"""
BubbleGS: Physics-Informed Gaussian Splatting for Iridescent Thin Films
=========================================================================
Complexity: ★★★☆☆ (new params, two-pass rendering, physics-informed losses)

Standard 3DGS fails on scenes with soap bubbles because:

    1. Alpha compositing drives bubble-region opacities toward 1.0,
       producing an opaque shell that blocks background visibility.
       Real soap films reflect only 2-8% of incident light.

    2. Spherical harmonics cannot capture the rapid sinusoidal color
       variation of thin-film interference at practical SH degree.

    3. No geometric prior encourages the flat, shell-like Gaussian
       arrangement matching a 2D soap film surface.

BubbleGS addresses these via per-Gaussian film thickness parameters
and physics-informed losses:

    - Per-Gaussian opacity ceiling: limits opacity of identified film
      Gaussians to physically motivated levels (~15%).
    - Thin-film chrominance loss: uses Fabry-Perot interference
      equations to guide SH coefficients toward correct iridescent
      colors based on the learned thickness map.
    - Shell flatness: encourages film Gaussians to form flat disks.
    - Thickness smoothness and sparsity: regularize the thickness map.

All film-specific losses activate AFTER densification completes
(default step 15000) to ensure the base scene is fully reconstructed
before any transparency pressure is applied. This prevents the
training collapse observed with early-activating transparency losses.

Architecture:
    Pass 1: Standard splatfacto RGB rendering (parent). Preserves
            densification controller, bilateral grid, etc.
    Pass 2: Film thickness rendering via lightweight second
            rasterization with sh_degree=None.

Train:
    ns-train bubble-gs \\
        --pipeline.model.background-color random \\
        blender-data --data <path>

References:
    Born & Wolf, Principles of Optics, Ch. 7
    Isenberg, The Science of Soap Films and Soap Bubbles (1992)
    Hecht, Optics, 5th ed., Section 9.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    opacity_entropy_loss,
    scale_flatten_loss,
    total_variation_loss,
)
from gs_template.rendering import (
    assemble_outputs,
    crop_custom_param,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# =====================================================================
# Thin-Film Interference Physics
# =====================================================================

LAMBDA_R = 650.0  # nm
LAMBDA_G = 550.0
LAMBDA_B = 450.0
N_FILM_DEFAULT = 1.33
N_AIR = 1.0


def thin_film_reflectance(
    thickness_nm: Tensor,
    cos_theta_i: Tensor,
    n_film: float = N_FILM_DEFAULT,
) -> Tensor:
    """RGB reflectance via Fabry-Perot interference for a thin film."""
    sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i
    sin2_theta_t = sin2_theta_i / (n_film * n_film)
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=1e-8))

    opd = 2.0 * n_film * thickness_nm * cos_theta_t
    r0 = ((n_film - N_AIR) / (n_film + N_AIR)) ** 2

    reflectances = []
    for lam in (LAMBDA_R, LAMBDA_G, LAMBDA_B):
        phi = 2.0 * math.pi * opd / lam + math.pi
        cos_phi = torch.cos(phi)
        R = 2.0 * r0 * (1.0 - cos_phi) / (
            1.0 + r0 * r0 - 2.0 * r0 * cos_phi + 1e-10
        )
        reflectances.append(R)

    return torch.stack(reflectances, dim=-1)


def thin_film_chrominance(
    thickness_nm: Tensor,
    cos_theta_i: Tensor,
    n_film: float = N_FILM_DEFAULT,
) -> Tensor:
    """Unit-chrominance vector from thin-film physics."""
    rgb = thin_film_reflectance(thickness_nm, cos_theta_i, n_film)
    norm = rgb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return rgb / norm


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class BubbleGSModelConfig(GSTemplateModelConfig):
    """Configuration for BubbleGS.

    DESIGN PRINCIPLE: all film-specific losses activate only after
    densification completes (step ~15000). This ensures the scene is
    fully reconstructed before any transparency pressure is applied.
    Earlier activation causes a feedback loop where opacity suppression
    spreads across the scene and collapses PSNR.

    The only losses active during early training are lightweight global
    regularizers (opacity entropy, max scale) that help scene quality
    without targeting the bubble specifically.
    """

    _target: Type = field(default_factory=lambda: BubbleGSModel)

    # -- Film parameters -----------------------------------------------

    film_thickness_range_nm: Tuple[float, float] = (50.0, 1200.0)
    """Physical thickness range mapped from sigmoid output."""

    film_n_refraction: float = 1.33
    """Refractive index of film medium (water/soap = 1.33)."""

    # -- Film loss activation ------------------------------------------

    film_loss_warmup: int = 15000
    """Step after which ALL film-specific losses activate. This must be
    after densification completes so the scene geometry is stable.
    Film losses ramp up linearly over film_loss_ramp_steps after this."""

    film_loss_ramp_steps: int = 5000
    """Number of steps to linearly ramp film losses from 0 to full
    weight after film_loss_warmup. Prevents abrupt changes."""

    # -- Per-Gaussian opacity ceiling ----------------------------------

    film_opacity_ceil: float = 0.15
    """Maximum opacity for individual film Gaussians. Real soap peak
    reflectance is ~8%; we allow up to 15% for modeling flexibility.
    This is the primary transparency mechanism."""

    film_opacity_ceil_weight: float = 0.05
    """Gentle weight. The loss is gated by film_prob, so it only
    affects Gaussians that have learned non-zero film thickness."""

    # -- Chrominance loss ----------------------------------------------

    film_chroma_weight: float = 0.005
    """Weight for thin-film chrominance consistency. This is the main
    value-add of BubbleGS: guiding SH colors toward physically
    correct interference patterns based on learned thickness."""

    # -- Shell flatness ------------------------------------------------

    film_flatness_weight: float = 0.005
    """Encourages flat disk-like film Gaussians."""

    # -- Thickness smoothness ------------------------------------------

    film_tv_weight: float = 0.001
    """Total variation on rendered thickness map."""

    # -- Thickness sparsity --------------------------------------------

    film_sparsity_weight: float = 0.001
    """L1 on film thickness to keep non-bubble Gaussians at zero."""

    # -- Global anti-floater (active from step 0) ----------------------

    global_opacity_entropy_weight: float = 0.005
    """Push ALL opacities toward binary {0, 1}. Active from step 0."""

    max_scale_weight: float = 0.01
    """Penalize oversized Gaussians. Active from step 0."""

    max_scale_threshold: float = 0.1
    """Scene-unit threshold for the max scale penalty."""

    # -- Incidence angle -----------------------------------------------

    default_cos_theta: float = 0.85


# =====================================================================
# Model
# =====================================================================

class BubbleGSModel(GSTemplateModel):
    """Two-pass rendering with physics-informed film losses."""

    config: BubbleGSModelConfig

    def _film_ramp(self) -> float:
        """Compute the linear ramp factor for film losses.

        Returns 0.0 before warmup, ramps linearly to 1.0 over
        film_loss_ramp_steps, then stays at 1.0.
        """
        if self.step < self.config.film_loss_warmup:
            return 0.0
        elapsed = self.step - self.config.film_loss_warmup
        return min(1.0, elapsed / max(1, self.config.film_loss_ramp_steps))

    # -- 1. populate_modules -------------------------------------------

    def populate_modules(self):
        """Add per-Gaussian film thickness parameter.

        Initialized at -5.0 (sigmoid = 0.007) so all Gaussians start
        firmly in the non-film state. The optimizer discovers the
        bubble region by driving thickness positive where it helps
        the photometric loss. Conservative init ensures film losses
        have near-zero effect during early training.
        """
        super().populate_modules()

        self.gauss_params["film_thickness_raw"] = Parameter(
            torch.full((self.num_points, 1), -5.0)
        )

    # -- 2. get_outputs (two-pass rendering) ---------------------------

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Two-pass rendering.

        Pass 1 (parent): Standard splatfacto RGB with full SH.
            Preserves xys, radii, densification state.
        Pass 2 (thickness): Lightweight rasterization of film
            thickness as a 1-channel pre-computed color.
        """
        # Pass 1: standard RGB rendering
        outputs = super().get_outputs(camera)

        # Pass 2: thickness rasterization
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return outputs

        means, quats, scales, opacities, _colors = gather_base_params(
            self, crop_ids
        )

        raw_thick = crop_custom_param(
            self.gauss_params["film_thickness_raw"], crop_ids
        )
        thickness_norm = torch.sigmoid(raw_thick)      # (N, 1)

        # Save densification state
        saved_xys = getattr(self, "xys", None)
        saved_radii = getattr(self, "radii", None)

        render_thick, _alpha, _info = rasterize_gaussians(
            self, means, quats, scales, opacities,
            thickness_norm,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            sh_degree=None,
            render_mode="RGB",
        )

        # Restore densification state
        if saved_xys is not None:
            self.xys = saved_xys
        if saved_radii is not None:
            self.radii = saved_radii

        thickness_map = render_thick.squeeze(0)        # (H, W, 1)
        lo, hi = self.config.film_thickness_range_nm
        outputs["film_thickness"] = thickness_map
        outputs["film_thickness_nm"] = lo + (hi - lo) * thickness_map

        return outputs

    # -- 3. compute_custom_losses --------------------------------------

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Two groups of losses with different activation schedules.

        Group 1 (Global, from step 0): Lightweight anti-floater
            regularization that improves overall scene quality.

        Group 2 (Film, from step 15000 with ramp): Physics-informed
            losses targeting the bubble region only. Activate after
            densification is complete and ramp up linearly.
        """
        losses = {}

        # ==============================================================
        # GROUP 1: Global anti-floater (active from step 0)
        # ==============================================================

        if self.config.global_opacity_entropy_weight > 0:
            losses["global_opacity_entropy"] = (
                self.config.global_opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        if self.config.max_scale_weight > 0:
            scales_exp = torch.exp(self.scales)
            max_s = scales_exp.max(dim=-1).values
            excess_s = F.relu(max_s - self.config.max_scale_threshold)
            losses["max_scale"] = (
                self.config.max_scale_weight * excess_s.mean()
            )

        # ==============================================================
        # GROUP 2: Film losses (after densification, with ramp)
        # ==============================================================

        ramp = self._film_ramp()
        if ramp <= 0.0:
            return losses

        thickness_nm = outputs.get("film_thickness_nm")
        if thickness_nm is None:
            return losses

        film_mask = (thickness_nm > 10.0).squeeze(-1)  # (H, W)
        film_frac = film_mask.float().mean()

        raw_thick = self.gauss_params["film_thickness_raw"]
        film_prob = torch.sigmoid(raw_thick).squeeze(-1)  # (N,)

        # --- (a) Per-Gaussian opacity ceiling -------------------------
        # The primary transparency mechanism. Limits opacity of
        # Gaussians identified as film to a physically motivated max.
        if self.config.film_opacity_ceil_weight > 0:
            opac = torch.sigmoid(self.opacities).squeeze(-1)
            excess = F.relu(opac - self.config.film_opacity_ceil)
            ceil_loss = (film_prob.detach() * excess * excess).mean()
            losses["film_opacity_ceil"] = (
                self.config.film_opacity_ceil_weight * ramp * ceil_loss
            )

        # --- (b) Thin-film chrominance --------------------------------
        # The main value-add: guides SH colors toward physically
        # correct interference patterns.
        if self.config.film_chroma_weight > 0 and film_frac > 0.005:
            rgb = outputs["rgb"]
            cos_theta = self._estimate_incidence_angle(outputs)
            pred_chroma = thin_film_chrominance(
                thickness_nm.squeeze(-1),
                cos_theta,
                n_film=self.config.film_n_refraction,
            )
            rgb_norm = rgb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            render_chroma = rgb / rgb_norm
            cos_sim = F.cosine_similarity(
                pred_chroma[film_mask],
                render_chroma[film_mask],
                dim=-1,
            )
            losses["film_chroma"] = (
                self.config.film_chroma_weight * ramp
                * (1.0 - cos_sim).mean()
            )

        # --- (c) Shell flatness ---------------------------------------
        if self.config.film_flatness_weight > 0:
            scales_exp = torch.exp(self.scales)
            sorted_s, _ = torch.sort(scales_exp, dim=-1)
            ratio = sorted_s[:, 0] / (sorted_s[:, 2] + 1e-8)
            flat_loss = (film_prob.detach() * ratio).mean()
            losses["film_flatness"] = (
                self.config.film_flatness_weight * ramp * flat_loss
            )

        # --- (d) Thickness smoothness (TV) ----------------------------
        if self.config.film_tv_weight > 0 and film_frac > 0.001:
            losses["film_tv"] = (
                self.config.film_tv_weight * ramp
                * total_variation_loss(thickness_nm)
            )

        # --- (e) Thickness sparsity -----------------------------------
        if self.config.film_sparsity_weight > 0:
            losses["film_sparsity"] = (
                self.config.film_sparsity_weight * ramp
                * torch.sigmoid(raw_thick).mean()
            )

        return losses

    # -- Incidence angle estimation ------------------------------------

    def _estimate_incidence_angle(
        self, outputs: Dict[str, Tensor]
    ) -> Tensor:
        """Estimate per-pixel cos(theta_i) from depth gradients."""
        depth = outputs.get("depth")
        H, W = outputs["rgb"].shape[:2]
        device = outputs["rgb"].device

        if depth is not None and depth.dim() >= 2:
            depth_2d = depth.squeeze(-1) if depth.dim() == 3 else depth

            if depth_2d.shape[0] >= 3 and depth_2d.shape[1] >= 3:
                dz_dy = (depth_2d[2:, 1:-1] - depth_2d[:-2, 1:-1]) / 2.0
                dz_dx = (depth_2d[1:-1, 2:] - depth_2d[1:-1, :-2]) / 2.0

                ones = torch.ones_like(dz_dx)
                normals = torch.stack([-dz_dx, -dz_dy, ones], dim=-1)
                normals = F.normalize(normals, dim=-1)

                cos_theta = normals[..., 2].abs()
                cos_theta = F.pad(
                    cos_theta.unsqueeze(0).unsqueeze(0),
                    (1, 1, 1, 1),
                    mode="replicate",
                ).squeeze(0).squeeze(0)
                cos_theta = cos_theta.clamp(0.05, 1.0)
                return cos_theta[:H, :W]

        return torch.full(
            (H, W), self.config.default_cos_theta, device=device
        )

    # -- 4. split_gaussians -------------------------------------------

    def split_gaussians(self, split_mask, samps):
        """Replicate film thickness during adaptive densification."""
        out = super().split_gaussians(split_mask, samps)
        out["film_thickness_raw"] = (
            self.gauss_params["film_thickness_raw"][split_mask]
            .repeat(samps, 1)
        )
        return out

    # -- 5. get_gaussian_param_groups ----------------------------------

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register film_thickness_raw for optimization."""
        groups = {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }
        return groups

    # -- 6. get_outputs_for_camera (viewer) ----------------------------

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        """Add thickness and iridescence visualizations."""
        outs = super().get_outputs_for_camera(camera, obb_box)

        if "film_thickness" in outs:
            thickness_nm = outs.get("film_thickness_nm")
            if thickness_nm is None:
                lo, hi = self.config.film_thickness_range_nm
                thickness_nm = lo + (hi - lo) * outs["film_thickness"]

            lo, hi = self.config.film_thickness_range_nm
            thick_vis = (thickness_nm - lo) / (hi - lo + 1e-8)
            outs["thickness_vis"] = thick_vis.clamp(0, 1).expand(
                -1, -1, 3
            )

            cos_theta = torch.full(
                thickness_nm.shape[:2],
                self.config.default_cos_theta,
                device=thickness_nm.device,
            )
            irid_rgb = thin_film_reflectance(
                thickness_nm.squeeze(-1),
                cos_theta,
                n_film=self.config.film_n_refraction,
            )
            outs["iridescence"] = (irid_rgb * 10.0).clamp(0, 1)

        return outs

    # -- 7. load_state_dict -------------------------------------------

    def load_state_dict(self, dict, **kwargs):
        """Handle checkpoints with film_thickness_raw."""
        self.step = 30000
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict.get(
                    p, dict.get(f"gauss_params.{p}")
                )

        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    # -- 8. get_metrics_dict ------------------------------------------

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Report film-related metrics for monitoring."""
        metrics = super().get_metrics_dict(outputs, batch)

        raw = self.gauss_params["film_thickness_raw"]
        film_prob = torch.sigmoid(raw).squeeze(-1)
        metrics["film_gaussians"] = (film_prob > 0.5).float().sum()
        metrics["mean_film_prob"] = film_prob.mean()

        if "film_thickness_nm" in outputs:
            t = outputs["film_thickness_nm"]
            mask = t > 10.0
            if mask.sum() > 0:
                metrics["mean_thickness_nm"] = t[mask].mean()

        return metrics