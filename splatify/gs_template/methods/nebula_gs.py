"""
NebulaGS — Physics-Informed Gaussian Splatting for Volumetric Nebula Scenes
=============================================================================
Complexity: ★★★☆☆ (new params + physics-informed losses + modified densification)

v2 — Rebalanced after empirical results:
    splatfacto baseline:  PSNR 31.19  SSIM 0.862  LPIPS 0.290
    nebula-gs v1:         PSNR 29.30  SSIM 0.888  LPIPS 0.255

    v1 diagnosis: physics priors improved structure (SSIM +0.026, LPIPS -0.035)
    but opacity softness and spectral loss were 5-10x too strong, actively
    fighting the L1+SSIM reconstruction objective and costing 1.9 dB PSNR.

    v2 changes:
    - All physics losses reduced to gentle-nudge regime (λ < 0.01)
    - Linear warmup schedule: physics losses fade in from 0 over first 3k steps
    - Opacity prior changed from "push to 0.5" to "penalize only >0.95"
    - Emission init changed from 0.5 (→1.24x boost) to 0.0 (→1.0x neutral)
    - Dropped radial spectrum loss (redundant with spectral) and emission
      decorrelation (over-constraining)
    - Raised accumulation ceiling from 0.85 to 0.95
    - Added multi-scale L1 loss (cooperates with main L1, cheap, stable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.rendering import (
    assemble_outputs,
    crop_custom_param,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# ═══════════════════════════════════════════════════════════════════
#  Physics-Informed Loss Functions
# ═══════════════════════════════════════════════════════════════════

def spectral_frequency_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Frequency-domain loss via 2D FFT log-amplitude matching.

    Nebulae exhibit Kolmogorov turbulence with power spectrum P(k) ∝ k^{-5/3}.
    Pixel-space losses (L1, SSIM) are dominated by low-frequency content and
    under-weight the high-frequency turbulent filaments. By matching log-amplitude
    spectra, we give equal optimization pressure across ALL spatial frequencies.

    Args:
        rendered: (H, W, C) rendered image (float, [0,1])
        target:   (H, W, C) ground-truth image (float, [0,1])
    Returns:
        Scalar loss
    """
    C = min(rendered.shape[-1], 3)
    loss = torch.tensor(0.0, device=rendered.device)

    for c in range(C):
        r_fft = torch.fft.rfft2(rendered[..., c])
        t_fft = torch.fft.rfft2(target[..., c])

        # Log-amplitude (log1p for numerical stability)
        r_logmag = torch.log1p(r_fft.abs())
        t_logmag = torch.log1p(t_fft.abs())

        loss = loss + F.mse_loss(r_logmag, t_logmag)

    return loss / C


def opacity_anti_binarization_loss(opacities_logit: Tensor, ceiling: float = 0.95) -> Tensor:
    """Gently prevent opacity binarization — penalize only extreme values.

    v1 used full entropy maximization (push everything to α=0.5), which
    fought the reconstruction loss hard (-1.9 dB PSNR). v2 only penalizes
    opacities above `ceiling` — a one-sided soft constraint that allows
    the optimizer to freely use the [0, 0.95] range.

    Physics basis: In optically thin emission nebulae, additive rendering
    I ≈ Σ αᵢcᵢ works when no single Gaussian dominates. We just prevent
    binarization to 1.0 (which creates shells), not force softness.

    Returns:
        mean(ReLU(σ(logit) - ceiling)²)
    """
    o = torch.sigmoid(opacities_logit)
    return F.relu(o - ceiling).pow(2).mean()


def scale_isotropy_loss(scales_log: Tensor) -> Tensor:
    """Penalize anisotropic Gaussians → encourage volumetric shapes.

    Nebula gas at sub-injection scales is approximately isotropic.
    Flat disk-like Gaussians (standard 3DGS bias) can't represent
    volumetric structure. Uses coefficient-of-variation of the 3 axes.
    """
    scales = torch.exp(scales_log)  # (N, 3)
    mean_s = scales.mean(dim=-1, keepdim=True)  # (N, 1)
    var_s = ((scales - mean_s) ** 2).mean(dim=-1)  # (N,)
    cv_sq = var_s / (mean_s.squeeze(-1) ** 2 + 1e-8)  # (N,)
    return cv_sq.mean()


def accumulation_sparsity_loss(accumulation: Tensor, max_alpha: float = 0.95) -> Tensor:
    """Penalize over-accumulation → keep nebula see-through.

    Soft upper bound on per-pixel accumulated opacity. Nebulae are
    optically thin — accumulated α near 1.0 creates opaque walls.
    Stars can still reach full opacity (penalty is soft quadratic).
    """
    excess = F.relu(accumulation - max_alpha)
    return (excess ** 2).mean()


def star_sharpness_loss(
    rendered: Tensor, target: Tensor, threshold: float = 0.92
) -> Tensor:
    """Extra reconstruction penalty on bright point sources (stars).

    Stars occupy few pixels but are perceptually important. Standard
    L1/L2 averages across all pixels, under-weighting star regions.
    This upweights bright pixels to ensure sharp point-source reconstruction.
    """
    t_lum = 0.2126 * target[..., 0] + 0.7152 * target[..., 1] + 0.0722 * target[..., 2]
    star_mask = (t_lum > threshold).float()

    if star_mask.sum() < 1:
        return torch.tensor(0.0, device=rendered.device)

    error = (rendered - target).abs().mean(dim=-1)
    return (error * star_mask).sum() / star_mask.sum().clamp(min=1)


def multi_scale_l1_loss(rendered: Tensor, target: Tensor, scales: int = 3) -> Tensor:
    """Multi-scale L1 loss — adds supervision at downsampled resolutions.

    Nebulae have structure at multiple spatial scales (Kolmogorov cascade).
    Standard L1 only supervises at full resolution. Downsampled L1 terms
    provide explicit gradient signal for large-scale structure. This is
    cheap, stable, and cooperates with (not fights) the main L1 loss.
    """
    loss = torch.tensor(0.0, device=rendered.device)
    r = rendered.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    t = target.permute(2, 0, 1).unsqueeze(0)

    for s in range(scales):
        if s > 0:
            r = F.avg_pool2d(r, 2)
            t = F.avg_pool2d(t, 2)
        loss = loss + F.l1_loss(r, t)

    return loss / scales


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NebulaGSModelConfig(GSTemplateModelConfig):
    """NebulaGS: Physics-informed Gaussian Splatting for volumetric nebula scenes.

    v2: All physics loss weights rebalanced to gentle-nudge regime.
    """

    _target: Type = field(default_factory=lambda: NebulaGSModel)

    # ── Rendering priors ─────────────────────────────────────
    sh_degree: int = 0
    """SH degree 0: nebulae have no view-dependent effects (no specular).
    All SH capacity goes to spatial color fidelity."""

    background_color: str = "black"
    """Space is black."""

    output_depth_during_training: bool = True
    """Enable depth output for accumulation-based losses."""

    # ── Loss warmup ──────────────────────────────────────────
    physics_loss_warmup_steps: int = 3000
    """Linear warmup for all physics losses. Reconstruction converges
    first, then physics priors gently steer the solution. At step 0
    the physics weight multiplier is 0; at warmup_steps it reaches 1."""

    # ── Spectral / frequency loss ────────────────────────────
    spectral_loss_weight: float = 0.008
    """Weight for FFT log-amplitude spectral loss.
    v1 was 0.05 — too strong, fought L1. v2: gentle high-freq nudge."""

    spectral_loss_start: int = 5000
    """Hard delay before spectral loss activates (on top of warmup).
    Gaussians need rough placement before frequency matching helps."""

    # ── Multi-scale reconstruction ───────────────────────────
    multiscale_loss_weight: float = 0.05
    """Weight for multi-scale L1 loss. Cheap, stable, cooperates
    with the main L1 loss instead of fighting it."""

    multiscale_levels: int = 3
    """Number of downsampling levels for multi-scale loss."""

    # ── Opacity prior ────────────────────────────────────────
    opacity_ceiling_weight: float = 0.005
    """Weight for anti-binarization. Only penalizes α > ceiling.
    v1 used full entropy maximization at 0.03 — way too aggressive."""

    opacity_ceiling: float = 0.95
    """Opacities above this get gently penalized. Allows [0, 0.95]
    freely; only prevents the hard binarization to 1.0 that creates shells."""

    # ── Accumulation sparsity ────────────────────────────────
    accumulation_sparsity_weight: float = 0.005
    """Weight for penalizing opaque walls. v1 was 0.01 with ceiling 0.85."""

    accumulation_max_alpha: float = 0.95
    """Soft ceiling for accumulated alpha. v1 was 0.85 (too aggressive)."""

    # ── Geometry priors ──────────────────────────────────────
    scale_isotropy_weight: float = 0.002
    """Weight for penalizing anisotropic Gaussians. v1 was 0.005."""

    # ── Star handling ────────────────────────────────────────
    star_loss_weight: float = 0.05
    """Extra weight on bright stars. v1 was 0.1."""

    star_luminance_threshold: float = 0.92
    """Luminance threshold for star detection."""

    # ── Densification tuning for volumetric media ────────────
    refine_every: int = 100
    """Densify frequently for volumetric coverage."""

    cull_alpha_thresh: float = 0.005
    """Keep soft Gaussians alive (standard culls at ~0.1)."""

    warmup_length: int = 500
    """Longer warmup for stable volumetric init."""


# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════

class NebulaGSModel(GSTemplateModel):
    """Physics-informed Gaussian Splatting for volumetric nebula rendering.

    v2: Losses rebalanced after empirical testing. Physics priors are
    gentle nudges that cooperate with (not fight) reconstruction.
    """

    config: NebulaGSModelConfig

    # ── Extension Point 1: New Parameters ──────────────────────

    def populate_modules(self):
        """Add per-Gaussian emission intensity parameter.

        Decouples brightness from opacity: emission scales the color
        independently, allowing bright-but-transparent Gaussians.

        v2: Initialize at 0.0 so emission_factor = exp(0) = 1.0 (neutral).
        v1 initialized at 0.5 → 2*sigmoid(0.5) ≈ 1.24x boost that hurt
        early reconstruction.
        """
        super().populate_modules()

        self.gauss_params["emission"] = Parameter(
            torch.zeros(self.num_points, 1)
        )

    # ── Extension Point 2: Custom Rendering ────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Emission-weighted rendering.

        emission_factor = exp(emission) clamped to [0.1, 5.0]:
          - init: exp(0) = 1.0 (neutral, identical to baseline at start)
          - can brighten (emission > 0) or dim (emission < 0) independently
            of opacity — this is the key physical decorrelation
        """
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        emission = crop_custom_param(
            self.gauss_params["emission"], crop_ids
        )

        # exp(0)=1 at init; clamp for numerical stability
        emission_factor = torch.exp(emission).clamp(0.1, 5.0)
        colors = colors * emission_factor

        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            sh_degree=None,
        )

        self.info = info

        outputs = assemble_outputs(self, render, alpha, info, cam)
        outputs["accumulation"] = alpha.squeeze(0)

        return outputs

    # ── Warmup helper ──────────────────────────────────────────

    def _physics_weight(self) -> float:
        """Linear warmup multiplier for physics losses.

        Returns 0 at step 0, linearly ramps to 1.0 at
        physics_loss_warmup_steps. Ensures reconstruction converges
        before physics priors kick in.
        """
        if self.step <= 0:
            return 0.0
        return min(1.0, self.step / max(1, self.config.physics_loss_warmup_steps))

    # ── Extension Point 3: Physics-Informed Losses ─────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute gently-weighted physics-informed losses with warmup."""
        losses = {}
        rgb_rendered = outputs.get("rgb")
        rgb_target = batch.get("image")

        # Cast to float — batch images may arrive as uint8
        if rgb_target is not None:
            if rgb_target.dtype == torch.uint8:
                rgb_target = rgb_target.float() / 255.0
            else:
                rgb_target = rgb_target.float()

        pw = self._physics_weight()

        # Helper: match target spatial dims to rendered
        def _match_target(target: Tensor, rendered: Tensor) -> Tensor:
            t = target.to(rendered.device)
            if t.shape[:2] != rendered.shape[:2]:
                t = F.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(rendered.shape[0], rendered.shape[1]),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            return t

        have_images = (rgb_rendered is not None and rgb_target is not None)

        if have_images:
            target_matched = _match_target(rgb_target, rgb_rendered)

        # ── 1. Multi-scale L1 (cooperates with main L1) ──────
        if have_images and self.config.multiscale_loss_weight > 0:
            losses["multiscale_l1"] = (
                self.config.multiscale_loss_weight
                * multi_scale_l1_loss(
                    rgb_rendered, target_matched,
                    scales=self.config.multiscale_levels,
                )
            )

        # ── 2. Spectral Frequency Loss (delayed + warmed up) ──
        if (have_images
                and self.config.spectral_loss_weight > 0
                and self.step > self.config.spectral_loss_start):
            losses["spectral_loss"] = (
                pw * self.config.spectral_loss_weight
                * spectral_frequency_loss(rgb_rendered, target_matched)
            )

        # ── 3. Opacity anti-binarization (gentle ceiling) ─────
        if self.config.opacity_ceiling_weight > 0:
            losses["opacity_ceiling"] = (
                pw * self.config.opacity_ceiling_weight
                * opacity_anti_binarization_loss(
                    self.opacities,
                    ceiling=self.config.opacity_ceiling,
                )
            )

        # ── 4. Accumulation sparsity ─────────────────────────
        if (self.config.accumulation_sparsity_weight > 0
                and "accumulation" in outputs):
            losses["accum_sparsity"] = (
                pw * self.config.accumulation_sparsity_weight
                * accumulation_sparsity_loss(
                    outputs["accumulation"],
                    max_alpha=self.config.accumulation_max_alpha,
                )
            )

        # ── 5. Scale isotropy ─────────────────────────────────
        if self.config.scale_isotropy_weight > 0:
            losses["scale_isotropy"] = (
                pw * self.config.scale_isotropy_weight
                * scale_isotropy_loss(self.scales)
            )

        # ── 6. Star sharpness ─────────────────────────────────
        if have_images and self.config.star_loss_weight > 0:
            star_l = star_sharpness_loss(
                rgb_rendered, target_matched,
                threshold=self.config.star_luminance_threshold,
            )
            if star_l > 0:
                losses["star_sharpness"] = (
                    self.config.star_loss_weight * star_l
                )

        return losses

    # ── Extension Point 4: Densification ───────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        out["emission"] = self.gauss_params["emission"][
            split_mask
        ].repeat(samps, 1)
        return out

    # ── Extension Point 5: Parameter Groups ────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    # ── Extension Point 7: Viewer Outputs ──────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)
        opacity_vals = torch.sigmoid(self.opacities)
        outs["mean_opacity"] = opacity_vals.mean()
        outs["opacity_std"] = opacity_vals.std()
        return outs

    # ── Extension Point 8: Checkpoint Loading ──────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict.get(
                    p, dict.get(f"gauss_params.{p}")
                )
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            key = f"gauss_params.{name}"
            if key not in dict:
                dict[key] = torch.zeros(
                    (newp,) + param.shape[1:],
                    device=param.device if param.is_cuda else "cpu",
                )
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    # ── Extension Point 9: Metrics ─────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)

        opacity_vals = torch.sigmoid(self.opacities)
        metrics["mean_opacity"] = opacity_vals.mean()
        metrics["opacity_entropy"] = (
            -(opacity_vals * torch.log(opacity_vals + 1e-6)
              + (1 - opacity_vals) * torch.log(1 - opacity_vals + 1e-6))
        ).mean()

        emission_vals = torch.exp(
            self.gauss_params["emission"]
        ).clamp(0.1, 5.0)
        metrics["mean_emission"] = emission_vals.mean()

        scales = torch.exp(self.scales)
        mean_s = scales.mean(dim=-1)
        std_s = scales.std(dim=-1)
        metrics["scale_cv"] = (std_s / (mean_s + 1e-8)).mean()

        if "accumulation" in outputs:
            acc = outputs["accumulation"]
            metrics["mean_accumulation"] = acc.mean()
            metrics["max_accumulation"] = acc.max()

        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )

        return metrics