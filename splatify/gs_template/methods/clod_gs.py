"""
CLoD-GS — Continuous Level-of-Detail via 3D Gaussian Splatting
================================================================
Paper: Cheng et al., 2025 (arXiv:2510.09997)
Complexity: ★★★☆☆ (new param, modified rendering, custom loss)

Core idea: attach a learnable distance decay factor sigma_d to each
Gaussian.  At render time, opacity is attenuated by a Gaussian falloff
in normalised camera distance, modulated by a virtual distance scale
s_v.  Training randomly samples s_v so the model learns a single
unified representation that supports continuous quality-performance
trade-off at inference.

Key equations (from the paper):
    d_i'  = |mu_i - cam_center| / max_j(d_j)       (normalised dist)
    alpha'' = alpha * exp(-(d' * s_v)^2 /
                          (2 * ReLU(sigma_d)^2 + eps))
    M_i   = (alpha'' > tau)                          (culling mask)
    eta_target = 1 / s_v^1.5
    L_reg = (s_v - 1)^2 * ReLU(eta_actual - eta_target)^2
    w_s   = (1 - 0.5 * s_v / max_sv)^2
    L     = w_s * (L_render + lambda_reg * L_reg)

Demonstrates:
    - populate_modules   : per-Gaussian sigma_d parameter
    - get_outputs        : distance-adaptive opacity before rasterisation
    - get_loss_dict      : w_s weighting + primitive-count regularisation
    - split_gaussians    : replicate sigma_d during densification
    - get_gaussian_param_groups : register sigma_d optimiser

Train:
    ns-train clod-gs --data <path>
    ns-train clod-gs --data <path> --pipeline.model.sv_max 5.0
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
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
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CLoDGSModelConfig(GSTemplateModelConfig):
    """CLoD-GS: continuous level-of-detail configuration."""

    _target: Type = field(default_factory=lambda: CLoDGSModel)

    # ── LoD schedule ───────────────────────────────────────────────
    lod_start_iter: int = 5000
    """Iteration at which the distance-adaptive LoD mechanism activates.
    Before this, training proceeds as standard 3DGS."""

    sv_min: float = 1.0
    """Lower bound when sampling the virtual distance scale s_v."""

    sv_max: float = 10.0
    """Upper bound when sampling s_v during training."""

    eval_sv: float = 1.0
    """s_v used at evaluation time (1.0 = full quality)."""

    # ── Decay / filtering ──────────────────────────────────────────
    epsilon: float = 1e-6
    """Numerical stability constant in the decay denominator."""

    lod_cull_thresh: float = 0.005
    """Opacity threshold below which a Gaussian is considered culled
    when computing eta_actual for the regularisation loss."""

    # ── Loss weights ───────────────────────────────────────────────
    lambda_reg: float = 1.0
    """Weight for the primitive-count regularisation loss."""

    # ── Initialisation ─────────────────────────────────────────────
    sigma_d_init: float = 1.0
    """Initial value for the per-Gaussian decay factor sigma_d.
    Larger values keep distant Gaussians visible early in training."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class CLoDGSModel(GSTemplateModel):
    """CLoD-GS: continuous level-of-detail through distance-adaptive opacity.

    Overrides:
        populate_modules        — adds sigma_d per-Gaussian param
        get_outputs             — applies distance-dependent opacity decay
        get_loss_dict           — applies w_s weighting + L_reg
        split_gaussians         — replicates sigma_d on densification
        get_gaussian_param_groups — registers sigma_d for optimisation
        get_outputs_for_camera  — eval with configurable s_v
    """

    config: CLoDGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        # One learnable scalar per Gaussian: the distance decay width.
        # Memory overhead is ~1.6% (one float vs ~248 bytes per Gaussian).
        self.gauss_params["sigma_d"] = Parameter(
            torch.full(
                (self.num_points, 1),
                self.config.sigma_d_init,
                dtype=torch.float32,
            )
        )

        # Per-iteration state shared between get_outputs and get_loss_dict
        self._sv: float = 1.0
        self._eta_actual: Tensor = torch.tensor(1.0)
        self._lod_active: bool = False

    # ── Rendering with distance-adaptive opacity ───────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Standard 3DGS rendering with LoD opacity attenuation.

        Before lod_start_iter this is identical to vanilla splatfacto.
        After that, each Gaussian's opacity is multiplied by a distance-
        dependent decay controlled by its learnable sigma_d.
        """
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )
        sigma_d = crop_custom_param(self.gauss_params["sigma_d"], crop_ids)

        # Decide whether the LoD mechanism is active this step
        lod_on = self.step >= self.config.lod_start_iter

        if lod_on:
            # --- Camera centre in world coordinates ---
            viewmat = cam["viewmat"]          # (1, 4, 4)
            R = viewmat[0, :3, :3]            # (3, 3)
            t = viewmat[0, :3, 3]             # (3,)
            cam_center = -(R.T @ t)           # (3,)

            # --- Per-Gaussian distance to camera ---
            dist = torch.norm(
                means - cam_center.unsqueeze(0), dim=-1
            )                                 # (N,)
            d_max = dist.max().clamp(min=1e-8)
            d_norm = dist / d_max             # normalised to [0, 1]

            # --- Sample virtual distance scale ---
            if self.training:
                sv = (
                    random.random()
                    * (self.config.sv_max - self.config.sv_min)
                    + self.config.sv_min
                )
            else:
                sv = self.config.eval_sv

            # --- Distance-adaptive opacity decay ---
            sigma_act = torch.relu(sigma_d.squeeze(-1)) + self.config.epsilon
            exponent = -((d_norm * sv) ** 2) / (2.0 * sigma_act ** 2)
            decay = torch.exp(exponent)       # (N,)

            # Attenuated opacity in probability space
            alpha_base = torch.sigmoid(opacities.squeeze(-1))
            alpha_att = alpha_base * decay    # (N,)

            # Convert back to logit space so the rasteriser's internal
            # sigmoid recovers the attenuated value.
            alpha_safe = alpha_att.clamp(1e-6, 1.0 - 1e-6)
            opacities = torch.log(
                alpha_safe / (1.0 - alpha_safe)
            ).unsqueeze(-1)

            # --- Book-keeping for the loss ---
            mask = alpha_att > self.config.lod_cull_thresh
            self._sv = sv
            self._eta_actual = mask.float().mean()
            self._lod_active = True
        else:
            self._sv = 1.0
            self._lod_active = False

        # Rasterise (this calls gsplat with sigmoid/exp activations)
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )
        self.info=info

        return assemble_outputs(self, render, alpha, info, cam)

    # ── Loss: w_s weighting + primitive-count regularisation ───────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Compute losses with CLoD-GS weighting.

        When the LoD mechanism is active every loss term is scaled by
        w_s = (1 - 0.5 * s_v / max_sv)^2, which down-weights supervision
        at large virtual distances where the rendering is inherently
        coarser.  The primitive-count regularisation L_reg encourages the
        model to cull an appropriate fraction of Gaussians at each s_v.
        """
        # Base rendering losses (L1 + D-SSIM from splatfacto)
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if not self._lod_active:
            return loss_dict

        sv = self._sv
        max_sv = self.config.sv_max

        # Supervision weight: weaker at large virtual distances
        w_s = (1.0 - 0.5 * sv / max_sv) ** 2

        # Scale every existing loss by w_s
        for key in list(loss_dict.keys()):
            loss_dict[key] = loss_dict[key] * w_s

        # Primitive-count regularisation
        eta_target = 1.0 / (sv ** 1.5)
        eta_excess = torch.relu(self._eta_actual - eta_target)
        l_reg = ((sv - 1.0) ** 2) * (eta_excess ** 2)
        loss_dict["lod_reg"] = w_s * self.config.lambda_reg * l_reg

        return loss_dict

    # ── Densification: replicate sigma_d on splits ─────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        out["sigma_d"] = self.gauss_params["sigma_d"][split_mask].repeat(
            samps, 1
        )
        return out

    # ── Optimiser registration ─────────────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_gaussian_param_groups()
        groups["sigma_d"] = [self.gauss_params["sigma_d"]]
        return groups

    # ── Viewer / eval outputs ──────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)
        return outs

    # ── Checkpoint loading ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    # ── Metrics ────────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["sv"] = torch.tensor(self._sv, dtype=torch.float32)
        if self._lod_active:
            metrics["eta_actual"] = self._eta_actual.detach().cpu()
        return metrics
