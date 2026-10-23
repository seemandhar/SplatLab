"""
MSAA-GS — Multi-Sample Anti-Aliasing and Constrained Optimization for 3DGS
=============================================================================
Paper: Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting

Three contributions:
    1. MSAA  — 4x subpixel multi-sample anti-aliasing during rasterization
    2. AWS   — Adaptive Weighting Strategy (error-adaptive per-pixel L1)
    3. GDC   — Gradient Difference Constraints (edge/gradient fidelity loss)

Architecture:
    - get_outputs()           -> MSAA via rendering helpers (single camera setup,
                                 multiple rasterizations, average)
    - compute_custom_losses() -> AWS + GDC as additive losses
    - get_loss_dict()         -> NOT overridden (splatfacto L1+DSSIM untouched)

Train:
    ns-train msaa-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    adaptive_weighted_l1_loss,
    gradient_difference_loss,
)
from gs_template.rendering import (
    assemble_outputs,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# MSAA Subpixel Offset Patterns

MSAA_OFFSETS_4X = [
    (0.375, 0.125),
    (0.875, 0.375),
    (0.125, 0.625),
    (0.625, 0.875),
]

MSAA_OFFSETS_2X = [
    (0.25, 0.25),
    (0.75, 0.75),
]

MSAA_OFFSETS_8X = [
    (0.5625, 0.3125),
    (0.4375, 0.6875),
    (0.8125, 0.5625),
    (0.3125, 0.1875),
    (0.1875, 0.8125),
    (0.0625, 0.4375),
    (0.6875, 0.9375),
    (0.9375, 0.0625),
]

MSAA_PATTERNS = {
    2: MSAA_OFFSETS_2X,
    4: MSAA_OFFSETS_4X,
    8: MSAA_OFFSETS_8X,
}


# Configuration

@dataclass
class MSAAGSModelConfig(GSTemplateModelConfig):
    """MSAA-GS: Multi-Sample Anti-Aliasing with Constrained Optimization."""

    _target: Type = field(default_factory=lambda: MSAAGSModel)

    msaa_samples: int = 4
    """Number of subpixel samples per pixel (2, 4, or 8). Paper uses 4."""

    msaa_enabled_during_eval: bool = True
    """Whether to apply MSAA during evaluation."""

    # AWS (Adaptive Weighting Strategy)
    aws_weight: float = 0.1
    """Weight for additive AWS loss."""

    aws_alpha: float = 0.5
    """Base weight floor in adaptive weighting."""

    aws_epsilon: float = 1e-6
    """Stability constant."""

    # GDC (Gradient Difference Constraints)
    grad_loss_weight: float = 0.1
    """Weight for gradient difference constraint loss."""

    output_depth_during_training: bool = True


# Model

class MSAAGSModel(GSTemplateModel):
    """MSAA-GS: Multi-Sample Anti-Aliasing + AWS + GDC for 3DGS.

    MSAA uses rendering helpers: one prepare_camera + gather_base_params
    call, then N rasterize_gaussians calls with shifted intrinsics,
    averaged. assemble_outputs uses first sample's info for densification.

    AWS+GDC are additive losses via compute_custom_losses().
    Splatfacto's standard L1 + DSSIM + bilateral grid etc. are untouched.
    """

    config: MSAAGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.msaa_samples not in MSAA_PATTERNS:
            raise ValueError(
                f"msaa_samples must be one of {list(MSAA_PATTERNS.keys())}, "
                f"got {self.config.msaa_samples}"
            )
        CONSOLE.print(
            f"[bold green]MSAA-GS[/bold green]: {self.config.msaa_samples}x "
            f"MSAA, aws_weight={self.config.aws_weight}, "
            f"grad_weight={self.config.grad_loss_weight}"
        )

    # MSAA Rendering

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """MSAA rendering via shifted intrinsics.

        Uses rendering helpers for a single camera setup + N rasterizations.
        Key: camera optimizer applied once, base params gathered once,
        only the rasterization is repeated with shifted K matrices.
        """
        use_msaa = self.training or self.config.msaa_enabled_during_eval
        n_samples = self.config.msaa_samples if use_msaa else 1

        # Phase 1: Camera setup (once)
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        # Gather Gaussian params (once)
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        if n_samples == 1:
            render, alpha, info = rasterize_gaussians(
                self, means, quats, scales, opacities, colors,
                cam["viewmat"], cam["K"], cam["W"], cam["H"],
            )
            self.info = info
            return assemble_outputs(self, render, alpha, info, cam)

        # Phase 2: Multi-sample rasterization
        offsets = MSAA_PATTERNS[n_samples]
        renders = []
        alphas = []
        first_info = None

        K_base = cam["K"].clone()

        for k, (dx, dy) in enumerate(offsets):
            K_shifted = K_base.clone()
            K_shifted[0, 0, 2] += (dx - 0.5)
            K_shifted[0, 1, 2] += (dy - 0.5)

            render_k, alpha_k, info_k = rasterize_gaussians(
                self, means, quats, scales, opacities, colors,
                cam["viewmat"], K_shifted, cam["W"], cam["H"],
            )
            renders.append(render_k)
            alphas.append(alpha_k)

            if k == 0:
                first_info = info_k

        # Phase 3: Average (Eq. 6)
        render_avg = torch.stack(renders, dim=0).mean(dim=0)
        alpha_avg = torch.stack(alphas, dim=0).mean(dim=0)

        # Use first sample's info for densification
        self.info = first_info
        return assemble_outputs(
            self, render_avg, alpha_avg, first_info, cam
        )

    # Additive Custom Losses

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """AWS + GDC as additive losses on top of splatfacto's standard loss."""
        losses = {}

        # Use splatfacto's own GT preprocessing
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_rgb = self.composite_with_background(
            outputs["rgb"], outputs["background"]
        )

        # AWS: Adaptive Weighted L1 (Eq. 8-9)
        if self.config.aws_weight > 0:
            losses["aws_l1"] = (
                self.config.aws_weight
                * adaptive_weighted_l1_loss(
                    pred_rgb, gt_rgb,
                    alpha=self.config.aws_alpha,
                    epsilon=self.config.aws_epsilon,
                )
            )

        # GDC: Gradient Difference Constraint (Eq. 11)
        if self.config.grad_loss_weight > 0:
            losses["grad_loss"] = (
                self.config.grad_loss_weight
                * gradient_difference_loss(pred_rgb, gt_rgb)
            )

        return losses