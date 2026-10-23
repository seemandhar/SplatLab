"""
Triple Regularization Gaussian Splatting
=========================================
Complexity: ★★☆☆☆ (custom losses only, no rendering changes)

Three orthogonal regularization losses applied jointly:

1. **Opacity entropy loss** — pushes per-Gaussian opacity toward 0 or 1
   (binary), reducing floaters. Activated after 10K iterations.
   lambda = 0.001

2. **Scale anisotropy regularization** — penalizes extreme anisotropy
   ratio max(s)/min(s) > threshold (default 10). Active from iter 0.
   lambda = 0.005

3. **Bilateral depth smoothness** — edge-aware L1 smoothness on rendered
   depth, weighted by image gradients so that depth edges align with
   color edges. Activated after 5K iterations.
   lambda = 0.002

Train:
    ns-train triple-reg-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import opacity_entropy_loss


@dataclass
class TripleRegGSModelConfig(GSTemplateModelConfig):
    """Triple Regularization GS configuration."""

    _target: Type = field(default_factory=lambda: TripleRegGSModel)

    # Must render depth during training for bilateral smoothness
    output_depth_during_training: bool = True

    # ── Opacity entropy ──────────────────────────────────────────
    opacity_entropy_weight: float = 0.001
    """Lambda for opacity entropy loss."""

    opacity_entropy_start_iter: int = 10000
    """Iteration at which opacity entropy loss activates."""

    # ── Scale anisotropy regularization ──────────────────────────
    scale_anisotropy_weight: float = 0.005
    """Lambda for scale anisotropy penalty."""

    scale_anisotropy_ratio_thresh: float = 10.0
    """Anisotropy ratio threshold: penalize when max(s)/min(s) > this."""

    scale_anisotropy_start_iter: int = 0
    """Iteration at which scale anisotropy loss activates."""

    # ── Bilateral depth smoothness ───────────────────────────────
    depth_smooth_weight: float = 0.002
    """Lambda for bilateral depth smoothness loss."""

    depth_smooth_start_iter: int = 5000
    """Iteration at which depth smoothness loss activates."""

    depth_smooth_color_sigma: float = 0.1
    """Sigma for bilateral color weighting (lower = stronger edge preservation)."""


class TripleRegGSModel(GSTemplateModel):
    """Triple Regularization Gaussian Splatting.

    Applies three orthogonal regularization losses on top of the
    standard splatfacto reconstruction losses. No rendering or
    densification changes.
    """

    config: TripleRegGSModelConfig

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}
        step = self.step

        # ── 1. Opacity entropy loss (after 10K iters) ───────────
        if step >= self.config.opacity_entropy_start_iter:
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        # ── 2. Scale anisotropy regularization (from iter 0) ────
        if step >= self.config.scale_anisotropy_start_iter:
            losses["scale_anisotropy"] = (
                self.config.scale_anisotropy_weight
                * _scale_anisotropy_loss(
                    self.scales,
                    self.config.scale_anisotropy_ratio_thresh,
                )
            )

        # ── 3. Bilateral depth smoothness (after 5K iters) ──────
        if (
            step >= self.config.depth_smooth_start_iter
            and outputs.get("depth") is not None
        ):
            rgb = outputs["rgb"]  # (H, W, 3)
            depth = outputs["depth"]  # (H, W, 1)
            losses["depth_smooth"] = (
                self.config.depth_smooth_weight
                * _bilateral_depth_smoothness(
                    depth, rgb, self.config.depth_smooth_color_sigma
                )
            )

        return losses


# ═══════════════════════════════════════════════════════════════════
# Loss helpers (module-private)
# ═══════════════════════════════════════════════════════════════════


def _scale_anisotropy_loss(
    scales_log: Tensor,
    ratio_thresh: float = 10.0,
) -> Tensor:
    """Hinge penalty on per-Gaussian anisotropy ratio.

    Penalizes Gaussians whose max(scale)/min(scale) exceeds the
    threshold, encouraging more isotropic shapes.

    Args:
        scales_log: (N, 3) log-space scale parameters
        ratio_thresh: maximum allowed anisotropy ratio
    """
    scales = torch.exp(scales_log)  # (N, 3)
    s_max = scales.max(dim=-1).values  # (N,)
    s_min = scales.min(dim=-1).values  # (N,)
    ratio = s_max / (s_min + 1e-8)  # (N,)
    # Hinge: only penalize when ratio > threshold
    penalty = torch.clamp(ratio - ratio_thresh, min=0.0)
    return penalty.mean()


def _bilateral_depth_smoothness(
    depth: Tensor,
    rgb: Tensor,
    color_sigma: float = 0.1,
) -> Tensor:
    """Bilateral-weighted L1 depth smoothness loss.

    Encourages smooth depth except at color edges. Uses image
    gradients as bilateral weights so depth discontinuities can
    align with color boundaries.

    Args:
        depth: (H, W, 1) rendered depth map
        rgb:   (H, W, 3) rendered RGB image
        color_sigma: controls edge sensitivity
    """
    # Depth gradients (L1)
    d = depth[..., 0]  # (H, W)
    depth_grad_x = (d[:, 1:] - d[:, :-1]).abs()  # (H, W-1)
    depth_grad_y = (d[1:, :] - d[:-1, :]).abs()  # (H-1, W)

    # Color gradients for bilateral weighting
    # Mean absolute color difference across channels
    rgb_grad_x = (rgb[:, 1:, :] - rgb[:, :-1, :]).abs().mean(dim=-1)  # (H, W-1)
    rgb_grad_y = (rgb[1:, :, :] - rgb[:-1, :, :]).abs().mean(dim=-1)  # (H-1, W)

    # Bilateral weights: high weight where color is smooth, low at edges
    weight_x = torch.exp(-rgb_grad_x.detach() / (color_sigma + 1e-8))  # (H, W-1)
    weight_y = torch.exp(-rgb_grad_y.detach() / (color_sigma + 1e-8))  # (H-1, W)

    # Weighted depth smoothness
    smooth_x = (weight_x * depth_grad_x).mean()
    smooth_y = (weight_y * depth_grad_y).mean()

    return smooth_x + smooth_y
