"""
Depth-Regularized Gaussian Splatting
======================================
Complexity: ★★☆☆☆ (custom loss + custom DataManager)

Adds depth supervision from pretrained monocular depth estimators
(Depth Anything V2, ZoeDepth, MiDAS) to improve geometry in
textureless or ambiguous regions.

Architecture (same pattern as Feature Splatting):
    DepthDataManager  — runs monocular depth estimation on all images at
                        startup, caches to disk, injects batch["depth_image"]
    DepthRegGSModel   — consumes batch["depth_image"], computes Pearson or
                        L1 depth loss. No heavy model dependency in model.py.

Demonstrates:
    - Custom DataManager for preprocessing (datamanagers.py)
    - Adding config fields for loss hyperparameters
    - Using compute_custom_losses() with losses from gs_template.losses
    - Zero rendering changes — inherited from SplatfactoModel

Train:
    ns-train depth-reg-gs --data <path>
    ns-train depth-reg-gs --data <path> --pipeline.model.depth_loss_type pearson
    ns-train depth-reg-gs --data <path> --pipeline.datamanager.depth_model zoedepth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import torch
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    l1_depth_loss,
    opacity_entropy_loss,
    pearson_depth_loss,
)


@dataclass
class DepthRegGSModelConfig(GSTemplateModelConfig):
    """Depth-regularized GS configuration."""

    _target: Type = field(default_factory=lambda: DepthRegGSModel)

    # Must enable depth rendering during training
    output_depth_during_training: bool = True

    # Depth loss settings
    depth_loss_weight: float = 0.5
    """Weight for depth supervision loss."""

    depth_loss_type: Literal["l1", "pearson"] = "pearson"
    """'l1' for absolute depth, 'pearson' for scale-invariant."""

    # Optional opacity regularization (helps with floaters)
    use_opacity_entropy: bool = True
    """Enable opacity entropy regularization."""

    opacity_entropy_weight: float = 0.01
    """Weight for opacity entropy loss."""


class DepthRegGSModel(GSTemplateModel):
    """Depth-regularized Gaussian Splatting.

    Overrides only compute_custom_losses(). Everything else
    (rendering, densification, etc.) is inherited from splatfacto.
    """

    config: DepthRegGSModelConfig

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        # ── Depth supervision ──────────────────────────────────────
        if "depth_image" in batch and outputs.get("depth") is not None:
            pred_depth = outputs["depth"]  # (H, W, 1) at training resolution
            gt_depth = batch["depth_image"].to(pred_depth.device)

            # Match dimensions
            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(-1)

            # Handle resolution mismatch: estimated depth is at full res,
            # but rendered depth is at (possibly downscaled) training res
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),  # (1, 1, H, W)
                    size=(int(pred_depth.shape[0]), int(pred_depth.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)  # back to (H, W, 1)

            # Valid mask: ignore zero / missing depth
            valid = gt_depth > 0
            if valid.sum() > 100:
                if self.config.depth_loss_type == "pearson":
                    losses["depth_loss"] = (
                        self.config.depth_loss_weight
                        * pearson_depth_loss(pred_depth[valid], gt_depth[valid])
                    )
                else:
                    losses["depth_loss"] = (
                        self.config.depth_loss_weight
                        * l1_depth_loss(pred_depth, gt_depth, valid)
                    )

        # ── Opacity entropy (optional) ─────────────────────────────
        if self.config.use_opacity_entropy:
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        return losses
