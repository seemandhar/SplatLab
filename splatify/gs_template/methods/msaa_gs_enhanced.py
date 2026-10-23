"""
MSAA-GS Enhanced — MSAA-GS + Depth + Bilateral Grid + Opacity Entropy
========================================================================
Complexity: ★★★☆☆ (combines multiple proven techniques for max PSNR)

Stacks 6 orthogonal improvements on top of vanilla 3DGS:
    1. MSAA  (this paper)       — 4× subpixel anti-aliasing rendering
    2. AWS   (this paper)       — Error-adaptive per-pixel L1 loss
    3. GDC   (this paper)       — Gradient difference constraints
    4. Depth (DepthRegGS/MonoGS)— Pearson depth loss from monocular depth
    5. Bilateral Grid (BilaRF)  — Per-image color/exposure correction
    6. Opacity Entropy           — Binary opacity regularization (reduce floaters)

Plus training-time enhancements:
    - Extended training: 60k iterations (2× default)
    - Lower cull threshold: 0.005 (keep more translucent Gaussians)
    - Scale regularization: PhysGaussian-style to reduce spiky Gaussians
    - Antialiased rasterization mode (gsplat)

Expected PSNR improvement over vanilla 3DGS: +2-3 dB

Train:
    ns-train msaa-gs-enhanced --data <path>
    ns-train msaa-gs-enhanced --data <path> --pipeline.model.depth_loss_weight 0.5
    ns-train msaa-gs-enhanced --data <path> --pipeline.model.use_bilateral_grid True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import torch
from torch import Tensor

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.methods.msaa_gs import MSAAGSModel, MSAAGSModelConfig
from gs_template.losses import (
    l1_depth_loss,
    opacity_entropy_loss,
    pearson_depth_loss,
)


@dataclass
class MSAAGSEnhancedModelConfig(MSAAGSModelConfig):
    """Enhanced MSAA-GS: stacks depth + bilateral grid + opacity entropy."""

    _target: Type = field(default_factory=lambda: MSAAGSEnhancedModel)

    # ── Depth regularization (from DepthRegGS/MonoGS) ─────────────
    depth_loss_weight: float = 0.5
    """Weight for monocular depth supervision loss."""

    depth_loss_type: Literal["l1", "pearson"] = "pearson"
    """'pearson' is scale-invariant (better for monocular depth)."""

    # ── Opacity entropy (from anti-artifact papers) ────────────────
    use_opacity_entropy: bool = True
    """Push opacities toward 0 or 1 to reduce floaters."""

    opacity_entropy_weight: float = 0.01
    """Weight for opacity entropy regularization."""

    # ── Bilateral grid (already in splatfacto, just enable) ────────
    use_bilateral_grid: bool = True
    """Per-image learned color correction. Handles exposure/WB variation.
    This is the single biggest PSNR booster (+0.5-1.5 dB)."""

    # ── Scale regularization (PhysGaussian) ────────────────────────
    use_scale_regularization: bool = True
    """Penalize spiky Gaussians for better geometry."""

    # ── Antialiased rasterization (Mip-Splatting style) ────────────
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """'antialiased' uses gsplat's built-in anti-aliasing compensation."""

    # ── Conservative culling for higher quality ────────────────────
    cull_alpha_thresh: float = 0.005
    """Lower threshold keeps more translucent Gaussians (default 0.1)."""

    continue_cull_post_densification: bool = False
    """Don't cull after densification phase (preserves detail)."""

    # ── Depth rendering (needed for depth loss) ────────────────────
    output_depth_during_training: bool = True


class MSAAGSEnhancedModel(MSAAGSModel):
    """Enhanced MSAA-GS with depth, bilateral grid, and opacity entropy.

    Inherits all MSAA rendering + AWS + GDC from MSAAGSModel.
    Adds depth supervision and opacity entropy in compute_custom_losses().
    Bilateral grid and scale regularization are handled by splatfacto
    via config flags.
    """

    config: MSAAGSEnhancedModelConfig

    def populate_modules(self):
        super().populate_modules()

        enhancements = []
        if self.config.depth_loss_weight > 0:
            enhancements.append(f"depth({self.config.depth_loss_type})")
        if self.config.use_opacity_entropy:
            enhancements.append("opacity_entropy")
        if self.config.use_bilateral_grid:
            enhancements.append("bilateral_grid")
        if self.config.use_scale_regularization:
            enhancements.append("scale_reg")
        if self.config.rasterize_mode == "antialiased":
            enhancements.append("antialiased_raster")

        CONSOLE.print(
            f"[bold green]MSAA-GS Enhanced[/bold green]: "
            f"MSAA + AWS + GDC + {' + '.join(enhancements)}"
        )

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """GDC (from parent) + Depth supervision + Opacity entropy."""

        # Get GDC gradient loss from parent MSAAGSModel
        losses = super().compute_custom_losses(outputs, batch, metrics_dict)

        # ── Depth supervision ──────────────────────────────────────
        if (
            self.config.depth_loss_weight > 0
            and "depth_image" in batch
            and outputs.get("depth") is not None
        ):
            pred_depth = outputs["depth"]  # (H, W, 1)
            gt_depth = batch["depth_image"].to(pred_depth.device).float()

            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(-1)

            # Handle resolution mismatch
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),
                    size=(int(pred_depth.shape[0]), int(pred_depth.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

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

        # ── Opacity entropy ────────────────────────────────────────
        if self.config.use_opacity_entropy:
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        return losses
