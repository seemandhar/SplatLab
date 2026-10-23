"""
AA-Wavelet-Entropy-Plus GS — Enhanced Anti-Aliasing + Wavelet + Entropy
========================================================================
Complexity: ★★☆☆☆ (custom losses + scale clamping, no new Gaussian params)

Builds on AA-Wavelet-Entropy GS with four additional improvements:

    1. **SSIM weight 0.3** (up from default 0.2): Stronger structural
       similarity emphasis in the main photometric loss.
    2. **Scale anisotropy regularization**: Penalizes Gaussians whose
       max/min scale ratio exceeds 8, preventing needle-like artifacts.
       lambda=0.003, active from step 0.
    3. **Gradient-Difference Constraint (GDC)**: Sobel-style edge loss
       between rendered and GT images for edge fidelity.
       lambda=0.002, activated after 5K iterations.
    4. **Higher wavelet weight**: Increased from 0.005 to 0.008 for
       stronger low-frequency supervision.

Key synergy: The stronger SSIM weight and GDC both improve structural
fidelity from complementary directions (global structure vs local edges),
while anisotropy regularization prevents degenerate Gaussians that would
confuse both losses.

Train:
    ns-train aa-wavelet-entropy-plus-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.methods.aa_wavelet_entropy_gs import (
    AAWaveletEntropyGSModel,
    AAWaveletEntropyGSModelConfig,
)
from gs_template.losses import (
    dwt_lf_loss,
    gradient_difference_loss,
    opacity_entropy_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AAWaveletEntropyPlusGSModelConfig(AAWaveletEntropyGSModelConfig):
    """AA-Wavelet-Entropy-Plus GS: enhanced AA + wavelet + entropy + GDC + anisotropy reg."""

    _target: Type = field(default_factory=lambda: AAWaveletEntropyPlusGSModel)

    # ── Override: higher SSIM weight ───────────────────────────────
    ssim_lambda: float = 0.3
    """SSIM weight in main photometric loss (default 0.2 -> 0.3)."""

    # ── Override: higher wavelet weight ────────────────────────────
    wavelet_loss_weight: float = 0.008
    """Weight for wavelet LF loss (increased from 0.005 to 0.008)."""

    # ── Scale anisotropy regularization ────────────────────────────
    scale_aniso_weight: float = 0.003
    """Weight for scale anisotropy regularization."""

    scale_aniso_max_ratio: float = 8.0
    """Maximum allowed ratio between max and min scales before penalty."""

    scale_aniso_start_step: int = 0
    """Activate scale anisotropy regularization from this step."""

    # ── Gradient-Difference Constraint (GDC) ───────────────────────
    gdc_weight: float = 0.002
    """Weight for gradient-difference constraint (edge loss)."""

    gdc_start_step: int = 5000
    """Activate GDC after this many training steps."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class AAWaveletEntropyPlusGSModel(AAWaveletEntropyGSModel):
    """AA-Wavelet-Entropy-Plus GS model.

    Inherits all AA-Wavelet-Entropy GS functionality (Mip-Splatting 3D
    smoothing filter, wavelet LF loss, opacity entropy) and adds:
        - Scale anisotropy regularization
        - Gradient-difference constraint (GDC)
        - Higher SSIM weight (via config ssim_lambda=0.3)
        - Higher wavelet weight (via config wavelet_loss_weight=0.008)

    Overrides:
        - compute_custom_losses(): adds anisotropy reg + GDC on top of
                                   parent's wavelet + entropy losses
    """

    config: AAWaveletEntropyPlusGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        CONSOLE.print(
            f"[bold green]AA-Wavelet-Entropy-Plus GS[/bold green]: "
            f"ssim_lambda={self.config.ssim_lambda}, "
            f"wavelet_w={self.config.wavelet_loss_weight}, "
            f"aniso(w={self.config.scale_aniso_weight}, "
            f"ratio={self.config.scale_aniso_max_ratio}, "
            f"start={self.config.scale_aniso_start_step}), "
            f"gdc(w={self.config.gdc_weight}, "
            f"start={self.config.gdc_start_step})"
        )

    # ── Custom losses: parent's wavelet+entropy + anisotropy + GDC ─

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        # Start with parent losses (wavelet LF + opacity entropy)
        losses = super().compute_custom_losses(outputs, batch, metrics_dict)

        # ── 1. Scale anisotropy regularization (from step 0) ──────
        if (
            self.config.scale_aniso_weight > 0
            and self.step >= self.config.scale_aniso_start_step
        ):
            scales = torch.exp(self.scales)  # (N, 3)
            scale_max = scales.max(dim=-1).values  # (N,)
            scale_min = scales.min(dim=-1).values  # (N,)
            ratio = scale_max / (scale_min + 1e-8)
            # Hinge penalty: only penalize ratios exceeding the threshold
            penalty = torch.clamp(
                ratio - self.config.scale_aniso_max_ratio, min=0.0
            )
            losses["scale_aniso_reg"] = (
                self.config.scale_aniso_weight * penalty.mean()
            )

        # ── 2. Gradient-Difference Constraint (after gdc_start_step) ─
        if (
            self.config.gdc_weight > 0
            and self.step >= self.config.gdc_start_step
        ):
            pred_rgb = outputs["rgb"]  # (H, W, 3)
            gt_rgb = batch["image"].to(pred_rgb.device).float()

            # Normalize uint8 [0,255] -> [0,1] if needed
            if gt_rgb.max() > 1.0:
                gt_rgb = gt_rgb / 255.0

            # Handle potential shape mismatch (training downscale)
            if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
                gt_rgb = torch.nn.functional.interpolate(
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            losses["gdc_loss"] = (
                self.config.gdc_weight
                * gradient_difference_loss(pred_rgb, gt_rgb)
            )

        return losses
