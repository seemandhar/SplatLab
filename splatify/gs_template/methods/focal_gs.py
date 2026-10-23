"""
Focal-GS: Error-Focused Gaussian Splatting with Adaptive Multi-Scale Supervision
==================================================================================
Paper: Focal-GS — Error-Focused Gaussian Splatting
Complexity: ★★★☆☆ (custom losses + config-level rendering improvements)

Key insight: Standard 3DGS uses uniform L1 + D-SSIM loss across all pixels,
treating well-reconstructed sky the same as under-reconstructed fine structures.
Meanwhile, the densification pipeline suffers from gradient dilution (Pixel-GS)
and unconstrained Gaussian growth (Micro-Splatting), while point-sample
rasterization introduces aliasing artifacts corrupting gradients (Mip-Splatting).

Focal-GS introduces a unified framework addressing BOTH the loss landscape AND
the optimization pipeline:

**Loss-side (compute_custom_losses):**
    1. Adaptive Weighted L1 (AWS): Error-adaptive per-pixel weighting
    2. L2 Photometric: Error-proportional gradients near convergence
    3. Wavelet Frequency Supervision: Multi-scale structural accuracy via Haar DWT
    4. Gradient Difference Constraint: Edge/fine structure preservation
    5. Covariance Trace Regularization: Prevent Gaussian bloat (causes blur)
    6. Isotropy Regularization: Prevent pathological anisotropy

**Config-side (no code changes needed):**
    7. Anti-aliased rasterization (rasterize_mode='antialiased')
    8. Absolute gradient accumulation (use_absgrad=True)

Extension points used:
    3. compute_custom_losses() — all loss components

Train:
    ns-train focal-gs --data <path> colmap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    adaptive_weighted_l1_loss,
    gradient_difference_loss,
    dwt_lf_loss,
    covariance_trace_loss,
    covariance_isotropy_loss,
    l2_photometric_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FocalGSModelConfig(GSTemplateModelConfig):
    """Focal-GS: Error-Focused Multi-Scale Supervision."""

    _target: Type = field(default_factory=lambda: FocalGSModel)

    # ── Adaptive Weighted L1 (AWS) ──────────────────────────────
    aws_weight: float = 0.1
    """Weight for additive AWS loss on top of splatfacto's base L1+DSSIM."""

    aws_alpha: float = 0.5
    """Base weight floor in adaptive weighting. 0=fully adaptive, 1=uniform."""

    # ── Wavelet Frequency Supervision ────────────────────────────
    dwt_lf_weight: float = 0.05
    """Weight for multi-level Haar DWT low-frequency loss."""

    dwt_levels: int = 2
    """Number of DWT decomposition levels."""

    # ── Gradient Difference Constraint ───────────────────────────
    gdc_weight: float = 0.01
    """Weight for gradient difference loss (edge preservation)."""

    # ── Covariance Trace Regularization ──────────────────────────
    cov_trace_weight: float = 0.001
    """Weight for covariance trace regularization."""

    trace_cap: float = 0.1
    """Upper bound on acceptable covariance trace."""

    # ── Isotropy Regularization ──────────────────────────────────
    isotropy_weight: float = 0.0005
    """Weight for isotropy loss."""

    # ── L2 Photometric Loss ──────────────────────────────────────
    l2_weight: float = 0.1
    """Weight for L2 photometric loss."""

    # ── Scheduling ───────────────────────────────────────────────
    loss_warmup_steps: int = 500
    """Steps before custom losses activate."""

    cov_reg_start_step: int = 1000
    """Step to start covariance regularization."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class FocalGSModel(GSTemplateModel):
    """Focal-GS: Error-Focused Gaussian Splatting.

    Overrides only compute_custom_losses() to add complementary loss
    components. Uses properly downscaled GT images via splatfacto's
    get_gt_img() to handle resolution schedule correctly.
    """

    config: FocalGSModelConfig

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Focal-GS multi-signal loss framework.

        All losses are additive on top of splatfacto's base
        (1-λ)*L1 + λ*DSSIM.

        CRITICAL: We must use self.get_gt_img() to handle the resolution
        schedule correctly — batch["image"] is full resolution while
        outputs["rgb"] may be downscaled during early training.
        """
        losses = {}
        step = self.step

        # Skip during warmup — let splatfacto's base loss stabilize first
        if step < self.config.loss_warmup_steps:
            return losses

        # Get rendered image
        pred = outputs["rgb"]    # (H, W, 3) — at current training resolution

        # Get GT image at matching resolution using splatfacto's built-in
        # resolution schedule handling
        gt = self.composite_with_background(
            self.get_gt_img(batch["image"]),
            outputs["background"]
        )

        # Safety check: ensure shapes match
        if pred.shape != gt.shape:
            return losses

        # ── 1. Adaptive Weighted L1 (AWS) ────────────────────────
        if self.config.aws_weight > 0:
            losses["aws_loss"] = self.config.aws_weight * adaptive_weighted_l1_loss(
                pred, gt,
                alpha=self.config.aws_alpha,
            )

        # ── 2. L2 Photometric Loss ──────────────────────────────
        if self.config.l2_weight > 0:
            losses["l2_photo_loss"] = self.config.l2_weight * l2_photometric_loss(
                pred, gt
            )

        # ── 3. Wavelet Frequency Supervision ─────────────────────
        if self.config.dwt_lf_weight > 0:
            losses["dwt_lf_loss"] = self.config.dwt_lf_weight * dwt_lf_loss(
                pred, gt,
                levels=self.config.dwt_levels,
            )

        # ── 4. Gradient Difference Constraint ────────────────────
        if self.config.gdc_weight > 0:
            losses["gdc_loss"] = self.config.gdc_weight * gradient_difference_loss(
                pred, gt
            )

        # ── 5. Covariance Trace Regularization ───────────────────
        if (self.config.cov_trace_weight > 0
                and step >= self.config.cov_reg_start_step):
            losses["cov_trace_loss"] = self.config.cov_trace_weight * covariance_trace_loss(
                self.scales, self.config.trace_cap
            )

        # ── 6. Isotropy Regularization ───────────────────────────
        if (self.config.isotropy_weight > 0
                and step >= self.config.cov_reg_start_step):
            losses["isotropy_loss"] = self.config.isotropy_weight * covariance_isotropy_loss(
                self.scales
            )

        return losses
