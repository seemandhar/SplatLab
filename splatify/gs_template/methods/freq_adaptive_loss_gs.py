"""
FreqAdaptiveLossGS — Frequency-Adaptive Loss with Progressive Wavelet Scheduling
=================================================================================
Complexity: Loss-only (no new Gaussian params, no rendering changes)

Replaces static L1+SSIM supervision with a frequency-aware decomposition:
    1. LF supervision: multi-level LL-subband L1 loss (strong from start)
    2. HF supervision: per-band (LH, HL, HH) L1 losses with linear ramp-up
    3. GDC: gradient-difference constraint for edge fidelity

Key insight: Unlike FreGS which adds frequency regularization as a separate term,
this replaces the core supervision signal with a frequency-aware decomposition.
Progressive scheduling ensures low-frequency structure is learned first, then
high-frequency detail is refined.

Train:
    ns-train freq-adaptive-loss-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    dwt_lf_loss,
    haar_dwt_2d,
    gradient_difference_loss,
)


@dataclass
class FreqAdaptiveLossGSModelConfig(GSTemplateModelConfig):
    """Frequency-adaptive loss with progressive wavelet scheduling + GDC edge loss."""

    _target: Type = field(default_factory=lambda: FreqAdaptiveLossGSModel)

    # -- DWT Low-Frequency Supervision ----------------------------------------
    dwt_lf_weight: float = 0.3
    """Weight for multi-level LL-subband L1 loss. Active from iteration 0."""

    dwt_levels: int = 2
    """Number of Haar DWT decomposition levels for LF loss.
    Level 1 = half-res LL, Level 2 = quarter-res LL."""

    # -- DWT High-Frequency Supervision (progressive ramp) --------------------
    dwt_hf_weight_start: float = 0.0
    """Initial weight for per-band HF losses (LH, HL, HH). Starts at 0."""

    dwt_hf_weight_end: float = 0.2
    """Final weight for per-band HF losses after ramp completes."""

    dwt_hf_ramp_start: int = 5000
    """Iteration at which HF weight ramp begins."""

    dwt_hf_ramp_end: int = 15000
    """Iteration at which HF weight ramp reaches dwt_hf_weight_end."""

    # -- Gradient Difference Constraint (GDC) ---------------------------------
    gdc_weight: float = 0.1
    """Weight for gradient-difference edge fidelity loss. Always active."""


class FreqAdaptiveLossGSModel(GSTemplateModel):
    """Frequency-Adaptive Loss GS — progressive wavelet-domain supervision.

    Overrides only:
        - compute_custom_losses(): DWT multi-band losses + GDC
        - get_metrics_dict(): per-band loss values and frequency error ratios

    Everything else (rendering, densification, etc.) inherited from splatfacto.
    """

    config: FreqAdaptiveLossGSModelConfig

    def _get_hf_weight(self) -> float:
        """Compute current HF loss weight via linear ramp schedule.

        Returns 0.0 for step < ramp_start,
        linearly interpolates for ramp_start <= step <= ramp_end,
        returns dwt_hf_weight_end for step > ramp_end.
        """
        step = self.step
        if step < self.config.dwt_hf_ramp_start:
            return self.config.dwt_hf_weight_start
        if step >= self.config.dwt_hf_ramp_end:
            return self.config.dwt_hf_weight_end

        # Linear interpolation
        progress = (step - self.config.dwt_hf_ramp_start) / max(
            self.config.dwt_hf_ramp_end - self.config.dwt_hf_ramp_start, 1
        )
        return (
            self.config.dwt_hf_weight_start
            + progress * (self.config.dwt_hf_weight_end - self.config.dwt_hf_weight_start)
        )

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Frequency-adaptive losses: DWT LF + progressive HF + GDC.

        Called by get_loss_dict() which adds these to splatfacto's base
        L1 + DSSIM losses.
        """
        losses = {}

        pred_rgb = outputs["rgb"]  # (H, W, 3)
        gt_rgb = batch["image"].to(pred_rgb.device).float()  # (H, W, 3)

        # Normalize uint8 [0,255] -> [0,1] if needed
        if gt_rgb.max() > 1.0:
            gt_rgb = gt_rgb / 255.0

        # Handle potential shape mismatch (training downscale)
        if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
            gt_rgb = F.interpolate(
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        # -- 1. LF wavelet supervision (always active) -----------------------
        losses["dwt_lf_loss"] = (
            self.config.dwt_lf_weight
            * dwt_lf_loss(pred_rgb, gt_rgb, levels=self.config.dwt_levels)
        )

        # -- 2. HF per-band supervision (progressive ramp) -------------------
        hf_weight = self._get_hf_weight()
        if hf_weight > 0:
            # Single-level DWT decomposition for HF bands
            pred_LL, pred_LH, pred_HL, pred_HH = haar_dwt_2d(pred_rgb)
            gt_LL, gt_LH, gt_HL, gt_HH = haar_dwt_2d(gt_rgb)

            losses["dwt_lh_loss"] = hf_weight * F.l1_loss(pred_LH, gt_LH)
            losses["dwt_hl_loss"] = hf_weight * F.l1_loss(pred_HL, gt_HL)
            losses["dwt_hh_loss"] = hf_weight * F.l1_loss(pred_HH, gt_HH)

        # -- 3. Gradient Difference Constraint (always active) ----------------
        if self.config.gdc_weight > 0:
            losses["gdc_loss"] = (
                self.config.gdc_weight
                * gradient_difference_loss(pred_rgb, gt_rgb)
            )

        return losses

    def get_metrics_dict(self, outputs, batch) -> Dict[str, float]:
        """Log per-band losses and frequency-band error ratios."""
        metrics = super().get_metrics_dict(outputs, batch)

        pred_rgb = outputs["rgb"]
        gt_rgb = batch["image"].to(pred_rgb.device).float()
        if gt_rgb.max() > 1.0:
            gt_rgb = gt_rgb / 255.0

        # Handle shape mismatch
        if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
            gt_rgb = F.interpolate(
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        with torch.no_grad():
            # DWT decomposition for metrics
            pred_LL, pred_LH, pred_HL, pred_HH = haar_dwt_2d(pred_rgb)
            gt_LL, gt_LH, gt_HL, gt_HH = haar_dwt_2d(gt_rgb)

            # Per-band L1 errors
            ll_err = F.l1_loss(pred_LL, gt_LL).item()
            lh_err = F.l1_loss(pred_LH, gt_LH).item()
            hl_err = F.l1_loss(pred_HL, gt_HL).item()
            hh_err = F.l1_loss(pred_HH, gt_HH).item()

            metrics["dwt_ll_error"] = ll_err
            metrics["dwt_lh_error"] = lh_err
            metrics["dwt_hl_error"] = hl_err
            metrics["dwt_hh_error"] = hh_err

            # Frequency-band error ratio (HF / LF)
            total_hf = lh_err + hl_err + hh_err
            metrics["hf_lf_ratio"] = total_hf / (ll_err + 1e-8)

            # Current HF weight (for monitoring schedule)
            metrics["hf_weight_current"] = self._get_hf_weight()

        return metrics
