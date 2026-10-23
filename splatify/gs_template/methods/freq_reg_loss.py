"""
FreqRegLoss — Progressive Fourier Frequency Regularization Loss (FreGS-style)
==============================================================================
Complexity: Loss-only (no new Gaussian params, no rendering changes)

Adds a progressive frequency regularization loss in the Fourier domain:
    1. Compute 2D FFT of rendered and GT images
    2. Apply progressive band-pass masks (coarse → fine schedule)
    3. Penalize spectral discrepancy (L1 on magnitude spectrum)
    4. Gradient-difference constraint for edge sharpness

Key insight (from FreGS, CVPR 2024): Gaussian splatting tends to under-fit
high-frequency details. By explicitly regularizing the Fourier spectrum with
a coarse-to-fine schedule, we force the model to first match low-frequency
structure, then progressively refine high-frequency details.

Train:
    ns-train freq-reg-loss --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import gradient_difference_loss


def fourier_spectral_loss(
    pred: Tensor,
    target: Tensor,
    freq_cutoff_ratio: float = 1.0,
) -> Tensor:
    """Fourier-domain spectral regularization loss.

    Computes L1 loss between magnitude spectra of predicted and target images,
    optionally masked to only include frequencies up to a cutoff ratio.

    Args:
        pred:   (H, W, C) predicted RGB image
        target: (H, W, C) ground-truth RGB image
        freq_cutoff_ratio: fraction of max frequency to include [0, 1]
                          1.0 = all frequencies, 0.5 = low half only

    Returns:
        Scalar spectral loss.
    """
    # Move channels first for FFT: (C, H, W)
    pred_chw = pred.permute(2, 0, 1)
    target_chw = target.permute(2, 0, 1)

    # 2D FFT
    pred_fft = torch.fft.rfft2(pred_chw, norm="ortho")
    target_fft = torch.fft.rfft2(target_chw, norm="ortho")

    # Magnitude spectra
    pred_mag = pred_fft.abs()
    target_mag = target_fft.abs()

    if freq_cutoff_ratio < 1.0:
        # Create frequency mask (low-pass up to cutoff)
        C, H, W_half = pred_mag.shape
        # Frequency grid (normalized 0 to 1)
        fy = torch.fft.fftfreq(H, device=pred.device).abs()
        fx = torch.fft.rfftfreq(H, device=pred.device).abs()  # rfft has W//2+1 bins
        # Use proper rfft frequency grid
        fx = torch.linspace(0, 0.5, W_half, device=pred.device)
        freq_grid = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)
        mask = (freq_grid <= freq_cutoff_ratio * 0.5).float()  # 0.5 = Nyquist
        mask = mask.unsqueeze(0)  # (1, H, W_half)

        pred_mag = pred_mag * mask
        target_mag = target_mag * mask

    return F.l1_loss(pred_mag, target_mag)


def fourier_phase_loss(
    pred: Tensor,
    target: Tensor,
    freq_cutoff_ratio: float = 1.0,
) -> Tensor:
    """Fourier-domain phase consistency loss.

    Penalizes phase differences weighted by magnitude (important frequencies
    get higher weight).

    Args:
        pred:   (H, W, C) predicted RGB image
        target: (H, W, C) ground-truth RGB image
        freq_cutoff_ratio: fraction of max frequency to include [0, 1]

    Returns:
        Scalar phase loss.
    """
    pred_chw = pred.permute(2, 0, 1)
    target_chw = target.permute(2, 0, 1)

    pred_fft = torch.fft.rfft2(pred_chw, norm="ortho")
    target_fft = torch.fft.rfft2(target_chw, norm="ortho")

    # Phase difference: angle between complex numbers
    # Use complex multiplication: pred_fft * conj(target_fft)
    cross = pred_fft * target_fft.conj()
    phase_diff = 1.0 - (cross.real / (cross.abs() + 1e-8))

    # Weight by target magnitude (important frequencies matter more)
    weights = target_fft.abs().detach()
    weights = weights / (weights.mean() + 1e-8)

    if freq_cutoff_ratio < 1.0:
        C, H, W_half = phase_diff.shape
        fy = torch.fft.fftfreq(H, device=pred.device).abs()
        fx = torch.linspace(0, 0.5, W_half, device=pred.device)
        freq_grid = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)
        mask = (freq_grid <= freq_cutoff_ratio * 0.5).float().unsqueeze(0)
        phase_diff = phase_diff * mask
        weights = weights * mask

    return (weights * phase_diff).mean()


@dataclass
class FreqRegLossModelConfig(GSTemplateModelConfig):
    """Progressive Fourier frequency regularization (FreGS-style)."""

    _target: Type = field(default_factory=lambda: FreqRegLossModel)

    # -- Spectral magnitude loss -----------------------------------------------
    freq_mag_weight: float = 0.15
    """Weight for Fourier magnitude spectrum L1 loss."""

    # -- Spectral phase loss ---------------------------------------------------
    freq_phase_weight: float = 0.05
    """Weight for magnitude-weighted phase consistency loss."""

    # -- Progressive frequency schedule ----------------------------------------
    freq_cutoff_start: float = 0.2
    """Initial frequency cutoff ratio (fraction of Nyquist). Start coarse."""

    freq_cutoff_end: float = 1.0
    """Final frequency cutoff ratio. End with full spectrum."""

    freq_ramp_start: int = 0
    """Iteration at which frequency ramp begins."""

    freq_ramp_end: int = 20000
    """Iteration at which frequency cutoff reaches freq_cutoff_end."""

    # -- Gradient Difference Constraint ----------------------------------------
    gdc_weight: float = 0.05
    """Weight for gradient-difference edge fidelity loss."""

    gdc_start_iter: int = 5000
    """Iteration to start applying GDC (let coarse structure form first)."""


class FreqRegLossModel(GSTemplateModel):
    """Progressive Fourier Frequency Regularization for 3D Gaussian Splatting.

    Implements FreGS-style coarse-to-fine spectral supervision:
    - Early training: regularize only low frequencies (global structure)
    - Mid training: progressively include higher frequencies
    - Late training: full-spectrum regularization (fine details)

    Overrides only compute_custom_losses() and get_metrics_dict().
    """

    config: FreqRegLossModelConfig

    def _get_freq_cutoff(self) -> float:
        """Current frequency cutoff ratio via linear ramp."""
        step = self.step
        if step < self.config.freq_ramp_start:
            return self.config.freq_cutoff_start
        if step >= self.config.freq_ramp_end:
            return self.config.freq_cutoff_end
        progress = (step - self.config.freq_ramp_start) / max(
            self.config.freq_ramp_end - self.config.freq_ramp_start, 1
        )
        return (
            self.config.freq_cutoff_start
            + progress * (self.config.freq_cutoff_end - self.config.freq_cutoff_start)
        )

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Fourier frequency regularization losses."""
        losses = {}

        pred_rgb = outputs["rgb"]  # (H, W, 3)
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

        cutoff = self._get_freq_cutoff()

        # 1. Fourier magnitude spectrum loss (progressive)
        if self.config.freq_mag_weight > 0:
            losses["freq_mag_loss"] = (
                self.config.freq_mag_weight
                * fourier_spectral_loss(pred_rgb, gt_rgb, cutoff)
            )

        # 2. Fourier phase consistency loss (progressive)
        if self.config.freq_phase_weight > 0:
            losses["freq_phase_loss"] = (
                self.config.freq_phase_weight
                * fourier_phase_loss(pred_rgb, gt_rgb, cutoff)
            )

        # 3. Gradient Difference Constraint (delayed start)
        if self.config.gdc_weight > 0 and self.step >= self.config.gdc_start_iter:
            losses["gdc_loss"] = (
                self.config.gdc_weight
                * gradient_difference_loss(pred_rgb, gt_rgb)
            )

        return losses

    def get_metrics_dict(self, outputs, batch) -> Dict[str, float]:
        """Log spectral metrics for monitoring."""
        metrics = super().get_metrics_dict(outputs, batch)

        with torch.no_grad():
            metrics["freq_cutoff"] = self._get_freq_cutoff()

            pred_rgb = outputs["rgb"]
            gt_rgb = batch["image"].to(pred_rgb.device).float()
            if gt_rgb.max() > 1.0:
                gt_rgb = gt_rgb / 255.0
            if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
                gt_rgb = F.interpolate(
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            # Full-spectrum error for monitoring
            metrics["full_spectral_error"] = fourier_spectral_loss(
                pred_rgb, gt_rgb, 1.0
            ).item()

        return metrics
