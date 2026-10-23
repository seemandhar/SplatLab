"""
WaveletLossGS — Multi-Scale Wavelet + Edge Loss for 3D Gaussian Splatting
==========================================================================
Complexity: ★★☆☆☆ (custom losses only, no new Gaussian params, no rendering changes)

Combines a multi-scale Haar wavelet loss with gradient-difference edge
supervision and a coarse-to-fine frequency schedule:

    1. HaarDWT (2-level): Decomposes rendered and GT images into LL/LH/HL/HH
       subbands at two scales. L1 loss on each subband with scheduled weights.
    2. Sobel-based gradient-difference loss: L_gdc = L1(sobel(rendered) - sobel(GT))
    3. Coarse-to-fine schedule: Gaussian blur on GT with sigma decaying from
       2.0 -> 0.0 over iters 0-20K. Wavelet and GDC losses activate after
       iter 5K with linear warmup 5K-8K.
    4. Total loss = (1-0.2)*L1 + 0.2*(1-SSIM) + 0.005*L_dwt + 0.002*L_gdc

All losses are purely additive — no rendering changes needed. The base
splatfacto L1+SSIM (main_loss) is inherited unchanged.

Train:
    ns-train wavelet-loss-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import haar_dwt_2d


# ═══════════════════════════════════════════════════════════════════
# Pure-PyTorch Sobel edge detection
# ═══════════════════════════════════════════════════════════════════

def _sobel_edges(img: Tensor) -> Tensor:
    """Compute Sobel edge magnitude for an (H, W, C) image.

    Uses fixed 3x3 Sobel kernels via grouped conv2d. Returns (H, W, C)
    gradient magnitude (sqrt(Gx^2 + Gy^2)).
    """
    # (H, W, C) -> (1, C, H, W)
    x = img.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    # Sobel kernels
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).unsqueeze(0).unsqueeze(0).expand(C, 1, 3, 3)  # (C, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device,
    ).unsqueeze(0).unsqueeze(0).expand(C, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # (1, C, H, W) -> (H, W, C)
    return mag.squeeze(0).permute(1, 2, 0)


def _gaussian_blur(img: Tensor, sigma: float) -> Tensor:
    """Apply Gaussian blur to (H, W, C) image using conv2d with a separable kernel.

    If sigma <= 0, returns the image unchanged.
    """
    if sigma <= 0:
        return img

    # Kernel radius: 3*sigma, at least 1
    radius = max(int(3.0 * sigma + 0.5), 1)
    ks = 2 * radius + 1

    # 1D Gaussian kernel
    coords = torch.arange(ks, dtype=img.dtype, device=img.device) - radius
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    # (H, W, C) -> (1, C, H, W)
    x = img.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    # Separable convolution: blur rows then columns
    k_h = g.view(1, 1, ks, 1).expand(C, 1, ks, 1)  # (C,1,ks,1)
    k_w = g.view(1, 1, 1, ks).expand(C, 1, 1, ks)  # (C,1,1,ks)

    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, k_h, groups=C)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k_w, groups=C)

    return x.squeeze(0).permute(1, 2, 0)


# ═══════════════════════════════════════════════════════════════════
# 2-level wavelet loss: L1 on all subbands at both levels
# ═══════════════════════════════════════════════════════════════════

def _multi_scale_dwt_loss(pred: Tensor, target: Tensor, levels: int = 2) -> Tensor:
    """Compute L1 loss across all subbands of a multi-level Haar DWT.

    At each level, computes L1 on LL, LH, HL, HH subbands and sums.
    The next level decomposes the LL subband.
    """
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    current_pred = pred
    current_target = target
    for _ in range(levels):
        p_LL, p_LH, p_HL, p_HH = haar_dwt_2d(current_pred)
        t_LL, t_LH, t_HL, t_HH = haar_dwt_2d(current_target)

        loss = loss + F.l1_loss(p_LL, t_LL)
        loss = loss + F.l1_loss(p_LH, t_LH)
        loss = loss + F.l1_loss(p_HL, t_HL)
        loss = loss + F.l1_loss(p_HH, t_HH)

        # Next level decomposes the LL subband
        current_pred = p_LL
        current_target = t_LL

    return loss


# ═══════════════════════════════════════════════════════════════════
# Sobel gradient-difference loss
# ═══════════════════════════════════════════════════════════════════

def _sobel_gdc_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L_gdc = L1(sobel(rendered) - sobel(GT))."""
    pred_edges = _sobel_edges(pred)
    target_edges = _sobel_edges(target)
    return F.l1_loss(pred_edges, target_edges)


# ═══════════════════════════════════════════════════════════════════
# Model Config + Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WaveletLossGSModelConfig(GSTemplateModelConfig):
    """Multi-Scale Wavelet + Edge Loss GS configuration."""

    _target: Type = field(default_factory=lambda: WaveletLossGSModel)

    # -- DWT loss ----------------------------------------------------------
    dwt_loss_weight: float = 0.005
    """Weight for multi-scale wavelet loss (L_dwt)."""

    dwt_levels: int = 2
    """Number of Haar DWT decomposition levels."""

    # -- Sobel GDC loss ----------------------------------------------------
    gdc_loss_weight: float = 0.002
    """Weight for Sobel gradient-difference loss (L_gdc)."""

    # -- Coarse-to-fine schedule -------------------------------------------
    blur_sigma_start: float = 2.0
    """Initial Gaussian blur sigma applied to GT (coarse-to-fine)."""

    blur_sigma_end: float = 0.0
    """Final Gaussian blur sigma (0 = no blur)."""

    blur_decay_end_step: int = 20000
    """Iteration at which blur sigma reaches blur_sigma_end."""

    freq_loss_activate_step: int = 5000
    """Iteration at which wavelet and GDC losses activate."""

    freq_loss_warmup_end_step: int = 8000
    """Iteration at which warmup ramp reaches full weight."""


class WaveletLossGSModel(GSTemplateModel):
    """Multi-Scale Wavelet + Edge Loss Gaussian Splatting.

    Overrides only compute_custom_losses(). Everything else
    (rendering, densification, etc.) is inherited from splatfacto.

    Loss schedule:
        - Iters 0-5K: Only base L1+SSIM with blurred GT (sigma 2.0 -> ~0.5)
        - Iters 5K-8K: Wavelet and GDC losses ramp up linearly; blur continues decaying
        - Iters 8K-20K: Full wavelet+GDC weights; blur decays to 0
        - Iters 20K+: All losses at full weight, no blur
    """

    config: WaveletLossGSModelConfig

    def _get_blur_sigma(self) -> float:
        """Compute current Gaussian blur sigma via linear decay."""
        step = self.step
        if step >= self.config.blur_decay_end_step:
            return self.config.blur_sigma_end
        progress = step / max(self.config.blur_decay_end_step, 1)
        return self.config.blur_sigma_start + progress * (
            self.config.blur_sigma_end - self.config.blur_sigma_start
        )

    def _get_freq_loss_weight_mult(self) -> float:
        """Compute warmup multiplier [0, 1] for wavelet and GDC losses."""
        step = self.step
        if step < self.config.freq_loss_activate_step:
            return 0.0
        if step >= self.config.freq_loss_warmup_end_step:
            return 1.0
        ramp_len = max(
            self.config.freq_loss_warmup_end_step - self.config.freq_loss_activate_step,
            1,
        )
        return (step - self.config.freq_loss_activate_step) / ramp_len

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Wavelet + Sobel GDC losses with coarse-to-fine schedule."""
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

        # -- Coarse-to-fine: blur GT image ---------------------------------
        sigma = self._get_blur_sigma()
        if sigma > 0:
            gt_rgb_for_freq = _gaussian_blur(gt_rgb, sigma)
        else:
            gt_rgb_for_freq = gt_rgb

        # -- Warmup multiplier for frequency losses ------------------------
        warmup = self._get_freq_loss_weight_mult()

        # -- 1. Multi-scale wavelet loss -----------------------------------
        if warmup > 0 and self.config.dwt_loss_weight > 0:
            dwt_loss = _multi_scale_dwt_loss(
                pred_rgb, gt_rgb_for_freq, levels=self.config.dwt_levels
            )
            losses["dwt_loss"] = self.config.dwt_loss_weight * warmup * dwt_loss

        # -- 2. Sobel gradient-difference loss -----------------------------
        if warmup > 0 and self.config.gdc_loss_weight > 0:
            gdc_loss = _sobel_gdc_loss(pred_rgb, gt_rgb_for_freq)
            losses["gdc_loss"] = self.config.gdc_loss_weight * warmup * gdc_loss

        return losses

    def get_metrics_dict(self, outputs, batch) -> Dict[str, float]:
        """Log wavelet/GDC loss values and schedule state."""
        metrics = super().get_metrics_dict(outputs, batch)

        with torch.no_grad():
            metrics["blur_sigma"] = self._get_blur_sigma()
            metrics["freq_warmup"] = self._get_freq_loss_weight_mult()

        return metrics
