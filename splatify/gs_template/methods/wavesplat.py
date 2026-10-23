"""
WaveSplat -- Wavelet-Domain Multi-Scale Supervision with Adaptive Subband Weighting
=====================================================================================
Complexity: ★★☆☆☆ (custom losses, no new Gaussian params, no rendering changes)

Improves 3DGS reconstruction quality by adding spatially-localized multi-scale
frequency supervision using multi-level Haar wavelet decomposition:

    1. Multi-level wavelet subband supervision: 3-level Haar DWT, L1 loss on
       all 12 subbands (LL, LH, HL, HH at each of 3 levels).
    2. Progressive coarse-to-fine scheduling: cosine-annealed weights --
       LL weights high early, HF weights high late.
    3. Convergence-aware weight adjustment: EMA tracking of per-subband loss,
       boost weights for slow-converging subbands.

Demonstrates:
    - Adding multi-level wavelet losses via compute_custom_losses()
    - Progressive training schedule with cosine annealing
    - EMA-based adaptive loss weighting
    - Config fields for loss hyperparameters and schedule control
    - Zero rendering / densification changes -- fully inherited

Train:
    ns-train wavesplat --data <path>
    ns-train wavesplat --data <path> --pipeline.model.dwt_levels 3
    ns-train wavesplat --data <path> --pipeline.model.wavelet_loss_weight 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import haar_dwt_2d


@dataclass
class WaveSplatModelConfig(GSTemplateModelConfig):
    """WaveSplat: wavelet-domain multi-scale supervision with adaptive subband weighting."""

    _target: Type = field(default_factory=lambda: WaveSplatModel)

    # -- Wavelet loss overall weight ------------------------------------------
    wavelet_loss_weight: float = 0.5
    """Overall weight lambda_wav for the combined wavelet loss."""

    # -- DWT decomposition ----------------------------------------------------
    dwt_levels: int = 3
    """Number of Haar DWT decomposition levels (paper default: 3).
    Produces 4 subbands per level = 12 total subband losses."""

    # -- Subband base weights -------------------------------------------------
    ll_weight_base: float = 0.5
    """Base weight for LL (approximation) subbands. Cosine-annealed high->low."""

    hf_weight_base: float = 0.3
    """Base weight for HF (LH, HL, HH) subbands. Cosine-annealed low->high."""

    # -- Convergence-aware adjustment -----------------------------------------
    ema_decay: float = 0.99
    """EMA decay factor for per-subband loss tracking."""

    convergence_gamma: float = 0.5
    """Exponent gamma for convergence-aware weight adjustment.
    Subbands with 4x higher EMA loss get 2x higher weight (sqrt scaling)."""

    # -- Schedule -------------------------------------------------------------
    warmup_steps: int = 500
    """No wavelet loss during initial warmup (let base L1+DSSIM stabilize)."""

    max_training_steps: int = 30000
    """Maximum training steps, used to compute progress ratio p = step/max."""


class WaveSplatModel(GSTemplateModel):
    """WaveSplat -- wavelet-domain multi-scale supervision.

    Overrides only:
        - populate_modules(): initialize EMA loss trackers
        - compute_custom_losses(): multi-level wavelet subband losses with
          progressive scheduling and convergence-aware weighting
        - get_metrics_dict(): log per-subband EMA values for monitoring

    Everything else (rendering, densification, etc.) inherited from splatfacto.
    """

    config: WaveSplatModelConfig

    def populate_modules(self):
        """Initialize EMA loss trackers for each subband at each level."""
        super().populate_modules()

        # EMA trackers: plain dict of floats (NOT nn.Parameters).
        # Keys: "L{level}_{subband}" e.g. "L1_LL", "L1_LH", "L2_HH", etc.
        self._loss_ema: Dict[str, float] = {}
        for level in range(1, self.config.dwt_levels + 1):
            for sb in ["LL", "LH", "HL", "HH"]:
                self._loss_ema[f"L{level}_{sb}"] = 0.0

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute multi-level wavelet subband losses with adaptive weighting.

        Loss formula:
            L_wav = sum_{n=1}^{dwt_levels} sum_{s in {LL,LH,HL,HH}}
                    w_{n,s}(t) * L1(DWT_s^n(pred), DWT_s^n(gt))

        Weight schedule (cosine annealing):
            w_{n,LL}(t) = ll_base / 2^(n-1) * (1 + cos(pi*p)) / 2   (high early)
            w_{n,HF}(t) = hf_base / 2^(n-1) * (1 - cos(pi*p)) / 2   (high late)

        Convergence-aware adjustment:
            w_{n,s}(t) *= (ema_{n,s} / ema_min) ^ gamma
        """
        losses = {}

        # Skip during warmup
        if self.step < self.config.warmup_steps:
            return losses

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

        # Training progress ratio p in [0, 1]
        progress = min(self.step / self.config.max_training_steps, 1.0)

        # Cosine schedule factors
        cos_val = math.cos(math.pi * progress)
        ll_schedule = (1.0 + cos_val) / 2.0   # 1.0 -> 0.0 over training
        hf_schedule = (1.0 - cos_val) / 2.0   # 0.0 -> 1.0 over training

        # Multi-level DWT: cascade through levels, feeding LL into next level
        total_wav_loss = torch.tensor(0.0, device=pred_rgb.device)
        current_pred = pred_rgb
        current_gt = gt_rgb

        for level in range(1, self.config.dwt_levels + 1):
            # Single-level Haar DWT (handles H,W,C input)
            ll_p, lh_p, hl_p, hh_p = haar_dwt_2d(current_pred)
            ll_g, lh_g, hl_g, hh_g = haar_dwt_2d(current_gt)

            # Level scale: deeper levels get lower base weight
            level_scale = 1.0 / (2 ** (level - 1))

            # Compute base weights with cosine schedule
            w_ll = self.config.ll_weight_base * level_scale * ll_schedule
            w_hf = self.config.hf_weight_base * level_scale * hf_schedule

            subbands = [
                (f"L{level}_LL", ll_p, ll_g, w_ll),
                (f"L{level}_LH", lh_p, lh_g, w_hf),
                (f"L{level}_HL", hl_p, hl_g, w_hf),
                (f"L{level}_HH", hh_p, hh_g, w_hf),
            ]

            for name, pred_sb, gt_sb, base_w in subbands:
                sb_loss = F.l1_loss(pred_sb, gt_sb)

                # Update EMA tracker (detached scalar)
                self._loss_ema[name] = (
                    self.config.ema_decay * self._loss_ema[name]
                    + (1.0 - self.config.ema_decay) * sb_loss.item()
                )

                # Convergence-aware weight adjustment
                ema_min = max(min(self._loss_ema.values()), 1e-8)
                ema_ratio = self._loss_ema[name] / ema_min
                conv_weight = ema_ratio ** self.config.convergence_gamma

                total_wav_loss = total_wav_loss + base_w * conv_weight * sb_loss

            # Next level operates on the LL subband (cascaded decomposition)
            current_pred = ll_p
            current_gt = ll_g

        losses["wavelet_loss"] = self.config.wavelet_loss_weight * total_wav_loss
        return losses

    def get_metrics_dict(self, outputs, batch):
        """Log per-subband EMA values for monitoring convergence."""
        metrics = super().get_metrics_dict(outputs, batch)
        for name, ema_val in self._loss_ema.items():
            metrics[f"wav_ema/{name}"] = ema_val
        return metrics
