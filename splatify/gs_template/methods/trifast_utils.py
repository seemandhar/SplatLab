"""
trifast_utils.py — Utility helpers for TriFast
================================================

Standalone, import-safe utilities used by TriFastModel:

    AdaptiveViewSampler      — per-view EMA error tracking (simplified)
    HaarDWT3Level            — 3-level Haar DWT as fixed conv2d kernels
    WaveletFrequencyLoss     — WFL: L1 on HH/HL/LH sub-bands with warm-up
    FrequencySaliencyPruner  — FSP: per-Gaussian saliency from DWT cache

Design notes
------------
- No new pip dependencies: only torch + existing gs_template imports.
- Haar filters are registered as module buffers so they follow device
  moves automatically and are not trained.
- DWT cache is shared between WFL and FSP to avoid redundant computation.
- All public methods handle edge cases (empty dicts, tiny images, etc.).
- AdaptiveViewSampler is simplified: only EMA tracking + hard-view lookup.
  The K-view softmax sampling was removed (ACVD fix: render 1 hard view
  every 10 steps instead of K=6..18 views every step).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# 1. Adaptive View Sampler — per-view EMA error tracking
# ═══════════════════════════════════════════════════════════════════

class AdaptiveViewSampler:
    """Maintains per-view EMA reconstruction errors.

    Simplified from the original ACVD implementation:
    - Only tracks EMA errors per camera index.
    - Provides lookup for the hardest view (max EMA error).
    - No K-view softmax sampling (removed to avoid training time blowup).

    The model calls update() each step and uses view_errors to find the
    hardest camera index for the cheap L1 supervision every 10 steps.

    Args:
        ema_alpha: EMA momentum for U_i (default 0.1 → 0.9 decay).
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        # Legacy parameters kept for backward-compat / config loading
        k_min: int = 6,
        k_max: int = 18,
        tau_max: float = 0.1,
        tau_min: float = 0.01,
    ) -> None:
        self.ema_alpha = ema_alpha
        # view_errors: {cam_idx: float}  — plain Python dict, NOT a tensor
        self.view_errors: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, cam_idx: int, l1_error: float) -> None:
        """Update EMA for camera `cam_idx` with new L1 error."""
        if cam_idx in self.view_errors:
            self.view_errors[cam_idx] = (
                (1.0 - self.ema_alpha) * self.view_errors[cam_idx]
                + self.ema_alpha * l1_error
            )
        else:
            self.view_errors[cam_idx] = l1_error

    def hardest_view(self) -> Optional[int]:
        """Return the camera index with the highest EMA error, or None."""
        if not self.view_errors:
            return None
        return max(self.view_errors, key=self.view_errors.get)

    def state_dict(self) -> Dict:
        return {"view_errors": dict(self.view_errors)}

    def load_state_dict(self, d: Dict) -> None:
        self.view_errors = dict(d.get("view_errors", {}))


# ═══════════════════════════════════════════════════════════════════
# 2. Haar DWT — 3-level, fixed conv2d kernels (no pywavelets)
# ═══════════════════════════════════════════════════════════════════

class HaarDWT3Level(nn.Module):
    """3-level 2D Haar Discrete Wavelet Transform via fixed conv2d.

    Filters are registered as buffers (device-portable, not trained).
    Each level decomposes the LL sub-band from the previous level.

    Input:  (H, W, C)  or  (1, C, H, W)  image tensor.
    Output: dict with keys like "L1_LL", "L1_LH", "L1_HL", "L1_HH",
            "L2_LL", ... , "L3_HH".

    The LL sub-band at each level is used as input to the next level.
    Odd-dimension inputs are reflected-padded to even dimensions.
    """

    def __init__(self) -> None:
        super().__init__()

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        lo = torch.tensor([inv_sqrt2, inv_sqrt2])   # low-pass
        hi = torch.tensor([inv_sqrt2, -inv_sqrt2])  # high-pass

        # Build (1, 1, 2, 2) 2D kernels for each sub-band
        # LL: lo x lo,  LH: hi x lo,  HL: lo x hi,  HH: hi x hi
        # Convention: first index = row direction, second = col direction
        for name, row_f, col_f in [
            ("k_ll", lo, lo),
            ("k_lh", hi, lo),
            ("k_hl", lo, hi),
            ("k_hh", hi, hi),
        ]:
            # outer product: (2,) x (2,) → (2,2)
            k2d = row_f.unsqueeze(1) * col_f.unsqueeze(0)  # (2, 2)
            self.register_buffer(name, k2d.view(1, 1, 2, 2))

    # ------------------------------------------------------------------

    def _dwt_single_level(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One level of 2D Haar DWT.

        Args:
            x: (1, C, H, W)  float tensor.

        Returns:
            LL, LH, HL, HH  — each (1, C, H//2, W//2).
        """
        B, C, H, W = x.shape

        # Pad to even dimensions
        ph = H % 2
        pw = W % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        # Grouped conv: each of C channels convolved independently
        def _conv(k):
            # k is (1,1,2,2); expand to (C,1,2,2) for groups=C
            return F.conv2d(x, k.expand(C, 1, 2, 2), stride=2, groups=C)

        LL = _conv(self.k_ll)
        LH = _conv(self.k_lh)
        HL = _conv(self.k_hl)
        HH = _conv(self.k_hh)
        return LL, LH, HL, HH

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute 3-level DWT.

        Args:
            x: (H, W, C) or (1, C, H, W).  Values should be in [0, 1].

        Returns:
            Dict with 12 entries: L{1,2,3}_{LL,LH,HL,HH}.
            All subbands stored as (1, C, h, w) tensors.
        """
        # Normalise input to (1, C, H, W)
        if x.dim() == 3:
            # (H, W, C) → (1, C, H, W)
            x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        elif x.dim() == 4 and x.shape[0] == 1 and x.shape[-1] <= 4:
            # (1, H, W, C) — convert; guard x.shape[0]==1 so (B,C,H,W)
            # tensors with small C are not accidentally treated as channel-last
            x = x.permute(0, 3, 1, 2).contiguous()
        # else: already (1, C, H, W)

        x = x.float()
        result: Dict[str, Tensor] = {}

        current = x
        for level in range(1, 4):
            # Check if spatial dims are large enough for DWT
            H, W = current.shape[2], current.shape[3]
            if H < 2 or W < 2:
                # Cannot decompose further — fill remaining levels with zeros
                for remaining in range(level, 4):
                    for sb in ("LL", "LH", "HL", "HH"):
                        result[f"L{remaining}_{sb}"] = torch.zeros(
                            1, current.shape[1], 1, 1,
                            device=current.device, dtype=current.dtype,
                        )
                break

            LL, LH, HL, HH = self._dwt_single_level(current)
            result[f"L{level}_LL"] = LL
            result[f"L{level}_LH"] = LH
            result[f"L{level}_HL"] = HL
            result[f"L{level}_HH"] = HH
            current = LL  # cascade: feed LL into next level

        return result


# ═══════════════════════════════════════════════════════════════════
# 3. Wavelet Frequency Loss (WFL)
# ═══════════════════════════════════════════════════════════════════

class WaveletFrequencyLoss(nn.Module):
    """WFL: L1 on HH/HL/LH sub-bands at 3 DWT levels with warm-up.

    Level weights:  mu_l = 2^(-(l-1))  →  [1, 0.5, 0.25]
    Only HF sub-bands (HH, HL, LH) contribute; LL is skipped.
    Total loss:
        L_wav = sum_{l=1}^{3} mu_l * (L1(HH_l) + L1(HL_l) + L1(LH_l))

    Warm-up: lambda_w ramps linearly from 0 to lambda_w_max
             between warmup_start and warmup_end steps.

    DWT results are cached internally after each forward call so that
    the FrequencySaliencyPruner can reuse them without re-computing.

    Args:
        lambda_w:      Target wavelet weight (default 0.02; lowered from
                       0.05 to prevent over-correction / blurring).
        warmup_start:  Step at which lambda_w starts ramping (default 1000).
        warmup_end:    Step at which lambda_w reaches lambda_w (default 5000).
    """

    # Level weights: mu_l = 2^(-(l-1))
    LEVEL_WEIGHTS = [1.0, 0.5, 0.25]

    def __init__(
        self,
        lambda_w: float = 0.02,
        warmup_start: int = 1000,
        warmup_end: int = 5000,
    ) -> None:
        super().__init__()
        self.lambda_w = lambda_w
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

        self.dwt = HaarDWT3Level()

        # Cache: set after each forward call, cleared after FSP consumes it.
        self._cache_pred: Optional[Dict[str, Tensor]] = None
        self._cache_gt: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------

    def effective_lambda(self, step: int) -> float:
        """Linearly ramped lambda_w for the current step."""
        if step < self.warmup_start:
            return 0.0
        if step >= self.warmup_end:
            return self.lambda_w
        ratio = (step - self.warmup_start) / max(self.warmup_end - self.warmup_start, 1)
        return self.lambda_w * ratio

    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        step: int,
    ) -> Tuple[Tensor, float]:
        """Compute WFL and cache DWT results.

        Args:
            pred: (H, W, 3) rendered image in [0, 1].
            gt:   (H, W, 3) ground-truth image in [0, 1].
            step: current training step.

        Returns:
            (loss_tensor, effective_lambda)
            loss_tensor is scalar, already weighted by effective_lambda.
        """
        lam = self.effective_lambda(step)
        device = pred.device

        if lam < 1e-9:
            # Warm-up not reached: still run DWT to populate cache for FSP
            with torch.no_grad():
                self._cache_pred = self.dwt(pred.detach())
                self._cache_gt = self.dwt(gt.detach())
            return torch.tensor(0.0, device=device), lam

        # Full DWT — pred must carry gradients, gt is detached
        self._cache_pred = self.dwt(pred)
        with torch.no_grad():
            self._cache_gt = self.dwt(gt.detach())

        # Initialize as a zero tensor tied to the computation graph via pred,
        # so that the returned loss always participates in backprop even when
        # the loop adds nothing (e.g. cache unexpectedly empty).
        loss = pred.sum() * 0.0
        for l_idx, mu in enumerate(self.LEVEL_WEIGHTS):
            lvl = l_idx + 1
            for sb in ("HH", "HL", "LH"):
                key = f"L{lvl}_{sb}"
                if key in self._cache_pred and key in self._cache_gt:
                    loss = loss + mu * F.l1_loss(self._cache_pred[key], self._cache_gt[key])

        return lam * loss, lam

    def pop_cache(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return and clear the DWT cache (consumed by FSP)."""
        pred_cache = self._cache_pred
        gt_cache = self._cache_gt
        self._cache_pred = None
        self._cache_gt = None
        return pred_cache, gt_cache


# ═══════════════════════════════════════════════════════════════════
# 4. Frequency-Saliency Pruning (FSP)
# ═══════════════════════════════════════════════════════════════════

class FrequencySaliencyPruner:
    """FSP: prune Gaussians that are low-opacity AND low HF saliency.

    Per-Gaussian saliency is estimated by projecting each Gaussian's
    2D screen-space footprint onto the HF energy map (sum of HH+HL+LH
    from the WFL DWT cache) and averaging the energy under that footprint.

    Dual pruning condition:
        prune if  opacity_i < alpha_thresh
               AND saliency_i < S_thresh(P%)

    S_thresh = P-th percentile of all Gaussian saliencies, where
    P anneals linearly from P_start (5%) to P_end (20%).

    Schedule:
        - Run every `prune_every` steps.
        - Active from `prune_start` to `prune_until` (fraction of T).
        - prune_start raised to 2000 (from 500) to let Gaussians densify
          before pruning begins (Issue 3 fix).

    Args:
        alpha_thresh:  Opacity threshold (default 0.005).
        p_start:       Starting percentile for S_thresh (default 5.0).
        p_end:         Ending percentile for S_thresh (default 20.0).
        prune_every:   Steps between pruning runs (default 100).
        prune_start:   First step to prune (default 2000).
        prune_until:   Fraction of total_steps after which pruning stops
                       (default 0.7, i.e., 70% of training).
    """

    def __init__(
        self,
        alpha_thresh: float = 0.005,
        p_start: float = 5.0,
        p_end: float = 20.0,
        prune_every: int = 100,
        prune_start: int = 2000,
        prune_until: float = 0.7,
    ) -> None:
        self.alpha_thresh = alpha_thresh
        self.p_start = p_start
        self.p_end = p_end
        self.prune_every = prune_every
        self.prune_start = prune_start
        self.prune_until = prune_until

    # ------------------------------------------------------------------

    def should_run(self, step: int, total_steps: int) -> bool:
        """Return True if FSP should run at this step."""
        if step < self.prune_start:
            return False
        if total_steps > 0 and step > int(self.prune_until * total_steps):
            return False
        return step % self.prune_every == 0

    def current_percentile(self, step: int, total_steps: int) -> float:
        """Linearly anneal P from p_start to p_end over training."""
        if total_steps <= 0:
            return self.p_start
        ratio = min(step / max(total_steps, 1), 1.0)
        return self.p_start + (self.p_end - self.p_start) * ratio

    def compute_saliency(
        self,
        xys: Optional[Tensor],
        radii: Optional[Tensor],
        dwt_cache: Optional[Dict[str, Tensor]],
        image_h: int,
        image_w: int,
    ) -> Optional[Tensor]:
        """Compute per-Gaussian HF saliency from DWT cache.

        Saliency is the average HF energy (sum over HH+HL+LH at all levels)
        within each Gaussian's 2D screen footprint.

        Args:
            xys:       (N, 2) 2D projected Gaussian centres (pixels), or None.
            radii:     (N,) or (N, 1) projected radii in pixels, or None.
            dwt_cache: DWT sub-bands from WaveletFrequencyLoss (may be None).
            image_h:   Image height (for the full-resolution image, pre-DWT).
            image_w:   Image width.

        Returns:
            (N,) float tensor of per-Gaussian saliency, or None if unavailable.
        """
        if xys is None or radii is None or dwt_cache is None:
            return None

        # Build HF energy map from DWT cache (level 1 — highest resolution)
        # We use L1 sub-bands only (HH, HL, LH) for saliency.
        # The L1 sub-bands have spatial size (H//2, W//2).
        hf_maps = []
        for sb in ("L1_HH", "L1_HL", "L1_LH"):
            if sb in dwt_cache:
                t = dwt_cache[sb]  # (1, C, h, w)
                if t.numel() > 0:
                    # Average over channels, take abs → (1, 1, h, w)
                    hf_maps.append(t.abs().mean(dim=1, keepdim=True))

        if not hf_maps:
            return None

        # Sum all HF sub-band energies → (1, 1, h, w)
        hf_energy = torch.stack(hf_maps, dim=0).sum(dim=0)  # (1, 1, h, w)

        # Upsample to full image resolution for direct pixel lookup
        # The DWT halved each spatial dim, so upsample by 2x.
        if hf_energy.shape[2] < image_h or hf_energy.shape[3] < image_w:
            hf_energy = F.interpolate(
                hf_energy,
                size=(image_h, image_w),
                mode="bilinear",
                align_corners=False,
            )
        hf_energy = hf_energy.squeeze(0).squeeze(0)  # (H, W)

        N = xys.shape[0]
        if N == 0:
            return torch.zeros(0, device=xys.device)

        # Detach everything — saliency is used for pruning, not for gradients
        with torch.no_grad():
            xy = xys.detach()
            r = radii.detach().float()
            if r.dim() > 1:
                r = r.squeeze(-1)

            # For each Gaussian, sample the average HF energy in its footprint.
            # Approach: grid_sample with a bounding box of size 2r x 2r.
            # We use a vectorised approximation: evaluate at center + 8 neighbours
            # and average, which is fast and avoids the per-Gaussian loop.
            # Offset grid in pixel units: scale to [-1, 1] for grid_sample.
            H, W = hf_energy.shape

            # Pixel coords of 9 sampling points per Gaussian (center + 8 ring)
            offsets = torch.tensor(
                [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                 [1, 1], [1, -1], [-1, 1], [-1, -1]],
                device=xy.device, dtype=torch.float32,
            )  # (9, 2)

            # r_safe: minimum radius of 1 pixel to avoid point-samples
            r_safe = r.clamp(min=1.0)

            # Build sampling grid: (N, 9, 2) in pixel coords
            # xy: (N, 2) — (x=col, y=row)
            sample_pts = xy.unsqueeze(1) + offsets.unsqueeze(0) * r_safe.unsqueeze(1).unsqueeze(2)
            # sample_pts shape: (N, 9, 2)

            # Normalize to [-1, 1] for grid_sample
            # x → (x / (W-1)) * 2 - 1,  y → (y / (H-1)) * 2 - 1
            norm_x = (sample_pts[..., 0] / max(W - 1, 1)) * 2.0 - 1.0
            norm_y = (sample_pts[..., 1] / max(H - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1)  # (N, 9, 2)

            # grid_sample expects (B, C, H_out, W_out) and grid (B, H_out, W_out, 2)
            # Reshape to (1, 1, N*9, 2) for a single bilinear sample call
            grid_flat = grid.reshape(1, 1, N * 9, 2)  # (1, 1, N*9, 2)
            energy_bchw = hf_energy.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            sampled = F.grid_sample(
                energy_bchw.float(),
                grid_flat,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, 1, 1, N*9)

            sampled = sampled.reshape(N, 9)  # (N, 9)
            saliency = sampled.mean(dim=1)   # (N,)

        return saliency

    def get_prune_mask(
        self,
        opacities_logit: Tensor,
        saliency: Optional[Tensor],
        step: int,
        total_steps: int,
    ) -> Optional[Tensor]:
        """Compute boolean prune mask.

        Returns None if saliency is unavailable (safe no-op).

        Args:
            opacities_logit: (N, 1) raw logit opacities.
            saliency:        (N,) per-Gaussian saliency, or None.
            step:            current step.
            total_steps:     total training steps.

        Returns:
            (N,) bool tensor: True = prune, or None if unavailable.
        """
        if saliency is None or saliency.numel() == 0:
            return None

        N = opacities_logit.shape[0]
        if N == 0 or saliency.shape[0] != N:
            return None

        with torch.no_grad():
            alpha = torch.sigmoid(opacities_logit.squeeze(-1))  # (N,)
            low_opacity = alpha < self.alpha_thresh

            p_pct = self.current_percentile(step, total_steps)
            # Clamp to valid range [0, 100]
            p_pct = max(0.0, min(p_pct, 100.0))

            # Compute S_thresh as p-th percentile of all saliencies
            s_flat = saliency.float().cpu()
            if s_flat.numel() < 2:
                return None
            s_thresh_val = torch.quantile(s_flat, p_pct / 100.0).item()

            low_saliency = saliency < s_thresh_val

            # Dual condition: prune if BOTH conditions met
            prune_mask = low_opacity & low_saliency

        return prune_mask
