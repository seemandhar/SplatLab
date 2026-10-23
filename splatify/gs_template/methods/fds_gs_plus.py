"""
FDS-GS-Plus — Frequency-Aligned Densification and Supervision for 3D Gaussian Splatting
=========================================================================================
Complexity: ★★★☆☆ (custom losses, densification modifications, no rendering changes)

Extends the FDS-GS framework (AAAI 2025) by aligning frequency-aware density
control with frequency-aware supervision losses. Key components:

    1. Density-scale coupling loss: soft constraint encouraging Gaussians in
       dense regions to have small scales, using KNN-based local density.
    2. Dynamic gradient-percentile densification threshold: replaces the fixed
       gradient threshold with the P-th percentile of accumulated gradients.
    3. Scale-based confidence filter: prunes overscaled Gaussians during
       refinement steps.
    4. DWT low-frequency loss: multi-level Haar DWT LL-subband L1 loss
       (from gs_template.losses).
    5. Gradient-difference loss: edge fidelity loss activated after step 7500
       (from gs_template.losses).

Demonstrates:
    - Modifying densification strategy parameters dynamically (grow_grad2d)
    - Additional pruning logic in step_post_backward via gsplat's remove()
    - Multi-loss scheduling with step-dependent activation and weight decay
    - KNN-based density estimation computed periodically for efficiency

Train:
    ns-train fds-gs-plus --data <path>
    ns-train fds-gs-plus --data <path> --pipeline.model.dwt_lf_weight 0.3
    ns-train fds-gs-plus --data <path> --pipeline.model.grad_percentile 90
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove as gsplat_remove

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import dwt_lf_loss, gradient_difference_loss


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FDSGSPlusModelConfig(GSTemplateModelConfig):
    """FDS-GS-Plus: frequency-aligned densification and supervision."""

    _target: Type = field(default_factory=lambda: FDSGSPlusModel)

    # ── Density-scale coupling loss ────────────────────────────────
    density_scale_weight: float = 0.01
    """Weight for the density-scale coupling loss L_density."""

    density_scale_weight_late: float = 0.005
    """Reduced weight for density-scale loss after fine-tuning phase begins."""

    density_scale_late_step: int = 15000
    """Step at which to reduce density-scale loss weight."""

    knn_k: int = 16
    """Number of nearest neighbors for local density estimation."""

    density_update_interval: int = 100
    """Recompute KNN density every N steps (matches refine_every)."""

    density_constant_percentile: float = 50.0
    """Percentile for calibrating the density constant C from initial
    point cloud. C = median(initial_spacing * density^(1/3))."""

    # ── Dynamic gradient-percentile densification threshold ────────
    use_dynamic_threshold: bool = True
    """Replace fixed grow_grad2d with dynamic P-th percentile."""

    grad_percentile: float = 90.0
    """Percentile P for dynamic densification threshold (0-100).
    Higher = more conservative densification."""

    # ── Scale-based confidence filter ──────────────────────────────
    use_scale_filter: bool = True
    """Enable pruning of overscaled Gaussians during refinement."""

    scale_filter_alpha: float = 2.0
    """Tolerance factor for scale filter. Gaussians with
    s_abs > alpha * C / density_k^(1/3) are pruned."""

    # ── DWT low-frequency loss ─────────────────────────────────────
    dwt_lf_weight: float = 0.3
    """Weight for multi-level Haar DWT LL-subband L1 loss."""

    dwt_levels: int = 2
    """Number of DWT decomposition levels for LF loss."""

    # ── Gradient-difference loss ───────────────────────────────────
    grad_diff_weight: float = 0.05
    """Weight for gradient-difference (edge fidelity) loss."""

    grad_diff_start_step: int = 7500
    """Activate gradient-difference loss after this step."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class FDSGSPlusModel(GSTemplateModel):
    """FDS-GS-Plus: frequency-aligned densification and supervision.

    Overrides:
        - populate_modules(): initialize KNN state and density constant C
        - compute_custom_losses(): density-scale + DWT LF + gradient-diff losses
        - step_post_backward(): dynamic percentile threshold + scale filter pruning
    """

    config: FDSGSPlusModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()

        # Cached density values per Gaussian (recomputed periodically)
        self._cached_density: Optional[Tensor] = None
        self._cached_density_step: int = -1

        # Calibrate density constant C from initial point cloud
        self._density_constant_C: float = self._calibrate_density_constant()

        # Store original threshold for fallback
        self._original_grow_grad2d: Optional[float] = None

    def _calibrate_density_constant(self) -> float:
        """Calibrate density constant C from initial point cloud spacing.

        C = percentile(s_abs_init * density_k^(1/3)) where s_abs_init
        is the initial absolute scale (geometric mean of exp(scales)).
        """
        with torch.no_grad():
            means = self.means.detach()
            n = means.shape[0]

            if n < self.config.knn_k + 1:
                return 1.0

            # Compute KNN distances for a subset to save memory
            sample_size = min(n, 10000)
            if n > sample_size:
                indices = torch.randperm(n, device=means.device)[:sample_size]
                sample_means = means[indices]
            else:
                indices = None
                sample_means = means

            # Compute pairwise distances
            dists = torch.cdist(sample_means, means)  # (sample, N)
            # Get K+1 nearest (including self), take K nearest excluding self
            knn_dists, _ = dists.topk(
                self.config.knn_k + 1, dim=1, largest=False
            )
            knn_dists = knn_dists[:, 1:]  # exclude self

            # Local density: K / volume of KNN ball
            max_knn_dist = knn_dists[:, -1].clamp(min=1e-8)  # radius
            volume = (4.0 / 3.0) * math.pi * max_knn_dist ** 3
            density = self.config.knn_k / volume  # (sample,)

            # Initial absolute scale = geometric mean of exp(scales)
            scales_exp = torch.exp(self.scales.detach())
            if indices is not None:
                scales_sample = scales_exp[indices]
            else:
                scales_sample = scales_exp
            s_abs = scales_sample.prod(dim=-1).pow(1.0 / 3.0)  # (sample,)

            # C = percentile(s_abs * density^(1/3))
            product = s_abs * density.pow(1.0 / 3.0)
            p = self.config.density_constant_percentile / 100.0
            k_idx = int(p * product.shape[0])
            k_idx = min(max(k_idx, 0), product.shape[0] - 1)
            sorted_product, _ = product.sort()
            C = sorted_product[k_idx].item()

            # Clamp to reasonable range
            C = max(C, 0.01)
            return C

    # ── 2. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        pred_rgb = outputs["rgb"]  # (H, W, 3)
        gt_rgb = batch["image"].to(pred_rgb.device).float()

        # Normalize uint8 [0,255] -> [0,1] if needed
        if gt_rgb.max() > 1.0:
            gt_rgb = gt_rgb / 255.0

        # Handle shape mismatch from training downscale
        if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
            gt_rgb = F.interpolate(
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        # ── 1. DWT low-frequency loss (always active) ─────────────
        losses["dwt_lf_loss"] = (
            self.config.dwt_lf_weight
            * dwt_lf_loss(pred_rgb, gt_rgb, levels=self.config.dwt_levels)
        )

        # ── 2. Gradient-difference loss (activated after step 7500) ─
        if self.step >= self.config.grad_diff_start_step:
            losses["grad_diff_loss"] = (
                self.config.grad_diff_weight
                * gradient_difference_loss(pred_rgb, gt_rgb)
            )

        # ── 3. Density-scale coupling loss ─────────────────────────
        density_loss = self._compute_density_scale_loss()
        if density_loss is not None:
            # Use reduced weight after fine-tuning phase
            weight = (
                self.config.density_scale_weight_late
                if self.step >= self.config.density_scale_late_step
                else self.config.density_scale_weight
            )
            losses["density_scale_loss"] = weight * density_loss

        return losses

    def _compute_density_scale_loss(self) -> Optional[Tensor]:
        """Compute density-scale coupling loss.

        L_density = mean(max(s_abs * density_k^(1/3) - C, 0)^2)

        Encourages Gaussians in dense regions to have appropriately
        small scales.
        """
        n = self.num_points
        if n < self.config.knn_k + 1:
            return None

        # Get or recompute cached density
        density = self._get_density()
        if density is None:
            return None

        # Compute absolute scale: geometric mean of exp(scales)
        scales_exp = torch.exp(self.scales)  # (N, 3)
        s_abs = scales_exp.prod(dim=-1).pow(1.0 / 3.0)  # (N,)

        # Coupling constraint: s_abs * density^(1/3) should be <= C
        density_cbrt = density.pow(1.0 / 3.0)
        violation = s_abs * density_cbrt - self._density_constant_C
        loss = torch.clamp(violation, min=0.0).pow(2).mean()

        return loss

    @torch.no_grad()
    def _get_density(self) -> Optional[Tensor]:
        """Get cached KNN density, recomputing if stale."""
        if (
            self._cached_density is not None
            and self._cached_density.shape[0] == self.num_points
            and (self.step - self._cached_density_step)
            < self.config.density_update_interval
        ):
            return self._cached_density

        return self._compute_knn_density()

    @torch.no_grad()
    def _compute_knn_density(self) -> Optional[Tensor]:
        """Compute KNN-based local density for all Gaussians.

        density_k = K / volume_of_KNN_ball

        For efficiency, uses batch computation to avoid OOM on large
        point clouds.
        """
        means = self.means.detach()
        n = means.shape[0]

        if n < self.config.knn_k + 1:
            return None

        k = self.config.knn_k
        density = torch.zeros(n, device=means.device)

        # Process in batches to limit memory
        batch_size = min(n, 8192)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_means = means[start:end]  # (B, 3)

            # Compute distances from batch to all points
            dists = torch.cdist(batch_means, means)  # (B, N)

            # Get K+1 nearest (including self)
            knn_dists, _ = dists.topk(k + 1, dim=1, largest=False)
            knn_dists = knn_dists[:, 1:]  # exclude self, (B, K)

            # Density = K / volume of sphere with radius = max KNN distance
            max_dist = knn_dists[:, -1].clamp(min=1e-8)
            volume = (4.0 / 3.0) * math.pi * max_dist ** 3
            density[start:end] = k / volume

        self._cached_density = density
        self._cached_density_step = self.step
        return density

    # ── 3. step_post_backward (dynamic threshold + scale filter) ──

    def step_post_backward(self, step):
        """Override to inject dynamic percentile threshold and scale filter.

        Modifies the strategy's grow_grad2d parameter to use the P-th
        percentile of accumulated gradients before the standard
        densification runs. After standard densification, applies
        additional scale-based confidence pruning via gsplat's remove().
        """
        assert step == self.step

        if not isinstance(self.strategy, DefaultStrategy):
            super().step_post_backward(step)
            return

        # Save original threshold on first call
        if self._original_grow_grad2d is None:
            self._original_grow_grad2d = self.strategy.grow_grad2d

        # Check if this is a refinement step
        is_refine_step = (
            step > self.strategy.refine_start_iter
            and step < self.strategy.refine_stop_iter
            and step % self.strategy.refine_every == 0
        )

        # ── Dynamic percentile threshold (before densification) ───
        if self.config.use_dynamic_threshold and is_refine_step:
            self._apply_dynamic_threshold()

        # Let the default strategy run its densification + pruning
        super().step_post_backward(step)

        # ── Scale-based confidence filter (after standard pruning) ─
        if self.config.use_scale_filter and is_refine_step:
            self._apply_scale_filter()

    @torch.no_grad()
    def _apply_dynamic_threshold(self):
        """Set grow_grad2d to the P-th percentile of accumulated gradients.

        The gradient state is accumulated by DefaultStrategy._update_state()
        between refine steps. We read it and compute the adaptive threshold.
        """
        state = self.strategy_state
        grad2d = state.get("grad2d")
        count = state.get("count")

        if grad2d is None or count is None:
            return

        # Average gradients over accumulation window
        valid = count > 0
        if valid.sum() < 10:
            return

        avg_grads = torch.zeros_like(grad2d)
        avg_grads[valid] = grad2d[valid] / count[valid]

        # Only consider Gaussians that were actually observed
        observed_grads = avg_grads[valid]

        # Compute P-th percentile
        p = self.config.grad_percentile / 100.0
        k = int(p * observed_grads.shape[0])
        k = min(max(k, 0), observed_grads.shape[0] - 1)

        sorted_grads, _ = observed_grads.sort()
        threshold = sorted_grads[k].item()

        # Clamp to reasonable range (don't go below 1e-6 or above 0.01)
        threshold = max(threshold, 1e-6)
        threshold = min(threshold, 0.01)

        self.strategy.grow_grad2d = threshold

    @torch.no_grad()
    def _apply_scale_filter(self):
        """Prune Gaussians whose absolute scale exceeds the density-implied maximum.

        Uses gsplat's remove() function for proper optimizer state management.
        Removes Gaussians where: s_abs > alpha * C / density_k^(1/3)
        """
        n = self.num_points
        if n < self.config.knn_k + 1:
            return

        # Recompute density after standard densification may have changed
        # the Gaussian count
        density = self._compute_knn_density()
        if density is None:
            return

        # Compute absolute scale: geometric mean of exp(scales)
        scales_exp = torch.exp(self.scales.detach())
        s_abs = scales_exp.prod(dim=-1).pow(1.0 / 3.0)  # (N,)

        # Maximum allowed scale
        density_cbrt = density.pow(1.0 / 3.0).clamp(min=1e-8)
        max_scale = (
            self.config.scale_filter_alpha
            * self._density_constant_C
            / density_cbrt
        )

        # Identify overscaled Gaussians
        overscaled_mask = s_abs > max_scale

        n_prune = overscaled_mask.sum().item()
        if n_prune == 0 or n_prune >= n - 10:
            # Don't prune everything or nothing meaningful
            return

        # Use gsplat's remove() for proper param + optimizer state management
        gsplat_remove(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            mask=overscaled_mask,
        )

        # Invalidate density cache since Gaussian count changed
        self._cached_density = None

    # ── 4. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["density_constant_C"] = torch.tensor(
            self._density_constant_C, dtype=torch.float32
        )
        return metrics
