"""
Micro-Splatting: Multistage Isotropy-informed Covariance Regularization
=========================================================================
Paper: Micro-Splatting
Complexity: ★★★☆☆ (custom losses + pruning/merging)

A two-stage pipeline for compact, high-fidelity 3D Gaussian Splatting:
    Stage I  (0 → T_refine):  Growth — covariance trace regularization +
                               splatfacto densification with absgrad
    Stage II (T_refine → end): Micro-Refine — importance-score pruning +
                               redundancy-driven pairwise merging

IMPLEMENTATION NOTES:
    1. We do NOT override get_loss_dict(). Splatfacto's base loss computation
       couples SSIM (via self.ssim module), bilateral grids, and other internal
       state. Overriding it breaks things. Instead we add all paper losses
       via compute_custom_losses() which the template's get_loss_dict() merges.

    2. The paper's L2 photometric loss is added as an ADDITIONAL term on top
       of splatfacto's (1-λ)*L1 + λ*SSIM. This gives complementary gradients
       near convergence (the paper's key insight about L2).

    3. The paper's gradient-guided splitting uses a custom CUDA densification
       kernel (Kerbl et al. codebase). We approximate this via splatfacto's
       gsplat strategy with absgrad=True, which similarly uses absolute
       gradients for densification decisions.

    4. Pruning/merging use soft opacity zeroing (→ -100) to avoid breaking
       optimizer state. Splatfacto's built-in culling removes dead Gaussians.

Train:
    ns-train micro-splatting --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    covariance_trace_loss,
    covariance_isotropy_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MicroSplattingModelConfig(GSTemplateModelConfig):
    """Micro-Splatting configuration."""

    _target: Type = field(default_factory=lambda: MicroSplattingModel)

    # ── Stage transition ───────────────────────────────────────────
    refine_transition_iter: int = 15000
    """Iteration to transition from Stage I (growth) to Stage II (refine)."""

    # ── L2 photometric loss (paper Section IV.B) ───────────────────
    l2_weight: float = 0.2
    """Weight for L2 photometric loss. Added on top of splatfacto's
    (1-ssim_lambda)*L1 + ssim_lambda*SSIM. Paper uses 0.5 but that's
    relative to their total loss; as an additive term 0.2 gives the
    same error-proportional gradient benefit without overwhelming."""

    # ── Covariance regularization (Stage I + II) ───────────────────
    cov_loss_weight: float = 0.001
    """Weight λ_cov for trace-based covariance regularization.
    Paper says 'moderate'. Must be small enough to not fight
    splatfacto's own scale optimization."""

    trace_cap: float = 0.1
    """Trace cap τ — max allowed covariance trace before penalty.
    Scene-dependent: indoor 0.05-0.1, outdoor 0.1-0.3.
    This is in scale^2 units (eigenvalue space)."""

    # ── Isotropy enforcement ───────────────────────────────────────
    use_isotropy_loss: bool = True
    """Enable isotropy regularization."""

    isotropy_loss_weight: float = 0.0005
    """Weight for isotropy loss. Very small — Gaussians need
    anisotropy for edges, thin structures, and leaves."""

    # ── Stage II: Pruning (Flow A) ─────────────────────────────────
    enable_pruning: bool = True
    """Enable importance-score pruning in Stage II."""

    prune_percent: float = 0.05
    """Fraction of lowest-importance splats to prune (paper: 5-10%)."""

    prune_interval: int = 2000
    """Run pruning every N iterations during Stage II.
    Conservative: ~7 prune passes total in 15k Stage II iters."""

    # ── Stage II: Merging (Flow B) ─────────────────────────────────
    enable_merging: bool = True
    """Enable redundancy-driven merging in Stage II."""

    merge_interval: int = 3000
    """Run merging every N iterations during Stage II.
    Conservative: ~5 merge passes total."""

    merge_spatial_thresh: float = 0.01
    """τ_xyz: max L2 distance between means for merge candidates.
    Tight threshold = only truly overlapping Gaussians merge."""

    merge_color_thresh: float = 0.03
    """τ_col: max L2 distance between DC SH coefficients.
    Tight = only visually identical Gaussians merge."""

    merge_scale_thresh: float = 0.01
    """τ_scale: max L2 distance between activated scale vectors."""

    max_merge_pairs: int = 50000
    """Max candidate pairs per merge pass (for efficiency)."""

    # ── Use absgrad for better densification ───────────────────────
    absgrad: bool = True
    """Absolute gradient accumulation — approximates the paper's
    gradient-guided splitting within splatfacto's framework."""

    # ── Depth during training ──────────────────────────────────────
    output_depth_during_training: bool = True


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class MicroSplattingModel(GSTemplateModel):
    """Micro-Splatting: two-stage compact Gaussian Splatting.

    Stage I  (Growth):       Splatfacto densification + covariance reg
    Stage II (Micro-Refine): Importance pruning + redundancy merging

    Does NOT override get_loss_dict() or get_outputs() rendering.
    All contributions go through compute_custom_losses() + soft ops
    in get_outputs() pre-rendering.
    """

    config: MicroSplattingModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()
        self._stage = 1
        self._last_prune_step = 0
        self._last_merge_step = 0
        self._pending_prune_mask = None
        self._pending_merge_ops = None

    # ── 2. get_outputs — apply pending soft ops before rendering ───

    def get_outputs(self, camera: Cameras):
        """Apply pending prune/merge via in-place .data mods, then render.

        This runs AFTER the previous step's backward + optimizer.step(),
        so the old computation graph is freed and .data is safe to touch.
        """
        if self._pending_prune_mask is not None:
            mask = self._pending_prune_mask
            if mask.any():
                with torch.no_grad():
                    self.opacities.data[mask] = -100.0
            self._pending_prune_mask = None

        if self._pending_merge_ops is not None:
            ops = self._pending_merge_ops
            if ops is not None and len(ops.get("keep", [])) > 0:
                with torch.no_grad():
                    keep_ids = ops["keep"]
                    kill_ids = ops["kill"]
                    for name in self.gauss_params:
                        p = self.gauss_params[name].data
                        p[keep_ids] = (p[keep_ids] + p[kill_ids]) / 2.0
                    self.opacities.data[kill_ids] = -100.0
            self._pending_merge_ops = None

        # Splatfacto's full rendering pipeline — untouched
        return super().get_outputs(camera)

    # ── 3. compute_custom_losses — ALL paper losses go here ────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict=None,
    ) -> Dict[str, Tensor]:
        """Paper's additional losses on top of splatfacto's base.

        Splatfacto already computes: (1-ssim_lambda)*L1 + ssim_lambda*SSIM
        We add:
            - L2 photometric (error-proportional gradients near convergence)
            - L_cov (trace-based covariance regularization)
            - L_iso (isotropy enforcement)

        Also schedules Stage II prune/merge operations.
        """
        losses = {}

        # Update stage
        if self.step >= self.config.refine_transition_iter:
            self._stage = 2

        # ── L2 photometric loss (paper Section IV.B) ───────────────
        # Complements L1: L1 gives constant gradients for large errors,
        # L2 gives error-proportional gradients near convergence.
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]
        losses["l2_photo"] = self.config.l2_weight * F.mse_loss(pred_img, gt_img)

        # ── Covariance trace regularization (both stages) ──────────
        # Hinge penalty: penalize Gaussians with trace(Σ) > τ
        # Encourages compact kernels, prevents bloated splats
        losses["cov_trace"] = (
            self.config.cov_loss_weight
            * covariance_trace_loss(self.scales, self.config.trace_cap)
        )

        # ── Isotropy enforcement (both stages) ─────────────────────
        # Variance of log-scales: 0 = perfectly isotropic
        if self.config.use_isotropy_loss:
            losses["isotropy"] = (
                self.config.isotropy_loss_weight
                * covariance_isotropy_loss(self.scales)
            )

        # ── Stage II: schedule pruning and merging ─────────────────
        if self._stage == 2 and self.training:
            if self.config.enable_pruning:
                self._schedule_prune(self.step)
            if self.config.enable_merging:
                self._schedule_merge(self.step)

        return losses

    # ── Stage II Flow A: Importance-Score Pruning ──────────────────

    @torch.no_grad()
    def _schedule_prune(self, step: int):
        """Importance score: s_k = sigmoid(α_k) * max(exp(σ_k)).
        Prune lowest q% by this score.
        """
        if step - self._last_prune_step < self.config.prune_interval:
            return
        self._last_prune_step = step

        n = self.num_points
        if n < 200:
            return

        opacities = torch.sigmoid(self.opacities.detach()).squeeze(-1)
        scales = torch.exp(self.scales.detach())
        max_scale = scales.max(dim=-1).values
        importance = opacities * max_scale

        n_prune = max(1, int(n * self.config.prune_percent))
        n_prune = min(n_prune, int(n * 0.15))  # Hard cap: max 15% per pass
        if n_prune >= n - 200:
            return

        threshold = torch.kthvalue(importance, n_prune).values
        prune_mask = importance <= threshold

        # Don't prune already-dead Gaussians (waste of mask)
        already_dead = opacities < 0.005
        prune_mask = prune_mask & (~already_dead)

        if prune_mask.sum() > 0:
            self._pending_prune_mask = prune_mask

    # ── Stage II Flow B: Redundancy-Driven Merging ─────────────────

    @torch.no_grad()
    def _schedule_merge(self, step: int):
        """Merge pairs passing spatial + color + scale proximity tests.
        Uses voxel hashing for O(N) candidate generation.
        """
        if step - self._last_merge_step < self.config.merge_interval:
            return
        self._last_merge_step = step

        n = self.num_points
        if n < 200:
            return

        means = self.means.detach()
        features_dc = self.features_dc.detach()
        scales_act = torch.exp(self.scales.detach())
        opacities = torch.sigmoid(self.opacities.detach()).squeeze(-1)

        alive = opacities > 0.01

        voxel_size = self.config.merge_spatial_thresh
        if voxel_size <= 0:
            return

        # Voxel hash for neighbor search
        voxel_coords = torch.floor(means / voxel_size).long()
        primes = torch.tensor(
            [73856093, 19349663, 83492791],
            device=means.device, dtype=torch.long,
        )
        hashes = (voxel_coords * primes).sum(dim=-1)

        sorted_indices = torch.argsort(hashes)
        sorted_hashes = hashes[sorted_indices]

        same_voxel = sorted_hashes[:-1] == sorted_hashes[1:]
        pair_starts = torch.where(same_voxel)[0]

        if len(pair_starts) == 0:
            return

        if len(pair_starts) > self.config.max_merge_pairs:
            perm = torch.randperm(len(pair_starts), device=means.device)
            pair_starts = pair_starts[perm[: self.config.max_merge_pairs]]

        idx_i = sorted_indices[pair_starts]
        idx_j = sorted_indices[pair_starts + 1]

        # Both must be alive
        both_alive = alive[idx_i] & alive[idx_j]
        idx_i = idx_i[both_alive]
        idx_j = idx_j[both_alive]

        if len(idx_i) == 0:
            return

        # Test merge conditions
        spatial_ok = (means[idx_i] - means[idx_j]).norm(dim=-1) <= self.config.merge_spatial_thresh
        color_ok = (features_dc[idx_i] - features_dc[idx_j]).norm(dim=-1) <= self.config.merge_color_thresh
        scale_ok = (scales_act[idx_i] - scales_act[idx_j]).norm(dim=-1) <= self.config.merge_scale_thresh

        merge_ok = spatial_ok & color_ok & scale_ok
        if merge_ok.sum() == 0:
            return

        merge_i = idx_i[merge_ok]
        merge_j = idx_j[merge_ok]

        # Greedy: each Gaussian in at most one merge
        used = torch.zeros(n, dtype=torch.bool, device=means.device)
        keep_list = []
        kill_list = []

        for mi, mj in zip(merge_i.tolist(), merge_j.tolist()):
            if not used[mi] and not used[mj]:
                keep_list.append(mi)
                kill_list.append(mj)
                used[mi] = True
                used[mj] = True

        if len(keep_list) == 0:
            return

        self._pending_merge_ops = {
            "keep": torch.tensor(keep_list, device=means.device, dtype=torch.long),
            "kill": torch.tensor(kill_list, device=means.device, dtype=torch.long),
        }

    # ── 4. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)

        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        metrics["micro_stage"] = torch.tensor(
            float(self._stage), dtype=torch.float32
        )

        with torch.no_grad():
            scales = torch.exp(self.scales)
            eigenvalues = scales ** 2
            traces = eigenvalues.sum(dim=-1)
            metrics["trace_mean"] = traces.mean()
            metrics["trace_max"] = traces.max()
            metrics["trace_violations"] = (
                (traces > self.config.trace_cap).float().mean()
            )
            metrics["anisotropy"] = self.scales.var(dim=-1).mean()

            alive = (torch.sigmoid(self.opacities) > 0.01).sum()
            metrics["alive_gaussians"] = alive.float()

        return metrics