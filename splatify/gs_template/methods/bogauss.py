"""
BOGausS — Better Optimized Gaussian Splatting
===============================================
Complexity: ★★★★☆ (custom densification, pruning, splitting)

Paper contributions implemented via splatfacto-compatible extension points:
    1. Effective opacity-based pruning (replaces opacity reset)
    2. Density-preserving splitting along longest axis (replaces cloning)
    3. Distortion-based split prioritization (P_SNR scoring)
    4. Densification every 500 iterations

What we intentionally keep from splatfacto (not worth fighting):
    - Log-space scale parameterization (exp activation)
    - Standard Adam optimizer (paper's modified Adam is a marginal gain)
    - Standard rendering pipeline

Train:
    ns-train bogauss --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BOGausSModelConfig(GSTemplateModelConfig):
    """BOGausS configuration.

    Inherits all splatfacto config fields. We only add BOGausS-specific
    knobs and override a few splatfacto defaults.
    """

    _target: Type = field(default_factory=lambda: BOGausSModel)

    # ── Override splatfacto defaults for BOGausS schedule ──────────
    refine_every: int = 500
    """Densification interval (paper: every 500 iterations)."""

    warmup_length: int = 500
    """Steps before densification starts."""

    # Use absgrad (same as AbsGS — helps with fine detail)
    absgrad: bool = True

    # ── BOGausS-specific: Splitting ────────────────────────────────
    split_axis_offset: float = 0.3
    """α: mean offset along longest axis during split (paper: 0.3)."""

    split_opacity_decay: float = 0.8
    """Opacity factor for children after split (density preservation)."""

    split_size_decay: float = 0.7
    """Size factor for children along split axis."""

    bogauss_split_enabled: bool = True
    """Enable BOGausS density-preserving splitting (alongside splatfacto's)."""

    bogauss_split_fraction: float = 0.01
    """Fraction of Gaussians to split per BOGausS step."""

    bogauss_split_start: int = 1500
    """Start BOGausS splitting after this step (let splatfacto warm up first)."""

    bogauss_split_stop: int = 25000
    """Stop BOGausS splitting after this step."""

    # ── BOGausS-specific: Pruning ──────────────────────────────────
    bogauss_prune_enabled: bool = True
    """Enable effective-opacity pruning."""

    effective_opacity_threshold: float = 0.02
    """τ: minimum effective opacity to survive (paper: 0.02)."""

    max_prune_fraction: float = 0.01
    """Cap: at most 1% of Gaussians pruned per step (paper: 1%)."""

    bogauss_prune_start: int = 2000
    """Start BOGausS pruning after this step."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class BOGausSModel(GSTemplateModel):
    """BOGausS — Better Optimized Gaussian Splatting.

    Works WITH splatfacto, not against it:
    - Splatfacto handles rendering, standard densification, optimizer wiring
    - BOGausS adds additional pruning and splitting logic on top
    - No parameterization changes — keeps log-space scales

    Overrides:
        - populate_modules: init tracking buffers
        - get_loss_dict: accumulate per-Gaussian SE for split priority
        - refinement_after: additional BOGausS split/prune after splatfacto's
        - get_metrics_dict: log Gaussian count
    """

    config: BOGausSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Standard splatfacto init + tracking buffers for BOGausS."""
        super().populate_modules()
        self._init_bogauss_buffers()

    def _init_bogauss_buffers(self):
        """Initialize per-Gaussian tracking for split/prune decisions."""
        n = self.num_points

        # Per-Gaussian accumulated squared error (for P_SNR split priority)
        self.register_buffer(
            "_se_accum", torch.zeros(n, device="cpu"), persistent=False
        )
        # Render count per Gaussian (for effective opacity)
        self.register_buffer(
            "_render_count", torch.zeros(n, dtype=torch.long, device="cpu"),
            persistent=False,
        )
        # Step counter for buffer resets
        self._last_refine_step = 0

    def _resize_bogauss_buffers(self, new_n: int):
        """Resize tracking buffers after split/prune."""
        device = self._se_accum.device
        self._se_accum = torch.zeros(new_n, device=device)
        self._render_count = torch.zeros(new_n, dtype=torch.long, device=device)

    # ── 2. Track per-Gaussian error (for split priority) ──────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Standard losses + accumulate per-Gaussian SE."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Track SE for split prioritization
        if self.training:
            self._accumulate_se(outputs, batch)

        return loss_dict

    @torch.no_grad()
    def _accumulate_se(self, outputs, batch):
        """Track per-Gaussian squared error for split prioritization.

        P_SNR(i) = Σ_c SE_{c,i} / Σ_j SE_{c,j}

        Approximation: distribute pixel-wise SE to visible Gaussians
        proportional to their opacity (since exact per-Gaussian weights
        require custom rasterization).
        """
        if "rgb" not in outputs or "image" not in batch:
            return

        pred = outputs["rgb"]
        gt = batch["image"].to(pred.device)
        if gt.shape[:2] != pred.shape[:2]:
            return

        # Total squared error for this frame
        pixel_se = ((pred - gt) ** 2).sum().item()

        # Which Gaussians are visible?
        if not hasattr(self, "radii") or self.radii is None:
            return
        visible = (self.radii > 0).cpu()
        n_visible = visible.sum().item()
        if n_visible == 0:
            return

        # Safety: buffer size must match current Gaussian count
        n = self.num_points
        if len(self._se_accum) != n:
            self._resize_bogauss_buffers(n)
            return

        if len(visible) != n:
            return

        # Weighted by opacity (more opaque → more responsible for error)
        opacities = torch.sigmoid(self.opacities.data).squeeze(-1).cpu()
        vis_opacities = opacities[visible]
        total_opacity = vis_opacities.sum().item()
        if total_opacity < 1e-10:
            return

        # Distribute SE weighted by opacity
        weights = vis_opacities / total_opacity
        self._se_accum[visible] += pixel_se * weights
        self._render_count[visible] += 1

    # ── 3. BOGausS refinement (runs AFTER splatfacto's) ────────────

    def refinement_after(self, optimizers, step):
        """Splatfacto densification + BOGausS-specific split/prune.

        Splatfacto's densification (gradient-based split/clone + opacity
        culling) runs first via the parent. Then BOGausS adds:
        - Effective opacity pruning
        - Distortion-based density-preserving splitting
        """
        # Let splatfacto do its thing first
        super().refinement_after(optimizers, step)

        # BOGausS additions at our schedule
        if step % self.config.refine_every != 0:
            return
        if step <= self.config.warmup_length:
            return

        # Sync buffer sizes with current Gaussian count
        # (splatfacto may have split/pruned in the super() call)
        n = self.num_points
        if len(self._se_accum) != n:
            self._resize_bogauss_buffers(n)
            return  # skip this round, buffers are stale

        # Effective opacity pruning
        if (self.config.bogauss_prune_enabled
                and step >= self.config.bogauss_prune_start):
            self._bogauss_prune(optimizers)

        # Density-preserving splitting
        if (self.config.bogauss_split_enabled
                and step >= self.config.bogauss_split_start
                and step < self.config.bogauss_split_stop):
            self._bogauss_split(optimizers)

        # Reset accumulators
        self._resize_bogauss_buffers(self.num_points)
        self._last_refine_step = step

    # ── Effective opacity pruning ──────────────────────────────────

    def _bogauss_prune(self, optimizers):
        """Prune Gaussians with low effective opacity.

        P_prune(i) = max_c P_i^{(c)}
        Approximated as: sigmoid(opacity_i) × (render_frequency_i)
        Prune if P_prune < τ, capped at 1% per step.
        """
        n = self.num_points
        if n < 200:
            return

        max_prune = max(1, int(n * self.config.max_prune_fraction))

        with torch.no_grad():
            opacities = torch.sigmoid(self.opacities.data).squeeze(-1)  # (N,)
            device = opacities.device

            # Render frequency: fraction of frames each Gaussian was visible
            refine_interval = self.config.refine_every
            render_freq = (
                self._render_count.float().to(device)
                / max(refine_interval, 1)
            ).clamp(0, 1)

            # Effective opacity = raw opacity × render frequency
            eff_opacity = opacities * render_freq

            # Prune low effective opacity
            prune_mask = eff_opacity < self.config.effective_opacity_threshold

            # Don't prune Gaussians that were never rendered
            # (they might just be outside recent camera views)
            never_rendered = self._render_count.to(device) == 0
            prune_mask = prune_mask & ~never_rendered

            # Cap at max_prune_fraction
            n_to_prune = prune_mask.sum().item()
            if n_to_prune > max_prune:
                scores = eff_opacity.clone()
                scores[~prune_mask] = float("inf")
                _, worst_indices = scores.topk(max_prune, largest=False)
                prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
                prune_mask[worst_indices] = True

            n_to_prune = prune_mask.sum().item()
            if n_to_prune == 0 or n_to_prune >= n - 100:
                return

            # Apply pruning
            keep_mask = ~prune_mask
            self._prune_gaussians(keep_mask, optimizers)

    # ── Density-preserving splitting ───────────────────────────────

    def _bogauss_split(self, optimizers):
        """Split top-scoring Gaussians along their longest axis.

        Priority: P_SNR(i) = SE_i / Σ_j SE_j (distortion-based)
        Split: offset means ±α*σ along longest principal axis,
               reduce child sizes and opacities for density preservation.
        """
        n = self.num_points
        if n < 100:
            return

        max_split = max(1, int(n * self.config.bogauss_split_fraction))
        device = self.means.device

        with torch.no_grad():
            # Compute split priority from accumulated SE
            se = self._se_accum.to(device)
            se_total = se.sum()
            if se_total < 1e-10:
                return

            priority = se / (se_total + 1e-10)

            # Boost priority for larger Gaussians (they benefit more from splitting)
            scales = torch.exp(self.gauss_params["scales"].data)  # (N, 3)
            max_scale = scales.max(dim=-1).values
            median_scale = max_scale.median()
            if median_scale < 1e-10:
                return
            scale_boost = (max_scale / (median_scale + 1e-8)).clamp(0.5, 5.0)
            priority = priority * scale_boost

            # Select top candidates
            n_to_split = min(max_split, n // 4)  # never split more than 25%
            if n_to_split == 0:
                return

            _, split_indices = priority.topk(n_to_split)
            split_mask = torch.zeros(n, dtype=torch.bool, device=device)
            split_mask[split_indices] = True

            # ── Build children ─────────────────────────────────────
            mother_means = self.means.data[split_mask]           # (M, 3)
            mother_quats = self.quats.data[split_mask]           # (M, 4)
            mother_scales_log = self.gauss_params["scales"].data[split_mask]  # (M, 3)
            mother_opacities = self.opacities.data[split_mask]   # (M, 1)

            scales_lin = torch.exp(mother_scales_log)  # (M, 3)
            m = split_mask.sum().item()

            # Longest axis per Gaussian
            longest_idx = scales_lin.argmax(dim=-1)  # (M,)
            longest_sigma = scales_lin.gather(1, longest_idx.unsqueeze(-1)).squeeze(-1)  # (M,)

            # Rotation matrices from quaternions
            quats_norm = mother_quats / mother_quats.norm(dim=-1, keepdim=True)
            rot = _quat_to_rotmat(quats_norm)  # (M, 3, 3)

            # Split direction = longest principal axis
            split_dir = rot[torch.arange(m, device=device), :, longest_idx]  # (M, 3)

            # Child means: ±offset
            offset = self.config.split_axis_offset * longest_sigma.unsqueeze(-1) * split_dir
            child1_means = mother_means + offset
            child2_means = mother_means - offset

            # Child scales: shrink along split axis (log-space)
            child_scales_log = mother_scales_log.clone()
            decay_log = math.log(self.config.split_size_decay)
            for i in range(m):
                child_scales_log[i, longest_idx[i]] += decay_log

            # Child opacities: reduce for density preservation
            # inverse_sigmoid(decay * sigmoid(o))
            mother_o = torch.sigmoid(mother_opacities)
            child_o = (self.config.split_opacity_decay * mother_o).clamp(1e-6, 1 - 1e-6)
            child_opacities = torch.log(child_o / (1 - child_o))

            # New Gaussian params = [kept] + [child1s] + [child2s]
            keep_mask_inv = ~split_mask

            new_gauss = {}
            for name, param in self.gauss_params.items():
                kept = param.data[keep_mask_inv]
                if name == "means":
                    new_gauss[name] = torch.cat([kept, child1_means, child2_means])
                elif name == "scales":
                    new_gauss[name] = torch.cat([kept, child_scales_log, child_scales_log])
                elif name == "opacities":
                    new_gauss[name] = torch.cat([kept, child_opacities, child_opacities])
                else:
                    # Duplicate mother values for children
                    mother_vals = param.data[split_mask]
                    new_gauss[name] = torch.cat([kept, mother_vals, mother_vals])

            # Apply
            self._replace_all_gaussians(new_gauss, optimizers)

    # ── Parameter management helpers ───────────────────────────────

    def _prune_gaussians(self, keep_mask: Tensor, optimizers):
        """Remove Gaussians using keep_mask. Updates optimizer state."""
        for name in list(self.gauss_params.keys()):
            old_param = self.gauss_params[name]
            new_data = old_param.data[keep_mask]
            new_param = Parameter(new_data)
            _replace_param_in_optimizers(optimizers, old_param, new_param)
            self.gauss_params[name] = new_param

    def _replace_all_gaussians(self, new_gauss: Dict[str, Tensor], optimizers):
        """Replace all Gaussian params with new tensors. Resets optimizer state."""
        for name in list(self.gauss_params.keys()):
            old_param = self.gauss_params[name]
            new_data = new_gauss[name]
            new_param = Parameter(new_data.to(old_param.device))
            _replace_param_in_optimizers(optimizers, old_param, new_param)
            self.gauss_params[name] = new_param

    # ── 4. Metrics ─────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        return metrics


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """(N, 4) normalized quaternions (w,x,y,z) → (N, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


def _replace_param_in_optimizers(optimizers, old_param, new_param):
    """Replace old_param with new_param across all optimizer groups.

    Works with nerfstudio's optimizer wrapper structure.
    """
    # nerfstudio wraps optimizers in an Optimizers object
    opt_dict = optimizers.optimizers if hasattr(optimizers, "optimizers") else {}
    for opt_name, opt in opt_dict.items():
        inner = opt
        # Unwrap if needed
        if hasattr(inner, "optimizer"):
            inner = inner.optimizer

        for group in inner.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_param or (p.data_ptr() == old_param.data_ptr()):
                    group["params"][i] = new_param
                    if p in inner.state:
                        del inner.state[p]
                    return