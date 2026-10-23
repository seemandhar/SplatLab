"""
Entropy-Guided Compact Splatting
=================================
Paper: "Entropy-Guided Compact Splatting" (Ours)
Complexity: ★★★☆☆ (entropy + scale reset + importance pruning)

Core idea: Combine ShorterSplatting's scale-reset mechanism with
    entropy-driven opacity regularization and importance-based pruning
    to reduce the total Gaussian count to ~1.5M while maintaining
    comparable PSNR to methods that keep 3.3M Gaussians.

The method operates in three phases:
    Phase 1 (0 → phase2_start):      Growth + Regularization
        - Standard splatfacto densification (inherited)
        - Opacity entropy loss pushes opacities toward 0 or 1
        - Covariance trace loss keeps Gaussians compact
        - These regularizations make pruning more effective later

    Phase 2 (phase2_start → phase3_start): Scale Reset + Progressive Pruning
        - Periodic scale reset (multiply all scales by zeta=0.7)
        - After each reset, wait for recovery, then prune bottom 3%
        - Importance score: sigmoid(opacity) * max(exp(scales))
        - Additional opacity L1 loss pushes weak Gaussians toward zero
        - Target: reduce to gaussian_budget (~1.5M)

    Phase 3 (phase3_start → end):     Fine-tuning
        - One-time hard cleanup removes dead Gaussians from memory
        - Mild entropy loss for clean-up
        - No more pruning or scale reset

Key implementation details:
    - Scale reset in log-space: s += log(zeta)  (equivalent to s *= zeta)
    - Soft pruning: set opacity to -100 (preserve optimizer state)
    - Hard cleanup: one-time Parameter replacement with keep_mask
    - Importance = sigmoid(opacity) * max(exp(scale_per_axis))

CRITICAL IMPLEMENTATION NOTE — Soft vs Hard Pruning:
    Same two-phase approach as CompactGS:
      Phase A (during pruning): SOFT pruning — clamp dead opacities to -100.
          sigmoid(-100) ≈ 0, invisible but still in memory.
          Preserves all optimizer states via normal gradient flow.
      Phase B (after pruning): ONE hard cleanup — remove dead Gaussians
          for memory efficiency. Fine-tuning phase rebuilds momentum.

Train:
    ns-train entropy-compact-gs --data <path>
    ns-train entropy-compact-gs --data <path> --pipeline.model.gaussian_budget 1000000
    ns-train entropy-compact-gs --data <path> --pipeline.model.scale_reset_zeta 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    opacity_entropy_loss,
    opacity_l1_loss,
    covariance_trace_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EntropyCompactGSModelConfig(GSTemplateModelConfig):
    """Entropy-Guided Compact Splatting configuration.

    The method has three phases:
        Phase 1 (0 → phase2_start):           Growth + regularization
        Phase 2 (phase2_start → phase3_start): Scale reset + progressive pruning
        Phase 3 (phase3_start → 30K):          Fine-tuning
    """

    _target: Type = field(default_factory=lambda: EntropyCompactGSModel)

    # ── Improved-GS backbone (absgrad densification) ───────────────
    absgrad: bool = True
    """Use absolute-value gradients for densification (Improved-GS).
    Better gradient signal for identifying where to split/clone."""

    # ── Gaussian budget ────────────────────────────────────────────
    gaussian_budget: int = 1_500_000
    """Target number of Gaussians after pruning. Default 1.5M for
    large scenes; reduce for simpler scenes (e.g., 500K indoor)."""

    # ── Phase schedule ─────────────────────────────────────────────
    phase2_start: int = 15_000
    """Iteration to begin Phase 2 (scale reset + pruning).
    Should match end of splatfacto densification period."""

    phase3_start: int = 20_000
    """Iteration to begin Phase 3 (fine-tuning).
    Gives ~10K iterations for fine-tuning before 30K total."""

    # ── Scale reset (ShorterSplatting-inspired) ────────────────────
    scale_reset_zeta: float = 0.7
    """Scale shrink factor. Each reset multiplies all alive scales
    by this value. In log-space: s += log(zeta). Values < 1 shrink
    Gaussians, forcing them to re-grow only if rendering demands it.
    zeta=0.7 provides gentle shrinkage that is easily recoverable."""

    scale_reset_interval: int = 4000
    """How often (in iterations) to apply scale reset during Phase 2.
    With phase2_start=15K and phase3_start=20K, this gives ~2 resets."""

    scale_reset_recovery: int = 500
    """Number of iterations to wait after a scale reset before pruning.
    Allows Gaussians to partially recover, so importance scores
    reflect post-reset fitness rather than transient shrinkage."""

    # ── Entropy regularization ─────────────────────────────────────
    entropy_weight_phase1: float = 0.002
    """Opacity entropy loss weight during Phase 1 (growth).
    Gentle push during growth; strong enough to bias opacities.
    H(α) = -(α log α + (1-α) log(1-α)), minimized at α∈{0,1}."""

    entropy_weight_phase3: float = 0.002
    """Reduced entropy weight during Phase 3 (fine-tuning).
    Maintains clean opacity distribution without over-constraining."""

    # ── Covariance trace regularization ────────────────────────────
    cov_trace_weight: float = 0.0001
    """Weight for covariance trace hinge loss. Penalizes Gaussians
    whose trace(Σ) exceeds trace_cap, encouraging compact kernels."""

    cov_trace_cap: float = 0.1
    """Maximum acceptable covariance trace before penalty kicks in.
    Scene-dependent: indoor 0.05-0.1, outdoor 0.1-0.3."""

    # ── Opacity L1 (Phase 2 only) ─────────────────────────────────
    opacity_l1_weight_phase2: float = 0.0
    """L1 penalty on sigmoid opacities during Phase 2.
    Pushes weak Gaussians toward zero, complementing entropy loss
    which pushes toward {0, 1}. Uses sum reduction per CompGS."""

    # ── Importance pruning ─────────────────────────────────────────
    prune_percent: float = 0.03
    """Fraction of alive Gaussians to prune per round (by importance).
    Conservative: 5% per round preserves quality while reducing count gradually.
    Applied after each scale reset + recovery period in Phase 2."""

    # ── Survival threshold ─────────────────────────────────────────
    survival_threshold: float = 0.005
    """τ: Gaussians with sigmoid(v) < τ are considered dead.
    Used during hard cleanup in Phase 3."""

    # ── Soft-kill value ────────────────────────────────────────────
    dead_opacity_logit: float = -100.0
    """Value to clamp v for dead Gaussians during soft pruning.
    sigmoid(-100) ≈ 0. Preserves optimizer state for survivors."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class EntropyCompactGSModel(GSTemplateModel):
    """Entropy-Guided Compact Splatting.

    Implements a three-phase pipeline:
    1. Growth + entropy/covariance regularization (inherited densification)
    2. Scale reset + importance-based progressive pruning
    3. Hard cleanup + mild entropy fine-tuning

    Uses SOFT pruning during Phase 2 (opacity → -100) to preserve
    optimizer states, then a single HARD cleanup at Phase 3 start
    for memory efficiency.
    """

    config: EntropyCompactGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Initialize entropy-compact tracking state."""
        super().populate_modules()

        # Phase tracking
        self._current_phase = 1
        self._hard_cleanup_done = False

        # Scale reset tracking
        self._last_scale_reset_step = 0
        self._scale_resets_done = 0

        # Pruning tracking
        self._pending_prune_after_step = None  # step at which to prune (after recovery)
        self._total_soft_killed = 0
        self._prune_rounds_done = 0
        self._peak_gaussians = 0

    # ── 2. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Phase-dependent regularization losses.

        Phase 1: opacity entropy + covariance trace
        Phase 2: opacity entropy + covariance trace + opacity L1
        Phase 3: mild opacity entropy only
        """
        losses = {}
        step = self.step

        if self._current_phase == 1:
            # ── Phase 1: Growth + Regularization ───────────────────
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase1
                * opacity_entropy_loss(self.opacities)
            )
            losses["cov_trace"] = (
                self.config.cov_trace_weight
                * covariance_trace_loss(self.scales, self.config.cov_trace_cap)
            )

        elif self._current_phase == 2:
            # ── Phase 2: Scale Reset + Pruning ─────────────────────
            # Continue entropy + cov trace from Phase 1
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase1
                * opacity_entropy_loss(self.opacities)
            )
            losses["cov_trace"] = (
                self.config.cov_trace_weight
                * covariance_trace_loss(self.scales, self.config.cov_trace_cap)
            )
            # Add opacity L1 to push weak Gaussians toward zero
            losses["opacity_l1"] = opacity_l1_loss(
                self.opacities, self.config.opacity_l1_weight_phase2
            )

        elif self._current_phase == 3:
            # ── Phase 3: Fine-tuning ──────────────────────────────
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase3
                * opacity_entropy_loss(self.opacities)
            )

        return losses

    # ── 3. get_loss_dict (phase transitions + pruning schedule) ────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Standard losses + phase scheduling, scale resets, and pruning.

        Handles:
        - Phase transitions (1 → 2 → 3)
        - Periodic scale resets in Phase 2
        - Importance-based soft pruning after recovery
        - One-time hard cleanup at Phase 3 start
        """
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        step = self.step

        # Track peak Gaussian count
        self._peak_gaussians = max(self._peak_gaussians, self.num_points)

        # ── Phase 1 → Phase 2 transition ──────────────────────────
        if step == self.config.phase2_start and self._current_phase == 1:
            self._current_phase = 2
            self._last_scale_reset_step = step  # don't reset immediately
            CONSOLE.print(
                f"\n[bold green]═══ Phase 2 ACTIVATED "
                f"(step {step}) ═══[/bold green]"
            )
            CONSOLE.print(f"  Current Gaussians: {self.num_points:,}")
            CONSOLE.print(f"  Target budget:     {self.config.gaussian_budget:,}")
            CONSOLE.print(
                f"  Scale reset zeta:  {self.config.scale_reset_zeta} "
                f"every {self.config.scale_reset_interval} iters"
            )
            CONSOLE.print(
                f"  Prune percent:     {self.config.prune_percent * 100:.0f}% "
                f"after {self.config.scale_reset_recovery} iter recovery"
            )

        # ── Phase 2: Scale reset + pruning ────────────────────────
        if self._current_phase == 2:
            # Periodic scale reset
            if (
                step - self._last_scale_reset_step
                >= self.config.scale_reset_interval
                and step + self.config.scale_reset_recovery
                < self.config.phase3_start
            ):
                self._apply_scale_reset()
                self._last_scale_reset_step = step
                self._scale_resets_done += 1
                # Schedule pruning after recovery period
                self._pending_prune_after_step = (
                    step + self.config.scale_reset_recovery
                )
                CONSOLE.print(
                    f"  [Scale reset #{self._scale_resets_done} at step {step}] "
                    f"Scales shrunk by zeta={self.config.scale_reset_zeta}. "
                    f"Pruning scheduled at step "
                    f"{self._pending_prune_after_step}."
                )

            # Execute pending pruning after recovery
            if (
                self._pending_prune_after_step is not None
                and step >= self._pending_prune_after_step
            ):
                n_killed = self._importance_prune()
                self._total_soft_killed += n_killed
                self._prune_rounds_done += 1
                self._pending_prune_after_step = None

                n_alive = self._count_alive()
                CONSOLE.print(
                    f"  [Prune round #{self._prune_rounds_done} at step {step}] "
                    f"Soft-killed {n_killed:,} | "
                    f"Alive: {n_alive:,} / {self.num_points:,} "
                    f"(budget: {self.config.gaussian_budget:,})"
                )

                # Early completion if budget met
                if n_alive <= self.config.gaussian_budget:
                    CONSOLE.print(
                        f"  [bold cyan]Budget met early! "
                        f"({n_alive:,} <= {self.config.gaussian_budget:,})"
                        f"[/bold cyan]"
                    )

        # ── Phase 2 → Phase 3 transition ──────────────────────────
        if step == self.config.phase3_start and self._current_phase == 2:
            self._current_phase = 3
            n_alive = self._count_alive()
            reduction_pct = (
                (1 - n_alive / max(self._peak_gaussians, 1)) * 100
            )
            CONSOLE.print(
                f"\n[bold green]═══ Phase 3 ACTIVATED "
                f"(step {step}) ═══[/bold green]"
            )
            CONSOLE.print(f"  Peak Gaussians:    {self._peak_gaussians:,}")
            CONSOLE.print(f"  Alive Gaussians:   {n_alive:,}")
            CONSOLE.print(
                f"  In-memory total:   {self.num_points:,} "
                f"(dead will be cleaned up shortly)"
            )
            CONSOLE.print(f"  Reduction:         {reduction_pct:.1f}%")
            CONSOLE.print(f"  Scale resets done: {self._scale_resets_done}")
            CONSOLE.print(f"  Prune rounds done: {self._prune_rounds_done}")
            CONSOLE.print(
                f"  Total soft-killed: {self._total_soft_killed:,}"
            )

        # ── Hard cleanup: remove dead Gaussians from memory ───────
        # Done ONCE, 500 iterations after Phase 3 starts, to let
        # optimizer states stabilize before the disruption.
        if (
            self._current_phase == 3
            and not self._hard_cleanup_done
            and step >= self.config.phase3_start + 500
        ):
            self._hard_cleanup()
            self._hard_cleanup_done = True

        return loss_dict

    # ═══════════════════════════════════════════════════════════════
    # Scale Reset — shrink all Gaussian scales
    # ═══════════════════════════════════════════════════════════════

    def _apply_scale_reset(self):
        """Shrink all alive Gaussian scales by factor zeta.

        In log-space, multiplying scales by zeta is equivalent to
        adding log(zeta) to the log-scale parameters. This forces
        Gaussians to re-grow through gradient descent; those that
        cannot recover are weak candidates for pruning.
        """
        with torch.no_grad():
            alive = self._get_alive_mask()
            self.scales.data[alive] += math.log(self.config.scale_reset_zeta)

    # ═══════════════════════════════════════════════════════════════
    # Importance Pruning — soft-kill lowest-importance Gaussians
    # ═══════════════════════════════════════════════════════════════

    def _importance_prune(self) -> int:
        """Prune bottom prune_percent of alive Gaussians by importance.

        Importance score: sigmoid(opacity) * max(exp(scales_per_axis))
        Combines visibility (opacity) with spatial extent (max scale).
        Gaussians that are both faint and small contribute least.

        Uses SOFT pruning: sets opacity logit to -100 rather than
        removing from memory, preserving all optimizer states.

        Returns: number of newly killed Gaussians this round.
        """
        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            alive_indices = torch.where(alive_mask)[0]

            if len(alive_indices) == 0:
                return 0

            # Compute importance scores for alive Gaussians
            opacities = torch.sigmoid(
                self.opacities.squeeze(-1)[alive_indices]
            )
            scales = torch.exp(self.scales[alive_indices])
            max_scale = scales.max(dim=-1).values
            importance = opacities * max_scale

            # Determine how many to prune
            n_alive = len(alive_indices)

            # Don't prune if already at or below budget
            if n_alive <= self.config.gaussian_budget:
                return 0

            n_to_kill = max(1, int(n_alive * self.config.prune_percent))

            # Don't prune below budget
            max_kill = max(0, n_alive - self.config.gaussian_budget)
            n_to_kill = min(n_to_kill, max_kill) if max_kill > 0 else n_to_kill

            # Don't prune below a minimum viable count
            if n_alive - n_to_kill < 1000:
                n_to_kill = max(0, n_alive - 1000)
            if n_to_kill <= 0:
                return 0

            # Find the weakest Gaussians
            _, weakest_local = torch.topk(
                importance, n_to_kill, largest=False
            )
            kill_indices = alive_indices[weakest_local]

            # Soft-kill: set opacity to dead value
            self.opacities.data[kill_indices] = self.config.dead_opacity_logit

            return n_to_kill

    # ═══════════════════════════════════════════════════════════════
    # Hard Cleanup — one-time removal for memory efficiency
    # ═══════════════════════════════════════════════════════════════

    def _hard_cleanup(self):
        """Remove dead Gaussians from memory (one-time, after Phase 2).

        This DOES destroy optimizer states, but:
        1. It only happens ONCE (not repeatedly during pruning)
        2. It happens after pruning when gradients are stable
        3. There are ~6.5K fine-tuning iterations remaining for recovery
        4. Only dead (v=-100) Gaussians are removed; survivors keep
           their positions/colors/etc. intact

        Adam optimizer rebuilds momentum estimates within ~1K iterations,
        well within the remaining training budget.
        """
        with torch.no_grad():
            keep_mask = self._get_alive_mask()
            n_before = self.num_points
            n_alive = keep_mask.sum().item()

            if n_alive == n_before:
                CONSOLE.print("  Hard cleanup: no dead Gaussians to remove")
                return

            for name in list(self.gauss_params.keys()):
                old_param = self.gauss_params[name]
                self.gauss_params[name] = Parameter(
                    old_param.data[keep_mask].contiguous().clone()
                )

            CONSOLE.print(
                f"  [bold cyan]Hard cleanup: {n_before:,} → "
                f"{self.num_points:,} (removed "
                f"{n_before - self.num_points:,} dead Gaussians, "
                f"freed memory)[/bold cyan]"
            )

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _get_alive_mask(self) -> Tensor:
        """Boolean mask: True for Gaussians NOT soft-killed.

        A Gaussian is considered alive if its opacity logit is above
        dead_opacity_logit + 10 (i.e., -90 with default settings).
        This margin avoids false positives from Gaussians that are
        merely low-opacity but not explicitly killed.
        """
        return (
            self.opacities.squeeze(-1) > (self.config.dead_opacity_logit + 10)
        )

    def _count_alive(self) -> int:
        """Count Gaussians that are still alive (not soft-killed)."""
        return self._get_alive_mask().sum().item()

    # ── 4. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add entropy-compact-specific metrics for logging."""
        metrics = super().get_metrics_dict(outputs, batch)

        n_alive = self._count_alive()
        metrics["num_gaussians_alive"] = torch.tensor(
            n_alive, dtype=torch.float32
        )
        metrics["num_gaussians_total"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        metrics["training_phase"] = torch.tensor(
            float(self._current_phase), dtype=torch.float32
        )
        metrics["scale_resets_done"] = torch.tensor(
            float(self._scale_resets_done), dtype=torch.float32
        )
        metrics["prune_rounds_done"] = torch.tensor(
            float(self._prune_rounds_done), dtype=torch.float32
        )

        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            if alive_mask.sum() > 0:
                alive_alpha = torch.sigmoid(self.opacities[alive_mask])
                metrics["mean_opacity_alive"] = alive_alpha.mean()

                alive_scales = torch.exp(self.scales[alive_mask])
                metrics["mean_max_scale_alive"] = (
                    alive_scales.max(dim=-1).values.mean()
                )

                # Covariance trace statistics
                eigenvalues = alive_scales ** 2
                traces = eigenvalues.sum(dim=-1)
                metrics["trace_mean"] = traces.mean()
                metrics["trace_violations"] = (
                    (traces > self.config.cov_trace_cap).float().mean()
                )

        return metrics

    # ── Convenience ────────────────────────────────────────────────

    @property
    def phase_name(self) -> str:
        """Current training phase for logging."""
        if self._current_phase == 1:
            return "growth_regularization"
        elif self._current_phase == 2:
            return "scale_reset_pruning"
        else:
            return "fine_tuning"
