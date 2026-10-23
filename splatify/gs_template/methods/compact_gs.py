"""
Compact 3DGS via Gradient-Driven Natural Selection
=====================================================
Paper: "Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting"
Complexity: ★★★☆☆ (custom loss + pruning schedule + optimizer manipulation)

Core idea: Simulate natural selection on Gaussians.
    - "Vitality" = opacity α
    - "Environmental pressure" = global regularization gradient field on
      pre-activation opacity v (logit space)
    - "Fitness" = rendering gradient ∇_{α_i} L_render

The regularization loss  L_reg = (E[v] - T)²  with T = -20 creates a
uniform downward pressure on all opacities. Gaussians whose rendering
gradients oppose this pressure survive; the rest decay below the survival
threshold τ and are pruned.

Key implementation details from the paper:
    - Densification (standard 3DGS/Improved-GS): 0 → 15K iterations
    - Natural selection: 15K → until budget met (or 23K fallback)
    - Opacity LR scaled 4× during natural selection
    - Regularization applied every 50 iterations
    - Survival threshold τ = 0.001 (lower than 3DGS's 0.005)
    - L_reg applied to v (pre-sigmoid), NOT α directly
    - After NS completes: 1K recovery iters, then restore opacity LR
    - Total training: 30K iterations

CRITICAL IMPLEMENTATION NOTE — Soft vs Hard Pruning:
    Nerfstudio models don't have direct access to optimizer state dicts.
    Creating new Parameter objects (hard pruning) during training destroys
    Adam momentum/variance for ALL remaining Gaussians, causing catastrophic
    PSNR collapse.

    Solution: TWO-PHASE pruning approach:
      Phase A (during NS): SOFT pruning — clamp dead Gaussians' v to -100
          sigmoid(-100) ≈ 0, so they're invisible but still in memory.
          This preserves all optimizer states via normal gradient flow.
      Phase B (after NS): ONE hard cleanup — remove dead Gaussians from
          memory for efficiency. Optimizer states for survivors are rebuilt
          quickly during the fine-tuning phase (23K→30K = 7K+ iters).

Train:
    ns-train compact-gs --data <path>
    ns-train compact-gs --data <path> --pipeline.model.gaussian_budget 50000
    ns-train compact-gs --data <path> --pipeline.model.reg_target_T -20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    natural_selection_reg_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CompactGSModelConfig(GSTemplateModelConfig):
    """Compact 3DGS with gradient-driven natural selection.

    The method has three phases:
        Phase 1 (0 → ns_start_iter):        Standard densification
        Phase 2 (ns_start_iter → budget met): Natural selection (pruning)
        Phase 3 (after NS → 30K):            Fine-tuning
    """

    _target: Type = field(default_factory=lambda: CompactGSModel)

    # ── Improved-GS backbone (absgrad densification) ───────────────
    absgrad: bool = True
    """Use absolute-value gradients for densification (Improved-GS).
    Paper: 'Use Improved-GS as the densification and rendering backbone.'"""

    # ── Gaussian budget ────────────────────────────────────────────
    gaussian_budget: int = 200_000
    """Target number of Gaussians. Paper uses ~15% of peak 3DGS count.
    Typical values: 50K (aggressive), 100K (moderate), 200K (conservative).
    Adjust per-scene based on complexity."""

    # ── Natural selection schedule ─────────────────────────────────
    ns_start_iter: int = 15_000
    """Iteration to begin natural selection (after densification ends)."""

    ns_max_iter: int = 23_000
    """Latest allowed iteration for natural selection.
    If budget not met by this point, staged fallback pruning is applied."""

    ns_reg_every: int = 50
    """Apply regularization gradient every N iterations.
    Paper: 'every N=50 rendering iterations, allowing fitness signals
    to accumulate from multiple views before applying pressure.'"""

    ns_soft_prune_every: int = 500
    """How often to soft-kill (clamp opacity) dead Gaussians during NS.
    These Gaussians remain in memory but render as invisible."""

    ns_warmup_iters: int = 3000
    """Linearly ramp reg_loss_weight from 0 → full over this many
    iterations after NS starts. Prevents the optimizer from being
    shocked by sudden introduction of a large new loss term.
    E.g. with ns_start_iter=15K and warmup=3K, full strength at 18K."""

    # ── Regularization gradient field ──────────────────────────────
    reg_target_T: float = -20.0
    """Target T for regularization: L_reg = (E[v] - T)².
    When |T| >> |E[v]|, the gradient ∇v ≈ 2T is dominated by T,
    providing stable, globally uniform environmental pressure.
    Paper default: T = -20."""

    reg_loss_weight: float = 0.01
    """Multiplier on L_reg. With T=-20 and E[v]≈1.4, raw L_reg ≈ 458.
    At weight=0.01, effective L_reg ≈ 4.6, comparable to L_render ≈ 0.05.
    The gradient per Gaussian is still small (2*(E[v]-T)*w/N), but
    Adam's momentum accumulation makes the consistent negative direction
    effective over many steps. Increase if NS converges too slowly;
    decrease if PSNR drops during NS."""

    # ── Opacity LR scaling ─────────────────────────────────────────
    opacity_lr_multiplier: float = 4.0
    """Scale factor for opacity learning rate during natural selection.
    Paper: '4× the original to amplify both recovery and decay terms,
    making survival depend mainly on dynamic gradient competition.'
    Tested range: 1×–5×, saturates around 4×."""

    opacity_recovery_iters: int = 1000
    """After NS completes, run this many iters for opacity recovery
    before reverting opacity LR to original value."""

    # ── Survival threshold ─────────────────────────────────────────
    survival_threshold: float = 0.001
    """τ: Gaussians with sigmoid(v) < τ are permanently removed.
    Paper: 'smaller than the 0.005 threshold in 3DGS'."""

    # ── Soft-kill value ────────────────────────────────────────────
    dead_opacity_logit: float = -100.0
    """Value to clamp v for dead Gaussians during soft pruning.
    sigmoid(-100) ≈ 0. These Gaussians contribute nothing to rendering
    but stay in memory to preserve optimizer states for survivors."""

    # ── Phase control (early vs late NS) ───────────────────────────
    use_finite_prior: bool = True
    """Whether to apply regularization to v (pre-sigmoid, finite prior)
    or to α (post-sigmoid, no prior).
    Paper strongly recommends finite prior for convergence speed."""

    early_ns_iters: int = 0
    """Duration of early NS phase with prior-free reg on α.
    Paper: 'early phase uses low effective prior under low learning rate.'
    Set to 0 to skip early phase and use finite prior throughout."""

    # ── Fallback staging ───────────────────────────────────────────
    fallback_stages: int = 4
    """If budget not met at ns_max_iter, split the remaining pruning into
    this many stages spread over ~1K iterations to soften the blow."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class CompactGSModel(GSTemplateModel):
    """Compact 3DGS via gradient-driven natural selection.

    Implements the full natural selection pipeline:
    1. Standard densification (inherited from splatfacto)
    2. Regularization gradient field (compute_custom_losses)
    3. SOFT opacity pruning — clamp dead Gaussians, preserve optimizer states
    4. Single HARD cleanup after NS for memory efficiency
    5. Staged fallback if budget not met by deadline
    """

    config: CompactGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Initialize natural selection tracking state."""
        super().populate_modules()

        # Natural selection state
        self._ns_active = False
        self._ns_completed = False
        self._ns_cleanup_done = False
        self._ns_recovery_start = None
        self._total_soft_killed = 0
        self._peak_gaussians = 0

        # Staged fallback state
        self._fallback_active = False
        self._fallback_prune_per_stage = 0
        self._fallback_stages_remaining = 0
        self._fallback_next_step = 0

    # ── 2. compute_custom_losses (CORE MECHANISM) ──────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Apply the regularization gradient field L_reg = (E[v] - T)².

        This is the "environmental pressure" in the natural selection
        analogy. The gradient ∇v = 2(E[v] - T) is globally constant
        for all Gaussians, creating uniform downward pressure on opacity.

        Gaussians with strong rendering gradients (high "fitness")
        resist this pressure and survive. Weak ones decay and die.

        IMPORTANT: We compute L_reg over ALIVE Gaussians only (excluding
        those already soft-killed at v=-100), so the mean E[v] reflects
        actual scene content, not dead padding.
        """
        losses = {}
        step = self.step

        # Only active during natural selection phase
        if step < self.config.ns_start_iter or self._ns_completed:
            return losses

        # Track peak Gaussian count
        self._peak_gaussians = max(self._peak_gaussians, self.num_points)

        # Apply regularization every ns_reg_every steps
        if step % self.config.ns_reg_every != 0:
            return losses

        # ── Warmup ramp ────────────────────────────────────────────
        # Linearly increase effective weight from 0 → reg_loss_weight
        # over ns_warmup_iters to avoid shocking the optimizer.
        ns_elapsed = step - self.config.ns_start_iter
        if self.config.ns_warmup_iters > 0 and ns_elapsed < self.config.ns_warmup_iters:
            warmup_factor = ns_elapsed / self.config.ns_warmup_iters
        else:
            warmup_factor = 1.0
        effective_weight = self.config.reg_loss_weight * warmup_factor

        if effective_weight < 1e-8:
            return losses  # Too early in warmup, skip

        # Mask out already-dead Gaussians from the mean computation
        # so they don't drag E[v] down and weaken pressure on survivors
        with torch.no_grad():
            alive_mask = self._get_alive_mask()

        if alive_mask.sum() == 0:
            return losses

        alive_opacities = self.opacities[alive_mask]

        # Determine which phase of NS we're in
        in_early_phase = (
            self.config.early_ns_iters > 0
            and ns_elapsed < self.config.early_ns_iters
            and not self.config.use_finite_prior
        )

        if in_early_phase:
            # Early phase: prior-free regularization on α (post-sigmoid)
            alpha = torch.sigmoid(alive_opacities)
            target_alpha = torch.sigmoid(
                torch.tensor(self.config.reg_target_T, device=alpha.device)
            )
            losses["ns_reg"] = (
                effective_weight * (alpha.mean() - target_alpha) ** 2
            )
        else:
            # Main phase: finite-prior regularization on v (pre-sigmoid)
            # L_reg = (E[v] - T)²
            losses["ns_reg"] = (
                effective_weight
                * natural_selection_reg_loss(
                    alive_opacities, self.config.reg_target_T
                )
            )

        return losses

    # ── 3. get_loss_dict (scheduling + pruning) ────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Standard losses + natural selection scheduling.

        Handles:
        - Activating/deactivating natural selection
        - Periodic SOFT pruning (clamp v → -100, preserve optimizer states)
        - Single HARD cleanup after NS (one-time Parameter replacement)
        - Staged fallback pruning if budget not met by deadline
        """
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        step = self.step

        # ── Activate natural selection ─────────────────────────────
        if (
            step == self.config.ns_start_iter
            and not self._ns_active
            and not self._ns_completed
        ):
            self._ns_active = True
            self._peak_gaussians = self.num_points
            CONSOLE.print(
                f"\n[bold green]═══ Natural Selection ACTIVATED "
                f"(step {step}) ═══[/bold green]"
            )
            CONSOLE.print(f"  Current Gaussians: {self.num_points:,}")
            CONSOLE.print(f"  Target budget:     {self.config.gaussian_budget:,}")
            CONSOLE.print(f"  Survival threshold: {self.config.survival_threshold}")
            CONSOLE.print(f"  Regularization T:  {self.config.reg_target_T}")
            CONSOLE.print(f"  Reg weight:        {self.config.reg_loss_weight}")
            CONSOLE.print(
                f"  Warmup:            {self.config.ns_warmup_iters} iters "
                f"(full strength at step {self.config.ns_start_iter + self.config.ns_warmup_iters})"
            )

        # ── Periodic SOFT pruning during NS ────────────────────────
        if (
            self._ns_active
            and not self._ns_completed
            and step >= self.config.ns_start_iter
            and step % self.config.ns_soft_prune_every == 0
        ):
            n_killed = self._soft_kill_low_opacity()
            self._total_soft_killed += n_killed

            n_alive = self._count_alive()

            if step % 1000 == 0:
                ns_elapsed = step - self.config.ns_start_iter
                if self.config.ns_warmup_iters > 0 and ns_elapsed < self.config.ns_warmup_iters:
                    warmup_pct = ns_elapsed / self.config.ns_warmup_iters * 100
                    warmup_str = f" | warmup: {warmup_pct:.0f}%"
                else:
                    warmup_str = " | full strength"
                CONSOLE.print(
                    f"  [NS step {step}] Soft-killed {n_killed:,} this round | "
                    f"Alive: {n_alive:,} / {self.num_points:,} "
                    f"(budget: {self.config.gaussian_budget:,}){warmup_str}"
                )

            # Check if budget is met (counting alive only)
            if n_alive <= self.config.gaussian_budget:
                self._complete_natural_selection(step, "budget_met")

        # ── Staged fallback pruning ────────────────────────────────
        if (
            step == self.config.ns_max_iter
            and not self._ns_completed
        ):
            n_alive = self._count_alive()
            if n_alive > self.config.gaussian_budget:
                excess = n_alive - self.config.gaussian_budget
                self._fallback_active = True
                self._fallback_stages_remaining = self.config.fallback_stages
                self._fallback_prune_per_stage = (
                    excess // self.config.fallback_stages + 1
                )
                self._fallback_next_step = step
                CONSOLE.print(
                    f"\n[bold yellow]  NS deadline reached at step {step}. "
                    f"Starting staged fallback: {n_alive:,} → "
                    f"{self.config.gaussian_budget:,} over "
                    f"{self.config.fallback_stages} stages[/bold yellow]"
                )
            else:
                # Budget already met through soft pruning alone
                self._complete_natural_selection(step, "budget_met_at_deadline")

        # Execute staged fallback steps
        if (
            self._fallback_active
            and self._fallback_stages_remaining > 0
            and step >= self._fallback_next_step
        ):
            self._staged_soft_kill()
            self._fallback_stages_remaining -= 1
            self._fallback_next_step = step + 250

            n_alive = self._count_alive()
            if (
                n_alive <= self.config.gaussian_budget
                or self._fallback_stages_remaining == 0
            ):
                self._fallback_active = False
                self._complete_natural_selection(step, "staged_fallback")

        # ── Hard cleanup: remove dead Gaussians from memory ────────
        # Done ONCE, 500 iterations after NS completes, giving the
        # optimizer a few steps to stabilize before the disruption.
        if (
            self._ns_completed
            and not self._ns_cleanup_done
            and self._ns_recovery_start is not None
            and step >= self._ns_recovery_start + 500
        ):
            self._hard_cleanup()
            self._ns_cleanup_done = True

        return loss_dict

    # ═══════════════════════════════════════════════════════════════
    # Soft Pruning — clamp opacity, preserve optimizer states
    # ═══════════════════════════════════════════════════════════════

    def _soft_kill_low_opacity(self) -> int:
        """Soft-kill Gaussians below survival threshold.

        Sets v = dead_opacity_logit (-100) for Gaussians with
        sigmoid(v) < τ. This makes them render as invisible
        (sigmoid(-100) ≈ 0) without removing them from memory,
        so all optimizer states for surviving Gaussians are preserved.

        Returns: number of newly killed Gaussians this round.
        """
        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            alpha = torch.sigmoid(self.opacities.squeeze(-1))

            # Find Gaussians that are alive but below threshold
            kill_mask = alive_mask & (alpha < self.config.survival_threshold)
            n_killed = kill_mask.sum().item()

            if n_killed > 0:
                self.opacities.data[kill_mask] = self.config.dead_opacity_logit

            return n_killed

    def _staged_soft_kill(self):
        """Kill the weakest alive Gaussians in a staged fallback.

        Instead of killing ALL excess Gaussians at once (catastrophic),
        kills the weakest `fallback_prune_per_stage` Gaussians.
        """
        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            alive_indices = torch.where(alive_mask)[0]

            if len(alive_indices) == 0:
                return

            n_to_kill = min(
                self._fallback_prune_per_stage,
                max(0, len(alive_indices) - self.config.gaussian_budget),
            )

            if n_to_kill <= 0:
                return

            # Find the weakest alive Gaussians
            alive_opacities = torch.sigmoid(
                self.opacities.squeeze(-1)[alive_indices]
            )
            _, weakest_local = torch.topk(
                alive_opacities, n_to_kill, largest=False
            )
            kill_indices = alive_indices[weakest_local]

            self.opacities.data[kill_indices] = self.config.dead_opacity_logit

            CONSOLE.print(
                f"    Staged fallback: soft-killed {n_to_kill:,} weakest "
                f"| Alive: {self._count_alive():,}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Hard Cleanup — one-time removal for memory efficiency
    # ═══════════════════════════════════════════════════════════════

    def _hard_cleanup(self):
        """Remove dead Gaussians from memory (one-time, after NS).

        This DOES destroy optimizer states, but:
        1. It only happens ONCE (not repeatedly during NS)
        2. It happens after NS when gradients are stable
        3. There are 6K+ fine-tuning iterations remaining for recovery
        4. Only dead (v=-100) Gaussians are removed; survivors keep
           their positions/colors/etc. intact

        Adam optimizer rebuilds momentum estimates within ~1K iterations
        from scratch, which is well within the remaining training budget.
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
        """Boolean mask: True for Gaussians NOT soft-killed."""
        return (
            self.opacities.squeeze(-1) > (self.config.dead_opacity_logit + 10)
        )

    def _count_alive(self) -> int:
        """Count Gaussians that are still alive (not soft-killed)."""
        return self._get_alive_mask().sum().item()

    def _complete_natural_selection(self, step: int, reason: str):
        """Mark NS as completed and begin recovery phase."""
        self._ns_active = False
        self._ns_completed = True
        self._ns_recovery_start = step

        n_alive = self._count_alive()
        reduction_pct = (
            (1 - n_alive / max(self._peak_gaussians, 1)) * 100
        )
        CONSOLE.print(
            f"\n[bold green]═══ Natural Selection COMPLETED "
            f"(step {step}, {reason}) ═══[/bold green]"
        )
        CONSOLE.print(f"  Peak Gaussians:  {self._peak_gaussians:,}")
        CONSOLE.print(f"  Alive Gaussians: {n_alive:,}")
        CONSOLE.print(
            f"  In-memory total: {self.num_points:,} "
            f"(dead will be cleaned up in 500 iters)"
        )
        CONSOLE.print(f"  Reduction:       {reduction_pct:.1f}%")
        CONSOLE.print(f"  Total soft-killed: {self._total_soft_killed:,}")

    # ── 4. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add natural-selection-specific metrics."""
        metrics = super().get_metrics_dict(outputs, batch)

        n_alive = self._count_alive()
        metrics["num_gaussians_alive"] = torch.tensor(
            n_alive, dtype=torch.float32
        )
        metrics["num_gaussians_total"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )

        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            if alive_mask.sum() > 0:
                alive_alpha = torch.sigmoid(self.opacities[alive_mask])
                metrics["mean_opacity_alive"] = alive_alpha.mean()

                if self._ns_active:
                    alpha_flat = alive_alpha.squeeze(-1)
                    metrics["opacity_below_thresh"] = (
                        (alpha_flat < self.config.survival_threshold)
                        .float()
                        .mean()
                    )
                    metrics["opacity_above_0.5"] = (
                        (alpha_flat > 0.5).float().mean()
                    )

        return metrics

    # ── Convenience ────────────────────────────────────────────────

    @property
    def ns_phase(self) -> str:
        """Current training phase for logging."""
        if self._ns_completed:
            return "fine_tuning"
        elif self._ns_active:
            return "natural_selection"
        else:
            return "densification"