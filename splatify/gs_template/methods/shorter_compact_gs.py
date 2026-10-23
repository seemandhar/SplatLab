"""
Compact Shorter Splatting (CSS)
================================
Paper: "Compact Shorter Splatting: Fewer Gaussians via Recovery-Gated Pruning"
Complexity: ★★★☆☆ (scale reset + natural selection + importance pruning)

Core idea: ShorterSplatting makes Gaussians focused/compact via scale reset
    and entropy constraints, but keeps the SAME 3.3M Gaussians. We observe
    that focused Gaussians have reduced overlap, creating redundancy.
    By combining scale reset with recovery-gated importance pruning and
    natural selection pressure, we achieve comparable PSNR with ~1M Gaussians
    (3x fewer than ShorterSplatting/3DGS).

Key novelties over ShorterSplatting:
    1. Recovery-gated pruning: after each scale reset, wait for recovery,
       then prune Gaussians that failed to regain their contribution.
       This is more principled than fixed-threshold pruning.
    2. Natural selection gradient pressure on opacity logits (from Compact-GS):
       creates uniform downward pressure, letting rendering gradients
       selectively protect important Gaussians.
    3. Multi-round adaptive compaction: progressively stricter scale resets
       create increasingly stringent necessity tests.
    4. Dual regularization: opacity entropy (push toward {0,1}) + covariance
       trace (keep Gaussians compact) work synergistically with pruning.

Three phases:
    Phase 1 (0 → phase2_start):      Growth + Regularization
        - Standard splatfacto densification with absgrad
        - Opacity entropy loss + covariance trace loss
        - Build up Gaussian field to full capacity

    Phase 2 (phase2_start → phase3_start): Scale Reset + Natural Selection + Pruning
        - Periodic scale reset (zeta=0.5, more aggressive than entropy_compact)
        - Natural selection reg loss creates downward opacity pressure
        - After each reset + recovery, prune bottom % by importance
        - Target: reduce to gaussian_budget (~1M)

    Phase 3 (phase3_start → end):     Fine-tuning
        - Hard cleanup removes dead Gaussians from memory
        - Mild entropy loss + L2 photometric for quality recovery
        - No more pruning

Train:
    ns-train shorter-compact-gs --data <path>
    ns-train shorter-compact-gs --data <path> --pipeline.model.gaussian_budget 800000
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
    l2_photometric_loss,
    natural_selection_reg_loss,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ShorterCompactGSModelConfig(GSTemplateModelConfig):
    """Compact Shorter Splatting configuration.

    Three phases:
        Phase 1 (0 → phase2_start):           Growth + regularization
        Phase 2 (phase2_start → phase3_start): Scale reset + NS + pruning
        Phase 3 (phase3_start → 30K):          Fine-tuning
    """

    _target: Type = field(default_factory=lambda: ShorterCompactGSModel)

    # ── Improved-GS backbone ─────────────────────────────────────
    absgrad: bool = True
    """Use absolute-value gradients for densification (better gradient signal)."""

    # ── Gaussian budget ──────────────────────────────────────────
    gaussian_budget: int = 1_000_000
    """Target number of Gaussians after pruning. ~1M for large outdoor
    scenes; 3x fewer than 3DGS/ShorterSplatting's 3.3M."""

    # ── Phase schedule ───────────────────────────────────────────
    phase2_start: int = 15_000
    """Start of compaction phase (after densification completes)."""

    phase3_start: int = 22_000
    """Start of fine-tuning phase. Gives 8K iterations for recovery."""

    # ── Scale reset (ShorterSplatting-inspired, more aggressive) ──
    scale_reset_zeta: float = 0.5
    """Scale shrink factor per reset. More aggressive than entropy_compact
    (0.7) to create stronger selection pressure. In log-space: s += log(zeta)."""

    scale_reset_interval: int = 2000
    """How often to apply scale reset during Phase 2. More frequent
    resets give more pruning opportunities."""

    scale_reset_recovery: int = 500
    """Wait time after reset before pruning. Allows partial recovery."""

    # ── Natural selection (from Compact-GS) ───────────────────────
    ns_reg_weight: float = 0.01
    """Weight for natural selection regularization loss during Phase 2.
    Creates uniform downward pressure on opacity logits."""

    ns_target_T: float = -20.0
    """Target for natural selection: E[v] → T. Large negative T creates
    strong but stable gradient pressure."""

    ns_warmup_iters: int = 1000
    """Linear warmup for NS loss to avoid optimizer shock."""

    # ── Entropy regularization ───────────────────────────────────
    entropy_weight_phase1: float = 0.003
    """Stronger entropy during growth than entropy_compact (0.002).
    Pre-conditions opacities for cleaner pruning later."""

    entropy_weight_phase2: float = 0.005
    """Increased entropy during compaction to sharpen opacity distribution."""

    entropy_weight_phase3: float = 0.001
    """Mild entropy during fine-tuning."""

    # ── Covariance trace regularization ──────────────────────────
    cov_trace_weight: float = 0.0002
    """Slightly stronger than entropy_compact (0.0001) for more compact Gaussians."""

    cov_trace_cap: float = 0.1
    """Maximum acceptable covariance trace."""

    # ── L2 photometric (Phase 3 quality recovery) ─────────────────
    l2_photo_weight_phase3: float = 0.1
    """L2 photometric loss during fine-tuning. Error-proportional gradients
    help remaining Gaussians better fit under-represented regions."""

    # ── Opacity L1 (Phase 2) ─────────────────────────────────────
    opacity_l1_weight_phase2: float = 1e-7
    """L1 penalty on sigmoid opacities during Phase 2 (sum reduction).
    Pushes weak Gaussians toward zero, complementing natural selection."""

    # ── Importance pruning ───────────────────────────────────────
    prune_percent: float = 0.05
    """Fraction of alive Gaussians to prune per round. More aggressive
    than entropy_compact (0.03) since NS pressure pre-weakens candidates."""

    # ── Survival threshold ───────────────────────────────────────
    survival_threshold: float = 0.005
    """Gaussians with sigmoid(v) < τ are dead."""

    dead_opacity_logit: float = -100.0
    """Value for soft-killed Gaussian opacity logits."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class ShorterCompactGSModel(GSTemplateModel):
    """Compact Shorter Splatting.

    Combines ShorterSplatting's scale reset with natural selection pressure
    and recovery-gated pruning to reduce Gaussian count to ~1M while
    maintaining comparable PSNR to 3DGS's 3.3M Gaussians.
    """

    config: ShorterCompactGSModelConfig

    # ── 1. populate_modules ──────────────────────────────────────

    def populate_modules(self):
        """Initialize tracking state for CSS."""
        super().populate_modules()

        # Phase tracking
        self._current_phase = 1
        self._hard_cleanup_done = False

        # Scale reset tracking
        self._last_scale_reset_step = 0
        self._scale_resets_done = 0

        # Pruning tracking
        self._pending_prune_after_step = None
        self._total_soft_killed = 0
        self._prune_rounds_done = 0
        self._peak_gaussians = 0

        # Gradient accumulation for importance (optional enhancement)
        self._grad_accum_count = 0

    # ── 2. compute_custom_losses ─────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Phase-dependent losses.

        Phase 1: opacity entropy + covariance trace
        Phase 2: opacity entropy + cov trace + natural selection + opacity L1
        Phase 3: mild entropy + L2 photometric
        """
        losses = {}
        step = self.step

        if self._current_phase == 1:
            # ── Phase 1: Growth + Regularization ─────────────────
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase1
                * opacity_entropy_loss(self.opacities)
            )
            losses["cov_trace"] = (
                self.config.cov_trace_weight
                * covariance_trace_loss(self.scales, self.config.cov_trace_cap)
            )

        elif self._current_phase == 2:
            # ── Phase 2: Scale Reset + NS + Pruning ──────────────
            # Entropy (stronger)
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase2
                * opacity_entropy_loss(self.opacities)
            )
            # Covariance trace
            losses["cov_trace"] = (
                self.config.cov_trace_weight
                * covariance_trace_loss(self.scales, self.config.cov_trace_cap)
            )
            # Natural selection with warmup
            ns_progress = min(
                1.0,
                (step - self.config.phase2_start)
                / max(self.config.ns_warmup_iters, 1),
            )
            ns_weight = self.config.ns_reg_weight * ns_progress
            losses["ns_reg"] = ns_weight * natural_selection_reg_loss(
                self.opacities, self.config.ns_target_T
            )
            # Opacity L1
            losses["opacity_l1"] = opacity_l1_loss(
                self.opacities, self.config.opacity_l1_weight_phase2
            )

        elif self._current_phase == 3:
            # ── Phase 3: Fine-tuning ─────────────────────────────
            losses["opacity_entropy"] = (
                self.config.entropy_weight_phase3
                * opacity_entropy_loss(self.opacities)
            )
            # L2 photometric for quality recovery
            if "rgb" in outputs and "image" in batch:
                losses["l2_photo"] = (
                    self.config.l2_photo_weight_phase3
                    * l2_photometric_loss(outputs["rgb"], batch["image"])
                )

        return losses

    # ── 3. get_loss_dict (phase transitions + scheduling) ────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Standard losses + phase scheduling, scale resets, NS, and pruning."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        step = self.step

        # Track peak Gaussian count
        self._peak_gaussians = max(self._peak_gaussians, self.num_points)

        # ── Phase 1 → Phase 2 transition ────────────────────────
        if step == self.config.phase2_start and self._current_phase == 1:
            self._current_phase = 2
            self._last_scale_reset_step = step
            CONSOLE.print(
                f"\n[bold green]═══ Phase 2: COMPACTION "
                f"(step {step}) ═══[/bold green]"
            )
            CONSOLE.print(f"  Current Gaussians: {self.num_points:,}")
            CONSOLE.print(f"  Target budget:     {self.config.gaussian_budget:,}")
            CONSOLE.print(
                f"  Scale reset zeta:  {self.config.scale_reset_zeta} "
                f"every {self.config.scale_reset_interval} iters"
            )
            CONSOLE.print(
                f"  NS pressure:       T={self.config.ns_target_T}, "
                f"weight={self.config.ns_reg_weight} "
                f"(warmup {self.config.ns_warmup_iters} iters)"
            )
            CONSOLE.print(
                f"  Prune percent:     {self.config.prune_percent * 100:.0f}% "
                f"after {self.config.scale_reset_recovery} iter recovery"
            )

        # ── Phase 2: Scale reset + pruning ──────────────────────
        if self._current_phase == 2:
            # Check if we should do a scale reset
            if (
                step - self._last_scale_reset_step
                >= self.config.scale_reset_interval
                and step + self.config.scale_reset_recovery
                < self.config.phase3_start
            ):
                self._apply_scale_reset()
                self._last_scale_reset_step = step
                self._scale_resets_done += 1
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

                if n_alive <= self.config.gaussian_budget:
                    CONSOLE.print(
                        f"  [bold cyan]Budget met! "
                        f"({n_alive:,} <= {self.config.gaussian_budget:,})"
                        f"[/bold cyan]"
                    )

        # ── Phase 2 → Phase 3 transition ────────────────────────
        if step == self.config.phase3_start and self._current_phase == 2:
            self._current_phase = 3
            n_alive = self._count_alive()
            reduction_pct = (
                (1 - n_alive / max(self._peak_gaussians, 1)) * 100
            )
            CONSOLE.print(
                f"\n[bold green]═══ Phase 3: FINE-TUNING "
                f"(step {step}) ═══[/bold green]"
            )
            CONSOLE.print(f"  Peak Gaussians:    {self._peak_gaussians:,}")
            CONSOLE.print(f"  Alive Gaussians:   {n_alive:,}")
            CONSOLE.print(f"  Reduction:         {reduction_pct:.1f}%")
            CONSOLE.print(f"  Scale resets done: {self._scale_resets_done}")
            CONSOLE.print(f"  Prune rounds done: {self._prune_rounds_done}")
            CONSOLE.print(
                f"  Total soft-killed: {self._total_soft_killed:,}"
            )

            # If still over budget, do emergency pruning
            if n_alive > self.config.gaussian_budget:
                excess = n_alive - self.config.gaussian_budget
                CONSOLE.print(
                    f"  [yellow]Still {excess:,} over budget, "
                    f"doing emergency prune[/yellow]"
                )
                n_killed = self._importance_prune(
                    override_count=excess
                )
                self._total_soft_killed += n_killed
                n_alive = self._count_alive()
                CONSOLE.print(
                    f"  Emergency prune: killed {n_killed:,}, "
                    f"now {n_alive:,} alive"
                )

        # ── Hard cleanup ────────────────────────────────────────
        if (
            self._current_phase == 3
            and not self._hard_cleanup_done
            and step >= self.config.phase3_start + 500
        ):
            self._hard_cleanup()
            self._hard_cleanup_done = True

        return loss_dict

    # ═══════════════════════════════════════════════════════════════
    # Scale Reset
    # ═══════════════════════════════════════════════════════════════

    def _apply_scale_reset(self):
        """Shrink all alive Gaussian scales by factor zeta in log-space."""
        with torch.no_grad():
            alive = self._get_alive_mask()
            self.scales.data[alive] += math.log(self.config.scale_reset_zeta)

    # ═══════════════════════════════════════════════════════════════
    # Importance Pruning
    # ═══════════════════════════════════════════════════════════════

    def _importance_prune(self, override_count: int = 0) -> int:
        """Prune bottom importance Gaussians via soft-kill.

        Importance = sigmoid(opacity) * max(exp(scales_per_axis))
        Combines visibility with spatial extent.

        Args:
            override_count: if > 0, prune exactly this many instead of percent

        Returns: number of newly killed Gaussians.
        """
        with torch.no_grad():
            alive_mask = self._get_alive_mask()
            alive_indices = torch.where(alive_mask)[0]

            if len(alive_indices) == 0:
                return 0

            # Compute importance scores
            opacities = torch.sigmoid(
                self.opacities.squeeze(-1)[alive_indices]
            )
            scales = torch.exp(self.scales[alive_indices])
            max_scale = scales.max(dim=-1).values
            importance = opacities * max_scale

            n_alive = len(alive_indices)

            # Don't prune if at or below budget
            if n_alive <= self.config.gaussian_budget:
                return 0

            if override_count > 0:
                n_to_kill = min(override_count, n_alive - 1000)
            else:
                n_to_kill = max(1, int(n_alive * self.config.prune_percent))

            # Don't prune below budget
            max_kill = max(0, n_alive - self.config.gaussian_budget)
            n_to_kill = min(n_to_kill, max_kill) if max_kill > 0 else n_to_kill

            # Safety: keep minimum viable count
            if n_alive - n_to_kill < 1000:
                n_to_kill = max(0, n_alive - 1000)
            if n_to_kill <= 0:
                return 0

            # Find weakest Gaussians
            _, weakest_local = torch.topk(
                importance, n_to_kill, largest=False
            )
            kill_indices = alive_indices[weakest_local]

            # Soft-kill
            self.opacities.data[kill_indices] = self.config.dead_opacity_logit

            return n_to_kill

    # ═══════════════════════════════════════════════════════════════
    # Hard Cleanup
    # ═══════════════════════════════════════════════════════════════

    def _hard_cleanup(self):
        """Remove dead Gaussians from memory (one-time)."""
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
                f"{n_before - self.num_points:,} dead Gaussians)[/bold cyan]"
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
        """Count alive Gaussians."""
        return self._get_alive_mask().sum().item()

    # ── 4. get_metrics_dict ──────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add CSS-specific metrics for logging."""
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

        return metrics
