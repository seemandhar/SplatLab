"""
Coarse-to-Fine Progressive Resolution Training (C2F-GS)
========================================================
Complexity: ★★★☆☆ (resolution schedule + SH control + densification phases)

A three-phase progressive resolution training schedule with synchronized
densification control and delayed SH activation:

    Phase 1 (0 → 5K):      1/4 resolution, SH degree 0, split-only densification
    Phase 2 (5K → 15K):    ramp to full resolution, SH degree 1→2, clone+split
    Phase 3 (15K → 30K):   full resolution, SH degree 3, prune-only (no grow)

Key ideas:
    - Training at low resolution first lets Gaussians find good initial
      positions cheaply before high-freq detail is introduced.
    - Restricting SH to degree 0 early avoids wasting capacity on view-dependent
      effects when geometry is still coarse.
    - Phase-aware densification: split-only in Phase 1 (grow count conservatively),
      full split+clone in Phase 2 (refine geometry), prune-only in Phase 3
      (compact the representation).
    - EMA loss tracking enables convergence-triggered phase transitions with
      fixed-iteration fallbacks for robustness.

IMPLEMENTATION NOTES:
    1. Resolution control overrides _get_downscale_factor() to implement the
       three-phase schedule instead of splatfacto's geometric ramp.
    2. SH degree control overrides get_outputs() to clamp the SH degree used
       at rasterization time per phase.
    3. Densification control monkey-patches the gsplat DefaultStrategy's
       _grow_gs method at phase transitions to disable clone or grow entirely.
    4. We do NOT override get_loss_dict(). All additions go through
       compute_custom_losses().

Train:
    ns-train c2f-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union, List, Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class C2FGSModelConfig(GSTemplateModelConfig):
    """Coarse-to-Fine GS configuration."""

    _target: Type = field(default_factory=lambda: C2FGSModel)

    # ── Phase boundaries (iteration counts) ─────────────────────────
    phase1_end: int = 5000
    """End of Phase 1 (low-res, SH=0, split-only). Phase 2 starts here."""

    phase2_end: int = 15000
    """End of Phase 2 (resolution ramp, SH 1→2, clone+split). Phase 3 starts here."""

    # ── Resolution schedule ─────────────────────────────────────────
    phase1_downscale: int = 4
    """Downscale factor during Phase 1. 4 means 1/4 resolution."""

    phase2_downscale_start: int = 4
    """Downscale factor at the start of Phase 2 (ramps to 1)."""

    phase2_downscale_end: int = 1
    """Downscale factor at the end of Phase 2 (full resolution)."""

    # ── SH degree schedule ──────────────────────────────────────────
    phase1_sh_degree: int = 0
    """Max SH degree in Phase 1."""

    phase2_sh_degree: int = 2
    """Max SH degree in Phase 2 (ramped from 1 to this value)."""

    phase3_sh_degree: int = 3
    """Max SH degree in Phase 3."""

    # ── EMA convergence detection ───────────────────────────────────
    use_ema_transitions: bool = True
    """Enable EMA-based convergence detection for phase transitions."""

    ema_decay: float = 0.99
    """EMA decay factor for loss tracking."""

    ema_plateau_threshold: float = 0.001
    """Relative change threshold below which we consider the loss plateaued."""

    ema_plateau_window: int = 500
    """Number of consecutive steps the loss must be plateaued to trigger transition."""

    # ── Densification overrides ─────────────────────────────────────
    # Phase 3 disables growing entirely (prune-only)
    phase3_prune_interval: int = 1000
    """How often to run pruning in Phase 3 (via opacity culling)."""

    phase3_prune_opacity_thresh: float = 0.05
    """Opacity threshold for Phase 3 pruning (more aggressive than default)."""

    # ── Output depth for all phases ─────────────────────────────────
    output_depth_during_training: bool = True

    # ── Use absgrad for better densification ────────────────────────
    absgrad: bool = True


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class C2FGSModel(GSTemplateModel):
    """Coarse-to-Fine Progressive Resolution Training.

    Three-phase training with synchronized resolution, SH degree,
    and densification scheduling.
    """

    config: C2FGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Phase tracking
        self._current_phase = 1
        self._phase_just_changed = False

        # EMA loss tracking for convergence-triggered transitions
        self._ema_loss: Optional[float] = None
        self._ema_loss_prev: Optional[float] = None
        self._plateau_count = 0

        # Store original strategy methods so we can restore/swap
        self._original_grow_gs = None
        self._original_prune_gs = None
        self._strategy_patched_phase = 0  # track which phase we patched for

        # Phase 3 pruning state
        self._last_phase3_prune_step = 0

    # ── Phase determination ──────────────────────────────────────────

    def _get_phase(self) -> int:
        """Determine current phase based on iteration count and EMA convergence."""
        step = self.step

        # Hard boundaries (fallbacks)
        if step >= self.config.phase2_end:
            return 3
        if step >= self.config.phase1_end:
            return 2

        # EMA-based early transition from Phase 1 → 2
        if (
            self.config.use_ema_transitions
            and self._current_phase == 1
            and step >= self.config.phase1_end // 2  # at least halfway
            and self._is_loss_plateaued()
        ):
            return 2

        # EMA-based early transition from Phase 2 → 3
        if (
            self.config.use_ema_transitions
            and self._current_phase == 2
            and step >= (self.config.phase1_end + self.config.phase2_end) // 2
            and self._is_loss_plateaued()
        ):
            return 3

        return self._current_phase

    def _is_loss_plateaued(self) -> bool:
        """Check if EMA loss has plateaued."""
        if self._ema_loss is None or self._ema_loss_prev is None:
            return False
        if abs(self._ema_loss_prev) < 1e-8:
            return False
        relative_change = abs(self._ema_loss - self._ema_loss_prev) / abs(self._ema_loss_prev)
        if relative_change < self.config.ema_plateau_threshold:
            self._plateau_count += 1
        else:
            self._plateau_count = 0
        return self._plateau_count >= self.config.ema_plateau_window

    def _update_ema_loss(self, loss_val: float):
        """Update EMA loss tracker."""
        if self._ema_loss is None:
            self._ema_loss = loss_val
        else:
            self._ema_loss_prev = self._ema_loss
            self._ema_loss = self.config.ema_decay * self._ema_loss + (1 - self.config.ema_decay) * loss_val

    # ── Resolution control ───────────────────────────────────────────

    def _get_downscale_factor(self):
        """Override splatfacto's resolution schedule with three-phase C2F schedule."""
        if not self.training:
            return 1

        phase = self._current_phase
        step = self.step

        if phase == 1:
            # Fixed low resolution
            return self.config.phase1_downscale

        elif phase == 2:
            # Linear ramp from phase2_downscale_start to phase2_downscale_end
            phase2_progress = (step - self.config.phase1_end) / max(
                self.config.phase2_end - self.config.phase1_end, 1
            )
            phase2_progress = min(max(phase2_progress, 0.0), 1.0)

            # Downscale must be power of 2 for splatfacto compatibility
            # Ramp: 4 → 2 → 1 across Phase 2
            ds_start = self.config.phase2_downscale_start
            ds_end = self.config.phase2_downscale_end

            # Use log-space interpolation for power-of-2 stepping
            import math
            log_start = math.log2(max(ds_start, 1))
            log_end = math.log2(max(ds_end, 1))
            log_ds = log_start + (log_end - log_start) * phase2_progress
            ds = 2 ** round(log_ds)
            return max(ds, 1)

        else:
            # Phase 3: full resolution
            return 1

    # ── SH degree control ────────────────────────────────────────────

    def _get_sh_degree_to_use(self) -> Optional[int]:
        """Compute the SH degree to use for the current phase/step."""
        if self.config.sh_degree == 0:
            return None  # color-only mode

        phase = self._current_phase
        step = self.step

        if phase == 1:
            return self.config.phase1_sh_degree

        elif phase == 2:
            # Ramp from 1 to phase2_sh_degree across Phase 2
            phase2_progress = (step - self.config.phase1_end) / max(
                self.config.phase2_end - self.config.phase1_end, 1
            )
            phase2_progress = min(max(phase2_progress, 0.0), 1.0)
            # 1 → phase2_sh_degree
            return min(
                1 + int(phase2_progress * (self.config.phase2_sh_degree - 1 + 0.5)),
                self.config.phase2_sh_degree,
            )

        else:
            return self.config.phase3_sh_degree

    # ── Override get_outputs for SH degree control ───────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[Tensor, List]]:
        """Override to control SH degree per phase.

        We temporarily override self.config.sh_degree_interval to force
        the desired SH degree, then restore it after calling super().
        """
        # Update phase
        new_phase = self._get_phase()
        if new_phase != self._current_phase:
            self._phase_just_changed = True
            self._current_phase = new_phase
            self._plateau_count = 0
            self._patch_strategy_for_phase(new_phase)

        # Phase 3 soft pruning: zero out low-opacity Gaussians periodically
        if (
            self.training
            and self._current_phase == 3
            and self.step - self._last_phase3_prune_step >= self.config.phase3_prune_interval
            and self.step > self.config.phase2_end
        ):
            self._last_phase3_prune_step = self.step
            self._soft_prune_phase3()

        # Temporarily manipulate sh_degree_interval to get desired SH degree
        target_sh = self._get_sh_degree_to_use()
        if target_sh is not None:
            # sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
            # We want: min(step // interval, sh_degree) == target_sh
            # So set interval = step // target_sh (if target_sh > 0)
            # or interval = step + 1 (if target_sh == 0, to get 0)
            original_interval = self.config.sh_degree_interval
            original_sh_degree = self.config.sh_degree
            if target_sh == 0:
                self.config.sh_degree_interval = max(self.step + 1, 1000000)
            else:
                # Ensure step // interval == target_sh
                self.config.sh_degree_interval = max(self.step // target_sh, 1)
            self.config.sh_degree = max(target_sh, self.config.sh_degree)

            outputs = super().get_outputs(camera)

            self.config.sh_degree_interval = original_interval
            self.config.sh_degree = original_sh_degree
        else:
            outputs = super().get_outputs(camera)

        return outputs

    # ── Densification control ────────────────────────────────────────

    def _patch_strategy_for_phase(self, phase: int):
        """Patch the gsplat strategy to control split/clone/prune behavior per phase.

        Phase 1: split-only (disable clone/duplicate)
        Phase 2: full split+clone (default behavior)
        Phase 3: no growing at all, prune-only
        """
        from gsplat.strategy import DefaultStrategy

        if not isinstance(self.strategy, DefaultStrategy):
            return

        # Save original methods on first call
        if self._original_grow_gs is None:
            self._original_grow_gs = self.strategy._grow_gs.__func__
            self._original_prune_gs = self.strategy._prune_gs.__func__

        if phase == 1:
            # Split-only: override _grow_gs to skip duplication
            def _grow_gs_split_only(strategy_self, params, optimizers, state, step):
                """Modified _grow_gs that only splits, never duplicates."""
                from gsplat.strategy.ops import split
                count = state["count"]
                grads = state["grad2d"] / count.clamp_min(1)
                device = grads.device

                is_grad_high = grads > strategy_self.grow_grad2d
                is_small = (
                    torch.exp(params["scales"]).max(dim=-1).values
                    <= strategy_self.grow_scale3d * state["scene_scale"]
                )
                is_large = ~is_small

                # Only split, no duplicate
                is_split = is_grad_high & (is_large | is_small)  # split both large and small
                if step < strategy_self.refine_scale2d_stop_iter and "radii" in state and state["radii"] is not None:
                    is_split = is_split | (state["radii"] > strategy_self.grow_scale2d)
                n_split = is_split.sum().item()

                if n_split > 0:
                    split(
                        params=params,
                        optimizers=optimizers,
                        state=state,
                        mask=is_split,
                        revised_opacity=strategy_self.revised_opacity,
                    )
                return 0, n_split  # 0 duplications

            import types
            self.strategy._grow_gs = types.MethodType(_grow_gs_split_only, self.strategy)

        elif phase == 2:
            # Full split+clone: restore original
            import types
            self.strategy._grow_gs = types.MethodType(self._original_grow_gs, self.strategy)

        elif phase == 3:
            # No growing at all: stop densification by setting refine_stop_iter
            # This is cleaner than patching _grow_gs
            self.strategy.refine_stop_iter = self.step  # effectively stops all future refinement

        self._strategy_patched_phase = phase

    @torch.no_grad()
    def _soft_prune_phase3(self):
        """Phase 3 pruning: soft-kill low-opacity Gaussians."""
        if self.num_points < 200:
            return
        opacities = torch.sigmoid(self.opacities.detach()).squeeze(-1)
        prune_mask = opacities < self.config.phase3_prune_opacity_thresh
        if prune_mask.sum() > 0:
            self.opacities.data[prune_mask] = -100.0

    # ── Custom losses (EMA tracking) ─────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Track EMA loss for convergence detection. No extra loss terms."""
        losses = {}

        # Track main loss for EMA convergence detection
        if self.training and "rgb" in outputs:
            gt_img = self.composite_with_background(
                self.get_gt_img(batch["image"]), outputs["background"]
            )
            pred_img = outputs["rgb"]
            with torch.no_grad():
                l1_val = torch.abs(gt_img - pred_img).mean().item()
                self._update_ema_loss(l1_val)

        return losses

    # ── Metrics ──────────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["c2f_phase"] = torch.tensor(float(self._current_phase), dtype=torch.float32)
        metrics["c2f_downscale"] = torch.tensor(float(self._get_downscale_factor()), dtype=torch.float32)
        sh_deg = self._get_sh_degree_to_use()
        metrics["c2f_sh_degree"] = torch.tensor(float(sh_deg if sh_deg is not None else 0), dtype=torch.float32)
        if self._ema_loss is not None:
            metrics["c2f_ema_loss"] = torch.tensor(self._ema_loss, dtype=torch.float32)
        return metrics
