"""
Time-Budget GS — Wall-clock-gated densification for short-budget regimes
=========================================================================
Precedent: Taming-3DGS (SIGGRAPH Asia 24), FastGS (2025).

Motivation: for a 50s total training budget, iter-based `stop_split_at=15000`
leaves the model densifying right up to wall-clock end and never allows
refinement to converge. Gate densification by `time.time()` so that the
first `densify_budget_sec` seconds perform normal grow/prune/reset and the
remaining time is pure refinement of a frozen Gaussian set.

Extension points used:
    populate_modules() — init wall-clock bookkeeping
    step_post_backward — gate densification by elapsed wall-clock seconds
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Type

from gsplat.strategy import DefaultStrategy

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class TimeBudgetGSModelConfig(GSTemplateModelConfig):
    """Wall-clock-budgeted densification schedule."""

    _target: Type = field(default_factory=lambda: TimeBudgetGSModel)

    densify_budget_sec: float = 25.0
    """Seconds of wall-clock densification before freeze."""

    freeze_opacity_reset: bool = True
    """Disable periodic opacity reset once frozen (avoids disturbing refinement)."""


class TimeBudgetGSModel(GSTemplateModel):

    config: TimeBudgetGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        self._train_t0: float | None = None
        self._densify_frozen: bool = False

    def step_post_backward(self, step):
        # Only wall-clock-gate the DefaultStrategy path. MCMC has different semantics.
        if not isinstance(self.strategy, DefaultStrategy):
            return super().step_post_backward(step)

        if self._train_t0 is None:
            self._train_t0 = time.time()

        elapsed = time.time() - self._train_t0

        if not self._densify_frozen and elapsed < self.config.densify_budget_sec:
            # Still within densify window: normal behavior.
            super().step_post_backward(step)
            return

        if not self._densify_frozen:
            # First step past the budget: do one final densify pass, then freeze.
            super().step_post_backward(step)
            self.strategy.refine_stop_iter = step
            if self.config.freeze_opacity_reset:
                # Practically-infinite reset interval => reset never fires again.
                self.strategy.reset_every = 10**9
            self._densify_frozen = True
            return

        # Post-freeze: strategy's own `step >= refine_stop_iter` guard makes
        # super() a no-op, but we avoid the overhead entirely.
        return
