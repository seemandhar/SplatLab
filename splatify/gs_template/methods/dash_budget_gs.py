"""
DashBudgetGS — Wall-Clock-Gated Resolution Ladder + Concave-Up Primitive Cap
============================================================================

A budget-first re-casting of DashGaussian / Turbo-GS style coarse-to-fine
training. Instead of gating stage transitions by iteration count (which is
unreliable because iter time varies with resolution and primitive count),
we gate by *elapsed wall-clock time* against a fixed training budget
(default 50 s).

Stage layout (default):
    Stage 1: [0,            0.25 * budget)  → downscale 4  (25 % res)
    Stage 2: [0.25 * budget, 0.55 * budget) → downscale 2  (50 % res)
    Stage 3: [0.55 * budget, budget]        → downscale 1  (100 % res)

Primitive cap schedule (concave-up in elapsed/budget):
    cap(t) = cap_min + (cap_max - cap_min) * (t/budget)**cap_exponent
with cap_exponent > 1.  This keeps primitive count low while low-frequency
content is fit, then ramps up sharply to fill in high-frequency detail
once full resolution is reached.  The cap is enforced by hard-pruning
the lowest-opacity Gaussians whenever num_points exceeds cap(t).

Train:
    ns-train dash-budget-gs --data <path>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DashBudgetGSModelConfig(GSTemplateModelConfig):
    """Wall-clock-budgeted coarse-to-fine GS with concave-up primitive cap."""

    _target: Type = field(default_factory=lambda: DashBudgetGSModel)

    # ── Wall-clock budget ───────────────────────────────────────────
    budget_seconds: float = 50.0
    """Total training budget in seconds. Stage transitions are fractions of this."""

    stage1_frac: float = 0.25
    """End of stage 1 as a fraction of budget_seconds (stage 2 begins here)."""

    stage2_frac: float = 0.55
    """End of stage 2 as a fraction of budget_seconds (stage 3 begins here)."""

    # ── Resolution ladder (power-of-2 downscale factors) ────────────
    stage1_downscale: int = 4
    stage2_downscale: int = 2
    stage3_downscale: int = 1

    # ── Primitive cap (concave-up growth curve) ─────────────────────
    cap_min: int = 200_000
    cap_max: int = 1_200_000
    cap_exponent: float = 2.0
    """Exponent on normalized elapsed time. >1 → concave-up (slow start, fast finish)."""

    cap_enforce_every: int = 100
    """Enforce the primitive cap every N training steps (and only while training)."""

    # Make sure depth is available if downstream methods need it
    output_depth_during_training: bool = False


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class DashBudgetGSModel(GSTemplateModel):
    """Wall-clock-gated 3-stage resolution ladder + concave-up primitive cap."""

    config: DashBudgetGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Wall-clock origin. Set lazily on first training call so we don't
        # count setup / dataloader warmup.
        self._dash_t0: Optional[float] = None
        self._last_cap_enforce_step: int = -1
        self._current_stage: int = 1

    # ── Clock helpers ───────────────────────────────────────────────

    def _elapsed(self) -> float:
        if self._dash_t0 is None:
            self._dash_t0 = time.time()
            return 0.0
        return time.time() - self._dash_t0

    def _elapsed_frac(self) -> float:
        if self.config.budget_seconds <= 0:
            return 1.0
        return min(max(self._elapsed() / self.config.budget_seconds, 0.0), 1.0)

    def _stage_from_elapsed(self) -> int:
        frac = self._elapsed_frac()
        if frac < self.config.stage1_frac:
            return 1
        if frac < self.config.stage2_frac:
            return 2
        return 3

    def _current_cap(self) -> int:
        """Concave-up cap as a function of elapsed / budget."""
        f = self._elapsed_frac()
        exp = max(self.config.cap_exponent, 0.0)
        scaled = f ** exp
        cap = self.config.cap_min + (self.config.cap_max - self.config.cap_min) * scaled
        return int(cap)

    # ── Resolution control ──────────────────────────────────────────

    def _get_downscale_factor(self):
        """Override splatfacto's schedule with wall-clock-gated stage downscale."""
        if not self.training:
            return 1
        # Triggers _dash_t0 init on first training call.
        stage = self._stage_from_elapsed()
        self._current_stage = stage
        if stage == 1:
            return max(int(self.config.stage1_downscale), 1)
        if stage == 2:
            return max(int(self.config.stage2_downscale), 1)
        return max(int(self.config.stage3_downscale), 1)

    # ── Primitive cap enforcement ───────────────────────────────────

    @torch.no_grad()
    def _enforce_primitive_cap(self):
        """Hard-prune the lowest-opacity Gaussians if num_points exceeds cap(t)."""
        cap = self._current_cap()
        n = self.num_points
        if n <= cap or n < 256:
            return
        # Rank by sigmoid(opacity) ascending; kill the lowest (n - cap) ones.
        # We do this by setting their raw opacity to a very negative value;
        # the standard opacity-cull path in splatfacto will then remove them
        # at the next refine_every boundary. This keeps us safe w.r.t. the
        # gsplat strategy's internal bookkeeping (it expects opacity-based
        # culls, not external param-surgery).
        opa = torch.sigmoid(self.opacities.detach()).squeeze(-1)
        k = n - cap
        # topk with largest=False → the k smallest opacities
        _, kill_idx = torch.topk(opa, k=int(k), largest=False)
        # In-place write on the *raw* opacity parameter; safe outside the
        # graph because we are in no_grad and between training steps.
        self.opacities.data[kill_idx] = -30.0

    # ── Training hook ───────────────────────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[Tensor, List]]:
        # Stage tracking happens inside _get_downscale_factor which is called
        # by super().get_outputs.
        outputs = super().get_outputs(camera)

        if self.training and self.step != self._last_cap_enforce_step:
            if self.step % max(self.config.cap_enforce_every, 1) == 0:
                self._enforce_primitive_cap()
                self._last_cap_enforce_step = self.step

        return outputs

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["dash_elapsed_s"] = torch.tensor(
            float(self._elapsed()), dtype=torch.float32
        )
        metrics["dash_stage"] = torch.tensor(
            float(self._current_stage), dtype=torch.float32
        )
        metrics["dash_downscale"] = torch.tensor(
            float(self._get_downscale_factor()), dtype=torch.float32
        )
        metrics["dash_cap"] = torch.tensor(
            float(self._current_cap()), dtype=torch.float32
        )
        return metrics
