"""
Fifty-Sec GS — 50-second-budget 3DGS with fixed SH=0 + hard Gaussian cap
========================================================================
Idea: idea_0004 (HOOK_PARAM_REPR)

Hypothesis: at a 50-second training budget, splatfacto's progressive SH
band activation and unbounded densification waste compute on parameters
that never converge. Fixing `sh_degree=0` (diffuse color only) and hard-
capping the Gaussian population at 600k via importance-based pruning
every 500 iters should gain +0.2-0.5 dB PSNR.

Inspired by: Speedy-Splat (Hanson 2025), Taming-3DGS (Mallick 2024).

Implementation:
  - `sh_degree=0` is a plain config override (inherited from splatfacto).
  - Hard cap: every `prune_every` iters, if num_points > budget, soft-kill
    (opacity logit = -100) the (num_points - budget) Gaussians with the
    lowest importance score = sigmoid(opacity) * exp(max_scale).
  - After the first cap event, schedule a one-time hard cleanup
    `hard_cleanup_delay` iters later to reclaim memory. Cleanup rebuilds
    `gauss_params` tensors in place; optimizer momentum resets but
    surviving positions/colors/scales are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class FiftySecGSModelConfig(GSTemplateModelConfig):
    """Config for Fifty-Sec GS — SH=0 + 600k hard cap."""

    _target: Type = field(default_factory=lambda: FiftySecGSModel)

    # Fix spherical harmonics at degree 0 (diffuse-only).
    # splatfacto interprets sh_degree as the max band; 0 → only features_dc.
    sh_degree: int = 0

    # Gaussian budget / hard cap.
    gaussian_budget: int = 600_000

    # Pruning cadence.
    prune_every: int = 500
    prune_start_iter: int = 500

    # One-time hard cleanup, triggered this many iters after the first
    # time we actually hit the cap.
    hard_cleanup_delay: int = 1000

    # Opacity logit assigned to soft-killed Gaussians (sigmoid ≈ 0).
    dead_opacity_logit: float = -100.0


class FiftySecGSModel(GSTemplateModel):
    """Fifty-Sec GS: diffuse-only + importance-pruning hard cap."""

    config: FiftySecGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        self._first_cap_step: Optional[int] = None
        self._hard_cleanup_done: bool = False
        self._total_soft_killed: int = 0

    # ── Hook: scheduling on get_loss_dict ──────────────────────────
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        step = self.step

        # Importance-based soft pruning to enforce hard cap.
        if (
            step >= self.config.prune_start_iter
            and step % self.config.prune_every == 0
        ):
            n_killed = self._enforce_budget()
            if n_killed > 0:
                self._total_soft_killed += n_killed
                if self._first_cap_step is None:
                    self._first_cap_step = step
                if step % (self.config.prune_every * 4) == 0:
                    CONSOLE.print(
                        f"  [fifty-sec-gs step {step}] soft-killed "
                        f"{n_killed:,} | alive={self._count_alive():,} "
                        f"/ total={self.num_points:,} "
                        f"(budget={self.config.gaussian_budget:,})"
                    )

        # One-time hard cleanup.
        if (
            not self._hard_cleanup_done
            and self._first_cap_step is not None
            and step >= self._first_cap_step + self.config.hard_cleanup_delay
        ):
            self._hard_cleanup()
            self._hard_cleanup_done = True

        return loss_dict

    # ── Budget enforcement ─────────────────────────────────────────
    def _enforce_budget(self) -> int:
        """Soft-kill lowest-importance Gaussians until alive <= budget."""
        with torch.no_grad():
            alive = self._alive_mask()
            n_alive = int(alive.sum().item())
            budget = self.config.gaussian_budget
            if n_alive <= budget:
                return 0

            n_to_kill = n_alive - budget
            alive_idx = torch.where(alive)[0]

            # Importance = sigmoid(opacity) * max(exp(scale)) per Gaussian.
            opa = torch.sigmoid(self.opacities.squeeze(-1)[alive_idx])
            max_scale = torch.exp(self.scales[alive_idx]).amax(dim=-1)
            importance = opa * max_scale

            _, weakest_local = torch.topk(
                importance, n_to_kill, largest=False
            )
            kill_idx = alive_idx[weakest_local]

            self.opacities.data[kill_idx] = self.config.dead_opacity_logit
            return int(n_to_kill)

    # ── One-time hard cleanup (rebuild gauss_params without dead) ──
    def _hard_cleanup(self):
        with torch.no_grad():
            keep = self._alive_mask()
            n_before = self.num_points
            n_keep = int(keep.sum().item())
            if n_keep == n_before:
                return
            for name in list(self.gauss_params.keys()):
                old = self.gauss_params[name]
                self.gauss_params[name] = Parameter(
                    old.data[keep].contiguous().clone()
                )
            CONSOLE.print(
                f"  [fifty-sec-gs] hard cleanup: {n_before:,} → "
                f"{self.num_points:,} (freed {n_before - n_keep:,})"
            )

    # ── Helpers ────────────────────────────────────────────────────
    def _alive_mask(self) -> Tensor:
        return self.opacities.squeeze(-1) > (
            self.config.dead_opacity_logit + 10.0
        )

    def _count_alive(self) -> int:
        return int(self._alive_mask().sum().item())

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians_alive"] = torch.tensor(
            float(self._count_alive())
        )
        return metrics
