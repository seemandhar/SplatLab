"""
Fused-Score GS — joint position+color+opacity score densification with
wall-clock one-shot co-prune (idea_0008, HOOK_DENSIFICATION).

Precedent: Turbo-GS (color-grad score), LiteGS (opacity-grad variance),
Taming-3DGS (budgeted top-K scoring replaces threshold).

Mechanism:
  - Accumulate three per-Gaussian statistics: position-grad (already tracked
    by DefaultStrategy as grad2d), color-grad magnitude, and opacity-grad
    variance across refine_every iterations.
  - At each refine event, densify the top-K fraction by a normalized
    weighted sum, small => duplicate, large => split.
  - At wall-clock t = coprune_t_sec (one shot), hard-remove any Gaussian
    with opacity < coprune_opa OR max_scale > coprune_scale_frac*scene_scale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Type

import torch
from torch import Tensor

from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, split, remove

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class FusedScoreGSModelConfig(GSTemplateModelConfig):
    """Config for fused-score densification + wall-clock co-prune."""

    _target: Type = field(default_factory=lambda: FusedScoreGSModel)

    # Weights for the fused score.
    w_pos: float = 1.0
    w_col: float = 0.5
    w_opa: float = 0.3

    # Fraction of Gaussians to densify each refine event (top-K selection).
    densify_topk_frac: float = 0.10

    # One-shot wall-clock co-prune parameters.
    coprune_t_sec: float = 25.0
    coprune_opa: float = 0.01
    coprune_scale_frac: float = 0.2


class FusedScoreGSModel(GSTemplateModel):
    """Fused-score densification + one-shot co-prune."""

    config: FusedScoreGSModelConfig

    # ── setup ──────────────────────────────────────────────────────
    def populate_modules(self):
        super().populate_modules()
        self._train_t0: float | None = None
        self._coprune_done: bool = False
        self._grow_patched: bool = False
        # Buffers allocated lazily to match the param device at train time.
        self._col_g_sum: Tensor | None = None
        self._opa_g_sum: Tensor | None = None
        self._opa_g_sqsum: Tensor | None = None
        self._score_count: Tensor | None = None

    # ── main hook ──────────────────────────────────────────────────
    def step_post_backward(self, step):
        if not isinstance(self.strategy, DefaultStrategy):
            return super().step_post_backward(step)

        if self._train_t0 is None:
            self._train_t0 = time.time()
        elapsed = time.time() - self._train_t0

        # Patch strategy._grow_gs once, so refine events use our fused score.
        if not self._grow_patched:
            self._install_fused_grow()
            self._grow_patched = True

        # Accumulate color and opacity gradient statistics BEFORE super,
        # because super() may zero strategy state on refine events.
        self._accumulate_grad_stats()

        # Resize buffers if population changed since last step.
        self._sync_buffer_size()

        # Normal behavior (which now invokes our patched _grow_gs).
        super().step_post_backward(step)

        # Reset our stats when strategy resets its own running stats.
        # DefaultStrategy zeros state["grad2d"]/state["count"] after refine.
        if self.strategy_state.get("count") is not None:
            if float(self.strategy_state["count"].sum().item()) == 0.0:
                self._reset_score_buffers()

        # One-shot wall-clock co-prune at coprune_t_sec.
        if (not self._coprune_done) and elapsed >= self.config.coprune_t_sec:
            self._coprune()
            self._coprune_done = True

    # ── buffer bookkeeping ────────────────────────────────────────
    def _sync_buffer_size(self):
        n = self.means.shape[0]
        dev = self.means.device
        if (
            self._col_g_sum is None
            or self._col_g_sum.shape[0] != n
            or self._col_g_sum.device != dev
        ):
            self._col_g_sum = torch.zeros(n, device=dev)
            self._opa_g_sum = torch.zeros(n, device=dev)
            self._opa_g_sqsum = torch.zeros(n, device=dev)
            self._score_count = torch.zeros(n, device=dev)

    def _reset_score_buffers(self):
        if self._col_g_sum is None:
            return
        self._col_g_sum.zero_()
        self._opa_g_sum.zero_()
        self._opa_g_sqsum.zero_()
        self._score_count.zero_()

    def _accumulate_grad_stats(self):
        # features_dc is [N, 1, 3]; opacities is [N, 1]. Gradients are only
        # non-None if loss.backward() just ran with these leaves requiring grad.
        self._sync_buffer_size()
        fdc = self.gauss_params.get("features_dc", None)
        opa = self.gauss_params.get("opacities", None)
        n = self.means.shape[0]
        if fdc is not None and fdc.grad is not None:
            g = fdc.grad.detach().reshape(n, -1)
            self._col_g_sum += g.abs().sum(dim=-1)
        if opa is not None and opa.grad is not None:
            g = opa.grad.detach().reshape(n)
            self._opa_g_sum += g
            self._opa_g_sqsum += g * g
        self._score_count += 1.0

    # ── strategy patch ────────────────────────────────────────────
    def _install_fused_grow(self):
        strat = self.strategy
        model = self

        def _fused_grow_gs(params, optimizers, state, step):
            count = state["count"].clamp_min(1)
            pos_grad = state["grad2d"] / count
            n = pos_grad.shape[0]

            # If our side buffers are missing or misaligned, fall back to
            # position-only scoring for this refine event.
            if (
                model._score_count is None
                or model._score_count.shape[0] != n
                or model._col_g_sum.device != pos_grad.device
            ):
                col_grad = torch.zeros_like(pos_grad)
                opa_var = torch.zeros_like(pos_grad)
            else:
                sc = model._score_count.clamp_min(1)
                col_grad = model._col_g_sum / sc
                opa_var = (model._opa_g_sqsum / sc) - (model._opa_g_sum / sc) ** 2
                opa_var = opa_var.clamp_min(0.0)

            def _norm(x: Tensor) -> Tensor:
                m = x.mean().clamp_min(1e-8)
                return x / m

            score = (
                model.config.w_pos * _norm(pos_grad)
                + model.config.w_col * _norm(col_grad)
                + model.config.w_opa * _norm(opa_var)
            )

            n = score.numel()
            k = max(1, int(model.config.densify_topk_frac * n))
            topk_idx = torch.topk(score, k, largest=True).indices
            is_grad_high = torch.zeros(n, dtype=torch.bool, device=score.device)
            is_grad_high[topk_idx] = True

            is_small = (
                torch.exp(params["scales"]).max(dim=-1).values
                <= strat.grow_scale3d * state["scene_scale"]
            )
            is_dupli = is_grad_high & is_small
            is_split = is_grad_high & (~is_small)
            if step < strat.refine_scale2d_stop_iter and state.get("radii") is not None:
                is_split = is_split | (state["radii"] > strat.grow_scale2d)

            n_dupli = int(is_dupli.sum().item())
            n_split = int(is_split.sum().item())

            if n_dupli > 0:
                duplicate(
                    params=params, optimizers=optimizers, state=state, mask=is_dupli
                )

            is_split = torch.cat(
                [
                    is_split,
                    torch.zeros(n_dupli, dtype=torch.bool, device=score.device),
                ]
            )

            if n_split > 0:
                split(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=is_split,
                    revised_opacity=strat.revised_opacity,
                )
            return n_dupli, n_split

        strat._grow_gs = _fused_grow_gs  # type: ignore[assignment]

    # ── one-shot co-prune ─────────────────────────────────────────
    @torch.no_grad()
    def _coprune(self):
        n = self.means.shape[0]
        opa = torch.sigmoid(self.opacities.flatten())
        max_scale = torch.exp(self.scales).amax(dim=-1)
        scene_scale = float(self.strategy_state.get("scene_scale", 1.0))
        is_prune = (opa < self.config.coprune_opa) | (
            max_scale > self.config.coprune_scale_frac * scene_scale
        )
        n_prune = int(is_prune.sum().item())
        if n_prune == 0 or n_prune >= n:
            return
        remove(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            mask=is_prune,
        )
        # Shrink our buffers: resync to new size.
        self._sync_buffer_size()
