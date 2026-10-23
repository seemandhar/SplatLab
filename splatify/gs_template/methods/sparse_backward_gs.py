"""
Sparse-Backward GS — visibility-gated per-tile sparse backward
==============================================================
Idea: idea_0006 (HOOK_4_OPTIMIZATION)

Forward the full render every iter, but only propagate gradients through
tiles whose per-tile L1 residual exceeds a per-view rolling-median
threshold. Post ``skip_start_iter`` additionally skip the entire backward
every ``skip_every``-th iter (SkipGS-style).

Precedent: SkipGS (li2026skipgs), Mini-Splatting2 (fang2024minisplatting2),
LiteGS (liao2025litegs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class SparseBackwardGSModelConfig(GSTemplateModelConfig):
    """Config for visibility-gated sparse-backward GS."""

    _target: Type = field(default_factory=lambda: SparseBackwardGSModel)

    # Tile size (pixels) for per-tile gating.
    tile_size: int = 16

    # Iters before any gating kicks in (model needs coarse convergence first).
    gate_start_iter: int = 500

    # EMA factor for per-view median buffer (new sample weight).
    view_median_ema: float = 0.1

    # SkipGS-style full-backward skip schedule.
    skip_start_iter: int = 2500
    skip_every: int = 3  # skip 1-in-3; 0 or 1 disables.

    # Safety: guarantee at least this fraction of tiles stay active.
    min_active_frac: float = 0.25


class SparseBackwardGSModel(GSTemplateModel):
    """Visibility-gated sparse-backward splatfacto variant."""

    config: SparseBackwardGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Per-view rolling median; keyed by image index.
        self._view_median: Dict[int, float] = {}
        self._gate_stats = {"active_frac_ema": 1.0, "n_calls": 0}

    # ── Hook: override loss to inject tile-gated residual ──────────
    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, Tensor]:
        step = int(self.step)

        # SkipGS full-iter skip: detach the entire main loss.
        skip_this_iter = (
            self.config.skip_every > 1
            and step >= self.config.skip_start_iter
            and (step % self.config.skip_every) == 0
        )

        # Below warmup: plain parent behaviour.
        if step < self.config.gate_start_iter and not skip_this_iter:
            return super().get_loss_dict(outputs, batch, metrics_dict)

        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask_b = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask_b.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask_b
            pred_img = pred_img * mask_b

        # Build tile gate (no grad).
        gate = self._build_tile_gate(pred_img, gt_img, batch, step)

        # Apply gate: outside active tiles, replace pred with its detached
        # value so loss is still finite but gradients vanish on those pixels.
        pred_for_loss = pred_img * gate + pred_img.detach() * (1.0 - gate)

        Ll1 = torch.abs(gt_img - pred_for_loss).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...],
            pred_for_loss.permute(2, 0, 1)[None, ...],
        )

        if self.config.use_scale_regularization and step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0, device=self.device)

        main_loss = (
            (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss
        )
        if skip_this_iter:
            main_loss = main_loss.detach()

        loss_dict = {"main_loss": main_loss, "scale_reg": scale_reg}

        # mcmc extras (mirror splatfacto).
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = (
                    self.config.mcmc_opacity_reg
                    * torch.abs(torch.sigmoid(self.opacities)).mean()
                )
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = (
                    self.config.mcmc_scale_reg
                    * torch.abs(torch.exp(self.scales)).mean()
                )

        return loss_dict

    # ── Gate construction ──────────────────────────────────────────
    def _build_tile_gate(
        self,
        pred_img: Tensor,
        gt_img: Tensor,
        batch: Dict,
        step: int,
    ) -> Tensor:
        """Returns a [H,W,1] float tensor in {0,1} (no grad)."""
        H, W, _ = pred_img.shape
        ts = self.config.tile_size
        with torch.no_grad():
            res = torch.abs(gt_img - pred_img).mean(dim=-1)  # [H,W]
            # pool to tile grid; pad to multiples of ts via avg_pool2d ceil.
            res_4d = res[None, None]  # [1,1,H,W]
            tile_l1 = F.avg_pool2d(
                res_4d, kernel_size=ts, stride=ts, ceil_mode=True
            )[0, 0]  # [Ht,Wt]

            view_key = int(batch.get("image_idx", -1))
            cur_median = float(tile_l1.median().item())
            ema = self.config.view_median_ema
            prev = self._view_median.get(view_key, cur_median)
            thresh = (1.0 - ema) * prev + ema * cur_median
            self._view_median[view_key] = thresh

            tile_mask = (tile_l1 >= thresh).float()  # [Ht,Wt]

            # Safety: keep at least `min_active_frac` of tiles active.
            active_frac = tile_mask.mean().item()
            if active_frac < self.config.min_active_frac:
                k = max(
                    1,
                    int(
                        self.config.min_active_frac
                        * tile_l1.numel()
                    ),
                )
                flat = tile_l1.reshape(-1)
                topk_vals, _ = torch.topk(flat, k)
                thresh2 = topk_vals.min()
                tile_mask = (tile_l1 >= thresh2).float()

            # Upsample to [H,W].
            gate_big = F.interpolate(
                tile_mask[None, None],
                size=(H, W),
                mode="nearest",
            )[0, 0]  # [H,W]

            # Running stats.
            s = self._gate_stats
            s["n_calls"] += 1
            s["active_frac_ema"] = (
                0.95 * s["active_frac_ema"]
                + 0.05 * float(tile_mask.mean().item())
            )

        return gate_big.unsqueeze(-1)  # [H,W,1]

    # ── Metrics ────────────────────────────────────────────────────
    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["gate_active_frac_ema"] = torch.tensor(
            float(self._gate_stats["active_frac_ema"])
        )
        return metrics
