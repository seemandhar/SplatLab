"""
Compound-50s GS — top-3 winners fused across three EVOLVE-BLOCKs
=================================================================
Idea: idea_0009_compound (HOOK_9_DATA + HOOK_1_LOSS + HOOK_DENSIFICATION)

Composes the three best-performing 50s-budget methods into ONE method:

  DATA     : view-curriculum-gs    (phase1_frac=0.30, phase1_secs=15, lambda_rot=0.5)
  LOSS     : two-phase-loss-gs     (L1-only early, +SSIM after t=15s)
  DENSIFY  : fused-score-gs        (pos+color+opacity-var score, co-prune at 25s)

Hypothesis: the three improvements are mechanistically independent:
  - view curriculum affects WHICH gradient signal enters the model (data side),
  - two-phase loss affects the COST of each supervision step,
  - fused-score densification affects WHERE capacity is placed.
Expected: their gains are largely additive for a 50-second wall-clock budget.

Implementation: subclass FusedScoreGSModel (inherits densify + co-prune
verbatim) and graft the two-phase loss switch on top of its
get_loss_dict. The view-curriculum DataManager is injected at the
TrainerConfig level (configs.py); no subclass needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.methods.fused_score_gs import (
    FusedScoreGSModel,
    FusedScoreGSModelConfig,
)


@dataclass
class CompoundFiftyGSModelConfig(FusedScoreGSModelConfig):
    """Compound-50s-gs config — inherits fused-score densify params."""

    _target: Type = field(default_factory=lambda: CompoundFiftyGSModel)

    # Two-phase loss switch. Aligned with view-curriculum phase1_secs so that
    # the transition to full-view + SSIM + (already ongoing) fused densify
    # happens in a single coherent "refinement phase" boundary at t=15s.
    ssim_switch_sec: float = 15.0
    ssim_switch_step: int = 10_000_000
    verbose_switch: bool = True


class CompoundFiftyGSModel(FusedScoreGSModel):
    """Fused densify + co-prune (inherited) + two-phase loss (grafted)."""

    config: CompoundFiftyGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        self._phase_start_t: Optional[float] = None
        self._ssim_active: bool = False

    def _check_phase(self) -> bool:
        """Return True if SSIM should be included in the loss this iter."""
        if self._phase_start_t is None:
            self._phase_start_t = time.time()
        if self._ssim_active:
            return True

        elapsed = time.time() - self._phase_start_t
        step = int(getattr(self, "step", 0))
        if (
            elapsed >= self.config.ssim_switch_sec
            or step >= self.config.ssim_switch_step
        ):
            self._ssim_active = True
            if self.config.verbose_switch:
                CONSOLE.print(
                    f"  [compound-50s-gs] SSIM ENABLED at step={step} "
                    f"elapsed={elapsed:.1f}s "
                    f"(threshold sec={self.config.ssim_switch_sec})"
                )
            return True
        return False

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        # Phase 2: full L1+SSIM via SplatfactoModel.get_loss_dict (through MRO).
        if self._check_phase():
            return super().get_loss_dict(outputs, batch, metrics_dict)

        # Phase 1: replicate splatfacto's loss body but skip SSIM entirely.
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio, device=self.device),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.zeros((), device=self.device)

        loss_dict: Dict[str, Tensor] = {
            "main_loss": Ll1,
            "scale_reg": scale_reg,
        }

        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = (
                    self.config.mcmc_opacity_reg
                    * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = (
                    self.config.mcmc_scale_reg
                    * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )

        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                from nerfstudio.model_components.losses import total_variation_loss
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        loss_dict.update(self.compute_custom_losses(outputs, batch, metrics_dict))
        return loss_dict
