"""
Two-Phase Loss GS — L1-only early, L1+SSIM late
================================================
Idea: idea_0005 (HOOK_1_LOSS)

Hypothesis: the 11×11 SSIM window-conv in splatfacto's photometric loss
costs ~18% of the per-iteration budget.  For a 50-second training
budget, skipping SSIM during the first ~30s (structure-forming phase,
where L1 alone is a sufficient supervisor) buys ~1.2k extra iters, and
reintroducing it for the final ~20s (refinement phase) recovers the
perceptual contribution.  Expected: +0.2–0.4 dB PSNR at 50s.

Precedent: SkipGS (Li 2026) shows selective backward-pass skipping
works; Taming-3DGS (Mallick 2024) shows SSIM's impact is concentrated
in the late refinement phase.

Implementation: override get_loss_dict() to compute SSIM only after a
wall-clock threshold.  The switch is time-gated (not iter-gated) so
that the cost saving is robust to per-iter speedups from other
optimizations.  A step-gated fallback is included for deterministic
testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class TwoPhaseLossGSModelConfig(GSTemplateModelConfig):
    """Two-phase photometric loss config."""

    _target: Type = field(default_factory=lambda: TwoPhaseLossGSModel)

    # Wall-clock (seconds) at which SSIM is re-enabled. 30s = 60% of a
    # 50s budget, matching the hypothesis.
    ssim_switch_sec: float = 30.0

    # Iter-based fallback trigger. Whichever fires first enables SSIM.
    # Default effectively disables step-gating; set finite for CI.
    ssim_switch_step: int = 10_000_000

    # Log the switch event once.
    verbose_switch: bool = True


class TwoPhaseLossGSModel(GSTemplateModel):
    """Two-phase photometric loss: L1-only → L1+SSIM at a wall-clock switch."""

    config: TwoPhaseLossGSModelConfig

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
                    f"  [two-phase-loss-gs] SSIM ENABLED at step={step} "
                    f"elapsed={elapsed:.1f}s "
                    f"(threshold sec={self.config.ssim_switch_sec}, "
                    f"step={self.config.ssim_switch_step})"
                )
            return True
        return False

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        # Phase 2: exact splatfacto behavior via parent.
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

        # Full weight on L1 (equivalent to ssim_lambda=0) but without
        # paying the SSIM conv cost.
        loss_dict: Dict[str, Tensor] = {
            "main_loss": Ll1,
            "scale_reg": scale_reg,
        }

        # mcmc regularizers (mirror splatfacto)
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

        # Plus any subclass/custom losses (none here but preserves the pattern).
        loss_dict.update(self.compute_custom_losses(outputs, batch, metrics_dict))
        return loss_dict
