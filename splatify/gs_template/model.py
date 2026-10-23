"""
gs_template/model.py — Core Gaussian Splatting Model Template
==============================================================

Extends SplatfactoModel to inherit ALL battle-tested logic.
Paper authors subclass GSTemplateModel and override only what they need.

EXTENSION POINTS (search "EXTENSION POINT"):
    1. populate_modules()          — Add new params, MLPs, state
    2. get_outputs()               — Custom rendering (use rendering.py helpers)
    3. compute_custom_losses()     — Novel loss terms (use losses.py functions)
    4. split_gaussians()           — Handle new params during densification
    5. get_gaussian_param_groups() — Register Gaussian param optimizers
    6. get_param_groups()          — Register non-Gaussian params (MLPs)
    7. get_outputs_for_camera()    — Custom viewer / eval outputs
    8. load_state_dict()           — Checkpoint loading for new params
    9. get_metrics_dict()          — Custom evaluation metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class GSTemplateModelConfig(SplatfactoModelConfig):
    """Configuration extending SplatfactoModelConfig.

    Inherits: sh_degree, warmup_length, refine_every, resolution_schedule,
    background_color, cull_alpha_thresh, cull_scale_thresh,
    use_scale_regularization, rasterize_mode, output_depth_during_training,
    strategy, camera_optimizer, and more.
    """

    _target: Type = field(default_factory=lambda: GSTemplateModel)


class GSTemplateModel(SplatfactoModel):
    """Base Gaussian Splatting method template.

    Inherits: rendering, densification, optimizer wiring, SH training,
    camera optimization, background compositing, bilateral grid,
    scale regularization.
    """

    config: GSTemplateModelConfig

    # EXTENSION POINT 1 ─────────────────────────────────────────────
    def populate_modules(self):
        """Add new parameters / modules. Always call super() first."""
        super().populate_modules()

    # EXTENSION POINT 2 ─────────────────────────────────────────────
    # Override get_outputs() only if rendering changes.
    # Use gs_template.rendering helpers to avoid boilerplate.
    # MUST set self.xys and self.radii for densification.

    # EXTENSION POINT 3 ─────────────────────────────────────────────
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Base losses + custom losses via compute_custom_losses()."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict.update(
            self.compute_custom_losses(outputs, batch, metrics_dict)
        )
        return loss_dict

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Override to add paper-specific losses. Return {name: scalar}."""
        return {}

    # EXTENSION POINT 4 ─────────────────────────────────────────────
    def split_gaussians(self, split_mask, samps):
        """Override to handle new params during densification."""
        return super().split_gaussians(split_mask, samps)

    # EXTENSION POINT 5 ─────────────────────────────────────────────
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Override to register new Gaussian params for optimization."""
        return super().get_gaussian_param_groups()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Override to register non-Gaussian params (MLPs, etc.)."""
        return super().get_param_groups()

    # EXTENSION POINT 6 ─────────────────────────────────────────────
    def get_outputs_for_camera(self, camera: Cameras, obb_box=None):
        """Override for custom viewer / eval outputs."""
        return super().get_outputs_for_camera(camera, obb_box)

    # EXTENSION POINT 7 ─────────────────────────────────────────────
    def load_state_dict(self, dict, **kwargs):
        """Override for checkpoint loading with new params."""
        lpips_keys = [k for k in list(dict.keys()) if "lpips" in k]
        for k in lpips_keys:
            dict.pop(k)

        super().load_state_dict(dict, **kwargs)

    # EXTENSION POINT 8 ─────────────────────────────────────────────
    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = self.num_points  # logs to wandb/tensorboard
        return metrics
    # Convenience ───────────────────────────────────────────────────
    @property
    def num_points(self) -> int:
        return self.means.shape[0]
