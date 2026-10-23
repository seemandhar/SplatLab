"""
AbsGS — Recovering Fine Details for 3D Gaussian Splatting
===========================================================
Paper: https://arxiv.org/abs/2404.10484
Complexity: ★☆☆☆☆ (config-only)

The paper's contribution: use absolute-value gradients (absgrad) for
densification instead of averaged gradients. This is already a flag
in gsplat/splatfacto — the method is pure configuration.

Demonstrates:
    - Subclassing GSTemplateModelConfig to change defaults
    - Zero code changes in the model class

Train:
    ns-train absgs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


@dataclass
class AbsGSModelConfig(GSTemplateModelConfig):
    """AbsGS: absolute gradients for densification.

    The only change from vanilla splatfacto is absgrad=True,
    which gsplat already supports natively.
    """

    _target: Type = field(default_factory=lambda: AbsGSModel)

    # Core contribution: use absolute gradient accumulation
    # (SplatfactoModelConfig already has this field; we override default)
    absgrad: bool = True


class AbsGSModel(GSTemplateModel):
    """AbsGS model — no code overrides needed.

    Everything is handled by setting absgrad=True in the config,
    which flows through to gsplat's rasterization call.
    """

    config: AbsGSModelConfig
