"""
Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting
===============================================================================
Paper: https://arxiv.org/abs/2403.15530
Complexity: ★★★☆☆ (custom rendering for gradient hooks, no new params)

Two modifications to densification:

1. Pixel-aware gradient weighting (Eq. 7):
   Replace the simple view-average of gradient norms with a weighted average
   where each view's contribution is scaled by the number of pixels the
   Gaussian participates in.

2. Scaled gradient field (Eq. 10):
   Multiply each view's gradient contribution by a depth-dependent factor
   f(i,k) = clip((z / (gamma * scene_radius))^2, 0, 1). Gaussians close
   to the camera receive near-zero weight, suppressing floater growth.

Critical implementation details:
   - Scene radius MUST be computed from camera centers (Eq. 8), NOT from
     Gaussian positions. Gaussians include far-away background primitives
     that inflate the radius, making depth scaling suppress everything.
   - Pixel weights are normalized to unit mean per view so that the
     existing densification threshold tau_pos remains effective.

Train:
    ns-train pixel-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.rendering import (
    assemble_outputs,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


@dataclass
class PixelGSModelConfig(GSTemplateModelConfig):
    """Pixel-GS configuration."""

    _target: Type = field(default_factory=lambda: PixelGSModel)

    gamma_depth: float = 0.37
    """Depth scaling hyperparameter (Eq. 9). Constant across all scenes."""

    use_pixel_weighting: bool = True
    """Enable pixel-count weighting of gradients (Eq. 7)."""

    use_depth_scaling: bool = True
    """Enable depth-based gradient scaling (Eq. 9-10)."""


class PixelGSModel(GSTemplateModel):
    """Pixel-GS: pixel-aware gradient and scaled gradient field.

    Overrides get_outputs() to apply gradient scaling via backward hooks.
    """

    config: PixelGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Scene radius from camera centers (computed lazily from training cameras).
        # We collect camera centers during early training steps and compute the
        # radius before the first densification round (step 500 by default).
        self._scene_radius: Optional[float] = None
        self._camera_centers: List[Tensor] = []
        self._radius_computed = False

    # ── Scene radius from camera centers (Eq. 8) ──────────────────

    def _collect_camera_center(self, viewmat: Tensor) -> None:
        """Extract and store the camera center from the current view matrix.

        Camera center in world coordinates: C = -R^T @ t, where the view
        matrix encodes [R | t] in its top-left 3x4 block.
        """
        if self._radius_computed:
            return
        with torch.no_grad():
            R = viewmat[0, :3, :3]
            t = viewmat[0, :3, 3]
            center = -R.T @ t
            self._camera_centers.append(center.cpu())

    def _maybe_compute_scene_radius(self) -> None:
        """Compute scene radius from collected camera centers.

        Paper Eq. 8:  radius = 1.1 * max_i ||C_i - mean(C_j)||_2

        This runs once, after collecting at least 100 camera centers
        (well before the first densification at step 500).
        """
        if self._radius_computed:
            return
        if len(self._camera_centers) < 100:
            return

        centers = torch.stack(self._camera_centers)
        centroid = centers.mean(dim=0)
        dists = (centers - centroid).norm(dim=-1)
        self._scene_radius = max(1.1 * dists.max().item(), 1e-3)
        self._radius_computed = True
        self._camera_centers = []  # free memory

    # ── Forward pass ───────────────────────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Standard rendering with pixel-aware gradient scaling."""

        # Phase 1: camera setup
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        # Gather base Gaussian parameters
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # Phase 2: rasterize (identical to vanilla splatfacto)
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )
        self.info=info

        # Collect camera centers for scene radius (early training only)
        if self.training:
            self._collect_camera_center(cam["viewmat"])
            self._maybe_compute_scene_radius()

        # Apply gradient scaling for densification (training only)
        if self.training and info["means2d"].requires_grad:
            self._apply_gradient_scaling(
                info, means, cam["viewmat"]
            )

        # Phase 3: assemble output dict
        outputs = assemble_outputs(self, render, alpha, info, cam)
        return outputs

    # ── Gradient scaling (core of Pixel-GS) ────────────────────────

    def _apply_gradient_scaling(
        self,
        info: Dict,
        means: Tensor,
        viewmat: Tensor,
    ) -> None:
        """Register a backward hook on means2d to scale densification gradients.

        Pixel-aware weighting (Eq. 7): multiplies each Gaussian's gradient by
        its pixel coverage, normalized to unit mean so tau_pos stays effective.
        Gaussians covering more pixels than average get up-weighted; those
        covering fewer get down-weighted.

        Depth scaling (Eq. 9): multiplies by f(i,k) = clip((z/(gamma*R))^2, 0, 1)
        to suppress near-camera floaters. Only active once scene_radius has been
        computed from camera centers.
        """
        means2d = info["means2d"]
        radii_raw = info["radii"]

        # Flatten batch dim: gsplat may return (1, N) or (N,)
        radii = radii_raw.squeeze(0) if radii_raw.dim() > 1 else radii_raw
        N = radii.shape[0]

        with torch.no_grad():
            scale_factor = torch.ones(N, device=means.device, dtype=torch.float32)
            visible = radii > 0

            # ── Pixel-count weighting (Eq. 7) ──────────────────────
            if self.config.use_pixel_weighting:
                # Approximate participating pixel count as pi * radius^2.
                # The rasterization radius is the 3-sigma boundary, so
                # this closely matches the paper's m_k^i definition.
                pixel_counts = math.pi * radii.float().clamp(min=0.5) ** 2
                pixel_counts[~visible] = 0.0

                # Normalize to unit mean across visible Gaussians.
                # This preserves relative weighting (large-projection
                # Gaussians get stronger signal) while keeping absolute
                # gradient magnitudes compatible with the unchanged
                # densification threshold tau_pos = 0.0002.
                if visible.any():
                    mean_px = pixel_counts[visible].mean().clamp(min=1.0)
                    pixel_weight = pixel_counts / mean_px
                else:
                    pixel_weight = pixel_counts

                pixel_weight[~visible] = 0.0
                scale_factor = scale_factor * pixel_weight

            # ── Depth-based scaling (Eq. 9) ────────────────────────
            if self.config.use_depth_scaling and self._scene_radius is not None:
                # Camera-space depth: z-component of (R @ mean + t)
                R = viewmat[0, :3, :3]
                t = viewmat[0, :3, 3]
                cam_z = (means @ R.T + t.unsqueeze(0))[:, 2]

                # f(i,k) = clip((z / (gamma * scene_radius))^2, 0, 1)
                gamma_r = self.config.gamma_depth * self._scene_radius
                depth_scale = (cam_z / gamma_r).pow(2).clamp(0.0, 1.0)
                depth_scale[~visible] = 0.0

                scale_factor = scale_factor * depth_scale

        # Register the backward hook on means2d.
        # This fires during loss.backward() and rescales the gradient before
        # the densification strategy reads it in step_post_backward.
        def _scale_hook(grad: Tensor, sf: Tensor = scale_factor) -> Tensor:
            if grad.dim() == 3:
                # (1, N, 2) batched layout
                return grad * sf.unsqueeze(0).unsqueeze(-1)
            else:
                # (N, 2) flat layout
                return grad * sf.unsqueeze(-1)

        means2d.register_hook(_scale_hook)
        means2d.retain_grad()