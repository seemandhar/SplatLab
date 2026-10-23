"""
2D Gaussian Splatting for Geometrically Accurate Radiance Fields
=================================================================
Paper: https://surfsplatting.github.io  (Huang et al., SIGGRAPH 2024)
Complexity: ★★★☆☆ (custom losses + rendering override for normals)

Core idea: represent the scene with 2D oriented disks rather than 3D
ellipsoids to improve surface geometry.  The paper contributes three
things that matter at training time:

  1.  Planar Gaussians -- one scale axis is driven toward zero so each
      primitive degenerates into a flat disk.
  2.  Depth distortion loss -- encourages the blending-weight
      distribution along each ray to concentrate at a single depth,
      suppressing semi-transparent "clouds".
  3.  Normal consistency loss -- aligns per-Gaussian surface normals
      with the normals derived from the rendered depth gradient,
      enforcing globally coherent orientation.

The perspective-correct ray-splat intersection that gives 2DGS its
name lives inside the CUDA rasterizer (gsplat handles this when its
2DGS backend is compiled).  Everything else is Python-level and
implemented here.

Architecture:
    Extends GSTemplateModel (inherits SplatfactoModel).
    Overrides get_outputs() to render per-Gaussian normals alongside
    RGB in a single extra rasterization call, then computes depth-
    gradient normals from the rendered depth map.
    Overrides compute_custom_losses() for the three regularizers.

Demonstrates:
    - Overriding get_outputs() for a lightweight second render pass
    - Adding geometry-aware losses via compute_custom_losses()
    - Quaternion-to-normal extraction for Gaussian orientation

Train:
    ns-train twodgs --data <path>
    ns-train twodgs --data <path> --pipeline.model.depth_distortion_weight 100
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    depth_distortion_loss_2dgs,
    normal_consistency_loss,
    scale_flatten_loss_min,
)
from gs_template.rendering import (
    assemble_outputs,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# ===================================================================
# Helpers
# ===================================================================

def quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert (N, 4) unit quaternions to (N, 3, 3) rotation matrices.

    Quaternion convention: (w, x, y, z).
    """
    quats = F.normalize(quats, dim=-1)
    w, x, y, z = quats.unbind(-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def gaussian_normals_from_quats(quats: Tensor, scales_log: Tensor) -> Tensor:
    """Extract per-Gaussian surface normals.

    For each Gaussian the normal is the column of R that corresponds
    to the *smallest* scale axis, which is the thinnest direction.
    When the scale-flatten loss is active this axis shrinks toward
    zero, turning the Gaussian into a disk whose face-normal is
    exactly this column.

    Args:
        quats:      (N, 4) raw quaternions (not necessarily unit)
        scales_log: (N, 3) log-space scales

    Returns:
        normals: (N, 3) unit normals
    """
    R = quat_to_rotmat(quats)                          # (N, 3, 3)
    scales = torch.exp(scales_log)                      # (N, 3)
    min_idx = scales.argmin(dim=-1)                     # (N,)
    # Gather the column of R corresponding to the smallest scale
    normals = R[torch.arange(R.shape[0], device=R.device), :, min_idx]
    normals = F.normalize(normals, dim=-1)
    return normals


def depth_to_normal(depth: Tensor, K: Tensor) -> Tensor:
    """Compute surface normals from a rendered depth map using finite
    differences in 3D.

    Args:
        depth: (H, W, 1) rendered depth
        K:     (1, 3, 3) camera intrinsics

    Returns:
        normal_map: (H, W, 3) unit normals in camera space
    """
    H, W, _ = depth.shape
    fx = K[0, 0, 0]
    fy = K[0, 1, 1]
    cx = K[0, 0, 2]
    cy = K[0, 1, 2]

    # Pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )

    z = depth[..., 0]                # (H, W)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # Finite differences
    dx = torch.zeros_like(pts)
    dy = torch.zeros_like(pts)
    dx[:-1, :, :] = pts[1:, :, :] - pts[:-1, :, :]
    dy[:, :-1, :] = pts[:, 1:, :] - pts[:, :-1, :]

    normal = torch.cross(dx, dy, dim=-1)              # (H, W, 3)
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    return normal


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class TwoDGSModelConfig(GSTemplateModelConfig):
    """2D Gaussian Splatting configuration.

    Hyperparameter defaults follow the original paper:
        alpha = 1000 for bounded scenes (100 for unbounded)
        beta  = 0.05
        gradient threshold = 0.0002
        prune opacity < 0.05 every 3000 steps
    """

    _target: Type = field(default_factory=lambda: TwoDGSModel)

    # Depth must be rendered during training for both losses
    output_depth_during_training: bool = True

    # -- Depth distortion loss (Section 5, Eq. 15) --
    depth_distortion_weight: float = 1000.0
    """alpha in the paper.  1000 for bounded, 100 for unbounded."""

    # -- Normal consistency loss (Section 5, Eq. 16) --
    normal_consistency_weight: float = 0.05
    """beta in the paper."""

    # -- Scale flattening (drives one axis toward zero) --
    scale_flatten_weight: float = 0.01
    """Penalty on the smallest scale to keep Gaussians planar."""

    # Densification gradient threshold (paper value)
    continue_cull_post_densification: bool = False
    cull_alpha_thresh: float = 0.05
    """Prune Gaussians with opacity below this threshold."""

    refine_every: int = 100
    """How often to run densification (default splatfacto value)."""


# ===================================================================
# Model
# ===================================================================

class TwoDGSModel(GSTemplateModel):
    """2D Gaussian Splatting for geometrically accurate radiance fields.

    Overrides:
        get_outputs           -- adds a second rasterization pass to
                                 render per-Gaussian normals, plus
                                 computes depth-gradient normals
        compute_custom_losses -- depth distortion, normal consistency,
                                 and scale flattening
    """

    config: TwoDGSModelConfig

    # ── get_outputs: render RGB+depth, then normals ───────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Two-pass rendering: RGB + depth, then surface normals."""

        # ── Phase 1: camera setup ─────────────────────────────────
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # ── Phase 2a: standard RGB + depth rasterization ──────────
        render_rgb, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )
        self.info = info

        outputs = assemble_outputs(
            self, render_rgb, alpha, info, cam,
        )

        # ── Phase 2b: render per-Gaussian normals ─────────────────
        # Compute normals from quaternions and orient toward camera
        raw_normals = gaussian_normals_from_quats(quats, scales)

        # Orient normals toward camera (the "near" side of the disk)
        viewmat = cam["viewmat"]                        # (1, 4, 4)
        cam_pos = torch.inverse(viewmat[0])[:3, 3]     # (3,)
        view_dirs = F.normalize(means - cam_pos, dim=-1)
        dot = (raw_normals * view_dirs).sum(-1, keepdim=True)
        oriented_normals = torch.where(dot > 0, -raw_normals, raw_normals)

        # Lightweight rasterization of just the normal vectors.
        # We reuse the same Gaussian geometry (means, quats, scales,
        # opacities) but swap colors for the 3-channel normals.
        render_n, _, _ = rasterize_gaussians(
            self, means, quats, scales, opacities, oriented_normals,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            sh_degree=None,          # pre-computed, no SH
            render_mode="RGB",       # normals only, no depth channel
        )

        rendered_normal = render_n.squeeze(0)           # (H, W, 3)
        rendered_normal = F.normalize(rendered_normal, dim=-1, eps=1e-8)
        outputs["rendered_normal"] = rendered_normal

        # ── Phase 3: depth-gradient normals ───────────────────────
        if outputs.get("depth") is not None:
            depth_normal = depth_to_normal(outputs["depth"], cam["K"])
            outputs["depth_normal"] = depth_normal

        return outputs

    # ── compute_custom_losses ─────────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}

        # 1. Scale flatten -- encourage planar Gaussians
        if self.config.scale_flatten_weight > 0:
            losses["scale_flatten"] = (
                self.config.scale_flatten_weight
                * scale_flatten_loss_min(self.scales)
            )

        # 2. Depth distortion -- concentrate ray weights in depth
        if (
            self.config.depth_distortion_weight > 0
            and outputs.get("depth") is not None
            and outputs.get("accumulation") is not None
        ):
            losses["depth_distortion"] = (
                self.config.depth_distortion_weight
                * depth_distortion_loss_2dgs(
                    outputs["depth"], outputs["accumulation"]
                )
            )

        # 3. Normal consistency -- align Gaussian normals with depth
        if (
            self.config.normal_consistency_weight > 0
            and "rendered_normal" in outputs
            and "depth_normal" in outputs
        ):
            losses["normal_consistency"] = (
                self.config.normal_consistency_weight
                * normal_consistency_loss(
                    outputs["rendered_normal"],
                    outputs["depth_normal"],
                    outputs.get("accumulation"),
                )
            )

        return losses

    # ── Viewer outputs ────────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box=None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)

        # Also expose normals in the viewer
        if "rendered_normal" in outs:
            # Map [-1, 1] normals to [0, 1] for visualization
            outs["normal_vis"] = outs["rendered_normal"] * 0.5 + 0.5

        return outs
