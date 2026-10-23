"""
gs_template/rendering.py — Rendering Helpers
==============================================

Reusable building blocks for papers that need to override get_outputs().
Extracted from the repeated boilerplate in SplatfactoModel and
Feature Splatting — use these to avoid copying 80+ lines.

Three-phase rendering pattern:
    1. prepare_camera()       → camera params, crop ids
    2. rasterize_gaussians()  → raw render, alpha, info
    3. assemble_outputs()     → final dict with rgb, depth, etc.

Usage (inside a GSTemplateModel subclass):
    def get_outputs(self, camera):
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids  # early-return empty outputs

        # Gather & fuse your custom Gaussian properties
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids)
        my_feats = self.gauss_params["my_feats"]
        if crop_ids is not None:
            my_feats = my_feats[crop_ids]
        fused = torch.cat([colors, my_feats], dim=-1)

        # Rasterize
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, fused,
            cam["viewmat"], cam["K"], cam["W"], cam["H"])

        # Assemble standard outputs + your extras
        outs = assemble_outputs(self, render, alpha, info, cam,
                                extra_channels={"feature": (3, 3 + 13)})
        return outs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from gsplat import rasterization
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Camera Preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_camera(
    model,
    camera: Cameras,
) -> Tuple[Optional[Dict], Optional[Tensor]]:
    """Set up camera parameters and handle cropping.

    Returns:
        (cam_dict, crop_ids)  on success
        (None, empty_outputs) if crop box eliminates all Gaussians

    cam_dict keys: viewmat, K, W, H, camera_scale_fac
    """
    if not isinstance(camera, Cameras):
        return None, {}

    # Camera optimization (training) vs raw poses (eval)
    if model.training:
        assert camera.shape[0] == 1, "Only one camera at a time during training"
        c2w = model.camera_optimizer.apply_to_camera(camera)
    else:
        c2w = camera.camera_to_worlds

    # Cropping (eval-only)
    crop_ids = None
    if model.crop_box is not None and not model.training:
        crop_ids = model.crop_box.within(model.means).squeeze()
        if crop_ids.sum() == 0:
            return None, model.get_empty_outputs(
                int(camera.width.item()),
                int(camera.height.item()),
                model.background_color,
            )

    # Downscale + intrinsics
    camera_scale_fac = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_scale_fac)
    viewmat = get_viewmat(c2w)
    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())
    model.last_size = (H, W)
    camera.rescale_output_resolution(camera_scale_fac)

    cam_dict = {
        "viewmat": viewmat,
        "K": K,
        "W": W,
        "H": H,
        "camera_scale_fac": camera_scale_fac,
    }
    return cam_dict, crop_ids


# ═══════════════════════════════════════════════════════════════════
# Gaussian Parameter Gathering
# ═══════════════════════════════════════════════════════════════════

def gather_base_params(
    model, crop_ids: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gather and optionally crop the 6 base Gaussian parameters.

    Returns activated colors (sigmoid of SH DC for degree-0, or
    full SH stack for higher degrees).

    Returns: (means, quats, scales, opacities, colors)
        - quats:     (N, 4) raw, NOT yet normalized
        - scales:    (N, 3) raw log-space, NOT yet exponentiated
        - opacities: (N, 1) raw logit-space, NOT yet sigmoided
        - colors:    (N, 3) sigmoid'd DC if sh_degree=0,
                     else (N, K, 3) SH coefficients
    """
    if crop_ids is not None:
        means = model.means[crop_ids]
        quats = model.quats[crop_ids]
        scales = model.scales[crop_ids]
        opacities = model.opacities[crop_ids]
        features_dc = model.features_dc[crop_ids]
        features_rest = model.features_rest[crop_ids]
    else:
        means = model.means
        quats = model.quats
        scales = model.scales
        opacities = model.opacities
        features_dc = model.features_dc
        features_rest = model.features_rest

    # Assemble color from SH bands
    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

    if model.config.sh_degree == 0:
        # No view-dependent color → sigmoid the single DC band
        colors = torch.sigmoid(colors).squeeze(1)  # (N, 1, 3) → (N, 3)

    return means, quats, scales, opacities, colors


def crop_custom_param(
    param: Tensor, crop_ids: Optional[Tensor]
) -> Tensor:
    """Crop a custom per-Gaussian parameter using the same crop_ids.

    Usage:
        my_feats = crop_custom_param(self.gauss_params["my_feats"], crop_ids)
    """
    if crop_ids is not None:
        return param[crop_ids]
    return param


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Rasterization
# ═══════════════════════════════════════════════════════════════════

def rasterize_gaussians(
    model,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmat: Tensor,
    K: Tensor,
    W: int,
    H: int,
    sh_degree: Optional[int] = None,
    render_mode: Optional[str] = None,
    **extra_kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Call gsplat rasterization with splatfacto-compatible defaults.

    Automatically determines render_mode and sh_degree if not given.
    Activates quats/scales/opacities from their unconstrained storage.

    Args:
        model:     the GSTemplateModel instance (for config access)
        means:     (N, 3) Gaussian centers
        quats:     (N, 4) raw quaternions (will be normalized)
        scales:    (N, 3) log-space scales (will be exponentiated)
        opacities: (N, 1) logit opacities (will be sigmoided)
        colors:    (N, C) fused render properties (rgb + features etc.)
        viewmat:   (1, 4, 4) world-to-camera
        K:         (1, 3, 3) intrinsics
        W, H:      image dimensions
        sh_degree: override SH degree (None = auto from colors shape)
        render_mode: override render mode (None = auto)

    Returns: (render, alpha, info)
    """
    # Determine render mode
    if render_mode is None:
        if model.config.output_depth_during_training or not model.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

    # Determine SH degree
    if sh_degree is None:
        if colors.dim() == 2:
            # Pre-computed colors (e.g. sigmoid'd DC), no SH
            sh_degree = None
        else:
            # SH coefficients: progressive training
            sh_degree = min(
                model.step // model.config.sh_degree_interval,
                model.config.sh_degree,
            )

    # Merge defaults with any user overrides
    kwargs = dict(
        means=means,
        quats=quats / quats.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode=render_mode,
        sh_degree=sh_degree,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode=model.config.rasterize_mode,
    )
    kwargs.update(extra_kwargs)

    render, alpha, info = rasterization(**kwargs)
    return render, alpha, info


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Output Assembly
# ═══════════════════════════════════════════════════════════════════

def assemble_outputs(
    model,
    render: Tensor,
    alpha: Tensor,
    info: Dict,
    cam: Dict,
    extra_channels: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Tensor]:
    """Assemble the standard output dictionary from raw rasterization.

    Handles:
    - Background compositing
    - Depth extraction (if present)
    - means2d/radii storage for densification
    - Extra feature channel slicing

    Args:
        model:  the GSTemplateModel instance
        render: (1, H, W, C) raw rasterization output
        alpha:  (1, H, W, 1) accumulated opacity
        info:   gsplat info dict (contains means2d, radii)
        cam:    dict from prepare_camera()
        extra_channels: optional {name: (start_ch, end_ch)} for slicing
                        additional channels from the render tensor.
                        E.g. {"feature": (3, 16)} slices channels 3..15.

    Returns:
        Dict with "rgb", "depth", "accumulation", "background", and
        any extra channels.
    """
    H, W = cam["H"], cam["W"]

    # Store for densification (splatfacto expects these)
    if model.training and info["means2d"].requires_grad:
        info["means2d"].retain_grad()
    model.xys = info["means2d"]
    model.radii = info["radii"]

    # Background compositing
    background = model._get_background_color()
    rgb = render[:, ..., :3] + (1 - alpha) * background
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # Depth (last channel in RGB+ED mode)
    depth = None
    if render.shape[-1] > 3:
        # Check if depth is appended (ED = expected depth)
        last_ch = render[:, ..., -1:]
        # Heuristic: depth is present if last channel values are
        # plausibly depth (not feature values). This is only reliable
        # when render_mode="RGB+ED" was used.
        depth = torch.where(
            alpha > 0, last_ch, last_ch.detach().max()
        ).squeeze(0)

    if background.shape[0] == 3 and not model.training:
        background = background.expand(H, W, 3)

    outputs = {
        "rgb": rgb.squeeze(0),
        "depth": depth,
        "accumulation": alpha.squeeze(0),
        "background": background,
    }

    # Slice extra channels
    if extra_channels:
        for name, (start, end) in extra_channels.items():
            outputs[name] = render[:, ..., start:end].squeeze(0)

    return outputs
