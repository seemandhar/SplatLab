"""
AA-Wavelet-Entropy GS — Anti-Aliasing + Wavelet Loss + Opacity Entropy
=========================================================================
Complexity: ★★☆☆☆ (custom losses + scale clamping, no new Gaussian params)

Combines three orthogonal, proven techniques:
    1. **Mip-Splatting 3D smoothing filter:** Clamp Gaussian scales so that
       no axis is smaller than the pixel footprint. Prevents sub-pixel
       aliasing artifacts.  scale_min = pixel_size / (2 * sqrt(2 * ln(2)))
    2. **Wavelet loss (DWTGS-style):** 2-level Haar DWT decomposition with
       L1 on LL subbands.  lambda = 0.005, activated after 5K iterations.
    3. **Opacity entropy regularization:** Pushes opacities toward 0 or 1,
       reducing boundary aliasing and floaters.
       L_ent = -mean(a*log(a+eps) + (1-a)*log(1-a+eps))
       lambda = 0.001, activated after 10K iterations.

Key synergy: Anti-aliasing cleans frequency content so wavelet loss
supervises real detail rather than aliasing noise.  Binary opacity from
entropy regularization further reduces boundary aliasing.

Train:
    ns-train aa-wavelet-entropy-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import dwt_lf_loss, opacity_entropy_loss
from gs_template.rendering import (
    assemble_outputs,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AAWaveletEntropyGSModelConfig(GSTemplateModelConfig):
    """AA-Wavelet-Entropy GS: anti-aliasing + wavelet loss + opacity entropy."""

    _target: Type = field(default_factory=lambda: AAWaveletEntropyGSModel)

    # ── Anti-aliasing: Mip-Splatting 3D smoothing filter ──────────
    use_mip_filter: bool = True
    """Enable Mip-Splatting-style minimum scale clamping."""

    # ── Anti-aliased rasterization (gsplat built-in) ──────────────
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """Use gsplat's built-in anti-aliased rasterization."""

    # ── Wavelet loss (DWTGS-style) ────────────────────────────────
    wavelet_loss_weight: float = 0.005
    """Weight for wavelet LF loss (L1 on DWT LL subbands)."""

    wavelet_dwt_levels: int = 2
    """Number of Haar DWT decomposition levels."""

    wavelet_start_step: int = 5000
    """Activate wavelet loss after this many training steps."""

    # ── Opacity entropy regularization ────────────────────────────
    opacity_entropy_weight: float = 0.001
    """Weight for opacity entropy loss."""

    opacity_entropy_start_step: int = 10000
    """Activate opacity entropy after this many training steps."""

    # ── Depth rendering for evaluation ────────────────────────────
    output_depth_during_training: bool = True


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class AAWaveletEntropyGSModel(GSTemplateModel):
    """AA-Wavelet-Entropy GS model.

    Overrides:
        - get_outputs():           applies Mip-Splatting 3D smoothing filter
                                   (scale clamping) before rasterization
        - compute_custom_losses(): adds wavelet LF loss + opacity entropy

    Everything else (densification, optimizer wiring, SH, etc.) inherited.
    """

    config: AAWaveletEntropyGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        CONSOLE.print(
            f"[bold green]AA-Wavelet-Entropy GS[/bold green]: "
            f"mip_filter={self.config.use_mip_filter}, "
            f"rasterize={self.config.rasterize_mode}, "
            f"wavelet(w={self.config.wavelet_loss_weight}, "
            f"start={self.config.wavelet_start_step}), "
            f"entropy(w={self.config.opacity_entropy_weight}, "
            f"start={self.config.opacity_entropy_start_step})"
        )

    # ── Rendering with Mip-Splatting 3D smoothing filter ──────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Standard rendering with Mip-Splatting scale clamping.

        The 3D smoothing filter ensures no Gaussian axis is smaller than
        the pixel footprint, preventing sub-pixel aliasing:

            scale_min = pixel_size / (2 * sqrt(2 * ln(2)))

        where pixel_size = focal_length_in_world_units / focal_length_pixels
        approximated as distance / focal_pixels for each view.
        """
        if not self.config.use_mip_filter:
            return super().get_outputs(camera)

        # Phase 1: Camera setup
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        # Phase 2: Gather Gaussian parameters
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # Phase 3: Compute pixel footprint and clamp scales
        #   pixel_size ~= 1 / focal_length (in NDC-like units)
        #   For a camera with focal length f pixels, the angular pixel size
        #   at unit distance is ~1/f. The Mip-Splatting minimum scale is:
        #   scale_min = pixel_size / (2 * sqrt(2 * ln(2)))
        #
        #   We compute per-Gaussian distances to get view-dependent pixel size,
        #   but for efficiency use the mean focal length and average distance.
        fx = cam["K"][0, 0, 0]  # focal length in pixels
        fy = cam["K"][0, 1, 1]
        focal_avg = (fx + fy) / 2.0

        # Compute distance of each Gaussian from camera
        viewmat = cam["viewmat"]  # (1, 4, 4)
        means_cam = (
            viewmat[0, :3, :3] @ means.T + viewmat[0, :3, 3:4]
        ).T  # (N, 3)
        depths = means_cam[:, 2].clamp(min=0.01)  # (N,)

        # pixel_size per Gaussian = depth / focal_avg
        pixel_sizes = depths / focal_avg  # (N,)

        # Mip-Splatting minimum scale (in world units, log-space)
        # scale_min = pixel_size / (2 * sqrt(2 * ln(2)))
        denom = 2.0 * math.sqrt(2.0 * math.log(2.0))
        log_scale_min = torch.log(pixel_sizes / denom + 1e-8)  # (N,)
        log_scale_min = log_scale_min.unsqueeze(-1)  # (N, 1) broadcast to 3 axes

        # Clamp: ensure scales (log-space) >= log_scale_min
        scales_clamped = torch.max(scales, log_scale_min)

        # Phase 4: Rasterize with clamped scales
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales_clamped, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )
        self.info = info
        return assemble_outputs(self, render, alpha, info, cam)

    # ── Custom losses: wavelet + opacity entropy ──────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        # ── 1. Wavelet LF loss (after wavelet_start_step) ────────
        if (
            self.config.wavelet_loss_weight > 0
            and self.step >= self.config.wavelet_start_step
        ):
            pred_rgb = outputs["rgb"]  # (H, W, 3)
            gt_rgb = batch["image"].to(pred_rgb.device).float()

            # Normalize uint8 [0,255] -> [0,1] if needed
            if gt_rgb.max() > 1.0:
                gt_rgb = gt_rgb / 255.0

            # Handle potential shape mismatch (training downscale)
            if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
                gt_rgb = torch.nn.functional.interpolate(
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=(int(pred_rgb.shape[0]), int(pred_rgb.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            losses["wavelet_lf_loss"] = (
                self.config.wavelet_loss_weight
                * dwt_lf_loss(pred_rgb, gt_rgb, levels=self.config.wavelet_dwt_levels)
            )

        # ── 2. Opacity entropy (after opacity_entropy_start_step) ─
        if (
            self.config.opacity_entropy_weight > 0
            and self.step >= self.config.opacity_entropy_start_step
        ):
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        return losses
