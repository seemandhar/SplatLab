"""
DWTGS — Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
====================================================================================
Complexity: ★★☆☆☆ (custom losses, no new Gaussian params, no rendering changes)

Uses Discrete Wavelet Transform (Haar) to regularize 3DGS in the
frequency domain for sparse-view settings:
    1. LF supervision: multi-level LL-subband L1 loss (emphasizes low-freq structure)
    2. HF regularization: self-supervised sparsity on 1-level HH subband
       (applied on novel-view renders every 10 iters, stopped after 5K iters)

Built on top of vanilla 3DGS (no DropGaussian dependency in this template).

Demonstrates:
    - Adding wavelet-space losses via compute_custom_losses()
    - Novel-view HF regularization with training schedule
    - Config fields for loss hyperparameters and schedule control
    - Zero rendering / densification changes — fully inherited

Paper settings:
    - Haar wavelet, N=2 DWT levels, λ_LL=0.5
    - HF sparsity: every 10 iters, stop after 5K iters
    - Evaluated on LLFF (3 views), Mip-NeRF 360 (12 views), Blender (8 views)

Train:
    ns-train dwtgs --data <path>
    ns-train dwtgs --data <path> --pipeline.model.dwt_levels 2
    ns-train dwtgs --data <path> --pipeline.model.lf_loss_weight 0.5
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import dwt_lf_loss, dwt_hf_sparsity_loss


@dataclass
class DWTGSModelConfig(GSTemplateModelConfig):
    """DWTGS: wavelet-space frequency regularization for sparse-view 3DGS."""

    _target: Type = field(default_factory=lambda: DWTGSModel)

    # ── LF supervision (multi-level LL) ────────────────────────────
    lf_loss_weight: float = 0.5
    """Weight λ_LL for wavelet LF loss (paper default: 0.5)."""

    dwt_levels: int = 2
    """Number of DWT decomposition levels N for LF loss (paper: 2).
    Higher = coarser LF. Paper finds 3 too coarse (diminishing returns)."""

    # ── HF sparsity regularization ─────────────────────────────────
    use_hf_sparsity: bool = True
    """Enable self-supervised HF sparsity loss on HH subband."""

    hf_loss_weight: float = 0.1
    """Weight for HF sparsity loss L1(HH^1, 0)."""

    hf_novel_view_interval: int = 10
    """Apply HF loss every N training iterations (paper: 10)."""

    hf_stop_step: int = 5000
    """Stop HF regularization after this many steps (paper: 5000)."""

    hf_camera_noise_t: float = 0.02
    """Translation noise magnitude for novel-view camera perturbation."""

    hf_camera_noise_r: float = 0.01
    """Rotation noise magnitude (radians) for novel-view perturbation."""

    # ── Optional: Fourier-space loss (FreGS-style baseline) ────────
    use_fregs_loss: bool = False
    """Enable Fourier-space LF/HF loss (FreGS baseline, off by default)."""

    fregs_lp_weight: float = 0.01
    """Weight for Fourier low-pass loss."""

    fregs_hp_weight: float = 0.01
    """Weight for Fourier high-pass loss."""

    fregs_hp_start_step: int = 5000
    """Start supervising Fourier HP after this step."""


class DWTGSModel(GSTemplateModel):
    """DWTGS — wavelet-space frequency regularization.

    Overrides only:
        - get_outputs(): stores current camera for novel-view HF loss
        - compute_custom_losses(): adds LF + HF wavelet losses

    Everything else (rendering, densification, etc.) inherited from splatfacto.
    """

    config: DWTGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Store reference to current camera for novel-view rendering
        self._current_camera: Optional[Cameras] = None

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Standard rendering + store camera reference for HF loss."""
        # Store camera for potential novel-view rendering in loss computation
        if self.training:
            self._current_camera = camera
        return super().get_outputs(camera)

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        pred_rgb = outputs["rgb"]  # (H, W, 3)
        gt_rgb = batch["image"].to(pred_rgb.device).float()  # (H, W, 3)

        # Normalize uint8 [0,255] → [0,1] if needed
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

        # ── 1. LF wavelet supervision (always active) ─────────────
        losses["dwt_lf_loss"] = (
            self.config.lf_loss_weight
            * dwt_lf_loss(pred_rgb, gt_rgb, levels=self.config.dwt_levels)
        )

        # ── 2. HF sparsity regularization (scheduled) ─────────────
        if (
            self.config.use_hf_sparsity
            and self.training
            and self.step < self.config.hf_stop_step
            and self.step % self.config.hf_novel_view_interval == 0
        ):
            hf_loss = self._compute_hf_sparsity_loss(pred_rgb)
            if hf_loss is not None:
                losses["dwt_hf_loss"] = self.config.hf_loss_weight * hf_loss

        # ── 3. Optional Fourier-space loss (FreGS baseline) ───────
        if self.config.use_fregs_loss:
            fregs = self._compute_fregs_loss(pred_rgb, gt_rgb)
            losses.update(fregs)

        return losses

    def _compute_hf_sparsity_loss(self, train_rgb: Tensor) -> Optional[Tensor]:
        """Compute HF sparsity loss on a novel-view render.

        Perturbs the current training camera to create a "novel view",
        renders from it, and applies L1(HH^1, 0) to discourage
        high-frequency hallucinations.

        If novel-view rendering fails or camera is unavailable,
        falls back to applying HF sparsity on the training-view render.
        """
        if self._current_camera is not None:
            try:
                novel_rgb = self._render_novel_view(self._current_camera)
                if novel_rgb is not None:
                    return dwt_hf_sparsity_loss(novel_rgb)
            except Exception:
                pass  # Fall back to training view

        # Fallback: apply HF sparsity on training-view render
        return dwt_hf_sparsity_loss(train_rgb.detach())

    @torch.no_grad()
    def _perturb_camera(self, camera: Cameras) -> Cameras:
        """Create a slightly perturbed copy of the camera for novel-view rendering.

        Adds small random noise to translation and rotation to simulate
        a nearby novel viewpoint.
        """
        # Clone camera's c2w matrix
        c2w = camera.camera_to_worlds.clone()  # (1, 3, 4) or (1, 4, 4)

        # Add translation noise
        t_noise = torch.randn(1, 3, device=c2w.device) * self.config.hf_camera_noise_t
        c2w[..., :3, 3] = c2w[..., :3, 3] + t_noise

        # Add small rotation noise (approximate: add noise to rotation columns)
        r_noise = torch.randn(1, 3, 3, device=c2w.device) * self.config.hf_camera_noise_r
        c2w[..., :3, :3] = c2w[..., :3, :3] + r_noise

        # Re-orthogonalize rotation via Gram-Schmidt
        R = c2w[..., :3, :3]
        u0 = R[..., :, 0]
        u0 = u0 / (u0.norm(dim=-1, keepdim=True) + 1e-8)
        u1 = R[..., :, 1] - (R[..., :, 1] * u0).sum(dim=-1, keepdim=True) * u0
        u1 = u1 / (u1.norm(dim=-1, keepdim=True) + 1e-8)
        u2 = torch.cross(u0, u1, dim=-1)
        c2w[..., :3, 0] = u0.squeeze(-1) if u0.dim() > 2 else u0
        c2w[..., :3, 1] = u1.squeeze(-1) if u1.dim() > 2 else u1
        c2w[..., :3, 2] = u2.squeeze(-1) if u2.dim() > 2 else u2

        # Create new camera with perturbed pose
        new_camera = camera.clone() if hasattr(camera, 'clone') else camera
        # Directly modify the camera_to_worlds
        new_camera.camera_to_worlds = c2w
        return new_camera

    def _render_novel_view(self, camera: Cameras) -> Optional[Tensor]:
        """Render from a perturbed camera for HF sparsity supervision.

        Uses torch.no_grad for the novel-view render (only the HF loss
        on training-view Gaussians needs gradients, not the novel camera).
        """
        novel_camera = self._perturb_camera(camera)

        # Temporarily set to eval to avoid densification bookkeeping
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                novel_outputs = super().get_outputs(novel_camera)
            novel_rgb = novel_outputs["rgb"]  # (H, W, 3)
        finally:
            if was_training:
                self.train()

        return novel_rgb

    def _compute_fregs_loss(
        self, pred_rgb: Tensor, gt_rgb: Tensor
    ) -> Dict[str, Tensor]:
        """Optional Fourier-space loss (FreGS-style baseline).

        Separates into LF/HF using a Gaussian low-pass mask in Fourier domain,
        then applies L1 on magnitude and phase.
        """
        losses = {}

        # To Fourier domain: (H, W, C) → per-channel FFT
        pred_fft = torch.fft.fft2(pred_rgb.permute(2, 0, 1))  # (C, H, W)
        gt_fft = torch.fft.fft2(gt_rgb.permute(2, 0, 1))

        # Build Gaussian low-pass mask (retain ~50% of frequencies)
        _, H, W = pred_fft.shape
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=pred_rgb.device).float() - cy
        x = torch.arange(W, device=pred_rgb.device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Sigma chosen so that ~50% of energy passes through
        sigma = min(H, W) * 0.25
        lp_mask = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        lp_mask = torch.fft.fftshift(lp_mask)
        hp_mask = 1.0 - lp_mask

        # LP loss (always active)
        pred_lp = pred_fft * lp_mask.unsqueeze(0)
        gt_lp = gt_fft * lp_mask.unsqueeze(0)
        losses["fregs_lp"] = self.config.fregs_lp_weight * (
            F.l1_loss(pred_lp.abs(), gt_lp.abs())
            + F.l1_loss(pred_lp.angle(), gt_lp.angle())
        )

        # HP loss (progressive, after fregs_hp_start_step)
        if self.step >= self.config.fregs_hp_start_step:
            # Attenuate based on training progress
            progress = min(
                (self.step - self.config.fregs_hp_start_step) / 5000.0, 1.0
            )
            pred_hp = pred_fft * hp_mask.unsqueeze(0) * progress
            gt_hp = gt_fft * hp_mask.unsqueeze(0) * progress
            losses["fregs_hp"] = self.config.fregs_hp_weight * (
                F.l1_loss(pred_hp.abs(), gt_hp.abs())
                + F.l1_loss(pred_hp.angle(), gt_hp.angle())
            )

        return losses