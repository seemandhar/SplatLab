"""
Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
===================================================================
Paper: https://arxiv.org/abs/2311.14654
Complexity: ★★★★★ (custom representation, MLPs, rendering, densification)

Replaces the standard per-Gaussian representation with a structured
anchor-based approach:
    - Anchor points are initialized from voxelized SfM points
    - Each anchor stores a learnable feature, scale factor, and k offsets
    - At render time, visible anchors spawn k neural Gaussians whose
      attributes (opacity, color, rotation, scale) are decoded by MLPs
      conditioned on view-dependent features
    - Anchor growing/pruning replaces standard split/clone densification

Key architecture differences from vanilla 3DGS:
    - Anchors (not Gaussians) are the stored representation
    - MLP-decoded attributes are view-dependent
    - Multi-resolution feature bank provides level-of-detail
    - Volume regularization replaces scale regularization

Train:
    ns-train scaffold-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from gsplat import rasterization
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# MLP building blocks (paper: 2-layer, ReLU, hidden=32)
# ═══════════════════════════════════════════════════════════════════

class MLP2Layer(nn.Module):
    """2-layer MLP with ReLU, as specified in the paper."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldGSModelConfig(GSTemplateModelConfig):
    """Scaffold-GS configuration."""

    _target: Type = field(default_factory=lambda: ScaffoldGSModel)

    # ── Anchor / neural Gaussian params ───────────────────────────
    feat_dim: int = 32
    """Per-anchor feature dimension."""

    n_offsets: int = 10
    """Number of neural Gaussians spawned per anchor (k in paper)."""

    voxel_size: float = 0.0
    """Voxel size for anchor initialization. 0 = auto (median NN distance)."""

    # ── MLP architecture ──────────────────────────────────────────
    mlp_hidden_dim: int = 32
    """Hidden dimension for all attribute MLPs (paper uses 32)."""

    # ── Feature bank ──────────────────────────────────────────────
    n_feature_levels: int = 3
    """Number of multi-resolution feature bank levels."""

    # ── Loss weights ──────────────────────────────────────────────
    lambda_vol: float = 0.001
    """Weight for volume regularization loss."""

    # ── Anchor refinement ─────────────────────────────────────────
    refine_interval: int = 100
    """Gradient averaging window N for anchor refinement."""

    refine_start_iter: int = 500
    """Start anchor refinement after this many iterations."""

    refine_stop_iter: int = 25000
    """Stop anchor refinement after this many iterations."""

    grow_grad_thresh: float = 0.0002
    """Base gradient threshold tau_g for anchor growing."""

    grow_n_levels: int = 3
    """Number of multi-resolution growing levels m."""

    prune_opacity_thresh: float = 0.5
    """Accumulated opacity threshold for anchor pruning."""

    opacity_filter_thresh: float = 0.005
    """Opacity threshold tau_alpha for filtering neural Gaussians."""

    # ── Override base config defaults ─────────────────────────────
    sh_degree: int = 0
    """Scaffold-GS uses direct color prediction via MLPs, not SH."""

    output_depth_during_training: bool = True


# ═══════════════════════════════════════════════════════════════════
# Voxelization utility
# ═══════════════════════════════════════════════════════════════════

def _voxelize_points(points: Tensor, voxel_size: float = 0.0) -> Tuple[Tensor, float]:
    """Voxelize a point cloud to create anchor positions.

    Implements: V = {round(P / epsilon)} * epsilon  (deduplicated)

    Args:
        points: (M, 3) SfM point cloud
        voxel_size: voxel size epsilon. 0 = auto (median NN distance).

    Returns:
        (anchor_positions, voxel_size)
    """
    if voxel_size <= 0:
        # Auto: estimate from median nearest-neighbor distance
        n = min(points.shape[0], 5000)
        idx = torch.randperm(points.shape[0], device=points.device)[:n]
        sub = points[idx]
        dists = torch.cdist(sub, sub)
        dists.fill_diagonal_(float("inf"))
        nn_dists = dists.min(dim=1).values
        voxel_size = float(nn_dists.median().item())
        voxel_size = max(voxel_size, 0.001)  # safety floor

    # Quantize to voxel grid and deduplicate
    voxel_indices = torch.round(points / voxel_size).long()
    unique_indices = torch.unique(voxel_indices, dim=0)
    anchor_positions = unique_indices.float() * voxel_size

    return anchor_positions.to(points.device), voxel_size


# ═══════════════════════════════════════════════════════════════════
# Spatial hashing for fast anchor lookup
# ═══════════════════════════════════════════════════════════════════

def _hash_coords(coords: Tensor) -> Tensor:
    """Simple spatial hash for 3D integer coordinates."""
    p1, p2, p3 = 73856093, 19349663, 83492791
    return coords[:, 0] * p1 ^ coords[:, 1] * p2 ^ coords[:, 2] * p3


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class ScaffoldGSModel(GSTemplateModel):
    """Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering.

    Key differences from vanilla 3DGS:
    1. Anchors replace individual Gaussians as the stored representation
    2. Neural Gaussians are spawned on-the-fly from visible anchors
    3. All Gaussian attributes are predicted by view-conditioned MLPs
    4. Anchor growing/pruning replaces standard densification
    """

    config: ScaffoldGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Initialize anchor representation and MLPs.

        Calls super() to get camera optimizer and SfM seed points,
        then replaces the Gaussian representation with anchors.
        """
        super().populate_modules()

        k = self.config.n_offsets
        feat_dim = self.config.feat_dim

        # Voxelize initial SfM points to create anchors
        init_points = self.means.data.clone()
        anchor_pos, voxel_size = _voxelize_points(
            init_points, self.config.voxel_size
        )
        self._voxel_size = voxel_size
        n_anchors = anchor_pos.shape[0]

        # ── Remove base gauss_params from computation graph ───────
        # SplatfactoModel.populate_modules() creates means, features_dc,
        # features_rest, opacities, scales, quats as gauss_params.
        # If any inherited method (get_loss_dict, etc.) references these,
        # they'd enter the computation graph with stale sizes → crash
        # after our anchor growing/pruning changes param counts.
        # Freeze them with requires_grad=False to prevent this.
        for name in list(self.gauss_params.keys()):
            self.gauss_params[name].requires_grad_(False)

        # ── Anchor parameters ─────────────────────────────────────
        self.anchor_positions = Parameter(anchor_pos)
        self.anchor_features = Parameter(
            torch.randn(n_anchors, feat_dim) * 0.01
        )
        self.anchor_offsets = Parameter(
            torch.zeros(n_anchors, k, 3)
        )
        self.anchor_scales = Parameter(
            torch.log(torch.ones(n_anchors, 3) * voxel_size)
        )

        # ── Feature bank weight MLP ───────────────────────────────
        # Input: (distance, direction) = (1 + 3) = 4
        self.mlp_feature_bank = MLP2Layer(
            in_dim=4,
            hidden_dim=self.config.mlp_hidden_dim,
            out_dim=self.config.n_feature_levels,
        )

        # ── Attribute MLPs ────────────────────────────────────────
        # Input: (mixed_feature + distance + direction) = (feat_dim + 4)
        mlp_in = feat_dim + 4

        self.mlp_opacity = MLP2Layer(mlp_in, self.config.mlp_hidden_dim, k)
        self.mlp_color = MLP2Layer(mlp_in, self.config.mlp_hidden_dim, k * 3)
        self.mlp_quat = MLP2Layer(mlp_in, self.config.mlp_hidden_dim, k * 4)
        self.mlp_scale = MLP2Layer(mlp_in, self.config.mlp_hidden_dim, k * 3)

        # ── Tracking buffers for anchor refinement ────────────────
        self._init_refinement_buffers(n_anchors)

        # Runtime state (set during forward pass)
        self._ng_anchor_ids: Optional[Tensor] = None
        self._ng_opacities: Optional[Tensor] = None
        self._optimizers = None  # set in get_training_callbacks

    def _init_refinement_buffers(self, n: int):
        """Initialize gradient/opacity accumulators for n anchors."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(self, "anchor_positions"):
            device = self.anchor_positions.device

        # Use register_buffer on first call, then just resize in-place
        if not hasattr(self, "anchor_grad_accum") or self.anchor_grad_accum is None:
            self.register_buffer("anchor_grad_accum", torch.zeros(n, device=device))
            self.register_buffer("anchor_grad_count", torch.zeros(n, dtype=torch.long, device=device))
            self.register_buffer("anchor_opacity_accum", torch.zeros(n, device=device))
            self.register_buffer("anchor_opacity_count", torch.zeros(n, dtype=torch.long, device=device))
        else:
            self.anchor_grad_accum = torch.zeros(n, device=device)
            self.anchor_grad_count = torch.zeros(n, dtype=torch.long, device=device)
            self.anchor_opacity_accum = torch.zeros(n, device=device)
            self.anchor_opacity_count = torch.zeros(n, dtype=torch.long, device=device)

    # ── CRITICAL: Disable splatfacto's built-in densification ───────
    #
    # SplatfactoModel registers a step_post_backward callback that runs
    # gsplat's DefaultStrategy. That strategy maintains state buffers
    # sized for self.means (the base SfM points) and indexes into them
    # using self.xys/self.radii. But our xys/radii come from neural
    # Gaussians (potentially 10x more) → out-of-bounds CUDA crash.
    #
    # Scaffold-GS uses its own anchor growing/pruning (after_train),
    # so we disable the inherited densification entirely.

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Return only our anchor refinement callback, NOT splatfacto's.

        Also captures a reference to the optimizers so we can update
        Adam state when anchors are added/removed.
        """
        # Save reference — needed for optimizer state updates during
        # anchor growing/pruning
        self._optimizers = training_callback_attributes.optimizers

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self._training_callback,
            ),
        ]

    def _training_callback(self, step: int, **kwargs):
        """Wrapper for after_train called by nerfstudio's callback system."""
        self.after_train(step)

    def step_post_backward(self, *args, **kwargs):
        """No-op — Scaffold-GS does not use gsplat's densification strategy."""
        pass

    # ── 2. get_outputs (core rendering pipeline) ───────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Spawn neural Gaussians from visible anchors and rasterize.

        Pipeline (from paper):
        1. Compute view direction and distance for each anchor
        2. Feature bank mixing via F_w MLP
        3. Decode neural Gaussian attributes via attribute MLPs
        4. Filter by opacity threshold tau_alpha
        5. Rasterize with gsplat
        """
        if not isinstance(camera, Cameras):
            return {}

        # ── Camera setup ──────────────────────────────────────────
        if self.training:
            assert camera.shape[0] == 1
            c2w = self.camera_optimizer.apply_to_camera(camera)
        else:
            c2w = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(c2w)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)

        cam_pos = c2w[0, :3, 3]  # camera position in world space

        # ── Step 1: View geometry ─────────────────────────────────
        anchor_pos = self.anchor_positions  # (Na, 3)
        delta_vec = anchor_pos - cam_pos.unsqueeze(0)  # (Na, 3)
        distance = delta_vec.norm(dim=-1, keepdim=True)  # (Na, 1)
        direction = delta_vec / (distance + 1e-8)  # (Na, 3)

        # ── Frustum culling (Filter 1) ────────────────────────────
        with torch.no_grad():
            pts_h = torch.cat(
                [anchor_pos, torch.ones_like(anchor_pos[:, :1])], dim=-1
            )
            pts_cam = (viewmat[0] @ pts_h.T).T
            visible = pts_cam[:, 2] > 0.01

            if visible.sum() == 0:
                return self._get_empty_outputs(W, H)

        vis_idx = visible.nonzero(as_tuple=True)[0]
        n_vis = vis_idx.shape[0]
        k = self.config.n_offsets
        feat_dim = self.config.feat_dim

        vis_dist = distance[vis_idx]
        vis_dir = direction[vis_idx]
        vis_feats = self.anchor_features[vis_idx]
        vis_offsets = self.anchor_offsets[vis_idx]
        vis_scales_log = self.anchor_scales[vis_idx]
        vis_pos = anchor_pos[vis_idx]

        # ── Step 2: Multi-resolution feature bank mixing ──────────
        bank_input = torch.cat([vis_dist, vis_dir], dim=-1)  # (Nv, 4)
        bank_weights = torch.softmax(
            self.mlp_feature_bank(bank_input), dim=-1
        )  # (Nv, n_levels)

        mixed_feat = torch.zeros_like(vis_feats)  # (Nv, feat_dim)
        for level in range(self.config.n_feature_levels):
            if level == 0:
                feat_level = vis_feats
            else:
                # Downsample by averaging groups of 2^level consecutive features
                ds = 2 ** level
                trunc_dim = (feat_dim // ds) * ds
                if trunc_dim == 0:
                    feat_level = vis_feats
                else:
                    pooled = vis_feats[:, :trunc_dim].reshape(
                        n_vis, -1, ds
                    ).mean(dim=-1)
                    feat_level = pooled.repeat(1, ds)[:, :feat_dim]

            mixed_feat = mixed_feat + bank_weights[:, level:level + 1] * feat_level

        # ── Step 3: Decode neural Gaussian attributes ─────────────
        mlp_input = torch.cat(
            [mixed_feat, vis_dist, vis_dir], dim=-1
        )  # (Nv, feat_dim + 4)

        # Opacity: Tanh → remap to [0, 1]
        ng_opacity = (torch.tanh(self.mlp_opacity(mlp_input)) + 1.0) / 2.0  # (Nv, k)

        # Color: Sigmoid → (0, 1)
        ng_color = torch.sigmoid(
            self.mlp_color(mlp_input).reshape(n_vis, k, 3)
        )

        # Quaternion: normalize
        raw_quat = self.mlp_quat(mlp_input).reshape(n_vis, k, 4)
        ng_quat = raw_quat / (raw_quat.norm(dim=-1, keepdim=True) + 1e-8)

        # Scale: Sigmoid * anchor_scale (exponentiated from log-space)
        anchor_scale_exp = torch.exp(vis_scales_log)  # (Nv, 3)
        ng_scale = (
            torch.sigmoid(self.mlp_scale(mlp_input).reshape(n_vis, k, 3))
            * anchor_scale_exp.unsqueeze(1)
        )

        # Position: anchor + offset * anchor_scale
        ng_pos = vis_pos.unsqueeze(1) + vis_offsets * anchor_scale_exp.unsqueeze(1)

        # ── Step 4: Flatten & opacity filter (Filter 2) ───────────
        ng_pos_f = ng_pos.reshape(-1, 3)
        ng_opacity_f = ng_opacity.reshape(-1)
        ng_color_f = ng_color.reshape(-1, 3)
        ng_quat_f = ng_quat.reshape(-1, 4)
        ng_scale_f = ng_scale.reshape(-1, 3)
        ng_anchor_map = vis_idx.unsqueeze(1).expand(-1, k).reshape(-1)

        mask = ng_opacity_f > self.config.opacity_filter_thresh
        if mask.sum() == 0:
            return self._get_empty_outputs(W, H)

        ng_pos_f = ng_pos_f[mask]
        ng_opacity_f = ng_opacity_f[mask]
        ng_color_f = ng_color_f[mask]
        ng_quat_f = ng_quat_f[mask]
        ng_scale_f = ng_scale_f[mask]
        ng_anchor_map = ng_anchor_map[mask]

        # Save for after_train refinement
        self._ng_anchor_ids = ng_anchor_map
        self._ng_opacities = ng_opacity_f.detach()

        # ── Step 5: Rasterize ─────────────────────────────────────
        render_mode = "RGB+ED" if (
            self.config.output_depth_during_training or not self.training
        ) else "RGB"

        render, alpha, info = rasterization(
            means=ng_pos_f,
            quats=ng_quat_f,
            scales=ng_scale_f,
            opacities=ng_opacity_f,
            colors=ng_color_f,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )

        # Store for gradient-based densification tracking
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]
        self.radii = info["radii"]

        # ── Assemble output dict ──────────────────────────────────
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        depth = None
        if render.shape[-1] > 3:
            depth = torch.where(
                alpha > 0,
                render[:, ..., -1:],
                render[:, ..., -1:].detach().max(),
            ).squeeze(0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        outputs = {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

        if self.training:
            outputs["_ng_scales"] = ng_scale_f  # for volume reg loss

        return outputs

    def _get_empty_outputs(self, W: int, H: int) -> Dict[str, Tensor]:
        bg = self._get_background_color()
        return {
            "rgb": bg.expand(H, W, 3),
            "depth": torch.zeros(H, W, 1, device=self.device),
            "accumulation": torch.zeros(H, W, 1, device=self.device),
            "background": bg.expand(H, W, 3),
        }

    # ── 3. Losses ──────────────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Compute L1 + SSIM + volume reg directly.

        CRITICAL: We do NOT call super().get_loss_dict() because
        SplatfactoModel's version may reference base gauss_params
        (self.scales for scale_reg, etc.) which are stale 44975-sized
        tensors. If they enter the computation graph, backward()
        crashes after anchor pruning changes our param sizes.

        Loss: L = L1 + 0.2 * L_SSIM + lambda_vol * L_vol
        (matches paper eq.)
        """
        gt_img = batch["image"].to(self.device)  # (H, W, 3)
        pred_img = outputs["rgb"]  # (H, W, 3)

        # L1 loss
        Ll1 = torch.abs(gt_img - pred_img).mean()

        # SSIM loss
        ssim_lambda = 0.2  # paper default
        gt_permuted = gt_img.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
        pred_permuted = pred_img.permute(2, 0, 1).unsqueeze(0)
        if hasattr(self, "ssim"):
            # Use parent's SSIM module (torchmetrics)
            Lssim = 1 - self.ssim(pred_permuted, gt_permuted)
        else:
            # Fallback: just use L1
            Lssim = Ll1

        loss_dict = {
            "main_loss": (1 - ssim_lambda) * Ll1 + ssim_lambda * Lssim,
        }

        # Volume regularization
        if "_ng_scales" in outputs:
            scales = outputs["_ng_scales"]
            loss_dict["vol_reg"] = self.config.lambda_vol * scales.prod(dim=-1).mean()

        return loss_dict

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Not used — losses computed in get_loss_dict directly."""
        return {}

    # ── 4. Anchor refinement (replaces split/clone) ────────────────

    def after_train(self, step: int):
        """Accumulate stats and periodically grow/prune anchors.

        Called by nerfstudio pipeline after each training step.
        """
        # ── Accumulate gradient stats from means2d ────────────────
        if (
            self._ng_anchor_ids is not None
            and self.xys is not None
            and self.xys.grad is not None
        ):
            ng_grads = self.xys.grad.detach()
            # gsplat may return (1, N, 2) with batch dim — squeeze it
            if ng_grads.dim() == 3:
                ng_grads = ng_grads.squeeze(0)
            ng_grads = ng_grads.norm(dim=-1)  # (N,)
            aids = self._ng_anchor_ids
            n_a = self.anchor_grad_accum.shape[0]
            valid = aids < n_a
            if valid.sum() > 0:
                self.anchor_grad_accum.scatter_add_(
                    0, aids[valid], ng_grads[valid]
                )
                self.anchor_grad_count.scatter_add_(
                    0, aids[valid],
                    torch.ones_like(aids[valid], dtype=torch.long),
                )

        # ── Accumulate opacity stats ──────────────────────────────
        if self._ng_anchor_ids is not None and self._ng_opacities is not None:
            aids = self._ng_anchor_ids
            n_a = self.anchor_opacity_accum.shape[0]
            valid = aids < n_a
            if valid.sum() > 0:
                self.anchor_opacity_accum.scatter_add_(
                    0, aids[valid], self._ng_opacities[valid]
                )
                self.anchor_opacity_count.scatter_add_(
                    0, aids[valid],
                    torch.ones_like(aids[valid], dtype=torch.long),
                )

        # ── Periodic refinement ───────────────────────────────────
        if (
            step >= self.config.refine_start_iter
            and step < self.config.refine_stop_iter
            and step % self.config.refine_interval == 0
            and step > 0
        ):
            # Compute decisions from current (consistent) accum buffers,
            # then apply. Order matters: prune first to shrink, then grow.
            #
            # 1. Compute prune mask (based on old-size accum buffers)
            prune_keep_mask = self._compute_prune_mask()
            # 2. Compute grow positions (based on old-size accum buffers + old params)
            grow_positions = self._compute_grow_positions()
            # 3. Apply prune (accum buffers match param size here)
            if prune_keep_mask is not None:
                self._remove_anchors_with_mask(prune_keep_mask)
            # 4. Apply grow (new positions are independent of current params)
            if grow_positions is not None and grow_positions.shape[0] > 0:
                self._add_anchors(grow_positions)
            # 5. Reset accumulators to match final param size
            self._init_refinement_buffers(self.anchor_positions.shape[0])

    def _compute_grow_positions(self) -> Optional[Tensor]:
        """Compute new anchor positions from gradient stats.

        Returns candidate positions (or None). Does NOT modify params.
        """
        count = self.anchor_grad_count.float().clamp(min=1)
        avg_grad = self.anchor_grad_accum / count
        eps = self._voxel_size
        all_new = []

        for level in range(self.config.grow_n_levels):
            eps_g = eps / (4.0 ** level)
            tau_g = self.config.grow_grad_thresh * (2.0 ** level)

            grow_mask = avg_grad > tau_g
            if grow_mask.sum() == 0:
                continue

            grow_pos = self.anchor_positions.data[grow_mask]

            # 6-connected neighbor candidates
            offsets = torch.tensor(
                [[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                 [0, 1, 0], [0, 0, -1], [0, 0, 1]],
                device=self.device, dtype=torch.float32,
            ) * eps_g

            candidates = (
                grow_pos.unsqueeze(1) + offsets.unsqueeze(0)
            ).reshape(-1, 3)

            # Deduplicate candidates
            cand_q = torch.round(candidates / eps_g).long()
            unique_q, inv = torch.unique(cand_q, dim=0, return_inverse=True)
            unique_pos = torch.zeros(
                unique_q.shape[0], 3, device=self.device
            )
            unique_pos.scatter_(0, inv.unsqueeze(1).expand(-1, 3), candidates)

            # Exclude positions that already have anchors
            existing_q = torch.round(
                self.anchor_positions.data / eps_g
            ).long()
            existing_hashes = set(_hash_coords(existing_q).cpu().tolist())
            cand_hashes = _hash_coords(unique_q)
            is_new = torch.tensor(
                [h not in existing_hashes for h in cand_hashes.cpu().tolist()],
                dtype=torch.bool, device=self.device,
            )

            if is_new.sum() == 0:
                continue

            new_pos = unique_pos[is_new]

            # Random elimination (~50%) to curb explosive growth
            keep = torch.rand(new_pos.shape[0], device=self.device) > 0.5
            new_pos = new_pos[keep]

            if new_pos.shape[0] > 0:
                all_new.append(new_pos)

        if all_new:
            return torch.cat(all_new, dim=0)
        return None

    def _compute_prune_mask(self) -> Optional[Tensor]:
        """Compute boolean keep-mask from opacity stats.

        Returns keep_mask (or None if no pruning needed). Does NOT modify params.
        """
        count = self.anchor_opacity_count.float().clamp(min=1)
        avg_op = self.anchor_opacity_accum / count
        prune = avg_op < self.config.prune_opacity_thresh
        n = prune.shape[0]
        if prune.sum() == 0 or prune.sum() >= n - 10:
            return None
        return ~prune

    def _add_anchors(self, new_positions: Tensor):
        """Grow anchors: extend param .data in-place + extend optimizer state."""
        n = new_positions.shape[0]
        k, fd = self.config.n_offsets, self.config.feat_dim
        dev = self.device

        # Build the new data to append for each param
        extensions = {
            "anchor_positions": new_positions,
            "anchor_features": torch.randn(n, fd, device=dev) * 0.01,
            "anchor_offsets": torch.zeros(n, k, 3, device=dev),
            "anchor_scales": torch.log(torch.ones(n, 3, device=dev) * self._voxel_size),
        }

        for name, new_data in extensions.items():
            param = getattr(self, name)
            new_val = torch.cat([param.data, new_data], dim=0)
            self._replace_param_data(name, param, new_val)

    def _remove_anchors(self, keep: Tensor):
        """Prune anchors: shrink param .data in-place + shrink optimizer state."""
        for name in ["anchor_positions", "anchor_features",
                     "anchor_offsets", "anchor_scales"]:
            param = getattr(self, name)
            new_val = param.data[keep]
            self._replace_param_data(name, param, new_val)

    def _replace_param_data(self, name: str, param: Parameter, new_data: Tensor):
        """Replace a parameter's data in-place AND update Adam optimizer state.

        This is the same pattern gsplat's DefaultStrategy uses for
        split/clone — the optimizer tracks the Parameter *object*,
        so we must keep the same object and only swap .data.

        For Adam, we must also resize exp_avg and exp_avg_sq to match
        the new parameter shape (zero-pad for new entries, slice for pruned).
        """
        old_n = param.data.shape[0]
        new_n = new_data.shape[0]

        # Update optimizer state BEFORE changing .data
        if self._optimizers is not None and hasattr(self._optimizers, 'optimizers'):
            opt = self._optimizers.optimizers.get(name)
            if opt is not None:
                for group in opt.param_groups:
                    for p in group['params']:
                        if p.data_ptr() == param.data_ptr() or p is param:
                            state = opt.state.get(p, {})
                            for key in ['exp_avg', 'exp_avg_sq']:
                                if key in state:
                                    old_state = state[key]
                                    if new_n > old_n:
                                        # Growing: zero-pad
                                        pad_shape = (new_n - old_n,) + old_state.shape[1:]
                                        pad = torch.zeros(
                                            pad_shape,
                                            dtype=old_state.dtype,
                                            device=old_state.device,
                                        )
                                        state[key] = torch.cat([old_state, pad], dim=0)
                                    elif new_n < old_n:
                                        # Pruning: must use same mask/indexing
                                        # new_data was already sliced, so
                                        # old_state needs same treatment.
                                        # We detect whether this is a keep-mask
                                        # or direct assignment case.
                                        state[key] = old_state[:new_n]
                                    # else same size — no change needed
                            break

        # Now swap .data in-place (optimizer still tracks same Parameter object)
        param.data = new_data
        # CRITICAL: Clear stale .grad to prevent shape mismatch on next backward.
        # If zero_grad(set_to_none=False) is used, old .grad retains old shape
        # and backward() tries to accumulate into wrong-sized tensor.
        param.grad = None

    def _remove_anchors_with_mask(self, keep: Tensor):
        """Prune anchors using a boolean mask, properly slicing optimizer state."""
        for name in ["anchor_positions", "anchor_features",
                     "anchor_offsets", "anchor_scales"]:
            param = getattr(self, name)
            new_val = param.data[keep]

            # Update optimizer state with same boolean mask
            if self._optimizers is not None and hasattr(self._optimizers, 'optimizers'):
                opt = self._optimizers.optimizers.get(name)
                if opt is not None:
                    for group in opt.param_groups:
                        for p in group['params']:
                            if p.data_ptr() == param.data_ptr() or p is param:
                                state = opt.state.get(p, {})
                                for key in ['exp_avg', 'exp_avg_sq']:
                                    if key in state:
                                        state[key] = state[key][keep]
                                break

            param.data = new_val
            param.grad = None  # Force fresh grad allocation

    # ── 5. Parameter groups ────────────────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Anchor params replace standard Gaussian params."""
        return {
            "anchor_positions": [self.anchor_positions],
            "anchor_features": [self.anchor_features],
            "anchor_offsets": [self.anchor_offsets],
            "anchor_scales": [self.anchor_scales],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return ALL param groups: anchors + MLPs + camera.

        Nerfstudio's Optimizers iterates over this dict and creates
        an optimizer for each key found in the trainer config.
        We must return every group that has a config entry.
        """
        # Start with anchor params (our 'Gaussian' equivalent)
        groups = self.get_gaussian_param_groups()

        # Add MLP params
        groups["mlp_feature_bank"] = list(self.mlp_feature_bank.parameters())
        groups["mlp_opacity"] = list(self.mlp_opacity.parameters())
        groups["mlp_color"] = list(self.mlp_color.parameters())
        groups["mlp_quat"] = list(self.mlp_quat.parameters())
        groups["mlp_scale"] = list(self.mlp_scale.parameters())

        # Camera optimizer
        cam_params = list(self.camera_optimizer.parameters())
        if cam_params:
            groups["camera_opt"] = cam_params

        return groups

    # ── 6. Viewer ──────────────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        return self.get_outputs(camera)

    # ── 7. Checkpoint ──────────────────────────────────────────────

    def load_state_dict(self, state_dict, **kwargs):
        self.step = 30000
        if "anchor_positions" in state_dict:
            n = state_dict["anchor_positions"].shape[0]
            k, fd = self.config.n_offsets, self.config.feat_dim
            self.anchor_positions = Parameter(torch.zeros(n, 3, device=self.device))
            self.anchor_features = Parameter(torch.zeros(n, fd, device=self.device))
            self.anchor_offsets = Parameter(torch.zeros(n, k, 3, device=self.device))
            self.anchor_scales = Parameter(torch.zeros(n, 3, device=self.device))
        super(GSTemplateModel, self).load_state_dict(state_dict, **kwargs)

    # ── 8. Metrics ─────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_anchors"] = torch.tensor(float(self.anchor_positions.shape[0]))
        return metrics

    # ── Interface compatibility ────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        """Not used — Scaffold-GS has anchor growing/pruning instead."""
        return super().split_gaussians(split_mask, samps)

    @property
    def num_points(self) -> int:
        if hasattr(self, "anchor_positions"):
            return self.anchor_positions.shape[0]
        return self.means.shape[0]