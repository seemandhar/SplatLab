"""
CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
=================================================================
Paper:  https://jiaw-z.github.io/CoR-GS
Complexity: ★★★★★ (dual fields, custom pruning, pseudo-view rendering)

Trains TWO 3D Gaussian radiance fields simultaneously and uses their
disagreement to (a) identify inaccurate reconstructions and (b) regularize
training under sparse-view conditions.

Key contributions implemented:
    1. Co-pruning — Prune Gaussians with high point disagreement across
       the two fields (KNN-based non-matching detection with threshold τ).
    2. Pseudo-view co-regularization — Sample novel-view cameras near
       training views, render both fields, minimize L1 + D-SSIM between
       the two renders.

Training loss:
    L = L_color^1 + L_color^2 + λ_p · R_pcolor
    where L_color is standard (1-λ)L1 + λ·D-SSIM supervised loss,
    R_pcolor is co-regularization at pseudo views.

After training, field 1 (the splatfacto-managed field) is kept for inference.

Demonstrates:
    - Dual-field Gaussian management with independent densification
    - Custom periodic pruning logic (co-pruning via KNN)
    - Pseudo-view camera sampling and dual-field rendering
    - Extension points 1–9

Architecture:
    Field 1 → managed by splatfacto (full densification, rendering)
    Field 2 → nn.ParameterDict with manual gsplat rendering + basic densification

Train:
    ns-train cor-gs --data <path>
    ns-train cor-gs --data <path> --pipeline.model.co_prune_tau 5.0
    ns-train cor-gs --data <path> --pipeline.model.pseudo_view_weight 1.0

Critical hyperparameters for reproducing paper results:
    - co_prune_tau = 5.0 (distance threshold; paper ablated 3/5/10/30)
    - pseudo_view_weight = 1.0 (λ_p)
    - dssim_weight = 0.2 (λ in L1+D-SSIM formula)
    - co_prune_every_n_refine = 5 (every 5 densification cycles)
    - Initialization: fused stereo point cloud (following FSGS [51])
    - LLFF/DTU/Blender: 10k iters; Mip-NeRF360: 30k iters

Ambiguities / approximations:
    - Paper does not specify pseudo-view noise magnitude ε precisely;
      we default to 0.05 (scaled by scene extent) as reasonable.
    - Field 2 densification uses a simplified split/clone/prune that
      mirrors splatfacto's schedule but with slightly different internals.
    - The paper uses FSGS-style stereo initialization; this is a data
      preprocessing step handled externally (not in the model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch import Tensor
from torch.nn import Parameter

from gsplat import rasterization
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, get_viewmat

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import color_co_regularization_loss, dssim_loss


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CoRGSModelConfig(GSTemplateModelConfig):
    """CoR-GS: Co-Regularized Gaussian Splatting for sparse views."""

    _target: Type = field(default_factory=lambda: CoRGSModel)

    # ── Co-pruning ─────────────────────────────────────────────
    co_prune_tau: float = 5.0
    """Max distance threshold τ for KNN non-matching detection.
    Paper ablation: τ=5 performs best; τ=3 too strict, τ=30 too loose."""

    co_prune_every_n_refine: int = 5
    """Run co-pruning every N densification (refine) steps.
    Paper: 'every 5 optimization/density-control interleaves'."""

    co_prune_start_step: int = 500
    """Don't co-prune before this step (let fields diverge first)."""

    # ── Pseudo-view co-regularization ──────────────────────────
    pseudo_view_weight: float = 1.0
    """λ_p — weight for pseudo-view co-regularization loss."""

    dssim_weight: float = 0.2
    """λ in (1−λ)·L1 + λ·D-SSIM for both supervised and co-reg losses."""

    pseudo_view_noise_std: float = 0.05
    """Standard deviation of ε noise added to camera position for
    pseudo-view sampling. Scaled by scene extent at runtime."""

    pseudo_view_start_step: int = 200
    """Begin pseudo-view co-regularization after this step."""

    # ── Field 2 densification ──────────────────────────────────
    field2_densify: bool = True
    """Enable densification for field 2 (recommended for faithful reproduction)."""

    field2_grad_thresh: float = 0.0002
    """Gradient threshold for split/clone in field 2 densification."""

    field2_max_screen_size: float = 0.05
    """Max 2D screen-space size before splitting (fraction of image dim)."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════


class CoRGSModel(GSTemplateModel):
    """CoR-GS: two-field co-regularized Gaussian Splatting.

    Field 1: self.gauss_params (managed by splatfacto — full densification)
    Field 2: self.field2 (nn.ParameterDict — manual rendering + densification)

    After training, field 1 is used for inference (field 2 is discarded).
    """

    config: CoRGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────

    def populate_modules(self):
        """Initialize both Gaussian fields from the same point cloud."""
        super().populate_modules()

        # ── Field 2: clone of field 1's initial parameters ─────
        self.field2 = nn.ParameterDict(
            {
                "means": Parameter(self.means.data.clone()),
                "features_dc": Parameter(self.features_dc.data.clone()),
                "features_rest": Parameter(self.features_rest.data.clone()),
                "opacities": Parameter(self.opacities.data.clone()),
                "scales": Parameter(self.scales.data.clone()),
                "quats": Parameter(self.quats.data.clone()),
            }
        )

        # ── Bookkeeping ───────────────────────────────────────
        # Accumulated 2D gradient norms for field 2 densification
        n2 = self.field2["means"].shape[0]
        self.register_buffer(
            "_f2_grad_accum", torch.zeros(n2, device=self.device)
        )
        self.register_buffer(
            "_f2_grad_count", torch.zeros(n2, device=self.device)
        )

        # Pool of training camera c2w matrices for pseudo-view sampling
        self._train_c2ws: List[Tensor] = []
        self._max_cam_pool = 200  # cap to avoid unbounded growth

        # Densification step counter (for co-pruning scheduling)
        self._refine_count = 0

        # Temporary storage for camera info (set in get_outputs, read in losses)
        self._cur_viewmat: Optional[Tensor] = None
        self._cur_K: Optional[Tensor] = None
        self._cur_WH: Optional[Tuple[int, int]] = None
        self._cur_c2w: Optional[Tensor] = None

    # ── 2. get_outputs (render both fields) ────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render field 1 (standard splatfacto) and field 2 (manual)."""

        # ── Field 1: standard splatfacto rendering ─────────────
        outputs = super().get_outputs(camera)

        if not self.training:
            return outputs  # only field 1 needed at eval

        # ── Extract camera parameters for field 2 ─────────────
        c2w = self.camera_optimizer.apply_to_camera(camera)
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(c2w)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)  # restore

        # Store for pseudo-view rendering in compute_custom_losses
        self._cur_viewmat = viewmat.detach()
        self._cur_K = K.detach()
        self._cur_WH = (W, H)
        self._cur_c2w = c2w.detach()

        # ── Store camera pose for pseudo-view pool ─────────────
        if len(self._train_c2ws) < self._max_cam_pool:
            self._train_c2ws.append(c2w.detach().cpu())

        # ── Field 2: manual rendering ──────────────────────────
        f2_rgb = self._render_field2(viewmat, K, W, H)
        outputs["field2_rgb"] = f2_rgb

        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Field 2 supervised loss + pseudo-view co-regularization + co-pruning."""
        losses = {}

        if not self.training:
            return losses

        # ── Field 2 supervised color loss ──────────────────────
        if "field2_rgb" in outputs:
            gt_rgb = batch["image"].to(outputs["field2_rgb"].device)
            # Crop GT to match rendered size if needed
            f2_rgb = outputs["field2_rgb"]
            if gt_rgb.shape[:2] != f2_rgb.shape[:2]:
                gt_rgb = Func.interpolate(
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    size=(f2_rgb.shape[0], f2_rgb.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            losses["field2_color"] = self._supervised_color_loss(f2_rgb, gt_rgb)

        # ── Pseudo-view co-regularization ──────────────────────
        if (
            self.step >= self.config.pseudo_view_start_step
            and self._cur_c2w is not None
            and len(self._train_c2ws) >= 2
        ):
            co_reg = self._pseudo_view_co_reg()
            if co_reg is not None:
                losses["pseudo_view_co_reg"] = (
                    self.config.pseudo_view_weight * co_reg
                )

        # ── Co-pruning (periodic, not a loss but modifies params) ──
        self._maybe_co_prune()

        # ── Field 2 densification bookkeeping ──────────────────
        if self.config.field2_densify:
            self._accumulate_field2_grads()

        return losses

    # ── Supervised color loss (shared formula) ─────────────────

    def _supervised_color_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Standard 3DGS color loss: (1-λ)·L1 + λ·D-SSIM.

        Same formula as splatfacto uses for field 1.
        """
        l1 = Func.l1_loss(pred, gt)
        ds = dssim_loss(pred, gt)
        lam = self.config.dssim_weight
        return (1.0 - lam) * l1 + lam * ds

    # ── Field 2 rendering ──────────────────────────────────────

    def _render_field2(
        self,
        viewmat: Tensor,
        K: Tensor,
        W: int,
        H: int,
        render_mode: str = "RGB",
    ) -> Tensor:
        """Render field 2 using gsplat directly.

        Does NOT overwrite self.xys / self.radii (those belong to field 1).
        Instead stores _f2_xys / _f2_radii for field 2 densification.
        """
        means2 = self.field2["means"]
        quats2 = self.field2["quats"]
        scales2 = self.field2["scales"]
        opacities2 = self.field2["opacities"]
        features_dc2 = self.field2["features_dc"]
        features_rest2 = self.field2["features_rest"]

        colors2 = torch.cat(
            (features_dc2[:, None, :], features_rest2), dim=1
        )

        # Progressive SH degree (same schedule as field 1)
        sh_degree = min(
            self.step // self.config.sh_degree_interval,
            self.config.sh_degree,
        )

        render, alpha, info = rasterization(
            means=means2,
            quats=quats2 / quats2.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales2),
            opacities=torch.sigmoid(opacities2).squeeze(-1),
            colors=colors2,
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
            rasterize_mode=self.config.rasterize_mode,
        )

        # Track field 2 gradients for densification
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self._f2_xys = info["means2d"]
        self._f2_radii = info["radii"]

        # Background compositing
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        return rgb.squeeze(0)  # (H, W, 3)

    def _render_field_at_pose(
        self,
        field_params: dict,
        viewmat: Tensor,
        K: Tensor,
        W: int,
        H: int,
    ) -> Tensor:
        """Render an arbitrary field (1 or 2) at a given pose. No grad tracking."""
        if isinstance(field_params, nn.ParameterDict):
            means = field_params["means"]
            quats = field_params["quats"]
            scales = field_params["scales"]
            opacities = field_params["opacities"]
            features_dc = field_params["features_dc"]
            features_rest = field_params["features_rest"]
        else:
            # Field 1: use model attributes
            means = self.means
            quats = self.quats
            scales = self.scales
            opacities = self.opacities
            features_dc = self.features_dc
            features_rest = self.features_rest

        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        sh_degree = min(
            self.step // self.config.sh_degree_interval,
            self.config.sh_degree,
        )

        render, alpha, _ = rasterization(
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
            render_mode="RGB",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=False,  # no grad tracking for pseudo-view
            rasterize_mode=self.config.rasterize_mode,
        )

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb.squeeze(0)

    # ── Pseudo-view co-regularization ──────────────────────────

    def _pseudo_view_co_reg(self) -> Optional[Tensor]:
        """Sample a pseudo-view camera, render both fields, compute co-reg loss.

        Pseudo-view sampling (following paper + FSGS [51]):
            P' = (t + ε, q)
            t: position of a training camera
            ε: random Gaussian noise (scaled by scene extent)
            q: averaged quaternion of two nearest training cameras
        """
        device = self.means.device

        # Current training camera pose
        cur_c2w = self._cur_c2w.to(device)  # (1, 3, 4) or (1, 4, 4)
        cur_t = cur_c2w[0, :3, 3]  # (3,)

        # Gather all stored training camera positions
        all_c2ws = torch.stack(
            [c.to(device) for c in self._train_c2ws], dim=0
        )  # (N, 3, 4) or (N, 4, 4)
        all_t = all_c2ws[:, :3, 3]  # (N, 3)

        # Find two nearest training cameras (by Euclidean distance)
        dists = torch.norm(all_t - cur_t.unsqueeze(0), dim=-1)  # (N,)
        _, nearest_idx = torch.topk(dists, k=min(2, len(dists)), largest=False)

        # ── Position: perturb current camera position ──────────
        # Scale noise by approximate scene extent
        scene_extent = (all_t.max(dim=0).values - all_t.min(dim=0).values).norm()
        noise_scale = self.config.pseudo_view_noise_std * max(scene_extent.item(), 0.1)
        epsilon = torch.randn(3, device=device) * noise_scale
        pseudo_t = cur_t + epsilon

        # ── Rotation: average of two nearest cameras ───────────
        R1 = all_c2ws[nearest_idx[0], :3, :3]  # (3, 3)
        if len(nearest_idx) > 1:
            R2 = all_c2ws[nearest_idx[1], :3, :3]
            # Simple rotation averaging: Slerp via matrix interpolation
            # (For small angular differences this is reasonable)
            pseudo_R = _average_rotations(R1, R2)
        else:
            pseudo_R = R1

        # ── Construct pseudo-view c2w ──────────────────────────
        pseudo_c2w = torch.eye(4, device=device).unsqueeze(0)  # (1, 4, 4)
        pseudo_c2w[0, :3, :3] = pseudo_R
        pseudo_c2w[0, :3, 3] = pseudo_t

        pseudo_viewmat = get_viewmat(pseudo_c2w)

        # Use same intrinsics as current camera
        K = self._cur_K.to(device)
        W, H = self._cur_WH

        # ── Render both fields at pseudo-view ──────────────────
        pseudo_rgb1 = self._render_field_at_pose(
            None, pseudo_viewmat, K, W, H  # None → use field 1
        )
        pseudo_rgb2 = self._render_field_at_pose(
            self.field2, pseudo_viewmat, K, W, H
        )

        # ── Co-regularization loss ─────────────────────────────
        co_reg = color_co_regularization_loss(
            pseudo_rgb1, pseudo_rgb2, dssim_weight=self.config.dssim_weight
        )

        return co_reg

    # ── Co-pruning ─────────────────────────────────────────────

    def _maybe_co_prune(self):
        """Run co-pruning if it's the right time.

        Schedule: every co_prune_every_n_refine densification steps,
        starting after co_prune_start_step.
        """
        if self.step < self.config.co_prune_start_step:
            return

        # Check if we're at a densification boundary
        # splatfacto refines every self.config.refine_every steps
        refine_every = self.config.refine_every
        if self.step > 0 and self.step % refine_every == 0:
            self._refine_count += 1

            # Field 2 densification
            if self.config.field2_densify:
                self._densify_field2()

            # Co-prune every N densification steps
            if self._refine_count % self.config.co_prune_every_n_refine == 0:
                self._co_prune()

    @torch.no_grad()
    def _co_prune(self):
        """Co-pruning: remove non-matching Gaussians between two fields.

        For each Gaussian in field 1, find nearest neighbor in field 2.
        If distance > τ, mark as non-matching and schedule for removal.

        Paper: uses KNN with threshold τ=5.

        Implementation: For field 1, we set opacity to very negative
        values so splatfacto's built-in culling removes them. For field 2,
        we directly remove the parameters.
        """
        tau = self.config.co_prune_tau
        means1 = self.means.detach()  # (N1, 3)
        means2 = self.field2["means"].detach()  # (N2, 3)

        if means1.shape[0] == 0 or means2.shape[0] == 0:
            return

        # ── Field 1 → Field 2 distances (chunked for memory) ──
        nn_dists_1to2 = _chunked_nn_distances(means1, means2)
        non_match_1 = nn_dists_1to2 > tau

        # ── Field 2 → Field 1 distances ───────────────────────
        nn_dists_2to1 = _chunked_nn_distances(means2, means1)
        non_match_2 = nn_dists_2to1 > tau

        n_prune_1 = non_match_1.sum().item()
        n_prune_2 = non_match_2.sum().item()

        # ── Prune field 1: set opacity very low ───────────────
        # Splatfacto will cull these in its next density control step
        if n_prune_1 > 0:
            self.opacities.data[non_match_1] = -10.0  # sigmoid(-10) ≈ 0

        # ── Prune field 2: direct removal ─────────────────────
        if n_prune_2 > 0:
            keep_mask = ~non_match_2
            self._resize_field2(keep_mask)

    @torch.no_grad()
    def _resize_field2(self, keep_mask: Tensor):
        """Remove field 2 Gaussians not in keep_mask."""
        for name in list(self.field2.keys()):
            self.field2[name] = Parameter(
                self.field2[name].data[keep_mask].contiguous()
            )
        # Resize grad tracking buffers
        n_new = keep_mask.sum().item()
        device = self._f2_grad_accum.device
        self._f2_grad_accum = torch.zeros(n_new, device=device)
        self._f2_grad_count = torch.zeros(n_new, device=device)

    # ── Field 2 densification ──────────────────────────────────

    def _accumulate_field2_grads(self):
        """Accumulate 2D gradient norms for field 2 (for densification)."""
        if not hasattr(self, "_f2_xys") or self._f2_xys is None:
            return
        if self._f2_xys.grad is None:
            return

        # Gradient norm of 2D projected means
        grad_norms = self._f2_xys.grad.detach().norm(dim=-1)  # (N,)

        # Only accumulate for visible Gaussians
        visible = self._f2_radii > 0
        n2 = self.field2["means"].shape[0]

        if grad_norms.shape[0] != n2:
            # Size mismatch after pruning; reset
            self._f2_grad_accum = torch.zeros(n2, device=self.device)
            self._f2_grad_count = torch.zeros(n2, device=self.device)
            return

        if self._f2_grad_accum.shape[0] != n2:
            self._f2_grad_accum = torch.zeros(n2, device=self.device)
            self._f2_grad_count = torch.zeros(n2, device=self.device)

        self._f2_grad_accum[visible] += grad_norms[visible]
        self._f2_grad_count[visible] += 1

    @torch.no_grad()
    def _densify_field2(self):
        """Basic densification for field 2: split, clone, prune.

        Mirrors standard 3DGS densification logic:
        - Split: Gaussians with high 2D gradient AND large 3D scale
        - Clone: Gaussians with high 2D gradient AND small 3D scale
        - Prune: Gaussians with very low opacity
        """
        n2 = self.field2["means"].shape[0]
        if n2 == 0 or self._f2_grad_count.shape[0] != n2:
            self._f2_grad_accum = torch.zeros(n2, device=self.device)
            self._f2_grad_count = torch.zeros(n2, device=self.device)
            return

        # Average gradients
        avg_grads = self._f2_grad_accum / self._f2_grad_count.clamp(min=1)
        thresh = self.config.field2_grad_thresh

        # Scale analysis
        scales = torch.exp(self.field2["scales"].data)  # (N, 3)
        max_scale = scales.max(dim=-1).values  # (N,)

        # Determine scene-adaptive scale threshold
        # (use same heuristic as splatfacto: fraction of scene extent)
        scene_extent = (
            self.field2["means"].data.max(dim=0).values
            - self.field2["means"].data.min(dim=0).values
        ).norm().item()
        scale_thresh = 0.01 * max(scene_extent, 0.1)

        high_grad = avg_grads > thresh

        # ── Split: large Gaussians with high gradients ─────────
        split_mask = high_grad & (max_scale > scale_thresh)
        n_split = min(split_mask.sum().item(), n2 // 4)  # cap to avoid explosion

        if n_split > 0:
            # Take top-N by gradient (avoid excessive splitting)
            if split_mask.sum() > n_split:
                topk_vals, topk_idx = avg_grads[split_mask].topk(n_split)
                new_split = torch.zeros_like(split_mask)
                new_split[split_mask.nonzero(as_tuple=True)[0][topk_idx]] = True
                split_mask = new_split

            self._split_field2(split_mask)

        # ── Clone: small Gaussians with high gradients ─────────
        clone_mask = high_grad & (max_scale <= scale_thresh)
        n_clone = min(clone_mask.sum().item(), n2 // 4)

        if n_clone > 0:
            if clone_mask.sum() > n_clone:
                topk_vals, topk_idx = avg_grads[clone_mask].topk(n_clone)
                new_clone = torch.zeros_like(clone_mask)
                new_clone[clone_mask.nonzero(as_tuple=True)[0][topk_idx]] = True
                clone_mask = new_clone

            self._clone_field2(clone_mask)

        # ── Prune: low opacity ─────────────────────────────────
        opacities = torch.sigmoid(self.field2["opacities"].data).squeeze(-1)
        prune_mask = opacities < self.config.cull_alpha_thresh

        # Also prune very large Gaussians
        if self.step > self.config.refine_every * 5:
            big_mask = max_scale > 0.1 * scene_extent
            prune_mask = prune_mask | big_mask

        if prune_mask.any():
            keep_mask = ~prune_mask
            self._resize_field2(keep_mask)

        # Reset accumulators
        n_new = self.field2["means"].shape[0]
        self._f2_grad_accum = torch.zeros(n_new, device=self.device)
        self._f2_grad_count = torch.zeros(n_new, device=self.device)

    @torch.no_grad()
    def _split_field2(self, split_mask: Tensor):
        """Split field 2 Gaussians: replace each with 2 smaller copies."""
        if not split_mask.any():
            return

        means = self.field2["means"].data
        scales = self.field2["scales"].data

        # Compute split offsets (along largest scale axis, halved)
        exp_scales = torch.exp(scales[split_mask])
        max_axis = exp_scales.argmax(dim=-1)

        # Sample two offsets from the Gaussian distribution
        stds = exp_scales  # (M, 3)
        samples = torch.randn(2, stds.shape[0], 3, device=self.device) * stds

        new_params = {}
        for name in self.field2.keys():
            old = self.field2[name].data
            selected = old[split_mask]
            if name == "means":
                # Two children offset from parent
                child1 = selected + samples[0]
                child2 = selected - samples[1]
                new_params[name] = torch.cat(
                    [old[~split_mask], child1, child2], dim=0
                )
            elif name == "scales":
                # Halve the scale (in log space: subtract ln(1.6))
                reduced = selected - math.log(1.6)
                new_params[name] = torch.cat(
                    [old[~split_mask], reduced, reduced], dim=0
                )
            else:
                # Duplicate other parameters
                new_params[name] = torch.cat(
                    [old[~split_mask], selected, selected], dim=0
                )

        for name in self.field2.keys():
            self.field2[name] = Parameter(new_params[name].contiguous())

    @torch.no_grad()
    def _clone_field2(self, clone_mask: Tensor):
        """Clone field 2 Gaussians: append duplicates."""
        if not clone_mask.any():
            return

        for name in list(self.field2.keys()):
            old = self.field2[name].data
            cloned = old[clone_mask]
            self.field2[name] = Parameter(
                torch.cat([old, cloned], dim=0).contiguous()
            )

    # ── 5. get_gaussian_param_groups ───────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register field 1 Gaussian params (standard)."""
        return super().get_gaussian_param_groups()

    # ── 6. get_param_groups ────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register both fields' parameters for optimization."""
        groups = super().get_param_groups()

        # Field 2 parameters — each gets its own optimizer group
        for name in self.field2.keys():
            groups[f"field2_{name}"] = [self.field2[name]]

        return groups

    # ── 7. get_outputs_for_camera (eval / viewer) ──────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        """Standard eval using field 1 only (field 2 is discarded)."""
        return super().get_outputs_for_camera(camera, obb_box)

    # ── 8. load_state_dict ─────────────────────────────────────

    def load_state_dict(self, state_dict, **kwargs):
        """Handle checkpoint loading with dual-field parameters."""
        self.step = 30000

        # Resize field 2 params to match checkpoint
        if "field2.means" in state_dict:
            n2 = state_dict["field2.means"].shape[0]
            for name in self.field2.keys():
                old_shape = self.field2[name].shape
                new_shape = (n2,) + old_shape[1:]
                self.field2[name] = Parameter(
                    torch.zeros(new_shape, device=self.device)
                )

        # Resize field 1 (inherited)
        if "gauss_params.means" in state_dict:
            n1 = state_dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                new_shape = (n1,) + param.shape[1:]
                self.gauss_params[name] = Parameter(
                    torch.zeros(new_shape, device=self.device)
                )

        super().load_state_dict(state_dict, **kwargs)

    # ── 9. get_metrics_dict ────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add field counts and disagreement metrics."""
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["field1_num_gaussians"] = torch.tensor(
            self.means.shape[0], dtype=torch.float32
        )
        metrics["field2_num_gaussians"] = torch.tensor(
            self.field2["means"].shape[0], dtype=torch.float32
        )
        return metrics


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def _chunked_nn_distances(
    pts1: Tensor, pts2: Tensor, chunk_size: int = 4096
) -> Tensor:
    """Compute nearest-neighbor distances from pts1 to pts2.

    Memory-efficient chunked computation to avoid OOM on large point clouds.

    Args:
        pts1: (N1, 3) query points
        pts2: (N2, 3) reference points
        chunk_size: process this many queries at once

    Returns:
        (N1,) tensor of NN distances
    """
    dists_list = []
    for i in range(0, pts1.shape[0], chunk_size):
        chunk = pts1[i : i + chunk_size]  # (C, 3)
        # Pairwise distances: (C, N2)
        d = torch.cdist(chunk, pts2, p=2)
        min_d = d.min(dim=1).values  # (C,)
        dists_list.append(min_d)
    return torch.cat(dists_list, dim=0)


def _average_rotations(R1: Tensor, R2: Tensor, t: float = 0.5) -> Tensor:
    """Approximate rotation averaging via geodesic interpolation.

    For small angular differences (typical in adjacent training views),
    a simple weighted average + re-orthogonalization is sufficient.

    Args:
        R1: (3, 3) rotation matrix
        R2: (3, 3) rotation matrix
        t:  interpolation parameter (0.5 = midpoint)

    Returns:
        (3, 3) averaged rotation matrix
    """
    # Weighted matrix average
    R_avg = (1 - t) * R1 + t * R2

    # Re-orthogonalize via SVD (closest rotation matrix)
    U, _, Vt = torch.linalg.svd(R_avg)
    R_ortho = U @ Vt

    # Ensure proper rotation (det = +1)
    if torch.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt

    return R_ortho
