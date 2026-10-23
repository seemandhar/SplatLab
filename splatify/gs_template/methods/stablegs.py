"""
StableGS — A Floater-Free Framework for 3D Gaussian Splatting
===============================================================
Paper: Wang et al., "StableGS: A Floater-Free Framework for 3DGS"
Complexity: ★★★★★ (new params, dual rendering, custom losses, custom DataManager)

Core idea: Floaters persist because L1/MSE gradients vanish at a
pseudo-equilibrium color. Depth-based regularization can remove them,
but strong depth losses degrade appearance. Solution: DUAL OPACITY.

    α   — standard "geometric" opacity   → depth losses applied here
    α'  — auxiliary "rendering" opacity   → appearance loss applied here

Two decoupled rendering paths per training step:
    1. Geometric path  (σ_k):       renders C^s, D^s  → depth losses
    2. Appearance path  (σ'_k·σ_k): renders C^o, D^o  → color loss

Losses:
    L_color   = L_3DGS(C^o, C^gt) + λ_s · L1(C^s, C^gt)
    L_geometry = λ_consis · L_consis(D^s) + λ_prior · L_prior(D^s)
    L_total   = L_color + λ_geo · L_geometry

Demonstrates:
    - populate_modules:          auxiliary opacity α'
    - get_outputs:               dual rasterization (geometric + appearance)
    - compute_custom_losses:     depth consistency + DUSt3R prior + dual color
    - split_gaussians:           replicate α' during densification
    - get_gaussian_param_groups: register α' optimizer
    - get_outputs_for_camera:    viewer outputs from appearance path
    - load_state_dict:           handle α' in checkpoints

Train:
    ns-train stablegs --data <path>
    ns-train stablegs --data <path> --pipeline.model.lambda_geo 0.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    depth_consistency_loss,
    weighted_depth_prior_loss,
)
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
class StableGSModelConfig(GSTemplateModelConfig):
    """StableGS configuration — dual opacity + depth regularization."""

    _target: Type = field(default_factory=lambda: StableGSModel)

    # ── Must enable depth rendering during training ────────────────
    output_depth_during_training: bool = True

    # ── Dual opacity settings ──────────────────────────────────────
    lambda_geo: float = 0.1
    """Weight for total geometry loss (L_geometry)."""

    lambda_s: float = 0.2
    """Weight for geometric-path color loss L1(C^s, C^gt)."""

    # ── Depth consistency loss (self-supervised) ───────────────────
    lambda_consis: float = 0.05
    """Weight for multi-view depth consistency loss."""

    min_covisible_points: int = 30
    """Minimum COLMAP co-visible feature points for pair selection."""

    min_rotation_deg: float = 16.0
    """Minimum relative rotation angle (degrees) for pair selection."""

    max_rotation_deg: float = 60.0
    """Maximum relative rotation angle for pair selection."""

    max_homography_inlier_ratio: float = 0.8
    """Maximum homography inlier ratio (suppress near-pure-rotation)."""

    num_warp_pairs: int = 4
    """Number of random pairs to sample per step for depth consistency."""

    # ── DUSt3R depth prior loss ────────────────────────────────────
    lambda_prior: float = 0.005
    """Weight for DUSt3R depth prior loss."""

    use_dust3r_prior: bool = True
    """Whether to use DUSt3R depth priors (requires StableGSDataManager)."""

    # ── Auxiliary opacity initialization ───────────────────────────
    aux_opacity_init: float = 0.0
    """Initial logit value for auxiliary opacities (0 → sigmoid=0.5)."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class StableGSModel(GSTemplateModel):
    """StableGS — floater-free 3DGS via dual opacity.

    Key innovation: decouple geometry optimization (strong depth losses)
    from appearance refinement (standard color losses) using two
    separate opacity values per Gaussian.
    """

    config: StableGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Add auxiliary opacity α' per Gaussian."""
        super().populate_modules()

        # Auxiliary opacity in logit space (same as standard opacities)
        # Initialized to config value; sigmoid(0) = 0.5
        self.gauss_params["aux_opacities"] = Parameter(
            torch.full(
                (self.num_points, 1),
                self.config.aux_opacity_init,
                dtype=torch.float32,
            )
        )

        # Cache for training pair info (populated by datamanager)
        self._pair_cache: Optional[Dict] = None

    # ── 2. get_outputs (DUAL RENDERING) ────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Dual-path rendering: geometric + appearance.

        Geometric path:  uses standard opacity σ_k
            → produces C^s (intermediate color), D^s (geometric depth)
            → depth losses applied to D^s

        Appearance path: uses σ'_k = α'_k · σ_k  (rendering opacity)
            → produces C^o (final color), D^o (appearance depth)
            → color loss applied to C^o
        """

        # Phase 1: camera setup
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids  # empty outputs

        # Gather base Gaussian params
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # Get auxiliary opacities (cropped)
        if crop_ids is not None:
            aux_opacities = self.gauss_params["aux_opacities"][crop_ids]
        else:
            aux_opacities = self.gauss_params["aux_opacities"]

        # ── Geometric path (standard opacity) ──────────────────────
        # Standard rasterization with σ_k = α_k · G(x)
        render_geo, alpha_geo, info_geo = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )

        # ── Appearance path (rendering opacity σ'_k = α'_k · σ_k) ─
        # Combine: logit(σ') = logit(α') + logit(α)
        # In logit space: sigmoid(a+b) ≈ sigmoid(a)·sigmoid(b) for
        # independent factors. We use the product in probability space:
        #   rendering_opacity_logit is computed so that
        #   sigmoid(rendering_opacity_logit) = sigmoid(aux) * sigmoid(std)
        std_prob = torch.sigmoid(opacities)
        aux_prob = torch.sigmoid(aux_opacities)
        combined_prob = (std_prob * aux_prob).clamp(1e-6, 1 - 1e-6)
        rendering_opacities = torch.log(combined_prob / (1 - combined_prob))

        render_app, alpha_app, info_app = rasterize_gaussians(
            self, means, quats, scales, rendering_opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
        )

        # ── Store means2d/radii for densification ──────────────────
        # Use the appearance path (primary) for densification signals
        if self.training and info_app["means2d"].requires_grad:
            info_app["means2d"].retain_grad()
        self.xys = info_app["means2d"]
        self.radii = info_app["radii"]

        # ── Assemble outputs ───────────────────────────────────────
        H, W = cam["H"], cam["W"]
        background = self._get_background_color()

        # Appearance path outputs (primary)
        rgb_app = render_app[:, ..., :3] + (1 - alpha_app) * background
        rgb_app = torch.clamp(rgb_app, 0.0, 1.0)

        # Geometric path outputs (for depth losses)
        rgb_geo = render_geo[:, ..., :3] + (1 - alpha_geo) * background
        rgb_geo = torch.clamp(rgb_geo, 0.0, 1.0)

        # Extract depth (last channel in RGB+ED mode)
        depth_app = None
        depth_geo = None
        if render_app.shape[-1] > 3:
            depth_app = torch.where(
                alpha_app > 0,
                render_app[:, ..., -1:],
                render_app[:, ..., -1:].detach().max(),
            ).squeeze(0)
        if render_geo.shape[-1] > 3:
            depth_geo = torch.where(
                alpha_geo > 0,
                render_geo[:, ..., -1:],
                render_geo[:, ..., -1:].detach().max(),
            ).squeeze(0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        outputs = {
            # Appearance path (primary outputs)
            "rgb": rgb_app.squeeze(0),           # C^o — final color
            "depth": depth_app,                   # D^o — appearance depth
            "accumulation": alpha_app.squeeze(0),
            "background": background,
            # Geometric path (for depth losses)
            "rgb_geometric": rgb_geo.squeeze(0),  # C^s — intermediate color
            "depth_geometric": depth_geo,         # D^s — geometric depth
            "accumulation_geometric": alpha_geo.squeeze(0),
        }

        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """StableGS losses:
            L_total = L_color + λ_geo · L_geometry

        where:
            L_color    = L_3DGS(C^o, C^gt) + λ_s · L1(C^s, C^gt)
            L_geometry = λ_consis · L_consis + λ_prior · L_prior
        """
        losses = {}

        # ── Geometric-path color supervision ───────────────────────
        # The primary L_3DGS(C^o, C^gt) is handled by splatfacto's
        # get_loss_dict via the "rgb" output. We add the geometric
        # path color loss: λ_s · L1(C^s, C^gt)
        if "rgb_geometric" in outputs and "image" in batch:
            gt_image = batch["image"].to(outputs["rgb_geometric"].device)
            # Handle resolution mismatch
            if gt_image.shape[:2] != outputs["rgb_geometric"].shape[:2]:
                gt_image = F.interpolate(
                    gt_image.permute(2, 0, 1).unsqueeze(0),
                    size=(
                        int(outputs["rgb_geometric"].shape[0]),
                        int(outputs["rgb_geometric"].shape[1]),
                    ),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            losses["geo_color_loss"] = (
                self.config.lambda_s
                * F.l1_loss(outputs["rgb_geometric"], gt_image)
            )

        # ── Depth consistency loss (self-supervised) ───────────────
        depth_geo = outputs.get("depth_geometric")
        if (
            depth_geo is not None
            and "warp_data" in batch
            and self.config.lambda_consis > 0
        ):
            warp_data = batch["warp_data"]
            consis_loss = _compute_depth_consistency(
                depth_geo, warp_data, self.config
            )
            if consis_loss is not None:
                losses["depth_consistency"] = (
                    self.config.lambda_geo
                    * self.config.lambda_consis
                    * consis_loss
                )

        # ── DUSt3R depth prior loss ────────────────────────────────
        if (
            depth_geo is not None
            and "dust3r_depth" in batch
            and self.config.use_dust3r_prior
            and self.config.lambda_prior > 0
        ):
            dust3r_depth = batch["dust3r_depth"].to(depth_geo.device)
            dust3r_conf = batch.get("dust3r_confidence")
            if dust3r_conf is not None:
                dust3r_conf = dust3r_conf.to(depth_geo.device)

            # Match spatial dims
            if dust3r_depth.shape[:2] != depth_geo.shape[:2]:
                dust3r_depth = F.interpolate(
                    dust3r_depth.permute(2, 0, 1).unsqueeze(0),
                    size=(int(depth_geo.shape[0]), int(depth_geo.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
                if dust3r_conf is not None:
                    dust3r_conf = F.interpolate(
                        dust3r_conf.permute(2, 0, 1).unsqueeze(0),
                        size=(
                            int(depth_geo.shape[0]),
                            int(depth_geo.shape[1]),
                        ),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)

            losses["depth_prior"] = (
                self.config.lambda_geo
                * self.config.lambda_prior
                * weighted_depth_prior_loss(
                    depth_geo, dust3r_depth, dust3r_conf
                )
            )

        return losses

    # ── 4. split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        """Replicate auxiliary opacities during densification."""
        out = super().split_gaussians(split_mask, samps)
        out["aux_opacities"] = self.gauss_params["aux_opacities"][
            split_mask
        ].repeat(samps, 1)
        return out

    # ── 5. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register ALL gauss_params including aux_opacities."""
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    # ── 7. get_outputs_for_camera (viewer) ─────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        """Viewer uses appearance-path outputs only."""
        outs = super().get_outputs_for_camera(camera, obb_box)
        # Remove geometric-path keys from viewer
        for key in ["rgb_geometric", "depth_geometric", "accumulation_geometric"]:
            outs.pop(key, None)
        return outs

    # ── 8. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        """Handle aux_opacities in checkpoints."""
        self.step = 30000
        # Handle legacy format
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict.get(p, dict.get(f"gauss_params.{p}"))

        # Resize gauss_params to match checkpoint
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            key = f"gauss_params.{name}"
            if key in dict:
                new_shape = dict[key].shape
            else:
                new_shape = (newp,) + param.shape[1:]
                # Initialize aux_opacities if not in checkpoint
                dict[key] = torch.full(
                    new_shape, self.config.aux_opacity_init
                )
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    # ── 9. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        # Track auxiliary opacity statistics
        aux_o = torch.sigmoid(self.gauss_params["aux_opacities"])
        metrics["aux_opacity_mean"] = aux_o.mean()
        metrics["aux_opacity_std"] = aux_o.std()
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        return metrics


# ═══════════════════════════════════════════════════════════════════
# Depth Consistency Computation (Self-supervised)
# ═══════════════════════════════════════════════════════════════════

def _compute_depth_consistency(
    depth_geo: Tensor,
    warp_data: Dict,
    config: StableGSModelConfig,
) -> Optional[Tensor]:
    """Compute multi-view depth consistency loss by warping.

    Warps depth from current view to paired views using known camera
    poses and computes L1 loss over valid regions.

    Args:
        depth_geo: (H, W, 1) geometric-path rendered depth
        warp_data: dict with keys:
            "paired_depths": list of (H, W, 1) depth maps from paired views
            "K_src", "K_tgt": (3, 3) intrinsics
            "T_src2tgt": (4, 4) relative transform src → tgt
            "valid_mask": (H, W, 1) valid projection mask
    """
    if not warp_data or "paired_depths" not in warp_data:
        return None

    total_loss = torch.tensor(0.0, device=depth_geo.device)
    count = 0

    paired_depths = warp_data["paired_depths"]
    transforms = warp_data.get("transforms", [])
    intrinsics_src = warp_data.get("K_src")
    intrinsics_tgt_list = warp_data.get("K_tgt_list", [])

    for idx, (paired_depth, T_rel) in enumerate(
        zip(paired_depths, transforms)
    ):
        paired_depth = paired_depth.to(depth_geo.device)
        T_rel = T_rel.to(depth_geo.device)

        K_tgt = (
            intrinsics_tgt_list[idx].to(depth_geo.device)
            if idx < len(intrinsics_tgt_list)
            else intrinsics_src.to(depth_geo.device)
        )
        K_src = intrinsics_src.to(depth_geo.device)

        warped, valid = _warp_depth(
            depth_geo.squeeze(-1), K_src, K_tgt, T_rel
        )

        if valid.sum() > 100:
            loss_fwd = F.l1_loss(
                warped[valid], paired_depth.squeeze(-1)[valid]
            )
            total_loss = total_loss + loss_fwd
            count += 1

    if count == 0:
        return None
    return total_loss / count


def _warp_depth(
    depth: Tensor,
    K_src: Tensor,
    K_tgt: Tensor,
    T_src2tgt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Warp a depth map from source to target view.

    Args:
        depth: (H, W) source depth map
        K_src: (3, 3) source intrinsics
        K_tgt: (3, 3) target intrinsics
        T_src2tgt: (4, 4) relative transformation

    Returns:
        warped_depth: (H, W) warped depth in target view
        valid_mask: (H, W) bool mask of validly projected pixels
    """
    H, W = depth.shape
    device = depth.device

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Unproject to 3D: x = D · K^-1 · [u, v, 1]^T
    ones = torch.ones_like(u)
    pixels = torch.stack([u, v, ones], dim=-1)  # (H, W, 3)
    K_inv = torch.linalg.inv(K_src[:3, :3])
    rays = (K_inv @ pixels.reshape(-1, 3).T).T.reshape(H, W, 3)
    points_3d = rays * depth.unsqueeze(-1)  # (H, W, 3)

    # Transform to target frame
    R = T_src2tgt[:3, :3]
    t = T_src2tgt[:3, 3]
    pts_flat = points_3d.reshape(-1, 3)
    pts_tgt = (R @ pts_flat.T).T + t  # (H*W, 3)

    # Project to target image
    pts_proj = (K_tgt[:3, :3] @ pts_tgt.T).T  # (H*W, 3)
    z_tgt = pts_proj[:, 2]
    u_tgt = (pts_proj[:, 0] / (z_tgt + 1e-8)).reshape(H, W)
    v_tgt = (pts_proj[:, 1] / (z_tgt + 1e-8)).reshape(H, W)
    z_tgt = z_tgt.reshape(H, W)

    # Validity mask
    valid = (
        (u_tgt >= 0)
        & (u_tgt < W)
        & (v_tgt >= 0)
        & (v_tgt < H)
        & (z_tgt > 0)
        & (depth > 0)
    )

    # Use grid_sample for differentiable warping
    grid_x = 2.0 * u_tgt / (W - 1) - 1.0
    grid_y = 2.0 * v_tgt / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

    warped_depth = z_tgt  # depth in target view coordinates

    return warped_depth, valid
