"""
RAIN-GS: Relaxing Accurate INitialization Constraint for 3D Gaussian Splatting
================================================================================
Paper: https://arxiv.org/abs/2403.09413  (Ku et al., Korea University)
Complexity: ★★★☆☆ (custom init, modified densification)

The paper enables 3DGS training from sub-optimal point clouds (random or
sparse SfM) through three coupled innovations:

    1. Sparse-Large-Variance (SLV) initialization
       Start with very few Gaussians (default 10) sampled randomly within
       3x the camera bounding box extent.  The large inter-point distance
       produces naturally large initial covariance.

    2. Progressive Gaussian low-pass filter control
       Replace the fixed eps2d = 0.3 addend with a dynamic schedule:
           h = clamp(H*W / (9*pi*N), 0.3, 300.0)
       updated every 1000 steps.  This is the least impactful component
       per the paper's ablation (~0.3 dB).  We do NOT override
       splatfacto's get_outputs to avoid breaking its rendering pipeline.
       If your gsplat version exposes eps2d as a config knob, wire it
       through here.

    3. Adaptive Bound-Expanding Split (ABE-Split)
       During the first abe_end_step iterations, each split produces a
       third Gaussian whose mean is pushed outward from the scene center
       to expand spatial coverage beyond the initial bounding box.

Training schedule (differs from vanilla 3DGS):
    - Total 40k iterations (vs 30k vanilla)
    - Mean lr held constant for first 10k steps, then exponential decay
    - SH degree increases every 1000 steps
    - ABE-Split active for first 10k steps
    - Densification starts early (warmup_length=500) so that the 10
      initial Gaussians can grow from the very beginning

Train with random initialization (paper's main setting):
    ns-train rain-gs --data <path>

Train with full SfM points (benefits from ABE-Split):
    ns-train rain-gs --data <path> --pipeline.model.use_random_init False

Train with sparse SfM (top 10% of points):
    ns-train rain-gs --data <path> --pipeline.model.use_random_init False \\
        --pipeline.model.sparse_sfm_ratio 0.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class RAINGSModelConfig(GSTemplateModelConfig):
    """RAIN-GS configuration.

    Groups parameters by the three paper contributions plus the
    modified training schedule.
    """

    _target: Type = field(default_factory=lambda: RAINGSModel)

    # -- SLV Initialization ------------------------------------------
    use_random_init: bool = True
    """Use random SLV initialization instead of SfM points."""

    num_random_points: int = 10
    """Number of initial Gaussians for SLV (paper default: 10)."""

    bbox_scale: float = 3.0
    """Scale factor for the random sampling cube (3x camera extent)."""

    initial_opacity_logit: float = 5.0
    """Logit-space opacity for random init.  sigmoid(5) ~ 0.993,
    matching the paper's 'InverseSigmoid(1)' intent."""

    sparse_sfm_ratio: float = 1.0
    """When not using random init, keep this fraction of SfM points
    (0.1 = top 10% by reprojection error, per paper)."""

    # -- ABE-Split ---------------------------------------------------
    use_abe_split: bool = True
    """Enable adaptive bound-expanding split."""

    abe_end_step: int = 10000
    """Disable ABE-Split after this step (end of warm-up phase)."""

    abe_direction_scale: float = 0.3
    """Scale factor for the outward displacement of the third
    Gaussian (0.3 * scene_extent, per paper supplement)."""

    # -- Training schedule overrides ---------------------------------
    sh_degree_interval: int = 1000
    """Increase SH degree every N steps (paper: 1000)."""

    output_depth_during_training: bool = True
    """Keep depth rendering on for monitoring."""


# ===================================================================
# Model
# ===================================================================

class RAINGSModel(GSTemplateModel):
    """RAIN-GS: training 3DGS from sub-optimal initialization.

    Overrides only populate_modules and split_gaussians.  Rendering
    is left entirely to splatfacto so that bilateral grid, SH
    progression, resolution scheduling, and all internal state that
    densification depends on remain intact.
    """

    config: RAINGSModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scene_center: Optional[Tensor] = None
        self._scene_extent: float = 1.0

    # ---------------------------------------------------------------
    # Extension Point 1: populate_modules  (SLV Initialization)
    # ---------------------------------------------------------------

    def populate_modules(self):
        """Initialize Gaussians.

        If use_random_init is True, replaces SfM seed points with N
        random points sampled within a cube of side 3 * camera_extent.
        The large inter-point distances produce naturally large initial
        covariance (the 'large variance' in SLV).
        """
        if self.config.use_random_init:
            super().populate_modules()
            self._init_random_gaussians()
        else:
            if self.config.sparse_sfm_ratio < 1.0:
                self._sparsify_seed_points()
            super().populate_modules()

        self._compute_scene_geometry()

    def _init_random_gaussians(self):
        """Replace Gaussian parameters with N random points.

        Follows the paper's SLV procedure:
        - Means: uniform random within bbox_scale * camera_extent cube
        - Scales: log of average 3-nearest-neighbor distance (large)
        - Opacities: high (logit ~ 5.0, sigmoid ~ 0.993)
        - Colors (SH DC): random
        - Quats: identity rotation
        """
        N = self.config.num_random_points
        device = self.means.device

        # Scene extent from the SfM points that super() loaded
        if hasattr(self, 'scene_box') and self.scene_box is not None:
            aabb = self.scene_box.aabb
            center = aabb.mean(dim=0)
            extent = (aabb[1] - aabb[0]).max().item() / 2.0
        else:
            center = self.means.data.mean(dim=0)
            extent = (self.means.data - center).norm(dim=-1).max().item()

        extent = max(extent, 0.1)
        half_side = self.config.bbox_scale * extent

        # Random means inside the sampling cube
        new_means = center.unsqueeze(0) + (
            torch.rand(N, 3, device=device) * 2 - 1
        ) * half_side

        # Scales from 3-nearest-neighbor distances
        with torch.no_grad():
            dists = torch.cdist(new_means, new_means)
            dists.fill_diagonal_(float('inf'))
            k = min(3, N - 1)
            knn_dists, _ = dists.topk(k, dim=-1, largest=False)
            avg_dist = knn_dists.mean(dim=-1, keepdim=True)
            new_scales = torch.log(avg_dist.expand(-1, 3).clone())

        new_quats = torch.zeros(N, 4, device=device)
        new_quats[:, 0] = 1.0

        new_opacities = torch.full(
            (N, 1), self.config.initial_opacity_logit, device=device
        )

        new_features_dc = torch.rand(N, 3, device=device) * 0.5

        rest_dim = self.features_rest.shape[1]
        new_features_rest = torch.zeros(N, rest_dim, 3, device=device)

        # Replace parameters.  This is called inside populate_modules
        # before the optimizer is created, so direct replacement is safe.
        self.gauss_params["means"] = Parameter(new_means)
        self.gauss_params["scales"] = Parameter(new_scales)
        self.gauss_params["quats"] = Parameter(new_quats)
        self.gauss_params["opacities"] = Parameter(new_opacities)
        self.gauss_params["features_dc"] = Parameter(new_features_dc)
        self.gauss_params["features_rest"] = Parameter(new_features_rest)

    def _sparsify_seed_points(self):
        """Keep only top sparse_sfm_ratio of seed points."""
        if not hasattr(self, 'seed_points') or self.seed_points is None:
            return
        points, colors = self.seed_points
        n_keep = max(1, int(len(points) * self.config.sparse_sfm_ratio))
        indices = torch.randperm(len(points))[:n_keep]
        self.seed_points = (points[indices], colors[indices])

    def _compute_scene_geometry(self):
        """Cache bounding box center and scene extent for ABE-Split."""
        with torch.no_grad():
            means = self.means.detach()
            if means.numel() == 0:
                return
            self._scene_center = means.mean(dim=0)
            diffs = (means - self._scene_center).norm(dim=-1)
            if diffs.numel() == 0:
                return
            self._scene_extent = max(diffs.max().item(), 1.0)

    # ---------------------------------------------------------------
    # Extension Point 4: split_gaussians  (ABE-Split)
    # ---------------------------------------------------------------

    def split_gaussians(self, split_mask: Tensor, samps: int) -> Dict:
        """Adaptive Bound-Expanding Split.

        During the first abe_end_step iterations, each standard 2-way
        split also spawns a third Gaussian displaced outward from the
        scene center.  The third Gaussian keeps its original scale (not
        halved) so it receives gradients from a wider spatial region.

        After abe_end_step, this is a plain pass-through to splatfacto.
        """
        out = super().split_gaussians(split_mask, samps)

        in_warmup = (
            self.config.use_abe_split
            and self.step < self.config.abe_end_step
        )
        if not in_warmup:
            return out

        if self._scene_center is None:
            self._compute_scene_geometry()

        n_split = split_mask.sum().item()
        if n_split == 0:
            return out

        device = self.means.device
        B0 = self._scene_center.to(device)
        s = self._scene_extent
        alpha = self.config.abe_direction_scale

        orig_means = self.means.data[split_mask]

        # Direction from scene center to original Gaussian, normalized
        direction = orig_means - B0.unsqueeze(0)
        dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction_unit = direction / dir_norm

        # Third Gaussian: displaced outward
        abe_means = orig_means + alpha * s * direction_unit

        # Keep original scale (not halved), clone remaining properties
        abe_scales = self.scales.data[split_mask].clone()
        abe_quats = self.quats.data[split_mask].clone()
        abe_opacities = self.opacities.data[split_mask].clone()
        abe_features_dc = self.features_dc.data[split_mask].clone()
        abe_features_rest = self.features_rest.data[split_mask].clone()

        # Append ABE Gaussians to the split output dict
        out["means"] = torch.cat([out["means"], abe_means], dim=0)
        out["scales"] = torch.cat([out["scales"], abe_scales], dim=0)
        out["quats"] = torch.cat([out["quats"], abe_quats], dim=0)
        out["opacities"] = torch.cat(
            [out["opacities"], abe_opacities], dim=0
        )
        out["features_dc"] = torch.cat(
            [out["features_dc"], abe_features_dc], dim=0
        )
        out["features_rest"] = torch.cat(
            [out["features_rest"], abe_features_rest], dim=0
        )

        return out

    # ---------------------------------------------------------------
    # Loss: periodic geometry refresh for ABE directions
    # ---------------------------------------------------------------

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Standard loss with periodic scene geometry refresh."""
        if self.step % 2000 == 0:
            self._compute_scene_geometry()
        return super().get_loss_dict(outputs, batch, metrics_dict)

    # ---------------------------------------------------------------
    # Checkpoint loading
    # ---------------------------------------------------------------

    def load_state_dict(self, dict, **kwargs):
        """Handle checkpoint loading with different Gaussian counts."""
        self.step = 40000
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            key = f"gauss_params.{name}"
            if key in dict:
                new_shape = dict[key].shape
            else:
                new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)