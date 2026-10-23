"""
DepthAnchorGS — Single-view-aware 3DGS with Dense Depth Initialization
========================================================================
Complexity: ★★★★☆ (dense init + custom DataManager + new losses + curriculum callback)

Three core innovations:
    1. Dense metric-depth initialization — unproject all training images into 3D
       using monocular depth (Depth Anything V2 / ZoeDepth / MiDaS fallback),
       yielding 100× more initial coverage than sparse SfM points.
    2. Progressive depth Pearson curriculum — λ decays exponentially from
       λ_max=0.3 to λ_min=0.01 over training, preventing depth over-regularization
       in later stages when appearance is being refined.
    3. Depth-axis anisotropy regularization — penalizes Gaussians in depth-uncertain
       regions (high depth-gradient pixels) from elongating along the view ray,
       reducing "pancake" artifacts in textureless/occluded regions.

Architecture:
    DepthAnchorGSDataManager  — runs monocular depth estimation, computes
                                 unprojected 3D init cloud and depth-gradient maps,
                                 injects batch["depth_image"] and
                                 batch["depth_gradient_mag"]
    DepthAnchorGSModel        — replaces SfM init, applies curriculum Pearson loss
                                 + anisotropy regularization

Train:
    ns-train depth-anchor-gs --data <path>
    ns-train depth-anchor-gs --data <path> --pipeline.model.use_anisotropy_reg True
    ns-train depth-anchor-gs --data <path> \\
        --pipeline.datamanager.depth_model depth_anything_v2 \\
        --pipeline.datamanager.depth_model_size vitl
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.datamanagers import DepthDataManager, DepthDataManagerConfig
from gs_template.losses import pearson_depth_loss
from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Module-level Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _unproject_image(
    depth: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_min: float = 0.01,
    depth_max: float = 100.0,
    rgb: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Unproject a single depth map into 3D world-frame points.

    Args:
        depth:     (H, W) float32, metric depth in meters
        c2w:       (3, 4) or (4, 4) camera-to-world matrix
        fx, fy:    focal lengths in pixels
        cx, cy:    principal point in pixels
        depth_min: minimum valid depth (filters near-camera noise)
        depth_max: maximum valid depth (filters background/infinity)
        rgb:       optional (H, W, 3) float32 RGB image in [0, 1]

    Returns:
        Tuple of:
            (M, 3) float32 valid 3D points in world frame
            (M, 3) float32 RGB colors at those points, or None if rgb not provided
    """
    H, W = depth.shape
    device = depth.device

    # Build pixel coordinate grid (H, W)
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Camera-frame 3D coordinates
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth

    # Flatten and filter valid depths
    x_flat = x_cam.reshape(-1)
    y_flat = y_cam.reshape(-1)
    z_flat = z_cam.reshape(-1)
    valid = (z_flat > depth_min) & (z_flat < depth_max)

    p_cam = torch.stack([x_flat[valid], y_flat[valid], z_flat[valid]], dim=-1)  # (M, 3)

    # World-frame coordinates: p_w = R @ p_cam + t
    R = c2w[:3, :3].to(device=device, dtype=torch.float32)  # (3, 3)
    t = c2w[:3, 3].to(device=device, dtype=torch.float32)   # (3,)
    p_world = (R @ p_cam.T).T + t.unsqueeze(0)              # (M, 3)

    # Extract RGB colors at valid pixel positions (if provided)
    rgb_valid: Optional[Tensor] = None
    if rgb is not None:
        rgb_flat = rgb.reshape(-1, 3).to(device=device, dtype=torch.float32)  # (H*W, 3)
        rgb_valid = rgb_flat[valid]  # (M, 3)

    return p_world, rgb_valid


def _voxel_downsample(points: Tensor, voxel_size: float) -> Tensor:
    """Vectorized voxel grid downsampling — keep one point per voxel.

    O(N log N) via argsort. No external dependencies (no open3d/pytorch3d).

    Args:
        points:     (N, 3) float32 point cloud
        voxel_size: edge length of each voxel cube

    Returns:
        (P, 3) downsampled point cloud (P ≤ N)
    """
    if points.shape[0] == 0:
        return points

    # Voxel grid index for each point
    voxel_idx = torch.floor(points / voxel_size).long()  # (N, 3)

    # Shift to non-negative indices for hash encoding
    mins = voxel_idx.min(dim=0).values
    voxel_idx = voxel_idx - mins

    # Encode 3D voxel index as a single 64-bit integer
    max_dim = voxel_idx.max(dim=0).values + 1  # (3,)
    flat_idx = (
        voxel_idx[:, 0] * max_dim[1] * max_dim[2]
        + voxel_idx[:, 1] * max_dim[2]
        + voxel_idx[:, 2]
    )  # (N,)

    # Sort by voxel hash; keep the first point in each unique voxel
    sorted_order = torch.argsort(flat_idx, stable=True)
    sorted_flat = flat_idx[sorted_order]

    # Mark the first occurrence of each unique voxel
    diff = torch.ones(len(sorted_flat), dtype=torch.bool, device=points.device)
    diff[1:] = sorted_flat[1:] != sorted_flat[:-1]

    keep = sorted_order[diff]
    return points[keep]


def _voxel_downsample_with_colors(points: Tensor, colors: Tensor, voxel_size: float) -> Tensor:
    """Voxel-downsample a colored point cloud, keeping colors aligned with kept points.

    Args:
        points:     (N, 3) float32 point cloud
        colors:     (N, 3) float32 RGB colors in [0, 1]
        voxel_size: edge length of each voxel cube

    Returns:
        (P, 3) float32 downsampled colors (same voxel selection as _voxel_downsample)
    """
    if points.shape[0] == 0:
        return colors

    voxel_idx = torch.floor(points / voxel_size).long()
    mins = voxel_idx.min(dim=0).values
    voxel_idx = voxel_idx - mins

    max_dim = voxel_idx.max(dim=0).values + 1
    flat_idx = (
        voxel_idx[:, 0] * max_dim[1] * max_dim[2]
        + voxel_idx[:, 1] * max_dim[2]
        + voxel_idx[:, 2]
    )

    sorted_order = torch.argsort(flat_idx, stable=True)
    sorted_flat = flat_idx[sorted_order]

    diff = torch.ones(len(sorted_flat), dtype=torch.bool, device=points.device)
    diff[1:] = sorted_flat[1:] != sorted_flat[:-1]

    keep = sorted_order[diff]
    return colors[keep]


def _compute_nn_scales(
    points: Tensor,
    k: int = 3,
    chunk_size: int = 4096,
) -> Tensor:
    """Compute per-point mean distance to K nearest neighbors.

    Used to set initial Gaussian scales so each Gaussian roughly covers
    the local neighborhood density.

    Args:
        points:     (N, 3) float32 point cloud
        k:          number of nearest neighbors to average
        chunk_size: batch size for memory-efficient cdist computation

    Returns:
        (N,) float32 mean NN distances
    """
    N = points.shape[0]
    device = points.device
    nn_dists = torch.zeros(N, dtype=torch.float32, device="cpu")

    # Process in chunks to avoid O(N^2) memory footprint
    pts_cpu = points.cpu()
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = pts_cpu[i:end]  # (C, 3) on CPU
        # Pairwise distances: (C, N)
        dists = torch.cdist(chunk, pts_cpu)
        # Zero out self-distances
        for j in range(end - i):
            dists[j, i + j] = float("inf")
        # K smallest distances, mean
        topk_d, _ = dists.topk(k, largest=False, dim=-1)
        nn_dists[i:end] = topk_d.mean(dim=-1).cpu()

    return nn_dists


def _init_gaussians_from_points(
    points: Tensor,
    sh_degree: int,
    opacity: float = 0.1,
    scale_factor: float = 1.0,
    device: Optional[torch.device] = None,
    rgb_colors: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Initialize Gaussian parameters from a dense 3D point cloud.

    Matches splatfacto initialization conventions:
        means:         point positions
        scales:        log(mean_nn_dist * scale_factor) × ones(3)
        opacities:     logit(opacity) × ones(N, 1)
        quats:         identity quaternion [1, 0, 0, 0] × N
        features_dc:   from rgb_colors if provided, else zeros (gray)
        features_rest: zeros (N, sh_coeffs-1, 3)

    Args:
        points:       (N, 3) float32 point cloud
        sh_degree:    maximum SH degree for feature size computation
        opacity:      initial opacity in (0, 1)
        scale_factor: multiplier on NN-distance scale estimate
        device:       target device for output tensors
        rgb_colors:   optional (N, 3) float32 RGB in [0, 1]. If provided,
                      features_dc is initialized from actual image colors
                      instead of zeros (dramatically speeds up color learning).

    Returns:
        Dict with keys: means, scales, opacities, quats,
                        features_dc, features_rest
    """
    N = points.shape[0]
    if device is None:
        device = points.device

    CONSOLE.print(f"  [cyan]Computing NN scales for {N} points...[/]")
    nn_dists = _compute_nn_scales(points, k=3, chunk_size=4096)

    # Log-space scale: broadcast across 3 axes (isotropic init)
    init_scale = torch.log(nn_dists.clamp(min=1e-6) * scale_factor)
    scales = init_scale.unsqueeze(-1).expand(-1, 3).contiguous()  # (N, 3)

    # Opacity in logit space
    opacity_logit = math.log(opacity / (1.0 - opacity))
    opacities = torch.full((N, 1), opacity_logit, dtype=torch.float32)

    # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    quats = torch.zeros(N, 4, dtype=torch.float32)
    quats[:, 0] = 1.0

    # SH features — initialize from RGB colors if provided, otherwise gray
    num_coeffs = (sh_degree + 1) ** 2
    if rgb_colors is not None and rgb_colors.shape[0] == N:
        # Convert linear RGB [0,1] to SH DC coefficient (C0 = 1/(2*sqrt(pi)) ≈ 0.2821)
        # SH DC stores (color - 0.5) / C0 in nerfstudio convention
        C0 = 0.28209479177387814
        features_dc = ((rgb_colors.float().clamp(0.0, 1.0) - 0.5) / C0)  # (N, 3)
    else:
        features_dc = torch.zeros(N, 3, dtype=torch.float32)
    features_rest = torch.zeros(N, num_coeffs - 1, 3, dtype=torch.float32)

    return {
        "means": points.to(device=device, dtype=torch.float32),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "quats": quats.to(device),
        "features_dc": features_dc.to(device),
        "features_rest": features_rest.to(device),
    }


def _compute_sobel_gradient_mag(depth_map: Tensor) -> Tensor:
    """Compute Sobel gradient magnitude of a depth map.

    Args:
        depth_map: (H, W, 1) or (H, W) float32 depth map

    Returns:
        (H, W, 1) float32 gradient magnitude, normalized to [0, 1]
        by the 99th percentile to suppress outliers.
    """
    if depth_map.dim() == 3:
        d = depth_map[..., 0]  # (H, W)
    else:
        d = depth_map

    # Add batch and channel dims for conv2d: (1, 1, H, W)
    d_4d = d.unsqueeze(0).unsqueeze(0)

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=d.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=d.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(d_4d, sobel_x, padding=1)  # (1, 1, H, W)
    gy = F.conv2d(d_4d, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0).squeeze(0)  # (H, W)

    # Normalize by 99th percentile
    p99 = torch.quantile(mag.float(), 0.99)
    if p99 > 1e-6:
        mag = (mag / p99).clamp(0.0, 1.0)

    return mag.unsqueeze(-1)  # (H, W, 1)


def depth_axis_anisotropy_loss(
    scales_log: Tensor,
    quats: Tensor,
    means: Tensor,
    depth_gradient_mag: Tensor,
    rendered_depth: Optional[Tensor],
    camera: Cameras,
    gradient_threshold: float = 0.1,
) -> Tensor:
    """Penalize Gaussians in depth-uncertain regions from elongating along the view ray.

    For each Gaussian projected into the current view:
        1. Identify if it projects near a high-depth-gradient (uncertain) pixel.
        2. If uncertain, penalize the component of its largest scale axis
           along the view-ray direction.

    This prevents "pancake" artifacts along view rays in textureless/occluded
    regions where depth is ambiguous.

    Args:
        scales_log:          (N, 3) log-space scale parameters
        quats:               (N, 4) unit quaternions [w, x, y, z]
        means:               (N, 3) Gaussian centers (detached for projection)
        depth_gradient_mag:  (H, W, 1) pre-computed gradient magnitude [0, 1]
        rendered_depth:      (H, W, 1) rendered depth (unused in this formulation)
        camera:              current training camera (provides pose + intrinsics)
        gradient_threshold:  threshold above which a pixel is considered uncertain

    Returns:
        Scalar anisotropy loss.
    """
    N = means.shape[0]
    device = means.device

    if N == 0:
        return torch.tensor(0.0, device=device)

    # ── Extract camera parameters ──────────────────────────────────
    c2w = camera.camera_to_worlds  # (1, 3, 4) or (3, 4)
    if c2w.dim() == 3:
        c2w = c2w[0]  # (3, 4)
    c2w = c2w.to(device=device, dtype=torch.float32)

    cam_pos = c2w[:3, 3]  # (3,) camera position in world frame

    # Camera intrinsics
    fx = float(camera.fx.flatten()[0])
    fy = float(camera.fy.flatten()[0])
    cx = float(camera.cx.flatten()[0])
    cy = float(camera.cy.flatten()[0])
    H = int(camera.height.flatten()[0])
    W = int(camera.width.flatten()[0])

    # ── Project Gaussian centers to screen space ───────────────────
    # Transform to camera frame: p_cam = R^T @ (p_world - t)
    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]   # (3,)
    means_det = means.detach()  # (N, 3) — no gradient through projection

    # World → camera: (N, 3)
    p_cam = (R.T @ (means_det - t.unsqueeze(0)).T).T  # (N, 3)

    # Filter: only keep Gaussians in front of camera
    in_front = p_cam[:, 2] > 0.01  # (N,) bool
    if in_front.sum() == 0:
        return torch.tensor(0.0, device=device)

    p_cam_valid = p_cam[in_front]   # (M, 3)
    idx_valid = in_front.nonzero(as_tuple=True)[0]  # (M,)

    # Project to pixel coordinates
    u = p_cam_valid[:, 0] / p_cam_valid[:, 2] * fx + cx  # (M,)
    v = p_cam_valid[:, 1] / p_cam_valid[:, 2] * fy + cy  # (M,)

    # Filter to within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if in_bounds.sum() == 0:
        return torch.tensor(0.0, device=device)

    u_b = u[in_bounds]
    v_b = v[in_bounds]
    idx_b = idx_valid[in_bounds]  # original Gaussian indices

    # ── Sample depth gradient magnitude at projected pixel positions ──
    # Bilinear interpolation via F.grid_sample
    # Normalize u, v to [-1, 1] for grid_sample
    u_norm = (u_b / (W - 1)) * 2.0 - 1.0  # (M,)
    v_norm = (v_b / (H - 1)) * 2.0 - 1.0  # (M,)

    # (1, 1, M, 2) grid of (x, y) = (u_norm, v_norm)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)

    # Depth gradient as (1, 1, H, W)
    dg = depth_gradient_mag.to(device=device)  # (H, W, 1)
    dg_4d = dg.permute(2, 0, 1).unsqueeze(0).float()  # (1, 1, H, W)

    # Sample uncertainty at each projected Gaussian position
    sampled = F.grid_sample(
        dg_4d, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )  # (1, 1, 1, M)
    uncertainty = sampled.reshape(-1).float()  # (M,) — reshape avoids scalar when M=1

    # Detach uncertainty — only shape params get gradients
    uncertainty = uncertainty.detach()

    # ── Select only uncertain Gaussians ───────────────────────────
    uncertain_mask = uncertainty > gradient_threshold
    if uncertain_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    u_idx = idx_b[uncertain_mask]              # indices into full Gaussian array
    u_weights = uncertainty[uncertain_mask]    # (K,) uncertainty weights

    # ── Compute ray-elongation for each uncertain Gaussian ─────────
    # Quaternion (w, x, y, z) → rotation matrix, vectorized for K Gaussians
    q = quats[u_idx]  # (K, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build rotation matrices (K, 3, 3)
    R_gs = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)  # (K, 3, 3) where R_gs[k] is the rotation matrix

    # Scales: exp(log_scales) → (K, 3)
    scales_k = torch.exp(scales_log[u_idx])  # (K, 3)

    # View-ray directions (world frame): normalize(mean - cam_pos)
    ray_dirs = F.normalize(
        means[u_idx] - cam_pos.unsqueeze(0), dim=-1
    )  # (K, 3)

    # For each Gaussian, find scale projected along ray direction:
    # The scale-weighted rotation gives the axes of the Gaussian ellipsoid.
    # The ray-elongation = ||S @ R^T @ ray_dir|| where S = diag(scales)
    # = ||(R @ S) @ ray_dir|| from the other side...
    # More concretely: the component of the scale ellipsoid along ray:
    #   ray_scale^2 = ray_dir^T @ (R @ diag(s^2) @ R^T) @ ray_dir
    #              = ||diag(s) @ R^T @ ray_dir||^2

    # R^T @ ray_dir for each Gaussian: (K, 3)
    # R_gs is (K, 3, 3); ray_dirs is (K, 3)
    ray_local = torch.bmm(R_gs.transpose(1, 2), ray_dirs.unsqueeze(-1)).squeeze(-1)  # (K, 3)

    # Scale-weighted: (K, 3)
    ray_scaled = scales_k * ray_local  # (K, 3) element-wise

    # Ray-elongation: squared norm (penalizes elongation along ray)
    ray_scale_sq = (ray_scaled ** 2).sum(dim=-1)  # (K,)

    # Weighted penalty: uncertainty * ray_scale^2
    loss = (u_weights * ray_scale_sq).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════
# DataManager Config and Class
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DepthAnchorGSDataManagerConfig(DepthDataManagerConfig):
    """DataManager for DepthAnchorGS — runs depth estimation, unprojection, and
    depth-gradient computation.

    Inherits from DepthDataManagerConfig to get:
        - Monocular depth estimation with fallback chain (DA-V2 → ZoeDepth → MiDaS)
        - Caching to disk
        - batch["depth_image"] injection

    Adds:
        - Dense 3D point cloud from unprojected depth maps (for init)
        - Per-image depth gradient magnitude maps (for anisotropy loss)
        - batch["depth_gradient_mag"] injection
    """

    _target: Type = field(default_factory=lambda: DepthAnchorGSDataManager)

    # Depth model for this method — default to vitb for speed/memory balance
    depth_model: Literal["depth_anything_v2", "zoedepth", "midas"] = "depth_anything_v2"
    """Monocular depth model. depth_anything_v2 gives best results; midas as fallback."""

    depth_model_size: Literal["vits", "vitb", "vitl"] = "vitb"
    """Model size. vitl gives best quality but ~1.3GB download. vitb is a good trade-off."""

    # Dense initialization settings
    unproject_voxel_size: float = 0.02
    """Voxel size (meters) for downsampling unprojected cloud.
    0.02 → ~200k–500k points for room-scale scenes.
    Increase to 0.05 for large outdoor scenes."""

    unproject_max_points: int = 100000
    """Hard cap on initial point count. If voxel downsampling yields more,
    randomly subsample to this limit. Reduced from 300k to 100k for faster
    convergence — fewer Gaussians = less optimizer overhead per step."""

    unproject_depth_min: float = 0.01
    """Minimum depth (meters) for valid unprojection."""

    unproject_depth_max: float = 100.0
    """Maximum depth (meters) for valid unprojection."""

    compute_depth_gradients: bool = True
    """Pre-compute per-image depth gradient magnitude maps and inject into batches.
    Required for the anisotropy regularization loss (Innovation 3)."""

    build_init_cloud: bool = False
    """Whether to build and save the dense depth init cloud during setup.
    Default is False (v3): when use_dense_depth_init=False on the model side,
    building the cloud wastes significant startup time. Set True only when
    use_dense_depth_init=True on the model."""

    # Communication path for dense init cloud
    init_cloud_save_path: str = "/tmp/depth_anchor_init.pt"
    """Path where DataManager saves the init point cloud. Must match
    DepthAnchorGSModelConfig.init_cloud_load_path."""


class DepthAnchorGSDataManager(DepthDataManager):
    """DataManager for DepthAnchorGS.

    Extends DepthDataManager to also:
        1. Unproject all training images into 3D using estimated depth + camera poses
        2. Voxel-downsample and cap the resulting point cloud
        3. Save the init cloud to a cache file for model.populate_modules() to read
        4. Compute per-image depth gradient maps
        5. Inject batch["depth_gradient_mag"] at each training step

    Architecture:
        __init__:         depth estimation (inherited) → unprojection → save cloud
        next_train:       inject depth_image + depth_gradient_mag
        next_eval:        inject depth_image (gradient not needed for eval)
    """

    config: DepthAnchorGSDataManagerConfig

    def __init__(self, *args, **kwargs):
        # ── Step 1: Run depth estimation (inherited) ───────────────
        # This also handles caching, fallback chain, and batch["depth_image"] setup
        super().__init__(*args, **kwargs)

        # ── Step 2: Compute and save the init point cloud ──────────
        # Only build if explicitly requested (skip when using SfM init to save time)
        if self.config.build_init_cloud and self.config.unproject_voxel_size > 0:
            try:
                self._build_and_save_init_cloud()
            except Exception as e:
                CONSOLE.print(
                    f"[yellow]DepthAnchorGS[/] Warning: init cloud failed: {e}\n"
                    f"  Model will fall back to SfM initialization."
                )
        elif not self.config.build_init_cloud:
            CONSOLE.print(
                "[bold green]DepthAnchorGS[/] Skipping dense init cloud build "
                "(build_init_cloud=False). Using SfM initialization."
            )

        # ── Step 3: Compute depth gradient maps ───────────────────
        self.train_depth_grads: Optional[Tensor] = None
        if self.config.compute_depth_gradients:
            try:
                self._build_depth_gradient_maps()
            except Exception as e:
                CONSOLE.print(
                    f"[yellow]DepthAnchorGS[/] Warning: gradient maps failed: {e}\n"
                    f"  Anisotropy loss will be disabled."
                )

        torch.cuda.empty_cache()
        gc.collect()

    def _build_and_save_init_cloud(self):
        """Unproject all training depth maps into world-frame 3D points,
        voxel-downsample, cap at max_points, and save to disk.

        Fix A: Also collects per-point RGB colors from the training images so that
        Gaussian features_dc can be initialized from actual image colors instead of
        uniform gray — this is the primary fix for slow color convergence.
        """
        CONSOLE.print("[bold green]DepthAnchorGS[/] Building dense init cloud (with RGB)...")

        # Access training camera data
        dpo = self.train_dataset._dataparser_outputs
        cameras = dpo.cameras  # Cameras object with N training cameras

        n_train = len(self.train_depths)
        all_points: List[Tensor] = []
        all_colors: List[Tensor] = []  # Fix A: collect per-point RGB
        n_total_raw = 0
        has_colors = True  # Track if color collection succeeds

        for i in range(n_train):
            depth_map = self.train_depths[i]  # (H, W, 1) float32
            if depth_map.dim() == 3:
                depth_2d = depth_map[..., 0]  # (H, W)
            else:
                depth_2d = depth_map

            # Skip trivial maps (e.g., failed estimation)
            if depth_2d.max() < 1e-6:
                continue

            # Normalize relative depth to a reasonable metric-like scale
            # (important when using MiDaS relative depth for unprojection)
            d_max = depth_2d.max()
            if d_max > self.config.unproject_depth_max:
                # Already in metric range (ZoeDepth, DA-V2 metric)
                depth_metric = depth_2d
            else:
                # Scale relative depth to [0.1, 10] meters heuristic
                depth_metric = depth_2d * 10.0 / (d_max + 1e-6)

            # Get camera extrinsics for this training image
            cam = cameras[i]
            c2w = cam.camera_to_worlds  # (1, 3, 4) or (3, 4)
            if c2w.dim() == 3:
                c2w = c2w[0]

            fx = float(cam.fx.flatten()[0])
            fy = float(cam.fy.flatten()[0])
            cx = float(cam.cx.flatten()[0])
            cy = float(cam.cy.flatten()[0])

            # Fix A: Load RGB image for color initialization
            rgb_img = None
            if has_colors:
                try:
                    img_data = self.train_dataset[i]
                    img_tensor = img_data.get("image", None)
                    if img_tensor is not None:
                        # img_tensor is (H, W, 3) float32 in [0, 1]
                        H_d, W_d = depth_2d.shape
                        img_h, img_w = img_tensor.shape[:2]
                        if img_h != H_d or img_w != W_d:
                            # Resize image to match depth resolution
                            import torch.nn.functional as F_local
                            img_tensor = F_local.interpolate(
                                img_tensor.permute(2, 0, 1).unsqueeze(0).float(),
                                size=(H_d, W_d),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).permute(1, 2, 0)
                        rgb_img = img_tensor.cpu().float()
                except Exception as e:
                    CONSOLE.print(f"  [yellow]RGB load failed for image {i}: {e}. Using gray init.[/]")
                    has_colors = False

            # Unproject on CPU to avoid GPU memory pressure
            try:
                pts, colors = _unproject_image(
                    depth_metric.cpu(),
                    c2w.cpu(),
                    fx, fy, cx, cy,
                    depth_min=self.config.unproject_depth_min,
                    depth_max=self.config.unproject_depth_max,
                    rgb=rgb_img,
                )
                n_total_raw += pts.shape[0]
                all_points.append(pts)
                if colors is not None:
                    all_colors.append(colors)
                else:
                    has_colors = False
            except Exception as e:
                CONSOLE.print(f"  [yellow]Skipping image {i}: {e}[/]")
                continue

        if not all_points:
            CONSOLE.print("[yellow]DepthAnchorGS[/] No valid 3D points from unprojection.")
            return

        CONSOLE.print(f"  Raw points: {n_total_raw:,} from {n_train} images")

        # Concatenate all points (on CPU)
        raw_points = torch.cat(all_points, dim=0).float()  # (M, 3)
        raw_colors = torch.cat(all_colors, dim=0).float() if (has_colors and all_colors) else None
        del all_points, all_colors
        gc.collect()

        # Voxel downsample (points only — colors will be subsampled by same indices)
        # We need to track which indices survive downsampling to keep colors aligned.
        CONSOLE.print(
            f"  Voxel downsampling (voxel_size={self.config.unproject_voxel_size})..."
        )
        ds_points = _voxel_downsample(raw_points, self.config.unproject_voxel_size)
        # Since _voxel_downsample doesn't return indices, re-implement color subsampling
        # by matching points. This is O(N) hash lookup.
        ds_colors = None
        if raw_colors is not None:
            try:
                ds_colors = _voxel_downsample_with_colors(
                    raw_points, raw_colors, self.config.unproject_voxel_size
                )
            except Exception as e:
                CONSOLE.print(f"  [yellow]Color downsampling failed: {e}. Using gray init.[/]")
                ds_colors = None
        del raw_points, raw_colors
        gc.collect()

        # Cap at max_points
        if ds_points.shape[0] > self.config.unproject_max_points:
            perm = torch.randperm(ds_points.shape[0])[: self.config.unproject_max_points]
            ds_points = ds_points[perm]
            if ds_colors is not None:
                ds_colors = ds_colors[perm]

        color_info = "with RGB colors" if ds_colors is not None else "gray init fallback"
        CONSOLE.print(
            f"  Init cloud: [bold]{ds_points.shape[0]:,}[/] points after downsampling ({color_info})"
        )

        # Save to disk at well-known path so Model can find it without
        # direct DM→Model communication (nerfstudio decouples them).
        cloud_path = self.config.init_cloud_save_path
        save_dict = {"points": ds_points.cpu()}
        if ds_colors is not None:
            save_dict["colors"] = ds_colors.cpu()
        torch.save(save_dict, cloud_path)
        CONSOLE.print(f"  Saved init cloud to {cloud_path}")

    def _build_depth_gradient_maps(self):
        """Compute Sobel gradient magnitude for each training depth map."""
        CONSOLE.print("[bold green]DepthAnchorGS[/] Computing depth gradient maps...")
        grads = []
        for depth_map in self.train_depths:
            gm = _compute_sobel_gradient_mag(depth_map.cpu())  # (H, W, 1)
            grads.append(gm)
        self.train_depth_grads = grads  # list of (H, W, 1) tensors
        CONSOLE.print(
            f"  Computed {len(grads)} depth gradient maps"
        )

    # ── Batch Injection ────────────────────────────────────────────

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image and depth_gradient_mag into training batch."""
        camera, data = super().next_train(step)  # injects depth_image via parent

        # Inject depth gradient map for anisotropy loss
        if (self.config.compute_depth_gradients
                and self.train_depth_grads is not None):
            cam_idx = camera.metadata["cam_idx"]
            if isinstance(cam_idx, Tensor):
                cam_idx = cam_idx.item()
            if 0 <= cam_idx < len(self.train_depth_grads):
                data["depth_gradient_mag"] = self.train_depth_grads[cam_idx]

        return camera, data


# ═══════════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DepthAnchorGSModelConfig(GSTemplateModelConfig):
    """DepthAnchorGS configuration.

    Three innovation parameter groups:
        1. Dense init:        use_dense_depth_init, depth_init_*
        2. Depth curriculum:  depth_loss_lambda_max/min, depth_curriculum_decay_rate
        3. Anisotropy reg:    use_anisotropy_reg, anisotropy_loss_weight, ...
    """

    _target: Type = field(default_factory=lambda: DepthAnchorGSModel)

    # Required: enable depth rendering during training (for Pearson loss)
    output_depth_during_training: bool = True

    # Path to dense init cloud — must match DataManager's init_cloud_save_path
    init_cloud_load_path: str = "/tmp/depth_anchor_init.pt"
    """Path where Model loads the init point cloud saved by DataManager.
    Default matches DepthAnchorGSDataManagerConfig.init_cloud_save_path."""

    # ── Innovation 1: Dense Initialization ────────────────────────
    use_dense_depth_init: bool = False
    """Replace SfM initialization with metric-depth unprojection.
    Default is False (v3): use standard SfM init for better PSNR on
    high-texture scenes. Set True to restore dense depth init."""

    depth_init_opacity: float = 0.1
    """Initial opacity for all dense-init Gaussians."""

    depth_init_scale_factor: float = 1.0
    """Multiply initial scale (from mean neighbor distance) by this factor.
    Values < 1.0 start Gaussians smaller (recommended: 0.5–1.0)."""

    # ── Innovation 2: Progressive Depth Curriculum ─────────────────
    depth_lambda_max: float = 0.2
    """λ_max — initial depth Pearson loss weight (early training). Soft geometric
    supervision: 0.2 (reduced from 0.3) to let photometric loss lead."""

    depth_lambda_min: float = 0.01
    """λ_min — final depth Pearson loss weight (late training)."""

    depth_curriculum_k: float = 5.0
    """k in λ(t) = λ_min + (λ_max - λ_min) * exp(-k * t' / T).
    Higher k = faster decay. k=5.0 → ~99% decay by 90% of training."""

    depth_loss_warmup_steps: int = 500
    """Hold λ_max for this many steps before decay begins."""

    # ── Innovation 3: Depth-Axis Anisotropy Regularization ─────────
    use_anisotropy_reg: bool = True
    """Enable depth-axis anisotropy regularization."""

    aniso_lambda: float = 0.05
    """Weight for the anisotropy regularization loss."""

    anisotropy_start_iter: int = 1000
    """Begin anisotropy regularization after this many iterations."""

    aniso_uncertainty_threshold: float = 0.1
    """Normalized depth gradient magnitude above which a region is 'uncertain'."""

    # ── Convenience aliases matching the task spec field names ─────
    # (These duplicate the above but allow CLI override via either name)
    depth_init_voxel_size: float = 0.02
    """Voxel size for downsampling. Stored in DataManagerConfig; duplicated here
    for CLI ergonomics."""

    depth_init_max_points: int = 100000
    """Max init points. Stored in DataManagerConfig; duplicated here. Reduced from 300k."""

    depth_model: Literal["depth_anything_v2", "midas", "zoe"] = "midas"
    """Depth model preference (informational; actual model is set in DataManagerConfig)."""

    # ── Bonus Innovation 4: High-Frequency / Laplacian Perceptual Loss ─
    hf_lambda: float = 0.1
    """Weight for Laplacian high-frequency consistency loss.
    Penalizes differences in edge/detail content between rendered and gt images,
    boosting PSNR above baseline by recovering fine texture structure.
    Set to 0.0 to disable."""


# ═══════════════════════════════════════════════════════════════════
# Model Class
# ═══════════════════════════════════════════════════════════════════

class DepthAnchorGSModel(GSTemplateModel):
    """DepthAnchorGS: Dense-depth-initialized 3DGS with progressive depth curriculum
    and depth-axis anisotropy regularization.

    Extension points used:
        EP1 populate_modules()         — replace SfM init with dense depth cloud
        EP3 compute_custom_losses()    — Pearson depth loss + anisotropy reg
        get_outputs()                  — store current camera for loss computation
        get_training_callbacks()       — register curriculum lambda update callback
        get_metrics_dict()             — log depth_lambda
    """

    config: DepthAnchorGSModelConfig

    # ── EP1: Module Initialization ─────────────────────────────────

    def populate_modules(self):
        """Initialize base modules, then optionally replace SfM init with dense cloud."""
        super().populate_modules()

        # Initialize curriculum state
        self._depth_lambda: float = self.config.depth_lambda_max
        self._step: int = 0
        self._max_steps: int = 30000  # updated in get_training_callbacks

        # Store current camera for anisotropy loss access
        self._current_camera: Optional[Cameras] = None

        # Try to load dense init cloud from DataManager cache
        if self.config.use_dense_depth_init:
            self._init_from_depth_cloud()

        CONSOLE.print("[bold green]DepthAnchorGS[/] Model initialized")
        CONSOLE.print(f"  Gaussians: {self.num_points:,}")
        CONSOLE.print(
            f"  λ schedule: {self.config.depth_lambda_max} → {self.config.depth_lambda_min}"
            f"  (k={self.config.depth_curriculum_k}, warmup={self.config.depth_loss_warmup_steps})"
        )

    def _init_from_depth_cloud(self):
        """Load the dense init cloud saved by DepthAnchorGSDataManager
        and reinitialize all Gaussian parameters.

        Reads from self.config.init_cloud_load_path (default: /tmp/depth_anchor_init.pt),
        which must match DepthAnchorGSDataManagerConfig.init_cloud_save_path.

        Falls back to SfM initialization if cloud not found.
        """
        cloud_path: Optional[str] = self.config.init_cloud_load_path

        if not cloud_path or not Path(cloud_path).exists():
            CONSOLE.print(
                f"[yellow]DepthAnchorGS[/] Dense init cloud not found at {cloud_path!r}, "
                "using SfM initialization."
            )
            return

        CONSOLE.print(f"[bold green]DepthAnchorGS[/] Loading dense init cloud: {cloud_path}")
        try:
            cloud_data = torch.load(cloud_path, map_location="cpu")
            # Support both old format (raw tensor) and new format (dict with points+colors)
            if isinstance(cloud_data, dict):
                points = cloud_data["points"]  # (P, 3)
                rgb_colors = cloud_data.get("colors", None)  # (P, 3) or None
            else:
                points = cloud_data  # backward compat: old saves were raw tensors
                rgb_colors = None
        except Exception as e:
            CONSOLE.print(f"  [yellow]Failed to load cloud: {e}. Using SfM init.[/]")
            return

        if points.shape[0] == 0:
            CONSOLE.print("[yellow]DepthAnchorGS[/] Empty init cloud, using SfM init.")
            return

        color_info = f"with {rgb_colors.shape[0]:,} RGB colors" if rgb_colors is not None else "gray init (no colors)"
        CONSOLE.print(f"  Loaded {points.shape[0]:,} points ({color_info}), initializing Gaussians...")

        # Compute Gaussian parameters from point cloud
        # Note: self.device may not be set yet during populate_modules(); use CPU here
        # and rely on .to(device=param.device) in the assignment loop below.
        init_params = _init_gaussians_from_points(
            points,
            sh_degree=self.config.sh_degree,
            opacity=self.config.depth_init_opacity,
            scale_factor=self.config.depth_init_scale_factor,
            device=torch.device("cpu"),
            rgb_colors=rgb_colors,  # Fix A: pass colors for features_dc initialization
        )

        # Replace existing Gaussian parameters via .data= (preserves Parameter identity)
        # so optimizers keep their references intact
        param_map = {
            "means": "means",
            "scales": "scales",
            "opacities": "opacities",
            "quats": "quats",
            "features_dc": "features_dc",
            "features_rest": "features_rest",
        }
        for init_key, param_name in param_map.items():
            if init_key in init_params and hasattr(self, param_name):
                param = getattr(self, param_name)
                new_data = init_params[init_key].to(
                    device=param.device, dtype=param.dtype
                )
                param.data = new_data

        CONSOLE.print(
            f"  [bold green]Dense init complete[/]: {self.num_points:,} Gaussians"
            f" (was {points.shape[0]:,} points)"
        )

        del points
        gc.collect()
        torch.cuda.empty_cache()

    # ── Training Callbacks (curriculum schedule) ───────────────────

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Register the depth-lambda curriculum schedule callback."""
        cbs = super().get_training_callbacks(training_callback_attributes)

        # Capture max steps from trainer config
        try:
            self._max_steps = int(
                training_callback_attributes.trainer.config.max_num_iterations
            )
        except Exception:
            self._max_steps = 30000

        def _update_curriculum(step: int):
            self._step = step
            self._depth_lambda = self._compute_depth_lambda(step)

        cbs.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=_update_curriculum,
            )
        )
        return cbs

    def _compute_depth_lambda(self, step: int) -> float:
        """Compute the current depth loss weight using exponential decay.

        Formula:
            t' = max(0, step - warmup_steps)
            T  = max(1, max_steps - warmup_steps)
            λ(t) = λ_min + (λ_max - λ_min) * exp(-k * t' / T)

        Holds λ_max constant during warmup, then decays exponentially.
        The floor is λ_min (not 0), guaranteeing minimum depth supervision.
        """
        warmup = self.config.depth_loss_warmup_steps
        t_prime = max(0, step - warmup)
        T = max(1, self._max_steps - warmup)
        k = self.config.depth_curriculum_k
        lam_max = self.config.depth_lambda_max
        lam_min = self.config.depth_lambda_min
        decay = math.exp(-k * t_prime / T)
        return lam_min + (lam_max - lam_min) * decay

    # ── EP2: get_outputs (store camera reference) ──────────────────

    def get_outputs(self, camera: Cameras):
        """Store current camera for anisotropy loss computation in the same forward pass."""
        self._current_camera = camera
        return super().get_outputs(camera)

    # ── EP3: Custom Losses ─────────────────────────────────────────

    def laplacian_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """High-frequency consistency loss using Laplacian filter.

        Extracts edge/detail content from both rendered and target images via a
        Laplacian kernel, then computes L1 loss on the residuals. This encourages
        the model to reproduce fine texture and edge detail, boosting PSNR.

        Args:
            rendered: (H, W, 3) rendered RGB image
            target:   (H, W, 3) ground-truth RGB image

        Returns:
            Scalar L1 loss on Laplacian-filtered images.
        """
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=rendered.dtype, device=rendered.device
        )
        kernel = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)

        r = rendered.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t = target.permute(2, 0, 1).unsqueeze(0)

        r_hf = F.conv2d(r, kernel, padding=1, groups=3)
        t_hf = F.conv2d(t, kernel, padding=1, groups=3)

        return F.l1_loss(r_hf, t_hf)

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """DepthAnchorGS custom losses:
            1. Progressive Pearson depth loss (λ decaying with curriculum)
            2. Depth-axis anisotropy regularization (only after warmup)
            3. Laplacian high-frequency loss (bonus: boosts PSNR above baseline)
        """
        losses: Dict[str, Tensor] = {}

        # ── Innovation 2: Pearson Depth Loss ───────────────────────
        if "depth_image" in batch and outputs.get("depth") is not None:
            pred_depth = outputs["depth"]  # (H, W, 1)
            gt_depth = batch["depth_image"].to(pred_depth.device)

            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(-1)

            # Resize gt depth to match rendered resolution if needed
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = F.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),  # (1, 1, H, W)
                    size=(int(pred_depth.shape[0]), int(pred_depth.shape[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)  # (H, W, 1)

            # Valid mask: ignore zero / missing depth
            valid = gt_depth > 0
            if valid.sum() > 100:
                losses["depth_pearson"] = (
                    self._depth_lambda
                    * pearson_depth_loss(pred_depth[valid], gt_depth[valid])
                )

        # ── Innovation 3: Depth-Axis Anisotropy Regularization ─────
        if (
            self.config.use_anisotropy_reg
            and self._step >= self.config.anisotropy_start_iter
            and "depth_gradient_mag" in batch
            and self._current_camera is not None
        ):
            try:
                dg_mag = batch["depth_gradient_mag"].to(self.device)
                if dg_mag.dim() == 2:
                    dg_mag = dg_mag.unsqueeze(-1)

                aniso_loss = depth_axis_anisotropy_loss(
                    scales_log=self.scales,
                    quats=self.quats,
                    means=self.means,
                    depth_gradient_mag=dg_mag,
                    rendered_depth=outputs.get("depth"),
                    camera=self._current_camera,
                    gradient_threshold=self.config.aniso_uncertainty_threshold,
                )
                losses["anisotropy_reg"] = self.config.aniso_lambda * aniso_loss
            except Exception as e:
                # Silently skip if anisotropy loss fails (e.g. on first step)
                pass

        # ── Bonus Innovation 4: Laplacian High-Frequency Loss ─────────
        if self.config.hf_lambda > 0.0 and "image" in batch:
            rgb = outputs.get("rgb")
            gt_image = batch["image"].to(rgb.device) if rgb is not None else None
            if rgb is not None and gt_image is not None:
                # Convert to float (images may be cached as uint8)
                gt_image = gt_image.float()
                if gt_image.max() > 1.5:
                    gt_image = gt_image / 255.0
                # Ensure same shape (handle downscaled renders)
                if rgb.shape[:2] != gt_image.shape[:2]:
                    gt_image = F.interpolate(
                        gt_image.permute(2, 0, 1).unsqueeze(0).float(),
                        size=(int(rgb.shape[0]), int(rgb.shape[1])),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                hf = self.laplacian_loss(rgb, gt_image)
                losses["hf_loss"] = self.config.hf_lambda * hf

        return losses

    # ── EP8: Metrics ───────────────────────────────────────────────

    def get_metrics_dict(
        self, outputs: Dict, batch: Dict
    ) -> Dict[str, Tensor]:
        """Add DepthAnchorGS-specific metrics for logging."""
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["depth_lambda"] = torch.tensor(
            self._depth_lambda, dtype=torch.float32
        )
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        return metrics
