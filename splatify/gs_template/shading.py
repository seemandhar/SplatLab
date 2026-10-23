"""
gs_template/shading.py — GaussianShader shading utilities
=========================================================

Utilities for reflective Gaussian Splatting methods:
- shortest-axis normal extraction from (quat, scale)
- residual SH color evaluation
- differentiable cubemap / mip-map environment lighting
- depth-to-normal conversion for the normal consistency loss
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# Spherical Harmonics
# ═══════════════════════════════════════════════════════════════════

def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def sh_components(degree: int, directions: Tensor) -> Tensor:
    """Return SH basis values up to the requested degree.

    Args:
        degree: maximum SH degree (0..3 used here)
        directions: (..., 3) unit vectors

    Returns:
        (..., (degree+1)^2) basis tensor
    """
    assert 0 <= degree <= 3, "This helper currently supports SH degree up to 3."
    assert directions.shape[-1] == 3

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]
    xx = x * x
    yy = y * y
    zz = z * z

    out = torch.zeros((*directions.shape[:-1], num_sh_bases(degree)), device=directions.device, dtype=directions.dtype)

    # l = 0
    out[..., 0] = 0.28209479177387814

    if degree > 0:
        # l = 1
        out[..., 1] = 0.4886025119029199 * y
        out[..., 2] = 0.4886025119029199 * z
        out[..., 3] = 0.4886025119029199 * x

    if degree > 1:
        # l = 2
        out[..., 4] = 1.0925484305920792 * x * y
        out[..., 5] = 1.0925484305920792 * y * z
        out[..., 6] = 0.9461746957575601 * zz - 0.31539156525252
        out[..., 7] = 1.0925484305920792 * x * z
        out[..., 8] = 0.5462742152960396 * (xx - yy)

    if degree > 2:
        # l = 3
        out[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        out[..., 10] = 2.890611442640554 * x * y * z
        out[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        out[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        out[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        out[..., 14] = 1.445305721320277 * z * (xx - yy)
        out[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    return out


def eval_sh_rgb(coeffs: Tensor, directions: Tensor, degree: int = 3) -> Tensor:
    """Evaluate RGB spherical harmonics.

    Args:
        coeffs: (N, K, 3)
        directions: (N, 3)
        degree: SH degree, default 3

    Returns:
        (N, 3) RGB values
    """
    bases = sh_components(degree, directions)  # (N, K)
    return torch.sum(coeffs * bases[..., None], dim=1)


# ═══════════════════════════════════════════════════════════════════
# Quaternion / shortest-axis normals
# ═══════════════════════════════════════════════════════════════════

def quaternion_to_rotation_matrix(quats: Tensor) -> Tensor:
    """Convert wxyz quaternions to rotation matrices.

    gsplat expects quaternions in wxyz convention. The implementation here
    follows the same convention.
    """
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def shortest_axis_directions(scales_log: Tensor, quats: Tensor) -> Tensor:
    """Return the world-space direction of the smallest Gaussian axis."""
    scales = torch.exp(scales_log)
    rot = quaternion_to_rotation_matrix(quats)  # (N, 3, 3)
    axis_idx = torch.argmin(scales, dim=-1)  # (N,)

    basis = torch.eye(3, device=scales.device, dtype=scales.dtype)[axis_idx]  # (N, 3)
    shortest = torch.bmm(rot, basis.unsqueeze(-1)).squeeze(-1)
    return F.normalize(shortest, dim=-1)


def select_view_aligned_normals(
    shortest_axis: Tensor,
    normal_delta_1: Tensor,
    normal_delta_2: Tensor,
    viewdirs: Tensor,
) -> Tensor:
    """Apply the GaussianShader normal formulation.

    n = v + Δn1 if ω·v > 0 else -(v + Δn2)
    followed by normalization.
    """
    align = torch.sum(viewdirs * shortest_axis, dim=-1, keepdim=True) > 0
    normals = torch.where(align, shortest_axis + normal_delta_1, -(shortest_axis + normal_delta_2))
    return F.normalize(normals, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Environment lighting
# ═══════════════════════════════════════════════════════════════════

def reflect(viewdirs: Tensor, normals: Tensor) -> Tensor:
    """Reflect outgoing viewing directions about the surface normal."""
    dot = torch.sum(viewdirs * normals, dim=-1, keepdim=True)
    reflected = 2.0 * dot * normals - viewdirs
    return F.normalize(reflected, dim=-1)


def gamma_tone_map(rgb_linear: Tensor, gamma: float = 2.2) -> Tensor:
    """Simple gamma tone mapping used for the shading function output."""
    rgb_linear = torch.clamp(rgb_linear, min=0.0, max=1.0)
    return torch.pow(rgb_linear + 1e-8, 1.0 / gamma)


def _direction_to_cubemap_uv(directions: Tensor):
    """Map 3D directions to cubemap face indices and uv coords in [-1, 1].

    Face order:
        0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    """
    dirs = F.normalize(directions, dim=-1)
    x, y, z = dirs.unbind(dim=-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()

    face = torch.zeros_like(x, dtype=torch.long)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    # ±X
    is_x = (ax >= ay) & (ax >= az)
    pos_x = is_x & (x >= 0)
    neg_x = is_x & (x < 0)
    face[pos_x] = 0
    face[neg_x] = 1
    u[pos_x] = -z[pos_x] / ax[pos_x].clamp_min(1e-8)
    v[pos_x] = -y[pos_x] / ax[pos_x].clamp_min(1e-8)
    u[neg_x] = z[neg_x] / ax[neg_x].clamp_min(1e-8)
    v[neg_x] = -y[neg_x] / ax[neg_x].clamp_min(1e-8)

    # ±Y
    is_y = (ay > ax) & (ay >= az)
    pos_y = is_y & (y >= 0)
    neg_y = is_y & (y < 0)
    face[pos_y] = 2
    face[neg_y] = 3
    u[pos_y] = x[pos_y] / ay[pos_y].clamp_min(1e-8)
    v[pos_y] = z[pos_y] / ay[pos_y].clamp_min(1e-8)
    u[neg_y] = x[neg_y] / ay[neg_y].clamp_min(1e-8)
    v[neg_y] = -z[neg_y] / ay[neg_y].clamp_min(1e-8)

    # ±Z
    is_z = ~(is_x | is_y)
    pos_z = is_z & (z >= 0)
    neg_z = is_z & (z < 0)
    face[pos_z] = 4
    face[neg_z] = 5
    u[pos_z] = x[pos_z] / az[pos_z].clamp_min(1e-8)
    v[pos_z] = -y[pos_z] / az[pos_z].clamp_min(1e-8)
    u[neg_z] = -x[neg_z] / az[neg_z].clamp_min(1e-8)
    v[neg_z] = -y[neg_z] / az[neg_z].clamp_min(1e-8)

    return face, u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)


def sample_cubemap(cubemap: Tensor, directions: Tensor) -> Tensor:
    """Differentiably sample a cubemap using manual bilinear interpolation.

    Args:
        cubemap: (6, H, W, 3)
        directions: (N, 3)

    Returns:
        (N, 3) sampled RGB
    """
    face, u, v = _direction_to_cubemap_uv(directions)
    _, H, W, _ = cubemap.shape

    x = (u + 1.0) * 0.5 * (W - 1)
    y = (v + 1.0) * 0.5 * (H - 1)

    x0 = torch.floor(x).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)

    c00 = cubemap[face, y0, x0]
    c01 = cubemap[face, y0, x1]
    c10 = cubemap[face, y1, x0]
    c11 = cubemap[face, y1, x1]

    c0 = c00 * (1 - wx) + c01 * wx
    c1 = c10 * (1 - wx) + c11 * wx
    return c0 * (1 - wy) + c1 * wy


def build_envmap_pyramid(envmap: Tensor, levels: int = 6) -> List[Tensor]:
    """Average-pool a cubemap into a differentiable mip pyramid."""
    pyr = [envmap]
    cur = envmap.permute(0, 3, 1, 2)  # (6, 3, H, W)
    for _ in range(1, levels):
        if cur.shape[-1] <= 1 or cur.shape[-2] <= 1:
            break
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=True)
        pyr.append(cur.permute(0, 2, 3, 1).contiguous())
    return pyr


def sample_prefiltered_envmap(
    envmap: Tensor,
    directions: Tensor,
    roughness: Tensor,
    levels: int = 6,
) -> Tensor:
    """Sample a cubemap mip pyramid with roughness-controlled LOD."""
    pyramid = build_envmap_pyramid(envmap, levels=levels)
    num_levels = len(pyramid)

    lod = roughness.squeeze(-1).clamp(0.0, 1.0) * max(num_levels - 1, 0)
    l0 = torch.floor(lod).long()
    l1 = torch.clamp(l0 + 1, max=num_levels - 1)
    t = (lod - l0.float()).unsqueeze(-1)

    all_samples = torch.stack([sample_cubemap(level, directions) for level in pyramid], dim=1)  # (N, L, 3)
    idx0 = l0[:, None, None].expand(-1, 1, 3)
    idx1 = l1[:, None, None].expand(-1, 1, 3)
    s0 = torch.gather(all_samples, 1, idx0).squeeze(1)
    s1 = torch.gather(all_samples, 1, idx1).squeeze(1)
    return s0 * (1 - t) + s1 * t


def cubemap_to_strip(cubemap: Tensor) -> Tensor:
    """Convert a cubemap to a simple 1x6 strip for viewer visualization."""
    return torch.cat([cubemap[i] for i in range(cubemap.shape[0])], dim=1)


# ═══════════════════════════════════════════════════════════════════
# Depth → normal map
# ═══════════════════════════════════════════════════════════════════

def depth_to_normal_map(depth: Tensor) -> Tensor:
    """Compute a Sobel-like normal map from a rendered depth image."""
    if depth.dim() == 3:
        depth = depth[..., 0]

    d = depth.unsqueeze(0).unsqueeze(0)
    kx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=d.device,
        dtype=d.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=d.device,
        dtype=d.dtype,
    ).view(1, 1, 3, 3)

    dzdx = F.conv2d(d, kx, padding=1)
    dzdy = F.conv2d(d, ky, padding=1)
    n = torch.cat([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=1)
    n = F.normalize(n, dim=1)
    return n.squeeze(0).permute(1, 2, 0).contiguous()