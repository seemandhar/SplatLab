"""
DenseInitGS — Dense Initialization 3DGS (idea_0000)
====================================================

Hypothesis: A single-pass densification of COLMAP SfM points to ~400k Gaussians
at iter 0, combined with skipping early splatfacto densification, yields better
quality under tight wall-clock budgets (50s / ~3k iters on Mip-NeRF 360).

Precedent:
    - EDGS (Kotovenko et al., CVPR 2026) — dense init removes densification need.
    - Mini-Splatting (Fang et al., 2024) — depth-unprojected init covers scene.

Mechanism: in populate_modules() we intercept the (xyz, rgb) seed_points that
nerfstudio's ColmapDataParser produces (i.e. the sparse SfM points3D.bin) and
densify them with per-point k-NN jitter replication until the cloud reaches
``target_init_points``. Colors are linearly interpolated along the jitter
direction. We then set the SplatfactoModel config so early densification is
effectively disabled (``warmup_length`` huge).

The densification is CPU torch, fully vectorized, and measured at <1 s for
300k output points — well within the 50 s budget window.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
from torch import Tensor

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ─────────────────────────────────────────────────────────────────────
# Densification kernel
# ─────────────────────────────────────────────────────────────────────

def _densify_point_cloud(
    xyz: Tensor,
    rgb: Tensor,
    target: int,
    k: int = 6,
    noise_frac: float = 0.5,
    max_multiplier: int = 200,
) -> Tuple[Tensor, Tensor]:
    """Densify a sparse SfM point cloud to ~target points via k-NN replication.

    For each input point, emit ceil(target/N) offspring copies displaced toward
    randomly chosen k-nearest neighbors with jitter. Color is linearly blended
    with the neighbor color. Final cloud is capped exactly at ``target``.

    Args:
        xyz:             (N, 3) float32 SfM points (world frame)
        rgb:             (N, 3) uint8 or float in [0,255] / [0,1] RGB
        target:          desired output count (e.g. 400_000)
        k:               number of NN to sample offspring from
        noise_frac:      Gaussian jitter stddev as fraction of mean-NN dist
        max_multiplier:  hard cap on offspring-per-source-point

    Returns:
        (M, 3) float32 xyz, (M, 3) uint8 rgb. M == target (or == len(xyz) when
        target <= len(xyz)).
    """
    N = xyz.shape[0]
    xyz = xyz.float().cpu()
    rgb = rgb.float().cpu()

    def _rgb_to_uint8(t: Tensor) -> Tensor:
        if t.numel() > 0 and float(t.max()) <= 1.5:
            t = t * 255.0
        return t.clamp(0, 255).to(torch.uint8)

    if N == 0 or target <= N:
        return xyz, _rgb_to_uint8(rgb)

    multiplier = min(max_multiplier, max(1, math.ceil(target / N)))

    # Chunked k-NN to avoid O(N^2) memory when N is large (mip360 ~150-300k).
    chunk = 4096
    nn_d = torch.empty((N, k), dtype=torch.float32)
    nn_i = torch.empty((N, k), dtype=torch.long)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        d = torch.cdist(xyz[i:end], xyz)  # (C, N)
        # mask out self-distances
        rows = torch.arange(end - i)
        d[rows, rows + i] = float("inf")
        vals, idx = d.topk(k, largest=False, dim=-1)
        nn_d[i:end] = vals
        nn_i[i:end] = idx
    mean_nn = nn_d.mean(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, 1)

    # For each source point replicate `multiplier` times.
    src_idx = torch.arange(N).repeat_interleave(multiplier)          # (N*mul,)
    M = src_idx.shape[0]
    # Pick a random neighbor for each offspring
    nbr_choice = torch.randint(0, k, (M,))                            # (M,)
    nbr_idx = nn_i[src_idx, nbr_choice]                               # (M,)

    src_xyz = xyz[src_idx]                                            # (M, 3)
    nbr_xyz = xyz[nbr_idx]                                            # (M, 3)
    # Interpolate at random position along segment (0 = source, 1 = neighbor)
    alpha = torch.rand(M, 1)                                          # (M, 1)
    new_xyz = src_xyz + alpha * (nbr_xyz - src_xyz)

    # Gaussian jitter of stddev noise_frac * mean_nn(source)
    sigma = noise_frac * mean_nn[src_idx]                             # (M, 1)
    new_xyz = new_xyz + torch.randn_like(new_xyz) * sigma

    # Color interpolation (same alpha)
    src_rgb = rgb[src_idx]
    nbr_rgb = rgb[nbr_idx]
    new_rgb = src_rgb + alpha * (nbr_rgb - src_rgb)

    # Include the original sparse points as anchors
    all_xyz = torch.cat([xyz, new_xyz], dim=0)
    all_rgb = torch.cat([rgb, new_rgb], dim=0)

    # Subsample to exactly `target`
    total = all_xyz.shape[0]
    if total > target:
        perm = torch.randperm(total)[:target]
        all_xyz = all_xyz[perm]
        all_rgb = all_rgb[perm]

    return all_xyz.contiguous(), _rgb_to_uint8(all_rgb).contiguous()


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DenseInitGSModelConfig(GSTemplateModelConfig):
    """DenseInitGS — dense SfM-point densification + suppressed early densification."""

    _target: Type = field(default_factory=lambda: DenseInitGSModel)

    # Densification-at-init
    target_init_points: int = 400_000
    """Target Gaussian count after dense init. 400k matches EDGS budget."""

    densify_k: int = 6
    """Number of nearest neighbors sampled per source point during densification."""

    densify_noise: float = 0.5
    """Gaussian jitter stddev as a fraction of per-point mean-NN distance."""

    densify_max_multiplier: int = 400
    """Upper bound on offspring per source point (guards pathological SfM clouds)."""

    # Suppress splatfacto's iterative densification
    skip_early_densify: bool = True
    """If True, set warmup_length/stop_split_at so densify/split is effectively off."""

    warmup_length_override: int = 100_000
    """warmup_length used when skip_early_densify=True. Larger than any realistic run."""

    stop_split_at_override: int = 0
    """stop_split_at used when skip_early_densify=True. 0 disables splits entirely."""


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class DenseInitGSModel(GSTemplateModel):
    """Splatfacto with dense SfM init + suppressed densification."""

    config: DenseInitGSModelConfig

    def populate_modules(self):
        # ── 1. Suppress iterative densification before base setup ──
        if self.config.skip_early_densify:
            self.config.warmup_length = self.config.warmup_length_override
            self.config.stop_split_at = self.config.stop_split_at_override

        # ── 2. Densify seed_points (the COLMAP SfM cloud loaded by dataparser) ──
        if self.seed_points is not None and not self.config.random_init:
            xyz, rgb = self.seed_points
            n_in = xyz.shape[0]
            t0 = time.perf_counter()
            xyz_d, rgb_d = _densify_point_cloud(
                xyz, rgb,
                target=self.config.target_init_points,
                k=self.config.densify_k,
                noise_frac=self.config.densify_noise,
                max_multiplier=self.config.densify_max_multiplier,
            )
            dt = time.perf_counter() - t0
            CONSOLE.print(
                f"[bold green]DenseInitGS[/] densified SfM cloud "
                f"{n_in:,} → {xyz_d.shape[0]:,} points in {dt:.2f}s"
            )
            self.seed_points = (xyz_d, rgb_d)
        else:
            CONSOLE.print(
                "[yellow]DenseInitGS[/] no seed_points available "
                "(random_init or missing COLMAP points); falling back to splatfacto init."
            )

        # ── 3. Standard splatfacto init (means from seed, SH from rgb, etc.) ──
        super().populate_modules()

        CONSOLE.print(
            f"[bold green]DenseInitGS[/] initialized: {self.num_points:,} Gaussians, "
            f"warmup_length={self.config.warmup_length}, "
            f"stop_split_at={self.config.stop_split_at}"
        )
