"""
GeoConsensus DataManager — Depth + Normals + Confidence
=========================================================

Extends DepthDataManager to additionally compute:
    1. Surface normals from depth maps via finite differences
    2. Depth confidence (edge-aware: smooth regions → high confidence)
    3. Normal confidence (Laplacian-based: smooth normals → high confidence)
    4. Joint confidence = depth_confidence * normal_confidence

Injects into each training/eval batch:
    batch["depth_image"]      — monocular depth (from parent DepthDataManager)
    batch["depth_normal"]     — (H, W, 3) surface normals derived from depth
    batch["depth_confidence"] — (H, W, 1) joint confidence map

Architecture (extends DepthDataManager):
    __init__() → super().__init__() [depth estimation] → compute normals + confidence
    next_train() → inject depth_image (super) + depth_normal + depth_confidence
    next_eval()  → same
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.datamanagers import DepthDataManager, DepthDataManagerConfig


@dataclass
class GeoConsensusDataManagerConfig(DepthDataManagerConfig):
    """DataManager that computes depth, normals, and confidence maps."""

    _target: Type = field(default_factory=lambda: GeoConsensusDataManager)


class GeoConsensusDataManager(DepthDataManager):
    """Extends DepthDataManager with surface normal and confidence computation.

    After depth estimation (handled by parent), computes:
        - Surface normals via finite differences of depth
        - Depth confidence from gradient magnitude (edge-aware)
        - Normal confidence from Laplacian magnitude
        - Joint confidence = depth_confidence * normal_confidence

    Caches all computed maps in memory for fast batch injection.
    """

    config: GeoConsensusDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute normals and confidence from cached depth maps
        CONSOLE.print("[GeoConsensus] Computing normals and confidence maps...")

        self.train_normals, self.train_confidences = self._compute_normals_and_confidence(
            self.train_depths
        )
        self.eval_normals, self.eval_confidences = self._compute_normals_and_confidence(
            self.eval_depths
        )

        CONSOLE.print(
            f"  Done: {len(self.train_normals)} train, "
            f"{len(self.eval_normals)} eval normal/confidence maps"
        )

        torch.cuda.empty_cache()
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Normal + Confidence Computation
    # ═══════════════════════════════════════════════════════════════

    def _compute_normals_and_confidence(
        self, depths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute surface normals and joint confidence from depth maps.

        Args:
            depths: (N, H, W, 1) depth maps from monocular estimation

        Returns:
            normals:     (N, H, W, 3) surface normal maps
            confidences: (N, H, W, 1) joint confidence maps
        """
        N = depths.shape[0]
        all_normals = []
        all_confidences = []

        for i in range(N):
            depth_2d = depths[i, :, :, 0]  # (H, W)

            # 1. Compute surface normals via finite differences
            normals = self._depth_to_normals(depth_2d)  # (H, W, 3)

            # 2. Compute depth confidence (edge-aware)
            c_depth = self._compute_depth_confidence(depth_2d)  # (H, W)

            # 3. Compute normal confidence (Laplacian-based)
            c_normal = self._compute_normal_confidence(depth_2d)  # (H, W)

            # 4. Joint confidence
            c_joint = c_depth * c_normal  # (H, W)

            all_normals.append(normals)
            all_confidences.append(c_joint.unsqueeze(-1))  # (H, W, 1)

        normals_tensor = torch.stack(all_normals)      # (N, H, W, 3)
        conf_tensor = torch.stack(all_confidences)      # (N, H, W, 1)

        return normals_tensor, conf_tensor

    @staticmethod
    def _depth_to_normals(depth: Tensor) -> Tensor:
        """Convert a depth map to surface normals via finite differences.

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W, 3) normalized surface normals
        """
        # Finite differences
        dz_dx = depth[:, 1:] - depth[:, :-1]  # (H, W-1)
        dz_dy = depth[1:, :] - depth[:-1, :]  # (H-1, W)

        # Pad to original size
        dz_dx = torch.cat([dz_dx, dz_dx[:, -1:]], dim=1)  # (H, W)
        dz_dy = torch.cat([dz_dy, dz_dy[-1:, :]], dim=0)  # (H, W)

        # Normal = (-dz/dx, -dz/dy, 1) normalized
        normals = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
        normals = F.normalize(normals, dim=-1)
        return normals  # (H, W, 3)

    @staticmethod
    def _compute_depth_confidence(depth: Tensor) -> Tensor:
        """Edge-aware depth confidence: smooth regions get high confidence.

        c_depth = 1 - (grad_magnitude / max_grad)

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W) confidence in [0, 1]
        """
        # Gradient magnitude via Sobel-like finite differences
        dx = depth[:, 1:] - depth[:, :-1]  # (H, W-1)
        dy = depth[1:, :] - depth[:-1, :]  # (H-1, W)

        # Pad back
        dx = torch.cat([dx, dx[:, -1:]], dim=1)
        dy = torch.cat([dy, dy[-1:, :]], dim=0)

        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        max_grad = grad_mag.max().clamp(min=1e-8)
        c_depth = 1.0 - (grad_mag / max_grad)
        return c_depth.clamp(0.0, 1.0)

    @staticmethod
    def _compute_normal_confidence(depth: Tensor) -> Tensor:
        """Normal confidence via Laplacian: smooth normals get high confidence.

        c_normal = 1 - (laplacian_magnitude / max_laplacian)

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W) confidence in [0, 1]
        """
        # Laplacian approximation: d2z/dx2 + d2z/dy2
        # Using finite differences on the depth map
        H, W = depth.shape

        # Pad depth for boundary handling
        depth_pad = F.pad(
            depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        ).squeeze(0).squeeze(0)

        # Laplacian = sum of second-order finite differences
        laplacian = (
            depth_pad[1:-1, 2:]
            + depth_pad[1:-1, :-2]
            + depth_pad[2:, 1:-1]
            + depth_pad[:-2, 1:-1]
            - 4 * depth_pad[1:-1, 1:-1]
        )

        lap_mag = laplacian.abs()
        max_lap = lap_mag.max().clamp(min=1e-8)
        c_normal = 1.0 - (lap_mag / max_lap)
        return c_normal.clamp(0.0, 1.0)

    # ═══════════════════════════════════════════════════════════════
    # Batch Injection
    # ═══════════════════════════════════════════════════════════════

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image, depth_normal, depth_confidence into batch."""
        camera, data = super().next_train(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_normal"] = self.train_normals[cam_idx]
        data["depth_confidence"] = self.train_confidences[cam_idx]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image, depth_normal, depth_confidence into batch."""
        camera, data = super().next_eval(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_normal"] = self.eval_normals[cam_idx]
        data["depth_confidence"] = self.eval_confidences[cam_idx]
        return camera, data
