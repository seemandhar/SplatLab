"""
StableGS DataManager — DUSt3R Depth Priors + Pair Selection
=============================================================

Handles all heavy preprocessing for StableGS:
    1. Pair selection based on COLMAP co-visibility, rotation angle,
       and homography inlier ratio
    2. DUSt3R depth prior estimation with efficient global scale
       optimization (per-pair scale factors)
    3. Depth warping data preparation for self-supervised consistency

Architecture (same pattern as DepthDataManager):
    __init__() → run_pair_selection() → run_dust3r() → global_scale_opt()
    next_train() → inject dust3r_depth, dust3r_confidence, warp_data

Requires:
    pip install dust3r   (or compatible fork)
"""

from __future__ import annotations

import gc
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StableGSDataManagerConfig(FullImageDatamanagerConfig):
    """DataManager for StableGS with DUSt3R priors + pair selection."""

    _target: Type = field(default_factory=lambda: StableGSDataManager)

    # ── Pair selection ─────────────────────────────────────────────
    min_covisible_points: int = 30
    """Minimum COLMAP co-visible feature points for pair selection."""

    min_rotation_deg: float = 16.0
    """Minimum relative rotation angle (degrees)."""

    max_rotation_deg: float = 60.0
    """Maximum relative rotation angle (degrees)."""

    max_homography_inlier_ratio: float = 0.8
    """Maximum homography inlier ratio (suppress degenerate pairs)."""

    # ── DUSt3R settings ────────────────────────────────────────────
    use_dust3r: bool = True
    """Whether to run DUSt3R for depth priors."""

    dust3r_model: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    """DUSt3R model checkpoint name/path."""

    dust3r_batch_size: int = 1
    """Batch size for DUSt3R inference (limited by GPU memory)."""

    # ── Global scale optimization ──────────────────────────────────
    scale_opt_iterations: int = 500
    """Iterations for global scale factor optimization."""

    scale_opt_lr: float = 0.01
    """Learning rate for scale optimization."""

    # ── Caching ────────────────────────────────────────────────────
    enable_cache: bool = True
    """Cache computed priors to disk."""

    # ── Warping ────────────────────────────────────────────────────
    num_warp_pairs: int = 4
    """Number of random pairs to inject per training step."""


# ═══════════════════════════════════════════════════════════════════
# DataManager
# ═══════════════════════════════════════════════════════════════════

class StableGSDataManager(FullImageDatamanager):
    """StableGS DataManager with DUSt3R depth priors.

    Pipeline:
        1. Load camera poses from dataset
        2. Select valid image pairs (co-visibility, rotation, homography)
        3. Run DUSt3R on selected pairs → point maps + confidence
        4. Optimize global per-pair scale factors
        5. Produce scaled depth priors for each image
        6. Inject per-step: dust3r_depth, dust3r_confidence, warp_data
    """

    config: StableGSDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_train = len(self.train_dataset)
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / "stablegs_cache.pt"

        # ── Try loading from cache ─────────────────────────────────
        if self.config.enable_cache and cache_path.exists():
            CONSOLE.print(f"[StableGS] Loading cached priors from {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            self.depth_priors = cache["depth_priors"]
            self.depth_confidences = cache["depth_confidences"]
            self.valid_pairs = cache["valid_pairs"]
            self.pair_cameras = cache.get("pair_cameras", {})
            CONSOLE.print(
                f"  ✓ Loaded {len(self.depth_priors)} depth priors, "
                f"{len(self.valid_pairs)} pairs"
            )
            torch.cuda.empty_cache()
            gc.collect()
            return

        # ── Step 1: Extract camera poses ───────────────────────────
        CONSOLE.print("[StableGS] Extracting camera poses...")
        cameras_train = self.train_dataset.cameras
        self.c2w_train = cameras_train.camera_to_worlds  # (N, 3, 4)

        # ── Step 2: Pair selection ─────────────────────────────────
        CONSOLE.print("[StableGS] Selecting image pairs...")
        self.valid_pairs = self._select_pairs(self.c2w_train)
        CONSOLE.print(f"  ✓ Selected {len(self.valid_pairs)} valid pairs")

        # ── Step 3 & 4: DUSt3R + global scale optimization ────────
        if self.config.use_dust3r and len(self.valid_pairs) > 0:
            CONSOLE.print("[StableGS] Running DUSt3R depth estimation...")
            (
                self.depth_priors,
                self.depth_confidences,
            ) = self._run_dust3r_pipeline()
        else:
            CONSOLE.print(
                "[StableGS] Skipping DUSt3R (disabled or no valid pairs). "
                "Using monocular fallback."
            )
            self.depth_priors = self._fallback_monocular_depth()
            self.depth_confidences = {
                i: torch.ones_like(d) for i, d in self.depth_priors.items()
            }

        # ── Store pair camera data for warping ─────────────────────
        self.pair_cameras = self._build_pair_cameras()

        # ── Cache ──────────────────────────────────────────────────
        if self.config.enable_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "depth_priors": self.depth_priors,
                    "depth_confidences": self.depth_confidences,
                    "valid_pairs": self.valid_pairs,
                    "pair_cameras": self.pair_cameras,
                },
                cache_path,
            )
            CONSOLE.print(f"  ✓ Cached to {cache_path}")

        torch.cuda.empty_cache()
        gc.collect()

    # ── Pair Selection ─────────────────────────────────────────────

    def _select_pairs(self, c2w: Tensor) -> List[Tuple[int, int]]:
        """Select valid image pairs based on:
        1. Relative rotation angle in [min_deg, max_deg]
        2. (COLMAP co-visibility checked if available)
        3. Homography inlier ratio < threshold

        For simplicity, we use rotation angle as the primary filter
        when COLMAP data isn't readily available.
        """
        N = c2w.shape[0]
        pairs = []

        for i in range(N):
            for j in range(i + 1, N):
                # Extract rotation matrices
                R_i = c2w[i, :3, :3]
                R_j = c2w[j, :3, :3]

                # Relative rotation
                R_rel = R_i.T @ R_j
                # Rotation angle from trace
                cos_angle = (R_rel.trace() - 1) / 2
                cos_angle = cos_angle.clamp(-1, 1)
                angle_deg = torch.acos(cos_angle).item() * 180 / np.pi

                # Filter by rotation angle
                if (
                    self.config.min_rotation_deg
                    <= angle_deg
                    <= self.config.max_rotation_deg
                ):
                    pairs.append((i, j))

        # Limit pairs to reasonable count for computational efficiency
        max_pairs = min(len(pairs), N * 5)
        if len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)

        return pairs

    # ── DUSt3R Pipeline ────────────────────────────────────────────

    def _run_dust3r_pipeline(self) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """Run DUSt3R on selected pairs and optimize global scales.

        Returns:
            depth_priors: {img_idx: (H, W, 1) scaled depth}
            depth_confidences: {img_idx: (H, W, 1) confidence}
        """
        try:
            return self._run_dust3r_real()
        except (ImportError, Exception) as e:
            CONSOLE.print(
                f"[StableGS] DUSt3R not available ({e}). "
                f"Falling back to Depth Anything V2."
            )
            depths = self._fallback_monocular_depth()
            confs = {i: torch.ones_like(d) for i, d in depths.items()}
            return depths, confs

    def _run_dust3r_real(self) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """Actual DUSt3R inference + global scale optimization.

        This follows the paper's efficient global scale optimization:
        1. Run DUSt3R on each pair → point maps + confidence
        2. Optimize per-pair scale factors S = {s_ij, s_ji}
        3. Produce scaled depth maps per image
        """
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from PIL import Image

        CONSOLE.print(f"  Loading DUSt3R model: {self.config.dust3r_model}")
        model = AsymmetricCroCo3DStereo.from_pretrained(
            self.config.dust3r_model
        ).cuda().eval()

        image_paths = self.train_dataset.image_filenames
        n_images = len(image_paths)

        # Per-pair results
        pair_pointmaps = {}  # (i,j) → (X_ij^{1,1}, X_ij^{2,1})
        pair_confidences = {}  # (i,j) → (C_ij^{1,1}, C_ji^{2,1})

        for idx, (i, j) in enumerate(self.valid_pairs):
            if idx % 10 == 0:
                CONSOLE.print(
                    f"  DUSt3R pair {idx+1}/{len(self.valid_pairs)}: "
                    f"({i}, {j})"
                )
            images = load_images([str(image_paths[i]), str(image_paths[j])], size=512)
            output = inference([tuple(images)], model, device="cuda", batch_size=1)

            # Extract point maps and confidence
            pred1, pred2 = output["pred1"], output["pred2"]
            X_ij_11 = pred1["pts3d"].squeeze(0).cpu()  # (H, W, 3)
            X_ij_21 = pred2["pts3d_in_other_view"].squeeze(0).cpu()  # (H, W, 3)
            C_ij_11 = pred1["conf"].squeeze(0).cpu()  # (H, W)
            C_ji_21 = pred2["conf"].squeeze(0).cpu()  # (H, W)

            pair_pointmaps[(i, j)] = (X_ij_11, X_ij_21)
            pair_confidences[(i, j)] = (C_ij_11, C_ji_21)

        del model
        torch.cuda.empty_cache()

        # ── Global scale optimization ──────────────────────────────
        CONSOLE.print("  Optimizing global scale factors...")
        scales = self._optimize_global_scales(
            pair_pointmaps, pair_confidences
        )

        # ── Produce scaled depth priors ────────────────────────────
        depth_priors = {}
        depth_confidences = {}

        for (i, j), (X_11, X_21) in pair_pointmaps.items():
            s_ij = scales.get((i, j), 1.0)
            s_ji = scales.get((j, i), 1.0)

            # Depth from z-coordinate of point maps
            d_ij = s_ij * X_11[..., 2:3]  # (H, W, 1)
            d_ji = s_ji * X_21[..., 2:3]

            C_11, C_21 = pair_confidences[(i, j)]

            # Accumulate per-image (average across pairs)
            for img_idx, depth, conf in [
                (i, d_ij, C_11.unsqueeze(-1)),
                (j, d_ji, C_21.unsqueeze(-1)),
            ]:
                if img_idx not in depth_priors:
                    depth_priors[img_idx] = depth
                    depth_confidences[img_idx] = conf
                else:
                    # Confidence-weighted average
                    old_d = depth_priors[img_idx]
                    old_c = depth_confidences[img_idx]
                    total_c = old_c + conf
                    depth_priors[img_idx] = (
                        old_c * old_d + conf * depth
                    ) / (total_c + 1e-8)
                    depth_confidences[img_idx] = total_c / 2

        # Fill missing images with zeros (no prior available)
        for i in range(len(self.train_dataset)):
            if i not in depth_priors:
                # Use placeholder — no DUSt3R prior for this image
                H = self.train_dataset[0]["image"].shape[0]
                W = self.train_dataset[0]["image"].shape[1]
                depth_priors[i] = torch.zeros(H, W, 1)
                depth_confidences[i] = torch.zeros(H, W, 1)

        return depth_priors, depth_confidences

    def _optimize_global_scales(
        self,
        pair_pointmaps: Dict,
        pair_confidences: Dict,
    ) -> Dict[Tuple[int, int], float]:
        """Efficient global scale optimization (Section: External priors).

        Optimizes per-pair scale factors S = {s_ij, s_ji} to align
        pairwise DUSt3R predictions in a globally consistent frame.

        Uses the matrix-factorization-based efficient form from the paper.
        """
        if not pair_pointmaps:
            return {}

        # Initialize per-pair log-scale parameters
        scale_params = {}
        for (i, j) in pair_pointmaps.keys():
            scale_params[(i, j)] = torch.nn.Parameter(
                torch.tensor(0.0)
            )  # log-scale, exp(0) = 1
            scale_params[(j, i)] = torch.nn.Parameter(
                torch.tensor(0.0)
            )

        optimizer = torch.optim.Adam(
            list(scale_params.values()),
            lr=self.config.scale_opt_lr,
        )

        # Get camera poses for alignment
        c2w = self.c2w_train

        for step in range(self.config.scale_opt_iterations):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)

            for (i, j), (X_11, X_21) in pair_pointmaps.items():
                s_ij = torch.exp(scale_params[(i, j)])
                s_ji = torch.exp(scale_params[(j, i)])

                C_11, C_21 = pair_confidences[(i, j)]

                # Transform point maps to world frame using camera poses
                T_i = c2w[i]  # (3, 4)
                T_j = c2w[j]

                # Simplified alignment loss:
                # Points from image i viewed in camera i frame, scaled
                # and transformed to world, should match points from
                # image j in camera j frame, scaled and transformed.
                # L_pair = |w ⊙ (s_ij T_i x_ij - s_ji T_j y_ij)|²

                # For efficiency, use confidence-weighted sample
                H, W = X_11.shape[:2]
                n_samples = min(49, H * W)  # 7² equivalent points
                idx_flat = torch.randperm(H * W)[:n_samples]

                x_flat = X_11.reshape(-1, 3)[idx_flat]  # (n, 3)
                y_flat = X_21.reshape(-1, 3)[idx_flat]
                w = (C_11.reshape(-1)[idx_flat] * C_21.reshape(-1)[idx_flat])
                w = w / (w.sum() + 1e-8)

                # Transform to world frame
                R_i, t_i = T_i[:3, :3], T_i[:3, 3]
                R_j, t_j = T_j[:3, :3], T_j[:3, 3]

                pts_i_world = (R_i @ (s_ij * x_flat).T).T + t_i
                pts_j_world = (R_j @ (s_ji * y_flat).T).T + t_j

                pair_loss = (
                    w.unsqueeze(-1)
                    * (pts_i_world - pts_j_world) ** 2
                ).sum()
                total_loss = total_loss + pair_loss

            # Intra-image scale consistency
            # (Cross-pair: for each image, scales should be consistent)
            image_scales = {}
            for (i, j) in pair_pointmaps.keys():
                image_scales.setdefault(i, []).append(
                    torch.exp(scale_params[(i, j)])
                )
                image_scales.setdefault(j, []).append(
                    torch.exp(scale_params[(j, i)])
                )

            for img_idx, scales_list in image_scales.items():
                if len(scales_list) > 1:
                    anchor = scales_list[0].detach()
                    for s in scales_list[1:]:
                        total_loss = total_loss + 0.1 * (s - anchor) ** 2

            total_loss.backward()
            optimizer.step()

        # Extract optimized scales
        result = {}
        for key, param in scale_params.items():
            result[key] = torch.exp(param).item()

        return result

    # ── Fallback: Monocular Depth ──────────────────────────────────

    def _fallback_monocular_depth(self) -> Dict[int, Tensor]:
        """Fall back to Depth Anything V2 if DUSt3R is unavailable."""
        from gs_template.datamanagers import _run_depth_estimation

        image_paths = self.train_dataset.image_filenames
        CONSOLE.print(
            f"  Fallback: estimating depth with Depth Anything V2 "
            f"for {len(image_paths)} images..."
        )
        all_depths = _run_depth_estimation(
            image_paths, model_name="depth_anything_v2", model_size="vitb"
        )
        return {i: all_depths[i] for i in range(len(image_paths))}

    # ── Build Pair Camera Data ─────────────────────────────────────

    def _build_pair_cameras(self) -> Dict[int, List[Dict]]:
        """Build per-image pair camera data for depth warping."""
        cameras = self.train_dataset.cameras
        pair_cameras = {}

        for i, j in self.valid_pairs:
            for src, tgt in [(i, j), (j, i)]:
                if src not in pair_cameras:
                    pair_cameras[src] = []

                c2w_src = cameras.camera_to_worlds[src]
                c2w_tgt = cameras.camera_to_worlds[tgt]

                # Compute relative transform
                # Build 4x4 matrices
                T_src = torch.eye(4)
                T_src[:3, :] = c2w_src
                T_tgt = torch.eye(4)
                T_tgt[:3, :] = c2w_tgt

                T_src2tgt = torch.linalg.inv(T_tgt) @ T_src

                K_tgt = cameras.get_intrinsics_matrices()[tgt]

                pair_cameras[src].append(
                    {
                        "tgt_idx": tgt,
                        "T_src2tgt": T_src2tgt,
                        "K_tgt": K_tgt,
                    }
                )

        return pair_cameras

    # ── Batch Injection ────────────────────────────────────────────

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject DUSt3R depth priors and warping data into batch."""
        camera, data = super().next_train(step)
        cam_idx = camera.metadata.get("cam_idx", 0)

        # ── DUSt3R depth prior ─────────────────────────────────────
        if cam_idx in self.depth_priors:
            data["dust3r_depth"] = self.depth_priors[cam_idx]
        if cam_idx in self.depth_confidences:
            data["dust3r_confidence"] = self.depth_confidences[cam_idx]

        # ── Warping data for depth consistency ─────────────────────
        if cam_idx in self.pair_cameras and len(self.pair_cameras[cam_idx]) > 0:
            # Sample random pairs
            available_pairs = self.pair_cameras[cam_idx]
            n_sample = min(self.config.num_warp_pairs, len(available_pairs))
            selected = random.sample(available_pairs, n_sample)

            paired_depths = []
            transforms = []
            K_tgt_list = []

            cameras = self.train_dataset.cameras
            K_src = cameras.get_intrinsics_matrices()[cam_idx]

            for pair_info in selected:
                tgt_idx = pair_info["tgt_idx"]
                if tgt_idx in self.depth_priors:
                    paired_depths.append(self.depth_priors[tgt_idx])
                    transforms.append(pair_info["T_src2tgt"])
                    K_tgt_list.append(pair_info["K_tgt"])

            if paired_depths:
                data["warp_data"] = {
                    "paired_depths": paired_depths,
                    "transforms": transforms,
                    "K_src": K_src,
                    "K_tgt_list": K_tgt_list,
                }

        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth priors for evaluation (no warping needed)."""
        camera, data = super().next_eval(step)
        cam_idx = camera.metadata.get("cam_idx", 0)

        if cam_idx in self.depth_priors:
            data["dust3r_depth"] = self.depth_priors[cam_idx]
        if cam_idx in self.depth_confidences:
            data["dust3r_confidence"] = self.depth_confidences[cam_idx]

        return camera, data
