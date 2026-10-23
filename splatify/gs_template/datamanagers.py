"""
gs_template/datamanagers.py — Custom DataManagers
====================================================

Nerfstudio's architecture: DataManager produces batches, Model consumes them.
Any preprocessing (depth estimation, feature extraction) lives HERE, not
in the model. This follows the feature-splatting pattern.

DataManager Pipeline:
    1. __init__: Run heavy preprocessing (depth estimation), cache to disk
    2. next_train(): Inject preprocessed data into each training batch
    3. next_eval():  Same for evaluation

Included:
    DepthDataManager — Runs monocular depth estimation on all images,
                       caches results, injects batch["depth_image"].

Few-shot usage (via colmap dataparser):
    ns-train <method> --data <path> colmap --num-training-views 12
    ns-train <method> --data <path> colmap --num-training-views 3
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════
# Model size mapping (vitX → HuggingFace model names)
# ═══════════════════════════════════════════════════════════════════

_DA_V2_SIZE_MAP = {"vits": "Small", "vitb": "Base", "vitl": "Large"}


# ═══════════════════════════════════════════════════════════════════
# Depth Estimation DataManager (with few-shot support)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DepthDataManagerConfig(FullImageDatamanagerConfig):
    """DataManager that runs monocular depth estimation on all images.

    Supports few-shot training via num_training_views: selects N
    evenly-spaced views from the full training set.
    """

    _target: Type = field(default_factory=lambda: DepthDataManager)

    depth_model: Literal["depth_anything_v2", "zoedepth", "midas"] = "depth_anything_v2"
    """Which pretrained monocular depth model to use."""

    depth_model_size: Literal["vits", "vitb", "vitl"] = "vitb"
    """Model size variant (for Depth Anything v2)."""

    enable_cache: bool = True
    """Cache estimated depths to disk for fast reload."""

    # ── Few-shot subsampling ───────────────────────────────────────

    num_training_views: int = -1
    """Number of training views for few-shot. -1 = use ALL training views.
    Positive values select N evenly-spaced views from the training set.

    Standard few-shot protocols:
        LLFF (RegNeRF):       3 views  (from ~20-60 total)
        Mip-NeRF360 (FSGS):  24 views (from ~100-300 total)
        NeRF-Synthetic:       8 views  (from 100 total)

    Usage:
        ns-train fsgs --data <path> --pipeline.datamanager.num_training_views 12
    """


class DepthDataManager(FullImageDatamanager):
    """Estimates monocular depth for all images and injects into batches.

    On first run, estimates depth for every image using the chosen model
    and caches results to {data_dir}/depth_cache_{model}.pt.
    On subsequent runs, loads from cache.

    When num_training_views > 0, subsamples the training set to N
    evenly-spaced views BEFORE depth estimation (saves compute).

    Injects batch["depth_image"] at each training/eval step.

    Architecture (same pattern as Feature Splatting):
        __init__() → subsample (if few-shot) → estimate_all_depths() → cache
        next_train() → inject depth into batch
        next_eval()  → inject depth into batch
    """

    config: DepthDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── Few-shot subsampling ───────────────────────────────────
        # Must happen BEFORE depth estimation to avoid wasting compute
        # on images that won't be used for training.
        if self.config.num_training_views > 0:
            self._subsample_training_views()

        # ── Estimate/load depth maps ───────────────────────────────
        all_depths = self._estimate_all_depths()

        # Split train/eval
        n_train = len(self.train_dataset)
        self.train_depths = all_depths[:n_train]
        self.eval_depths = all_depths[n_train:]
        assert len(self.eval_depths) == len(self.eval_dataset), (
            f"Depth count mismatch: {len(self.eval_depths)} vs {len(self.eval_dataset)}"
        )

        # Free GPU memory after estimation
        torch.cuda.empty_cache()
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Few-shot Training View Subsampling
    # ═══════════════════════════════════════════════════════════════

    def _subsample_training_views(self):
        """Select N evenly-spaced views from the training set.

        Modifies:
            - self.train_dataset._dataparser_outputs (image_filenames, cameras, etc.)
            - self.cached_train (cached image tensors)
            - self.train_unseen_cameras (epoch cycling list)

        The subsampling uses a uniform stride: indices = linspace(0, total-1, N)
        This matches the standard few-shot protocol used by RegNeRF, FSGS, etc.
        """
        n = self.config.num_training_views
        total = len(self.train_dataset)

        if n >= total:
            CONSOLE.print(
                f"[yellow]num_training_views={n} >= total training images={total}, "
                f"using all training views[/]"
            )
            return

        if n <= 0:
            return

        # ── Compute evenly-spaced indices ──────────────────────────
        # linspace for uniform spacing (matches RegNeRF/FSGS protocol)
        indices = torch.linspace(0, total - 1, n).long().tolist()
        # Remove duplicates (can happen if n is close to total)
        indices = sorted(set(indices))
        n_actual = len(indices)

        CONSOLE.print(
            f"[bold green]Few-shot subsampling[/]: selecting "
            f"{n_actual}/{total} training views"
        )
        CONSOLE.print(f"  Selected indices: {indices}")

        # ── Subsample dataparser outputs ───────────────────────────
        dpo = self.train_dataset._dataparser_outputs

        # Image filenames
        dpo.image_filenames = [dpo.image_filenames[i] for i in indices]

        # Cameras (nerfstudio Cameras supports tensor indexing)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        dpo.cameras = dpo.cameras[idx_tensor]

        # Optional: mask filenames
        if hasattr(dpo, "mask_filenames") and dpo.mask_filenames is not None:
            dpo.mask_filenames = [dpo.mask_filenames[i] for i in indices]

        # Optional: depth filenames (sensor depth, not our estimated depth)
        if hasattr(dpo, "depth_filenames") and dpo.depth_filenames is not None:
            dpo.depth_filenames = [dpo.depth_filenames[i] for i in indices]

        # Optional: metadata dict (per-image metadata tensors)
        if hasattr(dpo, "metadata") and dpo.metadata is not None:
            for key, val in dpo.metadata.items():
                if isinstance(val, Tensor) and val.shape[0] == total:
                    dpo.metadata[key] = val[idx_tensor]
                elif isinstance(val, list) and len(val) == total:
                    dpo.metadata[key] = [val[i] for i in indices]

        # ── Subsample cached training images ───────────────────────
        # FullImageDatamanager caches images in self.cached_train.
        # Two scenarios depending on nerfstudio version:
        #   (a) Eager: cached_train has `total` entries (pre-subsampled)
        #       → pick entries at `indices`
        #   (b) Lazy: accessing cached_train triggers caching AFTER we
        #       already modified _dataparser_outputs → it already has
        #       `n_actual` correct entries → just re-index image_idx
        if hasattr(self, "cached_train") and self.cached_train is not None:
            cache_len = len(self.cached_train)

            if cache_len == total:
                # (a) Eager caching — need to select subset
                CONSOLE.print(f"  Subsampling cached_train: {cache_len} → {n_actual}")
                self.cached_train = [self.cached_train[i] for i in indices]
            elif cache_len == n_actual:
                # (b) Lazy caching — already subsampled, nothing to filter
                CONSOLE.print(f"  cached_train already has {cache_len} entries (lazy)")
            else:
                CONSOLE.print(
                    f"  [yellow]Warning: cached_train has {cache_len} entries, "
                    f"expected {total} or {n_actual}[/]"
                )

            # Re-index image_idx in all entries
            for new_idx, entry in enumerate(self.cached_train):
                if isinstance(entry, dict) and "image_idx" in entry:
                    entry["image_idx"] = new_idx

        # ── Reset the unseen cameras list for epoch cycling ────────
        # FullImageDatamanager cycles through train_unseen_cameras;
        # after subsampling, valid indices are [0, n_actual).
        if hasattr(self, "train_unseen_cameras"):
            self.train_unseen_cameras = list(range(n_actual))
            random.shuffle(self.train_unseen_cameras)

        CONSOLE.print(
            f"[bold green]Few-shot subsampling complete[/]: "
            f"{n_actual} training views active"
        )

    # ═══════════════════════════════════════════════════════════════
    # Depth Estimation
    # ═══════════════════════════════════════════════════════════════

    def _estimate_all_depths(self) -> Tensor:
        """Run monocular depth estimation on all train+eval images.

        Returns:
            (N, H, W, 1) tensor of estimated depths for all images.
        """
        image_fnames = (
            self.train_dataset.image_filenames
            + self.eval_dataset.image_filenames
        )

        # Check cache — include num_training_views in cache name so
        # different few-shot configs don't collide
        cache_dir = self.config.dataparser.data
        nv_tag = ""
        if self.config.num_training_views > 0:
            nv_tag = f"_nv{self.config.num_training_views}"
        cache_name = (
            f"depth_cache_{self.config.depth_model}"
            f"_{self.config.depth_model_size}{nv_tag}.pt"
        )
        cache_path = cache_dir / cache_name

        if self.config.enable_cache and cache_path.exists():
            CONSOLE.print(f"Loading cached depths from {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            cached_fnames = cache.get("image_fnames")
            if cached_fnames == image_fnames:
                CONSOLE.print(
                    f"  ✓ Loaded {len(cache['depths'])} depth maps from cache"
                )
                return cache["depths"]
            else:
                CONSOLE.print("  Cache invalidated (image list changed), re-estimating...")

        # Estimate depths
        CONSOLE.print(
            f"Estimating depths with {self.config.depth_model} "
            f"({self.config.depth_model_size}) for {len(image_fnames)} images..."
        )
        depths = _run_depth_estimation(
            image_fnames,
            model_name=self.config.depth_model,
            model_size=self.config.depth_model_size,
        )

        # Cache
        if self.config.enable_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"image_fnames": image_fnames, "depths": depths},
                cache_path,
            )
            CONSOLE.print(f"  ✓ Cached depths to {cache_path}")

        return depths

    # ═══════════════════════════════════════════════════════════════
    # Batch Injection
    # ═══════════════════════════════════════════════════════════════

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image into training batch."""
        camera, data = super().next_train(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_image"] = self.train_depths[cam_idx]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Inject depth_image into eval batch."""
        camera, data = super().next_eval(step)
        cam_idx = camera.metadata["cam_idx"]
        data["depth_image"] = self.eval_depths[cam_idx]
        return camera, data


# ═══════════════════════════════════════════════════════════════════
# Depth Estimation Backend
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_depth_estimation(
    image_paths: list,
    model_name: str = "depth_anything_v2",
    model_size: str = "vitb",
) -> Tensor:
    """Run monocular depth estimation on a list of images.

    Supports multiple backends. Falls back gracefully if a model
    is not installed.

    NOTE: Many depth models (ZoeDepth, MiDAS) store numpy.int64 dimensions
    internally which newer PyTorch rejects in F.interpolate. We globally
    patch F.interpolate to auto-cast sizes to Python int for the duration
    of depth estimation, then restore the original.

    Returns:
        (N, H, W, 1) float32 tensor of metric (or relative) depth maps.
    """
    import torch.nn.functional as F
    import torch.nn as nn

    # ── Patch F.interpolate to accept numpy.int64 sizes ──────────
    _original_interpolate = F.interpolate

    def _safe_interpolate(input, size=None, **kwargs):
        if size is not None:
            if isinstance(size, (list, tuple)):
                size = tuple(int(s) for s in size)
            elif not isinstance(size, int):
                size = int(size)
        return _original_interpolate(input, size=size, **kwargs)

    F.interpolate = _safe_interpolate
    nn.functional.interpolate = _safe_interpolate

    try:
        if model_name == "depth_anything_v2":
            return _depth_anything_v2(image_paths, model_size)
        elif model_name == "zoedepth":
            return _zoedepth(image_paths)
        elif model_name == "midas":
            return _midas(image_paths)
        else:
            raise ValueError(
                f"Unknown depth model: {model_name}. "
                f"Supported: depth_anything_v2, zoedepth, midas"
            )
    finally:
        # Always restore original
        F.interpolate = _original_interpolate
        nn.functional.interpolate = _original_interpolate


def _depth_anything_v2(image_paths: list, model_size: str) -> Tensor:
    """Depth Anything v2 — strong default, metric depth."""
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Depth Anything v2 requires `transformers`. "
            "Install with: pip install transformers"
        )
    from PIL import Image

    hf_size = _DA_V2_SIZE_MAP.get(model_size, model_size.capitalize())
    model_id = f"depth-anything/Depth-Anything-V2-{hf_size}-hf"
    CONSOLE.print(f"  Loading {model_id}...")
    pipe = pipeline("depth-estimation", model=model_id, device="cuda")

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        result = pipe(img)
        # result["depth"] is a PIL Image of predicted depth
        depth_pil = result["depth"]
        import numpy as np
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_np)
        # Normalize to reasonable range
        dmax = depth_tensor.max()
        if dmax > 0:
            depth_tensor = depth_tensor / dmax
        depths.append(depth_tensor.unsqueeze(-1))  # (H, W, 1)

    del pipe
    torch.cuda.empty_cache()
    return torch.stack(depths)  # (N, H, W, 1)


def _zoedepth(image_paths: list) -> Tensor:
    """ZoeDepth — metric depth estimation."""
    try:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    except Exception:
        raise ImportError(
            "ZoeDepth failed to load. Install with: "
            "pip install timm==0.6.7 && pip install git+https://github.com/isl-org/ZoeDepth"
        )
    from PIL import Image

    model = model.cuda().eval()

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        depth = model.infer_pil(img)  # safe — F.interpolate patched by caller
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(-1)
        depths.append(depth_tensor)  # (H, W, 1)

    del model
    torch.cuda.empty_cache()
    return torch.stack(depths)


def _midas(image_paths: list) -> Tensor:
    """MiDAS v3.1 — relative depth (widely supported fallback)."""
    try:
        model = torch.hub.load("intel-isl/MiDAS", "DPT_BEiT_L_384", pretrained=True)
    except Exception:
        raise ImportError(
            "MiDAS failed to load. Install with: "
            "pip install timm"
        )
    from PIL import Image
    import torchvision.transforms as T

    model = model.cuda().eval()
    transform = T.Compose([
        T.Resize(384),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    depths = []
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            CONSOLE.print(f"  Processing image {i+1}/{len(image_paths)}...")
        img = Image.open(path).convert("RGB")
        orig_h, orig_w = int(img.height), int(img.width)  # force Python int
        input_tensor = transform(img).unsqueeze(0).cuda()
        depth = model(input_tensor)
        # Resize back to original resolution (use Python ints!)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu()
        # MiDAS outputs inverse depth — invert
        depth = 1.0 / (depth + 1e-6)
        depth = depth / depth.max()
        depths.append(depth.float().unsqueeze(-1))

    del model
    torch.cuda.empty_cache()
    return torch.stack(depths)
