"""
Opti3DGS — Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation
============================================================================================
Paper: Opti3DGS
Complexity: ★★☆☆☆ (custom loss override + image filtering — no renderer changes)

Core idea:
    Progressively increase image frequency content during training by applying
    a decaying low-pass (mean blur) filter to the ground-truth training images.
    Early iterations see only coarse structure → densification creates fewer,
    larger Gaussians. Later iterations see full detail → refinement adds
    necessary detail without over-densification.

Key properties:
    - Does NOT change the differentiable renderer or ADC logic
    - Only modifies the GT images used for loss computation
    - Mean blur filter with step-decay schedule over 5 levels
    - Filtering stops after 12,000 iterations; remaining 18,000 use original images
    - Reduces Gaussian count by up to 62%, GPU memory by up to 40%

Demonstrates:
    - Overriding get_loss_dict() to modify batch images before loss computation
    - Step-decay scheduling for kernel size
    - Config fields for filter hyperparameters

Train:
    ns-train opti3dgs --data <path>
    ns-train opti3dgs --data <path> --pipeline.model.initial_kernel_size 21
    ns-train opti3dgs --data <path> --pipeline.model.num_filter_levels 7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Image Frequency Modulation Utilities
# ═══════════════════════════════════════════════════════════════════

def mean_blur(image: Tensor, kernel_size: int) -> Tensor:
    """Apply mean (box) blur to an image tensor.

    Implements the low-pass filter from Eq. 2 of the paper:
        G[i,j] = Σ_u Σ_v H[u,v] * F[i-u, j-v]
    where H is a uniform kernel (mean filter).

    Args:
        image: (H, W, C) image tensor, values in [0, 1]
        kernel_size: odd integer, size of the mean filter kernel

    Returns:
        (H, W, C) blurred image, same spatial dimensions (padding='same')
    """
    if kernel_size <= 1:
        return image

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    C = image.shape[-1]
    # Reshape to (1, C, H, W) for F.avg_pool2d
    img = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Padding to maintain spatial dimensions (stride=1 as per paper)
    pad = kernel_size // 2
    blurred = F.avg_pool2d(img, kernel_size=kernel_size, stride=1, padding=pad)

    return blurred.squeeze(0).permute(1, 2, 0)  # back to (H, W, C)


def gaussian_blur(image: Tensor, kernel_size: int, sigma: float = 0.0) -> Tensor:
    """Apply Gaussian blur (alternative filter, not default).

    Args:
        image: (H, W, C)
        kernel_size: odd integer
        sigma: Gaussian sigma (0 = auto: 0.3*((k-1)*0.5 - 1) + 0.8)
    """
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
    coords -= kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()

    # Separable 2D convolution
    C = image.shape[-1]
    img = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    pad = kernel_size // 2

    # Horizontal pass
    kernel_h = g.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    img = F.conv2d(img, kernel_h, padding=(0, pad), groups=C)
    # Vertical pass
    kernel_v = g.view(1, 1, -1, 1).expand(C, -1, -1, -1)
    img = F.conv2d(img, kernel_v, padding=(pad, 0), groups=C)

    return img.squeeze(0).permute(1, 2, 0)


def compute_step_kernel_size(
    step: int,
    initial_kernel_size: int,
    num_levels: int,
    filter_stop_iter: int,
) -> int:
    """Compute the current kernel size using the step-decay schedule.

    The paper uses a step schedule where the kernel size decreases in
    discrete steps across `num_levels` levels. Each level maintains a
    fixed kernel size, allowing ADC to optimize under a stable regime.

    The progressive growing pipeline (Eq. 3-5):
        G_0 = SfM(I)
        G_n = 3DGS(G_{n-1}, I_n)   for n in {1,...,4}
        G_5 = 3DGS(G_4, I)         (final level: unfiltered)

    Args:
        step: current training iteration
        initial_kernel_size: kernel size at level 1 (default: 15)
        num_levels: number of filtering levels (default: 5)
        filter_stop_iter: iteration at which filtering stops (default: 12000)

    Returns:
        Current kernel size (1 = no filtering)
    """
    if step >= filter_stop_iter:
        return 1  # No filtering after stop iteration

    # Which level are we in?
    iters_per_level = filter_stop_iter // num_levels
    current_level = min(step // iters_per_level, num_levels - 1)

    # Linearly interpolate kernel sizes across levels
    # Level 0 → initial_kernel_size, Level (num_levels-1) → 3 (minimal blur)
    min_kernel = 3
    if num_levels <= 1:
        return initial_kernel_size

    kernel_size = int(
        initial_kernel_size
        - (initial_kernel_size - min_kernel) * current_level / (num_levels - 1)
    )

    # Ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return max(kernel_size, 1)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Opti3DGSModelConfig(GSTemplateModelConfig):
    """Opti3DGS configuration.

    Key hyperparameters from the paper:
        - initial_kernel_size: 15 (larger → fewer Gaussians, quality plateaus)
        - filter_stop_iter: 12000 (of 30000 total)
        - num_filter_levels: 5 (step schedule)
        - blur_type: "mean" (best overall per ablations)
    """

    _target: Type = field(default_factory=lambda: Opti3DGSModel)

    # ── Frequency modulation settings ─────────────────────────────

    initial_kernel_size: int = 15
    """Initial mean-blur kernel size. Larger = more aggressive coarsening.
    Paper ablation: quality improvements plateau quickly with larger sizes.
    Default 15 selected from ablations."""

    filter_stop_iter: int = 12000
    """Iteration at which frequency modulation stops. After this,
    training continues with unfiltered (original) images for the
    remaining iterations."""

    num_filter_levels: int = 5
    """Number of discrete filtering levels in the step schedule.
    Each level uses a fixed kernel size for (filter_stop_iter / num_levels)
    iterations, allowing ADC to optimize under a stable filtering regime."""

    blur_type: Literal["mean", "gaussian"] = "mean"
    """Low-pass filter type. Paper ablation: mean filter performs best.
    Gaussian and bilateral performed poorly in reported tests."""

    # ── Modified 3DGS hyperparameters (from paper Section Experiments) ──

    # These are set in configs.py via optimizer config, but we document
    # the paper's recommended values here:
    # scaling_lr = 0.01  (vs default 0.005)
    # densification_interval = 500  (vs default 100)
    # densify_grad_threshold = 0.0001  (vs default 0.0002)


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class Opti3DGSModel(GSTemplateModel):
    """Opti3DGS — Coarse-to-Fine Image Frequency Modulation.

    The only override is get_loss_dict(), where we apply the low-pass
    filter to the ground-truth batch images before computing the
    standard photometric loss. This causes densification to first fit
    coarse structure (large Gaussians) and later refine details.

    No changes to:
        - Differentiable renderer
        - Adaptive density control (ADC)
        - Gaussian splitting/cloning logic
        - Any other training component
    """

    config: Opti3DGSModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Apply frequency modulation to GT images, then compute standard losses.

        This is the core mechanism of Opti3DGS: by blurring the target
        images early in training, the photometric gradients only respond
        to low-frequency structure, causing ADC to create fewer, larger
        Gaussians. As training progresses, higher frequencies are revealed
        and finer detail is added.
        """
        # Compute current kernel size from step schedule
        kernel_size = compute_step_kernel_size(
            step=self.step,
            initial_kernel_size=self.config.initial_kernel_size,
            num_levels=self.config.num_filter_levels,
            filter_stop_iter=self.config.filter_stop_iter,
        )

        # Apply low-pass filter to GT image if we're still in the
        # frequency modulation phase
        if kernel_size > 1 and "image" in batch:
            gt_image = batch["image"]  # (H, W, 3)

            # DataManager may cache images as uint8 — blur requires float
            original_dtype = gt_image.dtype
            if gt_image.dtype == torch.uint8:
                gt_image = gt_image.float() / 255.0

            if self.config.blur_type == "mean":
                blurred = mean_blur(gt_image, kernel_size)
            elif self.config.blur_type == "gaussian":
                blurred = gaussian_blur(gt_image, kernel_size)
            else:
                blurred = gt_image

            # Cast back if needed (though splatfacto loss expects float anyway)
            if original_dtype == torch.uint8:
                blurred = (blurred * 255).clamp(0, 255).to(torch.uint8)

            batch["image"] = blurred

        # Standard splatfacto loss (L1 + SSIM) on the (possibly blurred) GT
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add Opti3DGS-specific metrics for monitoring."""
        metrics = super().get_metrics_dict(outputs, batch)

        # Log current kernel size for debugging
        kernel_size = compute_step_kernel_size(
            step=self.step,
            initial_kernel_size=self.config.initial_kernel_size,
            num_levels=self.config.num_filter_levels,
            filter_stop_iter=self.config.filter_stop_iter,
        )
        metrics["blur_kernel_size"] = torch.tensor(
            kernel_size, dtype=torch.float32
        )

        # Log Gaussian count
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )

        return metrics