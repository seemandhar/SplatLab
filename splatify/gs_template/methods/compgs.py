"""
CompGS: Compact Gaussian Splatting with Vector Quantization
=============================================================
Paper: https://github.com/UCDvision/compact3d
Complexity: ★★★☆☆ (new params + VQ forward + loss)

Two contributions:
  1. Vector quantization (VQ) of Gaussian parameters using K-means
     clustering with Straight-Through Estimator (STE) for gradients.
  2. L1 opacity regularization to encourage zero-opacity Gaussians,
     enabling aggressive pruning for smaller models + faster rendering.

Training schedule (30K total iterations):
    0–15K:   Standard 3DGS training
    15K–20K: + opacity L1 regularization  (λ = 1e-7, prune every 1K iters)
    20K–30K: VQ-aware training with K-means quantization + STE
             - Assignments updated every 100 iters until 25K, then frozen
             - Centroids updated every iteration from float params

Four independent codebooks (quantized before nonlinearities):
    - Color DC component:       K=4096,  dim=3
    - Color SH coefficients:    K=4096,  dim=45 (degree 3, 15 bands × 3 channels)
    - Covariance scales:        K=16384, dim=3  (before exp activation)
    - Covariance rotations:     K=16384, dim=4  (before quaternion normalization)

NOT quantized: positions μ (sharing causes overlap), opacity σ (scalar).

Demonstrates:
    - populate_modules(): codebook buffers, assignment tracking, VQ state
    - get_outputs():      VQ in forward pass with STE (phases 20K–30K)
    - compute_custom_losses(): opacity L1 regularization (phases 15K–20K)

Train:
    ns-train compgs --data <path>
    ns-train compgs --data <path> --pipeline.model.opacity_reg_weight 2e-7

Ambiguities / notes:
    - Paper uses sum (not mean) for L1 opacity penalty; we replicate this.
    - Pruning is handled by splatfacto's existing culling mechanism +
      our lowered cull_alpha_thresh; the paper's custom every-1K pruning
      is approximated by setting refine_every and ensuring culling persists.
    - K-means initialization uses random subset (not K-means++); paper
      doesn't specify initialization method.
    - Centroid update uses scatter_add for efficiency on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import opacity_l1_loss
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
class CompGSModelConfig(GSTemplateModelConfig):
    """CompGS configuration: VQ + opacity regularization."""

    _target: Type = field(default_factory=lambda: CompGSModel)

    # ── Vector Quantization schedule ──────────────────────────────
    vq_start_iter: int = 20000
    """Iteration at which VQ-aware training begins (K-means init + STE)."""

    vq_freeze_iter: int = 25000
    """Iteration after which cluster assignments are frozen."""

    vq_assign_freq: int = 100
    """How often (in iterations) to update K-means assignments."""

    vq_kmeans_init_iters: int = 10
    """Number of full K-means iterations for initial codebook at vq_start_iter."""

    # ── Codebook sizes ────────────────────────────────────────────
    dc_codebook_size: int = 4096
    """Number of codes for color DC component (dim=3)."""

    sh_codebook_size: int = 4096
    """Number of codes for SH coefficients (dim=45 for degree 3)."""

    scale_codebook_size: int = 16384
    """Number of codes for scale parameters (dim=3)."""

    quat_codebook_size: int = 16384
    """Number of codes for rotation quaternions (dim=4)."""

    # ── Opacity regularization ────────────────────────────────────
    opacity_reg_weight: float = 1e-7
    """L1 penalty weight on sigmoid opacities (applied as sum, not mean)."""

    opacity_reg_start: int = 15000
    """Iteration to start opacity regularization."""

    opacity_reg_end: int = 20000
    """Iteration to stop opacity regularization (before VQ phase)."""

    # ── Pruning ───────────────────────────────────────────────────
    cull_alpha_thresh: float = 0.005
    """Lower threshold to cull more aggressively during opacity reg phase.
    (Overrides splatfacto default of ~0.1.)"""


# ═══════════════════════════════════════════════════════════════════
# Vector Quantization Utilities
# ═══════════════════════════════════════════════════════════════════


def _ste_quantize(params: Tensor, codebook: Tensor, assignments: Tensor) -> Tensor:
    """Straight-Through Estimator (STE) quantization.

    Forward: returns codebook[assignments] (quantized values).
    Backward: gradients pass through to `params` as if identity.

    This is the core trick from Bengio et al. (2013) [7 in paper].
    """
    quantized = codebook[assignments]
    # STE: forward = quantized, backward = d(output)/d(params) = 1
    return params + (quantized - params).detach()


@torch.no_grad()
def _compute_assignments_batched(
    data: Tensor,
    centroids: Tensor,
    batch_size: int = 32768,
) -> Tensor:
    """Compute nearest-centroid assignments in batches to avoid OOM.

    Uses expanded squared distance: ||a-b||² = ||a||² + ||b||² - 2a·b

    Args:
        data:      (N, D) parameter vectors
        centroids: (K, D) codebook centroids
        batch_size: process this many Gaussians at a time

    Returns:
        (N,) long tensor of centroid indices
    """
    N = data.shape[0]
    assignments = torch.empty(N, dtype=torch.long, device=data.device)

    # Precompute ||centroid||²
    cent_sq = (centroids ** 2).sum(dim=-1).unsqueeze(0)  # (1, K)

    for i in range(0, N, batch_size):
        batch = data[i : i + batch_size]
        batch_sq = (batch ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
        # (B, K) squared distances
        dists = batch_sq + cent_sq - 2.0 * batch @ centroids.t()
        assignments[i : i + batch_size] = dists.argmin(dim=-1)

    return assignments


@torch.no_grad()
def _update_centroids_scatter(
    data: Tensor,
    assignments: Tensor,
    K: int,
    old_centroids: Tensor,
) -> Tensor:
    """Update centroids by averaging assigned data points.

    Uses scatter_add for GPU-efficient aggregation.
    Empty clusters retain their previous centroid.

    Args:
        data:           (N, D) parameter vectors
        assignments:    (N,) cluster indices
        K:              number of clusters
        old_centroids:  (K, D) previous centroids (fallback for empty clusters)

    Returns:
        (K, D) updated centroids
    """
    D = data.shape[-1]
    new_centroids = torch.zeros(K, D, device=data.device, dtype=data.dtype)
    counts = torch.zeros(K, device=data.device, dtype=data.dtype)

    # Accumulate
    idx_expand = assignments.unsqueeze(-1).expand_as(data)
    new_centroids.scatter_add_(0, idx_expand, data)
    counts.scatter_add_(
        0, assignments, torch.ones(data.shape[0], device=data.device, dtype=data.dtype)
    )

    # Average (avoid div-by-zero for empty clusters)
    mask = counts > 0
    new_centroids[mask] /= counts[mask].unsqueeze(-1)
    # Keep old centroids for empty clusters
    new_centroids[~mask] = old_centroids[~mask]

    return new_centroids


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════


# Names and metadata for the 4 VQ groups
_VQ_GROUPS = [
    # (group_name, param_source, codebook_size_attr, flatten)
    # flatten=True means we reshape (N, *, C) → (N, *C) before VQ
    ("dc", "features_dc", "dc_codebook_size", False),
    ("sh", "features_rest", "sh_codebook_size", True),
    ("scale", "scales", "scale_codebook_size", False),
    ("quat", "quats", "quat_codebook_size", False),
]


class CompGSModel(GSTemplateModel):
    """CompGS: Compact Gaussian Splatting with Vector Quantization.

    Extends GSTemplateModel with:
    - K-means VQ of 4 parameter groups (DC, SH, scale, rotation)
    - STE for gradient propagation through quantization
    - L1 opacity regularization for Gaussian count reduction
    """

    config: CompGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Initialize base modules + VQ state (codebooks & assignments)."""
        super().populate_modules()

        self._vq_initialized = False

        # Pre-register codebook buffers and assignment buffers.
        # They will be properly sized at vq_start_iter when we know
        # the actual parameter dimensions and Gaussian count.
        # Using register_buffer ensures they're saved in checkpoints.
        for group_name, _, _, _ in _VQ_GROUPS:
            self.register_buffer(
                f"_cb_{group_name}",
                torch.empty(0),
                persistent=True,
            )
            self.register_buffer(
                f"_assign_{group_name}",
                torch.empty(0, dtype=torch.long),
                persistent=True,
            )

    # ── VQ Internals ───────────────────────────────────────────────

    def _get_param_flat(self, param_name: str, flatten: bool) -> Tensor:
        """Get a Gaussian parameter, optionally flattening multi-dim to 2D.

        For SH: (N, num_bands, 3) → (N, num_bands*3)
        For others: (N, D) → (N, D) unchanged
        """
        p = self.gauss_params[param_name]
        if flatten and p.dim() > 2:
            return p.reshape(p.shape[0], -1)
        return p

    def _initialize_vq(self):
        """One-time K-means initialization at vq_start_iter.

        Runs full K-means (multiple iterations) on each parameter group
        to produce initial codebooks and assignments.
        """
        self._vq_initialized = True
        n_init = self.config.vq_kmeans_init_iters

        for group_name, param_name, cb_size_attr, flatten in _VQ_GROUPS:
            K = getattr(self.config, cb_size_attr)
            data = self._get_param_flat(param_name, flatten).detach()
            N, D = data.shape

            # Handle case where K > N (more codes than Gaussians)
            K_eff = min(K, N)

            # Initialize centroids by random sampling
            perm = torch.randperm(N, device=data.device)[:K_eff]
            centroids = data[perm].clone()

            # Pad with zeros if K_eff < K
            if K_eff < K:
                pad = torch.zeros(K - K_eff, D, device=data.device, dtype=data.dtype)
                centroids = torch.cat([centroids, pad], dim=0)

            # Run K-means iterations
            for _ in range(n_init):
                assignments = _compute_assignments_batched(data, centroids)
                centroids = _update_centroids_scatter(
                    data, assignments, K, centroids
                )

            # Store
            setattr(self, f"_cb_{group_name}", centroids)
            setattr(self, f"_assign_{group_name}", assignments)

    def _update_kmeans(self):
        """Periodic K-means update (called every training step during VQ phase).

        - Centroids: updated EVERY iteration (cheap: scatter average).
        - Assignments: updated every vq_assign_freq iters until vq_freeze_iter.
        """
        step = self.step

        for group_name, param_name, cb_size_attr, flatten in _VQ_GROUPS:
            K = getattr(self.config, cb_size_attr)
            data = self._get_param_flat(param_name, flatten).detach()
            cb = getattr(self, f"_cb_{group_name}")
            assign = getattr(self, f"_assign_{group_name}")

            # ── Update centroids (every iteration) ────────────────
            new_cb = _update_centroids_scatter(data, assign, K, cb)
            setattr(self, f"_cb_{group_name}", new_cb)

            # ── Update assignments (every vq_assign_freq, unfrozen) ─
            if step < self.config.vq_freeze_iter:
                steps_since_vq = step - self.config.vq_start_iter
                if steps_since_vq % self.config.vq_assign_freq == 0:
                    new_assign = _compute_assignments_batched(data, new_cb)
                    setattr(self, f"_assign_{group_name}", new_assign)

    def _apply_vq_ste(
        self, params: Tensor, group_name: str, flatten: bool, crop_ids: Optional[Tensor]
    ) -> Tensor:
        """Apply VQ with STE to a parameter tensor.

        Args:
            params:     (N, D) raw parameter values (from gauss_params)
            group_name: one of 'dc', 'sh', 'scale', 'quat'
            flatten:    whether params were flattened from higher dim
            crop_ids:   optional crop indices for eval-time cropping

        Returns:
            (N, D) quantized parameter values with STE gradient path
        """
        cb = getattr(self, f"_cb_{group_name}").to(params.device)
        assign = getattr(self, f"_assign_{group_name}").to(params.device)

        if crop_ids is not None:
            assign = assign[crop_ids]

        if flatten and params.dim() > 2:
            orig_shape = params.shape
            params_flat = params.reshape(params.shape[0], -1)
            quantized_flat = _ste_quantize(params_flat, cb, assign)
            return quantized_flat.reshape(orig_shape)
        else:
            return _ste_quantize(params, cb, assign)

    # ── 2. get_outputs (VQ-aware forward pass) ─────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render with optional VQ quantization.

        Before vq_start_iter: delegates to standard splatfacto rendering.
        After vq_start_iter:  applies VQ+STE to DC, SH, scales, quats,
                              then rasterizes with quantized parameters.
        """
        # ── K-means update (training only) ────────────────────────
        if self.training and self._vq_initialized:
            self._update_kmeans()
        elif (
            self.training
            and not self._vq_initialized
            and self.step >= self.config.vq_start_iter
        ):
            self._initialize_vq()

        # ── Decide whether to use VQ ──────────────────────────────
        use_vq = self._vq_initialized and self.step >= self.config.vq_start_iter

        if not use_vq:
            # Standard splatfacto rendering (phases 0–20K)
            return super().get_outputs(camera)

        # ── VQ-aware forward pass (phases 20K–30K) ────────────────
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids  # empty outputs from crop box

        # Gather raw base parameters
        means, quats_raw, scales_raw, opacities, colors_raw = gather_base_params(
            self, crop_ids
        )

        # Get the raw features for VQ (before they got assembled into colors)
        if crop_ids is not None:
            features_dc = self.features_dc[crop_ids]
            features_rest = self.features_rest[crop_ids]
        else:
            features_dc = self.features_dc
            features_rest = self.features_rest

        # ── Apply VQ+STE to each group ────────────────────────────
        # DC: (N, 3)
        dc_q = self._apply_vq_ste(features_dc, "dc", False, crop_ids)
        # SH: (N, num_bands, 3) → flatten → VQ → unflatten
        sh_q = self._apply_vq_ste(features_rest, "sh", True, crop_ids)
        # Scale: (N, 3) — quantized BEFORE exp activation
        scales_q = self._apply_vq_ste(scales_raw, "scale", False, crop_ids)
        # Quat: (N, 4) — quantized BEFORE normalization
        quats_q = self._apply_vq_ste(quats_raw, "quat", False, crop_ids)

        # Reassemble colors from quantized DC + SH
        colors_q = torch.cat((dc_q[:, None, :], sh_q), dim=1)

        # Determine SH degree (progressive training)
        if colors_q.dim() == 2:
            sh_degree = None
        else:
            sh_degree = min(
                self.step // self.config.sh_degree_interval,
                self.config.sh_degree,
            )

        # ── Rasterize with quantized parameters ──────────────────
        render, alpha, info = rasterize_gaussians(
            self,
            means,
            quats_q,
            scales_q,
            opacities,
            colors_q,
            cam["viewmat"],
            cam["K"],
            cam["W"],
            cam["H"],
            sh_degree=sh_degree,
        )

        # ── Assemble output dict ──────────────────────────────────
        outputs = assemble_outputs(self, render, alpha, info, cam)
        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Add opacity L1 regularization during the scheduled window.

        Active from opacity_reg_start to opacity_reg_end (default 15K–20K).
        Uses sum reduction (not mean) per the paper: L = λ * Σ σ_i
        """
        losses = {}

        step = self.step
        if self.config.opacity_reg_start <= step < self.config.opacity_reg_end:
            losses["opacity_l1"] = opacity_l1_loss(
                self.opacities, weight=self.config.opacity_reg_weight
            )

        return losses

    # ── 4. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Standard Gaussian param groups (no new Gaussian params added)."""
        return super().get_gaussian_param_groups()

    # ── 5. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add VQ-specific metrics for monitoring."""
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        if self._vq_initialized:
            # Report codebook utilization (fraction of codes assigned)
            for group_name, _, cb_size_attr, _ in _VQ_GROUPS:
                K = getattr(self.config, cb_size_attr)
                assign = getattr(self, f"_assign_{group_name}")
                if assign.numel() > 0:
                    n_used = assign.unique().numel()
                    metrics[f"vq_{group_name}_utilization"] = torch.tensor(
                        n_used / K, dtype=torch.float32
                    )
        return metrics

    # ── 6. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        """Handle loading checkpoints with VQ buffers.

        The VQ codebook and assignment buffers are registered as empty
        tensors in populate_modules() (since their size depends on
        training state). When loading a checkpoint that contains
        trained VQ state, we must resize these buffers to match the
        checkpoint BEFORE calling super().load_state_dict(), otherwise
        PyTorch raises a size mismatch error.
        """
        # Resize VQ buffers to match checkpoint shapes
        for group_name, _, _, _ in _VQ_GROUPS:
            cb_key = f"_cb_{group_name}"
            assign_key = f"_assign_{group_name}"

            if cb_key in dict and dict[cb_key].numel() > 0:
                self._vq_initialized = True
                # Resize codebook buffer to match checkpoint
                self.register_buffer(
                    cb_key,
                    torch.zeros_like(dict[cb_key]),
                    persistent=True,
                )
                # Resize assignment buffer to match checkpoint
                self.register_buffer(
                    assign_key,
                    torch.zeros_like(dict[assign_key]),
                    persistent=True,
                )

        # Also handle gauss_params resizing (same pattern as FeatureGS)
        if "gauss_params.means" in dict:
            newp = dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                new_shape = (newp,) + param.shape[1:]
                self.gauss_params[name] = Parameter(
                    torch.zeros(new_shape, device=param.device)
                )

        super().load_state_dict(dict, **kwargs)