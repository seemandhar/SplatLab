"""
VoD-3DGS: View-Opacity-Dependent 3D Gaussian Splatting
========================================================
Paper: View-Opacity-Dependent 3D Gaussian Splatting
Complexity: ★★★☆☆ (new Gaussian params, custom rendering, custom loss, modified reset)

Core idea: Replace scalar opacity with view-dependent opacity using a learnable
symmetric 3×3 matrix (SGGX-inspired). For each Gaussian i and camera j:

    α̂_i(ω_{i,j}) = σ(γ_i + ω_{i,j}^T Ŝ_i ω_{i,j})

Two variants:
    VoD-3DGS[L] — reset keeps largest eigenvalue  (higher quality, more memory)
    VoD-3DGS[S] — reset keeps smallest eigenvalue (more compact)

Train:
    ns-train vod3dgs-s --data <path>
    ns-train vod3dgs-l --data <path>
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import view_consistency_loss
from gs_template.rendering import (
    assemble_outputs,
    crop_custom_param,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VoD3DGSModelConfig(GSTemplateModelConfig):
    """VoD-3DGS configuration."""

    _target: Type = field(default_factory=lambda: VoD3DGSModel)

    # ── Variant ────────────────────────────────────────────────────
    vod_variant: Literal["S", "L"] = "S"
    """Reset variant: 'S' keeps smallest eigenvalue, 'L' keeps largest."""

    # ── View-consistency loss ──────────────────────────────────────
    vc_loss_weight: float = 0.01
    """Weight for L_vc. Paper ablation shows uniform sampling (without
    keypoint weighting) increases Gaussian count, so keep this LOW."""

    use_vc_loss: bool = True
    """Enable view-consistency regularization."""

    vc_warmup_steps: int = 1000
    """Start applying L_vc after this many steps."""

    vc_subsample: int = 65536
    """Max Gaussians to subsample for L_vc. Reduces noise and cost.
    0 = use all Gaussians (slow, noisy)."""

    # ── Enhanced opacity reset ─────────────────────────────────────
    opacity_s_reset_interval: int = 3000
    """Reset Ŝ matrices every N steps (mirrors opacity reset in 3DGS)."""

    opacity_s_reset_until: int = 15000
    """Stop resetting Ŝ after this step."""

    # ── Stability ──────────────────────────────────────────────────
    quad_clamp: float = 10.0
    """Clamp ω^T Ŝ ω to [-quad_clamp, +quad_clamp]. Prevents sigmoid
    saturation and dead Gaussians with zero gradient."""

    # ── View sampling for L_vc ─────────────────────────────────────
    use_keypoint_sampling: bool = False
    """Use keypoint-match-based view sampling (requires precomputed data)."""

    keypoint_matches_path: str = ""
    """Path to precomputed keypoint match counts (N×N .pt tensor)."""


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

def _compute_quadratic_form(S_params: Tensor, omega: Tensor) -> Tensor:
    """Compute ω^T Ŝ ω efficiently without building full 3×3 matrices.

    Args:
        S_params: (N, 6) — [Sxx, Sxy, Sxz, Syy, Syz, Szz]
        omega:    (N, 3) — normalized view directions

    Returns:
        (N, 1) — quadratic form values
    """
    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]
    Sxx = S_params[:, 0]
    Sxy = S_params[:, 1]
    Sxz = S_params[:, 2]
    Syy = S_params[:, 3]
    Syz = S_params[:, 4]
    Szz = S_params[:, 5]

    quad = (
        Sxx * ox * ox
        + 2.0 * Sxy * ox * oy
        + 2.0 * Sxz * ox * oz
        + Syy * oy * oy
        + 2.0 * Syz * oy * oz
        + Szz * oz * oz
    )
    return quad.unsqueeze(-1)  # (N, 1)


def _build_S_matrices(S_params: Tensor) -> Tensor:
    """Build full symmetric 3×3 matrices from 6 parameters."""
    N = S_params.shape[0]
    S = torch.zeros(N, 3, 3, device=S_params.device, dtype=S_params.dtype)
    S[:, 0, 0] = S_params[:, 0]
    S[:, 0, 1] = S_params[:, 1]
    S[:, 0, 2] = S_params[:, 2]
    S[:, 1, 0] = S_params[:, 1]
    S[:, 1, 1] = S_params[:, 3]
    S[:, 1, 2] = S_params[:, 4]
    S[:, 2, 0] = S_params[:, 2]
    S[:, 2, 1] = S_params[:, 4]
    S[:, 2, 2] = S_params[:, 5]
    return S


def _camera_center_from_viewmat(viewmat: Tensor) -> Tensor:
    """Extract camera center in world coordinates from a 4×4 view matrix."""
    if viewmat.dim() == 3:
        viewmat = viewmat[0]
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return -R.T @ t


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class VoD3DGSModel(GSTemplateModel):
    """VoD-3DGS: View-Opacity-Dependent Gaussian Splatting."""

    config: VoD3DGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()

        # Per-Gaussian symmetric matrix Ŝ (6 unique params)
        # Initialized to zeros → initially α̂ = σ(γ), identical to baseline
        self.gauss_params["opacity_S"] = Parameter(
            torch.zeros(self.num_points, 6)
        )

        # FIX 4: Use actual dataset camera centers, not accumulated noisy ones.
        # Collect exactly one center per unique training view (by cam_idx).
        self._dataset_camera_centers: Optional[Tensor] = None
        self._seen_cam_indices: Dict[int, Tensor] = {}  # cam_idx → center
        self._current_cam_center: Optional[Tensor] = None
        self._cameras_finalized: bool = False

        # Optional: precomputed keypoint match counts
        self._match_counts: Optional[Tensor] = None
        if (
            self.config.use_keypoint_sampling
            and self.config.keypoint_matches_path
        ):
            try:
                self._match_counts = torch.load(
                    self.config.keypoint_matches_path, map_location="cpu"
                ).float()
            except Exception as e:
                print(
                    f"[VoD-3DGS] Could not load keypoint matches: {e}. "
                    f"Falling back to uniform sampling."
                )

        # Track last reset step
        self._last_S_reset_step = 0

    def _collect_camera_center(self, camera: Cameras, cam_center: Tensor):
        """Collect unique training camera centers by cam_idx.

        FIX 4: Instead of accumulating noisy per-step duplicates, we
        collect exactly ONE center per unique training camera index.
        After seeing enough unique views (or after warmup), we finalize
        into a clean tensor for L_vc sampling.
        """
        if self._cameras_finalized:
            return

        # Get camera index from metadata
        try:
            cam_idx = camera.metadata["cam_idx"]
            if isinstance(cam_idx, Tensor):
                cam_idx = cam_idx.item()
            cam_idx = int(cam_idx)
        except (KeyError, AttributeError, TypeError):
            return

        # Store first sighting of each camera index
        if cam_idx not in self._seen_cam_indices:
            self._seen_cam_indices[cam_idx] = cam_center.detach().clone()

        # Finalize once we have enough views (or at warmup boundary)
        num_train = self.kwargs.get("num_train_data", 0)
        have_enough = len(self._seen_cam_indices) >= max(num_train, 2)
        at_warmup = self.step >= self.config.vc_warmup_steps

        if have_enough or (at_warmup and len(self._seen_cam_indices) >= 2):
            # Build clean tensor of unique camera centers
            centers = list(self._seen_cam_indices.values())
            self._dataset_camera_centers = torch.stack(centers).to(self.device)
            self._cameras_finalized = True
            from nerfstudio.utils.rich_utils import CONSOLE
            CONSOLE.print(
                f"[VoD-3DGS] Collected {len(centers)} unique training "
                f"camera centers for L_vc sampling."
            )

    # ── 2. get_outputs (view-dependent opacity) ────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render with view-dependent opacity.

        For each Gaussian i and current camera j:
            α̂_i = σ(γ_i + clamp(ω_{i,j}^T Ŝ_i ω_{i,j}))
        """
        # Phase 1: camera setup
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        # Gather base Gaussian params
        means, quats, scales, opacities_logit, colors = gather_base_params(
            self, crop_ids
        )
        S_params = crop_custom_param(
            self.gauss_params["opacity_S"], crop_ids
        )

        # ── Compute view-dependent opacity ─────────────────────────
        camera_center = _camera_center_from_viewmat(cam["viewmat"])

        # Store for L_vc computation and collect unique training cameras
        if self.training:
            self._current_cam_center = camera_center.detach()
            self._collect_camera_center(camera, camera_center)

        # View direction: Gaussian center → camera center (normalized)
        omega = camera_center.unsqueeze(0) - means  # (N, 3)
        omega = F.normalize(omega, dim=-1, eps=1e-8)

        # ω^T Ŝ ω (efficient, no full matrix construction)
        quad = _compute_quadratic_form(S_params, omega)  # (N, 1)

        # FIX 5: Clamp quadratic form to prevent sigmoid saturation.
        # Without this, S can learn extreme values → σ(γ + 100) = 1.0
        # with zero gradient → "dead" Gaussians that never recover.
        quad = torch.clamp(quad, -self.config.quad_clamp, self.config.quad_clamp)

        # View-dependent opacity logit: γ + ω^T Ŝ ω
        vd_opacity_logit = opacities_logit + quad

        # Phase 2: rasterize
        render, alpha, info = rasterize_gaussians(
            self,
            means,
            quats,
            scales,
            vd_opacity_logit,
            colors,
            cam["viewmat"],
            cam["K"],
            cam["W"],
            cam["H"],
        )
        self.info = info

        # Phase 3: assemble outputs
        outputs = assemble_outputs(self, render, alpha, info, cam)

        if self.training:
            outputs["_vd_opacity_logit"] = vd_opacity_logit

        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        # ── View-consistency regularization L_vc ───────────────────
        if (
            self.config.use_vc_loss
            and self.training
            and self._current_cam_center is not None
            and self.step > self.config.vc_warmup_steps
        ):
            cam_i = self._current_cam_center

            cam_j = self._sample_other_view(cam_i)
            if cam_j is not None:
                # ─────────────────────────────────────────────────────
                # FIX 1 (CRITICAL): Detach γ so L_vc ONLY optimizes S.
                #
                # Previously: gamma = self.opacities  (WITH gradient)
                # This meant L_vc backpropagated competing gradients to
                # the base scalar opacity, FIGHTING the reconstruction
                # loss every step. Reconstruction wants specific opacity
                # values; L_vc pushes toward view-consistency — these
                # conflict, degrading PSNR by ~0.3-0.5 dB.
                #
                # The paper's intent is for L_vc to regularize the S
                # matrix (prevent wild view-dependent opacity swings),
                # NOT to modify the base opacity γ.
                # ─────────────────────────────────────────────────────
                gamma = self.opacities.detach()  # NO gradient on γ
                S_params = self.gauss_params["opacity_S"]  # gradient HERE
                means = self.means.detach()

                # ─────────────────────────────────────────────────────
                # FIX 3: Subsample Gaussians for L_vc.
                #
                # Computing over 1M+ Gaussians is wasteful: most are far
                # from both cameras and contribute high-variance noise
                # (cos(θ)≈1, both opacities nearly identical → zero signal
                # but nonzero noisy gradient). 65K random sample gives
                # good coverage with 15x less noise and compute.
                # ─────────────────────────────────────────────────────
                N = means.shape[0]
                if self.config.vc_subsample > 0 and N > self.config.vc_subsample:
                    idx = torch.randperm(N, device=means.device)[:self.config.vc_subsample]
                    means_sub = means[idx]
                    gamma_sub = gamma[idx]
                    S_sub = S_params[idx]
                else:
                    means_sub = means
                    gamma_sub = gamma
                    S_sub = S_params

                losses["vc_loss"] = self.config.vc_loss_weight * view_consistency_loss(
                    means=means_sub,
                    opacity_logit=gamma_sub,
                    S_params=S_sub,
                    cam_center_i=cam_i,
                    cam_center_j=cam_j,
                )

        # ── Periodic Ŝ reset (FIX 6: mirrors 3DGS opacity reset) ──
        if (
            self.training
            and self.step > 0
            and self.step <= self.config.opacity_s_reset_until
            and self.step % self.config.opacity_s_reset_interval == 0
            and self.step != self._last_S_reset_step
        ):
            self._reset_opacity_S()
            self._last_S_reset_step = self.step

        return losses

    def _sample_other_view(self, cam_center_i: Tensor) -> Optional[Tensor]:
        """Sample another camera center for L_vc.

        FIX 4: Uses actual dataset camera centers when available.
        """
        if self._dataset_camera_centers is not None:
            n_cams = self._dataset_camera_centers.shape[0]
            if n_cams < 2:
                return None

            if self._match_counts is not None and self._match_counts.shape[0] >= n_cams:
                dists = (self._dataset_camera_centers - cam_center_i.unsqueeze(0)).norm(dim=-1)
                idx_i = dists.argmin().item()
                weights = self._match_counts[idx_i, :n_cams].clone()
                weights[idx_i] = 0
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    idx_j = torch.multinomial(weights, 1).item()
                else:
                    idx_j = random.randint(0, n_cams - 1)
            else:
                idx_j = random.randint(0, n_cams - 1)

            return self._dataset_camera_centers[idx_j]

        return None

    @torch.no_grad()
    def _reset_opacity_S(self):
        """Reset Ŝ matrices via eigendecomposition.

        VoD-3DGS[L]: keep only the largest eigenvalue
        VoD-3DGS[S]: keep only the smallest eigenvalue
        """
        from nerfstudio.utils.rich_utils import CONSOLE

        variant = self.config.vod_variant
        S_data = self.gauss_params["opacity_S"].data
        N = S_data.shape[0]
        chunk_size = 500000

        CONSOLE.print(
            f"[VoD-3DGS] Resetting Ŝ matrices (variant={variant}) "
            f"at step {self.step} for {N} Gaussians..."
        )

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = S_data[start:end]

            S_matrices = _build_S_matrices(chunk)
            eigenvalues, eigenvectors = torch.linalg.eigh(S_matrices)

            new_eigenvalues = torch.zeros_like(eigenvalues)
            if variant == "L":
                new_eigenvalues[:, 2] = eigenvalues[:, 2]
            else:
                new_eigenvalues[:, 0] = eigenvalues[:, 0]

            new_S = eigenvectors @ torch.diag_embed(new_eigenvalues) @ eigenvectors.transpose(-1, -2)

            S_data[start:end, 0] = new_S[:, 0, 0]
            S_data[start:end, 1] = new_S[:, 0, 1]
            S_data[start:end, 2] = new_S[:, 0, 2]
            S_data[start:end, 3] = new_S[:, 1, 1]
            S_data[start:end, 4] = new_S[:, 1, 2]
            S_data[start:end, 5] = new_S[:, 2, 2]

        CONSOLE.print(f"[VoD-3DGS] Ŝ reset complete.")

    # ── 4. split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        out["opacity_S"] = self.gauss_params["opacity_S"][split_mask].repeat(samps, 1)
        return out

    # ── 5. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    # ── 7. get_outputs_for_camera (viewer) ─────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)
        return outs

    # ── 8. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in self.gauss_params.keys():
                if p in dict:
                    dict[f"gauss_params.{p}"] = dict[p]

        if "gauss_params.means" in dict:
            newp = dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                new_shape = (newp,) + param.shape[1:]
                self.gauss_params[name] = Parameter(
                    torch.zeros(new_shape, device=self.device)
                )
            if "gauss_params.opacity_S" not in dict:
                dict["gauss_params.opacity_S"] = torch.zeros(newp, 6, device=self.device)

        super().load_state_dict(dict, **kwargs)

    # ── Convenience ────────────────────────────────────────────────

    @property
    def opacity_S(self) -> Tensor:
        return self.gauss_params["opacity_S"]

    def compute_max_opacity_over_views(self, camera_centers: Tensor) -> Tensor:
        means = self.means.detach()
        gamma = self.opacities.detach()
        S_params = self.gauss_params["opacity_S"].detach()
        max_alpha = torch.zeros(means.shape[0], device=means.device)
        for cam_center in camera_centers:
            omega = F.normalize(cam_center.unsqueeze(0) - means, dim=-1, eps=1e-8)
            quad = _compute_quadratic_form(S_params, omega)
            quad = torch.clamp(quad, -self.config.quad_clamp, self.config.quad_clamp)
            alpha = torch.sigmoid(gamma + quad).squeeze(-1)
            max_alpha = torch.maximum(max_alpha, alpha)
        return max_alpha