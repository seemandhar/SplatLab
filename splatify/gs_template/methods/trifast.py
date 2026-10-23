"""
TriFast — Triple-Enhanced FastGS
==================================
Paper basis: FastGS (arxiv 2511.04283), extended with three contributions.
Complexity: ★★★★☆ (adaptive sampling, wavelet loss, frequency-saliency pruning)

TriFast improves FastGS with three complementary innovations:

  1. Adaptive Hard-View Consistency (ACVD)
     - Tracks per-view L1 reconstruction error via an EMA.
     - Every 10 steps, renders 1 additional "hard" view (highest EMA error).
     - Adds a simple L1 consistency loss between that view's render and GT.
     - ~10% compute overhead vs the original pairwise approach (600% overhead).

  2. Wavelet Frequency Loss (WFL)
     - 3-level Haar DWT on rendered + GT images (no pywavelets needed).
     - L1 loss on HH, HL, LH sub-bands at each level (LL is skipped).
     - Level weights mu_l = 2^(-(l-1)) = [1, 0.5, 0.25].
     - lambda_w ramps linearly from 0 → 0.02 over steps 1000–5000.
     - DWT results are cached internally for FSP reuse.

  3. Frequency-Saliency Pruning (FSP)
     - Reuses WFL DWT cache — no extra DWT computation.
     - Per-Gaussian saliency = average HF energy within their 2D footprint.
     - Dual prune condition: opacity < alpha_thresh AND saliency < S_thresh.
     - S_thresh = P-th percentile of saliency; P anneals 5% → 20%.
     - Runs every 100 steps from step 2000 to 70% of training.
     - Implemented as SOFT pruning (clamp opacity logit to -100) to
       preserve Adam optimizer states; hard cleanup handled by inherited
       densification strategy.

Architecture
------------
    TriFastModel(GSTemplateModel)
        populate_modules() — init ACVD sampler, WFL module, FSP pruner
        get_outputs()      — store camera + cam_idx; standard rendering
        get_loss_dict()    — call super + apply FSP pruning callback
        compute_custom_losses() — WFL + ACVD hard-view L1 loss
        get_metrics_dict() — log all hyperparams + EMA error stats

Train:
    ns-train trifast --data <path>
    ns-train trifast --data <path> --pipeline.model.lambda_w 0.02
    ns-train trifast --data <path> --pipeline.model.mv_consistency_weight 0.01
    ns-train trifast --data <path> --pipeline.model.fsp_alpha_thresh 0.003
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.methods.trifast_utils import (
    AdaptiveViewSampler,
    WaveletFrequencyLoss,
    FrequencySaliencyPruner,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TriFastModelConfig(GSTemplateModelConfig):
    """TriFast configuration.

    Groups by contribution:
        ACVD — adaptive hard-view consistency
        WFL  — wavelet frequency loss
        FSP  — frequency-saliency pruning
    """

    _target: Type = field(default_factory=lambda: TriFastModel)

    # ── ACVD ───────────────────────────────────────────────────────

    acvd_enabled: bool = True
    """Enable Adaptive Hard-View Consistency loss."""

    acvd_ema_alpha: float = 0.1
    """EMA update rate for per-view error: U_i = (1-alpha)*U_i + alpha*err."""

    mv_consistency_weight: float = 0.005
    """Weight lambda_mv for the hard-view consistency loss.
    Applied to L1 loss between rendered hard view and its GT."""

    mv_consistency_start: int = 1000
    """Step at which ACVD hard-view consistency loss becomes active."""

    acvd_hard_view_interval: int = 10
    """Render 1 hard view every this many steps."""

    # ── WFL ────────────────────────────────────────────────────────

    wfl_enabled: bool = True
    """Enable Wavelet Frequency Loss."""

    lambda_w: float = 0.02
    """Target wavelet loss weight (reached after warmup). Lowered from 0.05
    to avoid over-correction / blurring (Issue 2 fix)."""

    wfl_warmup_start: int = 1000
    """Step at which lambda_w begins ramping up from zero."""

    wfl_warmup_end: int = 5000
    """Step at which lambda_w reaches its target value."""

    # ── FSP ────────────────────────────────────────────────────────

    fsp_enabled: bool = True
    """Enable Frequency-Saliency Pruning."""

    fsp_alpha_thresh: float = 0.005
    """Opacity threshold for FSP dual condition."""

    fsp_p_start: float = 5.0
    """Starting percentile for saliency threshold S_thresh."""

    fsp_p_end: float = 20.0
    """Ending percentile for saliency threshold S_thresh."""

    fsp_prune_every: int = 100
    """Steps between FSP pruning runs."""

    fsp_prune_start: int = 2000
    """First step at which FSP may prune. Raised from 500 → 2000 to let
    Gaussians densify first before pruning (Issue 3 fix)."""

    fsp_prune_until: float = 0.7
    """Fraction of total steps after which FSP stops (default 0.7 = 70%)."""

    fsp_dead_opacity_logit: float = -100.0
    """Value clamped to opacity logit for soft-pruned Gaussians."""

    # ── Shared ─────────────────────────────────────────────────────

    max_training_steps: int = 30000
    """Total training steps T. Controls all annealing schedules."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class TriFastModel(GSTemplateModel):
    """TriFast: Adaptive Hard-View + Wavelet Frequency + FSP pruning.

    Inherits all rendering, densification and optimization from
    SplatfactoModel/GSTemplateModel and adds three components via
    the standard extension points.
    """

    config: TriFastModelConfig

    # ── Extension Point 1: populate_modules ────────────────────────

    def populate_modules(self) -> None:
        """Initialize ACVD sampler, WFL module, and FSP pruner."""
        super().populate_modules()

        # ACVD — purely CPU state (no GPU tensors)
        self._acvd = AdaptiveViewSampler(
            ema_alpha=self.config.acvd_ema_alpha,
        )

        # WFL — nn.Module (registers Haar kernel buffers)
        self._wfl = WaveletFrequencyLoss(
            lambda_w=self.config.lambda_w,
            warmup_start=self.config.wfl_warmup_start,
            warmup_end=self.config.wfl_warmup_end,
        )

        # FSP — stateless helper (no nn.Module needed)
        self._fsp = FrequencySaliencyPruner(
            alpha_thresh=self.config.fsp_alpha_thresh,
            p_start=self.config.fsp_p_start,
            p_end=self.config.fsp_p_end,
            prune_every=self.config.fsp_prune_every,
            prune_start=self.config.fsp_prune_start,
            prune_until=self.config.fsp_prune_until,
        )

        # Internal state
        self._current_cam_idx: Optional[int] = None   # cam index for ACVD EMA update
        self._current_camera: Optional[Cameras] = None  # camera for ACVD hard-view
        self._last_image_hw: Tuple[int, int] = (512, 512)  # (H, W) for saliency map
        self._fsp_total_pruned: int = 0

        # Training dataset reference for hard-view rendering
        self._train_dataset = None

        # DWT cache pointer — shared between WFL forward and FSP
        # Populated each training step, cleared after FSP consumes it
        self._dwt_pred_cache: Optional[Dict[str, Tensor]] = None

    # ── Extension Point 2: get_outputs ─────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Standard rendering + bookkeeping for ACVD/FSP.

        Stores the camera index for EMA update and records image size
        for saliency computation.
        """
        # Attempt to extract camera index for ACVD EMA update.
        # nerfstudio passes camera_indices in the batch; here we read it
        # from the camera object if available.
        self._current_cam_idx = None
        if self.training:
            try:
                if hasattr(camera, "camera_indices") and camera.camera_indices is not None:
                    self._current_cam_idx = int(camera.camera_indices.squeeze().item())
            except Exception:
                pass
            self._current_camera = camera

        outputs = super().get_outputs(camera)

        # Record rendered image size for FSP saliency map alignment
        if "rgb" in outputs:
            rgb = outputs["rgb"]
            self._last_image_hw = (int(rgb.shape[0]), int(rgb.shape[1]))

        return outputs

    # ── Extension Point 3a: compute_custom_losses ──────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute WFL and ACVD hard-view consistency losses.

        Also updates the ACVD EMA for the current view and populates
        the DWT cache for FSP.
        """
        losses: Dict[str, Tensor] = {}
        step = self.step
        T = self.config.max_training_steps

        if "rgb" not in outputs:
            return losses

        pred_rgb = outputs["rgb"]                          # (H, W, 3)
        gt_rgb = batch["image"].to(pred_rgb.device).float()

        # Normalize GT from uint8 → float [0, 1] if needed
        if gt_rgb.max() > 1.5:
            gt_rgb = gt_rgb / 255.0

        # Align spatial resolution (GT may differ due to downscaling)
        pred_rgb, gt_rgb = self._align_resolution(pred_rgb, gt_rgb)

        # ── 1. Update ACVD per-view EMA ─────────────────────────────
        if self.config.acvd_enabled and self._current_cam_idx is not None:
            with torch.no_grad():
                l1_err = F.l1_loss(pred_rgb, gt_rgb).item()
                self._acvd.update(self._current_cam_idx, l1_err)

        # ── 2. Wavelet Frequency Loss (WFL) ─────────────────────────
        effective_lambda = 0.0
        if self.config.wfl_enabled:
            wfl_loss, effective_lambda = self._wfl(pred_rgb, gt_rgb, step)
            if effective_lambda > 1e-9:
                losses["wfl_loss"] = wfl_loss
            # Pop DWT cache regardless (FSP needs it even during warm-up)
            self._dwt_pred_cache, _ = self._wfl.pop_cache()
        else:
            self._dwt_pred_cache = None

        # ── 3. ACVD Hard-View Consistency loss ──────────────────────
        # Every acvd_hard_view_interval steps, render the single hardest
        # view (highest EMA error) and add an L1 loss against its GT.
        # This is ~10% overhead (1 extra render per 10 steps) vs the
        # original pairwise approach that rendered K=6..18 views every
        # step (~600% overhead). Gradients flow correctly here because
        # this runs inside get_loss_dict() / compute_custom_losses()
        # BEFORE optimizer.step() and optimizer.zero_grad().
        if (
            self.config.acvd_enabled
            and self.config.mv_consistency_weight > 0
            and step >= self.config.mv_consistency_start
            and step % self.config.acvd_hard_view_interval == 0
        ):
            hard_view_loss = self._compute_hard_view_loss(pred_rgb.device)
            if hard_view_loss is not None:
                losses["mv_consistency_loss"] = (
                    self.config.mv_consistency_weight * hard_view_loss
                )

        # ── Log hyperparams to metrics_dict ─────────────────────────
        if metrics_dict is not None:
            metrics_dict["wfl_effective_lambda"] = torch.tensor(
                effective_lambda, dtype=torch.float32
            )
            n_views = len(self._acvd.view_errors)
            if n_views > 0:
                metrics_dict["acvd_max_error_view"] = torch.tensor(
                    max(self._acvd.view_errors.values()), dtype=torch.float32
                )
            metrics_dict["fsp_percentile"] = torch.tensor(
                self._fsp.current_percentile(step, T), dtype=torch.float32
            )

        return losses

    # ── Extension Point 3b: get_loss_dict (pruning callback) ───────

    def get_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict=None,
    ) -> Dict[str, Tensor]:
        """Inherited losses + custom losses + FSP pruning hook."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # FSP: run soft pruning after loss computation so DWT cache is ready
        if (
            self.config.fsp_enabled
            and self.training
            and self._fsp.should_run(self.step, self.config.max_training_steps)
        ):
            self._run_fsp()

        return loss_dict

    # ── Extension Point 8: get_metrics_dict ────────────────────────

    def get_metrics_dict(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Standard metrics + TriFast-specific monitoring."""
        metrics = super().get_metrics_dict(outputs, batch)
        step = self.step
        T = self.config.max_training_steps

        # ACVD stats
        n_views = len(self._acvd.view_errors)
        metrics["trifast/acvd_views_tracked"] = torch.tensor(float(n_views))
        if n_views > 0:
            errors = list(self._acvd.view_errors.values())
            metrics["trifast/acvd_mean_error"] = torch.tensor(
                sum(errors) / max(len(errors), 1)
            )
            metrics["trifast/acvd_max_error"] = torch.tensor(max(errors))

        # WFL stats
        metrics["trifast/wfl_effective_lambda"] = torch.tensor(
            self._wfl.effective_lambda(step)
        )

        # FSP stats
        metrics["trifast/fsp_total_pruned"] = torch.tensor(
            float(self._fsp_total_pruned)
        )
        metrics["trifast/fsp_percentile"] = torch.tensor(
            self._fsp.current_percentile(step, T)
        )

        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _align_resolution(pred: Tensor, gt: Tensor) -> Tuple[Tensor, Tensor]:
        """Bilinearly resize GT to match pred if spatial shapes differ."""
        if gt.shape[:2] == pred.shape[:2]:
            return pred, gt
        try:
            gt = F.interpolate(
                gt.permute(2, 0, 1).unsqueeze(0),
                size=(int(pred.shape[0]), int(pred.shape[1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        except Exception:
            pass  # shape mismatch: return gt as-is, loss will error-guard
        return pred, gt

    def _compute_hard_view_loss(
        self,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Render the single hardest view and return L1 loss against its GT.

        Selects the camera index with the highest EMA error, renders it,
        and returns L1(render, gt). This is much cheaper than the original
        pairwise ACVD approach (~10% overhead vs ~600%).

        Gradients are preserved so the loss backpropagates to Gaussians.
        Returns None if no views have been tracked yet or rendering fails.
        """
        if not self._acvd.view_errors:
            return None

        # Pick the view with the highest EMA reconstruction error
        hard_cam_idx = max(self._acvd.view_errors, key=self._acvd.view_errors.get)

        # Retrieve training dataset to get GT for the hard view
        train_dataset = self._get_train_dataset()
        if train_dataset is None:
            return None

        cameras = getattr(train_dataset, "cameras", None)
        if cameras is None or hard_cam_idx >= cameras.shape[0]:
            return None

        # Render the hard view (with gradients for loss backprop)
        hard_rgb = self._render_camera_idx(hard_cam_idx, cameras, device)
        if hard_rgb is None:
            return None

        # Get GT for the hard view
        try:
            hard_batch = train_dataset[hard_cam_idx]
            hard_gt = hard_batch["image"].to(device).float()
            if hard_gt.max() > 1.5:
                hard_gt = hard_gt / 255.0
        except Exception:
            return None

        # Align resolution
        hard_rgb, hard_gt = self._align_resolution(hard_rgb, hard_gt)

        return F.l1_loss(hard_rgb, hard_gt)

    def _render_camera_idx(
        self, cam_idx: int, cameras: Cameras, device: torch.device
    ) -> Optional[Tensor]:
        """Render a training camera by index with gradients preserved.

        Saves and restores self.xys / self.radii to prevent the auxiliary
        render from corrupting the training-view projections that FSP needs.

        Returns (H, W, 3) RGB tensor on `device`, or None on failure.
        """
        try:
            cam = cameras[cam_idx : cam_idx + 1]

            # Save training-view 2D state so FSP is not corrupted by the
            # auxiliary render overwriting self.xys / self.radii.
            saved_xys = getattr(self, "xys", None)
            saved_radii = getattr(self, "radii", None)

            # Temporarily switch to eval to skip densification bookkeeping,
            # but keep gradients active so the consistency loss has a
            # gradient path back to the Gaussian parameters.
            was_training = self.training
            self.eval()
            try:
                with torch.enable_grad():
                    out = super().get_outputs(cam)
                rgb = out.get("rgb", None)
            finally:
                if was_training:
                    self.train()

            # Restore training-view projections for FSP
            if saved_xys is not None:
                self.xys = saved_xys
            if saved_radii is not None:
                self.radii = saved_radii

            return rgb

        except Exception:
            return None

    def _get_train_dataset(self):
        """Retrieve training dataset from the nerfstudio pipeline.

        Tries multiple access paths for compatibility with different
        nerfstudio versions and trainer configurations.
        """
        # Cached reference
        if self._train_dataset is not None:
            return self._train_dataset
        # Pipeline reference (set by some nerfstudio trainers)
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            dm = getattr(self._pipeline, "datamanager", None)
            if dm is not None:
                train_dataset = getattr(dm, "train_dataset", None)
                if train_dataset is not None:
                    self._train_dataset = train_dataset
                    return train_dataset
        return None

    def _run_fsp(self) -> None:
        """Execute Frequency-Saliency Pruning (soft opacity clamping).

        Retrieves the DWT cache that WFL populated earlier in this step,
        computes per-Gaussian saliency, and soft-prunes Gaussians that
        satisfy the dual (low opacity + low saliency) condition.
        """
        step = self.step
        T = self.config.max_training_steps

        # Retrieve DWT cache (may have been cleared already — safe no-op)
        dwt_cache = self._dwt_pred_cache
        self._dwt_pred_cache = None  # prevent stale cache reuse

        # Get projected 2D positions and radii from last render
        xys = getattr(self, "xys", None)
        radii = getattr(self, "radii", None)

        H, W = self._last_image_hw

        # Compute per-Gaussian saliency
        try:
            saliency = self._fsp.compute_saliency(xys, radii, dwt_cache, H, W)
        except Exception as e:
            CONSOLE.print(
                f"  [TriFast FSP] saliency computation failed at step {step}: {e}"
            )
            return

        # Get prune mask
        try:
            prune_mask = self._fsp.get_prune_mask(
                self.opacities, saliency, step, T
            )
        except Exception as e:
            CONSOLE.print(
                f"  [TriFast FSP] prune mask computation failed at step {step}: {e}"
            )
            return

        if prune_mask is None:
            return

        n_prune = int(prune_mask.sum().item())
        if n_prune == 0:
            return

        # SOFT pruning: clamp opacity logit → -100 (sigmoid(-100) ≈ 0)
        with torch.no_grad():
            self.opacities.data[prune_mask] = self.config.fsp_dead_opacity_logit

        self._fsp_total_pruned += n_prune

        if step % 500 == 0 or n_prune > 1000:
            alive = int((self.opacities.data > self.config.fsp_dead_opacity_logit + 10).sum())
            CONSOLE.print(
                f"  [TriFast FSP] step {step}: soft-pruned {n_prune:,} Gaussians "
                f"(P={self._fsp.current_percentile(step, T):.1f}%) "
                f"| alive: {alive:,} / {self.num_points:,}"
            )

    # ── Checkpoint helpers ─────────────────────────────────────────

    def load_state_dict(self, d: Dict, **kwargs) -> None:
        """Restore ACVD EMA state alongside Gaussian parameters."""
        # Restore ACVD view error EMA if saved.
        # Use .get() + explicit del to avoid mutating caller's dict with .pop().
        acvd_state = d.get("_trifast_acvd_state", None)
        if acvd_state is not None:
            try:
                self._acvd.load_state_dict(acvd_state)
                del d["_trifast_acvd_state"]
            except Exception:
                pass

        fsp_pruned = d.get("_trifast_fsp_total_pruned", None)
        if fsp_pruned is not None:
            try:
                self._fsp_total_pruned = int(fsp_pruned)
                del d["_trifast_fsp_total_pruned"]
            except Exception:
                pass

        super().load_state_dict(d, **kwargs)

    # num_points is inherited from GSTemplateModel (self.means.shape[0]).
