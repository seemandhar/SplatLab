"""
GeoConsensus-GS — Confidence-Weighted Depth-Normal Supervision
with Geometry-Aware Gaussian Dropout
================================================================
Complexity: ★★★☆☆ (custom loss + dropout + custom DataManager)

Combines three key ideas:
    1. Confidence-weighted depth supervision (Pearson loss scaled by
       joint depth-normal confidence from the DataManager)
    2. Self-supervised normal consistency (normals derived from rendered
       depth vs. normals derived from monocular depth)
    3. Geometry-aware Gaussian dropout during rendering (improves
       robustness by randomly dropping Gaussians with opacity
       compensation, similar to DropGaussian)

Extension Points Used:
    EP2 — get_outputs(): dropout via in-place opacity masking
    EP3 — compute_custom_losses(): depth, normal, opacity losses

Train:
    ns-train geo-consensus-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import (
    opacity_entropy_loss,
    pearson_depth_loss,
)


def _resize_to(tensor: Tensor, target_shape: tuple) -> Tensor:
    """Resize a (H, W, C) tensor to match target (H', W') spatial dims."""
    C = tensor.shape[-1] if tensor.dim() == 3 else 1
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    out = F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0),
        size=(int(target_shape[0]), int(target_shape[1])),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)
    return out


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GeoConsensusGSModelConfig(GSTemplateModelConfig):
    """GeoConsensus-GS configuration."""

    _target: Type = field(default_factory=lambda: GeoConsensusGSModel)

    # Must enable depth rendering during training
    output_depth_during_training: bool = True

    # ── Depth supervision ──────────────────────────────────────────
    depth_loss_weight: float = 0.2
    """Initial depth loss weight (Phase 1). Pearson loss range [0,2]."""

    depth_loss_weight_final: float = 0.02
    """Final depth loss weight (Phase 3)."""

    # ── Normal consistency ─────────────────────────────────────────
    normal_loss_weight: float = 0.05
    """Initial normal consistency loss weight."""

    normal_loss_weight_final: float = 0.01
    """Final normal consistency loss weight."""

    # ── Opacity regularization ─────────────────────────────────────
    use_opacity_entropy: bool = True
    """Enable opacity entropy regularization."""

    opacity_entropy_weight: float = 0.005
    """Weight for opacity entropy loss."""

    # ── Geometry-aware dropout ─────────────────────────────────────
    use_geo_dropout: bool = True
    """Enable geometry-aware Gaussian dropout during rendering."""

    dropout_start_iter: int = 10000
    """Iteration to begin dropout (start of Phase 2)."""

    dropout_base_rate: float = 0.05
    """Initial base dropout rate."""

    dropout_max_rate: float = 0.2
    """Maximum dropout rate (reached after ramp)."""

    dropout_ramp_iters: int = 10000
    """Number of iterations to ramp from base to max rate."""

    # ── Phase schedule ─────────────────────────────────────────────
    phase2_start: int = 10000
    """Iteration to start Phase 2 (decay supervision, start dropout)."""

    phase3_start: int = 22000
    """Iteration to start Phase 3 (minimal supervision, full dropout)."""

    max_iterations: int = 30000
    """Total training iterations."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class GeoConsensusGSModel(GSTemplateModel):
    """GeoConsensus-GS: joint depth-normal confidence regularization
    with geometry-aware Gaussian dropout.
    """

    config: GeoConsensusGSModelConfig

    def populate_modules(self):
        super().populate_modules()

    # ── EP2: get_outputs with in-place dropout ─────────────────────

    def get_outputs(self, camera):
        """Apply dropout by modifying opacities in-place before
        calling parent's get_outputs, then restore."""
        should_dropout = (
            self.training
            and self.config.use_geo_dropout
            and self.step >= self.config.dropout_start_iter
        )

        if should_dropout:
            drop_rate = self._get_dropout_rate()
            n = self.opacities.shape[0]
            device = self.opacities.device

            # Save original
            original_opacities = self.opacities.data.clone()

            # Create dropout mask
            keep_mask = torch.rand(n, device=device) > drop_rate

            # Drop: set to -100 (sigmoid -> ~0)
            self.opacities.data[~keep_mask] = -100.0

            # Compensate survivors: add log(1/(1-p)) to logits
            compensation = torch.log(
                torch.tensor(1.0 / (1.0 - drop_rate + 1e-8), device=device)
            )
            self.opacities.data[keep_mask] += compensation

        # Standard splatfacto rendering
        outputs = super().get_outputs(camera)

        if should_dropout:
            # Restore original opacities
            self.opacities.data.copy_(original_opacities)

        return outputs

    def _get_dropout_rate(self) -> float:
        """Compute current dropout rate based on training schedule."""
        step = self.step
        if step < self.config.dropout_start_iter:
            return 0.0
        progress = min(
            1.0,
            (step - self.config.dropout_start_iter) / self.config.dropout_ramp_iters,
        )
        return (
            self.config.dropout_base_rate
            + (self.config.dropout_max_rate - self.config.dropout_base_rate) * progress
        )

    # ── EP3: compute_custom_losses ─────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Compute confidence-weighted depth, normal, and regularization losses."""
        losses = {}
        step = self.step

        # Phase-aware weight scheduling
        depth_w = self._get_scheduled_weight(
            step, self.config.depth_loss_weight, self.config.depth_loss_weight_final
        )
        normal_w = self._get_scheduled_weight(
            step, self.config.normal_loss_weight, self.config.normal_loss_weight_final
        )

        # --- 1. Confidence-weighted depth supervision ---
        if "depth_image" in batch and outputs.get("depth") is not None:
            pred_depth = outputs["depth"]  # (H, W, 1)
            gt_depth = batch["depth_image"].to(pred_depth.device)

            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(-1)

            confidence = batch.get("depth_confidence", None)
            if confidence is not None:
                confidence = confidence.to(pred_depth.device)
                if confidence.dim() == 2:
                    confidence = confidence.unsqueeze(-1)

            # Resize if needed
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = _resize_to(gt_depth, pred_depth.shape[:2])
                if confidence is not None:
                    confidence = _resize_to(confidence, pred_depth.shape[:2])

            # Valid mask: ignore zero / missing depth
            valid = gt_depth > 0
            if valid.sum() > 100:
                if confidence is not None:
                    conf_weight = confidence[valid].sqrt().clamp(min=1e-6)
                    losses["depth_loss"] = depth_w * pearson_depth_loss(
                        pred_depth[valid] * conf_weight,
                        gt_depth[valid] * conf_weight,
                    )
                else:
                    losses["depth_loss"] = depth_w * pearson_depth_loss(
                        pred_depth[valid], gt_depth[valid]
                    )

        # --- 2. Normal consistency loss ---
        if "depth_image" in batch and outputs.get("depth") is not None:
            pred_depth_2d = outputs["depth"].squeeze(-1)  # (H, W)
            pred_normals = self._depth_to_normals(pred_depth_2d)

            if "depth_normal" in batch:
                gt_normals = batch["depth_normal"].to(pred_normals.device)

                if gt_normals.shape[:2] != pred_normals.shape[:2]:
                    gt_normals = _resize_to(gt_normals, pred_normals.shape[:2])

                # Dot product similarity (1 - dot = loss)
                dot = (pred_normals * gt_normals).sum(dim=-1)  # (H, W)
                normal_loss_map = (1.0 - dot).clamp(min=0.0)

                if "depth_confidence" in batch:
                    confidence = batch["depth_confidence"].to(pred_normals.device)
                    if confidence.shape[:2] != pred_normals.shape[:2]:
                        confidence = _resize_to(confidence, pred_normals.shape[:2])
                    conf_2d = confidence.squeeze(-1) if confidence.dim() == 3 else confidence
                    losses["normal_loss"] = normal_w * (conf_2d * normal_loss_map).mean()
                else:
                    losses["normal_loss"] = normal_w * normal_loss_map.mean()

        # --- 3. Opacity entropy ---
        if self.config.use_opacity_entropy:
            losses["opacity_entropy"] = (
                self.config.opacity_entropy_weight
                * opacity_entropy_loss(self.opacities)
            )

        return losses

    # ── Helpers ────────────────────────────────────────────────────

    def _get_scheduled_weight(
        self, step: int, initial: float, final: float
    ) -> float:
        """Linear weight decay across phases."""
        if step < self.config.phase2_start:
            return initial
        elif step < self.config.phase3_start:
            progress = (step - self.config.phase2_start) / (
                self.config.phase3_start - self.config.phase2_start
            )
            return initial + (final - initial) * progress
        else:
            return final

    @staticmethod
    def _depth_to_normals(depth: Tensor) -> Tensor:
        """Convert a depth map to surface normals via finite differences."""
        dz_dx = depth[:, 1:] - depth[:, :-1]
        dz_dy = depth[1:, :] - depth[:-1, :]

        dz_dx = torch.cat([dz_dx, dz_dx[:, -1:]], dim=1)
        dz_dy = torch.cat([dz_dy, dz_dy[-1:, :]], dim=0)

        normals = torch.stack(
            [-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1
        )
        normals = F.normalize(normals, dim=-1)
        return normals
