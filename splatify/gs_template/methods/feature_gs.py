"""
Feature Gaussian Splatting
===========================
Complexity: ★★★★☆ (new params, rendering, loss, densification)
Modeled after: Feature Splatting (Qiu et al., 2024)

Distills language-aligned features (CLIP/DINO) into per-Gaussian
latent vectors. Features are fused with RGB into a single
rasterization pass, then decoded by an MLP to produce high-dim
feature maps for open-vocabulary queries.

Demonstrates ALL extension points:
    1. populate_modules   — per-Gaussian features + decode MLP
    2. get_outputs        — fused rendering using rendering.py helpers
    3. compute_custom_losses — cosine feature distillation loss
    4. split_gaussians    — replicate features during densification
    5. get_gaussian_param_groups — register feature optimizer
    6. get_param_groups   — register MLP optimizer
    7. get_outputs_for_camera — PCA visualization in viewer
    8. load_state_dict    — resize params for checkpoint loading
    9. get_metrics_dict   — feature PSNR metric

Train:
    ns-train feature-gs --data <path>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import cosine_loss, total_variation_loss
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
class FeatureGSModelConfig(GSTemplateModelConfig):
    """Feature Gaussian Splatting configuration."""

    _target: Type = field(default_factory=lambda: FeatureGSModel)

    # SH degree 0 — feature splatting pre-computes color via sigmoid
    sh_degree: int = 0

    # Per-Gaussian feature dimension (latent, rasterized directly)
    feat_dim: int = 13
    """Latent dimension of per-Gaussian feature vectors."""

    # MLP decode head
    mlp_hidden_dim: int = 64
    """Hidden dim for the feature decode MLP."""

    output_feat_dim: int = 768
    """Output feature dim (e.g. 768 for CLIP ViT-L/14)."""

    # Loss weights
    feat_loss_weight: float = 1e-3
    """Weight for the feature cosine distillation loss."""

    feat_tv_weight: float = 0.0
    """Weight for total variation on rendered features (0 = off)."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class FeatureGSModel(GSTemplateModel):
    """Feature Gaussian Splatting — distill features into Gaussians."""

    config: FeatureGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()

        # Per-Gaussian latent features
        self.gauss_params["distill_features"] = Parameter(
            torch.zeros(self.num_points, self.config.feat_dim)
        )

        # Decode MLP: latent → high-dim features
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.config.feat_dim, self.config.mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dim, self.config.output_feat_dim),
        )

    # ── 2. get_outputs (custom rendering) ──────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Fuse RGB + latent features into one rasterization pass."""

        # Phase 1: camera setup
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids  # empty outputs (crop box eliminated all)

        # Gather base params + crop custom features
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )
        feats = crop_custom_param(
            self.gauss_params["distill_features"], crop_ids
        )

        # Fuse: (N, 3) RGB + (N, feat_dim) features → (N, 3+feat_dim)
        fused = torch.cat([colors, feats], dim=-1)

        # Phase 2: rasterize
        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, fused,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            sh_degree=None,  # colors are pre-computed (degree 0)
        )

        # Phase 3: assemble outputs with feature channel
        outputs = assemble_outputs(
            self, render, alpha, info, cam,
            extra_channels={
                "feature": (3, 3 + self.config.feat_dim),
            },
        )
        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        if "feature_dict" not in batch or "feature" not in outputs:
            return losses

        # Decode rendered latent features → high-dim
        rendered_feat = outputs["feature"]  # (H, W, feat_dim)
        decoded = self.feature_mlp(rendered_feat)  # (H, W, output_feat_dim)

        # Get target features from datamanager
        for name, target in batch["feature_dict"].items():
            target = target.to(decoded.device)

            # Resize decoded to match target spatial dims if needed
            if decoded.shape[:2] != target.shape[1:]:
                d = decoded.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                d = F.interpolate(
                    d, size=(target.shape[1], target.shape[2]),
                    mode="bilinear", align_corners=False,
                )
                d = d.squeeze(0)  # (C,h,w)
            else:
                d = decoded.permute(2, 0, 1)  # (C,H,W)

            # Ignore zero-valued feature regions (padding/invalid)
            ignore_mask = (target.sum(dim=0) == 0)
            target[:, ignore_mask] = d[:, ignore_mask].detach()

            losses["feature_loss"] = (
                self.config.feat_loss_weight * cosine_loss(d, target, dim=0)
            )

        # Optional TV regularization on rendered features
        if self.config.feat_tv_weight > 0:
            losses["feature_tv"] = (
                self.config.feat_tv_weight
                * total_variation_loss(rendered_feat)
            )

        return losses

    # ── 4. split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        out["distill_features"] = self.gauss_params["distill_features"][
            split_mask
        ].repeat(samps, 1)
        return out

    # ── 5. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register ALL gauss_params including distill_features."""
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    # ── 6. get_param_groups ────────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["feature_mlp"] = list(self.feature_mlp.parameters())
        return groups

    # ── 7. get_outputs_for_camera (viewer) ─────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)

        # PCA visualization of rendered features
        if "feature" in outs:
            feat = outs["feature"]  # (H, W, feat_dim)
            flat = feat.reshape(-1, feat.shape[-1])  # (HW, C)
            # Center and compute top-3 PCA components
            mean = flat.mean(dim=0, keepdim=True)
            centered = flat - mean
            _, _, V = torch.svd_lowrank(centered, q=3)
            projected = centered @ V  # (HW, 3)
            # Normalize to [0, 1] for visualization
            pmin = projected.min(dim=0).values
            pmax = projected.max(dim=0).values
            pca_rgb = (projected - pmin) / (pmax - pmin + 1e-6)
            outs["feature_pca"] = pca_rgb.reshape(
                feat.shape[0], feat.shape[1], 3
            )

        return outs

    # ── 8. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        # Handle legacy format (means → gauss_params.means)
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        # Resize all gauss_params to match checkpoint
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    # ── 9. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_features"] = torch.tensor(
            self.gauss_params["distill_features"].shape[0],
            dtype=torch.float32,
        )
        return metrics

    # ── Convenience ────────────────────────────────────────────────

    @property
    def distill_features(self):
        return self.gauss_params["distill_features"]
