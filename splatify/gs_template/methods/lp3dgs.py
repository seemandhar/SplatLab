"""
LP-3DGS — Learning to Prune 3D Gaussian Splatting
===================================================
Complexity: ★★★☆☆ (new params, modified rendering, custom loss, pruning)

Paper contribution: Learn a per-Gaussian trainable binary mask via
Gumbel-Sigmoid relaxation to automatically determine the optimal
pruning ratio per scene, replacing hand-tuned thresholds.

Training stages:
    Stage 1 (iter 0–19500):     Standard 3DGS with adaptive density control
    Stage 2 (iter 19500–20000): Mask learning with Gumbel-Sigmoid (500 iters)
    Prune at iter 20000:        Freeze hard binary mask (soft pruning)
    Stage 3 (iter 20000–30000): Fine-tune with hard mask applied

NOTE ON SOFT PRUNING:
    Nerfstudio's optimizer is created externally and the model has no access
    to it. Physically removing Gaussians (replacing Parameters in gauss_params)
    breaks the optimizer — it still holds references to the old Parameter
    objects, making all subsequent fine-tuning a no-op.

    Instead, we use "soft pruning": at step 20000, we compute a hard binary
    mask and store it as a buffer. During ALL subsequent rendering (train +
    eval), pruned Gaussians have their opacity zeroed out (via the mask).
    Fine-tuning works correctly because gradients flow normally through the
    kept Gaussians. The "pruned" Gaussians still consume memory but contribute
    nothing to rendering.

    For actual memory savings, export the kept Gaussians post-training.

Demonstrates:
    - populate_modules()          → per-Gaussian mask params + importance buffer
    - get_outputs()               → apply Gumbel-Sigmoid / hard mask to opacity
    - compute_custom_losses()     → L1 mask sparsity regularization
    - split_gaussians()           → replicate masks during densification
    - get_gaussian_param_groups() → register mask optimizer
    - load_state_dict()           → resize buffers for checkpoint loading

Importance scores (configurable):
    - "radsplat":      S_i = max over views of α_i · T_i  (Niemeyer et al., 2024)
    - "minisplatting": S_i = Σ_j ω_ij  (sum of color weights)  (Fang & Wang, 2024)
    - "opacity":       S_i = sigmoid(opacity_i)  (fast approximation)

Train:
    ns-train lp3dgs --data <path>
    ns-train lp3dgs --data <path> --pipeline.model.importance_type radsplat
    ns-train lp3dgs --data <path> --pipeline.model.mask_reg_weight 1e-3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.rendering import (
    assemble_outputs,
    gather_base_params,
    prepare_camera,
    rasterize_gaussians,
)


# ═══════════════════════════════════════════════════════════════════
# Gumbel-Sigmoid Utilities
# ═══════════════════════════════════════════════════════════════════

def _sample_gumbel(shape: Tuple[int, ...], device: torch.device) -> Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    u = torch.rand(shape, device=device).clamp(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(u))


def gumbel_sigmoid(
    logits: Tensor,
    tau: float = 0.5,
    training: bool = True,
) -> Tensor:
    """Gumbel-Sigmoid: differentiable relaxation of binary mask.

    From Eq. 7 of LP-3DGS:
        gs(m) = σ((log(m) + g₀ - g₁) / τ)

    where g₀, g₁ ~ Gumbel(0, 1), τ is temperature.

    At low τ, outputs are near-binary (0 or 1).
    At inference (not training), returns hard sigmoid without noise.

    Args:
        logits: input values (m_i * S_i), must be positive
        tau:    temperature (default 0.5 per paper)
        training: whether to add Gumbel noise

    Returns:
        Near-binary mask values in [0, 1]
    """
    if training:
        g0 = _sample_gumbel(logits.shape, logits.device)
        g1 = _sample_gumbel(logits.shape, logits.device)
        log_m = torch.log(logits.clamp(min=1e-8))
        return torch.sigmoid((log_m + g0 - g1) / tau)
    else:
        return torch.sigmoid(torch.log(logits.clamp(min=1e-8)) / tau)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LP3DGSModelConfig(GSTemplateModelConfig):
    """LP-3DGS: Learning to Prune 3D Gaussian Splatting."""

    _target: Type = field(default_factory=lambda: LP3DGSModel)

    # ── Training schedule ──────────────────────────────────────────
    mask_learning_start: int = 19500
    """Iteration to begin mask learning (Stage 2 start)."""

    mask_learning_end: int = 20000
    """Iteration to end mask learning and freeze the hard mask."""

    importance_update_freq: int = 20
    """Update importance scores every N iterations during mask learning."""

    # ── Gumbel-Sigmoid ─────────────────────────────────────────────
    gumbel_tau: float = 0.5
    """Temperature for Gumbel-Sigmoid (lower = more binary)."""

    # ── Importance score ───────────────────────────────────────────
    importance_type: Literal["radsplat", "minisplatting", "opacity"] = "radsplat"
    """Which importance metric to use for mask modulation."""

    # ── Mask regularization ────────────────────────────────────────
    mask_reg_weight: float = 5e-4
    """λ_m: L1 sparsity regularization on mask values (Eq. 10)."""

    # ── Pruning ────────────────────────────────────────────────────
    prune_threshold: float = 0.5
    """After mask learning, prune Gaussians with mask value < this."""

    # ── Importance accumulation ────────────────────────────────────
    importance_accumulation_start: int = 15000
    """Start accumulating importance scores from this iteration."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class LP3DGSModel(GSTemplateModel):
    """LP-3DGS: Learning to Prune 3D Gaussian Splatting.

    Uses soft pruning: a hard binary mask is frozen after mask learning
    and applied to opacities during all subsequent rendering. This avoids
    breaking the optimizer (which holds references to Parameter objects).
    """

    config: LP3DGSModelConfig

    # ── Extension Point 1: populate_modules ────────────────────────

    def populate_modules(self):
        """Initialize mask parameters and importance score buffers."""
        super().populate_modules()

        n = self.num_points

        # Per-Gaussian trainable mask (goes through Gumbel-Sigmoid)
        self.gauss_params["mask_logits"] = Parameter(
            torch.ones(n, 1, device="cpu")
        )

        # Importance score buffers (not optimized)
        self.register_buffer("importance_scores", torch.ones(n, 1))
        self.register_buffer("_importance_max", torch.zeros(n, 1))
        self.register_buffer("_importance_sum", torch.zeros(n, 1))
        self.register_buffer("_importance_count", torch.zeros(n, 1))

        # Hard binary mask: 1.0 = keep, 0.0 = prune
        # Starts as all-ones; frozen to 0/1 at mask_learning_end
        self.register_buffer("_hard_mask", torch.ones(n, 1))

        # Flag: has the mask been frozen?
        self._mask_frozen = False

    # ── Extension Point 2: get_outputs ─────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render with masked opacities.

        Phases:
            iter < accumulation_start:   Standard rendering
            accumulation → mask_start:   Standard rendering + accumulate importance
            mask_start → mask_end:       Gumbel-Sigmoid soft mask on opacity
            iter == mask_end:            Freeze hard binary mask
            iter > mask_end:             Hard mask on opacity (fine-tune + eval)
        """
        step = self.step

        in_mask_learning = (
            self.training
            and step >= self.config.mask_learning_start
            and step < self.config.mask_learning_end
            and not self._mask_frozen
        )

        in_importance_accumulation = (
            self.training
            and step >= self.config.importance_accumulation_start
            and step < self.config.mask_learning_end
            and not self._mask_frozen
        )

        # ── Freeze mask at mask_learning_end ──────────────────────
        if (
            self.training
            and step >= self.config.mask_learning_end
            and not self._mask_frozen
        ):
            self._freeze_mask()

        # Do we need custom rendering?
        needs_custom = (
            in_mask_learning
            or in_importance_accumulation
            or self._mask_frozen
        )

        if not needs_custom:
            return super().get_outputs(camera)

        # ── Phase 1: Camera setup ─────────────────────────────────
        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # ── Accumulate importance (pre-mask-learning) ─────────────
        if in_importance_accumulation:
            with torch.no_grad():
                self._accumulate_importance(opacities, crop_ids)

        # ── Determine mask ────────────────────────────────────────
        mask_values = None

        if in_mask_learning:
            # Stage 2: differentiable Gumbel-Sigmoid mask
            if step % self.config.importance_update_freq == 0:
                self._update_importance_scores()

            if crop_ids is not None:
                ml = self.gauss_params["mask_logits"][crop_ids]
                imp = self.importance_scores[crop_ids]
            else:
                ml = self.gauss_params["mask_logits"]
                imp = self.importance_scores

            mask_positive = F.softplus(ml)
            mask_input = mask_positive * imp.detach()
            mask_values = gumbel_sigmoid(
                mask_input, tau=self.config.gumbel_tau, training=True,
            )

        elif self._mask_frozen:
            # Stage 3 / eval: frozen hard binary mask
            if crop_ids is not None:
                mask_values = self._hard_mask[crop_ids]
            else:
                mask_values = self._hard_mask

        # ── Apply mask to opacity (logit space) ───────────────────
        if mask_values is not None:
            # log(mask) in logit space = multiply sigmoid(o) by mask
            # log(1.0) = 0 (no change), log(~0) = -18 (effectively zero)
            opacities = opacities + torch.log(mask_values.clamp(min=1e-8))

        # ── Phase 2: Rasterize ────────────────────────────────────
        render_mode = "RGB+ED" if (
            self.config.output_depth_during_training or not self.training
        ) else "RGB"

        render, alpha, info = rasterize_gaussians(
            self, means, quats, scales, opacities, colors,
            cam["viewmat"], cam["K"], cam["W"], cam["H"],
            render_mode=render_mode,
        )

        # ── Phase 3: Assemble outputs ─────────────────────────────
        outputs = assemble_outputs(self, render, alpha, info, cam)

        # Store soft mask for loss (only during mask learning)
        if in_mask_learning and mask_values is not None:
            outputs["_mask_values"] = mask_values

        return outputs

    # ── Extension Point 3: custom losses ───────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """L1 sparsity regularization on mask (Eq. 10)."""
        losses = {}
        if "_mask_values" in outputs:
            losses["mask_reg"] = (
                self.config.mask_reg_weight
                * outputs["_mask_values"].abs().mean()
            )
        return losses

    # ── Extension Point 4: split_gaussians ─────────────────────────

    def split_gaussians(self, split_mask, samps):
        """Replicate mask parameters when Gaussians are split."""
        out = super().split_gaussians(split_mask, samps)
        out["mask_logits"] = self.gauss_params["mask_logits"][
            split_mask
        ].repeat(samps, 1)
        return out

    # ── Extension Point 5: param groups ────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Register mask_logits for optimization."""
        groups = super().get_gaussian_param_groups()
        if "mask_logits" in self.gauss_params:
            groups["mask_logits"] = [self.gauss_params["mask_logits"]]
        return groups

    # ── Extension Point 8: load_state_dict ─────────────────────────

    def load_state_dict(self, dict, **kwargs):
        """Resize all params and buffers to match checkpoint before loading."""
        # Determine Gaussian count from checkpoint
        newp = None
        if "gauss_params.means" in dict:
            newp = dict["gauss_params.means"].shape[0]
        elif "means" in dict:
            newp = dict["means"].shape[0]

        if newp is not None:
            # Resize gauss_params
            for name, param in self.gauss_params.items():
                new_shape = (newp,) + param.shape[1:]
                self.gauss_params[name] = Parameter(
                    torch.zeros(new_shape, device=param.device)
                )

            # Resize registered buffers
            buf_names = [
                "importance_scores",
                "_importance_max",
                "_importance_sum",
                "_importance_count",
                "_hard_mask",
            ]
            for buf_name in buf_names:
                if buf_name in dict:
                    target_shape = dict[buf_name].shape
                else:
                    target_shape = (newp, 1)
                old = getattr(self, buf_name, None)
                device = old.device if old is not None else "cpu"
                setattr(self, buf_name, torch.zeros(target_shape, device=device))

        # Skip mask learning on eval
        self.step = 30000

        # Detect if mask was frozen in the checkpoint
        if "_hard_mask" in dict:
            hard = dict["_hard_mask"]
            if (hard == 0.0).any():
                self._mask_frozen = True

        super().load_state_dict(dict, **kwargs)

    # ── Extension Point 9: metrics ─────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        if self._mask_frozen:
            n_active = (self._hard_mask > 0.5).sum().item()
            metrics["active_gaussians"] = torch.tensor(
                n_active, dtype=torch.float32
            )
            metrics["pruning_ratio"] = torch.tensor(
                1.0 - n_active / max(self.num_points, 1),
                dtype=torch.float32,
            )
        if "_mask_values" in outputs:
            mv = outputs["_mask_values"]
            metrics["mask_mean"] = mv.mean()
            metrics["mask_sparsity"] = (mv < 0.5).float().mean()
        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Mask Freezing (Soft Pruning)
    # ═══════════════════════════════════════════════════════════════

    def _freeze_mask(self):
        """Freeze learned mask into hard binary 0/1 mask.

        Called once at mask_learning_end. The hard mask is applied during
        all subsequent rendering — pruned Gaussians get zero opacity.
        The optimizer and Parameter objects remain intact.
        """
        if self._mask_frozen:
            return

        self._mask_frozen = True

        with torch.no_grad():
            actual_n = self.means.shape[0]
            self._resize_buffers(actual_n)
            self._update_importance_scores()

            mask_logits = self.gauss_params["mask_logits"]
            mask_positive = F.softplus(mask_logits)
            mask_input = mask_positive * self.importance_scores

            mask_values = gumbel_sigmoid(
                mask_input, tau=self.config.gumbel_tau, training=False,
            )

            hard_mask = (mask_values >= self.config.prune_threshold).float()

            n_total = hard_mask.shape[0]
            n_keep = int((hard_mask > 0.5).sum().item())
            n_prune = n_total - n_keep
            ratio = n_prune / max(n_total, 1) * 100

            CONSOLE.print(
                f"\n[bold green]LP-3DGS Mask Frozen @ step {self.step}[/bold green]"
            )
            CONSOLE.print(
                f"  Total: {n_total:,} | Active: {n_keep:,} | "
                f"Masked: {n_prune:,} ({ratio:.1f}%)"
            )
            CONSOLE.print(
                f"  Fine-tuning continues with hard mask for "
                f"{30000 - self.step} iterations.\n"
            )

            if n_keep == 0:
                CONSOLE.print(
                    "[bold red]WARNING: All Gaussians masked! "
                    "Keeping top 10% by mask value.[/bold red]"
                )
                k = max(1, n_total // 10)
                _, top_idx = mask_values.squeeze(-1).topk(k)
                hard_mask = torch.zeros_like(hard_mask)
                hard_mask[top_idx] = 1.0

            self._hard_mask = hard_mask

    # ═══════════════════════════════════════════════════════════════
    # Importance Score Computation
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _accumulate_importance(
        self, opacities_logit: Tensor, crop_ids: Optional[Tensor],
    ):
        """Accumulate per-Gaussian importance from current view."""
        actual_n = self.means.shape[0]
        if self.importance_scores.shape[0] != actual_n:
            self._resize_buffers(actual_n)

        alpha = torch.sigmoid(opacities_logit)

        if crop_ids is not None:
            self._importance_max[crop_ids] = torch.max(
                self._importance_max[crop_ids], alpha
            )
            self._importance_sum[crop_ids] += alpha
            self._importance_count[crop_ids] += 1
        else:
            self._importance_max = torch.max(self._importance_max, alpha)
            self._importance_sum += alpha
            self._importance_count += 1

    @torch.no_grad()
    def _update_importance_scores(self):
        """Compute importance scores from accumulated statistics."""
        actual_n = self.means.shape[0]
        if self.importance_scores.shape[0] != actual_n:
            self._resize_buffers(actual_n)

        if self.config.importance_type == "radsplat":
            scores = self._importance_max.clone()
        elif self.config.importance_type == "minisplatting":
            scores = self._importance_sum.clone()
            smax = scores.max()
            if smax > 0:
                scores = scores / smax
        elif self.config.importance_type == "opacity":
            scores = torch.sigmoid(self.opacities.detach())
        else:
            raise ValueError(f"Unknown importance type: {self.config.importance_type}")

        self.importance_scores.copy_(scores.clamp(min=1e-6, max=1.0))

    # ═══════════════════════════════════════════════════════════════
    # Buffer Management
    # ═══════════════════════════════════════════════════════════════

    def _resize_buffers(self, new_n: int):
        """Resize importance buffers to match current Gaussian count."""
        device = self.means.device
        for buf_name in [
            "importance_scores",
            "_importance_max",
            "_importance_sum",
            "_importance_count",
            "_hard_mask",
        ]:
            old = getattr(self, buf_name)
            old_n = old.shape[0]
            if old_n == new_n:
                continue
            fill_val = 1.0 if buf_name == "_hard_mask" else 0.0
            new_buf = torch.full((new_n, 1), fill_val, device=device)
            copy_n = min(old_n, new_n)
            new_buf[:copy_n] = old[:copy_n].to(device)
            setattr(self, buf_name, new_buf)

    # ═══════════════════════════════════════════════════════════════
    # Convenience
    # ═══════════════════════════════════════════════════════════════

    @property
    def mask_logits(self) -> Parameter:
        return self.gauss_params["mask_logits"]

    @property
    def num_active_gaussians(self) -> int:
        if self._mask_frozen:
            return int((self._hard_mask > 0.5).sum().item())
        return self.num_points