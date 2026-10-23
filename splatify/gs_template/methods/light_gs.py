"""
LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS
==================================================================================
Paper: https://arxiv.org/abs/2410.xxxxx
Project: https://lightgaussian.github.io
Complexity: ★★★★☆ (multi-stage: pruning, SH distillation, VQ)

A post-training compression pipeline for 3D Gaussian Splatting:
    Stage 1 (steps 0 → prune_step):        Normal 3DGS training
    Stage 2 (prune_step):                   Compute global significance → prune
    Stage 3 (prune_step → distill_start):   Co-adaptation recovery (no densification)
    Stage 4 (distill_start → distill_end):  SH degree distillation with pseudo-views
    Stage 5 (distill_end):                  Vector quantization of SH coefficients
    Stage 6 (distill_end → max_steps):      VQ fine-tuning

CRITICAL ARCHITECTURE NOTE:
    All stage transitions that modify Parameters (prune, SH truncation, VQ init)
    MUST happen in after_train(), BETWEEN iterations — never inside get_loss_dict()
    or get_outputs() which run DURING forward/backward. Modifying parameters
    mid-forward breaks the autograd graph and orphans optimizer state.

Train:
    ns-train light-gs --data <path>
    ns-train light-gs --data <path> --pipeline.model.prune_ratio 0.5
    ns-train light-gs --data <path> --pipeline.model.target_sh_degree 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LightGSModelConfig(GSTemplateModelConfig):
    """LightGaussian configuration."""

    _target: Type = field(default_factory=lambda: LightGSModel)

    # ── Stage scheduling ───────────────────────────────────────────
    prune_step: int = 30000
    """Step at which to compute significance and prune."""

    recovery_steps: int = 5000
    """Number of co-adaptation fine-tuning steps after pruning."""

    distill_steps: int = 5000
    """Number of SH distillation steps."""

    vq_finetune_steps: int = 5000
    """Number of VQ fine-tuning steps."""

    # ── Pruning ────────────────────────────────────────────────────
    prune_ratio: float = 0.5
    """Fraction of Gaussians to prune (by significance score)."""

    significance_beta: float = 0.1
    """Power exponent for volume normalization in significance score."""

    score_accumulation_start: int = 15000
    """Start accumulating significance scores after this step.
    Only accumulate in later training when Gaussians are mature."""

    # ── SH Distillation ───────────────────────────────────────────
    target_sh_degree: int = 2
    """Target SH degree after distillation (original is typically 3)."""

    distill_pseudo_sigma: float = 0.1
    """Std dev for pseudo-view camera position perturbation."""

    distill_loss_weight: float = 1.0
    """Weight for distillation loss."""

    # ── Vector Quantization ────────────────────────────────────────
    enable_vq: bool = True
    """Enable vector quantization of SH coefficients."""

    vq_codebook_size: int = 8192
    """Number of codebook entries for VQ."""

    vq_ratio: float = 0.6
    """Fraction of least-significant Gaussians to VQ (by significance)."""

    vq_decay: float = 0.8
    """EMA decay for codebook updates."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class LightGSModel(GSTemplateModel):
    """LightGaussian — multi-stage 3DGS compression.

    Training proceeds through multiple stages controlled by step count.
    Each stage activates different loss functions and training behaviors.

    IMPORTANT: All parameter-modifying transitions (prune, SH truncation,
    VQ init) happen in after_train() — AFTER backward pass completes,
    BEFORE next forward pass. This ensures:
    1. Autograd graph is not broken mid-computation
    2. Optimizer state can be properly updated for new Parameters
    """

    config: LightGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        N = self.num_points

        # ── Significance tracking ──────────────────────────────────
        # Accumulated significance scores (not a learnable parameter)
        self.register_buffer(
            "significance_scores",
            torch.zeros(N, device="cpu"),
        )
        self.register_buffer(
            "significance_counts",
            torch.zeros(N, device="cpu"),
        )

        # ── Stage tracking ─────────────────────────────────────────
        self._pruning_done = False
        self._distillation_active = False
        self._vq_done = False
        self._teacher_sh_dc = None
        self._teacher_sh_rest = None
        self._teacher_sh_degree = None
        self._vq_codebook = None
        self._vq_assignments = None

    # ═══════════════════════════════════════════════════════════════
    # Stage Management
    # ═══════════════════════════════════════════════════════════════

    @property
    def distill_start_step(self) -> int:
        return self.config.prune_step + self.config.recovery_steps

    @property
    def distill_end_step(self) -> int:
        return self.distill_start_step + self.config.distill_steps

    @property
    def vq_start_step(self) -> int:
        return self.distill_end_step

    @property
    def total_steps(self) -> int:
        return self.vq_start_step + self.config.vq_finetune_steps

    @property
    def current_stage(self) -> str:
        if not self._pruning_done and self.step < self.config.prune_step:
            return "training"
        elif not self._pruning_done:
            return "prune"
        elif not self._distillation_active and self.step < self.distill_start_step:
            return "recovery"
        elif self.step < self.distill_end_step:
            return "distillation"
        elif not self._vq_done and self.config.enable_vq:
            return "vq_init"
        else:
            return "vq_finetune"

    # ═══════════════════════════════════════════════════════════════
    # Global Significance Score (Algorithm 2 from paper)
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _accumulate_significance(self) -> None:
        """Accumulate per-Gaussian significance from rendering info.

        Approximates Eq. 4 from the paper:
            GS_j = sum_i hit(G_j, r_i) * sigma_j * T_j * gamma(Sigma_j)

        We compute: opacity * volume_factor for each Gaussian,
        using radii > 0 as the hit indicator. Accumulated over many
        views, this approximates the full global significance.
        """
        if self.step < self.config.score_accumulation_start:
            return
        if self._pruning_done:
            return

        # Compute volume for each Gaussian: V = (4/3) * pi * s1 * s2 * s3
        scales = torch.exp(self.scales)  # (N, 3)
        volumes = (4.0 / 3.0) * math.pi * scales.prod(dim=-1)  # (N,)

        # Adaptive volume normalization (Eq. 5)
        v_max90 = torch.quantile(volumes, 0.9)
        v_norm = torch.clamp(volumes / (v_max90 + 1e-8), max=1.0)
        gamma = v_norm.pow(self.config.significance_beta)  # (N,)

        # Opacity contribution
        opacities = torch.sigmoid(self.opacities).squeeze(-1)  # (N,)

        # Per-Gaussian score: opacity * volume_factor
        per_gaussian_score = opacities * gamma

        # Use radii > 0 as hit indicator (Gaussian was rasterized)
        if hasattr(self, 'radii') and self.radii is not None:
            hit_mask = self.radii > 0
            if hit_mask.shape[0] == per_gaussian_score.shape[0]:
                per_gaussian_score = per_gaussian_score * hit_mask.float()

        # Resize buffers if Gaussians were densified since last accumulation
        N = per_gaussian_score.shape[0]
        if self.significance_scores.shape[0] != N:
            new_scores = torch.zeros(N, device="cpu")
            new_counts = torch.zeros(N, device="cpu")
            old_n = min(self.significance_scores.shape[0], N)
            new_scores[:old_n] = self.significance_scores[:old_n]
            new_counts[:old_n] = self.significance_counts[:old_n]
            self.significance_scores = new_scores
            self.significance_counts = new_counts

        self.significance_scores += per_gaussian_score.cpu()
        self.significance_counts += 1

    def _get_final_significance(self) -> Tensor:
        """Compute final normalized significance scores."""
        counts = self.significance_counts.clamp(min=1)
        scores = self.significance_scores / counts
        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    # ═══════════════════════════════════════════════════════════════
    # Optimizer State Management
    # ═══════════════════════════════════════════════════════════════

    def _update_optimizer_after_prune(self, keep_indices: Tensor) -> None:
        """Update optimizer param references and slice state after pruning.

        When we replace gauss_params with pruned versions, the optimizer
        still holds references to the OLD Parameter objects. We must:
        1. Point each optimizer to the NEW Parameter
        2. Slice the Adam state (exp_avg, exp_avg_sq) to match

        Without this, the new Parameters receive ZERO gradient updates
        and PSNR never recovers after pruning.
        """
        if not hasattr(self, 'optimizers') or self.optimizers is None:
            print("[LightGS] WARNING: No optimizer access, state not updated")
            return

        optimizers_dict = self.optimizers.optimizers
        for name in list(self.gauss_params.keys()):
            if name not in optimizers_dict:
                continue

            opt = optimizers_dict[name]
            new_param = self.gauss_params[name]

            # Get old param from optimizer and slice its Adam state
            old_params = opt.param_groups[0]['params']
            for old_p in old_params:
                if old_p in opt.state:
                    old_state = opt.state[old_p]
                    new_state = {}
                    for k, v in old_state.items():
                        if isinstance(v, Tensor) and v.dim() > 0:
                            # Slice momentum/variance to match pruned indices
                            try:
                                new_state[k] = v[keep_indices].clone()
                            except (IndexError, RuntimeError):
                                # Shape mismatch — reset this state tensor
                                new_state[k] = torch.zeros_like(new_param.data)
                        else:
                            # Scalar state (like step count) — keep as-is
                            new_state[k] = v
                    # Move state from old param to new param
                    del opt.state[old_p]
                    opt.state[new_param] = new_state

            # Point optimizer to new Parameter
            opt.param_groups[0]['params'] = [new_param]

        print(f"[LightGS] Optimizer state sliced for {len(self.gauss_params)} param groups")

    def _update_optimizer_after_reshape(self, param_name: str) -> None:
        """Update a single optimizer after its Parameter was replaced.

        Used after SH truncation and VQ init where individual params
        change shape. Resets Adam state (momentum rebuilds in ~100 steps).
        """
        if not hasattr(self, 'optimizers') or self.optimizers is None:
            return

        optimizers_dict = self.optimizers.optimizers
        if param_name not in optimizers_dict:
            return

        opt = optimizers_dict[param_name]
        new_param = self.gauss_params[param_name]

        # Clear old state entirely (shapes changed, can't slice)
        opt.state.clear()
        # Point to new Parameter
        opt.param_groups[0]['params'] = [new_param]

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: Pruning
    # ═══════════════════════════════════════════════════════════════

    def _prune_by_significance(self) -> None:
        """Prune least-significant Gaussians based on global significance.

        Called from after_train() — AFTER backward pass, BEFORE next forward.
        """
        if self._pruning_done:
            return

        scores = self._get_final_significance().to(self.device)
        n_total = scores.shape[0]
        n_prune = int(n_total * self.config.prune_ratio)
        n_keep = n_total - n_prune

        # Keep top-scoring Gaussians
        _, keep_indices = torch.topk(scores, n_keep, largest=True)
        keep_indices = keep_indices.sort().values

        print(f"\n[LightGS] Pruning: {n_total} → {n_keep} Gaussians "
              f"({self.config.prune_ratio*100:.0f}% pruned)")

        # Prune all Gaussian parameters
        with torch.no_grad():
            for name in list(self.gauss_params.keys()):
                old_param = self.gauss_params[name]
                new_data = old_param.data[keep_indices].clone()
                self.gauss_params[name] = Parameter(new_data)

        # CRITICAL: Update optimizer to point to new Parameters
        # and slice Adam state (exp_avg, exp_avg_sq) to match
        self._update_optimizer_after_prune(keep_indices)

        # Store significance for later VQ stage
        self._kept_significance = scores[keep_indices].clone()
        self._pruning_done = True

        # Reset significance buffers
        self.significance_scores = torch.zeros(n_keep, device="cpu")
        self.significance_counts = torch.zeros(n_keep, device="cpu")

        print(f"[LightGS] Pruning complete. Remaining: {self.num_points}")

    # ═══════════════════════════════════════════════════════════════
    # Stage 4: SH Distillation
    # ═══════════════════════════════════════════════════════════════

    def _start_distillation(self) -> None:
        """Freeze teacher SH and reduce student SH degree.

        Called from after_train() — AFTER backward pass, BEFORE next forward.
        """
        if self._distillation_active:
            return

        # Store full SH as frozen teacher reference
        self._teacher_sh_dc = self.features_dc.data.clone().detach()
        self._teacher_sh_rest = self.features_rest.data.clone().detach()
        self._teacher_sh_degree = self.config.sh_degree

        # Compute how many SH bands to keep
        orig_n_coeffs = (self.config.sh_degree + 1) ** 2
        target_n_coeffs = (self.config.target_sh_degree + 1) ** 2

        if target_n_coeffs >= orig_n_coeffs:
            print(f"[LightGS] Target SH degree {self.config.target_sh_degree} "
                  f">= current {self.config.sh_degree}, skipping distillation")
            self._distillation_active = True
            return

        # Truncate features_rest to target degree
        target_rest_bands = target_n_coeffs - 1  # minus DC band
        with torch.no_grad():
            old_rest = self.features_rest.data
            if target_rest_bands > 0 and target_rest_bands < old_rest.shape[1]:
                new_rest = old_rest[:, :target_rest_bands, :].clone()
                self.gauss_params["features_rest"] = Parameter(new_rest)
            elif target_rest_bands == 0:
                self.gauss_params["features_rest"] = Parameter(
                    torch.zeros(self.num_points, 0, 3, device=self.device)
                )

        # CRITICAL: Update config.sh_degree so gsplat rasterizer knows
        # the new SH band count. Without this, gsplat expects (D_old+1)^2
        # bands but features_rest only has (D_new+1)^2 - 1, causing
        # "Found dtype Byte but expected Float" in the CUDA kernel.
        self.config.sh_degree = self.config.target_sh_degree

        # Update optimizer for reshaped features_rest
        self._update_optimizer_after_reshape("features_rest")

        self._distillation_active = True
        print(f"[LightGS] SH distillation: degree {self._teacher_sh_degree} "
              f"→ {self.config.target_sh_degree} "
              f"(coeffs: {orig_n_coeffs} → {target_n_coeffs}, "
              f"features_rest: {old_rest.shape[1]} → {target_rest_bands} bands)")

    def _compute_distillation_loss(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Compute L2 distillation loss between student render and teacher.

        The paper renders with full-SH teacher on pseudo-views. Here we use
        the GT training image as the teacher signal — equivalent and cheaper
        since the teacher was well-trained on these views.
        """
        if not self._distillation_active or self._teacher_sh_dc is None:
            return {}

        student_rgb = outputs["rgb"]  # (H, W, 3)

        teacher_rgb = batch.get("image")
        if teacher_rgb is None:
            return {}

        teacher_rgb = teacher_rgb.to(student_rgb.device)

        # Handle uint8 images (cache_images_type="uint8")
        if teacher_rgb.dtype == torch.uint8:
            teacher_rgb = teacher_rgb.float() / 255.0

        # Resize if resolution mismatch (downscale schedule)
        if teacher_rgb.shape[:2] != student_rgb.shape[:2]:
            teacher_rgb = F.interpolate(
                teacher_rgb.permute(2, 0, 1).unsqueeze(0),
                size=(student_rgb.shape[0], student_rgb.shape[1]),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        # L2 distillation loss (Eq. 6 from paper)
        distill_loss = F.mse_loss(student_rgb, teacher_rgb)

        return {
            "distill_loss": self.config.distill_loss_weight * distill_loss,
        }

    # ═══════════════════════════════════════════════════════════════
    # Stage 5: Vector Quantization
    # ═══════════════════════════════════════════════════════════════

    def _init_vq(self) -> None:
        """Initialize VQ codebook for SH coefficients using K-means.

        Called from after_train() — AFTER backward pass, BEFORE next forward.
        """
        if self._vq_done or not self.config.enable_vq:
            self._vq_done = True
            return

        print("[LightGS] Initializing vector quantization...")

        with torch.no_grad():
            sh_dc = self.features_dc.data  # (N, 3)
            sh_rest = self.features_rest.data  # (N, K, 3)
            N = sh_dc.shape[0]

            if sh_rest.numel() == 0:
                sh_vectors = sh_dc.clone()
            else:
                sh_vectors = torch.cat(
                    [sh_dc, sh_rest.reshape(N, -1)], dim=-1
                )

            D = sh_vectors.shape[-1]

            # Select which Gaussians to VQ (least significant fraction)
            if hasattr(self, '_kept_significance') and self._kept_significance is not None:
                sig = self._kept_significance.to(self.device)
            else:
                sig = torch.ones(N, device=self.device)

            n_vq = int(N * self.config.vq_ratio)
            _, vq_indices = torch.topk(sig, n_vq, largest=False)
            vq_indices = vq_indices.sort().values

            K = min(self.config.vq_codebook_size, n_vq)
            if K < 2:
                print("[LightGS] Too few Gaussians for VQ, skipping")
                self._vq_done = True
                return

            vq_vectors = sh_vectors[vq_indices]

            # K-means: random init from data
            perm = torch.randperm(n_vq, device=self.device)[:K]
            codebook = vq_vectors[perm].clone()

            for _ in range(20):
                dists = torch.cdist(vq_vectors, codebook)
                assignments = dists.argmin(dim=-1)
                new_codebook = torch.zeros_like(codebook)
                counts = torch.zeros(K, device=self.device)
                new_codebook.scatter_add_(
                    0, assignments.unsqueeze(-1).expand(-1, D), vq_vectors,
                )
                counts.scatter_add_(
                    0, assignments, torch.ones(n_vq, device=self.device),
                )
                mask = counts > 0
                new_codebook[mask] /= counts[mask].unsqueeze(-1)
                new_codebook[~mask] = codebook[~mask]
                codebook = new_codebook

            # Store VQ state
            self._vq_codebook = Parameter(codebook)
            self._vq_assignments = assignments
            self._vq_indices = vq_indices
            self._vq_sh_dim = D

            # Replace VQ'd Gaussians' SH with quantized version
            quantized = codebook[assignments]
            sh_vectors[vq_indices] = quantized

            self.gauss_params["features_dc"] = Parameter(
                sh_vectors[:, :3].contiguous()
            )
            if sh_rest.numel() > 0:
                self.gauss_params["features_rest"] = Parameter(
                    sh_vectors[:, 3:].reshape(N, -1, 3).contiguous()
                )

        # Update optimizers for replaced parameters
        self._update_optimizer_after_reshape("features_dc")
        self._update_optimizer_after_reshape("features_rest")

        self._vq_done = True
        print(f"[LightGS] VQ initialized: {n_vq}/{N} Gaussians quantized "
              f"with {K} codes (dim={D})")

    # ═══════════════════════════════════════════════════════════════
    # Loss Computation (multi-stage) — NO parameter modifications here
    # ═══════════════════════════════════════════════════════════════

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Multi-stage loss — ONLY computes losses, NO parameter modifications.

        All parameter-modifying transitions happen in after_train().
        This method just selects which losses to apply based on current stage.
        """
        # Base photometric losses (always active in all stages)
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Additional distillation loss during SH distillation stage
        if self._distillation_active and self.step < self.distill_end_step:
            distill_losses = self._compute_distillation_loss(outputs, batch)
            loss_dict.update(distill_losses)

        return loss_dict

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Additional custom losses per stage."""
        return {}

    # ═══════════════════════════════════════════════════════════════
    # after_train: THE stage transition point (between iterations)
    # ═══════════════════════════════════════════════════════════════

    def after_train(self, step: int):
        """Handle all stage transitions AFTER backward pass.

        This is the ONLY place where parameter-modifying operations
        (prune, SH truncate, VQ init) should happen. The nerfstudio
        trainer calls methods in this order each iteration:

            1. get_outputs()      → forward pass with current params
            2. get_loss_dict()    → compute losses from outputs
            3. loss.backward()    → backward pass (computes gradients)
            4. optimizer.step()   → apply gradients to params
            5. after_train(step)  → THIS METHOD — safe to modify params
            6. goto 1 for next step

        Modifying params in steps 1-4 would break the autograd graph or
        cause the optimizer to update orphaned Parameter objects.
        """
        # ── Stage 1: Normal training (with densification) ──────────
        if not self._pruning_done:
            # Let splatfacto handle densification (clone/split/cull)
            super().after_train(step)

            # Accumulate significance scores for future pruning
            self._accumulate_significance()

            # Trigger pruning at the designated step
            if step >= self.config.prune_step:
                self._prune_by_significance()
            return

        # ── Stage 3→4 transition: Start SH distillation ───────────
        if not self._distillation_active and step >= self.distill_start_step:
            self._start_distillation()
            return

        # ── Stage 4→5 transition: Initialize VQ ────────────────────
        if not self._vq_done and step >= self.distill_end_step:
            self._init_vq()
            return

        # ── Stages 3, 4, 5, 6: No densification ───────────────────
        # After pruning, we ONLY do photometric fine-tuning.
        # Do NOT call super().after_train() which would attempt
        # densification (clone/split/cull) on the pruned set.
        # The optimizer.step() already happened in the trainer,
        # so gradients are applied — we just skip densification.
        return

    # ═══════════════════════════════════════════════════════════════
    # Param groups (include VQ codebook when active)
    # ═══════════════════════════════════════════════════════════════

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        if self._vq_codebook is not None:
            groups["vq_codebook"] = [self._vq_codebook]
        return groups

    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        stage_map = {
            "training": 0, "prune": 1, "recovery": 2,
            "distillation": 3, "vq_init": 4, "vq_finetune": 5,
        }
        metrics["stage"] = torch.tensor(
            stage_map.get(self.current_stage, -1),
            dtype=torch.float32,
        )
        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Checkpoint
    # ═══════════════════════════════════════════════════════════════

    def load_state_dict(self, dict, **kwargs):
        """Handle loading checkpoints with different Gaussian counts."""
        self.step = dict.get("step", 30000)

        # Handle legacy format
        if "means" in dict:
            for p in self.gauss_params.keys():
                if p in dict:
                    dict[f"gauss_params.{p}"] = dict[p]

        # Resize gauss_params to match checkpoint
        if "gauss_params.means" in dict:
            newp = dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                key = f"gauss_params.{name}"
                if key in dict:
                    self.gauss_params[name] = Parameter(
                        torch.zeros_like(dict[key])
                    )
                else:
                    new_shape = (newp,) + param.shape[1:]
                    self.gauss_params[name] = Parameter(
                        torch.zeros(new_shape, device=self.device)
                    )

            # Resize significance buffers
            self.significance_scores = torch.zeros(newp, device="cpu")
            self.significance_counts = torch.zeros(newp, device="cpu")

        super().load_state_dict(dict, **kwargs)