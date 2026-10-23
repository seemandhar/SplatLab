"""
Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting
=====================================================================
Paper: https://arxiv.org/abs/2501.14534
Complexity: ★★★☆☆ (new params, custom rendering, losses, pruning)

Combines five families of training-time tricks to reduce Gaussian
count, storage, and training time while keeping visual quality on
par with vanilla 3DGS:

    1. Learned Gaussian masks with straight-through estimator
    2. Per-Gaussian per-SH-band masking
    3. Gaussian significance score pruning (ray-hit x opacity x volume)
    4. Progressive training (blurring + resolution + Gaussian scale)
    5. Post-training FP16 quantization (handled outside the model)

Extension points used:
    1. populate_modules   — mask parameters + significance buffers
    2. get_outputs        — apply masks during rasterization
    3. compute_custom_losses — mask sparsity losses + progressive blur
    4. split_gaussians    — replicate masks during densification
    5. get_gaussian_param_groups — register mask optimizers
    7. load_state_dict    — resize mask params for checkpoint loading

Train:
    ns-train trick-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from gsplat import rasterization
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.rendering import (
    assemble_outputs,
    crop_custom_param,
    gather_base_params,
    prepare_camera,
)


# ═══════════════════════════════════════════════════════════════════
# Utility: straight-through hard mask
# ═══════════════════════════════════════════════════════════════════

def _straight_through_mask(logits: Tensor, threshold: float) -> Tensor:
    """Binary mask via straight-through estimator (Eq. 3 / Eq. 5).

    Forward pass returns hard {0, 1} values.
    Backward pass uses the smooth sigmoid gradient.
    """
    soft = torch.sigmoid(logits)
    hard = (soft > threshold).float()
    return hard - soft.detach() + soft


def _gaussian_blur_2d(img: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply 2D Gaussian blur to an image tensor (H, W, C).

    Uses separable 1D convolutions for efficiency (same trick the
    paper uses for SSIM).
    """
    if kernel_size <= 1:
        return img
    # Ensure float (images may arrive as uint8 from cache)
    orig_dtype = img.dtype
    if not img.is_floating_point():
        img = img.float() / 255.0
    # Force odd kernel
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Build 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=img.dtype, device=img.device)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    C = img.shape[-1]
    # Reshape for grouped conv: (1, C, H, W)
    x = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    pad = kernel_size // 2

    # Horizontal blur
    k_h = g.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    x = F.conv2d(x, k_h, padding=(0, pad), groups=C)
    # Vertical blur
    k_v = g.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(x, k_v, padding=(pad, 0), groups=C)

    return x.squeeze(0).permute(1, 2, 0)  # back to (H, W, C)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrickGSModelConfig(GSTemplateModelConfig):
    """Trick-GS configuration.

    Groups hyperparameters by the five trick families.
    Defaults follow the paper's recommended settings.
    """

    _target: Type = field(default_factory=lambda: TrickGSModel)

    # ── 1. Learned Gaussian mask ───────────────────────────────────
    mask_init_value: float = 2.0
    """Initial logit for Gaussian masks. sigmoid(2) ~ 0.88 so most
    Gaussians start unmasked."""

    mask_threshold: float = 0.05
    """Threshold epsilon_m for the hard mask (Eq. 3)."""

    lambda_mask: float = 0.05
    """Weight for the Gaussian mask sparsity loss."""

    mask_prune_interval: int = 500
    """After densification ends, prune masked Gaussians every this
    many iterations."""

    # ── 2. Gaussian significance score pruning ─────────────────────
    use_significance_pruning: bool = True
    """Enable significance-based pruning (Section 2 of paper)."""

    sig_prune_times: int = 6
    """Number of significance pruning rounds."""

    sig_first_rate: float = 0.60
    """Fraction of Gaussians removed in first significance round."""

    sig_decay: float = 0.70
    """Multiplicative decay of pruning rate across rounds."""

    sig_start_step: int = 15500
    """Start significance pruning AFTER densification ends. The paper
    interleaves pruning with densification, but significance scores
    are only meaningful once the Gaussian set has stabilised."""

    sig_end_step: int = 22000
    """Significance pruning is spread evenly up to this step."""

    sig_volume_norm_percentile: float = 0.90
    """Normalize volumes by this percentile (avoid floating Gaussians)."""

    # ── 3. SH band masking ─────────────────────────────────────────
    use_sh_masking: bool = True
    """Enable per-Gaussian per-band SH masking (Eq. 5)."""

    sh_mask_init_value: float = 2.0
    """Initial logit for SH masks."""

    sh_mask_threshold: float = 0.10
    """Threshold epsilon_sh for hard SH mask."""

    lambda_sh_mask: float = 0.05
    """Weight for the SH mask sparsity loss."""

    # ── 4. Progressive training ────────────────────────────────────
    use_progressive_blur: bool = True
    """Enable progressive blurring of training images."""

    blur_kernel_start: int = 9
    """Initial blur kernel size (square kernel)."""

    blur_sigma_start: float = 2.4
    """Initial Gaussian sigma for the blur kernel."""

    blur_reduce_every: int = 100
    """Reduce blur kernel every this many iterations."""

    blur_end_step: int = 19500
    """Stop blurring at this iteration."""

    blur_kernel_decay: float = 0.95
    """Multiplicative decay factor for kernel size reduction."""

    # Progressive Gaussian scale (low-pass, Sec. 4c)
    use_progressive_scale: bool = True
    """Add diagonal regularization to projected covariance (Eq. 7)."""

    min_scale_reg: float = 0.3
    """Lower bound on the diagonal regularization s."""

    # ── 5. Densification schedule ──────────────────────────────────
    densify_until_step: int = 15000
    """Standard densification stops here."""

    redensify_start: int = 20000
    """Re-enable densification from this step to recover from
    false-positive pruning."""

    redensify_end: int = 20500
    """Stop re-densification."""

    # Use absolute gradients for densification (AbsGS)
    absgrad: bool = True


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class TrickGSModel(GSTemplateModel):
    """Trick-GS: combines five trick families for efficient 3DGS.

    The only learnable additions to vanilla 3DGS are per-Gaussian
    binary masks and per-SH-band binary masks.  Everything else is
    a training-time schedule or loss term that does not change the
    underlying 3DGS representation.
    """

    config: TrickGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()

        N = self.num_points

        # Learned Gaussian mask (logit space)
        self.gauss_params["gaussian_mask"] = Parameter(
            torch.full((N, 1), self.config.mask_init_value)
        )

        # Per-Gaussian per-SH-band masks (logit space)
        num_bands = self.config.sh_degree  # bands 1 .. sh_degree
        if self.config.use_sh_masking and num_bands > 0:
            self.gauss_params["sh_masks"] = Parameter(
                torch.full((N, num_bands), self.config.sh_mask_init_value)
            )

        # Bookkeeping for significance pruning (not learned)
        self._sig_accum = torch.zeros(N, device="cpu")
        self._sig_view_count = 0
        self._sig_prune_round = 0

        # Track pruning steps for scheduling
        self._mask_prune_steps: List[int] = []

    # ── 2. get_outputs (custom rendering with masks) ───────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render with Gaussian masks and SH masks applied."""

        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        # Gather base Gaussian parameters
        means, quats, scales, opacities, colors = gather_base_params(
            self, crop_ids
        )

        # ── Apply Gaussian mask to opacity ─────────────────────────
        mask_logits = crop_custom_param(
            self.gauss_params["gaussian_mask"], crop_ids
        )
        g_mask = _straight_through_mask(
            mask_logits, self.config.mask_threshold
        )  # (N, 1), hard {0,1}

        # Activated masked opacity: sigmoid(logit) * mask
        activated_opacity = torch.sigmoid(opacities).squeeze(-1) * g_mask.squeeze(-1)

        # ── Apply SH band masks ────────────────────────────────────
        if (
            self.config.use_sh_masking
            and self.config.sh_degree > 0
            and "sh_masks" in self.gauss_params
            and colors.dim() == 3
        ):
            sh_mask_logits = crop_custom_param(
                self.gauss_params["sh_masks"], crop_ids
            )
            sh_hard = _straight_through_mask(
                sh_mask_logits, self.config.sh_mask_threshold
            )  # (N, num_bands)

            # Build a per-coefficient mask tensor (N, K, 1) where K is
            # total SH coefficients.  Band 0 (DC) is always 1.
            # This avoids inplace writes that break autograd.
            coeff_mask = torch.ones(
                colors.shape[0], colors.shape[1], 1,
                device=colors.device, dtype=colors.dtype,
            )
            for l in range(min(self.config.sh_degree, sh_hard.shape[-1])):
                band_start = (l + 1) ** 2
                band_end = (l + 2) ** 2
                if band_end > colors.shape[1]:
                    break
                coeff_mask[:, band_start:band_end, :] = sh_hard[:, l : l + 1].unsqueeze(-1)

            colors = colors * coeff_mask

        # ── Determine render settings ──────────────────────────────
        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if colors.dim() == 2:
            sh_degree = None
        else:
            sh_degree = min(
                self.step // self.config.sh_degree_interval,
                self.config.sh_degree,
            )

        # ── Rasterize (directly, since we have pre-activated opacity)
        render, alpha, info = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=activated_opacity,
            colors=colors,
            viewmats=cam["viewmat"],
            Ks=cam["K"],
            width=cam["W"],
            height=cam["H"],
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=self.config.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        self.info=info
        # ── Accumulate significance scores (training only) ─────────
        if self.training and self.config.use_significance_pruning:
            self._accumulate_significance(info, crop_ids)

        # ── Assemble outputs ───────────────────────────────────────
        outputs = assemble_outputs(self, render, alpha, info, cam)
        return outputs

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses = {}

        # ── Gaussian mask sparsity loss ────────────────────────────
        # Encourage masks toward 0 to prune unnecessary Gaussians.
        # L_m = mean(sigmoid(m))  (from [13], LightGaussian)
        losses["mask_loss"] = (
            self.config.lambda_mask
            * torch.sigmoid(self.gauss_params["gaussian_mask"]).mean()
        )

        # ── SH mask sparsity loss ──────────────────────────────────
        # Weight each band's loss by the number of coefficients in
        # that band: band l has (2l+1) coefficients.
        if (
            self.config.use_sh_masking
            and self.config.sh_degree > 0
            and "sh_masks" in self.gauss_params
        ):
            sh_soft = torch.sigmoid(self.gauss_params["sh_masks"])  # (N, num_bands)
            band_weights = []
            for l in range(sh_soft.shape[-1]):
                band_weights.append(2.0 * (l + 1) + 1.0)
            band_weights = torch.tensor(
                band_weights, device=sh_soft.device, dtype=sh_soft.dtype
            )
            # Normalize weights so the total is independent of SH degree
            band_weights = band_weights / band_weights.sum()
            weighted = (sh_soft * band_weights.unsqueeze(0)).sum(dim=-1).mean()
            losses["sh_mask_loss"] = self.config.lambda_sh_mask * weighted

        return losses

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Override to apply progressive blurring before loss computation
        and trigger step-based pruning."""

        # ── Progressive blurring ───────────────────────────────────
        # Paper blurs *training images* (GT targets) to create smoother
        # loss landscapes early in training, suppressing floaters and
        # reducing the Gaussian count needed.  The rendered output is
        # NOT blurred; the model naturally starts blurry because it
        # has few Gaussians, then sharpens as densification proceeds.
        if self.training and self.config.use_progressive_blur:
            blur_info = self._get_blur_params()
            if blur_info is not None:
                kernel_size, sigma = blur_info
                if kernel_size >= 3 and "image" in batch:
                    batch = dict(batch)  # shallow copy to avoid mutating
                    batch["image"] = _gaussian_blur_2d(
                        batch["image"], kernel_size, sigma
                    )

        # Standard losses + custom losses
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # ── Step-based pruning triggers ────────────────────────────
        if self.training:
            self._maybe_prune_masks()
            self._maybe_prune_significance()

        return loss_dict

    # ── 4. split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)

        # Replicate Gaussian masks for split Gaussians
        out["gaussian_mask"] = self.gauss_params["gaussian_mask"][
            split_mask
        ].repeat(samps, 1)

        # Replicate SH masks
        if "sh_masks" in self.gauss_params:
            out["sh_masks"] = self.gauss_params["sh_masks"][
                split_mask
            ].repeat(samps, 1)

        return out

    # ── 5. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }
        return groups

    # ── 8. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        # Reinitialize tracking buffers
        self._sig_accum = torch.zeros(newp, device="cpu")
        self._sig_view_count = 0
        self._sig_prune_round = 0
        super().load_state_dict(dict, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    # Internal: progressive blur schedule
    # ═══════════════════════════════════════════════════════════════

    def _get_blur_params(self) -> Optional[Tuple[int, float]]:
        """Compute current blur kernel size and sigma based on step.

        Returns None if blur is disabled at this step.
        """
        step = self.step
        if step >= self.config.blur_end_step:
            return None

        # Number of reductions so far
        n_reductions = step // self.config.blur_reduce_every
        decay = self.config.blur_kernel_decay ** n_reductions

        kernel_size = max(1, int(self.config.blur_kernel_start * decay))
        sigma = max(0.5, self.config.blur_sigma_start * decay)

        # Make sure kernel is odd
        if kernel_size % 2 == 0:
            kernel_size -= 1

        if kernel_size < 3:
            return None

        return kernel_size, sigma

    # ═══════════════════════════════════════════════════════════════
    # Internal: significance score accumulation
    # ═══════════════════════════════════════════════════════════════

    def _accumulate_significance(
        self, info: Dict, crop_ids: Optional[Tensor]
    ):
        """Accumulate per-Gaussian significance from the current render.

        Significance(j) = hit_count(j) * alpha(j) * volume(j)

        We approximate hit count per step as (radii > 0), and accumulate
        across training steps.
        """
        with torch.no_grad():
            radii = info.get("radii", None)
            if radii is None:
                return

            # Binary visibility: was this Gaussian rendered?
            # gsplat returns radii as (B, N) where B=1; squeeze it.
            if radii.dim() > 1:
                radii = radii.squeeze(0)
            visible = (radii > 0).float()  # (N_crop,)

            # Opacity contribution
            if crop_ids is not None:
                opacity = torch.sigmoid(
                    self.opacities[crop_ids]
                ).squeeze(-1)
                volume = torch.exp(
                    self.scales[crop_ids]
                ).prod(dim=-1)
            else:
                opacity = torch.sigmoid(self.opacities).squeeze(-1)
                volume = torch.exp(self.scales).prod(dim=-1)

            # Normalize volume by 90th percentile (paper Sec. 2)
            v_thresh = torch.quantile(
                volume, self.config.sig_volume_norm_percentile
            )
            volume = volume / (v_thresh + 1e-8)
            volume = volume.clamp(max=1.0)

            score = visible * opacity * volume  # (N_crop,)
            score = score.squeeze()  # remove any remaining batch dims

            # Accumulate into full-size buffer
            N = self.num_points
            if self._sig_accum.shape[0] != N:
                self._sig_accum = torch.zeros(N, device="cpu")
                self._sig_view_count = 0

            score_cpu = score.detach().cpu()
            if crop_ids is not None:
                self._sig_accum[crop_ids.cpu()] += score_cpu
            else:
                self._sig_accum += score_cpu
            self._sig_view_count += 1

    # ═══════════════════════════════════════════════════════════════
    # Internal: mask-based pruning
    # ═══════════════════════════════════════════════════════════════

    def _maybe_prune_masks(self):
        """Prune Gaussians whose learned mask is below threshold.

        Schedule:
          - During densification: prune at every densification step
          - After densification: prune every mask_prune_interval steps
        """
        step = self.step
        interval = self.config.mask_prune_interval

        should_prune = False
        # After densification stops, periodic mask pruning
        if step > self.config.densify_until_step and step % interval == 0:
            should_prune = True
        # During re-densification window, also prune
        if (
            self.config.redensify_start <= step <= self.config.redensify_end
            and step % 100 == 0
        ):
            should_prune = True

        if not should_prune:
            return

        with torch.no_grad():
            mask_soft = torch.sigmoid(
                self.gauss_params["gaussian_mask"].squeeze(-1)
            )
            prune_mask = mask_soft < self.config.mask_threshold

            n_pruned = prune_mask.sum().item()
            if n_pruned == 0 or n_pruned >= self.num_points - 10:
                return

            # Set opacity of pruned Gaussians to very low value.
            # Splatfacto's built-in culling will remove them.
            self.opacities.data[prune_mask] = -100.0

    # ═══════════════════════════════════════════════════════════════
    # Internal: significance-based pruning
    # ═══════════════════════════════════════════════════════════════

    def _maybe_prune_significance(self):
        """Prune lowest-significance Gaussians on schedule.

        6 rounds spread evenly between sig_start_step and sig_end_step,
        with decaying pruning rates starting at sig_first_rate.
        Pruning starts only after densification has stabilised.
        """
        if not self.config.use_significance_pruning:
            return
        if self._sig_prune_round >= self.config.sig_prune_times:
            return

        step = self.step
        start = self.config.sig_start_step
        end = self.config.sig_end_step

        if step < start:
            return

        # Spread rounds evenly across [start, end]
        window = end - start
        interval = window // (self.config.sig_prune_times)
        target_step = start + (self._sig_prune_round + 1) * interval

        if step < target_step:
            return

        with torch.no_grad():
            N = self.num_points
            if self._sig_view_count < 10 or N < 100:
                return

            # Compute per-Gaussian significance score
            scores = self._sig_accum[: N].to(self.device)
            scores = scores / (self._sig_view_count + 1e-8)

            # Compute pruning rate for this round
            rate = self.config.sig_first_rate * (
                self.config.sig_decay ** self._sig_prune_round
            )
            rate = min(rate, 0.5)  # safety cap

            # Find threshold: prune the bottom `rate` fraction
            k = max(1, int(N * rate))
            if k >= N - 10:
                self._sig_prune_round += 1
                return

            threshold = torch.kthvalue(scores, k).values
            prune_mask = scores <= threshold

            # Set opacity very low so splatfacto culling removes them
            self.opacities.data[prune_mask] = -100.0

            self._sig_prune_round += 1

            # Reset accumulation for next round
            self._sig_accum = torch.zeros(N, device="cpu")
            self._sig_view_count = 0