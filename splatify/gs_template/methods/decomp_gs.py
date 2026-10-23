"""
Decomposing Densification in Gaussian Splatting (DecompGS)
============================================================
Paper: "Decomposing Densification in Gaussian Splatting for
        Faster 3D Scene Reconstruction"
Complexity: ★★★☆☆ (modified densification + multi-res training + adaptive pruning)

Three contributions:
    1. G2L (Global-to-Local densification):
       - Global phase (iters 0..T_2): split-only → spatial diffusion
       - Local phase  (iters T_2..end): split+clone → detail refinement
       - Phase boundary T_2 = end of scale-2 training = schedule[1]

    2. C2F (Coarse-to-Fine multi-resolution training):
       - T_r = Round((E_r / E_1) * T_densify)  [end iter for scale r]
       - Scale r trains from T_{r+1} to T_r (coarse → fine)
       - Reduce positional LR after restoring full resolution

    3. Adaptive opacity pruning:
       - τ = min(α_sorted[⌊N·p⌋], τ_u)
       - Replaces FIXED prune threshold (0.005 → adaptive τ)
       - Opacity RESET is kept (it gives Gaussians a second chance)

Train:
    ns-train decomp-gs --data <path>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as torchF
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# FFT Energy Computation
# ═══════════════════════════════════════════════════════════════════

def compute_fft_energy(image: Tensor) -> float:
    """E(I) = |sqrt(Re(F(I))² + Im(F(I))²)|₁"""
    if image.dim() == 3:
        img = image.permute(2, 0, 1).float()
    else:
        img = image.unsqueeze(0).float()
    fft = torch.fft.fft2(img)
    energy = torch.sqrt(fft.real ** 2 + fft.imag ** 2)
    return energy.abs().sum().item()


def compute_c2f_schedule(
    image_paths: List[Path],
    max_scale_factor: int = 4,
    densify_iters: int = 25000,
    max_images_for_energy: int = 20,
) -> Dict[int, int]:
    """Compute C2F schedule from actual training images via FFT energy.

    Paper: T_r = Round((E_r / E_1) * T_densify)
    T_r = END iteration for scale r. Scale r trains from T_{r+1} to T_r.

    Returns: {scale_factor: start_iteration}
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    step = max(1, len(image_paths) // max_images_for_energy)
    sampled_paths = image_paths[::step][:max_images_for_energy]

    # Build scale factor list: K, K/2, ..., 2, 1
    scale_factors = []
    r = max_scale_factor
    while r >= 1:
        scale_factors.append(r)
        r //= 2
    if scale_factors[-1] != 1:
        scale_factors.append(1)

    sorted_sfs = sorted(scale_factors, reverse=True)  # coarsest first
    CONSOLE.print(f"[C2F] Computing FFT energy for scales {sorted_sfs}...")

    # Compute average energy at each resolution
    energies = {}
    for sf in scale_factors:
        total_energy = 0.0
        count = 0
        for path in sampled_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = TF.to_tensor(img).permute(1, 2, 0)
                if sf > 1:
                    h, w = img_tensor.shape[:2]
                    new_h, new_w = max(1, h // sf), max(1, w // sf)
                    img_down = torchF.interpolate(
                        img_tensor.permute(2, 0, 1).unsqueeze(0),
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ).squeeze(0).permute(1, 2, 0)
                else:
                    img_down = img_tensor
                energy = compute_fft_energy(img_down)
                energy *= sf ** 2  # Paper: E_r = |E(I_r)|₁ · r²
                total_energy += energy
                count += 1
            except Exception:
                continue
        energies[sf] = total_energy / max(count, 1)

    e_full = max(energies.get(1, 1.0), 1e-10)

    # T_r = Round((E_r / E_1) * T_densify) — END iteration for scale r
    t_end = {}
    for sf in scale_factors:
        t_end[sf] = round((energies[sf] / e_full) * densify_iters)
    t_end[1] = densify_iters  # T_1 always = T_densify

    # Enforce monotonicity: T_coarser < T_finer
    for i in range(len(sorted_sfs) - 1):
        coarser, finer = sorted_sfs[i], sorted_sfs[i + 1]
        if t_end[coarser] >= t_end[finer]:
            t_end[coarser] = max(t_end[finer] - 500, 0)

    # Convert to start iterations
    schedule = {sorted_sfs[0]: 0}  # Coarsest starts at 0
    for i in range(1, len(sorted_sfs)):
        schedule[sorted_sfs[i]] = t_end[sorted_sfs[i - 1]]

    CONSOLE.print(f"[C2F] T_end: {t_end}")
    CONSOLE.print(f"[C2F] Schedule (start iters): {schedule}")
    CONSOLE.print(
        f"[C2F] Energy ratios: "
        f"{ {sf: f'{e/e_full:.3f}' for sf, e in energies.items()} }"
    )
    return schedule


def get_default_c2f_schedule(
    max_scale_factor: int = 4,
    densify_iters: int = 25000,
) -> Dict[int, int]:
    """Approximate C2F schedule when images aren't accessible.

    Uses empirical energy ratios for natural images.
    Returns: {scale_factor: start_iteration}
    """
    scale_factors = []
    r = max_scale_factor
    while r >= 1:
        scale_factors.append(r)
        r //= 2
    if scale_factors[-1] != 1:
        scale_factors.append(1)
    sorted_sfs = sorted(scale_factors, reverse=True)

    # Empirical E_r/E_1 for natural images (conservative)
    # These give less time at low res than my v1/v2 estimates
    energy_ratios = {
        1: 1.0,
        2: 0.70,
        4: 0.30,
        8: 0.10,
    }

    # T_r = Round(ratio * T_densify) — end iteration for scale r
    t_end = {}
    for sf in sorted_sfs:
        ratio = energy_ratios.get(sf, max(0.05, 1.0 / sf))
        t_end[sf] = round(ratio * densify_iters)
    t_end[1] = densify_iters

    # Enforce monotonicity
    for i in range(len(sorted_sfs) - 1):
        coarser, finer = sorted_sfs[i], sorted_sfs[i + 1]
        if t_end[coarser] >= t_end[finer]:
            t_end[coarser] = max(t_end[finer] - 500, 0)

    # Convert to start iterations
    schedule = {sorted_sfs[0]: 0}
    for i in range(1, len(sorted_sfs)):
        schedule[sorted_sfs[i]] = t_end[sorted_sfs[i - 1]]

    return schedule


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DecompGSModelConfig(GSTemplateModelConfig):
    """DecompGS: Decomposed densification for faster 3DGS training."""

    _target: Type = field(default_factory=lambda: DecompGSModel)

    # ── G2L ────────────────────────────────────────────────────────
    enable_g2l: bool = True
    """Enable global-to-local densification decomposition."""

    g2l_phase_boundary: int = -1
    """Manual override for G2L boundary. -1 = auto (T_2 from C2F)."""

    # ── C2F ────────────────────────────────────────────────────────
    enable_c2f: bool = True
    """Enable coarse-to-fine multi-resolution training."""

    max_scale_factor: int = 4
    """Maximum downscale factor K. Paper=8, but 4 is safer with splatfacto
    since splatfacto natively goes to full-res by step 500."""

    densify_until_iter: int = 25000
    """Total densification iterations T_densify. Paper: 25K."""

    use_fft_energy: bool = True
    """Compute FFT energy from images (requires setup_c2f_from_images).
    Falls back to approximate schedule if not called."""

    # ── Adaptive Pruning ───────────────────────────────────────────
    enable_adaptive_pruning: bool = True
    """Replace fixed prune threshold with adaptive τ."""

    pruning_ratio: float = 0.03
    """Percentile p. Paper: 0.03."""

    pruning_upper_limit: float = 0.05
    """Upper bound τ_u. Paper: 0.05."""

    # ── LR ─────────────────────────────────────────────────────────
    position_lr_reduction: float = 0.5
    """Reduce positional LR by this factor after restoring full res."""


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class DecompGSModel(GSTemplateModel):
    """Decomposed Densification Gaussian Splatting.

    Key differences from vanilla splatfacto:
        1. C2F: energy-based resolution schedule (less time at low-res)
        2. G2L: split-only until T_2, then split+clone
        3. Adaptive pruning: τ = min(α_sorted[⌊N·p⌋], τ_u)
        4. Opacity RESET is KEPT (splatfacto-specific; paper's 3DGS-accel
           may differ, but reset helps with splatfacto's optimizer dynamics)
    """

    config: DecompGSModelConfig

    def populate_modules(self):
        super().populate_modules()

        self._c2f_schedule: Optional[Dict[int, int]] = None
        self._c2f_computed = False
        self._original_grow_scale3d: Optional[float] = None
        self._full_res_lr_adjusted = False
        self._prev_downscale: int = self.config.max_scale_factor
        self._logged_phase_switch = False

        if self.config.enable_c2f:
            self._c2f_schedule = get_default_c2f_schedule(
                self.config.max_scale_factor,
                self.config.densify_until_iter,
            )
            CONSOLE.print(f"[DecompGS] C2F schedule: {self._c2f_schedule}")
            CONSOLE.print(
                f"[DecompGS] G2L boundary (T_2): iter {self.g2l_phase_boundary}"
            )

    def setup_c2f_from_images(self, image_paths: List[Path]):
        """Compute C2F schedule from FFT energy of training images."""
        if self.config.use_fft_energy and len(image_paths) > 0:
            try:
                self._c2f_schedule = compute_c2f_schedule(
                    image_paths,
                    self.config.max_scale_factor,
                    self.config.densify_until_iter,
                )
                self._c2f_computed = True
                CONSOLE.print("[DecompGS] ✓ C2F from FFT energy")
                CONSOLE.print(
                    f"[DecompGS] G2L boundary: iter {self.g2l_phase_boundary}"
                )
            except Exception as e:
                CONSOLE.print(f"[DecompGS] FFT failed ({e}), using approx schedule")

    # ── G2L Phase Boundary ─────────────────────────────────────────

    @property
    def g2l_phase_boundary(self) -> int:
        """G2L boundary = T_2 = end of scale-2 training = start of scale-1.

        Paper: "Use iteration T_2 as the boundary between global
        spread and local refinement."

        In our schedule dict, T_2 = schedule[1] (start of full-res phase).
        NOT schedule[2] — that's T_{K/2} (start of scale-2 phase).
        """
        if self.config.g2l_phase_boundary > 0:
            return self.config.g2l_phase_boundary

        # T_2 = start of scale-1 training = schedule[1]
        if self._c2f_schedule is not None and 1 in self._c2f_schedule:
            return self._c2f_schedule[1]

        # Fallback: 75% of densification (paper: G2L global phase is long)
        return int(self.config.densify_until_iter * 0.75)

    # ── C2F: Resolution Schedule ───────────────────────────────────

    def _get_downscale_factor(self):
        """C2F energy-guided resolution: coarse → fine during densification."""
        if not self.training:
            return 1

        if not self.config.enable_c2f or self._c2f_schedule is None:
            return super()._get_downscale_factor()

        step = self.step
        if step >= self.config.densify_until_iter:
            return 1

        # Find active scale: last entry where start_iter <= step
        current_scale = 1
        for sf, start_iter in sorted(
            self._c2f_schedule.items(), key=lambda x: x[1]
        ):
            if step >= start_iter:
                current_scale = sf

        return current_scale

    # ── G2L + Adaptive Pruning ─────────────────────────────────────

    def after_train(self, step: int):
        """G2L densification control + adaptive pruning threshold.

        Runs BEFORE super().after_train() which triggers
        strategy.step_post_backward().

        NOTE: Opacity reset is KEPT. The paper's "adaptive pruning"
        replaces the fixed PRUNE THRESHOLD (0.005 → τ), not the
        periodic reset. Disabling reset with splatfacto causes
        excessive Gaussian death because splatfacto's densification
        creates fewer Gaussians than 3DGS-accel.
        """
        if not hasattr(self, "strategy"):
            super().after_train(step)
            return

        # ── G2L: split-only (global) vs split+clone (local) ───────
        if self.config.enable_g2l:
            if self._original_grow_scale3d is None:
                self._original_grow_scale3d = getattr(
                    self.strategy, "grow_scale3d", 0.01
                )
                CONSOLE.print(
                    f"[DecompGS] Saved grow_scale3d={self._original_grow_scale3d}"
                )

            boundary = self.g2l_phase_boundary
            if step < boundary:
                # GLOBAL PHASE: split-only (suppress cloning)
                # grow_scale3d=0 → max_scale > 0 always true → all split
                self.strategy.grow_scale3d = 0.0
            else:
                # LOCAL PHASE: restore split + clone
                if not self._logged_phase_switch and step >= boundary:
                    CONSOLE.print(
                        f"[DecompGS] G2L → LOCAL phase at step {step} "
                        f"(boundary={boundary}). Restoring grow_scale3d="
                        f"{self._original_grow_scale3d}"
                    )
                    self._logged_phase_switch = True
                self.strategy.grow_scale3d = self._original_grow_scale3d

        # ── Adaptive Pruning: τ = min(α_sorted[⌊N·p⌋], τ_u) ──────
        if self.config.enable_adaptive_pruning:
            self._update_pruning_threshold()

        # ── Run parent (triggers densification / pruning) ──────────
        super().after_train(step)

    def _update_pruning_threshold(self):
        """Set pruning threshold to τ = min(α_sorted[⌊N·p⌋], τ_u)."""
        with torch.no_grad():
            opacities_sig = torch.sigmoid(self.opacities.detach()).squeeze()
            n = opacities_sig.shape[0]
            if n == 0:
                return

            sorted_ops, _ = torch.sort(opacities_sig)
            k = min(int(n * self.config.pruning_ratio), n - 1)
            tau_k = sorted_ops[k].item()
            tau = min(tau_k, self.config.pruning_upper_limit)

            # Try multiple gsplat attribute names
            for attr in ["prune_opa", "prune_alpha", "cull_alpha_thresh"]:
                if hasattr(self.strategy, attr):
                    setattr(self.strategy, attr, tau)
                    return

    # ── LR Reduction at Full-Res Transition ────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Reduce positional LR when transitioning to full resolution."""
        if (
            self.config.enable_c2f
            and not self._full_res_lr_adjusted
            and self._c2f_schedule is not None
            and self.training
        ):
            current_ds = self._get_downscale_factor()
            if current_ds == 1 and self._prev_downscale > 1:
                self._adjust_position_lr()
                self._full_res_lr_adjusted = True
            self._prev_downscale = current_ds

        return super().get_loss_dict(outputs, batch, metrics_dict)

    def _adjust_position_lr(self):
        """Reduce positional LR by configured factor."""
        reduction = self.config.position_lr_reduction
        if reduction >= 1.0:
            return

        adjusted = False
        try:
            if hasattr(self, "optimizers") and self.optimizers is not None:
                for name, opt_dict in self.optimizers.items():
                    if "means" in name or "position" in name:
                        optimizer = (
                            opt_dict["optimizer"]
                            if isinstance(opt_dict, dict)
                            else opt_dict
                        )
                        for pg in optimizer.param_groups:
                            old_lr = pg["lr"]
                            pg["lr"] = old_lr * reduction
                            CONSOLE.print(
                                f"[DecompGS] Position LR: "
                                f"{old_lr:.2e} → {pg['lr']:.2e}"
                            )
                            adjusted = True
        except Exception as e:
            CONSOLE.print(f"[DecompGS] LR adjust failed: {e}")

        if not adjusted:
            CONSOLE.print("[DecompGS] Warning: could not find position optimizer")

    # ── Metrics ────────────────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)

        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        metrics["g2l_is_global"] = torch.tensor(
            1.0 if self.step < self.g2l_phase_boundary else 0.0,
            dtype=torch.float32,
        )
        metrics["downscale_factor"] = torch.tensor(
            float(self._get_downscale_factor()), dtype=torch.float32,
        )

        # Log pruning threshold
        if hasattr(self, "strategy"):
            tau = 0.005  # default
            for attr in ["prune_opa", "prune_alpha", "cull_alpha_thresh"]:
                if hasattr(self.strategy, attr):
                    tau = getattr(self.strategy, attr)
                    break
            metrics["prune_threshold"] = torch.tensor(
                float(tau), dtype=torch.float32
            )

        return metrics