"""
KFAC-Riemannian-GS — Per-primitive K-FAC with Riemannian retraction (HOOK_5_OPT)
=================================================================================

Idea: idea_0016 (cand_resume_opt). Replaces Adam's diagonal preconditioner with
a per-primitive block-diagonal approximate Fisher. Because nerfstudio instantiates
Adam-per-param-group via TrainerConfig, we leave Adam as the outer step and
precondition gradients BEFORE Adam consumes them via `Tensor.register_hook`.

Per-primitive factors:
  - A_i (3x3): geometry Fisher over means (EMA of outer product).
              Inverse applied to means AND to log-scales.
  - b_i (scalar): photometric "magnitude" factor over opacity + SH
                 (EMA of per-primitive gradient L2^2).
  - quats: tangent-space projection (remove radial component) — quats stay on S^3
           after the existing model-side normalisation in splatfacto.

Factors updated every `kfac_factor_update_freq` steps; Tikhonov damping
`kfac_damping` keeps inversion stable. Split inheritance: children receive
damped copies of the parent's factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from torch import Tensor
from torch.nn import Parameter

from gs_template.model import GSTemplateModel, GSTemplateModelConfig


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KFACRiemannianGSModelConfig(GSTemplateModelConfig):
    """K-FAC Riemannian preconditioner config."""

    _target: Type = field(default_factory=lambda: KFACRiemannianGSModel)

    kfac_damping: float = 1e-3
    """Tikhonov damping λ on the 3x3 geometry factor before inversion."""

    kfac_ema_beta: float = 0.95
    """EMA decay for Fisher factor updates."""

    kfac_factor_update_freq: int = 20
    """Refresh Fisher factors every N steps (amortises cost)."""

    kfac_inherit_alpha: float = 0.3
    """Damping factor applied to inherited factors on split."""

    kfac_photo_scale: float = 1.0
    """Scalar applied to the photometric-factor divide (acts as LR shift)."""

    kfac_enabled_after: int = 500
    """Step after which preconditioning turns on (warmup)."""

    kfac_min_norm: float = 1e-8
    """Floor on factor norms for numerical safety."""


# ═══════════════════════════════════════════════════════════════════
# Preconditioner
# ═══════════════════════════════════════════════════════════════════

class KFACPreconditioner:
    """Per-primitive block-diagonal approximate Fisher.

    Maintained state (all as plain tensors on GPU):
      - A (N,3,3): EMA of geometry gradient outer product (over means).
      - b_photo (N,): EMA of per-primitive photometric grad L2^2 (opac+SH).
      - step: internal counter.
    """

    def __init__(
        self,
        num_primitives: int,
        device: torch.device,
        damping: float,
        ema_beta: float,
        update_freq: int,
        photo_scale: float,
        min_norm: float,
    ):
        self.damping = damping
        self.ema_beta = ema_beta
        self.update_freq = update_freq
        self.photo_scale = photo_scale
        self.min_norm = min_norm

        self.A = torch.eye(3, device=device).expand(num_primitives, 3, 3).clone()
        self.b_photo = torch.ones(num_primitives, device=device)
        self.step = 0
        self.active = False

    def set_active(self, active: bool):
        self.active = active

    # ── updates ──────────────────────────────────────────────────────

    @torch.no_grad()
    def observe_means_grad(self, grad: Tensor):
        """Update A from a raw means gradient (N,3). EMA of outer products."""
        if self.step % self.update_freq != 0:
            return
        if grad.shape[0] != self.A.shape[0]:
            return
        beta = self.ema_beta
        outer = grad.unsqueeze(-1) * grad.unsqueeze(-2)  # (N,3,3)
        self.A.mul_(beta).add_(outer, alpha=1.0 - beta)

    @torch.no_grad()
    def observe_photo_grad(self, grad: Tensor):
        """Update b_photo from a photometric gradient (N,...). EMA of L2^2."""
        if self.step % self.update_freq != 0:
            return
        if grad.shape[0] != self.b_photo.shape[0]:
            return
        g2 = grad.reshape(grad.shape[0], -1).pow(2).sum(dim=-1)
        beta = self.ema_beta
        self.b_photo.mul_(beta).add_(g2, alpha=1.0 - beta)

    # ── transforms applied to gradients ──────────────────────────────

    @torch.no_grad()
    def precond_geometry(self, grad: Tensor) -> Tensor:
        """Apply A_i^{-1} to a (N,3) gradient. Returns new tensor."""
        if not self.active or grad.shape[0] != self.A.shape[0]:
            return grad
        A = self.A
        lam = self.damping * (A.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1) + self.min_norm)
        A_damped = A + lam * torch.eye(3, device=A.device).expand_as(A)
        try:
            A_inv = torch.linalg.inv(A_damped)
        except Exception:
            return grad
        # (N,3,3) @ (N,3,1) -> (N,3,1)
        out = torch.bmm(A_inv, grad.unsqueeze(-1)).squeeze(-1)
        # Normalise per-primitive scale so we stay on Adam's LR regime.
        nrm_in = grad.norm(dim=-1, keepdim=True) + self.min_norm
        nrm_out = out.norm(dim=-1, keepdim=True) + self.min_norm
        out = out * (nrm_in / nrm_out)
        return out

    @torch.no_grad()
    def precond_photo(self, grad: Tensor) -> Tensor:
        """Divide by sqrt(b_photo + damping) per-primitive; scale preserved."""
        if not self.active or grad.shape[0] != self.b_photo.shape[0]:
            return grad
        b = self.b_photo
        b_mean = b.mean().clamp_min(self.min_norm)
        denom = (b + self.damping * b_mean).clamp_min(self.min_norm).sqrt()
        factor = (b_mean.sqrt() / denom) * self.photo_scale  # (N,)
        while factor.dim() < grad.dim():
            factor = factor.unsqueeze(-1)
        return grad * factor

    @torch.no_grad()
    def retract_quat(self, quat: Tensor, grad: Tensor) -> Tensor:
        """Project grad onto the tangent space of S^3 at `quat`: g - <g,q> q."""
        if not self.active:
            return grad
        if grad.shape != quat.shape:
            return grad
        q = quat / (quat.norm(dim=-1, keepdim=True) + self.min_norm)
        radial = (grad * q).sum(dim=-1, keepdim=True) * q
        return grad - radial

    # ── lifecycle ────────────────────────────────────────────────────

    def tick(self):
        self.step += 1

    @torch.no_grad()
    def resize(self, new_n: int, split_mask: Optional[Tensor] = None, alpha: float = 0.3):
        """Handle densification.

        If split_mask is provided (same length as pre-split count), children
        inherit damped copies of parent factors; kept primitives preserved.
        Otherwise reset to identity (used for prune-without-split paths).
        """
        device = self.A.device
        if split_mask is None:
            self.A = torch.eye(3, device=device).expand(new_n, 3, 3).clone()
            self.b_photo = torch.ones(new_n, device=device)
            return
        old_n = split_mask.shape[0]
        if old_n != self.A.shape[0]:
            self.A = torch.eye(3, device=device).expand(new_n, 3, 3).clone()
            self.b_photo = torch.ones(new_n, device=device)
            return
        keep = ~split_mask
        n_kept = int(keep.sum().item())
        n_children = new_n - n_kept
        if n_children < 0 or n_children % max(1, int(split_mask.sum().item())) != 0:
            # fall back to safe reset
            self.A = torch.eye(3, device=device).expand(new_n, 3, 3).clone()
            self.b_photo = torch.ones(new_n, device=device)
            return
        A_new = torch.eye(3, device=device).expand(new_n, 3, 3).clone()
        b_new = torch.ones(new_n, device=device)
        A_new[:n_kept] = self.A[keep]
        b_new[:n_kept] = self.b_photo[keep]
        mother_A = self.A[split_mask]
        mother_b = self.b_photo[split_mask]
        n_split = int(split_mask.sum().item())
        if n_split > 0:
            copies = n_children // n_split
            damped_A = alpha * mother_A + (1 - alpha) * torch.eye(3, device=device).expand_as(mother_A)
            damped_b = alpha * mother_b + (1 - alpha) * torch.ones_like(mother_b)
            for s in range(copies):
                lo = n_kept + s * n_split
                hi = lo + n_split
                A_new[lo:hi] = damped_A
                b_new[lo:hi] = damped_b
        self.A = A_new
        self.b_photo = b_new


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class KFACRiemannianGSModel(GSTemplateModel):
    """Splatfacto + per-primitive K-FAC Riemannian gradient preconditioning."""

    config: KFACRiemannianGSModelConfig

    def populate_modules(self):
        super().populate_modules()
        device = self.means.device
        self._kfac = KFACPreconditioner(
            num_primitives=self.num_points,
            device=device,
            damping=self.config.kfac_damping,
            ema_beta=self.config.kfac_ema_beta,
            update_freq=self.config.kfac_factor_update_freq,
            photo_scale=self.config.kfac_photo_scale,
            min_norm=self.config.kfac_min_norm,
        )
        self._kfac_hooks_installed = False
        self._kfac_step = 0
        self._install_grad_hooks()

    # ── gradient hooks ───────────────────────────────────────────────

    def _install_grad_hooks(self):
        """Attach .register_hook on each gauss_param so Adam consumes
        the preconditioned gradient. Re-install after densification."""
        if not hasattr(self, "gauss_params"):
            return

        # means (N,3) : geometry factor A^{-1}
        means = self.gauss_params.get("means")
        if means is not None and means.requires_grad:
            def _means_hook(g: Tensor) -> Tensor:
                self._kfac.observe_means_grad(g)
                return self._kfac.precond_geometry(g)
            means.register_hook(_means_hook)

        # scales (N,3) (log-space) : reuse geometry factor A^{-1}
        scales = self.gauss_params.get("scales")
        if scales is not None and scales.requires_grad:
            def _scales_hook(g: Tensor) -> Tensor:
                return self._kfac.precond_geometry(g)
            scales.register_hook(_scales_hook)

        # quats (N,4) : tangent-space retraction
        quats = self.gauss_params.get("quats")
        if quats is not None and quats.requires_grad:
            def _quats_hook(g: Tensor) -> Tensor:
                return self._kfac.retract_quat(quats.data, g)
            quats.register_hook(_quats_hook)

        # opacities (N,1) : photometric scalar factor
        opacities = self.gauss_params.get("opacities")
        if opacities is not None and opacities.requires_grad:
            def _opac_hook(g: Tensor) -> Tensor:
                self._kfac.observe_photo_grad(g)
                return self._kfac.precond_photo(g)
            opacities.register_hook(_opac_hook)

        # features_dc, features_rest : photometric scalar factor (reuse b_photo)
        for name in ("features_dc", "features_rest"):
            p = self.gauss_params.get(name)
            if p is not None and p.requires_grad:
                def _feat_hook(g: Tensor) -> Tensor:
                    return self._kfac.precond_photo(g)
                p.register_hook(_feat_hook)

        self._kfac_hooks_installed = True

    # ── training-time activation ─────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            self._kfac_step += 1
            self._kfac.tick()
            if self._kfac_step >= self.config.kfac_enabled_after:
                self._kfac.set_active(True)
        return loss_dict

    # ── densification wiring ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        # splatfacto's splatfacto replaces gauss_params; re-size preconditioner
        # and re-install hooks against the new parameter tensors.
        new_n = self.num_points
        try:
            self._kfac.resize(
                new_n=new_n,
                split_mask=split_mask.to(self._kfac.A.device) if split_mask is not None else None,
                alpha=self.config.kfac_inherit_alpha,
            )
        except Exception:
            self._kfac.resize(new_n=new_n, split_mask=None)
        self._install_grad_hooks()
        return out

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        try:
            metrics["kfac_A_trace_mean"] = self._kfac.A.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().detach()
            metrics["kfac_b_photo_mean"] = self._kfac.b_photo.mean().detach()
        except Exception:
            pass
        return metrics
