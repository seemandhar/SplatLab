"""
gs_template/optimizers.py — BOGausS Custom Optimizer
======================================================

Implements the modified Adam optimizer from the BOGausS paper:
  - Filtered gradient states with α_{s,t} = max(1/t, 1 - β_s*)
  - Precision-aware Δ_θ scaling for position and size parameters
  - Per-Gaussian time counters (sparse, unbiased)
  - Optimizer state inheritance after Gaussian splitting

Usage:
    optimizer = BOGausSAdam(params, lr=1.6e-4, betas=(0.9, 0.999))
    # During training, set delta_scaling per param group:
    optimizer.param_groups[0]["delta_scaling"] = delta_tensor  # (N,) or scalar
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class BOGausSAdam(Optimizer):
    """Modified Adam optimizer from BOGausS paper.

    Key differences from standard Adam:
    1. α_{s,t} = max(1/t, 1 - β_s*) instead of constant β
    2. Per-parameter Δ_θ scaling (precision-aware for positions/sizes)
    3. Per-element time counter for sparse updates (Gaussians only update
       when visible in current frame)
    4. No bias correction — the adaptive α handles the cold start

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (η_θ)
        betas: (β1*, β2*) exponential decay rates
        eps: numerical stability constant
        is_gaussian: if True, enables per-element time tracking for sparse updates
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, delta_scaling=None)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        For Gaussian parameters with sparse gradients, only elements
        that received gradients are updated. The time counter t is
        per-element and only increments for updated elements.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1_star, beta2_star = group["betas"]
            eps = group["eps"]
            delta_scaling = group.get("delta_scaling", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BOGausSAdam does not support truly sparse grads; use dense with masking")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Per-element time counter (crucial for sparse unbiased updates)
                    state["t"] = torch.zeros_like(p.data, dtype=torch.float32)
                    # First moment (filtered gradient)
                    state["g_hat"] = torch.zeros_like(p.data)
                    # Second moment (filtered squared gradient)
                    state["g2_hat"] = torch.zeros_like(p.data)

                t = state["t"]
                g_hat = state["g_hat"]
                g2_hat = state["g2_hat"]

                # Determine which elements have non-zero gradient
                # (for sparse Gaussian updates)
                grad_mask = grad.abs().sum(dim=-1, keepdim=True) > 0 if grad.dim() > 1 else (grad.abs() > 0)
                grad_mask = grad_mask.expand_as(grad).float()

                # Increment time only for active elements
                t.add_(grad_mask)

                # Compute α_{s,t} = max(1/t, 1 - β_s*)
                # Avoid division by zero for t=0 elements
                t_safe = t.clamp(min=1.0)
                alpha1 = torch.maximum(1.0 / t_safe, torch.tensor(1.0 - beta1_star, device=p.device))
                alpha2 = torch.maximum(1.0 / t_safe, torch.tensor(1.0 - beta2_star, device=p.device))

                # Update filtered gradient states (only where grad is active)
                # ĝ_t = ĝ_{t-1} + α_{1,t} * (∇L_t - ĝ_{t-1})
                g_hat.add_(alpha1 * grad_mask * (grad - g_hat))

                # ĝ²_t = ĝ²_{t-1} + α_{2,t} * ((∇L_t)² - ĝ²_{t-1})
                g2_hat.add_(alpha2 * grad_mask * (grad * grad - g2_hat))

                # Compute update: η_θ * Δ_θ * ĝ_t / (ĝ²_t + ε)
                # Note: paper divides by g2_hat directly, not sqrt(g2_hat)
                denom = g2_hat.abs() + eps  # abs for safety
                update = g_hat / denom

                # Apply precision-aware scaling Δ_θ
                if delta_scaling is not None:
                    if isinstance(delta_scaling, Tensor):
                        # delta_scaling is (N,) for per-Gaussian scaling
                        if delta_scaling.dim() == 1 and update.dim() == 2:
                            update = update * delta_scaling.unsqueeze(-1)
                        else:
                            update = update * delta_scaling
                    else:
                        update = update * delta_scaling

                # Apply learning rate and update (only active elements)
                p.data.add_(lr * grad_mask * update)

        return loss

    def inherit_state(
        self,
        param: torch.nn.Parameter,
        split_mask: Tensor,
        n_splits: int,
        alpha_t: float = 1.0,
        alpha_g: float = 0.2,
    ):
        """Propagate optimizer states from mother to child Gaussians after splitting.

        Paper equations:
            t_child = α_t × t_mother
            ĝ_child = α_g × ĝ_mother
            ĝ²_child = α_g² × ĝ²_mother

        Args:
            param: the parameter that was split
            split_mask: (N,) bool mask of which Gaussians were split
            n_splits: number of children per mother (typically 2)
            alpha_t: time inheritance factor (default 1.0)
            alpha_g: gradient state inheritance factor (default 0.2)
        """
        if param not in self.state or len(self.state[param]) == 0:
            return

        state = self.state[param]
        old_t = state["t"]
        old_g_hat = state["g_hat"]
        old_g2_hat = state["g2_hat"]

        # After splitting, the parameter tensor has been resized.
        # The new layout is: [kept_gaussians] + [children_of_split_gaussians]
        # where children repeat n_splits times.
        n_old = split_mask.shape[0]
        n_kept = (~split_mask).sum().item()
        n_new = param.shape[0]  # total after split

        # Build new state tensors matching new parameter size
        new_shape = param.shape
        new_t = torch.zeros(new_shape, dtype=torch.float32, device=param.device)
        new_g_hat = torch.zeros(new_shape, device=param.device)
        new_g2_hat = torch.zeros(new_shape, device=param.device)

        # Copy kept Gaussians (not split)
        keep_mask = ~split_mask
        new_t[:n_kept] = old_t[keep_mask]
        new_g_hat[:n_kept] = old_g_hat[keep_mask]
        new_g2_hat[:n_kept] = old_g2_hat[keep_mask]

        # Inherit states for children of split Gaussians
        mother_t = old_t[split_mask]
        mother_g_hat = old_g_hat[split_mask]
        mother_g2_hat = old_g2_hat[split_mask]

        for s in range(n_splits):
            start = n_kept + s * split_mask.sum().item()
            end = start + split_mask.sum().item()
            new_t[start:end] = alpha_t * mother_t
            new_g_hat[start:end] = alpha_g * mother_g_hat
            new_g2_hat[start:end] = (alpha_g ** 2) * mother_g2_hat

        state["t"] = new_t
        state["g_hat"] = new_g_hat
        state["g2_hat"] = new_g2_hat

    def reset_state_for_param(self, param: torch.nn.Parameter):
        """Reset optimizer state when a parameter is fully replaced (e.g., after pruning)."""
        if param in self.state:
            del self.state[param]
