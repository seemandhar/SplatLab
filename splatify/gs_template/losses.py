"""
gs_template/losses.py — Reusable Loss Functions
=================================================

A library of common loss functions used across Gaussian Splatting papers.
Import and combine these in your compute_custom_losses() override.

Usage:
    from gs_template.losses import cosine_loss, pearson_depth_loss

    class MyModel(GSTemplateModel):
        def compute_custom_losses(self, outputs, batch, metrics_dict=None):
            return {"feat_loss": 0.001 * cosine_loss(pred, target)}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# Feature Distillation Losses
# ═══════════════════════════════════════════════════════════════════

def cosine_loss(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    """Cosine similarity loss: 1 - cos_sim, averaged over pixels.

    Used in: Feature Splatting, LangSplat, LEGaussians.

    Args:
        pred:   (C, H, W) or (H, W, C) predicted features
        target: same shape as pred, ground-truth features
        dim:    dimension along which to compute cosine similarity
    """
    return (1 - F.cosine_similarity(pred, target, dim=dim)).mean()


def l2_feature_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L2 feature reconstruction loss, per-pixel averaged.

    Used in: DINO-Splatting, generic feature distillation.
    """
    return F.mse_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════
# Depth Losses
# ═══════════════════════════════════════════════════════════════════

def pearson_depth_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Scale-invariant depth loss via Pearson correlation.

    Used in: DepthRegGS, MonoGS, DNGaussian.
    Ignores absolute scale — only penalizes structural disagreement.

    Args:
        pred:   (N,) or (H,W,1) predicted depth (valid pixels only)
        target: same shape, ground-truth or monocular depth
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_c = pred_flat - pred_flat.mean()
    target_c = target_flat - target_flat.mean()
    numer = (pred_c * target_c).sum()
    denom = pred_c.norm() * target_c.norm() + 1e-6
    return 1.0 - numer / denom


def l1_depth_loss(
    pred: Tensor, target: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    """L1 depth loss with optional validity mask.

    Args:
        pred:       (H, W, 1) predicted depth
        target:     (H, W, 1) ground-truth depth
        valid_mask: (H, W, 1) bool, True where depth is valid
    """
    if valid_mask is not None and valid_mask.sum() > 0:
        return F.l1_loss(pred[valid_mask], target[valid_mask])
    return F.l1_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════
# Regularization Losses
# ═══════════════════════════════════════════════════════════════════

def opacity_l1_loss(opacities_logit: Tensor, weight: float = 1e-7) -> Tensor:
    """L1 penalty on sigmoid opacities to encourage sparsity / pruning.

    Used in: CompGS.

    The loss is: weight * sum(sigmoid(opacities)).
    Uses SUM reduction (not mean) per the CompGS paper — the weight λ
    is calibrated for sum: with ~1M Gaussians and λ=1e-7, the loss
    contribution is ~0.05, comparable to the reconstruction loss.

    Args:
        opacities_logit: (N, 1) raw logit opacities (pre-sigmoid)
        weight: regularization strength (default: 1e-7 per paper)

    Returns:
        Scalar loss value.
    """
    o = torch.sigmoid(opacities_logit)
    return weight * o.sum()


def opacity_entropy_loss(opacities_logit: Tensor) -> Tensor:
    """Encourages opacities toward 0 or 1 (binary), reducing floaters.

    Used in: Many anti-artifact / compactness papers.

    Args:
        opacities_logit: (N, 1) raw logit opacities (pre-sigmoid)
    """
    o = torch.sigmoid(opacities_logit)
    return -(o * torch.log(o + 1e-6) + (1 - o) * torch.log(1 - o + 1e-6)).mean()


def scale_flatten_loss(scales_log: Tensor) -> Tensor:
    """Encourages Gaussians to be flat (one small axis), reducing spikes.

    Used in: 2DGS-inspired methods, SuGaR.

    Args:
        scales_log: (N, 3) log-space scales
    """
    scales = torch.exp(scales_log)
    # Penalize ratio between largest and smallest scale
    sorted_scales, _ = torch.sort(scales, dim=-1)
    ratio = sorted_scales[:, 2] / (sorted_scales[:, 0] + 1e-8)
    return ratio.mean()


def distortion_loss(weights: Tensor, t_mids: Tensor) -> Tensor:
    """Distortion loss from Mip-NeRF 360, adapted for Gaussians.

    Encourages weights to be concentrated (not spread along ray).

    Args:
        weights: (N_rays, N_samples) blending weights
        t_mids:  (N_rays, N_samples) midpoints of sample intervals
    """
    interval = t_mids[:, 1:] - t_mids[:, :-1]
    w_outer = weights[:, :-1] * weights[:, 1:]
    t_diff = (t_mids[:, :-1] - t_mids[:, 1:]).abs()
    return (w_outer * t_diff).sum(dim=-1).mean()


def total_variation_loss(features: Tensor) -> Tensor:
    """Total variation regularization on rendered feature maps.

    Encourages spatial smoothness of per-pixel features.

    Args:
        features: (H, W, C) rendered feature map
    """
    h_diff = (features[1:, :, :] - features[:-1, :, :]).abs().mean()
    w_diff = (features[:, 1:, :] - features[:, :-1, :]).abs().mean()
    return h_diff + w_diff


# ═══════════════════════════════════════════════════════════════════
# Micro-Splatting Losses
# ═══════════════════════════════════════════════════════════════════

def covariance_trace_loss(scales_log: Tensor, trace_cap: float) -> Tensor:
    """Trace-based covariance regularization (hinge penalty).

    Used in: Micro-Splatting.
    Penalizes Gaussians whose covariance trace exceeds a cap τ,
    encouraging compact, near-isotropic kernels.

    The covariance eigenvalues are λ_i = exp(s_i)^2 where s_i are
    log-space scale parameters. trace(Σ) = Σ λ_i.

    L_cov = Σ_k max(tr(Σ_k) - τ, 0)

    Args:
        scales_log: (N, 3) log-space scale parameters
        trace_cap:  τ, upper bound on acceptable trace (scene-dependent)

    Returns:
        Scalar hinge loss, sum of violations over all Gaussians.
    """
    # Eigenvalues of covariance = squared scales
    scales = torch.exp(scales_log)  # (N, 3)
    eigenvalues = scales ** 2       # (N, 3) — λ_1, λ_2, λ_3
    traces = eigenvalues.sum(dim=-1)  # (N,)

    # Hinge penalty: only penalize if trace > τ
    penalties = torch.clamp(traces - trace_cap, min=0.0)
    return penalties.mean()


def covariance_isotropy_loss(scales_log: Tensor) -> Tensor:
    """Isotropy regularization — penalizes anisotropy in Gaussians.

    Used in: Micro-Splatting.
    Encourages eigenvalue ratios close to 1 (spherical Gaussians).
    Computed as variance of log-scales across the 3 axes.

    Args:
        scales_log: (N, 3) log-space scale parameters

    Returns:
        Scalar isotropy loss (mean variance of log-scales).
    """
    # Variance of log-scales across axes measures anisotropy
    # If all 3 are equal → variance = 0 (perfect isotropy)
    return scales_log.var(dim=-1).mean()


def l2_photometric_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """L2 (MSE) photometric loss between rendered and target images.

    Used in: Micro-Splatting.
    Provides error-proportional gradients near convergence, complementing
    L1 which gives constant gradients.

    Args:
        rendered: (H, W, 3) rendered RGB image
        target:   (H, W, 3) ground-truth RGB image

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(rendered, target)



# ═══════════════════════════════════════════════════════════════════
# Natural Selection Losses (Compact 3DGS)
# ═══════════════════════════════════════════════════════════════════

def natural_selection_reg_loss(
    opacities_logit: Tensor, target_T: float = -20.0
) -> Tensor:
    """Regularization gradient field for natural selection pruning.

    Used in: "Gradient-Driven Natural Selection for Compact 3DGS"

    Implements L_reg = (E[v] - T)² where:
        v = pre-activation opacity (logit space)
        T = target value (default -20, providing strong downward pressure)

    The gradient ∇v = 2(E[v] - T) is globally constant for all Gaussians,
    creating uniform "environmental pressure". When |T| >> |E[v]|:
        ∇v ≈ -2T (stable, scene-agnostic)

    The finite prior arises from applying this to v (pre-sigmoid):
        Decay ratio R_o ≈ (1 - α)|∇v · lr|
        - High-opacity Gaussians (α→1): protected (R_o→0)
        - Low-opacity Gaussians (α→0): decay fastest

    Args:
        opacities_logit: (N, 1) raw logit opacities (v, pre-sigmoid)
        target_T:        target value T (default -20.0)
                         Increasing |T| linearly scales gradient strength.

    Returns:
        Scalar loss value.
    """
    mean_v = opacities_logit.mean()
    return (mean_v - target_T) ** 2


def natural_selection_reg_loss_no_prior(
    opacities_logit: Tensor, target_alpha: float = 0.0
) -> Tensor:
    """Prior-free variant: regularize α (post-sigmoid) directly.

    Used in: Early phase of natural selection (optional).
    All Gaussians experience identical opacity attenuation — no
    finite prior from sigmoid.

    Args:
        opacities_logit: (N, 1) raw logit opacities
        target_alpha:    target activated opacity (default 0.0)

    Returns:
        Scalar loss value.
    """
    alpha = torch.sigmoid(opacities_logit)
    mean_alpha = alpha.mean()
    return (mean_alpha - target_alpha) ** 2


def adaptive_weighted_l1_loss(
    pred: Tensor,
    target: Tensor,
    alpha: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    """Adaptive Weighting Strategy (AWS) — error-adaptive per-pixel L1.

    Used in: MSAA-GS.

    Per-pixel error e_{i,j} = |pred - target|_1, then weight:
        w_{i,j} = α + (1-α) * e_{i,j} / (max(e) + ε)

    Weighted loss:
        L_w = (1/HW) Σ w_{i,j} * |pred(i,j) - target(i,j)|_1

    This focuses optimization on high-error regions while maintaining
    a base gradient (α) everywhere.

    Args:
        pred:    (H, W, 3) predicted RGB image
        target:  (H, W, 3) ground-truth RGB image
        alpha:   base weight floor ∈ [0, 1] (α in paper Eq.8)
                 0 = fully error-adaptive, 1 = uniform (standard L1)
        epsilon: stability constant for normalization

    Returns:
        Scalar adaptive weighted L1 loss.
    """
    # Per-pixel L1 error across channels: (H, W)
    pixel_error = (pred - target).abs().mean(dim=-1)

    # Adaptive weight (Eq. 8)
    max_error = pixel_error.max().detach()
    weights = alpha + (1.0 - alpha) * pixel_error.detach() / (max_error + epsilon)

    # Weighted L1 (Eq. 9)
    weighted_error = weights * pixel_error
    return weighted_error.mean()


def gradient_difference_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Gradient Difference Constraint (GDC) — edge/gradient fidelity loss.

    Used in: MSAA-GS.

    Computes forward finite differences in x and y directions for both
    predicted and ground-truth images, then penalizes L1 difference
    between the gradient vectors.

    Paper Eq. 11:
        L_grad = 1/((H-1)(W-1)) Σ |G^pred_{i,j} - G^gt_{i,j}|_1

    where G_{i,j} = (∇_x I(i,j), ∇_y I(i,j)) are discrete gradient vectors.

    Args:
        pred:   (H, W, 3) predicted RGB image
        target: (H, W, 3) ground-truth RGB image

    Returns:
        Scalar gradient difference loss.
    """
    # Forward differences in x-direction (horizontal): ∇_x I(i,j) = I(i,j+1) - I(i,j)
    pred_grad_x = pred[:, 1:, :] - pred[:, :-1, :]      # (H, W-1, 3)
    target_grad_x = target[:, 1:, :] - target[:, :-1, :]

    # Forward differences in y-direction (vertical): ∇_y I(i,j) = I(i+1,j) - I(i,j)
    pred_grad_y = pred[1:, :, :] - pred[:-1, :, :]       # (H-1, W, 3)
    target_grad_y = target[1:, :, :] - target[:-1, :, :]

    # L1 on gradient differences (Eq. 11)
    # Crop to common region (H-1, W-1) to combine x and y gradients
    loss_x = (pred_grad_x[:-1, :, :] - target_grad_x[:-1, :, :]).abs().mean()
    loss_y = (pred_grad_y[:, :-1, :] - target_grad_y[:, :-1, :]).abs().mean()

    return loss_x + loss_y


def weighted_depth_prior_loss(
    pred: Tensor,
    prior: Tensor,
    confidence: Tensor | None = None,
) -> Tensor:
    """Confidence-weighted depth prior loss for DUSt3R-style priors.

    Used in: StableGS (D(i,j) = 1/|d|_0 * sum(C * |D^s - d|))

    Computes confidence-weighted L1 loss between rendered depth and
    the scaled DUSt3R depth prior. Only valid (non-zero) pixels
    contribute.

    Args:
        pred:       (H, W, 1) rendered geometric depth D^s
        prior:      (H, W, 1) scaled DUSt3R depth prior d_{i,j}
        confidence: (H, W, 1) DUSt3R confidence map (optional)
    """
    # Valid mask: non-zero prior values
    valid = (prior > 0) & (pred > 0)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=pred.device)

    diff = (pred - prior).abs()

    if confidence is not None:
        # Normalize confidence to sum to 1 over valid pixels
        conf_valid = confidence[valid]
        conf_norm = conf_valid / (conf_valid.sum() + 1e-8)
        loss = (conf_norm * diff[valid]).sum()
    else:
        loss = diff[valid].mean()

    return loss


def depth_consistency_loss(
    depth_src: Tensor,
    depth_tgt_warped: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Self-supervised multi-view depth consistency loss.

    Used in: StableGS (L_consis = |W_{i→j}(D^s_i) - D^s_j|_1)

    Computes L1 loss between warped source depth and target depth
    over validly projected pixels.

    Args:
        depth_src:        (H, W) or (H, W, 1) warped source depth
        depth_tgt_warped: (H, W) or (H, W, 1) target depth at warped coords
        valid_mask:       (H, W) bool mask of valid projections
    """
    src_flat = depth_src.reshape(-1)
    tgt_flat = depth_tgt_warped.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    if valid_flat.sum() < 10:
        return torch.tensor(0.0, device=depth_src.device)

    return F.l1_loss(src_flat[valid_flat], tgt_flat[valid_flat])


def view_consistency_loss(
    means: Tensor,
    opacity_logit: Tensor,
    S_params: Tensor,
    cam_center_i: Tensor,
    cam_center_j: Tensor,
    quad_clamp: float = 10.0,
) -> Tensor:
    """View-consistency regularization loss from VoD-3DGS.

    Penalizes opacity differences between nearby views, weighted by
    cosine of the angle between view vectors.

        L_vc = (1/|G|) Σ_g [ max(cos(θ), 0) · (α̂(ω_i) - α̂(ω_j))² ]

    Args:
        means:          (N, 3) Gaussian centers (should be detached)
        opacity_logit:  (N, 1) base opacity logits γ (should be detached)
        S_params:       (N, 6) symmetric matrix parameters (gradient flows here)
        cam_center_i:   (3,) camera center of view i
        cam_center_j:   (3,) camera center of view j
        quad_clamp:     clamp range for quadratic form (match forward pass)

    Returns:
        Scalar loss value.
    """
    # View directions from each Gaussian to each camera
    omega_i = F.normalize(cam_center_i.unsqueeze(0) - means, dim=-1, eps=1e-8)
    omega_j = F.normalize(cam_center_j.unsqueeze(0) - means, dim=-1, eps=1e-8)

    # Cosine of angle between view vectors
    cos_theta = (omega_i * omega_j).sum(dim=-1)
    cos_weight = torch.clamp(cos_theta, min=0.0)

    # Compute quadratic forms for both views
    ox_i, oy_i, oz_i = omega_i[:, 0], omega_i[:, 1], omega_i[:, 2]
    ox_j, oy_j, oz_j = omega_j[:, 0], omega_j[:, 1], omega_j[:, 2]
    Sxx, Sxy, Sxz, Syy, Syz, Szz = (
        S_params[:, 0], S_params[:, 1], S_params[:, 2],
        S_params[:, 3], S_params[:, 4], S_params[:, 5],
    )

    quad_i = (
        Sxx * ox_i * ox_i + 2 * Sxy * ox_i * oy_i + 2 * Sxz * ox_i * oz_i
        + Syy * oy_i * oy_i + 2 * Syz * oy_i * oz_i + Szz * oz_i * oz_i
    )
    quad_j = (
        Sxx * ox_j * ox_j + 2 * Sxy * ox_j * oy_j + 2 * Sxz * ox_j * oz_j
        + Syy * oy_j * oy_j + 2 * Syz * oy_j * oz_j + Szz * oz_j * oz_j
    )

    # Clamp to match forward pass behavior
    quad_i = torch.clamp(quad_i, -quad_clamp, quad_clamp)
    quad_j = torch.clamp(quad_j, -quad_clamp, quad_clamp)

    gamma = opacity_logit.squeeze(-1)
    alpha_i = torch.sigmoid(gamma + quad_i)
    alpha_j = torch.sigmoid(gamma + quad_j)

    per_gaussian_loss = cos_weight * (alpha_i - alpha_j).pow(2)

    return per_gaussian_loss.mean()

def scale_flatten_loss_min(scales_log: Tensor) -> Tensor:
    """Drive the smallest scale axis toward zero (planar Gaussians).

    2DGS represents each Gaussian as a 2D disk embedded in 3D.
    In the original formulation the third scaling factor is exactly
    zero; here we softly penalize the minimum scale so that the
    optimizer gradually collapses one axis.

    Used in: 2DGS (Huang et al., SIGGRAPH 2024).

    Args:
        scales_log: (N, 3) log-space scales
    """
    scales = torch.exp(scales_log)
    min_scale = scales.min(dim=-1).values       # (N,)
    return min_scale.mean()


def depth_distortion_loss_2dgs(
    depth: Tensor,
    accumulation: Tensor,
) -> Tensor:
    """Approximate depth distortion loss for 2DGS.

    The original formulation (Eq. 15 in the paper) sums over all
    pairs of intersections along each ray:

        L_d = sum_{i,j} w_i w_j |z_i - z_j|

    Computing this exactly requires per-ray per-intersection access
    inside the CUDA rasterizer.  We approximate it from the rendered
    outputs by penalizing depth variance: pixels whose blending
    weights are spread across multiple depths have high variance,
    which this loss drives toward zero.

    Specifically we encourage each pixel's depth to be smooth and
    concentrated by penalizing the spatial Laplacian of the depth
    map weighted by accumulated opacity.  Regions with high alpha
    but inconsistent depths incur the largest penalty.

    Used in: 2DGS (Huang et al., SIGGRAPH 2024).

    Args:
        depth:        (H, W, 1) rendered expected depth
        accumulation: (H, W, 1) accumulated alpha
    """
    d = depth[..., 0]                                   # (H, W)
    a = accumulation[..., 0].detach()                   # (H, W)

    # NDC-like normalization to down-weight distant Gaussians
    near, far = 0.2, 1000.0
    d_ndc = (1.0 / (d + 1e-6) - 1.0 / far) / (1.0 / near - 1.0 / far + 1e-8)
    d_ndc = d_ndc.clamp(0, 1)

    # Spatial Laplacian of the depth map (finite-difference approx)
    lap_h = d_ndc[2:, 1:-1] + d_ndc[:-2, 1:-1] - 2 * d_ndc[1:-1, 1:-1]
    lap_w = d_ndc[1:-1, 2:] + d_ndc[1:-1, :-2] - 2 * d_ndc[1:-1, 1:-1]
    laplacian = (lap_h.abs() + lap_w.abs())             # (H-2, W-2)

    # Weight by accumulated opacity so the loss focuses on occupied
    # regions and ignores the background
    weight = a[1:-1, 1:-1]
    loss = (weight * laplacian).mean()
    return loss


def normal_consistency_loss(
    rendered_normal: Tensor,
    depth_normal: Tensor,
    accumulation: Tensor | None = None,
) -> Tensor:
    """Normal consistency loss from the 2DGS paper (Eq. 16).

    Aligns per-Gaussian surface normals (rendered via alpha blending)
    with normals derived from the depth-map gradient.  The original
    per-intersection formulation is:

        L_n = sum_i w_i (1 - n_i^T N)

    We compute its pixel-level proxy: 1 - dot(rendered_normal, depth_normal),
    weighted by accumulated opacity so that background pixels are ignored.

    Used in: 2DGS (Huang et al., SIGGRAPH 2024).

    Args:
        rendered_normal: (H, W, 3) alpha-blended Gaussian normals
        depth_normal:    (H, W, 3) normals from depth gradient
        accumulation:    (H, W, 1) optional opacity weight
    """
    dot = (rendered_normal * depth_normal).sum(dim=-1)  # (H, W)
    loss_map = 1.0 - dot                                # (H, W)

    if accumulation is not None:
        w = accumulation[..., 0].detach()
        loss = (w * loss_map).sum() / (w.sum() + 1e-6)
    else:
        loss = loss_map.mean()

    return loss


def sparse_opacity_loss(opacities: Tensor) -> Tensor:
    """GaussianShader sparse opacity loss."""
    o = opacities.clamp(1e-6, 1 - 1e-6)
    return (torch.log(o) + torch.log(1 - o)).mean()


def normal_consistency_loss(
    pred_normal: Tensor,
    depth_normal: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Squared L2 normal consistency loss used by GaussianShader."""
    if valid_mask is not None and valid_mask.any():
        return ((pred_normal[valid_mask] - depth_normal[valid_mask]) ** 2).mean()
    return ((pred_normal - depth_normal) ** 2).mean()


def normal_residual_reg_loss(delta_1: Tensor, delta_2: Tensor) -> Tensor:
    """Quadratic penalty on the two learned normal residuals."""
    return 0.5 * (delta_1.pow(2).sum(dim=-1).mean() + delta_2.pow(2).sum(dim=-1).mean())




def _build_haar_filters(device: torch.device) -> tuple:
    """Build 2D Haar wavelet filter kernels for conv2d.

    Returns low-pass and high-pass 1D filter vectors (for separable application).
    Haar: L = [1/√2, 1/√2], H = [1/√2, -1/√2]
    """
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    lo = torch.tensor([inv_sqrt2, inv_sqrt2], device=device)
    hi = torch.tensor([inv_sqrt2, -inv_sqrt2], device=device)
    return lo, hi


def haar_dwt_2d(x: Tensor) -> tuple:
    """Single-level 2D Discrete Wavelet Transform using Haar wavelets.

    Differentiable implementation using strided convolutions.

    Args:
        x: (B, C, H, W) or (H, W, C) image tensor.
           If (H,W,C), will be reshaped internally.

    Returns:
        (LL, LH, HL, HH) — each is (B, C, H//2, W//2)
        If input was (H,W,C), outputs are (H//2, W//2, C).
    """
    squeeze_back = False
    if x.dim() == 3:
        # (H, W, C) → (1, C, H, W)
        squeeze_back = True
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = x.shape
    # Ensure even dimensions (pad if odd)
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    lo, hi = _build_haar_filters(x.device)

    # 2D convolution with separable Haar kernels
    # Build 2D kernels: LL, LH, HL, HH
    # LL = lo_col ⊗ lo_row, LH = hi_col ⊗ lo_row, HL = lo_col ⊗ hi_row, HH = hi_col ⊗ hi_row
    ll_k = (lo.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
    lh_k = (hi.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    hl_k = (lo.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    hh_k = (hi.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    # Expand kernels for grouped convolution (each channel independently)
    ll_k = ll_k.expand(C, 1, 2, 2)
    lh_k = lh_k.expand(C, 1, 2, 2)
    hl_k = hl_k.expand(C, 1, 2, 2)
    hh_k = hh_k.expand(C, 1, 2, 2)

    # Apply with stride=2 (downsampling) and groups=C (per-channel)
    LL = F.conv2d(x, ll_k, stride=2, groups=C)
    LH = F.conv2d(x, lh_k, stride=2, groups=C)
    HL = F.conv2d(x, hl_k, stride=2, groups=C)
    HH = F.conv2d(x, hh_k, stride=2, groups=C)

    if squeeze_back:
        # (1, C, H/2, W/2) → (H/2, W/2, C)
        LL = LL.squeeze(0).permute(1, 2, 0)
        LH = LH.squeeze(0).permute(1, 2, 0)
        HL = HL.squeeze(0).permute(1, 2, 0)
        HH = HH.squeeze(0).permute(1, 2, 0)

    return LL, LH, HL, HH


def multi_level_haar_dwt_ll(x: Tensor, levels: int = 2) -> list:
    """Multi-level Haar DWT returning LL subbands at each level.

    Args:
        x: (H, W, C) image tensor
        levels: number of DWT decomposition levels

    Returns:
        List of LL subbands: [LL^1, LL^2, ..., LL^levels]
        Each LL^n has shape (H/2^n, W/2^n, C)
    """
    ll_subbands = []
    current = x
    for _ in range(levels):
        LL, _, _, _ = haar_dwt_2d(current)
        ll_subbands.append(LL)
        current = LL
    return ll_subbands


def dwt_lf_loss(pred: Tensor, target: Tensor, levels: int = 2) -> Tensor:
    """DWTGS low-frequency loss: L1 on LL subbands at multiple DWT levels.

    L_DWTGS_LF = Σ_{n=1}^{N} L1(X_LL^n, X_LL^n_gt)

    Args:
        pred:   (H, W, C) rendered image
        target: (H, W, C) ground-truth image
        levels: number of DWT decomposition levels (paper uses N=2)

    Returns:
        Scalar loss
    """
    pred_lls = multi_level_haar_dwt_ll(pred, levels)
    target_lls = multi_level_haar_dwt_ll(target, levels)

    loss = torch.tensor(0.0, device=pred.device)
    for pred_ll, target_ll in zip(pred_lls, target_lls):
        loss = loss + F.l1_loss(pred_ll, target_ll)
    return loss


def dwt_hf_sparsity_loss(rendered: Tensor) -> Tensor:
    """DWTGS high-frequency sparsity loss: L1(HH^1, 0).

    Self-supervised: encourages the 1-level HH subband of a rendered
    image to be sparse (close to zero), reducing HF hallucinations.

    Args:
        rendered: (H, W, C) rendered image (typically from a novel view)

    Returns:
        Scalar loss
    """
    _, _, _, HH = haar_dwt_2d(rendered)
    return F.l1_loss(HH, torch.zeros_like(HH))




def dssim_loss(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    """D-SSIM (structural dissimilarity) loss between two images.

    Computes D-SSIM = (1 - SSIM) / 2 using a Gaussian-weighted window.

    Used in: CoR-GS, standard 3DGS color loss.

    Args:
        img1: (H, W, C) or (B, C, H, W) first image
        img2: same shape as img1, second image
        window_size: size of the Gaussian kernel (default 11)

    Returns:
        Scalar D-SSIM loss.
    """
    # Ensure BCHW format
    if img1.dim() == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    elif img1.dim() == 4 and img1.shape[-1] <= 4:
        # Likely (B, H, W, C), convert to (B, C, H, W)
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)

    C = img1.shape[1]

    # Create Gaussian window
    gauss = torch.exp(
        -0.5
        * (torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2) ** 2
        / (1.5 ** 2)
    )
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return (1.0 - ssim_map.mean()) / 2.0


def color_co_regularization_loss(
    img1: Tensor,
    img2: Tensor,
    dssim_weight: float = 0.2,
) -> Tensor:
    """Color co-regularization loss from CoR-GS.

    Computes: R_pcolor = (1 - λ) * L1(I'1, I'2) + λ * D-SSIM(I'1, I'2)

    Used in: CoR-GS pseudo-view co-regularization.

    Args:
        img1: (H, W, 3) rendered image from field 1
        img2: (H, W, 3) rendered image from field 2
        dssim_weight: λ weight for D-SSIM component (default 0.2)

    Returns:
        Scalar co-regularization loss.
    """
    l1 = F.l1_loss(img1, img2)
    dssim = dssim_loss(img1, img2)
    return (1.0 - dssim_weight) * l1 + dssim_weight * dssim
