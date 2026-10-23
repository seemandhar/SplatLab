"""
gs_template — Gaussian Splatting Method Template for Nerfstudio
================================================================

Minimal, extensible template for implementing 3DGS research papers
as nerfstudio plugins. Extends SplatfactoModel.

Installable methods:
    ns-train gs-template      --data <path>   # vanilla base
    ns-train absgs            --data <path>   # AbsGS (absgrad densification)
    ns-train depth-reg-gs     --data <path>   # Depth-regularized GS
    ns-train feature-gs       --data <path>   # Feature distillation GS
    ns-train micro-splatting  --data <path>   # Micro-Splatting (cov reg + prune/merge)
"""

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.datamanagers import DepthDataManager, DepthDataManagerConfig
from gs_template.losses import (
    cosine_loss,
    pearson_depth_loss,
    opacity_entropy_loss,
    covariance_trace_loss,
    covariance_isotropy_loss,
    l2_photometric_loss,
    natural_selection_reg_loss,
    adaptive_weighted_l1_loss,
    gradient_difference_loss,
    view_consistency_loss,
    depth_distortion_loss_2dgs,
    normal_consistency_loss,
    scale_flatten_loss_min,
    dwt_lf_loss, dwt_hf_sparsity_loss, haar_dwt_2d,

)
from gs_template.rendering import prepare_camera, rasterize_gaussians, assemble_outputs
from gs_template.optimizers import BOGausSAdam
from gs_template.losses import (
    cosine_loss,
    normal_consistency_loss,
    normal_residual_reg_loss,
    opacity_entropy_loss,
    pearson_depth_loss,
    sparse_opacity_loss,
    opacity_l1_loss
)
from gs_template.shading import depth_to_normal_map, sample_prefiltered_envmap, shortest_axis_directions
__all__ = [
    "GSTemplateModel",
    "GSTemplateModelConfig",
    "DepthDataManager",
    "DepthDataManagerConfig",
    "cosine_loss",
    "pearson_depth_loss",
    "opacity_entropy_loss",
    "covariance_trace_loss",
    "covariance_isotropy_loss",
    "l2_photometric_loss",
    "prepare_camera",
    "rasterize_gaussians",
    "natural_selection_reg_loss",
    "assemble_outputs",
    "adaptive_weighted_l1_loss",
    "gradient_difference_loss",
    "BOGausSAdam",
    "view_consistency_loss",
    "depth_distortion_loss_2dgs",
    "normal_consistency_loss",
    "scale_flatten_loss_min",
    "sparse_opacity_loss",
    "normal_consistency_loss",
    "normal_residual_reg_loss",
    "shortest_axis_directions",
    "sample_prefiltered_envmap",
    "depth_to_normal_map",
    "dwt_lf_loss",
    "dwt_hf_sparsity_loss",
    "haar_dwt_2d",
    "opacity_l1_loss"
]