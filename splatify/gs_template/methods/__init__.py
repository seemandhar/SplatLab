"""
gs_template.methods — Fully integrated example methods
=======================================================

Each method is a self-contained model + config demonstrating
the template at different complexity levels:

    absgs            ★☆☆☆☆  Config-only (absgrad=True)
    depth_reg        ★★☆☆☆  Custom loss function
    feature_gs       ★★★★☆  New params, rendering, loss, densification
    micro_splatting  ★★★☆☆  Custom losses + Stage II pruning/merging
"""

from gs_template.methods.absgs import AbsGSModel, AbsGSModelConfig
from gs_template.methods.depth_reg import DepthRegGSModel, DepthRegGSModelConfig
from gs_template.methods.feature_gs import FeatureGSModel, FeatureGSModelConfig
from gs_template.methods.micro_splatting import MicroSplattingModel, MicroSplattingModelConfig
from gs_template.methods.compact_gs import CompactGSModel, CompactGSModelConfig
from gs_template.methods.msaa_gs import MSAAGSModel, MSAAGSModelConfig
from gs_template.methods.opti3dgs import Opti3DGSModel, Opti3DGSModelConfig
from gs_template.methods.decomp_gs import DecompGSModel, DecompGSModelConfig
from gs_template.methods.bogauss import BOGausSModel, BOGausSModelConfig
from gs_template.methods.light_gs import LightGSModel, LightGSModelConfig
from gs_template.methods.vod3dgs import VoD3DGSModel, VoD3DGSModelConfig
from gs_template.methods.rain_gs import RAINGSModel, RAINGSModelConfig
from gs_template.methods.trick_gs import TrickGSModel, TrickGSModelConfig
from gs_template.methods.clod_gs import CLoDGSModel, CLoDGSModelConfig
from gs_template.methods.twodgs import TwoDGSModel, TwoDGSModelConfig
from gs_template.methods.pixel_gs import PixelGSModel, PixelGSModelConfig
from gs_template.methods.gaussian_shader import GaussianShaderModel, GaussianShaderModelConfig
from gs_template.methods.scaffold_gs import ScaffoldGSModel, ScaffoldGSModelConfig
from gs_template.methods.dwtgs import DWTGSModel, DWTGSModelConfig
from gs_template.methods.lp3dgs import LP3DGSModel, LP3DGSModelConfig
from gs_template.methods.compgs import CompGSModel, CompGSModelConfig
from gs_template.methods.cor_gs import CoRGSModel, CoRGSModelConfig
from gs_template.methods.fsgs import FSGSModel, FSGSModelConfig
from gs_template.methods.depth_anchor_gs import (
    DepthAnchorGSModel,
    DepthAnchorGSModelConfig,
    DepthAnchorGSDataManager,
    DepthAnchorGSDataManagerConfig,
)
from gs_template.methods.trifast import TriFastModel, TriFastModelConfig
from gs_template.methods.wavelet_loss_gs import WaveletLossGSModel, WaveletLossGSModelConfig
from gs_template.methods.c2f_gs import C2FGSModel, C2FGSModelConfig
from gs_template.methods.aa_wavelet_entropy_gs import AAWaveletEntropyGSModel, AAWaveletEntropyGSModelConfig
from gs_template.methods.aa_wavelet_entropy_plus_gs import AAWaveletEntropyPlusGSModel, AAWaveletEntropyPlusGSModelConfig


__all__ = [
    "AbsGSModel",
    "AbsGSModelConfig",
    "DepthRegGSModel",
    "DepthRegGSModelConfig",
    "FeatureGSModel",
    "FeatureGSModelConfig",
    "MicroSplattingModel",
    "MicroSplattingModelConfig",
    "CompactGSModel",
    "CompactGSModelConfig",
    "MSAAGSModel",
    "MSAAGSModelConfig",
    "Opti3DGSModel",
    "Opti3DGSModelConfig",
    "DecompGSModel",
    "DecompGSModelConfig",
    "BOGausSModel",
    "BOGausSModelConfig",
    "LightGSModel",
    "LightGSModelConfig",
    "VoD3DGSModel",
    "VoD3DGSModelConfig",
    "RAINGSModel",
    "RAINGSModelConfig",
    "TrickGSModel",
    "TrickGSModelConfig",
    "CLoDGSModel",
    "CLoDGSModelConfig",
    "TwoDGSModel",
    "TwoDGSModelConfig",
    "PixelGSModel",
    "PixelGSModelConfig",
    "GaussianShaderModel",
    "GaussianShaderModelConfig",
    "ScaffoldGSModel",
    "ScaffoldGSModelConfig",
    "DWTGSModel",
    "DWTGSModelConfig",
    "LP3DGSModel",
    "LP3DGSModelConfig",
    "CompGSModel",
    "CompGSModelConfig",
    "CoRGSModel",
    "CoRGSModelConfig",
    "FSGSModel",
    "FSGSModelConfig",
    "DepthAnchorGSModel",
    "DepthAnchorGSModelConfig",
    "DepthAnchorGSDataManager",
    "DepthAnchorGSDataManagerConfig",
    "TriFastModel",
    "TriFastModelConfig",
    "WaveletLossGSModel",
    "WaveletLossGSModelConfig",
    "C2FGSModel",
    "C2FGSModelConfig",
    "AAWaveletEntropyGSModel",
    "AAWaveletEntropyGSModelConfig",
    "AAWaveletEntropyPlusGSModel",
    "AAWaveletEntropyPlusGSModelConfig",
]
