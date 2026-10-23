# SPLATIFY CFG — Context-Free Grammar for GS Paper-to-Code

## Architecture Grammar

Every Gaussian Splatting nerfstudio plugin follows this grammar:

```
Plugin          → MethodSpec + Package
MethodSpec      → TrainerConfig(Pipeline, Optimizers, Viewer)
Pipeline        → VanillaPipelineConfig(DataManager, Model)
DataManager     → FullImageDatamanagerConfig(DataParser)
DataParser      → NerfstudioDataParserConfig(load_3D_points=True)
Model           → GSTemplateModelConfig → GSTemplateModel(SplatfactoModel)

Optimizers      → BaseOptimizers + CustomOptimizers
BaseOptimizers  → {means, features_dc, features_rest, opacities, scales, quats, camera_opt}
CustomOptimizers→ {param_name: AdamOptimizerConfig}*
```

## Extension Point Grammar

Given a paper P with modifications M = {m₁, m₂, ..., mₖ}, map each to:

```
Modification        → ExtensionPoint

New Gaussian param  → populate_modules() + split_gaussians()
                      + get_gaussian_param_groups() + optimizer in configs.py

New neural module   → populate_modules() + get_param_groups()
                      + optimizer in configs.py

New loss function   → compute_custom_losses() + (optionally losses.py)

Modified rendering  → get_outputs() using rendering.py helpers

Depth supervision   → DepthDataManager (datamanagers.py) + compute_custom_losses()

Feature distillation→ Custom DataManager (datamanagers.py) + populate_modules()
                      + get_outputs() + compute_custom_losses()

Viewer output       → get_outputs_for_camera()

Checkpoint compat   → load_state_dict()

Custom metric       → get_metrics_dict()
```

## Complexity Classification

```
★☆☆☆☆  Config-only       → Override config defaults (AbsGS)
★★☆☆☆  Loss + DataMgr    → DataManager + compute_custom_losses() (DepthRegGS)
★★★☆☆  New params + loss → populate_modules() + loss + split (OpacityGS)
★★★★☆  Full pipeline     → All above + get_outputs() (FeatureGS)
★★★★★  Custom data       → All above + custom DataManager (LangSplat)
```

## File Dependency DAG

```
configs.py ─────→ model.py ──→ SplatfactoModel (nerfstudio)
    │                 ↑
    ├──→ methods/ ────┘
    │    ├── absgs.py
    │    ├── depth_reg.py ──→ losses.py
    │    └── feature_gs.py ──→ losses.py + rendering.py
    │
    ├──→ datamanagers.py ──→ FullImageDatamanager (nerfstudio)
    │
    └──→ pyproject.toml (entry points)
```
