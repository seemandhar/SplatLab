# SteepGS Reproduction Results

## Method
- **Paper**: Steepest Descent Density Control for Compact 3D Gaussian Splatting (CVPR 2025)
- **Repository**: https://github.com/facebookresearch/SteepGS
- **Scene**: mipnerf360 / garden
- **Data path**: data/mipnerf360/garden
- **Image resolution**: images_4 (4x downsampled, as per paper's outdoor mipnerf360 protocol)
- **Densification strategy**: steepest (the paper's proposed method)
- **S_estimator**: inv_cov

## Environment
- **GPU**: NVIDIA GPU (CUDA_VISIBLE_DEVICES=0)
- **CUDA**: 12.6
- **PyTorch**: 2.1.0+cu121
- **Python**: 3.10
- **Conda env**: steepgs

## Setup Issues Encountered
1. `std::uintptr_t` compilation error in `diff-gaussian-rasterization` -- fixed by adding `#include <cstdint>` to `rasterizer_impl.h`
2. `FLT_MAX` undefined in `simple-knn` -- fixed by adding `#include <cfloat>` to `simple_knn.cu`
3. Default `S_estimator` value in `arguments/__init__.py` is `inv_covar` but the renderer expects `inv_cov` -- must pass `--S_estimator inv_cov` explicitly
4. NumPy 2.x incompatibility with PyTorch 2.1.0 -- downgraded to numpy<2
5. Editable pip installs (`-e .`) did not work for submodules; non-editable installs required

## Results (7000 iterations -- quick test)

| Metric | Value |
|--------|-------|
| PSNR   | 26.011 |
| SSIM   | 0.8075 |
| LPIPS  | 0.2088 |
| Training Time | ~4 min 44 sec |
| Num Gaussians | 1,905,540 |

## Results (30000 iterations -- full training, paper default)

| Metric | Value |
|--------|-------|
| PSNR   | 27.471 |
| SSIM   | 0.8587 |
| LPIPS  | 0.1293 |
| Training Time | ~18.6 min |
| Num Gaussians | 3,327,728 |

## Paper-Reported Results (Aggregate mipnerf360, from Table 1)

The paper does not report per-scene metrics. It reports aggregate metrics across mipnerf360 scenes (3 outdoor + 4 indoor = 7 scenes total, though the note in the paper says "3 outdoor"):

| Method | # Points | PSNR | SSIM | LPIPS |
|--------|----------|------|------|-------|
| 3DGS (original) | 3.339M | 29.037 | 0.872 | 0.183 |
| SteepGS | 1.606M | 28.734 | 0.857 | 0.211 |

Note: The paper's aggregate includes all mipnerf360 scenes (both indoor and outdoor). The paper constrains total point count to ~1.6M via a point budget, whereas our reproduction uses default settings without a point budget (resulting in 3.3M points at 30k iterations). The paper's reported numbers use a budget-constrained setting for compact representation.

## Comparison with Paper

- **Garden PSNR achieved (30k, default settings)**: 27.47 dB
- **Garden PSNR reported in paper**: Not reported per-scene; aggregate mipnerf360 PSNR is 28.734 (budget-constrained at 1.6M points)
- **Comparison note**: Our reproduction uses default 3DGS-style training without the point budget constraint used in the paper's main table. The per-scene garden result is not available from the paper for direct comparison. The aggregate paper PSNR of 28.73 includes indoor scenes (bonsai, kitchen, etc.) which tend to have higher PSNR, so a single outdoor scene like garden having 27.47 is reasonable and consistent with expectations.

## Output Paths
- 7k model: sota/SteepGS/output/garden_steepgs
- 30k model: sota/SteepGS/output/garden_steepgs_30k

## Training Command Used (30k)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/mipnerf360/garden \
  -i images_4 \
  -m output/garden_steepgs_30k \
  --no_gui --eval \
  --densify_strategy steepest \
  --S_estimator inv_cov \
  --quiet
```
