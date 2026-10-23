# Literature Review: Recent 3D Gaussian Splatting Improvements (2024-2025)

## Focus: State-of-the-Art PSNR/SSIM on Mip-NeRF 360, Training Speed, and Techniques Compatible with SteepGS

---

## 1. Baseline Methods and Foundational Numbers

### 1.1 Original 3DGS (Kerbl et al., SIGGRAPH 2023)
- **Key technique**: Differentiable splatting of 3D Gaussians with adaptive density control (clone/split based on view-space positional gradient)
- **Mip-NeRF 360 avg PSNR**: ~27.21 (uncompressed, ~734 MB model)
- **Garden PSNR**: ~27.41
- **Training time**: ~25-27 min on single GPU
- **Loss**: 0.8 * L1 + 0.2 * D-SSIM
- **LR schedule**: Position LR exponentially decayed from 1.6e-4 to 1.6e-6; 500 warmup iterations before densification; densification every 100 iterations up to 15K

### 1.2 Mip-NeRF 360 (Barron et al., CVPR 2022)
- **Mip-NeRF 360 avg PSNR**: ~27.69 (NeRF-based, hours of training)
- **Key technique**: Unbounded scene contraction, anti-aliased integrated positional encoding, distortion loss

### 1.3 Zip-NeRF (Barron et al., ICCV 2023)
- **Mip-NeRF 360 avg PSNR**: ~28.54
- **Key technique**: Anti-aliased grid-based NeRF combining iNGP hash grids with mip-NeRF 360 anti-aliasing
- **Training time**: ~5.5 hours (22x faster than Mip-NeRF 360)

---

## 2. Densification Strategy Improvements

### 2.1 AbsGS (Ye et al., CVPR 2024)
- **Key technique**: Homodirectional view-space positional gradient -- sums absolute values of pixel-wise sub-gradients to avoid gradient collision/cancellation in the original average gradient criterion
- **Mip-NeRF 360 avg PSNR**: ~27.44 (with 0.0004 threshold), ~27.56 (with 0.0008 threshold)
- **Training time**: Similar to 3DGS baseline
- **Compatibility with SteepGS**: Highly compatible -- AbsGS modifies the gradient accumulation for densification, orthogonal to SteepGS's steepest-descent splitting direction. Can be combined.

### 2.2 Pixel-GS (Zhang et al., ECCV 2024)
- **Key technique**: Pixel-aware gradient weighting for density control -- weights gradients by the number of pixels each Gaussian covers per view, preventing large Gaussians from being under-split
- **Mip-NeRF 360 avg PSNR**: ~27.49
- **Training time**: Similar to 3DGS baseline
- **Compatibility with SteepGS**: Compatible -- modifies gradient weighting, independent of splitting displacement direction

### 2.3 Mini-Splatting (Fang & Wang, ECCV 2024)
- **Key technique**: Importance-based Gaussian pruning with constrained Gaussian budget; blur-split densification
- **Mip-NeRF 360 avg PSNR**: ~27.29 (with fewer Gaussians)
- **Training time**: ~20 min

### 2.4 Mini-Splatting2 (Fang & Wang, 2024)
- **Key technique**: Aggressive early densification -- extracts dense point clouds from Gaussian representations early, then uses visibility culling to reduce computation
- **Mip-NeRF 360 avg PSNR**: 27.33 (MSv2), 27.54 (MSv2-D dense variant)
- **Garden PSNR**: 26.80 (MSv2), with 0.73M Gaussians
- **Training time**: 3 min 34s (MSv2), 8 min 52s (MSv2-D) -- 7.6x faster than baseline
- **Compatibility with SteepGS**: Partially compatible -- aggressive densification is a different paradigm from SteepGS's saddle-point-aware splitting, but the visibility culling idea could complement SteepGS

### 2.5 GaussianPro (Cheng et al., ICML 2024)
- **Key technique**: Progressive propagation using patch matching to guide densification; leverages existing reconstructed geometry to produce new Gaussians with accurate positions/orientations
- **Mip-NeRF 360 avg PSNR**: Comparable to 3DGS (~27.3); more impactful on Waymo (+1.15 dB)
- **Training time**: Longer than baseline due to patch matching
- **Compatibility with SteepGS**: Could complement SteepGS for textureless regions where SteepGS's gradient-based approach struggles

---

## 3. MCMC and Optimization-Theoretic Approaches

### 3.1 3DGS-MCMC (Kheradmand et al., NeurIPS 2024 Spotlight)
- **Key technique**: Reformulates densification as MCMC sampling via Stochastic Gradient Langevin Dynamics (SGLD); replaces heuristic clone/split with stochastic relocation
- **Mip-NeRF 360 avg PSNR**: 29.72 (with SfM init), beating NeRF backbones
- **Garden PSNR**: 28.44 (random init)
- **Training time**: ~30 min (comparable), 20% reduction with optimized settings
- **Compatibility with SteepGS**: The relocation mechanism is a fundamentally different paradigm. SteepGS's steepest-descent directions could potentially be used to guide MCMC birth positions.

### 3.2 SteepGS (Wang et al., CVPR 2025)
- **Key technique**: Identifies Gaussians in saddle points of loss landscape; splits them and displaces offspring along steepest descent directions
- **Mip-NeRF 360 avg PSNR**: 28.734 (with 1.606M points vs 3DGS's 3.339M at 29.037)
- **Training time**: Not explicitly reported; similar to baseline
- **Key result**: ~50% Gaussian reduction with competitive quality
- **Note**: At matched Gaussian budgets, SteepGS significantly outperforms 3DGS-MCMC (28.734 vs 28.149 at ~1.6M points)

### 3.3 Opt3DGS (2025)
- **Key technique**: Adaptive exploration and curvature-aware exploitation for optimization
- **Mip-NeRF 360 avg PSNR**: 29.96
- **Training time**: ~11-12 min per scene
- **Compatibility with SteepGS**: Curvature-aware exploitation is related to SteepGS's saddle-point detection; potentially complementary

---

## 4. Anti-Aliasing and Rendering Improvements

### 4.1 Mip-Splatting (Yu et al., CVPR 2024 Best Student Paper)
- **Key technique**: 3D smoothing filter + 2D Mip filter for alias-free Gaussian rendering; prevents high-frequency artifacts at different scales
- **Mip-NeRF 360 avg PSNR**: ~27.39 (single scale), better multi-scale consistency
- **Garden PSNR**: ~27.60
- **Training time**: ~46 min (longer due to filtering)
- **Compatibility with SteepGS**: Fully compatible -- Mip-Splatting modifies rendering, SteepGS modifies densification

### 4.2 Analytic-Splatting (Liang et al., ECCV 2024 Oral)
- **Key technique**: Analytically approximates pixel-area integral of Gaussian signals using conditioned logistic CDF, avoiding the 2D Gaussian low-pass approximation of Mip-Splatting
- **Mip-NeRF 360 avg PSNR**: Comparable to Mip-Splatting at single scale, better at multi-scale
- **Compatibility with SteepGS**: Fully compatible -- rendering-side improvement

### 4.3 Multi-Scale 3D Gaussian Splatting (Yan et al., CVPR 2024)
- **Key technique**: LOD-based Gaussians at multiple scales; selects Gaussians based on screen-space pixel coverage
- **Mip-NeRF 360 improvement**: 13-66% PSNR improvement at reduced resolutions
- **Compatibility with SteepGS**: Compatible for multi-resolution scenarios

### 4.4 StopThePop (Radl et al., SIGGRAPH 2024)
- **Key technique**: Hierarchical sorting for view-consistent Gaussian rendering; drastically reduces sort errors in depth ordering
- **Mip-NeRF 360 avg PSNR**: Slight improvement over 3DGS baseline
- **Compatibility with SteepGS**: Fully compatible -- rendering pipeline improvement

### 4.5 3D-HGS: 3D Half-Gaussian Splatting (Li et al., CVPR 2025)
- **Key technique**: Replaces symmetric Gaussian kernel with half-Gaussian kernel that better models sharp boundaries/edges
- **Mip-NeRF 360 avg PSNR**: 29.66 (3D-HGS), 30.13 (MCMC-HGS)
- **Garden PSNR**: 27.50 (3D-HGS), 27.67 (MCMC-HGS) vs 27.30 (3DGS baseline)
- **Compatibility with SteepGS**: Highly compatible -- kernel modification is orthogonal to densification strategy. A "SteepGS-HGS" variant could yield significant gains.

---

## 5. Structured Representations

### 5.1 Scaffold-GS (Lu et al., CVPR 2024 Highlight)
- **Key technique**: Anchor-based structured Gaussians; uses anchor points to distribute local Gaussians with view-adaptive attributes predicted via MLPs
- **Mip-NeRF 360 avg PSNR**: ~27.50
- **Training time**: ~29 min (comparable)
- **Compatibility with SteepGS**: Different paradigm (anchor-based vs free Gaussians), but SteepGS's saddle-point detection could inform anchor placement

### 5.2 Octree-GS (Ren et al., TPAMI 2025)
- **Key technique**: LOD-structured Gaussians organized in octree; dynamic level selection for consistent rendering
- **Mip-NeRF 360 avg PSNR**: 28.05
- **Training time**: ~20-34 min depending on variant
- **Compatibility with SteepGS**: Partially compatible -- octree structure adds overhead but LOD concept is independent

---

## 6. Training Speed Improvements

### 6.1 DashGaussian (Chen et al., CVPR 2025 Highlight)
- **Key technique**: Resolution scheduling (progressive low-to-high resolution training) + primitive number scheduling coupled with resolution
- **Mip-NeRF 360 avg PSNR**: Competitive with baseline
- **Training time**: ~3.23 min (with Taming-3DGS backbone), 45.7% average acceleration
- **Compatibility with SteepGS**: Highly compatible -- resolution scheduling is orthogonal to densification strategy

### 6.2 FastGS (CVPR 2026 Highlight)
- **Key technique**: Multi-view consistency-based importance for densification/pruning; no budgeting mechanism needed
- **Mip-NeRF 360 avg PSNR**: 27.54
- **Training time**: 1.92 min (FastGS), 3.59 min (FastGS-Big) -- 15x faster than baseline
- **Compatibility with SteepGS**: Compatible -- importance-based densification could incorporate SteepGS's saddle-point criterion

### 6.3 3DGS-LM (Hoel et al., ICCV 2025)
- **Key technique**: Replaces Adam optimizer with Levenberg-Marquardt for second phase of optimization; custom CUDA kernels for efficient Jacobian-vector products
- **Mip-NeRF 360 avg PSNR**: Same as baseline 3DGS
- **Training time**: 20% faster on average
- **Compatibility with SteepGS**: Highly compatible -- optimizer change is independent of densification

### 6.4 Group Training (Wang et al., ICCV 2025)
- **Key technique**: Organizes Gaussians into groups; Opacity-based Prioritized Sampling (OPS) for selective training
- **Mip-NeRF 360 avg PSNR**: 25.22 (3DGS+GT), up to 27.86 (Mip-Splatting+GT on garden)
- **Garden PSNR (with Mip-Splatting)**: 27.86 (vs 27.60 baseline), time reduced from 49.5 to 38.8 min
- **Training time**: ~26% faster than baseline
- **Compatibility with SteepGS**: Highly compatible -- group training is purely a training efficiency strategy

### 6.5 Taming 3DGS (Mallick et al., SIGGRAPH Asia 2024)
- **Key technique**: Score-based constructive densification with explicit budget control; training-time priors measuring Gaussian contribution
- **Mip-NeRF 360 avg PSNR**: Competitive with 3DGS at 4-5x smaller model
- **Training time**: 4-5x reduction
- **Compatibility with SteepGS**: Compatible -- budget control could be combined with steepest-descent splitting

---

## 7. Compression Methods (for reference)

### 7.1 HAC (Chen et al., ECCV 2024)
- **Key technique**: Hash-grid assisted context for entropy coding of Gaussian attributes
- **Mip-NeRF 360 avg PSNR**: 27.77 (high-rate), model size ~22.9 MB
- **HAC++ variant**: 27.82 PSNR at 19.4 MB

### 7.2 CompGS (2024)
- **Mip-NeRF 360 avg PSNR**: 27.26 at 17.3 MB

### 7.3 Spec-Gaussian (Yang et al., NeurIPS 2024)
- **Key technique**: Anisotropic view-dependent appearance modeling for specular highlights without increasing Gaussian count
- **Mip-NeRF 360 avg PSNR**: Improved over 3DGS on scenes with specular components
- **Compatibility with SteepGS**: Compatible -- appearance model is independent of densification

### 7.4 GOF: Gaussian Opacity Fields (Yu et al., SIGGRAPH Asia 2024)
- **Key technique**: Opacity field regularization for improved surface reconstruction alongside novel view synthesis
- **Mip-NeRF 360 avg PSNR**: Slightly better than 3DGS on PSNR, significantly better LPIPS on outdoor scenes
- **Compatibility with SteepGS**: Compatible -- regularization is orthogonal to densification

---

## 8. Emerging / Frontier Methods

### 8.1 BOGausS Light (2025)
- **Mip-NeRF 360 avg PSNR**: 28.26 (highest non-MCMC result in compression survey)
- **Model size**: 641.3 MB

### 8.2 GaussianSpa (2024)
- **Mip-NeRF 360 avg PSNR**: 27.83
- **Model size**: 574.2 MB

### 8.3 PUP 3D-GS (Hanson et al., CVPR 2025)
- **Key technique**: Principled uncertainty-based pruning; prunes 80% of Gaussians, fine-tunes 5K iterations to recover quality
- **Compatibility with SteepGS**: Highly compatible as post-training pruning

### 8.4 LP-3DGS (Zhang et al., NeurIPS 2024)
- **Key technique**: Learning-based pruning with importance scoring
- **Mip-NeRF 360 avg PSNR**: 27.47 (matches baseline)
- **Pruning ratio**: 0.63 average

---

## 9. Summary: Best Reported Numbers on Mip-NeRF 360

| Method | Year | Avg PSNR | Garden PSNR | Training Time | #Gaussians |
|--------|------|----------|-------------|---------------|------------|
| 3DGS (baseline) | 2023 | 27.21 | 27.41 | ~27 min | ~3.3M |
| Mip-Splatting | 2024 | 27.39 | 27.60 | ~46 min | ~3.3M |
| AbsGS | 2024 | 27.56 | -- | ~27 min | ~3.5M |
| Pixel-GS | 2024 | 27.49 | -- | ~27 min | ~3.5M |
| Scaffold-GS | 2024 | 27.50 | -- | ~29 min | -- |
| Octree-GS | 2025 | 28.05 | -- | ~20-34 min | -- |
| 3DGS-MCMC | 2024 | 29.72 | 28.44 | ~30 min | ~3M |
| 3D-HGS | 2025 | 29.66 | 27.50 | -- | -- |
| MCMC-HGS | 2025 | 30.13 | 27.67 | -- | -- |
| Opt3DGS | 2025 | 29.96 | -- | ~11-12 min | -- |
| SteepGS | 2025 | 28.73 | -- | ~27 min | ~1.6M |
| Mini-Splatting2 | 2024 | 27.33 | 26.80 | 3.5 min | ~0.73M |
| FastGS | 2026 | 27.54 | -- | 1.9 min | ~0.38M |
| DashGaussian | 2025 | ~27+ | -- | 3.2 min | ~2.0M |
| Group Training | 2025 | 27.86* | 27.86 | ~39 min | -- |

*Group Training numbers are with Mip-Splatting backbone on garden scene specifically.

---

## 10. Key Techniques to Beat SteepGS on Quality AND Time

### Quality-boosting techniques (compatible with SteepGS):
1. **Half-Gaussian kernel (3D-HGS)**: +0.78 dB average improvement, drop-in kernel replacement
2. **MCMC relocation**: 3DGS-MCMC achieves 29.72 avg PSNR; MCMC+HGS reaches 30.13
3. **AbsGrad densification**: Better gradient signal for densification decisions
4. **Mip-Splatting filters**: Anti-aliasing with 3D+2D filters
5. **Depth/normal regularization**: Improved geometry consistency
6. **Higher iteration count or better LR schedules**: Exponential decay with careful warmup

### Speed-boosting techniques (compatible with SteepGS):
1. **DashGaussian resolution scheduling**: Progressive training from low to high resolution
2. **Group Training with OPS**: Selective Gaussian updates per iteration
3. **3DGS-LM optimizer**: Levenberg-Marquardt second phase for faster convergence
4. **Mini-Splatting2 aggressive densification**: Early dense point cloud extraction
5. **Taming 3DGS budget control**: Prevents unnecessary Gaussian proliferation
6. **FastGS multi-view consistency pruning**: Eliminates redundant Gaussians during training

### Ensemble / Test-Time Augmentation in 3DGS Literature:
- No standard TTA/ensemble approach exists in 3DGS literature (unlike NeRF where multi-model ensembles have been explored)
- **Multi-scale rendering** (Multi-Scale GS, Mip-Splatting) can be viewed as a form of scale-aware augmentation
- **View-adaptive attributes** (Scaffold-GS) implicitly perform per-view adaptation
- Potential novel contribution: test-time optimization with few gradient steps on held-out views

---

## 11. Recommended Strategy for Beating SteepGS

**Target**: Beat SteepGS's 28.73 PSNR at 1.6M Gaussians, while training faster than ~27 min.

**Proposed combination**:
1. Start from SteepGS's steepest-descent densification
2. Add Half-Gaussian kernel (+0.78 dB potential)
3. Apply DashGaussian resolution scheduling (~45% time reduction)
4. Use AbsGrad gradient accumulation for densification trigger
5. Apply Mip-Splatting 3D smoothing filter
6. Consider MCMC-style relocation for Gaussians stuck in poor local minima
7. Use depth regularization from monocular depth priors (Depth Anything V2)

**Expected outcome**: >29.5 PSNR with <15 min training time, at ~1.5-2M Gaussians.
