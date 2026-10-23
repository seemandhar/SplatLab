# SteepGS: Steepest Descent Density Control for Compact 3D Gaussian Splatting

**Paper**: https://arxiv.org/abs/2505.05587
**Venue**: CVPR 2025
**Authors**: Peihao Wang, Yuehao Wang, Dilin Wang, Sreyas Mohan, Zhiwen Fan, Lemeng Wu, Ruisi Cai, Yu-Ying Yeh, Zhangyang Wang, Qiang Liu, Rakesh Ranjan (Facebook Research / VITA Group)

---

## 1. Aggregate Results (Table 1 from main paper)

### MipNeRF360

| Method           | # Points (M) | PSNR   | SSIM  | LPIPS |
|------------------|--------------|--------|-------|-------|
| 3DGS             | 3.339        | 29.037 | 0.872 | 0.183 |
| 3DGS + Thres.    | 1.632        | 27.851 | 0.848 | 0.227 |
| 3DGS-MCMC        | 1.606        | 28.149 | 0.853 | 0.204 |
| Revising 3DGS    | 1.606        | 28.085 | 0.850 | 0.212 |
| **SteepGS**      | **1.606**    | **28.734** | **0.857** | **0.211** |

### Tank & Temple

| Method           | # Points (M) | PSNR   | SSIM  | LPIPS |
|------------------|--------------|--------|-------|-------|
| 3DGS             | 1.830        | 23.743 | 0.848 | 0.177 |
| 3DGS + Thres.    | 0.973        | 22.415 | 0.812 | 0.218 |
| 3DGS-MCMC        | 0.957        | 22.545 | 0.817 | 0.204 |
| Revising 3DGS    | 0.957        | 22.339 | 0.811 | 0.216 |
| **SteepGS**      | **0.958**    | **23.684** | **0.840** | **0.194** |

### Deep Blending

| Method           | # Points (M) | PSNR   | SSIM  | LPIPS |
|------------------|--------------|--------|-------|-------|
| 3DGS             | 2.818        | 29.690 | 0.904 | 0.244 |
| 3DGS + Thres.    | 1.326        | 29.374 | 0.900 | 0.250 |
| 3DGS-MCMC        | 1.296        | 29.439 | 0.901 | 0.237 |
| Revising 3DGS    | 1.296        | 29.439 | 0.895 | 0.245 |
| **SteepGS**      | **1.296**    | **29.963** | **0.905** | **0.250** |

---

## 2. Per-Scene Breakdown (Tables 2-5 from Appendix B)

Note: The paper reports per-scene results for 7 mipnerf360 scenes (3 outdoor + 4 indoor). Scenes "flowers" and "treehill" are NOT included in the paper's evaluation.

### Table 2: Per-Scene PSNR

| Method   | Train  | Truck  | Dr. Johnson | Playroom | Bicycle | Garden | Stump  | Bonsai | Counter | Kitchen | Room   |
|----------|--------|--------|-------------|----------|---------|--------|--------|--------|---------|---------|--------|
| 3DGS     | 22.091 | 25.394 | 29.209      | 30.172   | 25.253  | 27.417 | 26.705 | 32.298 | 29.006  | 31.628  | 31.540 |
| SteepGS  | 21.974 | 25.395 | 29.478      | 30.447   | 24.890  | 27.159 | 26.115 | 31.911 | 28.737  | 31.030  | 31.401 |

### Table 3: Per-Scene SSIM

| Method   | Train | Truck | Dr. Johnson | Playroom | Bicycle | Garden | Stump | Bonsai | Counter | Kitchen | Room  |
|----------|-------|-------|-------------|----------|---------|--------|-------|--------|---------|---------|-------|
| 3DGS     | 0.813 | 0.882 | 0.901       | 0.907    | 0.766   | 0.867  | 0.908 | 0.773  | 0.942   | 0.928   | 0.919 |
| SteepGS  | 0.802 | 0.879 | 0.902       | 0.909    | 0.734   | 0.851  | 0.742 | 0.938  | 0.900   | 0.922   | 0.915 |

### Table 4: Per-Scene LPIPS

| Method   | Train | Truck | Dr. Johnson | Playroom | Bicycle | Garden | Stump | Bonsai | Counter | Kitchen | Room  |
|----------|-------|-------|-------------|----------|---------|--------|-------|--------|---------|---------|-------|
| 3DGS     | 0.207 | 0.147 | 0.244       | 0.244    | 0.210   | 0.107  | 0.215 | 0.203  | 0.200   | 0.126   | 0.219 |
| SteepGS  | 0.230 | 0.160 | 0.251       | 0.250    | 0.268   | 0.142  | 0.271 | 0.211  | 0.217   | 0.137   | 0.233 |

### Table 5: Per-Scene Number of Points Densified

| Method   | Train   | Truck   | Dr. Johnson | Playroom | Bicycle | Garden  | Stump   | Bonsai  | Counter | Kitchen | Room    |
|----------|---------|---------|-------------|----------|---------|---------|---------|---------|---------|---------|---------|
| 3DGS     | 1088197 | 2572172 | 3316036     | 2320830  | 6074705 | 5845401 | 4863462 | 1260017 | 1195896 | 1807771 | 1550152 |
| SteepGS  | 530476  | 1387065 | 1485567     | 1107604  | 2900640 | 2195185 | 3175021 | 746163  | 591508  | 922717  | 710212  |

---

## 3. Training Time and Resource Comparisons (Table 7 from Appendix B)

### MipNeRF360

| Method          | # Points (M) | PSNR   | SSIM  | LPIPS | GPU Elapsed (s) | Mean GPU Mem (GB) | Peak GPU Mem (GB) | FPS |
|-----------------|--------------|--------|-------|-------|-----------------|-------------------|-------------------|-----|
| 3DGS            | 3.339        | 29.037 | 0.872 | 0.183 | 1550.925        | 10.262            | 12.110            | 179 |
| LP-3DGS         | 1.303        | 28.640 | 0.865 | 0.198 | 1177.648        | 10.027            | 12.458            | 350 |
| Compact-3DGS    | 1.310        | 28.504 | 0.856 | 0.208 | 4063.203        | 7.274             | 9.044             | 98  |
| **SteepGS**     | **1.606**    | **28.734** | **0.857** | **0.211** | **1051.276** | **7.597**     | **8.957**         | **252** |

### Tank & Temple

| Method          | # Points (M) | PSNR   | SSIM  | LPIPS | GPU Elapsed (s) | Mean GPU Mem (GB) | Peak GPU Mem (GB) | FPS |
|-----------------|--------------|--------|-------|-------|-----------------|-------------------|-------------------|-----|
| 3DGS            | 1.830        | 23.743 | 0.848 | 0.177 | 803.542         | 5.193             | 6.241             | 248 |
| LP-3DGS         | 0.671        | 23.424 | 0.839 | 0.197 | 1021.806        | 5.045             | 6.489             | 150 |
| Compact-3DGS    | 0.836        | 23.319 | 0.835 | 0.200 | 1255.748        | 3.802             | 4.774             | 357 |
| **SteepGS**     | **0.958**    | **23.684** | **0.840** | **0.194** | **539.048** | **4.701**     | **5.607**         | **343** |

### Deep Blending

| Method          | # Points (M) | PSNR   | SSIM  | LPIPS | GPU Elapsed (s) | Mean GPU Mem (GB) | Peak GPU Mem (GB) | FPS |
|-----------------|--------------|--------|-------|-------|-----------------|-------------------|-------------------|-----|
| 3DGS            | 2.818        | 29.690 | 0.904 | 0.244 | 1429.878        | 8.668             | 10.218            | 187 |
| LP-3DGS         | 0.861        | 29.764 | 0.906 | 0.249 | 1697.793        | 8.354             | 10.115            | 134 |
| Compact-3DGS    | 1.054        | 29.896 | 0.905 | 0.255 | 1861.897        | 6.332             | 8.026             | 312 |
| **SteepGS**     | **1.296**    | **29.963** | **0.905** | **0.250** | **956.536** | **5.928**     | **9.506**         | **280** |

---

## 4. Improved Performance with More Densification Steps (Table 6)

Extending densification iterations to 25K and total training to 40K steps on select MipNeRF360 outdoor scenes:

| Method              | Bicycle | Garden | Stump |
|---------------------|---------|--------|-------|
| 3DGS                | 25.25   | 27.42  | 26.70 |
| SteepGS             | 24.89   | 27.16  | 26.11 |
| SteepGS (more steps)| 25.23   | 27.38  | 26.65 |

With more steps, SteepGS nearly matches 3DGS quality while using ~50% fewer points.

---

## 5. Key Method Details

### What makes SteepGS different from vanilla 3DGS?

1. **Splitting Matrix Framework**: Introduces a "splitting matrix" S(i) for each Gaussian that links first-order gradients to second-order Hessian information. This matrix characterizes the local curvature landscape around each Gaussian.

2. **Principled Splitting Criterion**: Instead of the heuristic gradient-magnitude threshold used in 3DGS's Adaptive Density Control (ADC), SteepGS splits a Gaussian ONLY if the minimum eigenvalue of its splitting matrix is negative (lambda_min < epsilon_split). A negative eigenvalue indicates a saddle point where splitting can decrease the loss.

3. **Optimal Offspring Configuration (Theorem 2)**:
   - Split into exactly **2 offspring** (not more -- mathematically proven to be sufficient)
   - Each offspring gets **half the parent's opacity** (o/2)
   - Offspring are displaced along the **eigenvector of the smallest eigenvalue** of the splitting matrix (positive and negative directions)
   - Displacement magnitude controlled by stepsize epsilon

4. **Compactest Splitting Strategy**: A Gaussian should be split only when its gradient is small (||G(i)|| <= epsilon_grad) AND the splitting matrix has a negative eigenvalue. This prevents splitting Gaussians that are already well-optimized.

5. **No post-hoc compression needed**: Unlike LightGaussian and similar methods, SteepGS prevents overproduction of Gaussians during training itself, rather than pruning after training.

### Core Contributions
- Theoretical framework proving densification helps escape saddle points
- Analytical solution for optimal splitting direction and offspring parameters
- ~50% reduction in Gaussian count without quality loss
- Faster training (1051s vs 1551s on MipNeRF360) and lower memory usage

---

## 6. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Density control frequency (T_split) | Every 100 steps | How often splitting is evaluated |
| Density control start | Step 500 | When density control begins |
| Splitting matrix threshold (epsilon_split) | -1e-6 | Minimum eigenvalue threshold for splitting |
| Stepsize (epsilon) | Not specified (from 3DGS default) | Offspring displacement magnitude |
| Gradient threshold (epsilon_grad) | Not specified | For compactest splitting variant |
| Total training iterations | 30K (default) / 40K (extended) | Standard 3DGS schedule |
| Densification iterations | 15K (default) / 25K (extended) | When densification stops |
| All other params | 3DGS defaults | Learning rates, SH degree, etc. |

**Hardware**: Single NVIDIA V100 GPU per scene

---

## 7. Observations and Potential Improvements

### Strengths
- Strong theoretical grounding (optimization-theoretic framework)
- Significant point reduction (~50%) with minimal quality drop
- Faster training and lower memory than 3DGS
- Better PSNR than all baselines at similar point budgets
- Compatible with existing 3DGS pipeline (drop-in replacement for ADC)

### Weaknesses / Areas for Improvement
- Per-scene LPIPS is consistently worse than 3DGS (higher values across all scenes)
- Per-scene SSIM is also generally lower than 3DGS
- Outdoor scenes (Bicycle, Garden, Stump) show larger quality gaps than indoor scenes
- The "flowers" and "treehill" scenes from mipnerf360 are not evaluated (only 7 of 9 scenes)
- PSNR gap on outdoor scenes: Bicycle -0.36 dB, Garden -0.26 dB, Stump -0.59 dB
- With extended training (40K steps), quality gap closes but training time advantage diminishes
- Custom CUDA kernel required for splitting matrix computation (implementation complexity)
- V100 GPU used (older hardware) -- unclear how it performs on modern GPUs

### Baselines NOT compared against (but relevant)
- Mip-Splatting
- Mini-Splatting
- GaussianPro
- Scaffold-GS
- HAC (Hash-grid Assisted Context)
