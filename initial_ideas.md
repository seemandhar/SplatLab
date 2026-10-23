# Initial Ideas for Improving SteepGS on MipNeRF360

## Idea 1: multi_scale_ssim_loss

**Hypothesis:** Replacing the single-scale DSSIM loss with a multi-scale SSIM (MS-SSIM) loss will capture structure at multiple frequency bands, improving reconstruction of both fine textures and large smooth regions that the steepest descent densification alone cannot resolve.

**Expected PSNR gain:** +0.15 to +0.25 dB

**Implementation complexity:** Low

**Pipeline component:** Loss

**Why it works on top of SteepGS:** SteepGS improves *where* new Gaussians are placed (via saddle-point escape and steepest descent directions), but does not change *what signal* drives the gradient. The current loss uses single-scale SSIM (window_size=11), which is blind to coarser structural errors. MS-SSIM evaluates at 2x, 4x, and 8x downsampled scales, providing gradient signal at frequencies that the single-window SSIM misses. This gives the steepest descent splitting matrices richer curvature information, making the saddle-point detection more accurate and the escape directions more informative. The implementation is a ~30-line change in `loss_utils.py` and a one-line swap in `train.py` (line 97). PyTorch-MS-SSIM is available as a drop-in. The key tuning knob is the weight vector across scales; the default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] from the MS-SSIM paper is a good starting point, but for GS we should truncate to 3 scales (the rendered images are not always large enough for 5 levels of 2x pooling).

---

## Idea 2: frequency_aware_densification_threshold

**Hypothesis:** Adapting the densification gradient threshold per-Gaussian based on the local spatial frequency content of the ground-truth image (high-frequency regions get a lower threshold, low-frequency regions get a higher one) will produce more Gaussians in textured areas and fewer in smooth skies, improving PSNR without inflating total Gaussian count.

**Expected PSNR gain:** +0.15 to +0.30 dB

**Implementation complexity:** Medium

**Pipeline component:** Densification

**Why it works on top of SteepGS:** SteepGS uses a global `densify_grad_threshold` (default 0.0002) and `densify_S_threshold` (default -1e-6) to decide which Gaussians to split. This one-size-fits-all threshold under-densifies complex texture regions (garden foliage, bicycle spokes) while over-densifying flat regions (sky, walls). The fix: at each densification step, project each Gaussian center into the most-recently-rendered view, compute a local Laplacian energy in a small patch around the projected 2D position, and scale the gradient threshold inversely with that energy. Concretely: `effective_threshold = base_threshold * (median_laplacian / (local_laplacian + eps))`. This is ~20 lines in `densify_and_prune` and reuses the already-available `viewspace_point_tensor` positions. It is orthogonal to the steepest descent direction computation -- it only changes *which* Gaussians are selected for splitting, not *how* they are split.

---

## Idea 3: opacity_entropy_regularization

**Hypothesis:** Adding a per-Gaussian opacity entropy regularization term (pushing opacities toward 0 or 1) during training will reduce floaters and semi-transparent artifacts, producing cleaner geometry that improves PSNR especially in the later training iterations after densification stops.

**Expected PSNR gain:** +0.10 to +0.20 dB

**Implementation complexity:** Low

**Pipeline component:** Loss / Regularization

**Why it works on top of SteepGS:** SteepGS focuses on position optimization via splitting matrix eigenanalysis, but does not regularize opacity. After the densification phase ends (iter 15000), many Gaussians settle at intermediate opacity values (0.2-0.8), creating semi-transparent layers that blur fine details. An entropy loss `L_ent = -mean(o * log(o + eps) + (1-o) * log(1-o + eps))` encourages binary opacities, effectively forcing the model to commit: either a Gaussian is fully present or it gets pruned. This is a 3-line addition to the loss in `train.py` with a weight schedule that ramps from 0 to 0.01 after densification ends. It is fully compatible with steepest descent because it only adds a diagonal term to the loss Hessian w.r.t. opacity (a parameter SteepGS does not compute splitting matrices for), so it does not interfere with the saddle-point analysis on xyz.

---

## Idea 4: progressive_resolution_training

**Hypothesis:** Training at 1/4 resolution for the first 3000 iterations, then 1/2 for 3000-7000, then full resolution thereafter will speed up early training by 3-4x (fewer pixels to rasterize) while providing a coarse-to-fine curriculum that helps the steepest descent optimizer place Gaussians at globally correct positions before refining detail.

**Expected PSNR gain:** +0.05 to +0.15 dB (with 1.3-1.5x total wall-clock speedup)

**Implementation complexity:** Medium

**Pipeline component:** Training schedule / Data loading

**Why it works on top of SteepGS:** The steepest descent splitting directions are computed from accumulated gradients over training views. At full resolution, early-stage gradients are noisy and dominated by high-frequency aliasing artifacts, which can mislead the splitting matrix eigenanalysis. Starting at low resolution provides smoother, more globally coherent gradients that produce better splitting directions for the initial densification rounds (iters 500-3000). As resolution increases, the splitting matrices naturally capture finer curvature. Implementation requires modifying `Scene.__init__` to precompute multi-resolution camera lists (SteepGS already supports `resolution_scales` in the Scene class), and adding a resolution schedule lookup in the training loop. The mipnerf360 images are large (1.6K+), so the speedup from early low-res training is substantial.

---

## Idea 5: scale_conditioned_split_ratio

**Hypothesis:** Instead of always splitting into N=2 children with a fixed scale reduction factor of 1/(0.8*N), using a scale-dependent split ratio (more aggressive shrinkage for very large Gaussians, gentler for medium ones) and allowing N=3 splits for the largest Gaussians will improve coverage of large scene regions while avoiding over-fragmentation of already-small Gaussians.

**Expected PSNR gain:** +0.10 to +0.20 dB

**Implementation complexity:** Low

**Pipeline component:** Densification

**Why it works on top of SteepGS:** In `densify_and_split_steepest` (gaussian_model.py line 619), the scale reduction is hardcoded as `self.get_scaling / (0.8*N)`. For very large Gaussians covering, say, an entire tree canopy, splitting into just 2 children with 62.5% scale often produces children that are still too large and require multiple subsequent split rounds to reach useful sizes -- wasting training iterations. Conversely, for medium Gaussians near the `percent_dense * scene_extent` boundary, the same aggressive shrinkage makes children too small, losing coverage. The fix: compute `scale_ratio = clamp(max_scale / (percent_dense * extent), 1, 4)` and set `N = 2 if scale_ratio < 2 else 3`, `shrink = 1 / (0.8 * N * sqrt(scale_ratio))`. This is a 5-line change in the split function. It leverages SteepGS's superior split directions (eigenvectors of the splitting matrix) by spreading children along those directions at more appropriate distances, making each split more effective.
