"""
FSGS — Real-Time Few-shot View Synthesis Using Gaussian Splatting
==================================================================
Paper: https://arxiv.org/abs/2404.xxxxx
Complexity: ★★★☆☆ (custom loss + DataManager + new densification + pseudo-views)

Three key contributions:
    1. Proximity-guided Gaussian Unpooling — densifies sparse regions by
       inserting new Gaussians at midpoints between isolated Gaussians
       and their K nearest neighbors.
    2. Pseudo-view Synthesis — augments training with virtual viewpoints
       interpolated between the two closest training cameras, with
       monocular depth consistency as regularization.
    3. Depth Regularization — Pearson correlation between rendered depth
       and monocular depth estimates on both real and pseudo views.

Architecture:
    DepthDataManager  — runs monocular depth estimation on training images
                        at startup, caches to disk, injects batch["depth_image"]
    FSGSModel         — consumes depth maps, synthesizes pseudo views,
                        performs proximity unpooling, and computes Pearson
                        depth losses.

Demonstrates:
    - Custom DataManager for preprocessing (DepthDataManager)
    - compute_custom_losses() with depth Pearson + pseudo-view depth losses
    - Custom densification via proximity-guided Gaussian unpooling
    - Opacity reset schedule for floater removal
    - SH degree progressive schedule (0→4 every 500 iters)

Train:
    ns-train fsgs --data <path>
    ns-train fsgs --data <path> --pipeline.model.proximity_k 3
    ns-train fsgs --data <path> --pipeline.model.pseudo_view_noise_std 0.01

Ambiguities / Notes:
    - The paper does not specify lambda_4 separately from lambda_3 for
      pseudo-view depth loss; we default both to 0.05.
    - The proximity threshold t_prox=10 is scene-dependent; may need tuning
      for unbounded scenes (Mip-NeRF360).
    - Pseudo-view depth estimation runs the depth model on rendered images
      during training, adding ~30% wall-clock overhead.
    - The paper references "open-source code" for depth estimation but
      does not provide a URL; we use Depth Anything V2 by default (paper
      showed robustness across estimators including DPT, MiDaS, DepthAnything).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.utils.rich_utils import CONSOLE

from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.losses import pearson_depth_loss


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FSGSModelConfig(GSTemplateModelConfig):
    """FSGS configuration — few-shot Gaussian Splatting.

    Key hyperparameters from the paper (Table in Sec. 4):
        lambda_1=0.8, lambda_2=0.2, lambda_3=0.05 (depth reg)
        K=3 (proximity neighbors), t_prox=10 (proximity threshold)
        Pseudo-views start at iter 2000
        SH degree increases every 500 iters up to degree 4
        Opacity reset at iters {2000, 5000, 7000}
        Total training: 10,000 iterations
    """

    _target: Type = field(default_factory=lambda: FSGSModel)

    # ── Rendering (must output depth for Pearson loss) ─────────────
    output_depth_during_training: bool = True

    # ── Loss weights (from paper grid search) ──────────────────────
    # NOTE: lambda_1 and lambda_2 are the base L1 and D-SSIM weights
    # handled by splatfacto (default 0.8/0.2 matches the paper).
    depth_loss_weight: float = 0.05
    """lambda_3: Pearson depth correlation weight on training views."""

    pseudo_depth_loss_weight: float = 0.05
    """lambda_4: Pearson depth correlation weight on pseudo views.
    Paper does not specify this separately; we use same as lambda_3."""

    # ── Pseudo-view synthesis ──────────────────────────────────────
    pseudo_view_start_iter: int = 2000
    """Start pseudo-view augmentation after this many iterations."""

    pseudo_view_noise_std: float = 0.01
    """Standard deviation (delta) for Gaussian noise on pseudo camera
    position. Controls how far virtual cameras deviate from the
    interpolated baseline."""

    # ── Proximity-guided Gaussian Unpooling ────────────────────────
    proximity_k: int = 3
    """K: number of nearest neighbors for proximity graph."""

    proximity_threshold: float = 10.0
    """t_prox: Gaussians with mean neighbor distance > this are
    candidates for unpooling. Scene-scale dependent."""

    unpooling_start_iter: int = 500
    """Start proximity unpooling after this iteration (matches
    densification start in the paper)."""

    unpooling_every: int = 100
    """Run proximity unpooling every N iterations (matches
    densification frequency)."""

    # ── Opacity reset schedule ─────────────────────────────────────
    opacity_reset_value: float = 0.05
    """Value to reset opacity logits to at reset iterations.
    Stored as inverse-sigmoid(0.05) ≈ -2.944."""

    opacity_reset_iters: Tuple[int, ...] = (2000, 5000, 7000)
    """Iterations at which to reset all Gaussian opacities."""

    # ── SH degree schedule ─────────────────────────────────────────
    sh_degree: int = 4
    """Maximum SH degree (paper uses up to degree 4)."""

    sh_degree_interval: int = 500
    """Increase SH degree by 1 every this many iterations."""

    # ── Depth estimator for pseudo-views ───────────────────────────
    pseudo_depth_model: Literal["depth_anything_v2", "dpt"] = "depth_anything_v2"
    """Depth estimator for pseudo-view regularization.
    Paper uses DPT by default but shows robustness across estimators."""

    pseudo_depth_model_size: str = "vitb"
    """Model size for depth estimator (vitb is a good speed/quality trade-off)."""

    # ── Blender dataset handling ───────────────────────────────────
    blender_depth_mask: bool = False
    """If True, compute Pearson correlation only on pixels where
    depth > 0 (for NeRF-Synthetic/Blender datasets)."""

    # ── Initialization ─────────────────────────────────────────────
    # Paper initializes opacity at 0.1 (vs splatfacto default of ~0.5)
    # Rotation init at 0 (quaternion [1,0,0,0] — identity)
    # Scale from mean point spacing
    # These are partially handled by splatfacto defaults.


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class FSGSModel(GSTemplateModel):
    """FSGS: Few-shot View Synthesis with Gaussian Splatting.

    Implements proximity-guided Gaussian unpooling, pseudo-view
    synthesis, and depth-regularized optimization.
    """

    config: FSGSModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        """Initialize base modules + FSGS-specific state."""
        super().populate_modules()

        # Storage for training camera poses (for pseudo-view synthesis)
        self._train_cam_positions: List[Tensor] = []
        self._train_cam_quats: List[Tensor] = []
        self._train_cam_intrinsics: Optional[Tensor] = None
        self._train_cam_width: int = 0
        self._train_cam_height: int = 0
        self._cameras_collected: bool = False

        # Lazy-loaded depth estimator for pseudo-views
        self._pseudo_depth_pipe = None

        # Optimizer reference — captured in get_training_callbacks()
        self._optimizers: Optional[Dict] = None

        CONSOLE.print("[bold green]FSGS[/] model initialized")
        CONSOLE.print(f"  Proximity K={self.config.proximity_k}, "
                      f"threshold={self.config.proximity_threshold}")
        CONSOLE.print(f"  Pseudo-views start at iter {self.config.pseudo_view_start_iter}")
        CONSOLE.print(f"  Opacity reset at iters {self.config.opacity_reset_iters}")

    # ── Training callbacks (captures optimizers for unpooling) ─────

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Register FSGS-specific training callbacks.

        Captures the optimizer dict (not available in get_loss_dict)
        so proximity unpooling and opacity reset can properly manage
        optimizer state when growing/modifying Gaussians.
        """
        # Capture optimizers from the trainer
        # nerfstudio wraps optimizers in an Optimizers object;
        # .optimizers gives us the raw {name: torch.optim.Optimizer} dict
        opt_obj = training_callback_attributes.optimizers
        if hasattr(opt_obj, "optimizers"):
            self._optimizers = opt_obj.optimizers  # dict of torch optimizers
        else:
            self._optimizers = opt_obj  # already a dict

        # Get splatfacto's callbacks (handles gsplat strategy)
        cbs = super().get_training_callbacks(training_callback_attributes)

        # FSGS callbacks run AFTER the full train iteration
        # (forward → backward → gsplat densification → optimizer step)
        # This is the only safe time to extend params because:
        #   - No computation graph holds references to old param storage
        #   - gsplat's strategy has finished its own grow/prune cycle
        #   - Optimizer has already stepped on the current params

        def _fsgs_after_train(step: int):
            """Run proximity unpooling and opacity reset after the
            full training iteration is complete."""
            # Opacity reset at scheduled iterations
            if step in self.config.opacity_reset_iters:
                self._reset_opacities()

            # Proximity unpooling
            if (self.training
                    and step >= self.config.unpooling_start_iter
                    and step % self.config.unpooling_every == 0):
                self._proximity_unpooling()

        cbs.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=_fsgs_after_train,
            )
        )

        return cbs

    # ── Depth estimator management ─────────────────────────────────

    def _get_depth_estimator(self):
        """Lazy-load depth estimator for pseudo-view depth estimation.

        Loaded on first use to avoid GPU memory overhead during early
        training when pseudo-views are not yet active.
        """
        if self._pseudo_depth_pipe is not None:
            return self._pseudo_depth_pipe

        CONSOLE.print("[bold green]FSGS[/] Loading depth estimator for "
                      f"pseudo-views: {self.config.pseudo_depth_model}")

        if self.config.pseudo_depth_model == "depth_anything_v2":
            try:
                from transformers import pipeline as hf_pipeline
                # Map vitX codes to HuggingFace model names
                _size_map = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
                hf_size = _size_map.get(
                    self.config.pseudo_depth_model_size,
                    self.config.pseudo_depth_model_size.capitalize(),
                )
                model_id = (
                    f"depth-anything/Depth-Anything-V2-{hf_size}-hf"
                )
                self._pseudo_depth_pipe = hf_pipeline(
                    "depth-estimation", model=model_id, device="cuda"
                )
            except ImportError:
                CONSOLE.print(
                    "[yellow]Warning: transformers not installed, "
                    "disabling pseudo-view depth. "
                    "Install with: pip install transformers[/]"
                )
                return None
        else:
            # DPT via torch hub (MiDaS DPT-Large)
            try:
                model = torch.hub.load(
                    "intel-isl/MiDAS", "DPT_Large", pretrained=True
                )
                model = model.cuda().eval()
                midas_transforms = torch.hub.load(
                    "intel-isl/MiDAS", "transforms"
                )
                self._pseudo_depth_pipe = {
                    "model": model,
                    "transform": midas_transforms.dpt_transform,
                }
            except Exception:
                CONSOLE.print(
                    "[yellow]Warning: DPT/MiDaS failed to load, "
                    "disabling pseudo-view depth.[/]"
                )
                return None

        return self._pseudo_depth_pipe

    @torch.no_grad()
    def _estimate_depth_from_image(self, rgb_tensor: Tensor) -> Optional[Tensor]:
        """Run monocular depth estimation on a rendered RGB image.

        Args:
            rgb_tensor: (H, W, 3) float tensor in [0, 1]

        Returns:
            (H, W, 1) depth tensor, or None if estimator unavailable
        """
        pipe = self._get_depth_estimator()
        if pipe is None:
            return None

        H, W = rgb_tensor.shape[:2]

        if self.config.pseudo_depth_model == "depth_anything_v2":
            from PIL import Image
            # Convert to PIL
            rgb_np = (rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb_np)
            result = pipe(pil_img)
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil, dtype=np.float32)
            depth_t = torch.from_numpy(depth_np).to(rgb_tensor.device)
            # Normalize
            dmax = depth_t.max()
            if dmax > 0:
                depth_t = depth_t / dmax
            # Resize to match rendered resolution
            if depth_t.shape[0] != H or depth_t.shape[1] != W:
                depth_t = F.interpolate(
                    depth_t.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
            return depth_t.unsqueeze(-1)  # (H, W, 1)
        else:
            # DPT/MiDaS
            model = pipe["model"]
            transform = pipe["transform"]
            rgb_np = (rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            from PIL import Image
            pil_img = Image.fromarray(rgb_np)
            input_batch = transform(pil_img).cuda()
            if input_batch.dim() == 3:
                input_batch = input_batch.unsqueeze(0)
            depth = model(input_batch)
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze()
            # MiDAS outputs inverse depth
            depth = 1.0 / (depth + 1e-6)
            depth = depth / (depth.max() + 1e-6)
            return depth.unsqueeze(-1).to(rgb_tensor.device)

    # ── Camera pose collection ─────────────────────────────────────

    def _collect_camera(self, camera: Cameras):
        """Collect training camera poses for pseudo-view synthesis.

        Called during the first pass through training images to store
        all camera positions and orientations.
        """
        if self._cameras_collected:
            return

        c2w = camera.camera_to_worlds  # (1, 3, 4)
        if c2w.dim() == 3:
            c2w = c2w.squeeze(0)  # (3, 4)

        position = c2w[:3, 3].detach().cpu()
        rotation_matrix = c2w[:3, :3].detach().cpu()

        # Convert rotation matrix to quaternion
        quat = _rotation_matrix_to_quaternion(rotation_matrix)

        self._train_cam_positions.append(position)
        self._train_cam_quats.append(quat)

        # Store intrinsics from first camera (assumed same for all)
        if self._train_cam_intrinsics is None:
            self._train_cam_intrinsics = camera.get_intrinsics_matrices().detach().cpu()
            self._train_cam_width = int(camera.width.item())
            self._train_cam_height = int(camera.height.item())

    # ── Pseudo-view synthesis ──────────────────────────────────────

    def _synthesize_pseudo_camera(self) -> Optional[Cameras]:
        """Synthesize a pseudo camera by interpolating between the two
        closest training cameras and adding positional noise.

        Returns a Cameras object for the pseudo viewpoint, or None if
        insufficient training cameras have been collected.
        """
        if len(self._train_cam_positions) < 2:
            return None

        positions = torch.stack(self._train_cam_positions)  # (N, 3)
        quats = torch.stack(self._train_cam_quats)          # (N, 4)

        # Pick a random training camera as anchor
        idx = torch.randint(0, len(positions), (1,)).item()
        anchor_pos = positions[idx]

        # Find the closest other camera
        dists = (positions - anchor_pos.unsqueeze(0)).norm(dim=-1)
        dists[idx] = float("inf")
        nearest_idx = dists.argmin().item()

        # Interpolate position (midpoint + noise)
        t = torch.rand(1).item()  # random interpolation factor
        pseudo_pos = (1 - t) * positions[idx] + t * positions[nearest_idx]

        # Add Gaussian noise to position
        noise = torch.randn(3) * self.config.pseudo_view_noise_std
        pseudo_pos = pseudo_pos + noise

        # Average quaternions (simple normalized mean — approximation
        # of SLERP for nearby orientations)
        q1 = quats[idx]
        q2 = quats[nearest_idx]
        # Ensure quaternions are in the same hemisphere
        if (q1 * q2).sum() < 0:
            q2 = -q2
        pseudo_quat = F.normalize((1 - t) * q1 + t * q2, dim=0)

        # Build c2w matrix from position + quaternion
        R = _quaternion_to_rotation_matrix(pseudo_quat)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pseudo_pos

        # Create Cameras object
        pseudo_camera = Cameras(
            camera_to_worlds=c2w[:3, :].unsqueeze(0).to(self.device),
            fx=self._train_cam_intrinsics[0, 0, 0].unsqueeze(0).to(self.device),
            fy=self._train_cam_intrinsics[0, 1, 1].unsqueeze(0).to(self.device),
            cx=self._train_cam_intrinsics[0, 0, 2].unsqueeze(0).to(self.device),
            cy=self._train_cam_intrinsics[0, 1, 2].unsqueeze(0).to(self.device),
            width=torch.tensor([self._train_cam_width]).to(self.device),
            height=torch.tensor([self._train_cam_height]).to(self.device),
        )
        return pseudo_camera

    # ── Proximity-Guided Gaussian Unpooling ────────────────────────

    @torch.no_grad()
    def _proximity_unpooling(self):
        """Insert new Gaussians between isolated Gaussians and their
        nearest neighbors.

        Algorithm:
            1. Build K-NN proximity graph on Gaussian means
            2. Compute proximity score P_i = mean distance to K neighbors
            3. For each Gaussian with P_i > t_prox:
               - Create new Gaussian at midpoint with each neighbor
               - Scale/opacity from destination neighbor
               - Rotation and SH initialized to zero
        """
        means = self.means.detach()  # (N, 3)
        N = means.shape[0]
        K = self.config.proximity_k

        if N < K + 1:
            return

        # ── Build K-NN graph ───────────────────────────────────────
        # For large point clouds, use chunked distance computation
        chunk_size = 8192
        knn_dists = torch.zeros(N, K, device=means.device)
        knn_indices = torch.zeros(N, K, dtype=torch.long, device=means.device)

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            # (chunk, N) pairwise distances
            dists = torch.cdist(means[i:end], means)
            # Set self-distance to inf
            for j in range(end - i):
                dists[j, i + j] = float("inf")
            # K smallest distances
            topk = dists.topk(K, largest=False)
            knn_dists[i:end] = topk.values
            knn_indices[i:end] = topk.indices

        # ── Proximity scores ───────────────────────────────────────
        proximity_scores = knn_dists.mean(dim=-1)  # (N,)

        # ── Identify candidates for unpooling ──────────────────────
        candidates = proximity_scores > self.config.proximity_threshold

        num_candidates = candidates.sum().item()
        if num_candidates == 0:
            return

        # Cap the number of new Gaussians to avoid memory explosion
        max_new = min(num_candidates * K, N)
        if num_candidates * K > max_new:
            # Randomly subsample candidates
            candidate_indices = candidates.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(candidate_indices))[:max_new // K]
            candidates = torch.zeros(N, dtype=torch.bool, device=means.device)
            candidates[candidate_indices[perm]] = True
            num_candidates = candidates.sum().item()

        if num_candidates == 0:
            return

        CONSOLE.print(
            f"[bold green]FSGS[/] Proximity unpooling: "
            f"{num_candidates} candidates → "
            f"{num_candidates * K} new Gaussians"
        )

        # ── Create new Gaussians at midpoints ──────────────────────
        src_indices = candidates.nonzero(as_tuple=True)[0]  # (M,)
        dst_indices = knn_indices[src_indices]  # (M, K)

        # Flatten: each source creates K new Gaussians
        src_expanded = src_indices.unsqueeze(1).expand(-1, K).reshape(-1)  # (M*K,)
        dst_flat = dst_indices.reshape(-1)  # (M*K,)

        # New means: midpoint between source and destination
        new_means = 0.5 * (means[src_expanded] + means[dst_flat])

        # New scales: from destination Gaussian
        new_scales = self.scales.data[dst_flat].clone()

        # New opacities: from destination Gaussian
        new_opacities = self.opacities.data[dst_flat].clone()

        # New rotation: zero quaternion → identity [1, 0, 0, 0] in
        # nerfstudio's parameterization (stored as raw quaternion)
        new_quats = torch.zeros(
            new_means.shape[0], 4, device=means.device
        )
        new_quats[:, 0] = 1.0  # identity quaternion

        # New SH coefficients: zero (will be optimized)
        new_features_dc = torch.zeros(
            new_means.shape[0],
            self.features_dc.shape[1],
            device=means.device,
        )
        new_features_rest = torch.zeros(
            new_means.shape[0],
            *self.features_rest.shape[1:],
            device=means.device,
        )

        # ── Concatenate with existing Gaussians ────────────────────
        self._extend_gauss_params({
            "means": new_means,
            "scales": new_scales,
            "opacities": new_opacities,
            "quats": new_quats,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
        })

    def _extend_gauss_params(self, new_params: Dict[str, Tensor]):
        """Concatenate new Gaussian parameters to existing ones,
        keeping gsplat's strategy state and optimizers in sync.

        Uses param.data= assignment to preserve Parameter identity
        (so optimizers keep their reference), then pads optimizer
        momentum/variance buffers and strategy state tensors.
        """
        n_new = new_params["means"].shape[0]

        # ── 1. Extend gauss_params via .data= (preserves identity) ─
        for name, param in self.gauss_params.items():
            if name in new_params:
                new_data = torch.cat(
                    [param.data, new_params[name].to(param.device)], dim=0
                )
            else:
                extra = torch.zeros(
                    n_new, *param.shape[1:],
                    device=param.device, dtype=param.dtype,
                )
                new_data = torch.cat([param.data, extra], dim=0)
            # .data= keeps the same Parameter object in memory,
            # so optimizer param_groups still reference it correctly
            param.data = new_data

        # ── 2. Pad optimizer state (exp_avg, exp_avg_sq for Adam) ──
        # Without this, Adam crashes when momentum buffer shape
        # doesn't match the extended parameter shape.
        if self._optimizers is not None:
            for name, param in self.gauss_params.items():
                # Find the optimizer that manages this param
                for opt_key, optimizer in self._optimizers.items():
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        for skey, sval in state.items():
                            if isinstance(sval, Tensor) and sval.dim() >= 1:
                                if sval.shape[0] == param.shape[0] - n_new:
                                    pad = torch.zeros(
                                        n_new, *sval.shape[1:],
                                        device=sval.device, dtype=sval.dtype,
                                    )
                                    state[skey] = torch.cat([sval, pad], dim=0)
                        break

        # ── 3. Pad gsplat strategy state ───────────────────────────
        # grad2d, count, etc. must match the new Gaussian count.
        if hasattr(self, "strategy_state") and self.strategy_state is not None:
            n_total = self.num_points  # already extended
            for key, val in self.strategy_state.items():
                if isinstance(val, Tensor) and val.dim() >= 1:
                    if val.shape[0] == n_total - n_new:
                        pad = torch.zeros(
                            n_new, *val.shape[1:],
                            device=val.device, dtype=val.dtype,
                        )
                        self.strategy_state[key] = torch.cat([val, pad], dim=0)

    # ── Opacity Reset ──────────────────────────────────────────────

    @torch.no_grad()
    def _reset_opacities(self):
        """Reset all Gaussian opacities to a low value.

        This removes floaters that accumulate during optimization.
        The paper resets to 0.05 (in opacity space), which corresponds
        to a logit value of inverse_sigmoid(0.05).
        """
        reset_logit = torch.logit(
            torch.tensor(self.config.opacity_reset_value)
        ).item()
        self.opacities.data.fill_(reset_logit)
        CONSOLE.print(
            f"[bold green]FSGS[/] Reset opacities to "
            f"{self.config.opacity_reset_value} at step {self.step}"
        )

    # ── 3. Loss computation ────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Compute all losses including FSGS-specific depth losses.

        Note: Proximity unpooling and opacity reset are handled by
        the training callback (AFTER_TRAIN_ITERATION) to ensure
        proper optimizer/strategy state management.
        """
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        return loss_dict

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """FSGS custom losses: depth regularization + pseudo-view depth.

        Loss terms:
            depth_reg:       lambda_3 * |Pearson(D_rasterized, D_estimated)|
            pseudo_depth:    lambda_4 * |Pearson(D_pseudo_ras, D_pseudo_est)|
        """
        losses = {}

        # ── Collect training camera poses ──────────────────────────
        # (needed for pseudo-view synthesis)
        if self.training and "camera" in batch:
            self._collect_camera(batch["camera"])

        # ── Depth regularization on training views ─────────────────
        if "depth_image" in batch and outputs.get("depth") is not None:
            pred_depth = outputs["depth"]  # (H, W, 1)
            gt_depth = batch["depth_image"].to(pred_depth.device)

            if gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(-1)

            # Handle resolution mismatch
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = F.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),
                    size=(int(pred_depth.shape[0]), int(pred_depth.shape[1])),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            # Valid mask (for Blender: only where depth > 0)
            if self.config.blender_depth_mask:
                valid = gt_depth > 0
            else:
                valid = torch.ones_like(gt_depth, dtype=torch.bool)

            if valid.sum() > 100:
                losses["depth_reg"] = (
                    self.config.depth_loss_weight
                    * pearson_depth_loss(
                        pred_depth[valid], gt_depth[valid]
                    )
                )

        # ── Pseudo-view depth regularization ───────────────────────
        if (self.training
                and self.step >= self.config.pseudo_view_start_iter
                and len(self._train_cam_positions) >= 2):
            pseudo_loss = self._compute_pseudo_view_loss()
            if pseudo_loss is not None:
                losses["pseudo_depth"] = pseudo_loss

        return losses

    def _compute_pseudo_view_loss(self) -> Optional[Tensor]:
        """Synthesize a pseudo view, render it, estimate depth,
        and compute Pearson correlation loss.

        Returns:
            Scalar loss tensor, or None if estimation fails.
        """
        # Synthesize pseudo camera
        pseudo_cam = self._synthesize_pseudo_camera()
        if pseudo_cam is None:
            return None

        # Render from pseudo camera (no gradients through camera synthesis,
        # but gradients flow through Gaussian parameters)
        pseudo_outputs = self.get_outputs(pseudo_cam)

        if pseudo_outputs.get("rgb") is None or pseudo_outputs.get("depth") is None:
            return None

        rendered_rgb = pseudo_outputs["rgb"]      # (H, W, 3)
        rendered_depth = pseudo_outputs["depth"]   # (H, W, 1)

        # Estimate depth from rendered RGB using monocular estimator
        estimated_depth = self._estimate_depth_from_image(rendered_rgb)
        if estimated_depth is None:
            return None

        estimated_depth = estimated_depth.to(rendered_depth.device)

        # Pearson correlation loss
        if self.config.blender_depth_mask:
            valid = rendered_depth > 0
        else:
            valid = torch.ones_like(rendered_depth, dtype=torch.bool)

        if valid.sum() < 100:
            return None

        return (
            self.config.pseudo_depth_loss_weight
            * pearson_depth_loss(
                rendered_depth[valid], estimated_depth[valid]
            )
        )

    # ── Convenience / metrics ──────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Add FSGS-specific metrics for logging."""
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["num_gaussians"] = torch.tensor(
            self.num_points, dtype=torch.float32
        )
        return metrics


# ═══════════════════════════════════════════════════════════════════
# Quaternion Utilities
# ═══════════════════════════════════════════════════════════════════

def _rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    R = R.float()
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z])
    return q / q.norm()


def _quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.zeros(3, 3)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R