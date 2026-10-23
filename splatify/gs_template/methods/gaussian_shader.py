"""
GaussianShader — 3D Gaussian Splatting with Shading Functions
================================================================

Paper: GaussianShader (CVPR 2024)
Complexity: ★★★★★ (new params + custom rendering + custom losses)

This implementation adapts the paper to the gs_template / nerfstudio plugin
architecture by:
    - reusing Splatfacto's geometry / densification core
    - adding per-Gaussian shading attributes
    - using a differentiable trainable cubemap for direct specular lighting
    - rendering RGB + normal + expected depth in one rasterization pass
    - enforcing the paper's normal consistency and opacity sparsity losses

Important note:
    The paper summary does not fully specify the original CUDA-side lighting
    implementation or exact optimizer hyperparameters for all new parameters.
    This file therefore implements a faithful PyTorch approximation that fits
    the provided template, with the key paper components preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.model_components.lib_bilagrid import total_variation_loss as bilateral_tv_loss

from gs_template.losses import (
    normal_consistency_loss,
    normal_residual_reg_loss,
    sparse_opacity_loss,
)
from gs_template.model import GSTemplateModel, GSTemplateModelConfig
from gs_template.rendering import crop_custom_param, gather_base_params, prepare_camera, rasterize_gaussians
from gs_template.shading import (
    cubemap_to_strip,
    depth_to_normal_map,
    eval_sh_rgb,
    gamma_tone_map,
    reflect,
    sample_prefiltered_envmap,
    select_view_aligned_normals,
    shortest_axis_directions,
)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GaussianShaderModelConfig(GSTemplateModelConfig):
    """GaussianShader configuration."""

    _target: Type = field(default_factory=lambda: GaussianShaderModel)

    # The paper uses 3rd-order SH residual colors.
    sh_degree: int = 3

    # Need depth during training for the normal-from-depth loss.
    output_depth_during_training: bool = True

    # Lighting
    envmap_resolution: int = 64
    envmap_num_levels: int = 6
    envmap_init: float = 0.8
    gamma: float = 2.2

    # Shading attribute initialization
    specular_tint_init: float = 0.1
    roughness_init: float = 0.2

    # Residual color schedule (paper says introduced later, exact step unspecified)
    residual_start_step: int = 15000
    residual_ramp_steps: int = 2000

    # Paper loss weights
    lambda_normal: float = 0.01
    lambda_sparse: float = 0.001
    lambda_residual_reg: float = 0.001


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class GaussianShaderModel(GSTemplateModel):
    """GaussianShader implemented as a nerfstudio Splatfacto extension."""

    config: GaussianShaderModelConfig

    # ── 1. populate_modules ────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()
        device = self.means.device
        dtype = self.means.dtype

        # Initialize diffuse color from the seed RGB used by vanilla splatfacto.
        base_rgb = self.colors.detach().clamp(1e-4, 1 - 1e-4)

        # Repurpose the base SH color coefficients as residual-color SH.
        # The paper introduces this term later in training, so start from zero.
        with torch.no_grad():
            self.gauss_params["features_dc"].zero_()
            self.gauss_params["features_rest"].zero_()

        self.gauss_params["diffuse_color"] = Parameter(torch.logit(base_rgb))
        self.gauss_params["specular_tint"] = Parameter(
            torch.logit(torch.full((self.num_points, 3), self.config.specular_tint_init, device=device, dtype=dtype))
        )
        self.gauss_params["roughness"] = Parameter(
            torch.logit(torch.full((self.num_points, 1), self.config.roughness_init, device=device, dtype=dtype))
        )
        self.gauss_params["normal_delta_1"] = Parameter(torch.zeros(self.num_points, 3, device=device, dtype=dtype))
        self.gauss_params["normal_delta_2"] = Parameter(torch.zeros(self.num_points, 3, device=device, dtype=dtype))

        # Global trainable environment map (6 faces, 64x64, RGB), initialized to 0.8.
        self.environment_map = Parameter(
            torch.logit(
                torch.full(
                    (6, self.config.envmap_resolution, self.config.envmap_resolution, 3),
                    self.config.envmap_init,
                    device=device,
                    dtype=dtype,
                )
            )
        )

    # ── helpers ────────────────────────────────────────────────────

    @property
    def residual_sh_coeffs(self) -> Tensor:
        return torch.cat([self.features_dc[:, None, :], self.features_rest], dim=1)

    def _residual_weight(self) -> float:
        if self.step < self.config.residual_start_step:
            return 0.0
        if self.config.residual_ramp_steps <= 0:
            return 1.0
        t = (self.step - self.config.residual_start_step) / float(self.config.residual_ramp_steps)
        return float(max(0.0, min(1.0, t)))

    def _camera_pose(self, camera: Cameras) -> Tensor:
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time during training"
            return self.camera_optimizer.apply_to_camera(camera)
        return camera.camera_to_worlds

    def _assemble_outputs(
        self,
        render: Tensor,
        alpha: Tensor,
        H: int,
        W: int,
    ) -> Dict[str, Tensor]:
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        normal = render[:, ..., 3:6].squeeze(0)
        normal = F.normalize(normal, dim=-1)
        valid = alpha.squeeze(0) > 1e-4
        normal = torch.where(valid.expand_as(normal), normal, torch.zeros_like(normal))

        depth = render[:, ..., -1:]
        depth = torch.where(alpha > 0, depth, depth.detach().max()).squeeze(0)

        if self.config.use_bilateral_grid and self.training:
            if self.camera is not None and self.camera.metadata is not None and "cam_idx" in self.camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, self.camera.metadata["cam_idx"], H, W)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "normal": normal,
            "accumulation": alpha.squeeze(0),
            "background": background,
            "envmap_strip": cubemap_to_strip(torch.sigmoid(self.environment_map)).detach(),
        }

    # ── 2. get_outputs (custom rendering) ──────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        if not isinstance(camera, Cameras):
            return {}

        self.camera = camera
        optimized_camera_to_world = self._camera_pose(camera)
        camera_position = optimized_camera_to_world[0, :3, 3]

        cam, crop_ids = prepare_camera(self, camera)
        if cam is None:
            return crop_ids

        means, quats, scales, opacities, _ = gather_base_params(self, crop_ids)
        diffuse_color = torch.sigmoid(crop_custom_param(self.gauss_params["diffuse_color"], crop_ids))
        specular_tint = torch.sigmoid(crop_custom_param(self.gauss_params["specular_tint"], crop_ids))
        roughness = torch.sigmoid(crop_custom_param(self.gauss_params["roughness"], crop_ids))
        normal_delta_1 = crop_custom_param(self.gauss_params["normal_delta_1"], crop_ids)
        normal_delta_2 = crop_custom_param(self.gauss_params["normal_delta_2"], crop_ids)
        residual_coeffs = crop_custom_param(self.residual_sh_coeffs, crop_ids)

        viewdirs = F.normalize(camera_position[None, :] - means, dim=-1)
        shortest_axis = shortest_axis_directions(scales, quats)
        normals_world = select_view_aligned_normals(shortest_axis, normal_delta_1, normal_delta_2, viewdirs)
        reflection = reflect(viewdirs, normals_world)

        envmap = torch.sigmoid(self.environment_map)
        specular_light = sample_prefiltered_envmap(
            envmap=envmap,
            directions=reflection,
            roughness=roughness,
            levels=self.config.envmap_num_levels,
        )

        residual_weight = self._residual_weight()
        residual_color = residual_weight * eval_sh_rgb(residual_coeffs, viewdirs, degree=self.config.sh_degree)

        shaded_rgb = gamma_tone_map(
            diffuse_color + specular_tint * specular_light + residual_color,
            gamma=self.config.gamma,
        )

        # Render normals in the same camera convention as gsplat depth.
        normal_cam = torch.matmul(cam["viewmat"][0, :3, :3], normals_world.T).T
        normal_cam = F.normalize(normal_cam, dim=-1)

        fused = torch.cat([shaded_rgb, normal_cam], dim=-1)
        render, alpha, self.info = rasterize_gaussians(
            self,
            means,
            quats,
            scales,
            opacities,
            fused,
            cam["viewmat"],
            cam["K"],
            cam["W"],
            cam["H"],
            sh_degree=None,
            render_mode="RGB+ED",
            absgrad=self.strategy.absgrad if hasattr(self.strategy, "absgrad") else False,
        )

        # Store 2D means / radii for densification, matching Splatfacto.
        if self.training and self.info["means2d"].requires_grad:
            self.info["means2d"].retain_grad()
        self.xys = self.info["means2d"]
        self.radii = self.info["radii"]

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params,
                self.optimizers,
                self.strategy_state,
                self.step,
                self.info,
            )

        return self._assemble_outputs(render, alpha, cam["H"], cam["W"])

    # ── custom paper-faithful loss ─────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # Paper uses an L2 color reconstruction loss.
        loss_dict = {"color_loss": F.mse_loss(pred_img, gt_img)}
        loss_dict.update(self.compute_custom_losses(outputs, batch, metrics_dict))

        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * bilateral_tv_loss(self.bil_grids.grids)
        return loss_dict

    # ── 3. compute_custom_losses ───────────────────────────────────

    def compute_custom_losses(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}

        # Normal-geometry consistency from the paper.
        if outputs.get("depth") is not None and outputs.get("normal") is not None:
            depth_normal = depth_to_normal_map(outputs["depth"])
            valid_mask = outputs["accumulation"][..., 0] > 1e-4
            if valid_mask.any():
                losses["normal_loss"] = self.config.lambda_normal * normal_consistency_loss(
                    outputs["normal"],
                    depth_normal,
                    valid_mask=valid_mask,
                )

        # Sparse opacity loss.
        losses["sparse_loss"] = self.config.lambda_sparse * sparse_opacity_loss(torch.sigmoid(self.opacities))

        # Residual-normal regularization.
        losses["normal_reg"] = self.config.lambda_residual_reg * normal_residual_reg_loss(
            self.gauss_params["normal_delta_1"],
            self.gauss_params["normal_delta_2"],
        )

        return losses

    # ── 4. split_gaussians ─────────────────────────────────────────

    def split_gaussians(self, split_mask, samps):
        out = super().split_gaussians(split_mask, samps)
        for name in ["diffuse_color", "specular_tint", "roughness", "normal_delta_1", "normal_delta_2"]:
            out[name] = self.gauss_params[name][split_mask].repeat(samps, 1)
        return out

    # ── 5. get_gaussian_param_groups ───────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {name: [self.gauss_params[name]] for name in self.gauss_params.keys()}

    # ── 6. get_param_groups ────────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["lighting"] = [self.environment_map]
        return groups

    # ── 7. get_outputs_for_camera ──────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, Tensor]:
        outs = super().get_outputs_for_camera(camera, obb_box)
        if "normal" in outs:
            outs["normal_vis"] = 0.5 * (outs["normal"] + 1.0)
        if "envmap_strip" in outs:
            outs["lighting"] = outs["envmap_strip"]
        return outs

    # ── 8. load_state_dict ─────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]

        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            new_shape = (newp,) + param.shape[1:]
            self.gauss_params[name] = Parameter(torch.zeros(new_shape, device=self.device))

        if "environment_map" in dict and tuple(self.environment_map.shape) != tuple(dict["environment_map"].shape):
            self.environment_map = Parameter(torch.zeros_like(dict["environment_map"], device=self.device))

        super().load_state_dict(dict, **kwargs)

    # ── 9. get_metrics_dict ────────────────────────────────────────

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["avg_roughness"] = torch.sigmoid(self.gauss_params["roughness"]).mean()
        metrics["residual_weight"] = torch.tensor(self._residual_weight(), device=self.device)
        return metrics
