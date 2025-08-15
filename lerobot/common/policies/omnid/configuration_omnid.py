#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.types import NormalizationMode
import numpy as np
import torch
import  lerobot.common.policies.omnid.utils as utils
import lerobot.common.policies.omnid.utils.vox 
import json
def get_camera_dict(camera_para_path: str) -> dict: 
        try:
            with open(camera_para_path, 'r', encoding='utf-8') as f:
                camera_dict = json.load(f)
                
            return camera_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"相机参数文件未找到: {camera_para_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON 解析错误: {e.msg}", e.doc, e.pos)
        
@PreTrainedConfig.register_subclass("omnid")
@dataclass
class OmnidConfig(PreTrainedConfig):
    # Inputs / output structure.
    
    # input_image_features = {}
    # for name in ['observation.image_0','observation.image_1','observation.image_2','observation.image_3','observation.image_4']:
    #     input_image_features[name] =  PolicyFeature(type=FeatureType.VISUAL, shape=(3,480,480))
    bev_encoder: str = "Bevformernet"
    n_obs_steps: int = 2
    avgpool_out: int =1
    horizon: int = 16
    n_action_steps: int = 8
    camera_para_path: str = "./config/camera_para.json"
    # camera_para_path: str = "./config/camera_para_old.json"
    camera_para = get_camera_dict(camera_para_path)
    T_base2cam_transformed= []
    intrinsics_matrices = []
    for key in camera_para.keys():
        T_base2cam_transformed.append(np.linalg.inv(camera_para[key]["extrinsic"]["matrix"]))
        intrinsics_matrices.append(camera_para[key]["intrinsic"]["matrix"])
    T_base2cam_transformed = torch.tensor(T_base2cam_transformed)
    intrinsics_matrices = torch.tensor(intrinsics_matrices)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    # crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    rand_flip: bool = False
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Z, Y, X = 16, 8, 16
    x_size,y_size,z_size = 0.128/2, 0.128/2, 0.128/2  # y

    scene_centroid_x,scene_centroid_y,scene_centroid_z = -0.1,0,0
    scene_centroid_py = np.array([scene_centroid_x,
                            scene_centroid_y,
                            scene_centroid_z]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid_py).float()
    
    XMIN, XMAX =  0, 1.024+0.128 # x 朝前
    # ZMIN, ZMAX = -1.024/4, 1.024/4
    YMIN, YMAX = -1.024/2-0.128, 1.024/2+0.128  # Y 朝左
    # YMIN, YMAX = -1.024/2, 1.024/2  # Y 朝左, X,Y,Z 需要时 8 的倍数
    # ZMIN, ZMAX = -0.512 , 1.536 # z 朝上
    ZMIN, ZMAX = 0 , 1.024/2+0.256 # z 朝上
    #  = -0.512/4 , 1.536/4
    X : int = 64
    # X = int((XMAX - XMIN)/x_size)
    Y : int = 16
    # Y = int((YMAX - YMIN)/y_size)
    Z  :int =  64
    # Z = int((ZMAX - ZMIN)/z_size)
    
    # XMIN, XMAX = -1.024/2 - 0.1024, 1.024/2 + 0.1024 * 1.5  # +X:左
    # ZMIN, ZMAX = -1.024/2 + 0.2048, 1.024 * 0.4  # +Z:前
    # YMIN, YMAX =  0, 1.024/3  # +Y:上
    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX) 
    pad = None  # 可选填充
    assert_cube = False 
    
    T_base2cam_transformed[:, :3, 3] /= 1000
    
    
    
    vox_util = utils.vox.Vox_util(Z, Y, X, scene_centroid, bounds, pad, assert_cube)
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
        
    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
    # @property
    # def image_features(self) -> dict[str, PolicyFeature]:
    #     return {key: ft for key, ft in self.input_image_features.items() if ft.type is FeatureType.VISUAL}

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
