# lerobot/scripts/serve_policy.py
import dataclasses
import enum
import logging
import socket
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import json
import numpy as np
import asyncio
import websockets
import time
import traceback
import msgpack_numpy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.common.policies.factory import get_policy_class

from contextlib import nullcontext
from dataclasses import asdict, field
from pprint import pformat
from termcolor import colored
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.eval import EvalPipelineConfig

from lerobot.scripts.webpolicy import WebPolicyServer


class EnvMode(enum.Enum):
    """支持的环境类型"""
    FRANKA = "franka"
    XARM = "xarm"
    UR5 = "ur5"


HOST = "127.0.0.1"
PORT = 8000



# 预设配置映射
PRESETS = {
    "franka_default": {
        "env": EnvMode.FRANKA,
        "policy": {
            "config_path": "configs/vqbet/franka_default.yaml",
            "model_path": "checkpoints/vqbet/franka_default",
            "model_type": "vqbet"
        }
    },
    "xarm_default": {
        "env": EnvMode.XARM,
        "policy": {
            "config_path": "configs/vqbet/xarm_default.yaml",
            "model_path": "checkpoints/vqbet/xarm_default",
            "model_type": "vqbet"
        }
    },
    "ur5_default": {
        "env": EnvMode.UR5,
        "policy": {
            "config_path": "configs/vqbet/ur5_default.yaml",
            "model_path": "checkpoints/vqbet/ur5_default",
            "model_type": "vqbet"
        }
    }
}


@parser.wrap()
def server_main(cfg: EvalPipelineConfig):
    """主函数"""
    # 设置日志
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    #env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    logging.info("Making policy.")
    dataset_metadata = LeRobotDatasetMetadata("/media/jerry/data/lerobot/set_study_table_simple_100")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta = dataset_metadata
    )
    policy.eval()
    # 确定设备
    device = cfg.policy.device
    # 创建上下文管理器
    ctx = torch.autocast(device_type=device if device != "cpu" else "cpu") if cfg.policy.use_amp else nullcontext()


    with torch.no_grad(), ctx:
        # 获取主机信息
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")
        
        # 创建并启动服务器
        server = WebPolicyServer(
            policy=policy,
            device=device,
            host=HOST,
            port=PORT

        )
        
        try:
            server.server_forever()
            
        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        except Exception as e:
            logging.error(f"Server error: {e}")
            import traceback
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    init_logging()
    server_main()

"""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一个 GPU

# 启动服务器
python lerobot/scripts/server.py --policy.path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model --policy.config_path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model/config.json



eval:
usage: server.py [-h] [--config_path str] [--env str] [--env.type {aloha,pusht,xarm}] [--env.task str] [--env.fps str]
                 [--env.features str] [--env.features_map str] [--env.episode_length str] [--env.obs_type str]
                 [--env.render_mode str] [--env.visualization_width str] [--env.visualization_height str] [--eval str]
                 [--eval.n_episodes str] [--eval.batch_size str] [--eval.use_async_envs str] [--policy str]
                 [--policy.type {act,diffusion,pi0,tdmpc,vqbet,pi0fast}]
                 [--policy.replace_final_stride_with_dilation str] [--policy.pre_norm str] [--policy.dim_model str]
                 [--policy.n_heads str] [--policy.dim_feedforward str] [--policy.feedforward_activation str]
                 [--policy.n_encoder_layers str] [--policy.n_decoder_layers str] [--policy.use_vae str]
                 [--policy.n_vae_encoder_layers str] [--policy.temporal_ensemble_coeff str] [--policy.kl_weight str]
                 [--policy.optimizer_lr_backbone str] [--policy.drop_n_last_frames str]
                 [--policy.use_separate_rgb_encoder_per_camera str] [--policy.down_dims str]
                 [--policy.kernel_size str] [--policy.n_groups str] [--policy.diffusion_step_embed_dim str]
                 [--policy.use_film_scale_modulation str] [--policy.noise_scheduler_type str]
                 [--policy.num_train_timesteps str] [--policy.beta_schedule str] [--policy.beta_start str]
                 [--policy.beta_end str] [--policy.prediction_type str] [--policy.clip_sample str]
                 [--policy.clip_sample_range str] [--policy.num_inference_steps str]
                 [--policy.do_mask_loss_for_padding str] [--policy.scheduler_name str] [--policy.num_steps str]
                 [--policy.attention_implementation str] [--policy.train_expert_only str]
                 [--policy.train_state_proj str] [--policy.n_action_repeats str] [--policy.horizon str]
                 [--policy.image_encoder_hidden_dim str] [--policy.state_encoder_hidden_dim str]
                 [--policy.latent_dim str] [--policy.q_ensemble_size str] [--policy.mlp_dim str]
                 [--policy.discount str] [--policy.use_mpc str] [--policy.cem_iterations str] [--policy.max_std str]
                 [--policy.min_std str] [--policy.n_gaussian_samples str] [--policy.n_pi_samples str]
                 [--policy.uncertainty_regularizer_coeff str] [--policy.n_elites str]
                 [--policy.elite_weighting_temperature str] [--policy.gaussian_mean_momentum str]
                 [--policy.max_random_shift_ratio str] [--policy.reward_coeff str] [--policy.expectile_weight str]
                 [--policy.value_coeff str] [--policy.consistency_coeff str] [--policy.advantage_scaling str]
                 [--policy.pi_coeff str] [--policy.temporal_decay_coeff str] [--policy.target_model_momentum str]
                 [--policy.n_action_pred_token str] [--policy.action_chunk_size str] [--policy.vision_backbone str]
                 [--policy.crop_shape str] [--policy.crop_is_random str] [--policy.pretrained_backbone_weights str]
                 [--policy.use_group_norm str] [--policy.spatial_softmax_num_keypoints str]
                 [--policy.n_vqvae_training_steps str] [--policy.vqvae_n_embed str] [--policy.vqvae_embedding_dim str]
                 [--policy.vqvae_enc_hidden_dim str] [--policy.gpt_block_size str] [--policy.gpt_input_dim str]
                 [--policy.gpt_output_dim str] [--policy.gpt_n_layer str] [--policy.gpt_n_head str]
                 [--policy.gpt_hidden_dim str] [--policy.dropout str] [--policy.mlp_hidden_dim str]
                 [--policy.offset_loss_weight str] [--policy.primary_code_loss_weight str]
                 [--policy.secondary_code_loss_weight str] [--policy.bet_softmax_temperature str]
                 [--policy.sequentially_select str] [--policy.optimizer_vqvae_lr str]
                 [--policy.optimizer_vqvae_weight_decay str] [--policy.input_camera str] [--policy.n_obs_steps str]
                 [--policy.normalization_mapping str] [--policy.input_features str] [--policy.output_features str]
                 [--policy.device str] [--policy.use_amp str] [--policy.chunk_size str] [--policy.n_action_steps str]
                 [--policy.max_state_dim str] [--policy.max_action_dim str] [--policy.resize_imgs_with_padding str]
                 [--policy.interpolate_like_pi str] [--policy.empty_cameras str] [--policy.adapt_to_pi_aloha str]
                 [--policy.use_delta_joint_actions_aloha str] [--policy.tokenizer_max_length str]
                 [--policy.proj_width str] [--policy.max_decoding_steps str] [--policy.fast_skip_tokens str]
                 [--policy.max_input_seq_len str] [--policy.use_cache str] [--policy.freeze_vision_encoder str]
                 [--policy.freeze_lm_head str] [--policy.optimizer_lr str] [--policy.optimizer_betas str]
                 [--policy.optimizer_eps str] [--policy.optimizer_weight_decay str]
                 [--policy.scheduler_warmup_steps str] [--policy.scheduler_decay_steps str]
                 [--policy.scheduler_decay_lr str] [--policy.checkpoint_path str] [--policy.padding_side str]
                 [--policy.precision str] [--policy.grad_clip_norm str] [--policy.relaxed_action_decoding str]
                 [--output_dir str] [--job_name str] [--seed str]


                 python server.py --policy.path=/mnt/data/310_jiarui/lerobot/outputs/train/2025-04-09/21-32-06_vqbet/checkpoints/last/pretrained_model

                 python server.py --policy.path=/root/workspace/OmniD/outputs/train/set_study_table_pos_100/1/checkpoints/120000/pretrained_model
"""