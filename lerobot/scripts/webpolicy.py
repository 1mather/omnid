import logging


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

from lerobot.common.policies.factory import get_policy_class

from contextlib import nullcontext
from dataclasses import asdict, field
from pprint import pformat
from termcolor import colored
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.eval import EvalPipelineConfig

import torch
from torch import Tensor


import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np

# 从 factory.py 获取的 ImageNet 统计量
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

def normalize_image_with_imagenet_stats(image):
    """
    使用 ImageNet 统计量对输入图像进行标准化
    
    参数:
        image: PIL图像、numpy数组或PyTorch张量
    
    返回:
        标准化后的PyTorch张量
    """
    # 定义转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为[0,1]范围的张量并调整通道顺序为 (C,H,W)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # RGB通道的均值
            std=[0.229, 0.224, 0.225]     # RGB通道的标准差
        )
    ])
    
    # 应用转换
    normalized_image = transform(image)
    return normalized_image

# 也可以使用PyTorch张量直接操作
def normalize_tensor_with_imagenet_stats(image_tensor):
    """
    使用 ImageNet 统计量对已经是PyTorch张量的图像进行标准化
    假设输入已经是 [0,1] 范围的 (C,H,W) 张量
    """
    # 将统计量转换为张量并确保形状正确以便广播
    mean = torch.tensor([0,0,0]).reshape(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([255,255,255]).reshape(3, 1, 1).to(image_tensor.device)
    
    # 应用标准化
    normalized_tensor = (image_tensor - mean) / std
    return normalized_tensor


class WebPolicyServer:
    def __init__(self, policy: VQBeTPolicy, device: str, host: str = "0.0.0.0", port: int = 8000):
        """初始化策略服务器"""
        self.policy = policy
        self.policy.reset()
        self.device = torch.device(device)
        self.host = host
        self.port = port
        self.server = None
        logging.info(f"PolicyServer initialized with device={device}, host={host}, port={port}")
        logging.info(f"Policy type: {type(policy).__name__}")
        
        # 记录模型输入输出特征
        logging.info(f"Policy input features: {policy.config.input_features}")
        logging.info(f"Policy output features: {policy.config.output_features}")

    def server_forever(self):
        asyncio.run(self.run())
    async def run(self):
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        while True:
            try:
                message=await websocket.recv()
                if isinstance(message,bytes) and message==b'reset':
                    print("reset")
                    self.policy.reset()
                    await websocket.send(b'reset_complete')
                    continue
                else:
                    obs= msgpack_numpy.unpackb(message)
                    print(f"Received observation: ")
                    for key,value in obs.items():
                        obs[key]=torch.from_numpy(value).to(self.device)
                        print(f"shape of {key}:{obs[key].shape}")
                #print(f"observation device: {obs['observation.image_0'].device}")
                for key,value in obs.items():
                    if key.startswith("observation.image"):
                        #obs[key]=normalize_tensor_with_imagenet_stats(value)
                        img = value


                        # 假设img形状为 (B, C, H, W) 或 (C, H, W)
                        import os
                        from torchvision.utils import save_image
                        save_dir = "omnid_images"
                        os.makedirs(save_dir, exist_ok=True)
                        # 只保存第一个batch
                        print(f"img shape: {img}")
                        img_to_save = img
                        save_path = os.path.join(save_dir, f"{key}.png")
                        print(f"Saving observation to {save_path}")
                        try:
                            save_image(img_to_save, save_path)
                        except Exception as e:
                            pass

                        obs[key]=normalize_tensor_with_imagenet_stats(value)
                        print(f"normalized {key}")
                    else:
                        print(f"not normalized {key}")
                action = self.policy.select_action(obs)

                print(f"action: {action.shape}")
                #time.sleep()

                packed_action = msgpack_numpy.packb(action.cpu().numpy(), use_bin_type=True)
                await websocket.send(packed_action)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise