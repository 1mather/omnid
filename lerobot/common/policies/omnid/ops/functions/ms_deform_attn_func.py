# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        # print(im2col_step)  # 源码默认64
        # print(value.shape)  # 必须满足(B*n_obs_steps mod im2col_step == 0 && B*n_obs_steps*S mod im2col_step == 0)
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
    #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
    # torch.Size([256, 1006, 4, 3, 8])
    '''
    value: 输入特征，形状为 (N_, S_, M_, D_)
    value_spatial_shapes: 特征图的空间形状
    sampling_locations: 采样位置，形状为 (N_, Lq_, M_, P_, 2)
    attention_weights: 注意力权重，形状为 (N_, Lq_, M_, L_)
    
    N_: batch size
    S_: 特征图的空间维度数（例如，一个图像可能有多个空间特征图）
    M_: 通道数（例如，深度特征图的通道数）
    D_: 每个通道的特征维度
    Lq_: 查询的数量（即要采样的位置数量）
    L_: 注意力分配给的特征数量
    P_: 额外的维度，通常与采样的空间位置相关
    '''
    N_, S_, M_, D_ = value.shape  # (256, 1287, 4, 32)  1287=972+252+63
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # (256, 1006, 4, 3, 8, 2)
    # tensor([[27, 36],
    #     [14, 18],
    #     [ 7,  9]], device='cuda:0')  972 252 63
    # (256, 972, 4, 32) (256, 252, 4, 32) (256, 63, 4, 32)
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # (256, 1006, 4, 3, 8, 2)
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)  # (1024, 32, 27, 36)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)  # (1024, 1006, 8, 2)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # 传入的attention_weights:(256, 1006, 4, 1, 8)有问题,倒数第二维度应该是3！！！ (256,1006,4,3,8) 256,4,1,1006,3,8 
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
