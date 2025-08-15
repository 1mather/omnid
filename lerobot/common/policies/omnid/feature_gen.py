
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from lerobot.common.policies.omnid.ops.modules import MSDeformAttn, MSDeformAttn3D
import  lerobot.common.policies.omnid.utils as utils
import lerobot.common.policies.omnid.utils.vox
import lerobot.common.policies.omnid.utils.geom
import lerobot.common.policies.omnid.utils.misc
import lerobot.common.policies.omnid.utils.basic
import lerobot.common.policies.omnid.utils.add_noise
# 手眼标定得到的是从camera到base的变换
from torchvision.models.resnet import resnet18
from PIL import Image
import os



def save_reference_points_XYZ(reference_points_cam, images, H, W,X=16,Y=16,Z=16, save_dir="debug_vis", prefix="ref_points_xyz"):
    """
    Draw reference points on the original camera images and save the result.

    Args:
        reference_points_cam (tensor): [S, B, Z*X, Y, 2]
        images (tensor or np.array): [B, S, 3, H, W] or list of PIL images
        H, W: image height and width
    """
    import matplotlib.pyplot as plt
    index = 1
    while True:
            dir_name = f"{index}"
            dir_path = os.path.join(save_dir, dir_name)
            if not os.path.exists(dir_path):
                save_dir = dir_path
                break
            index += 1
    # save_dir = save_dir +str(index)
    os.makedirs(save_dir, exist_ok=True)
    S, B, ZX, Y, _ = reference_points_cam.shape

    for s in range(S):
        for b in range(1):
                # 获取原始图像
                if isinstance(images, torch.Tensor):
                    img = images[b, s].detach().cpu().permute(1, 2, 0).numpy()  # [3, H, W] -> [H, W, 3]
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, np.ndarray):
                    img = images[b, s]
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, list):
                    img = np.array(images[b][s])
                else:
                    raise TypeError("Unsupported image type")
                ref_points = reference_points_cam.view(S,B,Z*X*Y,2)[s, b,:,:].detach().cpu().numpy()

                # 绘图
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img)
                plt.savefig(os.path.join(save_dir, f"{prefix}_cam{s}_batch{b}_slice{1}_image.png"))
                ax.scatter(ref_points[:, 0]*W, ref_points[:, 1]*H, s=0.5, c='lime', alpha=0.7)
                ax.axis("off")

                # 保存
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(save_dir, f"{prefix}_cam{s}_batch{b}_slice{1}.png"))
                plt.close()
def save_reference_points_XY(reference_points_cam, images, H, W,X=16,Y=16,Z=16, save_dir="debug_vis", prefix="ref_points_xy"):
    """
    Draw reference points on the original camera images and save the result.

    Args:
        reference_points_cam (tensor): [S, B, Z*X, Y, 2]
        images (tensor or np.array): [B, S, 3, H, W] or list of PIL images
        H, W: image height and width
    """
    os.makedirs(save_dir, exist_ok=True)
    S, B, ZX, Y, _ = reference_points_cam.shape

    for s in range(S):
        for b in range(4):
            for z in range(1):
                # 获取原始图像
                if isinstance(images, torch.Tensor):
                    img = images[b, s].detach().cpu().permute(1, 2, 0).numpy()  # [3, H, W] -> [H, W, 3]
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, np.ndarray):
                    img = images[b, s]
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, list):
                    img = np.array(images[b][s])
                else:
                    raise TypeError("Unsupported image type")
                ref_points = reference_points_cam.view(S,B,Z,X*Y,2)[s, b,z,:,:].detach().cpu().numpy()

                # 绘图
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img)
                ax.scatter(ref_points[:, 0]*W, ref_points[:, 1]*H, s=0.5, c='lime', alpha=0.7)
                ax.axis("off")

                # 保存
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(save_dir, f"{prefix}_cam{s}_batch{b}_slice{z}.png"))
                plt.close()

def save_reference_bev_projection_(
    reference_points_cam, images, H, W,X=16,Y=16,Z=16, save_dir="debug_vis", prefix="bev_color"
):
    """
    将原始图像投影到 BEV 平面并保存 BEV 多视角融合图像。

    Args:
        reference_points_cam: [S, B, Z*X, Y, 2] -> 归一化图像坐标 (x,y)
        images: [B, S, 3, H, W]
        H, W: 原图尺寸
    """
    os.makedirs(save_dir, exist_ok=True)
    S, B, ZX, Y, _ = reference_points_cam.shape
    # Z = X = int(np.sqrt(ZX))

    for b in range(B):
        bev_img = torch.zeros(3, Z, X)  # BEV 图像，三通道（RGB）

        for s in range(S):
            ref_points = reference_points_cam[s, b]  # [ZX, Y, 2]
            ref_points = ref_points.view(Z, X, Y, 2).detach()  # [Z, X, Y, 2]
            ref_points_2d = ref_points.mean(dim=2)  # 在 Y 上平均，形状 [Z, X, 2]

            # 反归一化到图像像素坐标
            uv = ref_points_2d.clone()
            uv[..., 0] *= W
            uv[..., 1] *= H

            # 获取图像并做 grid_sample 采样
            img = images[b, s].unsqueeze(0)  # [1, 3, H, W]
            uv_norm = ref_points_2d.clone()
            uv_norm[..., 0] = uv_norm[..., 0] / (W - 1) * 2 - 1  # -> [-1, 1]
            uv_norm[..., 1] = uv_norm[..., 1] / (H - 1) * 2 - 1

            grid = uv_norm.unsqueeze(0)  # [1, Z, X, 2]
            sampled = F.grid_sample(img, grid, mode='bilinear', align_corners=True)  # [1, 3, Z, X]

            bev_img += sampled[0].cpu()  # 累加多视角图像

        bev_img = torch.clamp(bev_img / S, 0, 1)  # 平均并限制范围
        bev_np = bev_img.permute(1, 2, 0).cpu().numpy()  # [Z, X, 3]
        save_path = os.path.join(save_dir, f"{prefix}_batch{b}.png")
        plt.imsave(save_path, bev_np)
        # plt.figure(figsize=(6, 6))
        # # plt.imshow(bev_np)
        # plt.title("Projected to BEV from Multi-view")
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f"{prefix}_batch{b}.png"))
        # plt.close()

def save_reference_points_(reference_points_cam, images, H, W, save_dir="debug_vis", prefix="ref_points"):
    """
    Draw reference points on the original camera images and save the result.

    Args:
        reference_points_cam (tensor): [S, B, Z*X, Y, 2]
        images (tensor or np.array): [B, S, 3, H, W] or list of PIL images
        H, W: image height and width
    """
    os.makedirs(save_dir, exist_ok=True)
    S, B, ZX, Y, _ = reference_points_cam.shape

    for s in range(S):
        for b in range(1):
            for y in range(1):
                # 获取原始图像
                if isinstance(images, torch.Tensor):
                    img = images[b, s].detach().cpu().permute(1, 2, 0).numpy()  # [3, H, W] -> [H, W, 3]
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, np.ndarray):
                    img = images[b, s]
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                elif isinstance(images, list):
                    img = np.array(images[b][s])
                else:
                    raise TypeError("Unsupported image type")
                # img = np.flipud(img)
                # img = np.fliplr(img)
                ref_points = reference_points_cam[s, b, :, y].detach().cpu().numpy()

                # 绘图
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img)
                ax.scatter(ref_points[:, 0]*W, ref_points[:, 1]*H, s=0.5, c='lime', alpha=0.7)
                ax.axis("off")

                # 保存
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(save_dir, f"{prefix}_cam{s}_batch{b}_slice{y}.png"))
                plt.close()

def save_reference_points(reference_points_cam, H, W, save_dir="debug_vis", prefix="ref_points"):
    os.makedirs(save_dir, exist_ok=True)
    S, B, ZX, Y, _ = reference_points_cam.shape

    for s in range(S):
        for b in range(B):
            fig, axs = plt.subplots(1, Y, figsize=(Y*2, 2))
            for y in range(Y):
                ref_points = reference_points_cam[s, b, :, y].detach().cpu().numpy()
                fig_ax = axs[y] if Y > 1 else axs
                fig_ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
                fig_ax.scatter(ref_points[:, 0]*W, ref_points[:, 1]*H, s=0.5, c='lime', alpha=0.7)
                fig_ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_cam{s}_batch{b}.png"))
            plt.close()
class Bevformernet(nn.Module):
    def __init__(self, Z, Y, X, 
                 rand_flip=False,
                 latent_dim=128,  # 128
                 encoder_type="res101",  # 自己写了res18，源码res101
                 pix_T_cams = None, 
                 cam0_T_camXs = None, 
                 vox_util = None,
                 resize_shape = None,
                 crop_shape = None,  # (216, 288)
                 random_crop = None,
                 imagenet_norm = None,
                 drop_cam = False,
                 add_noise = False,
                 avgpool_out =1,
                 pool_size = 1,
                 ):
        super(Bevformernet, self).__init__()
        assert (encoder_type in ["res101", "res50", "res18", "effb0", "effb4"])
        
        self.pix_T_cams = pix_T_cams
        self.cam0_T_camXs = cam0_T_camXs
        self.vox_util = vox_util

        self.Z, self.Y, self.X = Z, Y, X  
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.use_radar = False
        self.use_lidar = False
        self.pool_size = pool_size
        # preprocess
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        self.resize_shape = resize_shape,
        self.crop_shape = crop_shape,
        self.random_crop = random_crop,
        self.imagenet_norm = imagenet_norm
        self.drop_cam = drop_cam
        self.add_noise = add_noise
        self.avgpool_out = avgpool_out

        # Encoder
        
        self.feat2d_dim = feat2d_dim = latent_dim
        # if encoder_type == "res101":
        #     self.encoder = Encoder_res101(feat2d_dim)
        # elif encoder_type == "res50":
        #     self.encoder = Encoder_res50(feat2d_dim)
        # elif encoder_type == "res18":
        #     self.encoder = Encoder_res18(feat2d_dim)
        # elif encoder_type == "effb0":
        #     self.encoder = Encoder_eff(feat2d_dim, version='b0')
        # else:
        #     # effb4
        #     self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEVFormer self & cross attention layers
        self.bev_queries = nn.Parameter(0.1*torch.randn(latent_dim, Z, X)) # C, Z, X  128,64,64
        self.bev_queries_pos = nn.Parameter(0.1*torch.randn(latent_dim, Z, X)) # C, Z, X  位置嵌入
        num_layers = 6  
        self.num_layers = num_layers
        self.self_attn_layers = nn.ModuleList([
            VanillaSelfAttention(dim=latent_dim,X=X,Y=Y,Z=Z) for _ in range(num_layers)
        ]) # deformable self attention
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=latent_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        ffn_dim = 1028
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, latent_dim)) for _ in range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )
        # 使得raw_features适配diffusion的conditional features
        if self.pool_size ==1:
             self.avgpool =  nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.pool_size, stride=1, padding=0)
                                        , nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.pool_size, stride=1, padding=0),
                                        nn.Conv2d(in_channels=32, out_channels=self.avgpool_out, kernel_size=self.pool_size, stride=1, padding=0))
        elif self.pool_size==3:
            
            self.avgpool =  nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.pool_size, stride=1, padding=1)
                                        , nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.pool_size, stride=1, padding=1),
                                        nn.Conv2d(in_channels=32, out_channels=self.avgpool_out, kernel_size=self.pool_size, stride=1, padding=1))
        self.fc = nn.Identity()

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def visualize_cameras(self, rgb_camXs, n, save_path, save_name):
        """
        Save images from the nth batch for all cameras in a single image file using subplots.
        :param rgb_camXs: Tensor of shape (B, S, C, H, W)
        :param n: The batch index to save images from
        :param save_path: The file path to save the combined image
        """
        # 检查 save_path 是否存在，如果不存在则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 从 batch 中选择第 n 个样本，假设 S >= 5
        images = rgb_camXs[n]  # Shape: (S, C, H, W)
        num_cameras = images.shape[0]  # 获取 S 的数量，即相机的数量
        # 创建一个包含多个子图的画布
        fig, axes = plt.subplots(1, num_cameras, figsize=(15, 5))
        # 如果只有一个子图，axes 就是一个单一的对象，需要处理成列表
        if num_cameras == 1:
            axes = [axes]
        for i in range(num_cameras):  # 遍历所有相机
            img = images[i].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            axes[i].imshow(img)
            axes[i].axis("off")  # 去掉坐标轴
            axes[i].set_title(f"Camera {i+1}")
        plt.savefig(os.path.join(save_path, save_name))  # 保存图像到指定路径
        plt.close(fig)  # 关闭图像以释放内存
        print(f"Images for batch {n} saved to {os.path.join(save_path, save_name)}")

    def forward(self, rgb_camXs):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)  # 相机内参（intrinsics）矩阵
        cam0_T_camXs: (B,S,4,4)  # 从 cam0 到其他相机视角 camX 的相机外参（extrinsics）矩阵
        vox_util: vox util object
        '''
        # rgb_camXs = rgb_camXs[:,:,0,:,:,:]
        B,_,S, C, H, W = rgb_camXs.shape 
        rgb_camXs = rgb_camXs[:,-1,:,:,:,:]
        B0 = B*S 
        assert(C==3)
        
        cam0_T_camXs = self.cam0_T_camXs.repeat(B, 1, 1, 1).to(rgb_camXs.device)
        cam0_T_camXs = cam0_T_camXs.float()
        pix_T_cams = self.pix_T_cams.repeat(B, 1, 1, 1).to(rgb_camXs.device)
        if self.add_noise:
            noisy_transforms = []
            # 对每个变换矩阵应用噪声并计算位置和角度
            for b in range(B):
                for s in range(S):
                    T_noisy = utils.add_noise.add_noise_to_transform_matrix(cam0_T_camXs[b, s], angle_change_threshold=1.0)
                    noisy_transforms.append(T_noisy)
            # 将处理后的噪声变换矩阵堆叠回原来的形状
            cam0_T_camXs_noise = torch.stack(noisy_transforms).reshape(B, S, 4, 4)
            cam0_T_camXs = cam0_T_camXs_noise

        # 在训练过程中随机丢弃一个相机
        if self.drop_cam:
            # 为每个样本随机选择一个要丢弃的相机索引
            drop_indices = torch.randint(0, S, (B,), device=rgb_camXs.device)  # [B]
            mask = torch.ones(B, S, device=rgb_camXs.device)  # [B, S]
            mask[torch.arange(B), drop_indices] = 0  # 将选中的相机设为0
            mask_expanded = mask.view(B, S, 1, 1, 1)  # [B, S, 1, 1, 1]
            
            # 应用掩码到rgb_camXs: 将被丢弃的相机图像置零
            rgb_camXs = rgb_camXs * mask_expanded  # [B, S, C, H, W]
            
            # 创建一个零矩阵用于被丢弃的相机参数
            zero_matrix = torch.zeros(4, 4, device=rgb_camXs.device)
            zero_matrix = zero_matrix.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)  # [B, S, 4, 4]
            
            # 应用掩码到pix_T_cams和cam0_T_camXs: 将被丢弃的相机参数设为零
            pix_T_cams = pix_T_cams * mask.view(B, S, 1, 1) + zero_matrix * (1 - mask.view(B, S, 1, 1))
            cam0_T_camXs = cam0_T_camXs * mask.view(B, S, 1, 1) + zero_matrix * (1 - mask.view(B, S, 1, 1))
        
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        
        # save_path = "/root/lei.mao/code/OmniD/outputs/debug"
        # save_name = "input_camera_images_5cam_drop1.png"
        # self.visualize_cameras(rgb_camXs, 0, save_path = save_path, save_name = save_name)  # 可视化输入图像,用于调试

        rgb_camXs_ = __p(rgb_camXs)  # (64,2,3,240,320)->(128,3,240,320)
        pix_T_cams_ = __p(pix_T_cams)  # (64,2,4,4)->(128,4,4)
        cam0_T_camXs_ = __p(cam0_T_camXs)  # (64,2,4,4)->(128,4,4)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)  # (128,4,4)

        # rgb encoder
        # configure resize
        input_shape = C, H, W
        this_resizer = nn.Identity()
        if self.resize_shape[0] is not None:
            (h, w) = self.resize_shape[0]
            # (h, w) = self.resize_shape
            this_resizer = torchvision.transforms.Resize(size=(h,w))
            input_shape = (C,h,w)

        # configure randomizer
        this_randomizer = nn.Identity()
        if self.crop_shape[0] is not None:
            h, w = self.crop_shape[0]
            if self.random_crop:
                this_randomizer = CropRandomizer(
                    input_shape=input_shape,
                    crop_height=h,
                    crop_width=w,
                    num_crops=1,
                    pos_enc=False
                )
            else:
                this_normalizer = torchvision.transforms.CenterCrop(
                    size=(h,w)
                )
        # configure normalizer
        this_normalizer = nn.Identity()

        device = rgb_camXs_.device
        # rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        # rgb_camXs_ = rgb_camXs_.contiguous()
        # rgb_camXs_ = this_transform(rgb_camXs_)
        feat_camXs_8_, feat_camXs_16_, feat_camXs_32_ = self.encoder2(rgb_camXs_)
        # feat_camXs_8_, feat_camXs_16_, feat_camXs_32_ = self.encoder(rgb_camXs_)  # (128,128,30,40)1/8 (128,128,15,20)1/16  (128,128,8,10)1/32
        if self.rand_flip:
            feat_camXs_8_[self.rgb_flip_index] = torch.flip(feat_camXs_8_[self.rgb_flip_index], [-1])
            feat_camXs_16_[self.rgb_flip_index] = torch.flip(feat_camXs_16_[self.rgb_flip_index], [-1])
            feat_camXs_32_[self.rgb_flip_index] = torch.flip(feat_camXs_32_[self.rgb_flip_index], [-1])
        feat_camXs_8 = __u(feat_camXs_8_)  # (128,128,30,40)->(64,2,128,30,40)  (128, 2, 128, 27, 36)
        feat_camXs_16 = __u(feat_camXs_16_)  # (128,128,15,20)->(64,2,128,15,20)  (128, 2, 128, 14, 18)
        feat_camXs_32 = __u(feat_camXs_32_)  # (128,128,8,10)->(64,2,128,8,10)  (128, 2, 128, 7, 9)
        # feat_camXs = __u(feat_camXs_) # (B, S, C, Hf, Wf)

        Z, Y, X = self.Z, self.Y, self.X

        # compute the image locations (no flipping for now)
        xyz_mem_ = utils.basic.gridcloud3d(B0, Z, Y, X, norm=False, device=rgb_camXs.device) # B0, Z*Y*X, 3
        xyz_cam0_ = self.vox_util.Mem2Ref(xyz_mem_, Z, Y, X, assert_cube=False)  # mem -> ref  B0, Z*Y*X, 3
        xyz_camXs_ = utils.geom.apply_4x4(camXs_T_cam0_, xyz_cam0_)  # ref -> cam  B0, Z*Y*X, 3
        xy_camXs_ = utils.geom.camera2pixels(xyz_camXs_, pix_T_cams_) # B0, N, 2
        xy_camXs = __u(xy_camXs_) # B, S, N, 2, where N=Z*Y*X
        reference_points_cam = xy_camXs_.reshape(B, S, Z, Y, X, 2).permute(1, 0, 2, 4, 3, 5).reshape(S, B, Z*X, Y, 2)  # (2, 64, 4096, 32, 2) 
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(W)
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)
        bev_mask = ((reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)).squeeze(-1)

        # self & cross attentions  bev_queries是随机生成的nn.Parameter,(C,Z,X)->(1,C,Z,X)->(B,C,Z,X)->(B,C,Z*X)->(B,Z*X,C) 128,64,64->1,128,64,64->64,128,64,64->64,128,4096
        bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.latent_dim, -1).permute(0,2,1) # B, Z*X, C  (64,4096,128)
        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.latent_dim, -1).permute(0,2,1) # B, Z*X, C
        spatial_shapes = bev_queries.new_zeros([3, 2])  # 新张量spatial_shapes与 bev_queries在相同的设备上，并且具有相同的数据类型
        bev_keys = []

        _, _, _, H8, W8 = feat_camXs_8.shape  # (72, 5, 128, 45, 80)
        spatial_shapes[0, 0] = H8  # 30
        spatial_shapes[0, 1] = W8  # 40  (64,2,128,30,40)->(64,2,128,1200)->(2,1200,64,128)
        bev_keys.append(feat_camXs_8.reshape(B, S, self.latent_dim, H8*W8).permute(1, 3, 0, 2))  # (2,1200,64,128)

        _, _, _, H16, W16 = feat_camXs_16.shape  # 23,40
        spatial_shapes[1, 0] = H16  # 15
        spatial_shapes[1, 1] = W16  # 20
        bev_keys.append(feat_camXs_16.reshape(B, S, self.latent_dim, H16*W16).permute(1, 3, 0, 2))  # (2, 300, 64, 128)

        _, _, _, H32, W32 = feat_camXs_32.shape  # 12,20
        spatial_shapes[2, 0] = H32  # 8
        spatial_shapes[2, 1] = W32  # 10
        bev_keys.append(feat_camXs_32.reshape(B, S, self.latent_dim, H32*W32).permute(1, 3, 0, 2))  # (5, 1580, 48, 128)  (5, 4760, 48, 128)
        # 972,252,63
        bev_keys = torch.cat(bev_keys, dim=1).contiguous() # S, M_all, B, C  (5, 1580, 48, 128)  1200 + 300 + 80
        spatial_shapes = spatial_shapes.long()
        level_start_index = torch.cat((spatial_shapes.new_zeros(  # (0,1200,1500)  360,640分辨率:(0, 45*80=3600, 3600+23*40=4520)
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        for i in range(self.num_layers):
            # self attention within the features (B, N, C)
            bev_queries = self.self_attn_layers[i](bev_queries, bev_queries_pos.clone())  # (64,4096,128)

            # normalize (B, N, C)
            bev_queries = self.norm1_layers[i](bev_queries)
            # save_reference_points_XYZ(reference_points_cam,rgb_camXs,H=480,W=480,X=X,Y=Y,Z=Z,save_dir="/root/lei.mao/code/OmniD/outputs/debug/test")
            bev_queries = self.cross_attn_layers[i](bev_queries, 
                bev_keys[:, level_start_index[2]:], bev_keys[:, level_start_index[2]:], 
                query_pos=bev_queries_pos,
                reference_points_cam = reference_points_cam.detach(),
                spatial_shapes = spatial_shapes[2:3].detach(), 
                bev_mask = bev_mask.detach(),
                level_start_index = level_start_index[:1].detach())
            
            # normalize (B, N, C)
            bev_queries = self.norm2_layers[i](bev_queries)

            # feedforward layer (B, N, C)
            bev_queries = bev_queries.clone() + self.ffn_layers[i](bev_queries)

            # normalize (B, N, C)
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.latent_dim, self.Z, self.X)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_bev[self.bev_flip1_index] = torch.flip(feat_bev[self.bev_flip1_index], [-1])
            feat_bev[self.bev_flip2_index] = torch.flip(feat_bev[self.bev_flip2_index], [-3])

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        raw_e_flatten = self.avgpool(raw_e)
        # raw_e_flatten = self.avgpool(raw_e).squeeze(0) # (1, 128, 64, 64)->(1, 128)
        return raw_e_flatten  # (1, 128) (1, 128, 64, 64) (1, 1, 64, 64) (1, 1, 64, 64) (1, 2, 64, 64)


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.depth_layer2 = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.depth_layer3 = nn.Conv2d(1024, self.C, kernel_size=1, padding=0)
        self.depth_layer4 = nn.Conv2d(2048, self.C, kernel_size=1, padding=0)

    def forward(self, x):
        outs = []
        x = self.backbone(x) # 1/8
        outs.append(self.depth_layer2(x))
        x = self.layer3(x) # 1/16
        outs.append(self.depth_layer3(x))
        x = self.layer4(x) # 1/32
        outs.append(self.depth_layer4(x))

        return tuple(outs)


class Encoder_res18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet18(pretrained=True)
        
        # 去掉最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(resnet.children())[:-4]) 
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 根据 ResNet18 通道数定义深度层
        self.depth_layer2 = nn.Conv2d(128, self.C, kernel_size=1, padding=0)  # ResNet18的第2层输出通道为128
        self.depth_layer3 = nn.Conv2d(256, self.C, kernel_size=1, padding=0)  # ResNet18的第3层输出通道为256
        self.depth_layer4 = nn.Conv2d(512, self.C, kernel_size=1, padding=0)  # ResNet18的第4层输出通道为512

    def forward(self, x):
        outs = []
        x = self.backbone(x)  # 1/8 下采样
        outs.append(self.depth_layer2(x))
        
        x = self.layer3(x)  # 1/16 下采样
        outs.append(self.depth_layer3(x))
        
        x = self.layer4(x)  # 1/32 下采样
        outs.append(self.depth_layer4(x))

        return tuple(outs)

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x
    
    

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x
    

class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.1,X=16,Y=16,Z=16):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim 
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8, im2col_step = 16 * 5 * 2)
        self.output_proj = nn.Linear(dim, dim)
        self.X = X
        self.Y = Y
        self.Z=  Z
        
    def forward(self, query, query_pos=None):
        '''
        query: (B, N, C)  # 批次大小 B, 序列长度(或查询个数)N,以及每个查询的特征维度 C
        '''
        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        device = query.device
        
         # 需要与模型初始化的时候一致
        Z, X = self.Z, self.X
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z-0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X-0.5, X, dtype=torch.float, device=query.device)
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)  # 归一化的坐标(1, 256, 2)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, N, 1, 2)

        B, N, C = query.shape
        input_spatial_shapes = query.new_zeros([1,2]).long()
        # input_spatial_shapes[:] = 200
        input_spatial_shapes[0, 0] = Z  # 设置高度
        input_spatial_shapes[0, 1] = X  # 设置宽度

        input_level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query, reference_points, query.clone(), 
            input_spatial_shapes.detach(), input_level_start_index.detach())

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual  # 输出shape和输入一致

class SpatialCrossAttention(nn.Module):
    # From https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim=128, dropout=0.1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=32 * 4, im2col_step = 16 * 5 * 2)  # num_points是总的采样点数，分配到每个num_Z_anchors上(即BEV空间的不同高度)，必须满足num_points>=Y;源码num_levels=1,num_points=8,没有设置im2col_step
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None, level_start_index=None):
        '''
        query: (B, N, C)  torch.Size([64, 4096, 128])
        key: (S, M, B, C)  torch.Size([2, 1580, 64, 128]) key和value都是bev_key, 即选择使用的features
        reference_points_cam: (S, B, N, D, 2), in 0-1  torch.Size([2, 64, 4096, 32, 2])
        bev_mask: (S. B, N, D)  torch.Size([2, 64, 4096, 32]) N = X*Z, D=Y
        '''
        inp_residual = query.clone()
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)  # 32,就是Y
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)  # 每个 mask_per_img（即每个相机）中符合条件的行索引
        max_len = max([len(each) for each in indexes])
        # # 看一下哪些有效点
        # print("N:{}".format(N))
        # for i in range(len(indexes)):
        #     print(len(indexes[i]))
        #     print("index_query_img{}:{}".format(i, indexes[i]))

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)  # (2, 1580, 64, 128)->(64,2,1580,128)->(128,1580,128)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        queries = self.deformable_attention(query=queries_rebatch.view(B*S, max_len, self.dim),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(B*S, max_len, D, 2),
            spatial_shapes=spatial_shapes.to(query.device),
            level_start_index=level_start_index.to(query.device)).view(B, S, max_len, self.dim)  # torch.Size([64, 2, 1008, 128])

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                x1 = slots[j, index_query_per_img]
                x2 = queries[j, i, :len(index_query_per_img)]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
    

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import numpy as np
import torch
import torch.nn.functional as F


class Vox_util(object):
    def __init__(self, Z, Y, X, scene_centroid, bounds, pad=None, assert_cube=False):
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        B, D = list(scene_centroid.shape)
        self.Z, self.Y, self.X = Z, Y, X

        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX-self.XMIN)/float(X)
        self.default_vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        self.default_vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Z))

    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert(C==3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        xyz = utils.geom.apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube, device=xyz_mem.device)
        xyz_ref = utils.geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):  # 求步距
        vox_size_X = (self.XMAX-self.XMIN)/float(X)  # 每个voxel在X、Y、Z方向上的大小，(10,-10)/64=0.3125
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)  # (10,-10)/32=0.625
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)  # (1,-1)/64=0.3125

        if assert_cube:  # 断言变换是基于一个立方体（所有维度的voxel大小应该相同）
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert(np.isclose(vox_size_X, vox_size_Y))
            assert(np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = utils.geom.eye_4x4(B, device=device)
        center_T_ref[:,0,3] = -self.XMIN-vox_size_X/2.0  # 设置平移量，使内存坐标系的原点与参考坐标系中的左下角对齐
        center_T_ref[:,1,3] = -self.YMIN-vox_size_Y/2.0
        center_T_ref[:,2,3] = -self.ZMIN-vox_size_Z/2.0

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = utils.geom.eye_4x4(B, device=device)
        mem_T_center[:,0,0] = 1./vox_size_X
        mem_T_center[:,1,1] = 1./vox_size_Y
        mem_T_center[:,2,2] = 1./vox_size_Z
        mem_T_ref = utils.basic.matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(X-0.5)).byte()
        y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(Y-0.5)).byte()
        z_valid = ((z-padding)>-0.5).byte() & ((z+padding)<float(Z-0.5)).byte()
        nonzero = (~(z==0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return vox

    def voxelize_xyz_and_feats(self, xyz_ref, feats, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        B2, N2, D2 = list(feats.shape)
        assert(D==3)
        assert(B==B2)
        assert(N==N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        feats = self.get_feat_occupancy(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return feats

    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert(C==3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B*Z*Y*X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        # B x 1 x Z x Y x X
        return voxels

    def get_feat_occupancy(self, xyz, feat, Z, Y, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # feat is B x N x D
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert(C==3)
        assert(B==B2)
        assert(N==N2)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask # B, N
        y = y*mask
        z = z*mask
        feat = feat*mask.unsqueeze(-1) # B, N, D

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        # permute point orders
        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)
        feat = feat.view(B*N, -1)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        feat_voxels = torch.zeros((B*Z*Y*X, D2), device=xyz.device).float()
        feat_voxels[vox_inds.long()] = feat
        # zero out the singularity
        feat_voxels[base.long()] = 0.0
        feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
        # B x C x Z x Y x X
        return feat_voxels

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False, xyz_camA=None):
        # rgb_camB is B x C x H x W
        # pixB_T_camA is B x 4 x 4

        # rgb lives in B pixel coords
        # we want everything in A memory coords

        # this puts each C-dim pixel in the rgb_camB
        # along a ray in the voxelgrid
        B, C, H, W = list(rgb_camB.shape)

        if xyz_camA is None:
            xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)
            xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)
        z = xyz_camB[:,:,2]

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        # z = xyz_pixB[:,:,2]
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
        y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
        z_valid = (z>0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        if (0):
            # handwritten version
            values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
            for b in list(range(B)):
                values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
        else:
            # native pytorch version
            y_pixB, x_pixB = utils.basic.normalize_grid2d(y, x, H, W)
            # since we want a 3d output, we need 5d tensors
            z_pixB = torch.zeros_like(x)
            xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
            rgb_camB = rgb_camB.unsqueeze(2)
            xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
            values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values

    def warp_tiled_to_mem(self, rgb_tileB, pixB_T_camA, camB_T_camA, Z, Y, X, DMIN, DMAX, assert_cube=False):
        # rgb_tileB is B,C,D,H,W
        # pixB_T_camA is B,4,4
        # camB_T_camA is B,4,4

        # rgb_tileB lives in B pixel coords but it has been tiled across the Z dimension
        # we want everything in A memory coords

        # this resamples the so that each C-dim pixel in rgb_tilB
        # is put into its correct place in the voxelgrid
        # (using the pinhole camera model)
        
        B, C, D, H, W = list(rgb_tileB.shape)

        xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_camA, xyz_camA)
        z_camB = xyz_camB[:,:,2]

        # rgb_tileB has depth=DMIN in tile 0, and depth=DMAX in tile D-1
        z_tileB = (D-1.0) * (z_camB-float(DMIN)) / float(DMAX-DMIN)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        # z = xyz_pixB[:,:,2]
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
        y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
        z_valid = (z_camB>0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()

        z_tileB, y_pixB, x_pixB = utils.basic.normalize_grid3d(z_tileB, y, x, D, H, W)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_tileB], axis=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_tileB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Z, Y, X))
        values = values * valid_mem
        return values
    

    def apply_mem_T_ref_to_lrtlist(self, lrtlist_cam, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in cam coordinates
        # transforms them into mem coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_cam.shape)
        assert(C==19)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=lrtlist_cam.device)

    def xyz2circles(self, xyz, radius, Z, Y, X, soft=True, already_mem=True, also_offset=False, grid=None):
        # xyz is B x N x 3
        # radius is B x N or broadcastably so
        # output is B x N x Z x Y x X
        B, N, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        if grid is None:
            grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X, stack=False, norm=False, device=xyz.device)
            # note the default stack is on -1
            grid = torch.stack([grid_x, grid_y, grid_z], dim=1)
            # this is B x 3 x Z x Y x X
            
        xyz = xyz.reshape(B, N, 3, 1, 1, 1)
        grid = grid.reshape(B, 1, 3, Z, Y, X)
        # this is B x N x Z x Y x X

        # round the xyzs, so that at least one value matches the grid perfectly,
        # and we get a value of 1 there (since exp(0)==1)
        xyz = xyz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        if soft:
            off = grid - xyz # B,N,3,Z,Y,X
            # interpret radius as sigma
            dist_grid = torch.sum(off**2, dim=2, keepdim=False)
            # this is B x N x Z x Y x X
            if torch.is_tensor(radius):
                radius = radius.reshape(B, N, 1, 1, 1)
            mask = torch.exp(-dist_grid/(2*radius*radius))
            # zero out near zero
            mask[mask < 0.001] = 0.0
            # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            # h[h < np.finfo(h.dtype).eps * h.max()] = 0
            # return h
            if also_offset:
                return mask, off
            else:
                return mask
        else:
            assert(False) # something is wrong with this. come back later to debug

            dist_grid = torch.norm(grid - xyz, dim=2, keepdim=False)
            # this is 0 at/near the xyz, and increases by 1 for each voxel away

            radius = radius.reshape(B, N, 1, 1, 1)

            within_radius_mask = (dist_grid < radius).float()
            within_radius_mask = torch.sum(within_radius_mask, dim=1, keepdim=True).clamp(0, 1)
            return within_radius_mask

    def xyz2circles_bev(self, xyz, radius, Z, Y, X, already_mem=True, also_offset=False):
        # xyz is B x N x 3
        # radius is B x N or broadcastably so
        # output is B x N x Z x Y x X
        B, N, D = list(xyz.shape)
        assert(D==3)
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X)

        xz = torch.stack([xyz[:,:,0], xyz[:,:,2]], dim=2)

        grid_z, grid_x = utils.basic.meshgrid2d(B, Z, X, stack=False, norm=False, device=xyz.device)
        # note the default stack is on -1
        grid = torch.stack([grid_x, grid_z], dim=1)
        # this is B x 2 x Z x X

        xz = xz.reshape(B, N, 2, 1, 1)
        grid = grid.reshape(B, 1, 2, Z, X)
        # these are ready to broadcast to B x N x Z x X

        # round the points, so that at least one value matches the grid perfectly,
        # and we get a value of 1 there (since exp(0)==1)
        xz = xz.round()

        if torch.is_tensor(radius):
            radius = radius.clamp(min=0.01)

        off = grid - xz # B,N,2,Z,X
        # interpret radius as sigma
        dist_grid = torch.sum(off**2, dim=2, keepdim=False)
        # this is B x N x Z x X
        if torch.is_tensor(radius):
            radius = radius.reshape(B, N, 1, 1, 1)
        mask = torch.exp(-dist_grid/(2*radius*radius))
        # zero out near zero
        mask[mask < 0.001] = 0.0
        
        # add a Y dim
        mask = mask.unsqueeze(-2)
        off = off.unsqueeze(-2)
        # # B,N,2,Z,1,X
        
        if also_offset:
            return mask, off
        else:
            return mask
        



class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)  # x 是低分辨率特征图，需要上采样到与 x_skip 相同的分辨率。
        return x + x_skip
