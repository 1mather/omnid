import torch
import math
import numpy as np

def add_noise_to_transform_matrix(T_base2cam, position_noise_range=0.05, angle_noise_range=0.01, angle_change_threshold=1.0, device = "cuda"):
    """
    对变换矩阵T_base2cam的所有轴施加一致的噪声，并确保加噪后的角度变化不超过给定的阈值。
    
    :param T_base2cam: 变换矩阵 (4x4 tensor)
    :param position_noise_range: 位置噪声范围 (单位: 米)
    :param angle_noise_range: 角度噪声范围 (单位: 弧度)
    :param angle_change_threshold: 角度变化的阈值 (单位: 度)
    :return: 加噪后的变换矩阵
    """
    # 生成随机噪声
    position_noise = torch.FloatTensor(3).uniform_(-position_noise_range, position_noise_range)
    angle_noise = torch.FloatTensor(3).uniform_(-angle_noise_range, angle_noise_range)
    
    # 生成加噪后的变换矩阵
    T_noisy = T_base2cam.clone().cpu()
    
    # 添加位置噪声
    T_noisy[:3, 3] += position_noise  # 加噪到位置部分
    
    # 旋转噪声：绕X, Y, Z轴分别加入噪声
    rot_x = torch.tensor([
        [1, 0, 0],
        [0, math.cos(angle_noise[0]), -math.sin(angle_noise[0])],
        [0, math.sin(angle_noise[0]), math.cos(angle_noise[0])]
    ])
    rot_y = torch.tensor([
        [math.cos(angle_noise[1]), 0, math.sin(angle_noise[1])],
        [0, 1, 0],
        [-math.sin(angle_noise[1]), 0, math.cos(angle_noise[1])]
    ])
    rot_z = torch.tensor([
        [math.cos(angle_noise[2]), -math.sin(angle_noise[2]), 0],
        [math.sin(angle_noise[2]), math.cos(angle_noise[2]), 0],
        [0, 0, 1]
    ])
    
    # 总旋转矩阵为绕X、Y、Z轴的旋转矩阵的乘积
    rotation_matrix = torch.mm(torch.mm(rot_z, rot_y), rot_x)
    
    T_noisy[:3, :3] = torch.mm(T_noisy[:3, :3], rotation_matrix)  # 加噪到旋转部分

    # 提取加噪后的旋转矩阵和位置
    R_noisy = T_noisy[:3, :3].cpu().numpy()  # 旋转矩阵
    position_noisy = T_noisy[:3, 3].cpu().numpy()  # 位置
    
    # 计算加噪后的欧拉角
    yaw_rad_noisy, pitch_rad_noisy, roll_rad_noisy, yaw_deg_noisy, pitch_deg_noisy, roll_deg_noisy = rotation_matrix_to_euler_angles(R_noisy)
    
    # 提取原始的旋转矩阵和位置
    R_original = T_base2cam[:3, :3].cpu().numpy()  # 旋转矩阵
    position_original = T_base2cam[:3, 3].cpu().numpy()  # 位置
    
    # 计算原始欧拉角
    yaw_rad, pitch_rad, roll_rad, yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_angles(R_original)
    
    # 如果加噪后的角度变化大于阈值，则使用原始矩阵
    angle_diff = max(abs(yaw_deg_noisy - yaw_deg), abs(pitch_deg_noisy - pitch_deg), abs(roll_deg_noisy - roll_deg))
    
    if angle_diff > angle_change_threshold:
        # print(f"  Noisy angle change is too large ({angle_diff}°), using original transform.")
        T_noisy = T_base2cam.cpu()  # 使用原始矩阵
        T_noisy[:3, 3] += position_noise  # 仅仅进行位置加噪
    return T_noisy.to(device)


def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角（Yaw, Pitch, Roll）
    假设旋转矩阵为 3x3
    返回角度（度）和弧度（rad）
    """
    # 提取旋转矩阵的元素
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # 计算俯仰角（Pitch）
    pitch_rad = math.asin(-r31)
    pitch_deg = math.degrees(pitch_rad)  # 弧度转角度
    
    # 计算偏航角（Yaw）和滚转角（Roll）
    if abs(r31) != 1:
        yaw_rad = math.atan2(r21, r11)
        roll_rad = math.atan2(r32, r33)
        yaw_deg = math.degrees(yaw_rad)  # 弧度转角度
        roll_deg = math.degrees(roll_rad)  # 弧度转角度
    else:
        yaw_rad = 0  # 当 r31 == ±1 时，yaw 和 roll 不能确定
        roll_rad = math.atan2(r12, r13) if r31 == -1 else 0
        yaw_deg = 0
        roll_deg = math.degrees(roll_rad)  # 弧度转角度

    return yaw_rad, pitch_rad, roll_rad, yaw_deg, pitch_deg, roll_deg


if __name__ == "__main__":
    # 旋转部分提取出来
    T_base2cam_transformed = torch.tensor([[[ 5.9114e-02,  5.7252e-01, -8.1775e-01,  1.0261e+03],
            [-2.7772e-02, -8.1793e-01, -5.7465e-01,  5.9127e+02],
            [-9.9786e-01,  5.6681e-02, -3.2451e-02,  5.9830e+02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # 1 左
            [[ 6.4774e-01,  2.8021e-01, -7.0846e-01,  6.6006e+02],
            [-1.3446e-02, -9.2555e-01, -3.7838e-01,  4.9597e+02],
            [-7.6174e-01,  2.5461e-01, -5.9575e-01,  1.0897e+03],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # 2左上
            [[ 9.9885e-01,  2.4114e-02,  4.1406e-02, -3.5038e+01],
            [ 4.5388e-02, -7.5313e-01, -6.5631e-01,  7.7399e+02],
            [ 1.5357e-02,  6.5743e-01, -7.5336e-01,  1.3819e+03],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # 3前
            [[ 8.3849e-01, -2.8281e-01,  4.6579e-01, -6.6151e+02],
            [-6.4586e-02, -9.0033e-01, -4.3039e-01,  6.4252e+02],
            [ 5.4108e-01,  3.3080e-01, -7.7318e-01,  1.3967e+03],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # 4右上
            [[ 2.8836e-02, -3.4259e-01,  9.3904e-01, -9.8231e+02],
            [ 1.6485e-03, -9.3941e-01, -3.4278e-01,  5.4320e+02],
            [ 9.9958e-01,  1.1432e-02, -2.6524e-02,  5.4594e+02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],  # 5右
        device='cpu')
    # List to store noisy transformation matrices
    noisy_transforms = []

    # Apply noise to each transformation matrix and compute position and angles
    for i in range(T_base2cam_transformed.shape[0]):
        T_noisy = add_noise_to_transform_matrix(T_base2cam_transformed[i], angle_change_threshold=1.0)

        # Store the noisy transformation matrix
        noisy_transforms.append(T_noisy)

        # Output original transformation matrix and angles
        R_original = T_base2cam_transformed[i][:3, :3].cpu().numpy()  # Original rotation matrix
        position_original = T_base2cam_transformed[i][:3, 3].cpu().numpy()  # Original position
        yaw_rad_original, pitch_rad_original, roll_rad_original, yaw_deg_original, pitch_deg_original, roll_deg_original = rotation_matrix_to_euler_angles(R_original)

        print(f"Transform {i+1} (Original):")
        print(f"  Position = {position_original}")
        print(f"  Yaw = {yaw_deg_original}°, Pitch = {pitch_deg_original}°, Roll = {roll_deg_original}°")

        # Output noisy transformation matrix and angles
        R_noisy = T_noisy[:3, :3].cpu().numpy()  # Noisy rotation matrix
        position_noisy = T_noisy[:3, 3].cpu().numpy()  # Noisy position
        yaw_rad_noisy, pitch_rad_noisy, roll_rad_noisy, yaw_deg_noisy, pitch_deg_noisy, roll_deg_noisy = rotation_matrix_to_euler_angles(R_noisy)

        print(f"Transform {i+1} (Noisy):")
        print(f"  Position = {position_noisy}")
        print(f"  Yaw = {yaw_deg_noisy}°, Pitch = {pitch_deg_noisy}°, Roll = {roll_deg_noisy}°")

    # Return all noisy transformation matrices
    noisy_transforms_tensor = torch.stack(noisy_transforms)
    print("\nAll Noisy Transform Matrices:")
    print(noisy_transforms_tensor)