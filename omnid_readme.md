# 环境配置
1. 安装 lerobot环境
2. 重新安装 torch： 
```
pip install torch==2.6.0+cu118  torchvision==0.21.0+cu118  --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.21.0+cu118  --index-url https://download.pytorch.org/whl/cu118
```
3. 安装 DETR 环境
https://github.com/fundamentalvision/Deformable-DETR
需要注意cudatoolkit 版本要和cuda 版本匹配
# 训练
1. 参考./config/omnid/training_200k_01234_pretrain.json在对应目录下生成训练文件，如config/act/training_test.json
2. 修改config 中"steps"，“type”，“input_features”等参数，通过“input_features”来控制训练、评测时使用的相机。
3. 发起训练
```
python3 lerobot/scripts/train.py --config_path=/home/lei/code/lerobot/config/omnid/training_200k_01234_pretrain.json
```

coffee
```
python3 lerobot/scripts/train.py --config_path=/root/workspace/OmniD/config/vqbet/get_coffee_random_pos_100/training_200k_01234_pretrain.json 
```

set_study_table
```
python3 lerobot/scripts/train.py --config_path=/media/jerry/code/OmniD/config/vqbet/get_coffee_random_pos_100/training_200k_01234_pretrain.json 
```


ur_5e_singleArm-gripper-4cameras_2
```
python3 lerobot/scripts/train.py --config_path=/media/jerry/code/OmniD/config/dp/ur_5e_200/training_200k_01234_pretrain.json
```