````markdown
# Environment Setup & Training Guide

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/1mather/omnid.git
cd lerobot
````

2. **Create a Python 3.10 virtual environment (recommended: Miniconda)**

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

3. **Install ffmpeg if not already present**

```bash
conda install ffmpeg
```

4. **Install 🤗 LeRobot**

```bash
pip install --no-binary=av -e .
```

5. **Reinstall Torch with CUDA 11.8 support**

```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

6. **Install Deformable-DETR dependencies**
   Follow the instructions at: [https://github.com/fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)

   > ⚠️ Ensure that your `cudatoolkit` version matches your local CUDA version.

---

## Training Instructions

1. **Prepare a training config file**
   Use `./config/omnid/training_200k_01234_pretrain.json` as a reference and create a new config file in the corresponding directory, e.g. `config/act/training_test.json`.

2. **Edit the config parameters**

   * `"steps"`: number of training steps
   * `"type"`: task type
   * `"input_features"`: select the cameras to use for training/evaluation

3. **Start training**
   **General command:**

   ```bash
   python3 lerobot/scripts/train.py --config_path=/path/to/config.json
   ```

### Example Tasks

**Coffee**

```bash
python3 lerobot/scripts/train.py \
    --config_path=/root/workspace/OmniD/config/vqbet/get_coffee_random_pos_100/training_200k_01234_pretrain.json
```

**Set Study Table**

```bash
python3 lerobot/scripts/train.py \
    --config_path=/media/jerry/code/OmniD/config/vqbet/get_coffee_random_pos_100/training_200k_01234_pretrain.json
```



### Dataset
https://huggingface.co/datasets/jiaruiguan/omnid_vlabench_dataset_v_0

You can evaluate your models and create custom datasets using vla_evaluation ：https://github.com/1mather/vla_evaluation.git. This repository provides six benchmark tasks for experimentation.

### Models
https://huggingface.co/jiaruiguan/Omnid



