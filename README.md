# Project Context: VQ-VAE for Human-to-Robot Motion Retargeting

**Update Date:** 2025-12-17
**Current Phase:** Data Preprocessing Completed / Ready for Model Training
**Robot Target:** Unitree G1 (29 DoF)

## 1. Project Overview
本项目旨在利用 VQ-VAE (Vector Quantized Variational Autoencoder) 技术，将人类动作数据对齐并重定向到人形机器人 (Unitree G1) 上。项目核心要求是实现多种 VQ-VAE 变体 (Standard, EMA, RVQ) 进行消融实验，并产出可视化训练曲线与指标对比。

## 2. Directory Structure & File Manifest
(纯文本结构，无代码框)

project_root/
├── data/
│   ├── raw/                       # 原始下载及解压数据
│   │   ├── seulzx/smplx_datasets/ # .tar.bz2 文件存放处
│   │   └── unzipped/              # 解压后的根目录
│   └── processed/                 # 预处理后的 .npy 文件
│       ├── g1_train.npy           # Shape: (N, 64, 29)
│       ├── mean.npy               # 用于归一化
│       └── std.npy                # 用于归一化
├── models/
│   ├── __init__.py
│   └── vqvae.py                   # 核心模型 (含 Standard, EMA, RVQ 实现)
├── scripts/
│   ├── download_data.py           # ModelScope 数据下载脚本
│   ├── inspect_npz.py             # 数据结构探查脚本
│   └── process_data.py            # 数据清洗与切片脚本
├── utils/
│   ├── __init__.py
│   └── visualizer.py              # 绘图工具 (待完善)
└── project_context.md             # 项目上下文文档

## 3. Dataset Specifications (I/O Protocols)

### 3.1 Raw Data Source
- **Source:** ModelScope (`seulzx/smplx_datasets`)
- **Structure Path:** `data/raw/unzipped/extended_datasets/lafan1_dataset/g1`
- **File Format:** `.npz` (Numpy Archive)
- **Key Discovery:**
  在 `inspect_npz.py` 探查中发现数据集已包含重定向好的机器人数据。
  - **Selected Key:** `joint_pos` (机器人关节角度)
  - **Robot:** Unitree G1
  - **DoF (Degrees of Freedom):** 29
  - **Ignored Keys:** `fps`, `joint_vel`, `body_pos_w`, `smplx_pose_body` (原始SMPL数据), `robot_keypoints_trans` 等。

### 3.2 Preprocessed Data (.npy)
预处理脚本 `process_data.py` 将原始的不定长序列转换为定长的时间窗口切片，用于 VQ-VAE 训练。

- **Input Pipeline:**
  - **Window Size:** 64 Frames
  - **Stride:** 32 Frames (Train set overlap), 64 Frames (Test set non-overlap)
- **Output Files:**
  - `g1_train.npy`:
    - **Shape:** `(13726, 64, 29)`
    - **Dim 0:** Samples (切片数量)
    - **Dim 1:** Time (时间窗口)
    - **Dim 2:** Channels/DoF (关节自由度)
  - `mean.npy`: 全局均值 (Shape: 29,)
  - `std.npy`: 全局标准差 (Shape: 29,)
- **Normalization Strategy:** Standard Scaling `(x - mean) / std`.

## 4. Development Log & Execution History

### 4.1 Environment Setup
安装了 ModelScope, Torch, Pinocchio (未实际使用但已装), Datasets 库。
> pip install modelscope torch pandas pinocchio datasets

### 4.2 Data Acquisition
运行 `scripts/download_data.py`。
- **Issue:** 初始报错 `Repo not exists`。
- **Fix:** 在 `snapshot_download` 中添加参数 `repo_type='dataset'`。
- **Result:** 成功下载 `extended_datasets.tar.bz2` (1.2GB) 和 `lafan1_smplx_datasets.tar.bz2` (834M)。

### 4.3 Data Inspection & Logic Pivot
**Critical Step:** 运行 `scripts/inspect_npz.py` 检查数据内部。
- **Input:** 检查了 `.../g1/train/dance1_subject2.npz`
- **Output Keys:** `['fps', 'joint_pos', 'joint_vel', ..., 'smplx_pose_body', ...]`
- **Decision:** 发现数据集中已包含名为 `g1` (Unitree G1) 的机器人 `joint_pos` 数据。
- **Action:** **取消** 原定的 Inverse Kinematics (IK) 和 Retargeting 开发计划，直接使用现成的 `joint_pos` 进行 VQ-VAE 训练。

### 4.4 Data Preprocessing
运行 `scripts/process_data.py`。
- **Version 1 Issue:** 简单提取并将数据存为 `object` 数组，无法被 DataLoader 直接使用。
- **Version 2 Fix:** 引入滑动窗口 (Sliding Window) 机制。
- **Execution Log (2025-12-17):**
  > 在 ./data/raw/unzipped/extended_datasets/lafan1_dataset/g1 下发现了子目录: ['train']
  > 正在处理 train 集 (来源: ['train'])...
  > Loading train: 100%|...| 40/40 [00:00<00:00, 400.56it/s]
  > --> train 完成。Shape: (13726, 64, 29) (Samples, Time, Dof)
  > 警告: 没有找到 test 相关的文件夹，跳过。
  > 正在计算归一化统计量 (Mean/Std)...
  > Mean/Std 已保存。数据预处理全部完成！

## 5. Model Architecture Specifications (Current Plan)

### 5.1 VQ-VAE Variants
代码位于 `models/vqvae.py`，支持通过参数 `method` 切换：
1.  **Standard:** Euclidean distance + Gradient Descent.
2.  **EMA:** Exponential Moving Average update (无 Codebook 梯度).
3.  **RVQ (Residual VQ):** 多层量化器 (Multi-layer Quantizer) 逼近残差。

### 5.2 Interface
> class MotionVQVAE(nn.Module):
>     def __init__(self, input_dim=29, hidden_dim=64, method='standard', n_layers=4):
>         ...
>     def forward(self, x):
>         # Input x: [Batch, DoF, Time]
>         pass

## 6. Next Steps
1.  **Split Data:** 由于原始数据只有 `train` 文件夹，需要在 DataSet Loader 层面手动划分 Train/Val 集合 (e.g., 90/10 split)。
2.  **Train:** 编写 `scripts/train_ablation.py`，加载 `.npy` 数据，实例化三种模型进行训练。
3.  **Visual:** 完善 `utils/visualizer.py` 绘制 Loss 曲线和 Codebook Perplexity。



