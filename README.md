# Project Context: VQ-VAE for Human-to-Robot Motion Retargeting

**Update Date:** 2025-12-17
**Current Phase:** Data Preprocessing Completed / Ready for Model Training
**Robot Target:** Unitree G1 (29 DoF)

## 1. Project Overview
本项目旨在利用 VQ-VAE (Vector Quantized Variational Autoencoder) 技术，将人类动作数据对齐并重定向到人形机器人 (Unitree G1) 上。项目核心要求是实现多种 VQ-VAE 变体 (Standard, EMA, RVQ) 进行消融实验，并产出可视化训练曲线与指标对比。

## 2. Directory Structure & File Manifest
(纯文本结构，无代码框)
```bash
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
│   └── vqvae.py                   # 核心模型 (新增 arch 参数，支持 Simple/ResNet/Transformer)
├── scripts/
│   ├── download_data.py           # ModelScope 数据下载脚本
│   ├── inspect_npz.py             # 数据结构探查脚本
│   ├── process_data.py            # 数据清洗与切片脚本
│   ├── train_ablation.py          # 训练主脚本 (保存日志至 json，不含绘图)
│   └── plot_results.py            # 独立绘图脚本 (读取 json 生成 Loss/PPL/Radar 图)
├── utils/
│   ├── __init__.py
│   └── alignment.py              # 不知道干啥的
└── project_context.md             # 项目上下文文档
```

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

## 5. Model Architecture Specifications

### 5.1 Architecture Variants (New)
代码位于 `models/vqvae.py`，经过重构后支持 `arch` (骨干网络) 和 `method` (量化方法) 的组合配置：

1.  **Backbones (`arch`):**
    -   **Simple:** 2层 Conv1d (Baseline)。
    -   **ResNet:** 引入 `ResBlock1D`，支持更深的网络和高频特征保持 (Proposed)。
    -   **Transformer:** Conv 降采样 + Transformer Encoder 处理全局时序 (Ablation)。
2.  **Quantizers (`method`):**
    -   **EMA:** 指数移动平均更新，无 Codebook 梯度 (Standard approach)。
    -   **RVQ (Residual VQ):** 多层残差量化，解决 Codebook Collapse 问题 (Advanced)。

### 5.2 Training Interface
> class MotionVQVAE(nn.Module):
>     def __init__(self, input_dim=29, hidden_dim=64, arch='resnet', method='ema', ...):
>         ...

**Ablation Study Configs:**
1.  **Baseline:** Arch=`simple`, Method=`ema`
2.  **Proposed:** Arch=`resnet`, Method=`ema` (主力模型)
3.  **Advanced:** Arch=`resnet`, Method=`rvq` (高性能探索)

### 修改 3：更新下一步计划 (Section 6)
**操作说明：** 请用以下内容替换整个 `Section 6`，反映最新的“训练-绘图”解耦流程。

## 6. Next Steps & Workflow

1.  **Training:**
    运行 `scripts/train_ablation.py`。
    -   自动加载 `.npy` 数据。
    -   执行 Baseline/Proposed/Advanced 三组实验。
    -   训练过程中计算 **Reconstruction Loss**, **Velocity Loss** (平滑度), **Perplexity**。
    -   结果保存为 `results/training_log.json`，模型保存为 `checkpoints/*.pth`。

2.  **Visualization:**
    运行 `scripts/plot_results.py`。
    -   读取 JSON 日志。
    -   生成 Loss 对比曲线、PPL 变化曲线。
    -   生成最终性能对比雷达图 (Radar Chart)。

3.  **Evaluation:**
    分析图表，确认 ResNet+RVQ 是否解决了 Codebook Collapse 问题，并选择最佳模型部署到 Unitree G1 机器人上。



