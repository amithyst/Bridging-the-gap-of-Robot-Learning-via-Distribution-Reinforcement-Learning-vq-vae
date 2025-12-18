# Project Context: VQ-VAE for Human-to-Robot Motion Retargeting

**Update Date:** 2025-12-17
**Robot Target:** Unitree G1 (29 DoF)

## 1. Project Overview
本项目旨在利用 VQ-VAE (Vector Quantized Variational Autoencoder) 技术，将人类动作数据对齐并重定向到人形机器人 (Unitree G1) 上。项目核心要求是实现多种 VQ-VAE 变体 (Standard, EMA, RVQ) 进行消融实验，并产出可视化训练曲线与指标对比。


### Directory Structure & File Manifest

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
│   └── vqvae.py                   # 核心模型 (支持 Simple/ResNet + EMA/RVQ/FSQ/LFQ)
├── scripts/
│   ├── download_data.py           # ModelScope 数据下载脚本
│   ├── inspect_npz.py             # 数据结构探查脚本
│   ├── process_data.py            # 数据清洗与切片脚本
│   ├── train_ablation.py          # 训练主脚本 (多 Seed, 支持 Jerk/DCR 指标)
│   ├── plot_results.py            # 绘图脚本 (含方差带, Loss/PPL/Jerk/DCR/Radar)
│   └── export_latex_table.py      # (新增) 读取日志并生成 LaTeX 格式的 Mean±Std 表格
├── utils/
│   ├── __init__.py
│   └── alignment.py
└── project_context.md             # 项目上下文文档
```

## 2. Quick Start & Environment Setup

### 2.1 Dependencies Installation
运行以下命令安装项目所需依赖：
```bash
pip install modelscope torch pandas pinocchio datasets matplotlib
```
或者
```bash
pip install -r requirements.txt
```
### 2.2 Standard Workflow
按顺序执行以下指令即可完成从数据准备到结果导出的全流程：

1. **Data Download:**
```bash
   python scripts/download_data.py
```

2. **Unzip Data:**
   (需手动解压至指定目录)
```bash
   mkdir -p ./data/raw/unzipped
   tar -xjf ./data/raw/seulzx/smplx_datasets/lafan1_smplx_datasets.tar.bz2 -C ./data/raw/unzipped
   tar -xjf ./data/raw/seulzx/smplx_datasets/extended_datasets.tar.bz2 -C ./data/raw/unzipped
```

3. **Data Inspection (Optional):**
```bash
   python scripts/inspect_npz.py
```

4. **Preprocessing:**
```bash
   python scripts/process_data.py
```

5. **Training (Multi-Seed Ablation):**
```bash
   python scripts/train_ablation.py
```

6. **Result Export & Visualization:**
```bash
   python scripts/export_latex_table.py
   python scripts/plot_results.py
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

### 5.1 Architecture Variants (Updated)
代码位于 models/vqvae.py，支持 arch (骨干) 和 method (量化) 的组合，新增 SOTA 方法：

1.  **Backbones (`arch`):**
    -   **Simple:** 2层 Conv1d (Baseline)。
    -   **ResNet:** 引入 ResBlock1D，高频特征保持 (Proposed)。
    -   **Transformer:** (可选) Conv 降采样 + Transformer Encoder。
2.  **Quantizers (`method`):**
    -   **EMA:** 指数移动平均更新，无 Codebook 梯度 (Standard)。
    -   **RVQ (Residual VQ):** 多层残差量化，解决 Codebook Collapse (Advanced)。
    -   **FSQ (Finite Scalar Quantization):** (SOTA 2024) 无显式 Codebook，通过标量投影与取整实现极高利用率。
    -   **LFQ (Lookup-Free Quantization):** 基于符号判断 (Sign) 的二值化量化，适合低码率场景。

## 6. Next Steps & Workflow (Updated)
(请替换原有的 Section 6，反映多种子训练和新指标流程)

1.  **Multi-Seed Training:**
    运行 scripts/train_ablation.py。
    -   **配置:** 自动遍历 Baseline, Proposed (ResNet), Advanced (RVQ), SOTA (FSQ, LFQ)。
    -   **机制:** 每个实验运行多个随机种子 (SEEDS=[42, 1024, ...]) 以评估稳定性。
    -   **新增指标:**
        -   **Jerk Loss:** 加加速度 (平滑度指标，越小越好)。
        -   **Dead Code Ratio (DCR):** 死码率 (Codebook 利用率指标，越小越好)。
    -   **输出:** 也就是 logs/log_{config}_seed_{seed}.json。

2.  **Visualization (Variance Band):**
    运行 scripts/plot_results.py。
    -   读取多 Seed 日志，自动对齐 Epoch。
    -   绘制 **Mean ± Std** 阴影曲线图：
        -   Reconstruction Loss
        -   Velocity Loss & Jerk Loss (Smoothness)
        -   Perplexity & Dead Code Ratio (Utilization)
        -   Train Loss (Convergence Check)
    -   生成五维雷达图 (Radar Chart): 综合对比各模型性能。

3.  **Paper Resource Generation:**
    运行 scripts/export_latex_table.py。
    -   自动聚合所有 Seed 的最后 5 Epoch 数据。
    -   计算 Mean ± Std。
    -   直接输出 LaTeX `booktabs` 格式表格代码，用于论文/报告粘贴。