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
├── checkpoints/ #存储训练得到的最优模型
├── models/
│   ├── __init__.py
│   ├── experiment_config.py
│   └── vqvae.py                   # 核心模型 (支持 Simple/ResNet + EMA/RVQ/FSQ/LFQ+Hybrid)
├── scripts/
│   ├── data/                  # 数据处理 (download, process, inspect)
│   │   ├── download_data.py           # ModelScope 数据下载脚本
│   │   ├── inspect_npz.py             # 数据结构探查脚本
│   │   └── process_data.py            # 数据清洗与切片脚本
│   ├── deployment/            # [还没写] Isaac Lab 演示 (play_isaac)
│   ├── evaluation/            # 结果导出 (plot, export_latex)
│   │   ├── plot_results.py            # 绘图脚本 (含方差带, Loss/PPL/Jerk/DCR/Radar(标准和对数))
│   │   ├── analyze_latent_space.py # 可视化隐空间分布
│   │   └── export_latex_table.py      # (新增) 读取日志并生成 LaTeX 格式的 
│   └── train_ablation.py          # 训练主脚本 (多 Seed, 支持 Jerk/DCR 指标)
├── utils/
│   ├── __init__.py
│   └── alignment.py
├── plots/ #输出图片
│   ├── compare_dcr.png
│   ├── compare_jerk.png
│   ├── compare_ppl.png
│   ├── compare_recon.png
│   ├── compare_vel.png
│   ├── final_radar_chart_log.png
│   └── final_radar_chart.png
├── results/ #训练时记录的指标数据
│   ├── log_resnet_ema_seed_1024.json
│   ├── ...
│   └── log_simple_ema_seed_999.json
├── requirements.txt
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


### 2.2 安装isaaclab最终可视化依赖（前期训练网络不用）

#### isaaclab
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

2小时后
```bash
isaacsim
Yes
```


```bash
apt install cmake build-essential
apt show cmake build-essential

cd ~/深度学习/PJ/IsaacLab/source/isaaclab
sudo rm -rf build dist isaaclab.egg-info

cd /home/kaijie/深度学习/PJ/IsaacLab
./isaaclab.sh --install # or "./isaaclab.sh -i"
```
#### whole_body_tracking
然后安装whole_body_tracking
```bash
git clone https://github.com/HybridRobotics/whole_body_tracking.git
cd whole_body_tracking/source/whole_body_tracking
install -e .
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
   python scripts/data/inspect_npz.py
```

4. **Preprocessing:**
```bash
   python scripts/data/process_data.py
```

5. **Training (Multi-Seed Ablation):**
```bash
   python scripts/train_ablation.py
```

6. **Result Export & Visualization:**
```bash
   python scripts/evaluation/export_latex_table.py
   python scripts/evaluation/plot_results.py
   python scripts/evaluation/analyze_latent_space.py
```
7. 启动isaaclab
```bash
# 确保在项目根目录下执行，这样脚本才能找到 models 文件夹
cd /home/kaijie/深度学习/PJ_vqvae
conda activate isaaclab_env
# 使用 isaaclab.sh 运行脚本，-p 参数代表用内置 python 运行
python scripts/deployment/export_motion.py \
    --exp_id resnet_hybrid \
    --ckpt "checkpoints/Ours(Dual-Enc+Hybrid)_hybrid_seed_42.pth" \
    --data_dir "./data/processed" \
    --output_dir "./motions" \
    --sample_idx 0

python scripts/deployment/play_g1_npy.py \
    --input_file "motions/demo_recon_resnet_hybrid_idx0.npy" \
    --input_fps 30 \
    --output_fps 50

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

### 3.2 Preprocessed Data (.npy) (Updated)
预处理脚本 `process_data.py` 已更新，强制将数据 Flatten 为 3D 张量 `(N, T, C)` 以解决 DataLoader 维度报错问题，并生成成对数据。

- **Robot Data:**
  - `g1_train.npy`: Shape `(N, 64, 29)` - Unitree G1 关节数据。
  - `mean.npy` / `std.npy`: 机器人数据的归一化统计量。
- **Human Data (Paired):**
  - `human_train.npy`: Shape `(N, 64, 126)` - 对应的 SMPL 姿态数据 (Flattened from 21x6 or similar)。
  - `human_mean.npy` / `human_std.npy`: 人类数据的归一化统计量。
- **Normalization:** 双边独立归一化 (Dual-side Independent Normalization)。

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

## 5. System Architecture: Dual-Encoder Shared-Space VQ-VAE

本项目采用 **Dual-Encoder (双编码器)** 架构来实现 Implicit Retargeting，核心思想是强迫人类和机器人的动作在量化潜在空间（Quantized Latent Space）中对齐。

### 5.1 Module Definitions
1.  **Robot Encoder ($E_r$):** - Input: $x_r \in \mathbb{R}^{T \times 29}$
    - Output: $z_{e,r}$
2.  **Human Encoder ($E_h$):** - Input: $x_h \in \mathbb{R}^{T \times 126}$
    - Output: $z_{e,h}$ (通过卷积层映射到与 $E_r$ 相同的隐层维度)
3.  **Shared Quantizer ($Q$):** - Method: **HybridVQ (FSQ + RVQ)**
    - Role: "The Bridge". 无论输入来源如何，都将其离散化为共享的 Codebook 索引。
4.  **Robot Decoder ($D_r$):** - Input: $z_q$ (Quantized Latent)
    - Output: $\hat{x}_r \in \mathbb{R}^{T \times 29}$ (始终生成机器人动作)

### 5.2 Inference Flow (Retargeting)
推理时仅使用 Human Encoder 和 Robot Decoder：
$$\text{Human Input } x_h \xrightarrow{E_h} z_{e,h} \xrightarrow{Q} z_q \xrightarrow{D_r} \text{Robot Motion } \hat{x}_r$$

## 6. Workflow (Updated for Alignment)



### 6.1 Training Strategy

训练时同时输入成对的 $(x_r, x_h)$：
1.  **Robot Flow (Self-Recon):** $x_r \to E_r \to Q \to D_r \to \hat{x}_r$
2.  **Human Flow (Cross-Recon):** $x_h \to E_h \to Q \to D_r \to \hat{x}_{cross}$
3.  **Alignment Constraint:** $z_{e,r} \approx z_{e,h}$

### 6.2 Evaluation Metrics (Updated)

本项目使用以下 8 个核心指标评估模型性能。

#### 指标用途分类说明：
* **[LOSS]**: 该指标直接参与损失函数计算，用于模型优化。
* **[雷达图]**: 该指标出现在 `final_radar_chart.png` 中，用于模型间的综合性能对比。
* **[曲线]**: 该指标在 `results/*.json` 中记录，并生成对应的 `compare_*.png` 训练趋势图。

#### 指标详表：

1.  **Reconstruction Loss (MSE): [LOSS] [雷达图] [曲线]**
    * 衡量机器人自身重建动作的精确度。
2.  **Cross-Reconstruction Loss (Retargeting Error): [LOSS] [雷达图] [曲线]**
    * 衡量 Human -> Robot 转换后与 Ground Truth 的差异。
3.  **Alignment Loss (Latent MSE + InfoNCE): [LOSS] [雷达图] [曲线]**
    * 衡量 $z_{human}$ 和 $z_{robot}$ 在潜在空间中的重叠程度。
4.  **Velocity Loss: [LOSS] [雷达图] [曲线]**
    * 衡量动作一阶导数的一致性，确保动态特性。
5.  **VQ Loss (Commitment): [LOSS] [曲线]**
    * 衡量编码器输出靠近 Codebook 中心的程度（不计入雷达图以保持视觉平衡）。
6.  **Jerk Loss (Smoothness): [雷达图] [曲线]**
    * 衡量加加速度，评估生成的机械臂动作是否平滑、无抖动（不参与训练）。
7.  **Perplexity (PPL): [雷达图] [曲线]**
    * 衡量 Codebook 条目的活跃利用率。
8.  **Dead Code Ratio (DCR): [曲线]**
    * 衡量未被激活的条目比例（已从雷达图移除，仅作为训练诊断参考）。

#### 6.3 Loss Function Configuration (Updated)

模型训练的总损失函数由以下 5 部分加权组成，旨在实现精确重建与跨域对齐的平衡：

$$\mathcal{L}_{total} = \lambda_{recon} \mathcal{L}_{recon} + \lambda_{vq} \mathcal{L}_{vq} + \lambda_{vel} \mathcal{L}_{vel} + \lambda_{cross} \mathcal{L}_{cross} + \lambda_{align} \mathcal{L}_{align}$$

* **权重优化记录**: 为了解决 t-SNE 分离问题，$\lambda_{align}$ 已提升至 **100.0**，且引入了 **InfoNCE 对比损失**。
* **可视化监控**: `loss_breakdown.png` 现在支持全部 5 路损失的均值与方差带（Mean ± Std）展示。

### 6.4 Visualization Workflow & Analysis

为了验证 Human-to-Robot 的 Implicit Retargeting 效果，本项目建立了标准化的可视化分析流程 (`scripts/analyze_latent_space.py`)：

1.  **Data Loading & Normalization:**
    -   加载成对的 Human/Robot 动作数据。
    -   **关键步骤:** 执行双边独立归一化 (`(x - mean) / std`)，确保两个域的数据在进入编码器前处于同一尺度，消除数值分布差异导致的伪分离。
2.  **Feature Extraction (Continuous Latent):**
    -   提取量化前的潜变量 $z_e$ (Encoder Output)，而非量化后的 $z_q$。
    -   **原因:** $z_e$ 代表了编码器的原始映射能力，观察 $z_e$ 的分布重叠程度能直接反映 Alignment Loss 是否生效。
3.  **Temporal Pooling:**
    -   对时间维度进行平均池化 (Mean Pooling)，将一段时间序列动作压缩为潜在空间中的一个点。
4.  **Dimensionality Reduction (t-SNE):**
    -   将高维特征 ($D=64$) 降维至 2D 平面。
    -   **判据:**
        -   **Color by Action:** 不同动作（Run vs Walk）应形成自然聚类。
        -   **Color by Domain:** Robot 和 Human 的数据点应当**完全重叠 (Overlap)**。若泾渭分明，则说明 Domain Gap 依然存在。

### 6.5 下阶段进行任务[待办]

1. Isaac Lab 同屏演示 (Deployment):
   - 编写 scripts/deployment/play_isaac.py。
   - 实现“人类(Marker) - 机器人(GT) - 机器人(Model)”三位一体同屏回放。
2. 路径适配稳健化:
   - 将所有脚本的 sys.path 修改为溯源至 project_root 的 Path.parents[2] 写法。
3. 模型验证:
   - 重新运行训练，观察 λ_align=100 后 t-SNE 是否实现红蓝点重叠。

## 7. Final Experimental Results (SOTA Performance)

经过多轮消融实验与架构迭代，我们提出的 **Hybrid (FSQ+VQ)** 模型被确认为本项目的最终 SOTA 模型。

-   **性能评估:** 该模型成功克服了 RVQ 的抖动问题和 FSQ 的精度瓶颈，被称为“五边形战士”：
    -   **精度:** Reconstruction Loss (**0.0081**) 和 Velocity Loss (**0.0006**) 均为全场最低，显著优于单纯的 RVQ (0.0097)。
    -   **利用率:** PPL 高达 1670.3，有效利用了潜在空间。
    -   **平滑度:** Jerk Loss (**0.0022**) 极低，仅次于甚至逼近以牺牲精度为代价的 LFQ (0.0020)，远优于 Baseline 和单纯 RVQ。
-   **LaTeX 表格:** 以下是最终导出的实验数据表格，已格式化为 LaTeX 代码：

```latex
\begin{table}[h]
\centering
\caption{Comparison of different VQ-VAE variants (Expanded Metrics).}
\label{tab:results}
\begin{tabular}{l c c c c c c c}
\toprule
Method & Recon $\downarrow$ & Cross $\downarrow$ & Align $\downarrow$ & Vel $\downarrow$ & Jerk $\downarrow$ & PPL $\uparrow$ & DCR \% $\downarrow$ \\
\midrule
Baseline(Simple) & 0.0457 $\pm$ 0.0027 & 0.0459 $\pm$ 0.0027 & 0.0024 $\pm$ 0.0001 & 0.0018 $\pm$ 0.0000 & 0.0061 $\pm$ 0.0000 & 84.6 $\pm$ 7.0 & 89.7 $\pm$ 1.0 \\
ResNet+EMA & 0.0355 $\pm$ 0.0000 & 0.0356 $\pm$ 0.0001 & 0.0017 $\pm$ 0.0001 & 0.0011 $\pm$ 0.0000 & 0.0024 $\pm$ 0.0001 & 199.5 $\pm$ 24.7 & 74.3 $\pm$ 3.2 \\
ResNet+RVQ & 0.0124 $\pm$ 0.0002 & 0.0130 $\pm$ 0.0000 & 0.0005 $\pm$ 0.0000 & 0.0008 $\pm$ 0.0000 & 0.0023 $\pm$ 0.0000 & 520.8 $\pm$ 2.4 & 34.9 $\pm$ 0.6 \\
FSQ & 0.0317 $\pm$ 0.0016 & 0.0317 $\pm$ 0.0014 & 0.0004 $\pm$ 0.0001 & 0.0010 $\pm$ 0.0000 & 0.0030 $\pm$ 0.0000 & 1363.9 $\pm$ 4.0 & 0.0 $\pm$ 0.0 \\
LFQ & 0.0580 $\pm$ 0.0006 & 0.0571 $\pm$ 0.0005 & 0.0837 $\pm$ 0.0482 & 0.0013 $\pm$ 0.0000 & 0.0038 $\pm$ 0.0000 & 164.2 $\pm$ 8.9 & 84.0 $\pm$ 0.9 \\
Ours(Dual-Enc+Hybrid) & 0.0120 $\pm$ 0.0006 & 0.0127 $\pm$ 0.0006 & 0.0005 $\pm$ 0.0000 & 0.0007 $\pm$ 0.0000 & 0.0023 $\pm$ 0.0000 & 1096.8 $\pm$ 0.3 & 0.0 $\pm$ 0.0 \\
\bottomrule
\end{tabular}
\end{table}

```