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
│       ├── g1_train_full_raw.npy # 完整长序列数据
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
│   ├── deployment/            # Isaac Lab 演示 (play_isaac)
│   │   ├── export_motion.py             # 数据输出格式转换
│   │   ├── render_viewport.py #[待修复]视角Bug
│   │   └── play_g1_npy.py            # 使用isaaclab模拟
│   ├── evaluation/            # 结果导出 (plot, export_latex)
│   │   ├── plot_results.py            # 绘图脚本 (含方差带, Loss/PPL/Jerk/DCR/Radar(标准和对数))
│   │   ├── analyze_latent_space.py # 可视化隐空间分布
│   │   └── export_latex_table.py      # (新增) 读取日志并生成 LaTeX 格式的 
│   └── train_ablation.py          # 训练主脚本 (多 Seed, 支持 Jerk/DCR 指标)
├── utils/
│   ├── __init__.py
│   └── alignment.py
├── plots/ #输出图片
│   ├── latent_space/
│   │   ├──...
│   │   └── align_Baseline(Simple)_ema_seed_42.png
│   ├── metrics/
│   │   ├── compare_dcr.png
│   │   ├── compare_jerk.png
│   │   ├── compare_ppl.png
│   │   ├── compare_recon.png
│   │   ├── ...
│   │   └── compare_vel.png
│   ├── radar/
│   │   ├── final_radar_chart_log.png
│   │   ├── ...
│   │   └── final_radar_chart.png
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

#### 1. **Data Download:**
```bash
   python scripts/download_data.py
```

#### 2. **Unzip Data:**
   (需手动解压至指定目录)
```bash
   mkdir -p ./data/raw/unzipped
   tar -xjf ./data/raw/seulzx/smplx_datasets/lafan1_smplx_datasets.tar.bz2 -C ./data/raw/unzipped
   tar -xjf ./data/raw/seulzx/smplx_datasets/extended_datasets.tar.bz2 -C ./data/raw/unzipped
```

#### 3. **Data Inspection (Optional):**
```bash
   python scripts/data/inspect_npz.py
```

#### 4. **Preprocessing:**
```bash
# 生成包含完整长序列的数据集
# --overwrite 是必须的，否则它会发现 output_dir 有文件而跳过
python scripts/data/process_data.py \
    --input_dir "./data/raw/unzipped/extended_datasets/lafan1_dataset/g1" \
    --output_dir "./data/processed" \
    --window 10 \
    --step 1\
    --overwrite
```

#### 5. **Training (Multi-Seed Ablation):**

##### 大架构
```bash
python scripts/train_ablation.py \
    --mode teacher \
    --arch transformer \
    --method hybrid \
    --window 10 \
    --epochs 500 \
    --patience 50 \
    --batch_size 512 \
    --resume

python scripts/train_ablation.py \
    --mode student \
    --arch transformer \
    --method hybrid \
    --window 10 \
    --epochs 1000 \
    --patience 50 \
    --batch_size 1024 \
    --teacher_ckpt "checkpoints/Exp_transformer_W10_hybrid_teacher_seed_42_best.pth" \
    --resume

```
##### 小架构

```bash
python scripts/train_ablation.py \
    --mode teacher \
    --arch resnet_no_down \
    --method ae \
    --window 10 \
    --epochs 500 \
    --batch_size 4096 \
    --patience 20 \
    --seed 42 1024

python scripts/train_ablation.py \
    --mode student \
    --arch resnet_no_down \
    --method ae \
    --window 10 \
    --epochs 500 \
    --batch_size 4096 \
    --patience 20 \
    --teacher_ckpt "checkpoints/Exp_resnet_no_down_W10_ae_teacher_seed_42_best.pth" \
    --seed 42 1024
```

#### 6. **Result Export & Visualization:**
```bash
python scripts/evaluation/export_latex_table.py
# 1. 绘制训练曲线 (Loss, Recon, Align, DCR 等)
# 现在的实验名称通常包含 "transformer" 或 "Ours"，建议用 filter 筛选
python scripts/evaluation/plot_results.py --dir results --filter transformer

# 2. 隐空间可视化 (t-SNE)
# !!! 关键变化：--window 必须设为 64，否则数据加载形状对不上模型
python scripts/evaluation/analyze_latent_space.py --window 10 --filter student
```



#### 7. **启动isaaclab可视化转换数据**
```bash
# 确保在项目根目录下执行，这样脚本才能找到 models 文件夹
cd /home/kaijie/深度学习/PJ_vqvae
conda activate isaaclab_env
# 使用 isaaclab.sh 运行脚本，-p 参数代表用内置 python 运行
# 即使 custom_ae 不在 EXPERIMENTS 里，指定 --method ae 也能跑
# 从第 1000 个样本开始，连续导出 20 个
# 导出 Robot 重构动作 (用于测试 VQ-VAE 的还原能力)
python scripts/deployment/export_motion.py \
    --ckpt "checkpoints/Exp_transformer_W10_hybrid_teacher_seed_42_best.pth" \
    --arch transformer \
    --method hybrid \
    --window 10 \
    --start_idx 0 \
    --num_samples 1 \
    --step_size 5 \
    --max_len 600

# 纯AE
python scripts/deployment/export_motion.py \
    --ckpt "checkpoints/Exp_resnet_no_down_W10_ae_teacher_seed_42_best.pth" \
    --arch resnet_no_down \
    --method ae \
    --window 10 \
    --start_idx 0 \
    --num_samples 1 \
    --step_size 5 \
    --max_len 600

# [可不用]显示多帧动画
python scripts/deployment/play_g1_npy.py \
    --input_file "motions/demo_recon_resnet_hybrid_idx0.npy" \
    --input_fps 30 \
    --output_fps 50

```

#### 8.逐帧可视化

```bash
# 渲染原始数据
python scripts/deployment/render_viewport.py --input_file "motions/idx0_gt.npy"

# 渲染重构数据 (请根据您实际生成的 npy 文件名修改路径)
python scripts/deployment/render_viewport.py --input_file "motions/recon_transformer_FullSeq_W10_idx0.npy"

python scripts/deployment/render_viewport.py --input_file "motions/recon_resnet_no_down_FullSeq_W10_idx0.npy"

```
或者
```bash
# 渲染原始数据
python scripts/deployment/render_viewport.py --max_shots 50 --input_file "motions/idx0_gt.npy" "motions/recon_transformer_FullSeq_W10_idx0.npy"
```

## 5. System Architecture: Transformer-based VQ-VAE (Teacher-Student)

本项目采用 **Transformer** 架构配合 **Teacher-Student (两阶段)** 训练策略。核心改进在于将时序动作（Window Size，默认 10 帧）压缩为潜在空间中的 **唯一 Token (Global Latent Token)**，从而实现基于意图的强对齐。

### 5.1 Module Definitions
1.  **Transformer Encoder ($E$):**
    -   **Structure:** `Linear` -> `Transformer Encoder (Global Attention)` -> `Global Mean Pooling` -> `Linear`
    -   **Input Data Dimensions:**
        -   **Robot ($x_r$):** $\mathbb{R}^{10 \times 29}$ (29 DoF Joint Positions)
        -   **Human ($x_h$):** $\mathbb{R}^{10 \times 126}$ (SMPL-X Body Pose: 21 Joints $\times$ 6D Rotation)
    -   **Output:** $z_e \in \mathbb{R}^{1 \times 64}$ (Single Token representing the whole motion sequence)
    -   **Mechanism:** 利用 Transformer 的全局注意力机制捕捉窗口内的完整运动意图。

2.  **Shared Quantizer ($Q$):**
    -   **Method:** **HybridVQ (FSQ + RVQ)**
    -   **Role:** 将连续的意图 Token $z_e$ 离散化为 Codebook 索引。

3.  **Transformer Decoder ($D_r$):**
    -   **Input:** $z_q \in \mathbb{R}^{1 \times 64}$
    -   **Mechanism:** `Repeat(10)` -> `Positional Encoding` -> `Transformer Encoder`
    -   **Output:** $\hat{x}_r \in \mathbb{R}^{10 \times 29}$ (一次性生成完整序列)

### 5.2 Inference Flow (Retargeting)
推理时使用训练好的 Human Encoder (Student) 和冻结的 Robot Decoder (Teacher)：
$$\text{Human Input } x_h (10\text{ frames}) \xrightarrow{E_h} \text{Token } z_{e,h} \xrightarrow{Q} z_q \xrightarrow{D_r} \text{Robot Motion } \hat{x}_r$$

## 6. Workflow (Teacher-Student Strategy)

为了解决“域分离”问题，本项目放弃了不稳定的同步对抗训练，转而采用 **Teacher-Student 两阶段训练法**。

### 6.1 Training Strategy

**Stage 1: Teacher Training (Robot VQ-VAE)**
-   **目标:** 建立完美的机器人动作“字典” (Codebook) 和重构能力。
-   **输入:** 仅机器人数据 $x_r$。
-   **模型:** 训练 $E_r, Q, D_r$。
-   **产出:** 训练完成后**冻结 (Freeze)** 所有参数。

**Stage 2: Student Alignment (Human-to-Robot)**
-   **目标:** 训练 Human Encoder 将人类动作映射到已经冻结的 Robot 潜在空间中。
-   **输入:** 成对数据 $(x_r, x_h)$。
-   **模型:** 仅训练 $E_h$ (Student)。$E_r$ 仅作为目标生成器 (Target Generator)。
-   **约束:** $z_{e,h} \approx z_{e,r}$ (Direct MSE)。

### 6.2 Loss Function Configuration (Updated)

根据训练阶段不同，损失函数分为两套：

**Stage 1 (Teacher):**
$$\mathcal{L}_{Teacher} = \lambda_{recon} \mathcal{L}_{recon} + \lambda_{vq} \mathcal{L}_{vq} + \lambda_{vel} \mathcal{L}_{vel}$$
* 重点关注机器人的物理动作还原与平滑度。

**Stage 2 (Student):**
$$\mathcal{L}_{Student} = \lambda_{align} \| z_{e,human} - z_{e,robot} \|^2_2$$
* **Token-level Alignment:** 由于潜在空间已被压缩为 1 个 Token，直接计算 MSE 即可实现强对齐，不再需要 InfoNCE 或时间维度的平均池化。

### 6.3 Visualization Analysis (Updated)

由于现在的模型架构直接输出单个隐变量 Token ($T=1$)：
1.  **无需 Temporal Pooling:** 代码中的可视化脚本不再需要手动对时间维度取平均。
2.  **t-SNE 含义:** 图上的每一个点直接代表一个**完整的 64 帧动作片段**。
3.  **对齐判据:** 在 Stage 2 训练良好的情况下，Human 和 Robot 的点在 t-SNE 图中应完全重合。

## 7. Experimental Results (Transformer Era)

**Update:** 转入 Transformer + Teacher-Student 架构后的最新实验数据 (Running...)。

### Current SOTA Configuration
-   **Architecture:** Transformer (4 Layers, d_model=64)
-   **Latent:** Single Token Compression (10 frames -> 1 frame)
-   **Quantizer:** Hybrid (FSQ + RVQ)
-   **Training:** Teacher-Student Two-Stage

*(待训练完成后填入新表格)*
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