import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import argparse
import math

# === 默认配置 ===
# 定义雷达图关注的 5-6 个核心维度
RADAR_METRICS_MAP = {
    'Recon': 'val_recon',        # 重构误差 (越低越好)
    'Align': 'val_align',        # 对齐误差 (Student模式, 越低越好)
    'Smooth': 'val_vel',         # 速度平滑度 (越低越好)
    'Jerk': 'val_jerk',          # 加速度变化率 (越低越好)
    'Usage': 'dcr',              # Codebook 死码率 (越低越好, 0最好)
    'PPL': 'perplexity'          # 困惑度 (越高越好，代表多样性)
}

# 需要绘制曲线的所有指标
ALL_METRICS = [
    'val_recon', 'val_vel', 'val_jerk', 'val_align', 
    'perplexity', 'dcr', 'rvq_ppl',
    'train_loss', 'train_recon_loss', 'train_vq_loss'
]

def smooth(scalars, weight=0.8):
    if len(scalars) == 0: return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + point * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def load_and_aggregate(log_dir, filter_str=None):
    data = {}
    files = glob.glob(os.path.join(log_dir, "log_*.json"))
    
    if not files: 
        print(f"No log files found in {log_dir}")
        return {}

    print(f"Found {len(files)} logs. Filtering with '{filter_str}'...")

    for fpath in files:
        fname = os.path.basename(fpath)
        if filter_str and filter_str not in fname: continue
            
        # 尝试提取实验名 (去除 log_ 和 _seed_xxx)
        # e.g. log_Exp_transformer_hybrid_teacher_seed_42.json
        try:
            clean_name = fname.replace("log_", "").replace(".json", "")
            # 分割出 seed 之前的部分作为实验 ID
            exp_id = clean_name.split("_seed_")[0]
        except:
            exp_id = fname
        
        try:
            with open(fpath, 'r') as f:
                log = json.load(f)
            
            if exp_id not in data: data[exp_id] = {}
            
            # 将该 seed 的数据加入列表
            for metric in ALL_METRICS:
                if metric in log and len(log[metric]) > 0:
                    if metric not in data[exp_id]: data[exp_id][metric] = []
                    data[exp_id][metric].append(log[metric])
                    
        except Exception as e: 
            print(f"Error reading {fname}: {e}")

    return data

def plot_metric_curve(data, metric_key, save_dir):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    has_data = False
    for i, (exp_id, metrics) in enumerate(data.items()):
        if metric_key not in metrics: continue
        
        # List of lists (seeds)
        raw_vals = metrics[metric_key] 
        # 对齐长度
        min_len = min(len(x) for x in raw_vals)
        if min_len < 2: continue
        
        arr = np.array([x[:min_len] for x in raw_vals])
        
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        mean_smooth = smooth(mean_curve) # 平滑用于绘图
        
        epochs = range(1, len(mean_smooth) + 1)
        
        label = exp_id.replace('_', ' ')
        color = colors[i % len(colors)]
        
        plt.plot(epochs, mean_smooth, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, mean_smooth - std_curve, mean_smooth + std_curve, color=color, alpha=0.1)
        has_data = True
        
    if has_data:
        plt.title(f'Comparison: {metric_key}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_key)
        if 'loss' in metric_key or 'recon' in metric_key:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'compare_{metric_key}.png'), dpi=150)
        print(f"Saved compare_{metric_key}.png")
    plt.close()

def plot_radar_chart(data, save_dir):
    """
    绘制相对性能雷达图。
    注意：为了美观，通常会对指标进行归一化（相对于 Baseline 或 Max 值）。
    这里采用 '越小越好' 统一转换逻辑（除了 PPL）。
    """
    categories = list(RADAR_METRICS_MAP.keys())
    N = len(categories)
    
    # 1. Calculate Final Means per Experiment
    summary = {}
    for exp_id, metrics in data.items():
        vals = []
        for cat, key in RADAR_METRICS_MAP.items():
            if key in metrics:
                # 取最后 10 个 epoch 的平均值作为最终性能
                all_seeds = np.array([s[-10:] for s in metrics[key]])
                val = np.mean(all_seeds)
            else:
                val = 0.0 # 缺失数据
            vals.append(val)
        summary[exp_id] = vals

    if len(summary) < 1: return

    # 2. Normalize Data (Min-Max scaling for visualization)
    # 我们希望雷达图的面积越大代表"综合性能越好"。
    # 对于 Recon/Jerk/DCR: 值越小越好 -> 转换成 1/x 或 (Max - x)
    # 对于 PPL: 值越大越好 -> 保持
    
    # 这里简单处理：将所有负向指标倒数，然后归一化到 [0, 1]
    plot_data = {}
    
    # 获取每一列的最大最小值
    vals_array = np.array(list(summary.values())) # (Exp, Metrics)
    
    # 处理负向指标 (Recon, Align, Smooth, Jerk, Usage/DCR 都是越小越好)
    # 只有 PPL (Index 5) 是越大越好
    # 为了统一：这里绘制 "Score" (分数)，分数越高越好
    scores = np.zeros_like(vals_array)
    
    for i in range(N):
        col_vals = vals_array[:, i]
        if categories[i] == 'PPL':
            # 越大越好: Normalize to [0.2, 1.0]
            if col_vals.max() > col_vals.min():
                scores[:, i] = 0.2 + 0.8 * (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min())
            else:
                scores[:, i] = 1.0
        else:
            # 越小越好: 反向 Normalize
            # val -> (max - val) / (max - min)
            if col_vals.max() > col_vals.min():
                scores[:, i] = 0.2 + 0.8 * (col_vals.max() - col_vals) / (col_vals.max() - col_vals.min())
            else:
                scores[:, i] = 1.0

    # 3. Plotting
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += [angles[0]] # Close the loop
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for idx, (exp_id, _) in enumerate(summary.items()):
        values = scores[idx].tolist()
        values += [values[0]]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=exp_id.replace('_', ' '), color=colors[idx % len(colors)])
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.1)

    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Relative Performance (Higher Area = Better)")
    plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=150, bbox_inches='tight')
    print("Saved radar_chart.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results', help='Directory containing json logs')
    parser.add_argument('--out', type=str, default='plots/metrics', help='Output directory')
    parser.add_argument('--filter', type=str, default=None, help='Filter logs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    data = load_and_aggregate(args.dir, args.filter)
    if not data: return

    # Plot Curves
    for m in ALL_METRICS:
        plot_metric_curve(data, m, args.out)
    
    # Plot Radar
    plot_radar_chart(data, os.path.dirname(args.out)) # Save in plots/ root

if __name__ == "__main__":
    main()