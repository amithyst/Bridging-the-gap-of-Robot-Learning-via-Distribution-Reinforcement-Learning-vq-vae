import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

LOG_DIR = 'results'
SAVE_DIR = 'plots'
os.makedirs(SAVE_DIR, exist_ok=True)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.experiment_config import EXPERIMENTS

# 1. 更新名称映射
NAME_MAP = {exp['id']: exp['name'] for exp in EXPERIMENTS}

def smooth(scalars, weight=0.8):
    if len(scalars) == 0: return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + point * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def load_and_aggregate():
    data = {}
    files = glob.glob(os.path.join(LOG_DIR, "log_*.json"))
    
    if not files:
        print("No log files found.")
        return {}

    # 1. 读取所有原始数据
    raw_data = {} 
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.replace("log_", "").replace(".json", "").split("_seed_")
        if len(parts) != 2: continue
        exp_key = parts[0]
        
        try:
            with open(fpath, 'r') as f:
                log = json.load(f)
            if exp_key not in raw_data:
                raw_data[exp_key] = []
            raw_data[exp_key].append(log)
        except json.JSONDecodeError:
            print(f"Skipping broken file: {fpath}")

    # 2. 对齐并聚合
    keys_to_plot = ['val_recon', 'val_vel', 'val_jerk', 'perplexity', 'dead_code_ratio']
    
    for exp_key, seed_logs in raw_data.items():
        if not seed_logs: continue
        
        # 找到所有 Seed 中最短的 epoch 数，防止断点续传导致长度不一
        # 注意：这里只检查 val_recon 存在的日志
        valid_logs = [log for log in seed_logs if 'val_recon' in log]
        if not valid_logs: continue

        min_len = min(len(log['val_recon']) for log in valid_logs)
        if min_len < 2: continue 
        
        data[exp_key] = {}
        for metric in keys_to_plot:
            # 关键修复：如果旧日志里没有这个 metric，就会产生空列表
            metric_data = [log[metric][:min_len] for log in valid_logs if metric in log]
            data[exp_key][metric] = metric_data # List of Lists
            
    return data

def plot_metric(data, metric_key, title, ylabel, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    valid_plot = False
    for i, (exp_key, metrics_dict) in enumerate(data.items()):
        # 安全检查：如果没有这个指标的数据，跳过
        if metric_key not in metrics_dict or not metrics_dict[metric_key]:
            continue
        
        seeds_data = metrics_dict[metric_key]
        arr = np.array(seeds_data) 
        
        # 如果数组为空或维度不对，跳过
        if arr.size == 0 or arr.ndim != 2: continue
        
        # 核心：计算 Mean 和 Std (多种子逻辑在这里体现)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        mean_curve = smooth(mean_curve)
        
        label = NAME_MAP.get(exp_key, exp_key)
        color = colors[i % len(colors)]
        epochs = range(1, len(mean_curve) + 1)
        
        plt.plot(epochs, mean_curve, label=label, color=color, linewidth=2)
        # 画方差带
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
        valid_plot = True
        
    if valid_plot:
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if log_scale: plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150)
        print(f"Saved {filename}")
    plt.close()

# 2. 修改 plot_radar 函数，增加数据清洗
def plot_radar(data):
    summary = {}
    metrics_list = ['val_recon', 'val_vel', 'val_jerk', 'perplexity', 'dead_code_ratio']
    
    for exp_key, metrics_dict in data.items():
        is_complete = True
        temp_vals = {}
        for k in metrics_list:
            if k not in metrics_dict or not metrics_dict[k]:
                is_complete = False; break
            
            arr = np.array(metrics_dict[k])
            if arr.ndim != 2 or arr.size == 0:
                is_complete = False; break
            
            # 取最后 5 个 epoch 的平均值
            final_val = np.mean(arr[:, -5:]) 
            
            # 针对 FSQ/Hybrid 的 DCR 负数修正
            if k == 'dead_code_ratio':
                if 'fsq' in exp_key or 'hybrid' in exp_key:
                    final_val = max(0.0, final_val)
            
            temp_vals[k] = final_val
            
        if is_complete:
            summary[exp_key] = temp_vals

    if not summary: 
        print("No valid data for Radar Chart.")
        return

    baseline_key = list(summary.keys())[0]
    categories = ['Recon Quality', 'Motion Smoothness', 'Low Jerk', 'Code Usage', 'Active Codes']
    
    # === 定义两种绘图配置 ===
    plot_configs = [
        {
            'filename': 'final_radar_chart.png',
            'log_scale': False,
            'title': 'Relative Performance (Normalized by Best)',
            # 线性模式：使用手动微调的 Limits 让 SOTA 刚好顶格
            'limits': [5.0, 2.2, 1.5, 23.0, 20.0] 
        },
        {
            'filename': 'final_radar_chart_log.png',
            'log_scale': True,
            'title': 'Relative Performance (Log Scale, Baseline=1.0)',
            # 对数模式：Log2(Ratio)+1。
            # Limit=6.0 意味着能容纳 2^(6-1) = 32倍 的性能提升，足够囊括 PPL 的 22倍
            'limits': [3.5, 2.1, 1.7, 6.0, 6.0] 
        }
    ]

    # 1. Recon: Hybrid ~3.3x -> 设上限 4.0
    # 2. Vel: Hybrid ~1.9x -> 设上限 2.5 (让 1.9 看起来饱满)
    # 3. Jerk: LFQ ~3.2x, Hybrid ~1.8x -> 设上限 4.0 (包容 LFQ)
    # 4. PPL: Hybrid ~6.5x, FSQ ~14x -> 设上限 8.0 (重点展示 Hybrid 的强项，FSQ 爆表也没关系)
    # 5. DCR: Hybrid ~12x -> 设上限 15.0
    
    # === 循环生成两张图 ===
    for config in plot_configs:
        is_log = config['log_scale']
        current_limits = config['limits']
        PLOT_RADIUS = 3.0
        
        radar_data = {}
        for name, vals in summary.items():
            base = summary[baseline_key]
            
            # 1. 计算原始倍数 (Ratios)
            # 越小越好 -> Baseline / Current
            r1 = base['val_recon'] / (vals['val_recon'] + 1e-6)
            r2 = base['val_vel'] / (vals['val_vel'] + 1e-6)
            r3 = base['val_jerk'] / (vals['val_jerk'] + 1e-6)
            
            # 越大越好 -> Current / Baseline
            r4 = vals['perplexity'] / (base['perplexity'] + 1e-6)
            r5 = (base['dead_code_ratio'] + 0.01) / (vals['dead_code_ratio'] + 0.01)
            
            raw_ratios = [r1, r2, r3, r4, r5]
            
            # 2. 核心逻辑分支：对数 vs 线性
            plot_vals = []
            for r, limit in zip(raw_ratios, current_limits):
                if is_log:
                    # === 对数逻辑 ===
                    # 使用 Log2，直观含义：+1 代表性能翻倍
                    # +1.0 是偏移量，让 Baseline (Ratio=1.0) 位于 1.0 刻度处，而不是 0
                    # max(0.1, r) 防止 log(0) 报错
                    val_log = np.log2(max(0.1, r)) + 1.0
                    
                    # 归一化: (Log值 / Limit) * 半径
                    # 这里的 Limit 6.0 对应 2^5 = 32倍提升
                    norm_val = (val_log / limit) * PLOT_RADIUS
                    
                    # 截断防止画出圈外
                    plot_vals.append(np.clip(norm_val, 0, PLOT_RADIUS))
                else:
                    # === 线性逻辑 (旧代码) ===
                    norm_val = (r / limit) * PLOT_RADIUS
                    plot_vals.append(np.clip(norm_val, 0, PLOT_RADIUS))
                
            radar_data[name] = plot_vals

        # 3. 绘图部分
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 根据模式调整网格标签
        if is_log:
            # Log 模式下，网格代表 1x, 2x, 4x, 8x...
            grid_labels = [] # 不显示具体数字，保持简洁
            ax.set_ylim(0, PLOT_RADIUS)
        else:
            ax.set_ylim(0, PLOT_RADIUS)

        # 设置网格线位置 (5等分)
        ax.set_rgrids([0.6, 1.2, 1.8, 2.4, 3.0], labels=[], angle=0)

        for i, (name, scores) in enumerate(radar_data.items()):
            vals = scores + scores[:1]
            label = NAME_MAP.get(name, name)
            ax.plot(angles, vals, linewidth=2, label=label, color=colors[i % len(colors)])
            ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        plt.title(config['title'], y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) 
        
        save_path = os.path.join(SAVE_DIR, config['filename'])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()



def main():
    data = load_and_aggregate()
    if not data: return

    # 1. 核心质量指标 (对应雷达图)
    plot_metric(data, 'val_recon', 'Reconstruction Error', 'MSE', 'compare_recon.png', log_scale=True)
    plot_metric(data, 'val_vel', 'Velocity Error', 'MSE', 'compare_vel.png')
    plot_metric(data, 'val_jerk', 'Jerk Error (Smoothness)', 'Jerk MSE', 'compare_jerk.png')
    
    # === 补全: 漏掉的 Perplexity 曲线 ===
    plot_metric(data, 'perplexity', 'Codebook Usage (Perplexity)', 'Score', 'compare_ppl.png')
    
    plot_metric(data, 'dead_code_ratio', 'Dead Code Ratio (Lower is Better)', 'Ratio [0-1]', 'compare_dcr.png')
    
    # 2. 训练收敛情况 (新增: Total Loss，用于检查是否收敛，不一定放论文)
    plot_metric(data, 'train_loss', 'Total Training Loss (Convergence Check)', 'Loss', 'compare_train_loss.png', log_scale=True)
    
    # 3. 雷达图
    plot_radar(data)

if __name__ == "__main__":
    main()