import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

LOG_DIR = 'results'
SAVE_DIR = 'plots'
os.makedirs(SAVE_DIR, exist_ok=True)

# 映射名字让图表更好看
NAME_MAP = {
    'simple_ema': 'Baseline',
    'resnet_ema': 'ResNet+EMA',
    'resnet_rvq': 'ResNet+RVQ',
    'resnet_fsq': 'SOTA-FSQ',
    'resnet_lfq': 'SOTA-LFQ'
}

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

def plot_radar(data):
    summary = {}
    
    # 这里的 keys 顺序必须和雷达图轴的顺序对应
    metrics_list = ['val_recon', 'val_vel', 'val_jerk', 'perplexity', 'dead_code_ratio']
    
    for exp_key, metrics_dict in data.items():
        # 检查是否所有指标都存在 (防止旧日志报错)
        is_complete = True
        temp_vals = {}
        for k in metrics_list:
            if k not in metrics_dict or not metrics_dict[k]:
                is_complete = False
                break
            
            arr = np.array(metrics_dict[k])
            # 安全检查：确保是二维数组 [Seeds, Epochs]
            if arr.ndim != 2 or arr.size == 0:
                is_complete = False
                break
                
            # 取最后5轮的平均值 (此时把所有 Seed 混合在一起取平均)
            final_val = np.mean(arr[:, -5:]) 
            temp_vals[k] = final_val
            
        if is_complete:
            summary[exp_key] = temp_vals
        else:
            print(f"Warning: Skipping {exp_key} in Radar Chart due to missing metrics (likely old logs).")

    if not summary: 
        print("No valid data for Radar Chart.")
        return

    # 选取第一个有效实验作为 Baseline
    baseline_key = list(summary.keys())[0]
    categories = ['Recon Quality', 'Motion Smoothness', 'Low Jerk', 'Code Usage', 'Active Codes']
    
    radar_data = {}
    for name, vals in summary.items():
        base = summary[baseline_key]
        
        # 计算相对分数
        # 越小越好 (Loss, Jerk, DCR) -> Baseline / Current
        # 越大越好 (PPL) -> Current / Baseline
        
        s1 = np.clip(base['val_recon'] / (vals['val_recon'] + 1e-6), 0, 3)
        s2 = np.clip(base['val_vel'] / (vals['val_vel'] + 1e-6), 0, 3)
        s3 = np.clip(base['val_jerk'] / (vals['val_jerk'] + 1e-6), 0, 3)
        
        # PPL 越大越好
        s4 = np.clip(vals['perplexity'] / (base['perplexity'] + 1e-6), 0, 3)
        
        # Dead Code Ratio 越小越好 (加个小常数防止除以0)
        s5 = np.clip((base['dead_code_ratio'] + 0.01) / (vals['dead_code_ratio'] + 0.01), 0, 3)
        
        radar_data[name] = [s1, s2, s3, s4, s5]

    # 绘图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, scores) in enumerate(radar_data.items()):
        vals = scores + scores[:1]
        label = NAME_MAP.get(name, name)
        ax.plot(angles, vals, linewidth=2, label=label, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title('Relative Performance (vs Baseline)', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(SAVE_DIR, 'final_radar_chart.png'), dpi=150)
    print("Saved final_radar_chart.png")
    plt.close()

def main():
    data = load_and_aggregate()
    if not data: return

    # 1. Basic Losses
    plot_metric(data, 'val_recon', 'Reconstruction Error', 'MSE', 'compare_recon.png', log_scale=True)
    plot_metric(data, 'val_vel', 'Velocity Error', 'MSE', 'compare_vel.png')
    
    # 2. New Metrics
    plot_metric(data, 'val_jerk', 'Jerk Error (Smoothness)', 'Jerk MSE', 'compare_jerk.png')
    plot_metric(data, 'dead_code_ratio', 'Dead Code Ratio (Lower is Better)', 'Ratio [0-1]', 'compare_dcr.png')
    
    # 3. Radar
    plot_radar(data)

if __name__ == "__main__":
    main()