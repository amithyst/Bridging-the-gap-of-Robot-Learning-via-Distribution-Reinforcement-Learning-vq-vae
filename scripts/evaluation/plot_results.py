import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

LOG_DIR = 'results'
SAVE_DIR = 'plots'
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_PLOT_DIR = 'plots'
METRIC_DIR = os.path.join(BASE_PLOT_DIR, 'metrics')
RADAR_DIR = os.path.join(BASE_PLOT_DIR, 'radar')
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(RADAR_DIR, exist_ok=True)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.experiment_config import EXPERIMENTS

# 1. 更新名称映射
NAME_MAP = {exp['id']: exp['name'] for exp in EXPERIMENTS}
RADAR_Number=6 # 雷达图指标数量
RADAR_COEFFS = [1.05]*RADAR_Number

# === 指标配置中心 [新增/修改] ===
# 雷达图显示的指标 (顺序必须与下方 r1-r7 一致)
RADAR_METRICS = ['val_recon', 'val_vel', 'val_jerk', 'perplexity',
                #   'dead_code_ratio',
                  'val_align', 'val_cross_recon']
RADAR_categories = ['Recon Quality', 'Motion Smoothness', 'Low Jerk', 'Code Usage',
                    #  'Active Codes',
                       'Alignment', 'Retargeting']
# 画曲线的指标
EXTRA_METRICS = [
    'train_recon_loss', 'train_align_loss', 'train_cross_loss', 
    'train_vq_loss', 'train_vel_loss',  # <--- [新增] 确保 main 会加载并画图
    'dead_code_ratio'
]
# 全部需要加载的指标
ALL_METRICS = RADAR_METRICS + EXTRA_METRICS

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
    
    if not files: return {}

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
        except json.JSONDecodeError: pass

    # === 修改：增加要提取的 Key ===
    keys_to_plot = ALL_METRICS
    
    for exp_key, seed_logs in raw_data.items():
        if not seed_logs: continue
        
        valid_logs = [log for log in seed_logs if 'val_recon' in log]
        if not valid_logs: continue

        min_len = min(len(log['val_recon']) for log in valid_logs)
        if min_len < 2: continue 
        
        data[exp_key] = {}
        for metric in keys_to_plot:
            metric_data = [log[metric][:min_len] for log in valid_logs if metric in log]
            data[exp_key][metric] = metric_data
            
    return data

def plot_metric(data, metric_key, title, ylabel, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    valid_plot = False
    for i, (exp_key, metrics_dict) in enumerate(data.items()):
        if metric_key not in metrics_dict or not metrics_dict[metric_key]:
            continue
        
        seeds_data = metrics_dict[metric_key]
        arr = np.array(seeds_data) 
        if arr.size == 0 or arr.ndim != 2: continue
        
        mean_curve = smooth(np.mean(arr, axis=0))
        std_curve = np.std(arr, axis=0)
        
        label = NAME_MAP.get(exp_key, exp_key)
        color = colors[i % len(colors)]
        epochs = range(1, len(mean_curve) + 1)
        
        plt.plot(epochs, mean_curve, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
        valid_plot = True
        
    if valid_plot:
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if log_scale: plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(METRIC_DIR, filename), dpi=150)
        print(f"Saved {filename}")
    plt.close()

def plot_loss_breakdown(data, target_exp_key='ours_hybrid'):
    if target_exp_key not in data:
        print(f"Skipping breakdown plot: {target_exp_key} not found.")
        return

    metrics = data[target_exp_key]
    plt.figure(figsize=(10, 6))
    
    loss_types = {
        'train_recon_loss': ('Reconstruction', 'blue'),
        'train_cross_loss': ('Cross-Recon', 'orange'),
        'train_align_loss': ('Alignment (Latent)', 'red'),
        'train_vq_loss': ('VQ Commitment', 'green'),
        'train_vel_loss': ('Velocity Consistency', 'purple')
    }
    
    for key, (label, color) in loss_types.items():
        if key in metrics and metrics[key]:
            arr = np.array(metrics[key])
            if arr.size == 0 or arr.ndim != 2: continue
            
            # 计算均值和标准差
            mean_curve = smooth(np.mean(arr, axis=0))
            std_curve = np.std(arr, axis=0) # [新增] 计算标准差
            epochs = range(1, len(mean_curve) + 1)
            
            # 绘制均值线
            plt.plot(epochs, mean_curve, label=label, color=color, linewidth=2)
            # [新增] 绘制阴影区间 (方差带)
            plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, 
                             color=color, alpha=0.15)
            
    plt.title(f'Loss Breakdown (Mean ± Std) for {NAME_MAP.get(target_exp_key, target_exp_key)}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(METRIC_DIR, 'loss_breakdown.png'), dpi=150)
    print(f"Saved loss_breakdown.png")
    plt.close()

def plot_radar(data):
    summary = {}
    metrics_list = RADAR_METRICS
    
    # --- 1. 数据预清洗 (保持不变) ---
    for exp_key, metrics_dict in data.items():
        is_complete = True
        temp_vals = {}
        for k in metrics_list:
            if k not in metrics_dict or not metrics_dict[k]:
                is_complete = False; break
            arr = np.array(metrics_dict[k])
            if arr.ndim != 2 or arr.size == 0:
                is_complete = False; break
            final_val = np.mean(arr[:, -5:]) 
            if k == 'dead_code_ratio':
                if 'fsq' in exp_key or 'hybrid' in exp_key:
                    final_val = max(0.0, final_val)
            temp_vals[k] = final_val
        if is_complete:
            summary[exp_key] = temp_vals

    if not summary: 
        print("No valid data for Radar Chart.")
        return

    # --- 2. 计算所有实验的 Ratios 并寻找各轴最大值 (SOTA) ---
    baseline_key = list(summary.keys())[0]
    all_exp_ratios = [] # 用于计算 Linear Limit
    all_exp_logs = []   # 用于计算 Log Limit
    
    # 临时存储每个实验计算好的原始 ratio
    temp_ratios_storage = {}

    for name, vals in summary.items():
        base = summary[baseline_key]
        # 越小越好 (r = Base / Current)
        r1 = base['val_recon'] / (vals['val_recon'] + 1e-6)
        r2 = base['val_vel'] / (vals['val_vel'] + 1e-6)
        r3 = base['val_jerk'] / (vals['val_jerk'] + 1e-6)
        # 越大越好 (r = Current / Base)
        r4 = vals['perplexity'] / (base['perplexity'] + 1e-6)
        # 越小越好 (DCR, 加上 0.01 防止除零)
        # r5 = (base['dead_code_ratio'] + 0.01) / (vals['dead_code_ratio'] + 0.01)
        # 越小越好
        r6 = base['val_align'] / (vals['val_align'] + 1e-6)
        r7 = base['val_cross_recon'] / (vals['val_cross_recon'] + 1e-6)
        
        ratios = [r1, r2, r3, r4, r6, r7]
        temp_ratios_storage[name] = ratios
        all_exp_ratios.append(ratios)
        # 预计算 Log 映射后的值：Log2(r) + 1.0
        all_exp_logs.append([np.log2(max(0.1, r)) + 1.0 for r in ratios])

    # 寻找每个维度的最大表现 (SOTA)
    max_ratios = np.max(np.array(all_exp_ratios), axis=0)
    max_logs = np.max(np.array(all_exp_logs), axis=0)

    # 自动计算 limits (SOTA * 系数)
    auto_linear_limits = max_ratios * np.array(RADAR_COEFFS)
    auto_log_limits = max_logs * np.array(RADAR_COEFFS)

    # --- 3. 定义绘图配置 ---
    plot_configs = [
        {
            'filename': 'final_radar_chart.png',
            'log_scale': False,
            'title': 'Relative Performance (Auto Linear Scale)',
            'limits': auto_linear_limits
        },
        {
            'filename': 'final_radar_chart_log.png',
            'log_scale': True,
            'title': 'Relative Performance (Auto Log Scale)',
            'limits': auto_log_limits
        },
        {
            'filename': 'final_radar_chart_uniform.png',
            'log_scale': False,
            'title': 'Relative Performance (Uniform Linear Scale)',
            'limits': [np.max(auto_linear_limits)] * RADAR_Number
        },
        {
            'filename': 'final_radar_chart_log_uniform.png',
            'log_scale': True,
            'title': 'Relative Performance (Uniform Log Scale)',
            'limits': [np.max(auto_log_limits)] * RADAR_Number
        }
    ]

    # --- 4. 循环生成图片 ---
    for config in plot_configs:
        is_log = config['log_scale']
        current_limits = config['limits']
        PLOT_RADIUS = 3.0
        
        radar_data = {}
        for name, ratios in temp_ratios_storage.items():
            plot_vals = []
            for i, r in enumerate(ratios):
                limit = current_limits[i]
                if is_log:
                    val = np.log2(max(0.1, r)) + 1.0
                else:
                    val = r
                norm_val = (val / limit) * PLOT_RADIUS
                plot_vals.append(np.clip(norm_val, 0, PLOT_RADIUS))
            radar_data[name] = plot_vals

        # --- 5. 绘图细节 (拓宽画布防止遮挡) ---
        angles = np.linspace(0, 2 * np.pi, len(RADAR_categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # 增加 figsize 宽度，从 (6, 6) 改为 (10, 7)
        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(polar=True))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        ax.set_ylim(0, PLOT_RADIUS)
        ax.set_rgrids([0.6, 1.2, 1.8, 2.4, 3.0], labels=[], angle=0)

        for i, (name, scores) in enumerate(radar_data.items()):
            vals = scores + scores[:1]
            label = NAME_MAP.get(name, name)
            ax.plot(angles, vals, linewidth=2, label=label, color=colors[i % len(colors)])
            ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_categories)
        
        # 调整标题和图例位置，防止重合
        plt.title(config['title'], y=1.1, fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1)) 
        
        save_path = os.path.join(RADAR_DIR, config['filename'])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()



def main():
    data = load_and_aggregate()
    if not data: return

    # 1. 遍历所有全局指标画曲线图 (自动化，不需要一个个写)
    for m in ALL_METRICS:
        # 特殊处理：对 Recon 和 Align 类指标用对数坐标
        use_log = any(kw in m for kw in ['recon', 'align', 'loss'])
        plot_metric(data, m, m.replace('_', ' ').title(), 'Value', f'compare_{m}.png', log_scale=use_log)
    
    # 3. === 新增：Loss 拆解图 ===
    # 假设你的 Hybrid 实验 ID 是 'ours_hybrid' 或类似的，请根据 config 调整 key
    # 这里自动寻找名字里带 'hybrid' 的 key
    hybrid_key = next((k for k in data.keys() if 'hybrid' in k), None)
    if hybrid_key:
        plot_loss_breakdown(data, hybrid_key)
    
    # 4. 雷达图 (需要确保你的 plot_radar 函数还在)
    plot_radar(data) 

if __name__ == "__main__":
    main()