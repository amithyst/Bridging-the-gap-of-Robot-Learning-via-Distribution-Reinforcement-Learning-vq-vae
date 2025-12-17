import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

LOG_DIR = 'results'
SAVE_DIR = 'plots'

# 实验名称映射 (用于图例)
NAME_MAP = {
    'simple_ema': 'Baseline (Simple)',
    'resnet_ema': 'Proposed (ResNet)',
    'resnet_rvq': 'Advanced (RVQ)',
    'resnet_fsq': 'SOTA (FSQ)',
    'resnet_lfq': 'SOTA (LFQ)'
}

def smooth(scalars, weight=0.8):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + point * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def load_and_aggregate():
    # 数据结构: data[exp_key] = {'recon': [[seed1], [seed2]...], 'vel': ...}
    data = {}
    
    files = glob.glob(os.path.join(LOG_DIR, "log_*.json"))
    if not files:
        print("No log files found in results/")
        return {}

    for fpath in files:
        # 解析文件名: log_{arch}_{method}_seed_{seed}.json
        fname = os.path.basename(fpath)
        parts = fname.replace("log_", "").replace(".json", "").split("_seed_")
        exp_key = parts[0] # e.g., resnet_ema
        
        with open(fpath, 'r') as f:
            log = json.load(f)
            
        if exp_key not in data:
            data[exp_key] = {'val_recon': [], 'val_vel': [], 'perplexity': []}
            
        data[exp_key]['val_recon'].append(log['val_recon'])
        data[exp_key]['val_vel'].append(log['val_vel'])
        data[exp_key]['perplexity'].append(log['perplexity'])
        
    return data

def plot_metric(data, metric_key, title, ylabel, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (exp_key, seeds_data) in enumerate(data.items()):
        # seeds_data: List of lists (Seed x Epochs)
        arr = np.array(seeds_data) # [Seeds, Epochs]
        
        # Calculate Mean and Std
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        # Smooth for cleaner plot
        mean_curve = smooth(mean_curve)
        
        label = NAME_MAP.get(exp_key, exp_key)
        color = colors[i % len(colors)]
        
        epochs = range(1, len(mean_curve) + 1)
        
        plt.plot(epochs, mean_curve, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, 
                         color=color, alpha=0.15)
        
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150)
    print(f"Saved {filename}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    data = load_and_aggregate()
    
    if not data:
        return

    # 1. Recon Loss
    plot_metric(data, 'val_recon', 'Reconstruction Error (Mean ± Std)', 'MSE Loss', 'compare_recon_loss.png', log_scale=True)
    
    # 2. Velocity Loss
    plot_metric(data, 'val_vel', 'Motion Jitter / Velocity Error', 'Velocity MSE', 'compare_velocity_loss.png')
    
    # 3. Perplexity
    plot_metric(data, 'perplexity', 'Codebook Usage (Perplexity)', 'Perplexity', 'compare_perplexity.png')

if __name__ == "__main__":
    main()