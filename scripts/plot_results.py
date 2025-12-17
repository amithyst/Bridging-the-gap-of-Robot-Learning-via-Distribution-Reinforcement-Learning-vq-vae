import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

LOG_DIR = 'results'
SAVE_DIR = 'plots'
os.makedirs(SAVE_DIR, exist_ok=True)

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

    # 1. Load all raw data
    raw_data = {} # {exp_key: [history_seed1, history_seed2]}
    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse: log_{arch}_{method}_seed_{seed}.json
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

    # 2. Align and Aggregate
    keys_to_plot = ['val_recon', 'val_vel', 'val_jerk', 'perplexity', 'dead_code_ratio']
    
    for exp_key, seed_logs in raw_data.items():
        if not seed_logs: continue
        
        # Find min epochs to handle interrupted runs
        min_len = min(len(log['val_recon']) for log in seed_logs)
        if min_len < 2: continue # Skip if too short
        
        data[exp_key] = {}
        for metric in keys_to_plot:
            # Truncate each seed to min_len
            metric_data = [log[metric][:min_len] for log in seed_logs if metric in log]
            data[exp_key][metric] = metric_data # List of Lists
            
    return data

def plot_metric(data, metric_key, title, ylabel, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    valid_plot = False
    for i, (exp_key, metrics_dict) in enumerate(data.items()):
        if metric_key not in metrics_dict: continue
        
        seeds_data = metrics_dict[metric_key]
        if not seeds_data: continue
        
        # Convert to numpy [Seeds, Epochs]
        arr = np.array(seeds_data) 
        if arr.ndim != 2: continue # Safety check
        
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        mean_curve = smooth(mean_curve)
        
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
        plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150)
        print(f"Saved {filename}")
    plt.close()

def plot_radar(data):
    # Normalize metrics for radar chart
    # We take the final epoch value (average of last 5 epochs)
    summary = {}
    
    for exp_key, metrics_dict in data.items():
        summary[exp_key] = {}
        for k in ['val_recon', 'val_vel', 'val_jerk', 'perplexity', 'dead_code_ratio']:
            arr = np.array(metrics_dict[k])
            # Mean of last 5 epochs across all seeds
            final_val = np.mean(arr[:, -5:]) 
            summary[exp_key][k] = final_val

    # Define Axis (Inverse where lower is better)
    # 1. Recon (Lower better) -> 1/x
    # 2. Vel (Lower better) -> 1/x
    # 3. Jerk (Lower better) -> 1/x
    # 4. PPL (Higher better) -> x
    # 5. DCR (Lower better) -> 1/x (Low dead code is good)
    
    if not summary: return
    baseline_key = list(summary.keys())[0]
    categories = ['Recon Quality', 'Motion Smoothness', 'Low Jerk', 'Code Usage', 'Active Codes']
    
    radar_data = {}
    for name, metrics in summary.items():
        base = summary[baseline_key]
        
        # Calculate Relative Score (Baseline / Current for Loss, Current / Baseline for Metric)
        # Add epsilon to avoid div by zero
        s1 = np.clip(base['val_recon'] / (metrics['val_recon'] + 1e-6), 0, 3)
        s2 = np.clip(base['val_vel'] / (metrics['val_vel'] + 1e-6), 0, 3)
        s3 = np.clip(base['val_jerk'] / (metrics['val_jerk'] + 1e-6), 0, 3)
        s4 = np.clip(metrics['perplexity'] / (base['perplexity'] + 1e-6), 0, 3)
        # DCR: Lower is better. 
        s5 = np.clip((base['dead_code_ratio'] + 0.01) / (metrics['dead_code_ratio'] + 0.01), 0, 3)
        
        radar_data[name] = [s1, s2, s3, s4, s5]

    # Plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
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