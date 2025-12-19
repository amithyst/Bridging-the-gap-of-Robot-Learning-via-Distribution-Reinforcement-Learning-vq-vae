import json
import numpy as np
import os
import glob
import sys

# 配置
LOG_DIR = 'results'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.experiment_config import EXPERIMENTS

# 1. 修改配置部分的名称映射
ORDERED_EXP = [(exp['id'], exp['name']) for exp in EXPERIMENTS]

# 定义指标、显示名称、格式化精度、是否为百分比
# === 修改：新增 Align Loss 和 Cross Recon ===
METRICS = [
    ('val_recon', r'Recon $\downarrow$', 4, False),
    ('val_cross_recon', r'Cross $\downarrow$', 4, False), # 新增：Retargeting 误差
    ('val_align', r'Align $\downarrow$', 4, False),       # 新增：对齐误差
    ('val_vel', r'Vel $\downarrow$', 4, False),
    ('val_jerk', r'Jerk $\downarrow$', 4, False),
    ('perplexity', r'PPL $\uparrow$', 1, False),
    ('dead_code_ratio', r'DCR \% $\downarrow$', 1, True) 
]

def get_final_metrics(json_data, metric_key, n_epochs=5):
    """
    获取单个日志文件中，最后 n_epochs 的平均值
    """
    if metric_key not in json_data or not json_data[metric_key]:
        return None
    
    values = json_data[metric_key]
    data_slice = values[-n_epochs:] if len(values) >= n_epochs else values
    return np.mean(data_slice)

def load_data():
    # 存储结构: data[exp_key][metric_key] = [seed1_val, seed2_val, ...]
    data = {exp_key: {m[0]: [] for m in METRICS} for exp_key, _ in ORDERED_EXP}
    
    files = glob.glob(os.path.join(LOG_DIR, "log_*.json"))
    
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            parts = fname.replace("log_", "").replace(".json", "").split("_seed_")
            if len(parts) != 2: continue
            
            exp_key = parts[0]
            if exp_key not in data: continue 

            with open(fpath, 'r') as f:
                log = json.load(f)
            
            for m_key, _, _, is_percent in METRICS:
                val = get_final_metrics(log, m_key)
                
                # 特殊处理 FSQ/Hybrid 的 Dead Code Ratio
                if m_key == 'dead_code_ratio':
                    if 'fsq' in exp_key and val is None: 
                        val = 0.0 
                    elif val is not None and val < 0:
                        val = 0.0 

                if val is not None:
                    if is_percent and not ('fsq' in exp_key and m_key == 'dead_code_ratio'): 
                        val *= 100.0
                    data[exp_key][m_key].append(val)
                    
        except Exception as e:
            print(f"Error loading {fname}: {e}", file=sys.stderr)
            
    return data

def generate_latex():
    data = load_data()
    
    print("-" * 60)
    print("LaTeX Table Code (Copy below):")
    print("-" * 60)
    print()
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of different VQ-VAE variants (Expanded Metrics).}")
    print(r"\label{tab:results}")
    
    # 动态生成列格式
    col_def = "l" + " c" * len(METRICS)
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(r"\toprule")
    
    headers = ["Method"] + [m[1] for m in METRICS]
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")
    
    for exp_key, exp_name in ORDERED_EXP:
        row_str = [exp_name]
        
        for m_key, _, decimal, _ in METRICS:
            values = data[exp_key][m_key]
            
            if not values:
                row_str.append("N/A")
            else:
                mean = np.mean(values)
                std = np.std(values)
                val_str = f"{mean:.{decimal}f} $\\pm$ {std:.{decimal}f}"
                row_str.append(val_str)
        
        print(" & ".join(row_str) + r" \\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

if __name__ == "__main__":
    generate_latex()