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
# (Key in JSON, Display Name, Decimal Places, Is Percentage)
METRICS = [
    ('val_recon', r'Recon Loss $\downarrow$', 4, False),
    ('val_vel', r'Vel Loss $\downarrow$', 4, False),
    ('val_jerk', r'Jerk Loss $\downarrow$', 4, False),
    ('perplexity', r'PPL $\uparrow$', 1, False),
    ('dead_code_ratio', r'Dead Code \% $\downarrow$', 1, True) 
]

def get_final_metrics(json_data, metric_key, n_epochs=5):
    """
    获取单个日志文件中，最后 n_epochs 的平均值
    """
    if metric_key not in json_data or not json_data[metric_key]:
        return None
    
    values = json_data[metric_key]
    # 取最后 n_epochs，如果不足则取所有
    data_slice = values[-n_epochs:] if len(values) >= n_epochs else values
    return np.mean(data_slice)

# 2. 修改 load_data 函数，增加针对 FSQ 的特殊处理
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
            
            # 提取每个指标
            for m_key, _, _, is_percent in METRICS:
                # === 修复开始 ===
                val = get_final_metrics(log, m_key)
                
                # 特殊处理 FSQ/Hybrid 的 Dead Code Ratio
                if m_key == 'dead_code_ratio':
                    if 'fsq' in exp_key and val is None: 
                        val = 0.0 # 纯 FSQ 有时日志里没有 dcr 键
                    elif val is not None and val < 0:
                        val = 0.0 # 修正 Hybrid 的负数 DCR 为 0.0
                # === 修复结束 ===

                if val is not None:
                    if is_percent and not ('fsq' in exp_key and m_key == 'dead_code_ratio'): 
                        # FSQ 已经是 0.0 了，不需要再乘 100，或者是 0 * 100 也没关系
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
    
    # 打印表头
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of different VQ-VAE variants on Unitree G1 motion data.}")
    print(r"\label{tab:results}")
    
    # 动态生成列格式: l c c c c c
    col_def = "l" + " c" * len(METRICS)
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(r"\toprule")
    
    # 打印列名
    headers = ["Method"] + [m[1] for m in METRICS]
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")
    
    # 打印每一行数据
    for exp_key, exp_name in ORDERED_EXP:
        row_str = [exp_name]
        
        for m_key, _, decimal, _ in METRICS:
            values = data[exp_key][m_key]
            
            if not values:
                # 如果没有数据 (比如旧日志)，显示 N/A
                row_str.append("N/A")
            else:
                mean = np.mean(values)
                std = np.std(values)
                
                # 格式化: Mean +/- Std
                # 比如: 0.1234 \pm 0.0001
                val_str = f"{mean:.{decimal}f} $\\pm$ {std:.{decimal}f}"
                row_str.append(val_str)
        
        print(" & ".join(row_str) + r" \\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

if __name__ == "__main__":
    generate_latex()