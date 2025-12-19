# scripts/deployment/export_motion.py
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径，以便导入 models
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

from models.vqvae import DualMotionVQVAE
from models.experiment_config import EXPERIMENTS

def load_stats(data_dir, device):
    mean = np.load(os.path.join(data_dir, 'mean.npy'))
    std = np.load(os.path.join(data_dir, 'std.npy'))
    return torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID defined in config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./motions', help='Where to save .npy files')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to export')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 配置模型
    config = next((item for item in EXPERIMENTS if item['id'] == args.exp_id), None)
    if not config:
        raise ValueError(f"Experiment ID {args.exp_id} not found in config.")

    print(f"Loading model: {config['name']} ({config['arch']}+{config['method']})")
    model = DualMotionVQVAE(
        human_input_dim=126, # 根据你的数据调整
        robot_input_dim=29,
        hidden_dim=64,
        arch=config['arch'],
        method=config['method']
    ).to(device)
    
    # 2. 加载权重
    # 注意：如果你的权重里包含 'model_state_dict' 键，请自行调整
    state_dict = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 加载数据 & 统计量
    print("Loading data and stats...")
    data_path = os.path.join(args.data_dir, 'g1_train.npy')
    raw_data = np.load(data_path) # (N, 64, 29)
    mean, std = load_stats(args.data_dir, device)

    # 4. 提取样本 & 归一化
    sample = raw_data[args.sample_idx] # (64, 29)
    input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device) # (1, 64, 29)
    # 手动归一化 (模拟 Dataset 行为)
    norm_input = (input_tensor - mean) / std

    # 5. 推理
    with torch.no_grad():
        outputs = model(x_robot=norm_input)
        recon_norm = outputs['robot']['recon'] # (1, 64, 29)

    # 6. 反归一化
    recon_denorm = recon_norm * std + mean
    
    # 转回 Numpy
    gt_np = sample # 原始数据本身就是未归一化的 (process_data.py 保存的是 raw value 还是 normalized? 
                   # 根据代码看 process_data 保存的是 robot_data，没有归一化，mean/std 是另存的。
                   # 所以这里 sample 就是真实角度)
    recon_np = recon_denorm.squeeze(0).cpu().numpy()

    # 7. 保存
    gt_path = os.path.join(args.output_dir, f"demo_gt_idx{args.sample_idx}.npy")
    recon_path = os.path.join(args.output_dir, f"demo_recon_{args.exp_id}_idx{args.sample_idx}.npy")
    
    np.save(gt_path, gt_np)
    np.save(recon_path, recon_np)
    
    print(f"Done! Saved to:\n  {gt_path}\n  {recon_path}")
    print(f"Shape: {recon_np.shape}")

if __name__ == '__main__':
    main()