import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

from models.vqvae import DualMotionVQVAE

def load_stats(data_dir, device):
    try:
        mean = np.load(os.path.join(data_dir, 'mean.npy'))
        std = np.load(os.path.join(data_dir, 'std.npy'))
    except FileNotFoundError:
        print("[WARN] Stats not found, using identity normalization.")
        return torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
    return torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device)

def reconstruct_long_sequence(model, full_seq, window_size, step_size, mean, std, device):
    """
    使用滑动窗口+平均策略重建长序列
    """
    seq_len, dim = full_seq.shape
    
    if seq_len <= window_size:
        return None 

    # 结果缓冲区
    recon_buffer = torch.zeros((seq_len, dim)).to(device)
    count_buffer = torch.zeros((seq_len, 1)).to(device)
    
    model.eval()
    
    # 策略：从 0 开始滑动，直到触底
    current_idx = 0
    while current_idx + window_size <= seq_len:
        # 1. 取出窗口数据
        chunk = full_seq[current_idx : current_idx + window_size]
        
        # 2. 归一化 & 推理
        input_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device) # (1, W, D)
        norm_input = (input_tensor - mean) / std
        
        with torch.no_grad():
            outputs = model(x_robot=norm_input)
            recon_norm = outputs['robot']['recon'] 
        
        # 3. 反归一化
        recon_denorm = recon_norm * std + mean
        recon_chunk = recon_denorm.squeeze(0) # (W, D)
        
        # 4. 累加到缓冲区
        recon_buffer[current_idx : current_idx + window_size] += recon_chunk
        count_buffer[current_idx : current_idx + window_size] += 1.0
        
        current_idx += step_size
        
    # 处理末尾
    if current_idx < seq_len:
        last_start = seq_len - window_size
        chunk = full_seq[last_start : ]
        input_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        norm_input = (input_tensor - mean) / std
        with torch.no_grad():
            outputs = model(x_robot=norm_input)
            recon_norm = outputs['robot']['recon']
        recon_denorm = recon_norm * std + mean
        recon_chunk = recon_denorm.squeeze(0)
        
        recon_buffer[last_start:] += recon_chunk
        count_buffer[last_start:] += 1.0

    # 5. 取平均
    count_buffer[count_buffer == 0] = 1.0 
    final_recon = recon_buffer / count_buffer
    
    return final_recon.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Batch export motions from VQ-VAE (Supports Long Sequence).")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='./motions', help='Where to save .npy files')
    
    # 模型配置
    parser.add_argument('--arch', type=str, default='transformer', help='Model architecture')
    parser.add_argument('--method', type=str, default='hybrid', help='Quantization method')
    parser.add_argument('--window', type=int, default=10, help='Model Window size.')
    
    # 批量与长序列参数
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of sample in dataset')
    parser.add_argument('--num_samples', type=int, default=1, help='How many samples to export')
    parser.add_argument('--step_size', type=int, default=None, help='Stride for sliding window. Default = window // 2')
    # [新增 1] 添加 max_len 参数，默认为 -1 (表示不限制)
    parser.add_argument('--max_len', type=int, default=-1, help='Max frames to export for long sequence. -1 means full length.')
    
    args = parser.parse_args()

    if args.step_size is None:
        args.step_size = max(1, args.window // 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 预加载 Checkpoint 以探测维度
    print(f"[INFO] Inspecting checkpoint: {args.ckpt}")
    try:
        raw_state_dict = torch.load(args.ckpt, map_location=device)
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        return

    if 'model_state_dict' in raw_state_dict:
        raw_state_dict = raw_state_dict['model_state_dict']
        
    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('module.', '')] = v

    detected_h_dim = 126
    detected_r_dim = 29
    
    if 'human_encoder.input_proj.weight' in state_dict:
        detected_h_dim = state_dict['human_encoder.input_proj.weight'].shape[1]
    if 'robot_encoder.input_proj.weight' in state_dict:
        detected_r_dim = state_dict['robot_encoder.input_proj.weight'].shape[1]
    
    print(f"[INFO] Init Model: Arch={args.arch}, Win={args.window}, Stride={args.step_size}, R={detected_r_dim}")
    
    # 2. 初始化模型
    model = DualMotionVQVAE(
        human_input_dim=detected_h_dim, 
        robot_input_dim=detected_r_dim,
        hidden_dim=64, 
        arch=args.arch,
        method=args.method,
        window_size=args.window 
    ).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[WARN] Loading failed: {e}")
        
    model.eval()

    # 3. 加载数据 (优先加载完整序列)
    mean, std = load_stats(args.data_dir, device)

    raw_data_path = os.path.join(args.data_dir, 'g1_train_full_raw.npy')
    sliced_data_path = os.path.join(args.data_dir, 'g1_train.npy')
    
    raw_data = []
    is_full_sequence = False
    
    if os.path.exists(raw_data_path):
        print(f"[INFO] Found FULL sequence data: {raw_data_path}")
        # 加载 object array, 里面是 list of arrays
        raw_data = np.load(raw_data_path, allow_pickle=True)
        is_full_sequence = True
    elif os.path.exists(sliced_data_path):
        print(f"[WARN] Full sequence data not found. Falling back to SLICED data: {sliced_data_path}")
        raw_data = np.load(sliced_data_path)
        is_full_sequence = False
    else:
        print(f"[ERROR] No data found in {args.data_dir}")
        return

    print(f"[INFO] Data loaded. Total samples: {len(raw_data)}")
    
    # 4. 批量处理
    end_idx = args.start_idx + args.num_samples
    print(f"[INFO] Exporting samples {args.start_idx} to {end_idx}...")
    
    for i in tqdm(range(args.start_idx, end_idx)):
        if i >= len(raw_data): 
            print(f"[WARN] Index {i} out of bounds (Total: {len(raw_data)}). Stopping.")
            break
            
        full_sample = raw_data[i] # (T, D)

        # [新增 2] 如果设置了 max_len 且数据超长，进行截断
        if args.max_len > 0 and full_sample.shape[0] > args.max_len:
            full_sample = full_sample[:args.max_len]

        seq_len = full_sample.shape[0]

        gt_filename = f"idx{i}_gt.npy"
        
        # === 分支 A: 长序列 (且使用 full sequence 数据源) ===
        if seq_len > args.window and is_full_sequence:
            print(f"  -> Sequence {i}: Length {seq_len} (Full). Sliding Window Reconstruction.")
            recon_np = reconstruct_long_sequence(model, full_sample, args.window, args.step_size, mean, std, device)
            
            if recon_np is not None:
                suffix = f"{args.arch}_FullSeq_W{args.window}_idx{i}"
                np.save(os.path.join(args.output_dir, gt_filename), full_sample)
                np.save(os.path.join(args.output_dir, f"recon_{suffix}.npy"), recon_np)
                continue

        # === 分支 B: 短序列 或 切片数据 ===
        if seq_len < args.window:
            print(f"[WARN] Sample {i} length {seq_len} < window {args.window}, skipping.")
            continue
            
        # 截断只取前 window 帧
        sample = full_sample[:args.window]
        
        input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
        norm_input = (input_tensor - mean) / std

        with torch.no_grad():
            outputs = model(x_robot=norm_input)
            recon_norm = outputs['robot']['recon'] 

        recon_denorm = recon_norm * std + mean
        recon_np = recon_denorm.squeeze(0).cpu().numpy()
        
        suffix = f"{args.arch}_W{args.window}_{args.method}_idx{i}"
        np.save(os.path.join(args.output_dir, gt_filename), sample)
        np.save(os.path.join(args.output_dir, f"recon_{suffix}.npy"), recon_np)

    print(f"[SUCCESS] Saved to {args.output_dir}")

if __name__ == '__main__':
    main()