import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import argparse
from scipy.spatial.transform import Rotation as R

# 引入项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.vqvae import DualMotionVQVAE

def compute_6d_rotation(data):
    """将 (T, J*3) 或 (T, J*4) 转为 (T, J*6)"""
    T = data.shape[0]
    total_features = data.size // T
    
    if total_features % 3 == 0: # Axis-Angle
        data_flat = data.reshape(-1, 3)
        rot_mats = R.from_rotvec(data_flat).as_matrix()
        J = total_features // 3
    elif total_features % 4 == 0: # Quaternion
        data_flat = data.reshape(-1, 4)
        rot_mats = R.from_quat(data_flat).as_matrix()
        J = total_features // 4
    else:
        if total_features % 6 == 0: return data.reshape(T, -1)
        raise ValueError(f"Unknown input feature dim: {total_features}")

    rot_6d_flat = rot_mats[:, :, :2].reshape(-1, 6)
    return rot_6d_flat.reshape(T, J * 6)

def slice_sequence(motion, window_size, stride):
    slices = []
    n_frames = motion.shape[0]
    if n_frames < window_size: return []
    for i in range(0, n_frames - window_size + 1, stride):
        slices.append(motion[i:i+window_size])
    return slices

def load_normalization_stats(processed_dir, r_dim, h_dim):
    try:
        r_mean = np.load(os.path.join(processed_dir, 'mean.npy'))
        r_std = np.load(os.path.join(processed_dir, 'std.npy'))
    except:
        print("Warning: Robot stats not found, using identity.")
        r_mean, r_std = np.zeros(r_dim), np.ones(r_dim)

    try:
        h_mean = np.load(os.path.join(processed_dir, 'human_mean.npy'))
        h_std = np.load(os.path.join(processed_dir, 'human_std.npy'))
    except:
        # 如果没有专门的 human stats，通常在 Retargeting 任务中可能复用 robot 的或单独处理
        # 这里给个默认值防止报错
        print("Warning: Human stats not found, using identity.")
        h_mean, h_std = np.zeros(h_dim), np.ones(h_dim)
        
    return (r_mean, r_std), (h_mean, h_std)

def load_paired_data_by_action(keywords, raw_root, window_size, r_stats, h_stats):
    """
    按动作关键词加载数据，用于可视化分类效果
    """
    r_mean, r_std = r_stats
    h_mean, h_std = h_stats
    
    data_dict = {k: {'robot': [], 'human': []} for k in keywords}
    search_path = os.path.join(raw_root, '**', '*.npz')
    files = glob.glob(search_path, recursive=True)
    
    print(f"Scanning for actions {keywords}...")
    
    for f in files:
        fname = os.path.basename(f).lower()
        found_key = None
        for k in keywords:
            if k in fname:
                found_key = k
                break
        
        if found_key:
            try:
                content = np.load(f, allow_pickle=True)
                if 'joint_pos' not in content or 'smplx_pose_body' not in content: continue
                
                r_motion = content['joint_pos'] 
                h_motion = content['smplx_pose_body']
                
                if r_motion.ndim > 2: r_motion = r_motion.reshape(r_motion.shape[0], -1)
                h_motion = compute_6d_rotation(h_motion)
                
                min_len = min(len(r_motion), len(h_motion))
                r_motion = r_motion[:min_len]
                h_motion = h_motion[:min_len]

                # Normalize
                r_motion = (r_motion - r_mean) / r_std
                h_motion = (h_motion - h_mean) / h_std
                
                # Slice
                stride = max(1, window_size // 2) # 50% overlap
                r_slices = slice_sequence(r_motion, window_size, stride)
                h_slices = slice_sequence(h_motion, window_size, stride)
                
                if r_slices:
                    data_dict[found_key]['robot'].extend(r_slices)
                    data_dict[found_key]['human'].extend(h_slices)
            except Exception as e:
                # print(f"Error loading {fname}: {e}")
                pass

    final_dict = {}
    for k, v in data_dict.items():
        if len(v['robot']) > 0:
            # 限制采样数，防止 t-SNE 跑太久
            n_samples = min(len(v['robot']), 300)
            indices = np.random.choice(len(v['robot']), n_samples, replace=False)
            
            r_tensor = torch.tensor(np.array(v['robot'])[indices], dtype=torch.float32)
            h_tensor = torch.tensor(np.array(v['human'])[indices], dtype=torch.float32)
            
            final_dict[k] = (r_tensor, h_tensor)
            print(f"  Action '{k}': Collected {n_samples} samples.")
            
    return final_dict

def get_latent_vectors(model, x, encoder_type='robot', device='cuda'):
    """
    获取隐变量。
    关键修改：对于 ResNet (T>1)，执行 Flatten 操作而非 Mean Pooling。
    对于 Transformer (T=1)，直接 Squeeze。
    """
    model.eval()
    latents = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(x), batch_size):
            batch = x[i : i+batch_size].to(device)
            # Input: (B, T, C) -> Permute to (B, C, T) for model
            batch = batch.permute(0, 2, 1) 
            
            if encoder_type == 'robot':
                z_e = model.robot_encoder(batch) # Output: (B, Hidden, T')
            else:
                z_e = model.human_encoder(batch) # Output: (B, Hidden, T')
            
            # === 核心修改逻辑 ===
            # Transformer: z_e is (B, Hidden, 1) -> Squeeze to (B, Hidden)
            if z_e.shape[2] == 1:
                z_flat = z_e.squeeze(2)
            else:
                # ResNet: z_e is (B, Hidden, 16) -> Flatten to (B, Hidden * 16)
                # 这样保留了时空特征，一个点代表“一条完整的时空轨迹”
                z_flat = z_e.reshape(z_e.size(0), -1)
                
            latents.append(z_flat.cpu().numpy())
            
    return np.concatenate(latents, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=64, help='Sequence length (Must match training!)')
    parser.add_argument('--filter', type=str, default='final', help='Filter checkpoint filenames')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 指向 raw data 用于获取带标签的动作数据
    RAW_DATA = "./data/raw/unzipped/extended_datasets/lafan1_dataset/g1" 
    PROCESSED = "./data/processed"
    PLOT_DIR = 'plots/latent_space'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    ckpts = glob.glob(os.path.join('checkpoints', '*.pth'))
    if args.filter:
        ckpts = [c for c in ckpts if args.filter in c]
    
    if not ckpts:
        print(f"No checkpoints found matching '{args.filter}'.")
        return

    # 1. Load Data
    # 假设 Human 维度 126 (6D rot), Robot 维度 29
    r_stats, h_stats = load_normalization_stats(PROCESSED, 29, 126)
    
    # 定义想看的动作类别
    # 定义想看的动作类别 (根据你的文件名扩展)
    actions = ['walk', 'run', 'jump', 'dance', 'fight', 'sprint', 'fall']
    data_pairs = load_paired_data_by_action(actions, RAW_DATA, args.window, r_stats, h_stats)
    
    if not data_pairs: 
        print("No paired data found via keywords. Check raw data path.")
        return

    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        print(f"\n>>> Analyzing: {fname}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            config = checkpoint.get('config', {}) # 尝试获取训练配置
            
            # 兼容 DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # --- Auto Detect Dimensions from weights ---
        if 'human_encoder.input_proj.weight' in state_dict: # Transformer
            h_dim = state_dict['human_encoder.input_proj.weight'].shape[1]
            r_dim = state_dict['robot_encoder.input_proj.weight'].shape[1]
            arch = 'transformer'
        elif 'human_encoder.model.0.weight' in state_dict: # ResNet/Simple
            h_dim = state_dict['human_encoder.model.0.weight'].shape[1]
            r_dim = state_dict['robot_encoder.model.0.weight'].shape[1]
            arch = config.get('arch', 'resnet')
        else:
            print("Unknown architecture.")
            continue
            
        # Detect Method
        method = config.get('method', 'hybrid') # Default to hybrid if not found
        
        print(f"    Config: H_dim={h_dim}, R_dim={r_dim}, Arch={arch}, Method={method}")

        # --- Init Model ---
        try:
            model = DualMotionVQVAE(
                human_input_dim=h_dim,
                robot_input_dim=r_dim,
                hidden_dim=config.get('hidden_dim', 64), 
                arch=arch,
                method=method
            ).to(DEVICE)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"    Model init failed: {e}. Check config consistency.")
            continue

        # --- Extract Latents ---
        X, labels, domains = [], [], []
        
        for action, (r_tensor, h_tensor) in data_pairs.items():
            # Check Dim
            if h_tensor.shape[-1] != h_dim:
                print(f"    Skipping {action} (dim mismatch: data {h_tensor.shape[-1]} != model {h_dim})")
                continue

            z_r = get_latent_vectors(model, r_tensor, 'robot', DEVICE)
            z_h = get_latent_vectors(model, h_tensor, 'human', DEVICE)
            
            X.append(z_r); X.append(z_h)
            # Create labels
            labels.extend([action] * (len(z_r) + len(z_h)))
            domains.extend(['Robot'] * len(z_r) + ['Human'] * len(z_h))
            
        if not X: continue
        X = np.concatenate(X, axis=0)
        
        # --- t-SNE ---
        print(f"    Running t-SNE on shape {X.shape}...")
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # --- Plotting ---
        # 图 1: Domain Alignment (Robot vs Human)
        plt.figure(figsize=(5, 5))
        
        mask_r = np.array(domains) == 'Robot'
        mask_h = np.array(domains) == 'Human'
        
        # 绘制 Robot (蓝色)
        plt.scatter(X_tsne[mask_r, 0], X_tsne[mask_r, 1], c='blue', label='Robot', alpha=0.2, s=20, marker='o', edgecolors='none')
        # 绘制 Human (红色)
        plt.scatter(X_tsne[mask_h, 0], X_tsne[mask_h, 1], c='red', label='Human', alpha=0.2, s=20, marker='x')
        
        plt.title(f'Domain Alignment: {arch}+{method}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        save_name_align = f"align_{fname.replace('.pth','')}.png"
        plt.savefig(os.path.join(PLOT_DIR, save_name_align), dpi=150)
        plt.close()
        
        # 图 2: Action Distribution (不同动作分布)
        plt.figure(figsize=(6, 5)) # 稍微宽一点，以便在右侧放置 Legend
        
        unique_actions = np.unique(labels)
        # 动态选择颜色映射
        cmap = plt.get_cmap('tab10') if len(unique_actions) <= 10 else plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(unique_actions)))
        
        for i, action in enumerate(unique_actions):
            mask = np.array(labels) == action
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[i], label=action, alpha=0.5, s=20, edgecolors='none')
            
        plt.title(f'Action Distribution: {arch}+{method}')
        # Legend 放外面防止遮挡散点
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        save_name_action = f"action_{fname.replace('.pth','')}.png"
        plt.savefig(os.path.join(PLOT_DIR, save_name_action), dpi=150)
        plt.close()

        print(f"    Saved: {save_name_align} & {save_name_action}")

if __name__ == "__main__":
    main()