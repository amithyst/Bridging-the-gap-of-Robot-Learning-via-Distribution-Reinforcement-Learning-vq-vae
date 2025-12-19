import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import DualMotionVQVAE

# === Config ===


ACTION_KEYWORDS = ['walk', 'run'] 
RAW_DATA_ROOT = "./data/raw/unzipped/extended_datasets/lafan1_dataset/g1"
PROCESSED_DIR = "./data/processed"
LATENT_PLOT_DIR = 'plots/latent_space'

os.makedirs(LATENT_PLOT_DIR, exist_ok=True)

ROBOT_DIM = 29
HUMAN_DIM = 126
HIDDEN_DIM = 64
ARCH = 'resnet'
METHOD = 'hybrid'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_normalization_stats():
    """
    同时加载 Robot 和 Human 的统计量
    """
    r_mean_path = os.path.join(PROCESSED_DIR, 'mean.npy')
    r_std_path = os.path.join(PROCESSED_DIR, 'std.npy')
    
    h_mean_path = os.path.join(PROCESSED_DIR, 'human_mean.npy')
    h_std_path = os.path.join(PROCESSED_DIR, 'human_std.npy')

    # Robot Stats
    if os.path.exists(r_mean_path):
        r_mean, r_std = np.load(r_mean_path), np.load(r_std_path)
    else:
        r_mean, r_std = np.zeros(ROBOT_DIM), np.ones(ROBOT_DIM)
        
    # Human Stats
    if os.path.exists(h_mean_path):
        h_mean, h_std = np.load(h_mean_path), np.load(h_std_path)
    else:
        print("Warning: Human stats not found! Using Identity.")
        h_mean, h_std = np.zeros(HUMAN_DIM), np.ones(HUMAN_DIM)
        
    return (r_mean, r_std), (h_mean, h_std)

def slice_sequence(motion, window_size=64, stride=64):
    slices = []
    n_frames = motion.shape[0]
    for i in range(0, n_frames - window_size + 1, stride):
        slices.append(motion[i:i+window_size])
    return slices

def load_paired_data_by_action(keywords, r_stats, h_stats):
    r_mean, r_std = r_stats
    h_mean, h_std = h_stats
    
    data_dict = {k: {'robot': [], 'human': []} for k in keywords}
    search_path = os.path.join(RAW_DATA_ROOT, '**', '*.npz')
    files = glob.glob(search_path, recursive=True)
    
    print(f"Scanning files for keywords {keywords}...")
    
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
                
                # === Flatten ===
                if r_motion.ndim > 2: r_motion = r_motion.reshape(r_motion.shape[0], -1)
                if h_motion.ndim > 2: h_motion = h_motion.reshape(h_motion.shape[0], -1)
                
                # Length Check
                min_len = min(len(r_motion), len(h_motion))
                r_motion = r_motion[:min_len]
                h_motion = h_motion[:min_len]

                # === 关键修改：双边归一化 ===
                r_motion = (r_motion - r_mean) / r_std
                h_motion = (h_motion - h_mean) / h_std
                # ===========================
                
                r_slices = slice_sequence(r_motion)
                h_slices = slice_sequence(h_motion)
                
                if r_slices:
                    data_dict[found_key]['robot'].extend(r_slices)
                    data_dict[found_key]['human'].extend(h_slices)
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    final_dict = {}
    for k, v in data_dict.items():
        if len(v['robot']) > 0:
            n_samples = min(len(v['robot']), 200)
            indices = np.random.choice(len(v['robot']), n_samples, replace=False)
            
            r_tensor = torch.tensor(np.array(v['robot'])[indices], dtype=torch.float32)
            h_tensor = torch.tensor(np.array(v['human'])[indices], dtype=torch.float32)
            
            final_dict[k] = (r_tensor, h_tensor)
            print(f"Action '{k}': Collected {n_samples} paired samples")
            
    return final_dict

def get_latent_vectors(model, x, encoder_type='robot'):
    model.eval()
    latents = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(x), batch_size):
            batch = x[i : i+batch_size].to(DEVICE)
            batch = batch.permute(0, 2, 1) # (B, C, T)
            
            if encoder_type == 'robot':
                z_e = model.robot_encoder(batch)
            else:
                z_e = model.human_encoder(batch)
                
            z_pool = torch.mean(z_e, dim=2)
            latents.append(z_pool.cpu().numpy())
            
    return np.concatenate(latents, axis=0)


def main():
    os.makedirs(LATENT_PLOT_DIR, exist_ok=True)
    
    # 1. 扫描所有权重文件
    ckpts = glob.glob(os.path.join('checkpoints', '*.pth'))
    if not ckpts:
        print("No checkpoints found in 'checkpoints/'.")
        return

    # 2. 加载统计量与原始数据 (保持 numpy 格式以便动态处理)
    r_stats, h_stats = load_normalization_stats()
    data_pairs_raw = load_paired_data_by_action(ACTION_KEYWORDS, r_stats, h_stats)
    if not data_pairs_raw: return

    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        
        # 排除 FSQ/LFQ (这些离散化模型不适合直接做连续隐空间池化可视化)
        if any(k in fname.lower() for k in ['fsq', 'lfq']):
            print(f"\n>>> Skipping {fname} (Quantization type not ideal for vis)")
            continue
            
        print(f"\n>>> Analyzing Checkpoint: {fname}")
        
        try:
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
        except Exception as e:
            print(f"!!! Error loading {fname}: {e}")
            continue

        # --- 健壮性检查：过滤单编码器模型 ---
        # 如果没有 human_encoder 关键字，说明是旧的单路重建模型
        if 'human_encoder.model.0.weight' not in state_dict:
            print(f"    [Skip] {fname} is a Single-Encoder model or incompatible. Skipping.")
            continue

        # --- 动态探测参数 ---
        # 探测 Human Dimension
        h_dim_detected = state_dict['human_encoder.model.0.weight'].shape[1]
        
        # 探测 Architecture (通过检查是否有 resnet 特有的 'net' 关键字)
        is_resnet = any('net' in k for k in state_dict.keys())
        current_arch = 'resnet' if is_resnet else 'simple'
        
        # 探测 Method
        current_method = 'ema'
        if any('quantizer.layers' in k for k in state_dict.keys()): current_method = 'rvq'
        elif any('quantizer.fsq' in k for k in state_dict.keys()): current_method = 'hybrid'
        # 补充文件名判断
        for m in ['rvq', 'hybrid', 'ema', 'standard']:
            if m in fname.lower():
                current_method = m
                break

        print(f"    [Detected] Config -> Dim: {h_dim_detected} | Arch: {current_arch} | Method: {current_method}")

        # --- 实例化适配的模型 ---
        try:
            model = DualMotionVQVAE(
                human_input_dim=h_dim_detected,
                robot_input_dim=ROBOT_DIM,
                hidden_dim=HIDDEN_DIM, 
                arch=current_arch, 
                method=current_method
            ).to(DEVICE)
            
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except Exception as e:
            print(f"    [Error] Model initialization failed: {e}")
            continue

        # --- 提取特征 ---
        X = []; labels = []; domains = []
        for action, (r_tensor, h_tensor) in data_pairs_raw.items():
            # 核心：检查当前提取的数据维度是否匹配该模型的训练维度
            if h_tensor.shape[-1] != h_dim_detected:
                print(f"    [Skip Action] Data dim {h_tensor.shape[-1]} != Model expected {h_dim_detected}")
                continue

            z_r = get_latent_vectors(model, r_tensor, 'robot')
            z_h = get_latent_vectors(model, h_tensor, 'human')
            
            X.append(z_r); X.append(z_h)
            labels.extend([action] * (len(z_r) + len(z_h)))
            domains.extend(['Robot'] * len(z_r) + ['Human'] * len(z_h))
            
        if not X: 
            print("    [Warning] No compatible data found for this model's dimensions.")
            continue
            
        X = np.concatenate(X, axis=0)

        # --- t-SNE 与绘图 ---
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        unique_labels = np.unique(labels)
        for act in unique_labels:
            mask = np.array(labels) == act
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=act, alpha=0.5, s=15)
        plt.title(f'Action Clustering ({current_method})')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        mask_r = np.array(domains) == 'Robot'
        plt.scatter(X_tsne[mask_r, 0], X_tsne[mask_r, 1], c='blue', label='Robot', alpha=0.3, s=15)
        mask_h = np.array(domains) == 'Human'
        plt.scatter(X_tsne[mask_h, 0], X_tsne[mask_h, 1], c='red', label='Human', alpha=0.3, s=15)
        plt.title('Domain Alignment (Overlap Check)')
        plt.legend()
        
        plt.suptitle(f"File: {fname}\nDim: {h_dim_detected} | Arch: {current_arch}")
        save_path = os.path.join(LATENT_PLOT_DIR, f"align_{fname.replace('.pth','')}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f">>> Successfully saved: {save_path}")

if __name__ == "__main__":
    main()