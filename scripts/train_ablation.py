import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
import sys
import os
import json
import random
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import DualMotionVQVAE
from models.experiment_config import EXPERIMENTS

# === Config ===
SEEDS = [42, 2025] 
BATCH_SIZE = 256
EPOCHS = 400
LEARNING_RATE = 2e-4
HIDDEN_DIM = 64
GPU_Work_Multipliers = 3
LOG_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'

# === Loss Weights ===
LAMBDA_RECON = 1.0
LAMBDA_VQ    = 1.0
LAMBDA_VEL   = 0.5
LAMBDA_CROSS = 5.0   # 强迫 Human 输入能重构出 Robot 动作
LAMBDA_ALIGN = 100.0  # 强迫 Latent Space 强制对齐
TEMPERATURE = 0.07    # [新增] 用于 InfoNCE

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_jerk_loss(real, recon):
    # (B, T, C)
    if real.shape[1] < 4: return torch.tensor(0.0).to(real.device)
    real_jerk = torch.diff(real, n=3, dim=1)
    recon_jerk = torch.diff(recon, n=3, dim=1)
    return F.mse_loss(recon_jerk, real_jerk)

def info_nce_loss(z_h, z_r, temperature=0.07):
    # 对特征做 L2 归一化，将 MSE 转化为余弦相似度相关的对齐
    z_h = F.normalize(z_h, dim=-1)
    z_r = F.normalize(z_r, dim=-1)
    
    # 计算余弦相似度矩阵 (B, B)
    logits = torch.matmul(z_h, z_r.T) / temperature
    labels = torch.arange(z_h.size(0)).to(z_h.device)
    
    # 双向对比：Human 找 Robot，Robot 找 Human
    loss_h = F.cross_entropy(logits, labels)
    loss_r = F.cross_entropy(logits.T, labels)
    return (loss_h + loss_r) / 2

def load_paired_data():
    p_root = os.path.join('data', 'processed')
    r_path = os.path.join(p_root, 'g1_train.npy')
    h_path = os.path.join(p_root, 'human_train.npy')
    
    if not os.path.exists(r_path) or not os.path.exists(h_path):
        print("Error: 数据文件缺失，请先运行 process_data.py")
        return None, None, 29, 63 

    r_data = np.load(r_path).astype(np.float32)
    h_data = np.load(h_path).astype(np.float32)
    
    min_n = min(len(r_data), len(h_data))
    r_data = r_data[:min_n]
    h_data = h_data[:min_n]
    
    robot_dim = r_data.shape[-1]
    human_dim = h_data.shape[-1]
    
    dataset = TensorDataset(torch.from_numpy(r_data), torch.from_numpy(h_data))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False), \
           robot_dim, human_dim

def train_seed(config, seed, train_loader, val_loader, robot_dim, human_dim, device):
    set_seed(seed)
    print(f"  > [Device: {device}] Seed {seed} | {config['name']}")
    
    model = DualMotionVQVAE(
        robot_input_dim=robot_dim,
        human_input_dim=human_dim,
        hidden_dim=HIDDEN_DIM, 
        arch=config['arch'],
        method=config['method']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # === 修改：扩充 History 记录详细 Loss ===
    history = {
        'train_loss': [],
        'train_recon_loss': [], # 新增
        'train_cross_loss': [], # 新增
        'train_align_loss': [], # 新增
        'train_vq_loss': [],    # 新增
        'train_vel_loss': [],   # 新增
        'val_recon': [],      
        'val_cross_recon': [], 
        'val_align': [],        # 新增：验证集对齐损失
        'val_vel': [], 
        'val_jerk': [], 
        'perplexity': [], 
        'dead_code_ratio': []
    }
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        t_loss = 0
        # === 新增累加器 ===
        t_recon_acc = 0
        t_cross_acc = 0
        t_align_acc = 0
        t_vq_acc = 0    # 新增
        t_vel_acc = 0   # 新增
        
        for batch in train_loader:
            x_r = batch[0].to(device)
            x_h = batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(x_robot=x_r, x_human=x_h)
            
            # Robot Self-Recon
            out_r = outputs['robot']
            loss_recon = F.mse_loss(out_r['recon'], x_r)
            
            # Velocity Loss
            loss_vel = F.mse_loss(out_r['recon'][:,1:]-out_r['recon'][:,:-1], x_r[:,1:]-x_r[:,:-1])
            
            # Alignment & Cross
            z_e_r = out_r['z_e']
            z_e_h = outputs['human']['z_e']
            # 对时间维度取平均池化 (同 t-SNE 逻辑)，然后计算对比损失
            z_e_r_pool = torch.mean(z_e_r, dim=2)
            z_e_h_pool = torch.mean(z_e_h, dim=2)
            
            # 混合对齐：MSE (点) + InfoNCE (分布)
            loss_align_mse = F.mse_loss(z_e_h, z_e_r)
            loss_align_nce = info_nce_loss(z_e_h_pool, z_e_r_pool, temperature=TEMPERATURE)
            loss_align = loss_align_mse + loss_align_nce
            
            loss_cross = F.mse_loss(outputs['human']['retargeted'], x_r)
            
            # VQ Losses
            loss_vq = out_r['loss_vq'] + outputs['human']['loss_vq']
            
            # Total Loss (Updated with Config Constants)
            loss = (LAMBDA_RECON * loss_recon + 
                    LAMBDA_VQ    * loss_vq + 
                    LAMBDA_VEL   * loss_vel + 
                    LAMBDA_CROSS * loss_cross + 
                    LAMBDA_ALIGN * loss_align)
            
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            # === 累加分项 Loss ===
            t_recon_acc += loss_recon.item()
            t_cross_acc += loss_cross.item()
            t_align_acc += loss_align.item()
            t_vq_acc += loss_vq.item()    # 新增
            t_vel_acc += loss_vel.item()  # 新增
        
        # --- Validation ---
        model.eval()
        v_recon = 0; v_cross = 0; v_align = 0; # 新增 v_align
        v_vel = 0; v_jerk = 0; v_ppl = 0; v_dcr = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_r = batch[0].to(device)
                x_h = batch[1].to(device)
                
                outputs = model(x_robot=x_r, x_human=x_h)
                x_recon = outputs['robot']['recon']
                metrics = outputs['robot']['metrics'] 
                
                # Metrics Calculation
                v_recon += F.mse_loss(x_recon, x_r).item()
                v_cross += F.mse_loss(outputs['human']['retargeted'], x_r).item()
                
                # === 新增：计算验证集 Alignment Loss ===
                z_e_r = outputs['robot']['z_e']
                z_e_h = outputs['human']['z_e']
                v_align += F.mse_loss(z_e_h, z_e_r).item()
                
                v_vel += F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x_r[:,1:]-x_r[:,:-1]).item()
                v_jerk += calculate_jerk_loss(x_r, x_recon).item()
                v_ppl += metrics['perplexity'].item()
                v_dcr += metrics['dcr'].item()
        
        steps = len(val_loader)
        avg_recon = v_recon / steps
        
        # Log History
        train_steps = len(train_loader)
        history['train_loss'].append(t_loss / train_steps)
        history['train_recon_loss'].append(t_recon_acc / train_steps)
        history['train_cross_loss'].append(t_cross_acc / train_steps)
        history['train_align_loss'].append(t_align_acc / train_steps)
        history['train_vq_loss'].append(t_vq_acc / train_steps)   # 新增
        history['train_vel_loss'].append(t_vel_acc / train_steps) # 新增

        history['val_recon'].append(avg_recon)
        history['val_cross_recon'].append(v_cross / steps)
        history['val_align'].append(v_align / steps) # 记录
        history['val_vel'].append(v_vel / steps)
        history['val_jerk'].append(v_jerk / steps)
        history['perplexity'].append(v_ppl / steps)
        history['dead_code_ratio'].append(v_dcr / steps)

        # Early Stopping
        if avg_recon < best_val_loss:
            best_val_loss = avg_recon
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f"    Ep {epoch}: Recon {avg_recon:.4f} | Align {v_align/steps:.4f} | PPL {v_ppl/steps:.1f}")
            
        if patience_counter >= patience:
            break

    return history, model

def run_task(args):
    config, seed, device_id = args
    if torch.cuda.is_available() and device_id >= 0:
        torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    
    train_loader, val_loader, r_dim, h_dim = load_paired_data()
    if train_loader is None: return "Failed to load data"
    
    try:
        history, model = train_seed(config, seed, train_loader, val_loader, r_dim, h_dim, device)
        
        # Save Log
        log_file = os.path.join(LOG_DIR, f"log_{config['id']}_seed_{seed}.json")
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=4)
            
        # Save Model
        ckpt_name = f"{config['name']}_{config['method']}_seed_{seed}.pth"
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, ckpt_name))
        
        return f"Success: {config['name']} (Seed {seed})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in {config['name']}: {e}"

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus)) if num_gpus > 0 else [-1]
    MAX_WORKERS = max(1, num_gpus * GPU_Work_Multipliers)
    
    tasks = []
    idx = 0
    # 预加载一次检查数据
    load_paired_data()
    
    for config in EXPERIMENTS:
        for seed in SEEDS:
            # === 修改：如果旧日志缺少新字段，强制重新训练 ===
            log_path = os.path.join(LOG_DIR, f"log_{config['id']}_seed_{seed}.json")
            needs_run = True
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if 'val_align' in data: # 检查新字段是否存在
                            needs_run = False
                    except: pass
            
            if needs_run:
                tasks.append((config, seed, device_ids[idx % len(device_ids)]))
                idx += 1
            
    print(f"Tasks: {len(tasks)}")
    
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            print(f.result())

if __name__ == "__main__":
    main()