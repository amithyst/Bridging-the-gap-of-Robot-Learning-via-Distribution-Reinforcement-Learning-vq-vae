import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
import json
import random
import concurrent.futures # <--- 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import MotionVQVAE
from models.experiment_config import EXPERIMENTS


SEEDS = [42, 1024,999,2024,2025] # 减少为2个以快速演示，实际可用更多

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 2e-4
INPUT_DIM = 29
HIDDEN_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_DIR = 'results'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_jerk_loss(real, recon):
    """
    Jerk (加加速度) = d(Acc)/dt = d(d(Vel))/dt
    Simple Finite Difference:
    Vel = x[t+1] - x[t]
    Acc = Vel[t+1] - Vel[t] = x[t+2] - 2x[t+1] + x[t]
    Jerk = Acc[t+1] - Acc[t]
    """
    if real.shape[1] < 4: return torch.tensor(0.0).to(real.device)
    
    # 使用 torch.diff 计算高阶差分
    # dim=1 is Time
    real_jerk = torch.diff(real, n=3, dim=1)
    recon_jerk = torch.diff(recon, n=3, dim=1)
    
    return F.mse_loss(recon_jerk, real_jerk)

def load_data():
    data_path = os.path.join('data', 'processed', 'g1_train.npy')
    if not os.path.exists(data_path):
        # Fallback dummy data if file missing (for testing script logic)
        print("Warning: .npy file not found, using dummy data.")
        raw_data = np.random.randn(100, 64, 29).astype(np.float32)
    else:
        raw_data = np.load(data_path).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(raw_data))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# 替换 scripts/train_ablation.py 中的 train_seed 函数

def train_seed(config, seed, train_loader, val_loader, device): # <--- 修改这里
    set_seed(seed)

    # print 改为包含设备信息，方便调试
    print(f"  > [Device: {device}] Seed {seed} | {config['name']}")
    
    model = MotionVQVAE(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        arch=config['arch'],
        method=config['method']
    ).to(device) # <--- 修改这里，使用传入的 device
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    history = {
        'train_loss': [], 'val_recon': [], 'val_vel': [], 
        'val_jerk': [], 'perplexity': [], 'dead_code_ratio': []
    }
    
    # === Early Stopping 初始化 ===
    best_val_loss = float('inf')
    patience = 50         # 连续 10 轮不提升就停止
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        t_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_recon, loss_vq, metrics = model(x)
            
            loss_recon = F.mse_loss(x_recon, x)
            loss_vel = F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1])
            
            # Loss Function: 可以根据需要调整权重
            loss = loss_recon + loss_vq + 0.5 * loss_vel
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        # Val
        model.eval()
        v_recon = 0; v_vel = 0; v_jerk = 0; v_ppl = 0; v_dcr = 0
        current_val_loss = 0 # 用于 Early Stopping
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                x_recon, _, metrics = model(x)
                
                # 计算各项指标
                recon_err = F.mse_loss(x_recon, x).item()
                vel_err = F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1]).item()
                
                v_recon += recon_err
                v_vel += vel_err
                v_jerk += calculate_jerk_loss(x, x_recon).item()
                v_ppl += metrics['perplexity'].item()
                v_dcr += metrics['dcr'].item()
                
                # 验证集 Total Loss (用于判断是否停止)
                current_val_loss += recon_err + 0.5 * vel_err

        # Log
        steps = len(val_loader)
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_recon'].append(v_recon / steps)
        history['val_vel'].append(v_vel / steps)
        history['val_jerk'].append(v_jerk / steps)
        history['perplexity'].append(v_ppl / steps)
        history['dead_code_ratio'].append(v_dcr / steps)
        
        avg_val_loss = current_val_loss / steps

        # === Early Stopping 逻辑 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 # 只要有提升，计数器归零
        else:
            patience_counter += 1
            
        # 打印简略进度条，方便观察谁停了
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Val Loss {avg_val_loss:.5f} | Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"    [Stop] Early stopping at epoch {epoch} (Best Val Loss: {best_val_loss:.5f})")
            break # 跳出 Epoch 循环

    return history

# === 新增：单次任务包装器 (解决多进程数据加载问题) ===
def run_task(args):
    config, seed, device_id = args

    # === 新增：关键修改 ===
    # 强制设定当前进程的默认 GPU，防止 CuDNN 算子去 cuda:0 找数据
    if torch.cuda.is_available() and device_id >= 0:
        torch.cuda.set_device(device_id)
    # ====================
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    
    # 在子进程中重新加载数据，避免 Dataloader 多进程 pickle 问题
    # 因为数据是 .npy 文件且很小，这样最稳妥
    train_loader, val_loader = load_data()
    
    try:
        history = train_seed(config, seed, train_loader, val_loader, device)
        
        # 保存日志
        log_file = os.path.join(LOG_DIR, f"log_{config['id']}_seed_{seed}.json")
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=4)
        return f"Success: {config['name']} (Seed {seed})"
    except Exception as e:
        return f"Error in {config['name']} Seed {seed}: {e}"

# === 修改：主函数改为并行模式 ===
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. 自动检测 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU detected, using CPU (slow).")
        device_ids = [-1] # -1 代表 CPU
    else:
        print(f"Detected {num_gpus} GPUs. Enabling parallel acceleration.")
        device_ids = list(range(num_gpus))
    
    # 2. 配置并行参数
    # 建议：如果单卡，设置为 4 (看显存大小)；如果多卡，设置为 卡数 * 2
    MAX_WORKERS = max(1, num_gpus * 2) if num_gpus > 0 else 2 
    print(f"Parallel Workers: {MAX_WORKERS}")

    # 3. 准备任务列表
    tasks = []
    task_idx = 0
    for config in EXPERIMENTS:
        for seed in SEEDS:
            log_file = os.path.join(LOG_DIR, f"log_{config['id']}_seed_{seed}.json")
            if os.path.exists(log_file):
                print(f"Skipping {config['name']} (Seed {seed}) - Already exists.")
                continue
            
            # Round-Robin 分配 GPU: 任务0->卡0, 任务1->卡1, 任务2->卡0...
            target_device_id = device_ids[task_idx % len(device_ids)]
            if num_gpus == 0: target_device_id = -1 # CPU 标志
            
            tasks.append((config, seed, target_device_id))
            task_idx += 1
    
    print(f"Total tasks to run: {len(tasks)}")

    # 4. 启动进程池
    # 设置启动方式为 spawn (Linux/Windows CUDA 必须)
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(run_task, task) for task in tasks]
        
        # 监控完成进度
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()