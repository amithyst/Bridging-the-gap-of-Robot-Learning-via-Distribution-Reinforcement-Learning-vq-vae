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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import MotionVQVAE

# 实验配置
ABLATION_EXPERIMENTS = [
    {'name': 'Baseline (Simple+EMA)', 'arch': 'simple', 'method': 'ema'},
    {'name': 'Proposed (ResNet+EMA)', 'arch': 'resnet', 'method': 'ema'},
    {'name': 'Advanced (ResNet+RVQ)', 'arch': 'resnet', 'method': 'rvq'},
    {'name': 'SOTA-FSQ (ResNet+FSQ)', 'arch': 'resnet', 'method': 'fsq'},
]
SEEDS = [42, 1024] # 减少为2个以快速演示，实际可用更多

BATCH_SIZE = 256
EPOCHS = 100
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

def train_seed(config, seed, train_loader, val_loader):
    set_seed(seed)
    print(f"  > Seed {seed} | {config['name']}")
    
    model = MotionVQVAE(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        arch=config['arch'],
        method=config['method']
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 增加新的 Metric keys
    history = {
        'train_loss': [], 'val_recon': [], 'val_vel': [], 
        'val_jerk': [], 'perplexity': [], 'dead_code_ratio': []
    }
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        t_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            # New Return: metrics dict
            x_recon, loss_vq, metrics = model(x)
            
            loss_recon = F.mse_loss(x_recon, x)
            loss_vel = F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1])
            
            loss = loss_recon + loss_vq + 0.5 * loss_vel
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        # Val
        model.eval()
        v_recon = 0; v_vel = 0; v_jerk = 0; v_ppl = 0; v_dcr = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                x_recon, _, metrics = model(x)
                
                v_recon += F.mse_loss(x_recon, x).item()
                v_vel += F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1]).item()
                v_jerk += calculate_jerk_loss(x, x_recon).item()
                v_ppl += metrics['perplexity'].item()
                v_dcr += metrics['dcr'].item()
        
        # Log
        steps = len(val_loader)
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_recon'].append(v_recon / steps)
        history['val_vel'].append(v_vel / steps)
        history['val_jerk'].append(v_jerk / steps)
        history['perplexity'].append(v_ppl / steps)
        history['dead_code_ratio'].append(v_dcr / steps)

    return history

def main():
    train_loader, val_loader = load_data()
    os.makedirs(LOG_DIR, exist_ok=True)
    
    for config in ABLATION_EXPERIMENTS:
        for seed in SEEDS:
            log_file = os.path.join(LOG_DIR, f"log_{config['arch']}_{config['method']}_seed_{seed}.json")
            if os.path.exists(log_file):
                continue
            
            try:
                history = train_seed(config, seed, train_loader, val_loader)
                with open(log_file, 'w') as f:
                    json.dump(history, f, indent=4)
            except Exception as e:
                print(f"Error in {config['name']} Seed {seed}: {e}")

if __name__ == "__main__":
    main()