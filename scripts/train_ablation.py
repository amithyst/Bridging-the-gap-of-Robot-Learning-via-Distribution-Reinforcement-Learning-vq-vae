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

# ==========================================
# 1. Enhanced Configs
# ==========================================
# 增加 SOTA 方法 (FSQ, LFQ) 和 Transformer 架构
ABLATION_EXPERIMENTS = [
    # 1. Baseline
    {'name': 'Baseline (Simple+EMA)', 'arch': 'simple', 'method': 'ema'},
    
    # 2. Main Proposal (Strongest Classic VQ)
    {'name': 'Proposed (ResNet+EMA)', 'arch': 'resnet', 'method': 'ema'},
    
    # 3. Advanced VQ (Fix Collapse)
    {'name': 'Advanced (ResNet+RVQ)', 'arch': 'resnet', 'method': 'rvq'},
    
    # 4. SOTA 1: Finite Scalar Quantization (2024 Trending)
    {'name': 'SOTA-FSQ (ResNet+FSQ)', 'arch': 'resnet', 'method': 'fsq'},
    
    # 5. SOTA 2: Lookup-Free Quantization (Binary)
    {'name': 'SOTA-LFQ (ResNet+LFQ)', 'arch': 'resnet', 'method': 'lfq'},
]

# 多随机种子，用于画 Variance Error Band
SEEDS = [42, 2024, 999] 

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

def load_data():
    data_path = os.path.join('data', 'processed', 'g1_train.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    raw_data = np.load(data_path).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(raw_data))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_seed(config, seed, train_loader, val_loader):
    set_seed(seed)
    print(f"  > Seed {seed} | Arch: {config['arch']} | Method: {config['method']}")
    
    model = MotionVQVAE(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        arch=config['arch'],
        method=config['method']
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    history = {'train_loss': [], 'val_recon': [], 'val_vel': [], 'perplexity': []}
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        t_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_recon, loss_vq, _ = model(x)
            loss_recon = F.mse_loss(x_recon, x)
            loss_vel = F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1])
            loss = loss_recon + loss_vq + 0.5 * loss_vel
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        # Validation
        model.eval()
        v_recon = 0; v_vel = 0; v_ppl = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                x_recon, _, ppl = model(x)
                v_recon += F.mse_loss(x_recon, x).item()
                v_vel += F.mse_loss(x_recon[:,1:]-x_recon[:,:-1], x[:,1:]-x[:,:-1]).item()
                v_ppl += ppl.item()
        
        # Log
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_recon'].append(v_recon / len(val_loader))
        history['val_vel'].append(v_vel / len(val_loader))
        history['perplexity'].append(v_ppl / len(val_loader))

    return history

def main():
    train_loader, val_loader = load_data()
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Starting Multi-Seed Ablation Study on {DEVICE}...")
    
    for config in ABLATION_EXPERIMENTS:
        exp_name = config['name']
        print(f"\n>>> Experiment: {exp_name}")
        
        # 运行多个 Seed
        for seed in SEEDS:
            log_file = os.path.join(LOG_DIR, f"log_{config['arch']}_{config['method']}_seed_{seed}.json")
            
            # 如果跑过了就跳过 (断点续传)
            if os.path.exists(log_file):
                print(f"  Skipping Seed {seed} (Already exists)")
                continue
                
            history = train_seed(config, seed, train_loader, val_loader)
            
            # 单独保存每个 Seed 的结果
            with open(log_file, 'w') as f:
                json.dump(history, f, indent=4)
                
    print("\nAll experiments completed.")
    print("Run 'python scripts/plot_results.py' to visualize with error bands.")

if __name__ == "__main__":
    main()