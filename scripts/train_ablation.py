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
import argparse
# 在原有的 import 下面添加：
import time
import datetime
from tqdm import tqdm # 需要确保 pip install tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import DualMotionVQVAE
try:
    from models.experiment_config import EXPERIMENTS
except ImportError:
    EXPERIMENTS = [] 

# ANSI 颜色代码
class TermColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m' # 补上了这个
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RED = '\033[91m'    # 顺便补上 RED
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# === Default Config ===
DEFAULT_SEEDS = [42]
# 基础 Batch Size (单卡)。如果是多卡，代码会自动乘以卡数
# HIDDEN_DIM = 256     # <--- 从 64 提升到 128 或 256
BATCH_SIZE = 512     # <--- 显存够的话 (4090肯定够)，Batch大一点更稳
EPOCHS = 400
LEARNING_RATE = 2e-4
HIDDEN_DIM = 64
GPU_WORK_MULTIPLIERS = 3
LOG_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'

# === Loss Weights ===
LAMBDA_RECON = 1.0
LAMBDA_VQ    = 1.0
LAMBDA_VEL   = 0.5
LAMBDA_CROSS = 5.0   
LAMBDA_ALIGN = 100.0 
TEMPERATURE = 0.07   

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_jerk_loss(real, recon):
    if real.shape[1] < 4: return torch.tensor(0.0).to(real.device)
    real_jerk = torch.diff(real, n=3, dim=1)
    recon_jerk = torch.diff(recon, n=3, dim=1)
    return F.mse_loss(recon_jerk, real_jerk)

def info_nce_loss(z_h, z_r, temperature=0.07):
    z_h = F.normalize(z_h, dim=-1)
    z_r = F.normalize(z_r, dim=-1)
    logits = torch.matmul(z_h, z_r.T) / temperature
    labels = torch.arange(z_h.size(0)).to(z_h.device)
    loss_h = F.cross_entropy(logits, labels)
    loss_r = F.cross_entropy(logits.T, labels)
    return (loss_h + loss_r) / 2

def load_paired_data(batch_size):
    """
    修改：接收 dynamic batch_size
    """
    p_root = os.path.join('data', 'processed')
    r_path = os.path.join(p_root, 'g1_train.npy')
    h_path = os.path.join(p_root, 'human_train.npy')
    
    if not os.path.exists(r_path) or not os.path.exists(h_path):
        print("Error: 数据文件缺失，请先运行 process_data.py")
        return None, None, 0, 0 

    r_data = np.load(r_path).astype(np.float32)
    h_data = np.load(h_path).astype(np.float32)
    
    min_n = min(len(r_data), len(h_data))
    r_data = r_data[:min_n]
    h_data = h_data[:min_n]
    
    robot_dim = r_data.shape[-1]
    human_dim = h_data.shape[-1]
    
    print(f"Dataset Loaded. Dim: R={robot_dim}/H={human_dim}, N={min_n}, Batch={batch_size}")
    
    dataset = TensorDataset(torch.from_numpy(r_data), torch.from_numpy(h_data))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Num_workers > 0 可以加速数据加载，防止 GPU 等 CPU
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), \
           robot_dim, human_dim

# === 替换整个 train_seed 函数 ===
def train_seed(config, seed, train_loader, val_loader, robot_dim, human_dim, device, use_multi_gpu=False):
    set_seed(seed)
    
    epochs = config.get('epochs', 400)
    mode = config.get('mode', 'teacher')
    resume = config.get('resume', False)
    teacher_ckpt = config.get('teacher_ckpt', None)
    
    # 定义日志及路径
    log_name = f"log_{config['id']}_seed_{seed}.json" if 'id' in config else f"log_{config['name']}_seed_{seed}.json"
    log_file = os.path.join(LOG_DIR, log_name)
    run_name = f"{config['name']}_{config['method']}_{mode}_seed_{seed}"
    
    # 打印启动信息 (彩色)
    print(f"{TermColor.HEADER}🚀 Start: {run_name} | Mode: {mode.upper()} | Device: {device}{TermColor.ENDC}")
    
    # === Model Init ===
    model = DualMotionVQVAE(
        robot_input_dim=robot_dim,
        human_input_dim=human_dim,
        hidden_dim=config.get('hidden_dim', 64), 
        arch=config.get('arch', 'transformer'), 
        method=config['method'],
        window_size=config.get('window', 64) # <--- [新增] 从 config 读取 window
    ).to(device)

    # === Resume Logic ===
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_last.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    # [新增] 初始化早停计数器
    patience = config.get('patience', -1) # -1 代表不启用
    patience_counter = 0

    history = {k: [] for k in ['train_loss', 'val_loss', 'val_recon', 'val_align']}

    if resume and os.path.exists(ckpt_path):
        print(f"{TermColor.CYAN}    [Resume] Loading checkpoint: {ckpt_path}{TermColor.ENDC}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 尝试加载 json
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f: history = json.load(f)
            except: pass
            
        if checkpoint.get('config', {}).get('mode') == mode:
             start_epoch = checkpoint['epoch'] + 1
             best_val_loss = checkpoint.get('best_loss', float('inf'))
    
    elif mode == 'student':
        if teacher_ckpt and os.path.exists(teacher_ckpt):
            print(f"{TermColor.BLUE}    [Student] Loading Teacher: {teacher_ckpt}{TermColor.ENDC}")
            teacher_state = torch.load(teacher_ckpt, map_location=device)
            # 过滤 human_encoder，只加载 robot 部分
            pretrained_dict = {k: v for k, v in teacher_state['model_state_dict'].items() if 'human_encoder' not in k}
            model_state = model.state_dict()
            model_state.update(pretrained_dict)
            model.load_state_dict(model_state)
            # 冻结
            for name, param in model.named_parameters():
                if 'human_encoder' not in name: param.requires_grad = False
        else:
            raise ValueError(f"Student mode requires valid --teacher_ckpt")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=config.get('lr', 2e-4), weight_decay=1e-4)
    if resume and 'optimizer_state' in locals():
        # 只有当参数数量匹配时才加载 optimizer (防止 teacher->student 切换时报错)
        try: optimizer.load_state_dict(optimizer_state)
        except: pass

    if use_multi_gpu: model = nn.DataParallel(model)

    # === Timer ===
    total_start_time = time.time()

    # === Training Loop ===
    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = 0.0
        
        # === TQDM Color Setup ===
        # Teacher 用青色，Student 用绿色，区分一下
        bar_color = 'cyan' if mode == 'teacher' else 'green'
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False, dynamic_ncols=True, colour=bar_color)
        
        for batch in pbar:
            x_r = batch[0].to(device)
            x_h = batch[1].to(device)
            
            optimizer.zero_grad()
            
            if mode == 'teacher':
                outputs = model(x_robot=x_r, x_human=None)
                out_r = outputs['robot']
                loss_recon = F.mse_loss(out_r['recon'], x_r)
                loss_vel = F.mse_loss(out_r['recon'][:,:,1:]-out_r['recon'][:,:,:-1], x_r[:,:,1:]-x_r[:,:,:-1])
                if use_multi_gpu: loss_vq = out_r['loss_vq'].mean()
                else: loss_vq = out_r['loss_vq']
                loss = (LAMBDA_RECON * loss_recon + LAMBDA_VQ * loss_vq + LAMBDA_VEL * loss_vel)
                
            elif mode == 'student':
                outputs = model(x_robot=x_r, x_human=x_h)
                z_e_r = outputs['robot']['z_e'].detach()
                z_e_h = outputs['human']['z_e']
                loss = LAMBDA_ALIGN * F.mse_loss(z_e_h, z_e_r)

            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # --- Validation ---
        model.eval()
        v_recon = 0.0
        v_align = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_r = batch[0].to(device)
                x_h = batch[1].to(device)
                outputs = model(x_robot=x_r, x_human=x_h)
                if mode == 'teacher':
                    v_recon += F.mse_loss(outputs['robot']['recon'], x_r).item()
                elif mode == 'student':
                    v_align += F.mse_loss(outputs['human']['z_e'], outputs['robot']['z_e']).item()

        # Metrics & ETA
        avg_t_loss = t_loss / len(train_loader)
        avg_v_recon = v_recon / len(val_loader)
        avg_v_align = v_align / len(val_loader)
        current_val_metric = avg_v_recon if mode == 'teacher' else avg_v_align

        history['train_loss'].append(avg_t_loss)
        if mode == 'teacher': history['val_recon'].append(avg_v_recon)
        else: history['val_align'].append(avg_v_align)
        
        # ETA Calculation
        elapsed = time.time() - total_start_time
        done = epoch - start_epoch + 1
        eta_seconds = (epochs - epoch - 1) * (elapsed / done)
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        now_str = datetime.datetime.now().strftime("%H:%M")

        # === Colored Log Output ===
        if epoch % 5 == 0 or epoch == epochs - 1:
            metric_name = "Recon" if mode == 'teacher' else "Align"
            # 构造彩色字符串
            log_msg = (
                f"{TermColor.BOLD}[{now_str}]{TermColor.ENDC} "
                f"Ep {epoch}: "
                f"Train {TermColor.WARNING}{avg_t_loss:.4f}{TermColor.ENDC} | "
                f"Val({metric_name}) {TermColor.GREEN}{current_val_metric:.4f}{TermColor.ENDC} | "
                f"ETA {TermColor.CYAN}{eta_str}{TermColor.ENDC}"
            )
            tqdm.write(log_msg)

        # === Save ===
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if use_multi_gpu else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_val_loss,
            'config': config
        }
        torch.save(save_dict, ckpt_path)
        with open(log_file, 'w') as f: json.dump(history, f, indent=4)
        
        # === [修改] 早停逻辑与保存最佳模型 ===
        if current_val_metric < best_val_loss:
            best_val_loss = current_val_metric
            best_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pth")
            torch.save(save_dict, best_path)
            tqdm.write(f"    {TermColor.YELLOW}[*] New Best Saved!{TermColor.ENDC}")
            
            # [新增] 如果性能提升，重置计数器
            patience_counter = 0 
        else:
            # [新增] 如果没有提升，增加计数器
            if patience > 0:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n{TermColor.RED}!!! Early Stopping Triggered at Epoch {epoch} (No improvement for {patience} epochs) !!!{TermColor.ENDC}")
                    break # 跳出循环，结束训练

    return history, model

def run_task(args):
    # === 1. 修改：接收 4 个参数 (对应 main 函数中的打包) ===
    config_base, seed, device_id, extra_args = args 
    
    # === 2. 修改：合并参数 ===
    # 将命令行传入的 extra_args (如 mode, resume, teacher_ckpt) 合并到 config 中
    config = config_base.copy()
    config.update(extra_args)
    
    use_multi_gpu = False
    # 获取基础 batch_size (优先从 config 取，没有则默认 256)
    current_batch_size = config.get('batch_size', 256)

    # === 设备逻辑 ===
    if torch.cuda.is_available():
        if device_id == -999: 
            # 模式: 使用所有 GPU (DataParallel)
            use_multi_gpu = True
            device = 'cuda:0' # 主卡必须设为 0
            
            # 关键优化: 多卡时扩大 Batch Size
            gpu_count = torch.cuda.device_count()
            current_batch_size = current_batch_size * gpu_count
            print(f"!!! Scaling Batch Size to {current_batch_size} for {gpu_count} GPUs !!!")
            
        elif device_id >= 0:
            # 模式: 单卡运行
            torch.cuda.set_device(device_id)
            device = f'cuda:{device_id}'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    
    # === 3. 加载数据 (使用计算后的 batch_size) ===
    train_loader, val_loader, r_dim, h_dim = load_paired_data(current_batch_size)
    if train_loader is None: return "Failed to load data"
    
    try:
        # === 4. 开始训练 ===
        history, model = train_seed(config, seed, train_loader, val_loader, r_dim, h_dim, device, use_multi_gpu)
        
        # === 5. 保存结果 (文件名加入 mode 以区分) ===
        # Log 文件名: log_Ours(transformer+hybrid)_teacher_seed_42.json
        mode = config.get('mode', 'unknown')
        log_name = f"log_{config['name']}_{mode}_seed_{seed}.json"
        log_file = os.path.join(LOG_DIR, log_name)
        
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=4)
            
        # Model 保存 (保存最终状态，虽然 train_seed 已经保存了 best/last，这里作为双重保险)
        if use_multi_gpu and isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        # Checkpoint 文件名: Ours(transformer+hybrid)_hybrid_teacher_seed_42_final.pth
        ckpt_name = f"{config['name']}_{config['method']}_{mode}_seed_{seed}_final.pth"
        torch.save(state_dict, os.path.join(CHECKPOINT_DIR, ckpt_name))
        
        return f"Success: {config['name']} | Mode: {mode} | Seed: {seed}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in {config['name']}: {e}"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--method', type=str, default='hybrid', help='quantization method')
    parser.add_argument('--arch', type=str, default='transformer', help='encoder/decoder architecture')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--window', type=int, default=64, help='Sequence length (e.g., 64 or 10)') # <--- [新增] 命令行参数
    # [新增] 早停参数，默认 -1 表示不开启
    parser.add_argument('--patience', type=int, default=-1, help='Early stopping patience (epochs). -1 to disable.')
    
    # Teacher-Student 相关参数
    parser.add_argument('--mode', type=str, default='teacher', choices=['teacher', 'student'], help='Training mode')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if exists')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='Path to best teacher checkpoint (for student mode)')
    
    # 设备
    parser.add_argument('--force_multi_gpu', action='store_true', help='Force using all GPUs with DataParallel')
    
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 构建传给 run_task 的额外参数字典
    extra_args = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'mode': args.mode,
        'resume': args.resume,
        'teacher_ckpt': args.teacher_ckpt,
        'arch': args.arch,
        'method': args.method,
        'window': args.window,  # <--- [新增] 放入 extra_args
        'patience': args.patience, # [新增] 传入配置
        'name': f"Exp_{args.arch}_W{args.window}" # [可选] 名字里加上 W10 方便区分
    }
    
    # Config 模板
    config_template = {
        'id': f"{args.arch}_{args.method}",
        'name': f"Ours({args.arch}+{args.method})",
        'arch': args.arch,
        'method': args.method
    }

    tasks = []
    # 总是使用 DataParallel (User Request: 只有一个模型，占满显卡)
    print(f"!!! Launching Single Task in [{args.mode.upper()}] Mode using ALL GPUs !!!")
    for seed in args.seed:
        # device_id = -999 触发 DataParallel
        tasks.append((config_template, seed, -999, extra_args))

    MAX_WORKERS = 1 
    
    # 执行
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_task, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            print(f.result())