# scripts/data/process_data.py
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def compute_6d_rotation(data):
    """
    将输入姿态转换为连续的 6D 旋转表示。
    """
    T = data.shape[0]
    total_features = data.size // T
    
    if total_features % 3 == 0:
        data_flat = data.reshape(-1, 3) 
        rot_mats = R.from_rotvec(data_flat).as_matrix()
        J = total_features // 3
    elif total_features % 4 == 0:
        data_flat = data.reshape(-1, 4)
        rot_mats = R.from_quat(data_flat).as_matrix()
        J = total_features // 4
    else:
        raise ValueError(f"不支持的特征维度: 每帧 {total_features} 维")

    rot_6d_flat = rot_mats[:, :, :2].reshape(-1, 6)
    rot_6d = rot_6d_flat.reshape(T, J * 6)
    return rot_6d

def slice_sequence(motion, window_size, stride):
    num_frames = motion.shape[0]
    if num_frames < window_size:
        return []
    slices = []
    for i in range(0, num_frames - window_size + 1, stride):
        slices.append(motion[i : i + window_size])
    return slices

def process_paired_data(args):
    source_root = args.input_dir
    output_dir = args.output_dir
    window = args.window
    stride = args.step
    
    # === [新增] 缓存检查 ===
    main_file = os.path.join(output_dir, 'g1_train.npy')
    raw_file = os.path.join(output_dir, 'g1_train_full_raw.npy')
    
    if os.path.exists(main_file) and os.path.exists(raw_file) and not args.overwrite:
        print(f"✅ Data already exists in {output_dir}. Skipping processing.")
        print(f"   (Use --overwrite to force re-processing)")
        return

    subdirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    train_dirs = [d for d in subdirs if 'train' in d.lower()]
    
    if not train_dirs:
        print(f"警告: 在 {source_root} 下未找到 train 目录")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 切片数据（用于训练）
    robot_slices_all = []
    human_slices_all = []
    
    # === [新增] 完整序列数据（用于推理/长序列可视化）===
    robot_raw_all = [] 
    
    print(f"🚀 Processing Data | Window: {window} | Stride: {stride} | Mode: 6D Rotation")

    for d in train_dirs:
        search_path = os.path.join(source_root, d, '**', '*.npz')
        files = glob.glob(search_path, recursive=True)
        
        for f in tqdm(files, desc=f"Scanning {d}"):
            try:
                data = np.load(f, allow_pickle=True)
                if 'joint_pos' not in data or 'smplx_pose_body' not in data: continue
                
                # Robot
                robot_motion = data['joint_pos'] 
                if robot_motion.ndim > 2: robot_motion = robot_motion.reshape(robot_motion.shape[0], -1)
                
                # Human (6D)
                human_motion = compute_6d_rotation(data['smplx_pose_body'])

                # Length Check & Align
                min_len = min(len(robot_motion), len(human_motion))
                robot_motion = robot_motion[:min_len]
                human_motion = human_motion[:min_len]
                
                if np.isnan(robot_motion).any() or np.isnan(human_motion).any(): continue

                # === [新增] 保存完整序列到列表 (不切片) ===
                # 只有足够长的才值得存，或者全部存
                if min_len >= window:
                    robot_raw_all.append(robot_motion)

                # Slicing (用于训练)
                r_slices = slice_sequence(robot_motion, window, stride)
                h_slices = slice_sequence(human_motion, window, stride)

                if len(r_slices) == len(h_slices) and len(r_slices) > 0:
                    robot_slices_all.extend(r_slices)
                    human_slices_all.extend(h_slices)
                
            except Exception as e:
                print(f"Error reading {f}: {e}")

    if not robot_slices_all:
        print("❌ Error: No data found.")
        return

    # 保存训练切片
    robot_data = np.array(robot_slices_all, dtype=np.float32)
    human_data = np.array(human_slices_all, dtype=np.float32)
    
    # === [新增] 保存完整序列列表 ===
    # 注意：因为长度不一，必须指定 allow_pickle=True，它会被存为 object array (list)
    np.save(raw_file, np.array(robot_raw_all, dtype=object))
    print(f"   Saved {len(robot_raw_all)} full sequences to {raw_file}")

    print(f"✅ Data Processed!")
    print(f"   Robot Train Shape: {robot_data.shape} (N, {window}, 29)")
    
    np.save(os.path.join(output_dir, 'g1_train.npy'), robot_data)
    np.save(os.path.join(output_dir, 'human_train.npy'), human_data)
    
    print("Computing Stats...")
    r_flat = robot_data.reshape(-1, robot_data.shape[-1])
    np.save(os.path.join(output_dir, 'mean.npy'), np.mean(r_flat, axis=0))
    np.save(os.path.join(output_dir, 'std.npy'), np.std(r_flat, axis=0) + 1e-6)
    
    h_flat = human_data.reshape(-1, human_data.shape[-1])
    np.save(os.path.join(output_dir, 'human_mean.npy'), np.mean(h_flat, axis=0))
    np.save(os.path.join(output_dir, 'human_std.npy'), np.std(h_flat, axis=0) + 1e-6)

    print("🎉 Done. Ready for training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/raw/unzipped/extended_datasets/lafan1_dataset/g1")
    parser.add_argument('--output_dir', type=str, default="./data/processed")
    parser.add_argument('--window', type=int, default=64) 
    parser.add_argument('--step', type=int, default=20)   
    # [新增] 覆盖开关
    parser.add_argument('--overwrite', action='store_true', help="Force re-processing even if files exist")
    args = parser.parse_args()
    
    process_paired_data(args)