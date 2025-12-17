import numpy as np
import os
import glob
from tqdm import tqdm

# === 配置参数 ===
WINDOW_SIZE = 64   # VQ-VAE 输入的时间窗口长度
STRIDE = 32        # 滑动步长 (重叠一半，增加数据量)
# ================

def slice_sequence(motion, window_size, stride):
    """
    将长动作序列切片成固定长度的小段
    Input: (T, Dof)
    Output: (N, Window, Dof)
    """
    num_frames = motion.shape[0]
    if num_frames < window_size:
        return [] # 太短的扔掉
        
    slices = []
    # 滑动窗口切片
    for i in range(0, num_frames - window_size + 1, stride):
        slices.append(motion[i : i + window_size])
        
    return slices

def process_g1_data(source_root, output_dir):
    """
    处理 G1 机器人数据：读取 -> 清洗 -> 切片 -> 保存
    """
    # 1. 自动寻找 train 和 test/val 文件夹
    subdirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print(f"在 {source_root} 下发现了子目录: {subdirs}")
    
    # 简单的逻辑：名字里带 'train' 的是训练集，其他的是测试集
    train_dirs = [d for d in subdirs if 'train' in d.lower()]
    test_dirs = [d for d in subdirs if 'train' not in d.lower()]
    
    datasets = {
        'train': train_dirs,
        'test': test_dirs
    }
    
    # 统计量 (Mean/Std) 将只根据 train 计算
    train_slices_cache = [] 

    os.makedirs(output_dir, exist_ok=True)

    for split_name, dir_list in datasets.items():
        if not dir_list:
            print(f"警告: 没有找到 {split_name} 相关的文件夹，跳过。")
            continue
            
        print(f"\n正在处理 {split_name} 集 (来源: {dir_list})...")
        
        all_slices = []
        file_count = 0
        
        for d in dir_list:
            # 搜索所有 .npz
            search_path = os.path.join(source_root, d, '**', '*.npz')
            files = glob.glob(search_path, recursive=True)
            file_count += len(files)
            
            for f in tqdm(files, desc=f"Loading {d}"):
                try:
                    data = np.load(f, allow_pickle=True)
                    if 'joint_pos' not in data: continue
                    
                    motion = data['joint_pos'] # (T, Dof)
                    
                    # 简单清洗
                    if np.isnan(motion).any() or np.isinf(motion).any(): continue
                    
                    # === 关键：切片 ===
                    # 只有 Train 切片重叠 (Data Augmentation)，Test 不重叠或步长更大
                    current_stride = STRIDE if split_name == 'train' else WINDOW_SIZE
                    
                    slices = slice_sequence(motion, WINDOW_SIZE, current_stride)
                    all_slices.extend(slices)
                    
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        if not all_slices:
            print(f"{split_name} 集提取结果为空！")
            continue
            
        # 转换为 Numpy 数组 (N, Window, Dof)
        # 这就是 VQ-VAE 喜欢的标准格式，不再是 object 了
        final_data = np.array(all_slices, dtype=np.float32)
        print(f"--> {split_name} 完成。Shape: {final_data.shape} (Samples, Time, Dof)")
        
        # 保存数据
        save_path = os.path.join(output_dir, f'g1_{split_name}.npy')
        np.save(save_path, final_data)
        
        # 如果是训练集，缓存下来算均值方差
        if split_name == 'train':
            train_slices_cache = final_data

    # === 计算并保存归一化统计量 (只用 Train) ===
    if len(train_slices_cache) > 0:
        print("\n正在计算归一化统计量 (Mean/Std)...")
        # reshape 成 (N*T, Dof) 来算
        flat_data = train_slices_cache.reshape(-1, train_slices_cache.shape[-1])
        
        mean = np.mean(flat_data, axis=0)
        std = np.std(flat_data, axis=0) + 1e-6
        
        np.save(os.path.join(output_dir, 'mean.npy'), mean)
        np.save(os.path.join(output_dir, 'std.npy'), std)
        print("Mean/Std 已保存。数据预处理全部完成！")

def main():
    # 你的路径
    raw_root = "./data/raw/unzipped/extended_datasets/lafan1_dataset/g1"
    processed_root = "./data/processed"
    
    process_g1_data(raw_root, processed_root)

if __name__ == "__main__":
    main()