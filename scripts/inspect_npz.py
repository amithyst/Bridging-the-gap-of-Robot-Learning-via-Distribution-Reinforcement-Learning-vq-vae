import numpy as np
import os
import glob

def inspect_data():
    # 搜索解压目录下的任意一个 .npz 文件
    search_path = "./data/raw/unzipped/**/*.npz"
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("错误：在 ./data/raw/unzipped 下没有找到 .npz 文件，请检查解压是否成功。")
        return

    # 选取第一个文件进行检查
    target_file = files[0]
    print(f"正在检查文件: {target_file}")
    
    try:
        data = np.load(target_file, allow_pickle=True)
        print("\n=== 文件内部 Keys ===")
        print(list(data.keys()))
        
        print("\n=== 数据形状示例 ===")
        for key in data.keys():
            # 打印前5个key的形状，避免刷屏
            if isinstance(data[key], (np.ndarray, list)):
                try:
                    print(f"{key}: {data[key].shape}")
                except:
                    print(f"{key}: (无法获取 shape)")
            else:
                print(f"{key}: Type={type(data[key])}")
                
    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    inspect_data()