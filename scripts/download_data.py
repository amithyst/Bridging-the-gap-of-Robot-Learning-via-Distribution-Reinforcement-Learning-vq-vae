from modelscope.hub.snapshot_download import snapshot_download
import os

# 确保数据目录存在
# 注意：这里 cache_dir 建议设置为绝对路径或确保相对路径正确
save_dir = './data/raw' 
os.makedirs(save_dir, exist_ok=True)

print("开始下载数据集...")
try:
    # 关键修改：添加 repo_type='dataset'
    model_dir = snapshot_download(
        'seulzx/smplx_datasets', 
        repo_type='dataset',      # <--- 必须加这一行
        cache_dir=save_dir
    )
    print(f"下载成功！数据位于: {model_dir}")
    print("请手动解压 .tar.bz2 文件到 data/raw 文件夹中以便后续处理。")
except Exception as e:
    print(f"下载失败: {e}")