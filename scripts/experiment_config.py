# scripts/experiment_config.py

# ==========================================
# 统一实验配置列表
# 所有脚本 (Train, Plot, Latex) 都从这里读取
# ==========================================

EXPERIMENTS = [
    # 格式: (ID/Filename_Key, Display Name, Arch, Method)
    # ID 会用于生成文件名: log_{ID}_seed_{seed}.json
    # 程序会自动根据 arch 和 method 参数去实例化模型
    
    # 1. Baseline
    {
        'id': 'simple_ema',
        'name': 'Baseline (Simple)',
        'arch': 'simple',
        'method': 'ema'
    },
    # 2. Proposed (ResNet)
    {
        'id': 'resnet_ema',
        'name': 'ResNet+EMA',
        'arch': 'resnet',
        'method': 'ema'
    },
    # 3. Advanced (RVQ)
    {
        'id': 'resnet_rvq',
        'name': 'ResNet+RVQ',
        'arch': 'resnet',
        'method': 'rvq'
    },
    # 4. SOTA (FSQ)
    {
        'id': 'resnet_fsq',
        'name': 'FSQ',
        'arch': 'resnet',
        'method': 'fsq'
    },
    # 5. SOTA (LFQ)
    {
        'id': 'resnet_lfq',
        'name': 'LFQ',
        'arch': 'resnet',
        'method': 'lfq'
    },
    # === 6. NEW: Hybrid (你的新模型) ===
    # 只要在这里加了，所有脚本都会自动运行它
    {
        'id': 'resnet_hybrid',      # 对应的文件名将是 log_resnet_hybrid_seed_xxx.json
        'name': 'Hybrid (Ours)',    # 论文图表中显示的名字
        'arch': 'resnet',           # 传给 MotionVQVAE 的参数
        'method': 'hybrid'          # 传给 MotionVQVAE 的参数
    },
]