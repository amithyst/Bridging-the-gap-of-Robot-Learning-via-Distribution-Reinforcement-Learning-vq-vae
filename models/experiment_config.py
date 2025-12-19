# configs/experiment_config.py

# 这是一个中心化的配置文件
# id: 对应日志文件名中的标识 (例如 log_resnet_hybrid_seed_42.json)
# name: 论文/图表中显示的名字
# arch: 传递给模型的 arch 参数
# method: 传递给模型的 method 参数

EXPERIMENTS = [
    {'id': 'simple_ema',    'name': 'Baseline(Simple)',   'arch': 'simple', 'method': 'ema'},
    {'id': 'resnet_ema',    'name': 'ResNet+EMA',          'arch': 'resnet', 'method': 'ema'},
    {'id': 'resnet_rvq',    'name': 'ResNet+RVQ',          'arch': 'resnet', 'method': 'rvq'},
    {'id': 'resnet_fsq',    'name': 'FSQ',                 'arch': 'resnet', 'method': 'fsq'},
    {'id': 'resnet_lfq',    'name': 'LFQ',                 'arch': 'resnet', 'method': 'lfq'},
    
    # === 这里是你新加的 ===
    {'id': 'resnet_hybrid', 'name': 'Ours(Dual-Enc+Hybrid)',     'arch': 'resnet', 'method': 'hybrid'},
]