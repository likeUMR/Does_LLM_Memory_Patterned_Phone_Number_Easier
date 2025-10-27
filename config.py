"""
配置文件 - 统一管理实验参数
"""

from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Qwen3-1___7B"  # 使用非FP8版本
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# 数据生成配置
NUM_SAMPLES = 100  # 每个组生成的电话号码数量

# 训练配置
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_epochs": 10,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 128,
    "fp16": True,
    "lora_r": 32,  # rank提升到32，大幅增加可训练参数
    "lora_alpha": 64,  # alpha设置为2倍rank
    "lora_dropout": 0.1,
}

# LoRA目标模块（针对Qwen模型）
# 包含Attention层和FFN层，扩大训练范围
LORA_TARGET_MODULES = [
    "q_proj",      # Attention Query
    "k_proj",      # Attention Key
    "v_proj",      # Attention Value
    "o_proj",      # Attention Output
    "gate_proj",   # FFN Gate Projection
    "up_proj",     # FFN Up Projection
    "down_proj"    # FFN Down Projection
]

# 组标签映射
GROUP_LABELS = {
    "GroupA": "随机号码",
    "GroupB": "反对称号码",
    "GroupC": "高度重复号码"
}

# 绘图配置
PLOT_CONFIG = {
    "figsize": (10, 6),
    "dpi": 300,
    "colors": ['#1f77b4', '#ff7f0e', '#2ca02c']
}

