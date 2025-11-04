"""
主运行脚本 - 执行完整实验流程
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from src.data_generator import PhoneNumberGenerator
from src.trainer import SSTrainer
from src.visualizer import LossVisualizer


def main():
    """执行完整实验流程"""
    print("=" * 60)
    print("电话号码记忆实验")
    print("=" * 60)
    
    # 步骤1: 生成数据集
    print("\n步骤1: 生成数据集...")
    generator = PhoneNumberGenerator(num_samples=config.NUM_SAMPLES)
    datasets = generator.generate_all_datasets()
    generator.save_datasets(datasets, config.DATA_DIR)
    
    # 步骤2: 训练模型
    print("\n步骤2: 训练模型...")
    print("警告: 训练过程可能需要较长时间，请耐心等待...")
    
    # 检查模型路径
    if not config.MODEL_DIR.exists():
        print(f"错误: 找不到模型路径 {config.MODEL_DIR}")
        return
    
    trainer = SSTrainer(
        model_path=str(config.MODEL_DIR),
        output_dir=str(config.MODELS_DIR),
        lr=config.TRAINING_CONFIG["learning_rate"],
        epochs=config.TRAINING_CONFIG["num_epochs"],
        batch_size=config.TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=config.TRAINING_CONFIG["gradient_accumulation_steps"],
        max_length=config.TRAINING_CONFIG["max_length"]
    )
    
    trainer.train_all_groups(str(config.DATA_DIR))
    trainer.save_losses(str(config.RESULTS_DIR / "training_losses.json"))
    
    # 步骤3: 可视化结果
    print("\n步骤3: 可视化结果...")
    visualizer = LossVisualizer(str(config.RESULTS_DIR / "training_losses.json"))
    visualizer.print_statistics()
    visualizer.plot_loss_curves()
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()

