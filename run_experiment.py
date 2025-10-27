"""
主运行脚本 - 执行完整实验流程
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_generator import PhoneNumberGenerator
from src.trainer import SSTrainer
from src.visualizer import LossVisualizer


def main():
    """执行完整实验流程"""
    print("=" * 60)
    print("电话号码记忆实验")
    print("=" * 60)
    
    # 获取基础路径
    base_dir = Path(__file__).parent
    model_path = base_dir / "Qwen3-1___7B"  # 使用非FP8版本
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    # 步骤1: 生成数据集
    print("\n步骤1: 生成数据集...")
    generator = PhoneNumberGenerator(num_samples=100)
    datasets = generator.generate_all_datasets()
    generator.save_datasets(datasets, data_dir)
    
    # 步骤2: 训练模型
    print("\n步骤2: 训练模型...")
    print("警告: 训练过程可能需要较长时间，请耐心等待...")
    
    # 检查模型路径
    if not model_path.exists():
        print(f"错误: 找不到模型路径 {model_path}")
        return
    
    trainer = SSTrainer(
        model_path=str(model_path),
        output_dir=str(models_dir),
        lr=2e-4,
        epochs=10,
        batch_size=4,
        gradient_accumulation_steps=4
    )
    
    trainer.train_all_groups(str(data_dir))
    trainer.save_losses(str(results_dir / "training_losses.json"))
    
    # 步骤3: 可视化结果
    print("\n步骤3: 可视化结果...")
    visualizer = LossVisualizer(str(results_dir / "training_losses.json"))
    visualizer.print_statistics()
    visualizer.plot_loss_curves()
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()

