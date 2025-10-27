"""
使用示例 - 展示如何使用各个模块
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import PhoneNumberGenerator
from src.trainer import SSTrainer
from src.visualizer import LossVisualizer


def example_generate_data():
    """示例1: 生成数据集"""
    print("=" * 60)
    print("示例1: 生成数据集")
    print("=" * 60)
    
    generator = PhoneNumberGenerator(num_samples=10)  # 生成10个样本作为示例
    
    # 生成单个组的数据
    group_a = generator.generate_group_a()
    print(f"\nGroupA示例 (前5个):")
    for phone in group_a[:5]:
        print(f"  {phone}")
    
    group_b = generator.generate_group_b()
    print(f"\nGroupB示例 (前5个):")
    for phone in group_b[:5]:
        print(f"  {phone}")
    
    group_c = generator.generate_group_c()
    print(f"\nGroupC示例 (前5个):")
    for phone in group_c[:5]:
        print(f"  {phone}")
    
    # 生成格式化数据集
    datasets = generator.generate_all_datasets()
    print(f"\n格式化数据示例:")
    print(f"  {datasets['GroupA'][0]['text']}")


def example_train_model():
    """示例2: 训练模型（需要模型文件）"""
    print("\n" + "=" * 60)
    print("示例2: 训练模型")
    print("=" * 60)
    
    model_path = Path(__file__).parent.parent / "Qwen3-1___7B-FP8"
    
    if not model_path.exists():
        print("模型文件不存在，跳过训练示例")
        return
    
    trainer = SSTrainer(
        model_path=str(model_path),
        output_dir="models_example",
        lr=2e-4,
        epochs=2,  # 示例中只训练2个epoch
        batch_size=2,
        gradient_accumulation_steps=2
    )
    
    # 这里只是展示配置，不实际运行训练
    print("训练器配置:")
    print(f"  学习率: {trainer.lr}")
    print(f"  训练轮数: {trainer.epochs}")
    print(f"  批次大小: {trainer.batch_size}")


def example_visualize():
    """示例3: 可视化结果"""
    print("\n" + "=" * 60)
    print("示例3: 可视化结果")
    print("=" * 60)
    
    results_dir = Path(__file__).parent.parent / "results"
    loss_file = results_dir / "training_losses.json"
    
    if not loss_file.exists():
        print("Loss文件不存在，创建示例数据...")
        
        # 创建示例loss数据
        example_data = [
            {
                "group": "GroupA",
                "losses": [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35]
            },
            {
                "group": "GroupB",
                "losses": [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2]
            },
            {
                "group": "GroupC",
                "losses": [2.5, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
            }
        ]
        
        results_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        print(f"已创建示例loss文件: {loss_file}")
    
    visualizer = LossVisualizer(str(loss_file))
    visualizer.print_statistics()
    
    # 注意：这里不调用plot_loss_curves()避免弹出图片窗口
    print("\n(可视化图片已跳过，实际使用时会保存到文件)")


if __name__ == "__main__":
    example_generate_data()
    example_train_model()
    example_visualize()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

