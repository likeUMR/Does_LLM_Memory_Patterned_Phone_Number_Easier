"""
可视化模块 - 绘制训练loss曲线
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import platform

# 设置中文字体
def setup_chinese_font():
    """设置支持中文的字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常用中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['Arial Unicode MS', 'STHeiti', 'PingFang SC']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    
    # 尝试找到可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return font
    
    # 如果没有找到，使用默认字体并警告
    print("警告: 未找到中文字体，可能无法正确显示中文")
    return None

# 初始化字体
setup_chinese_font()

# 设置matplotlib为非交互模式，避免阻塞程序
plt.ioff()


class LossVisualizer:
    """Loss可视化器"""
    
    def __init__(self, loss_file: str):
        """
        初始化可视化器
        
        Args:
            loss_file: loss数据文件路径
        """
        self.loss_file = Path(loss_file)
        self.data = None
    
    def load_data(self):
        """加载loss数据"""
        if not self.loss_file.exists():
            raise FileNotFoundError(f"找不到loss文件: {self.loss_file}")
        
        with open(self.loss_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def plot_loss_curves(self, output_path: str = None):
        """
        绘制loss曲线
        
        Args:
            output_path: 输出图片路径
        """
        if self.data is None:
            self.load_data()
        
        plt.figure(figsize=(10, 6))
        
        # 定义组名和对应的中文标签
        group_labels = {
            'GroupA': '随机号码',
            'GroupB': '反对称号码',
            'GroupC': '高度重复号码'
        }
        
        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 绘制每个组的loss曲线
        for idx, group_data in enumerate(self.data):
            group_name = group_data['group']
            losses = group_data['losses']
            
            # Epoch从0开始（0表示未训练基线）
            epochs = list(range(len(losses)))
            label = group_labels.get(group_name, group_name)
            
            plt.plot(
                epochs,
                losses,
                marker='o',
                label=label,
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=6
            )
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('三种电话号码模式的训练Loss对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        if output_path is None:
            output_path = self.loss_file.parent / "loss_curves.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存到: {output_path}")
        
        # 关闭图形，避免阻塞程序
        plt.close()
    
    def print_statistics(self):
        """打印统计信息"""
        if self.data is None:
            self.load_data()
        
        print("\n训练统计信息:")
        print("=" * 60)
        
        for group_data in self.data:
            group_name = group_data['group']
            losses = group_data['losses']
            
            initial_loss = losses[0] if losses else 0  # Epoch 0 (未训练)
            final_loss = losses[-1] if losses else 0   # 最终epoch
            reduction = initial_loss - final_loss
            reduction_percent = (reduction / initial_loss * 100) if initial_loss > 0 else 0
            
            print(f"\n{group_name}:")
            print(f"  Epoch 0 (未训练) Loss: {initial_loss:.4f}")
            print(f"  最终 Loss: {final_loss:.4f}")
            print(f"  Loss降低: {reduction:.4f} ({reduction_percent:.2f}%)")
            print(f"  训练轮数: {len(losses) - 1}")  # 减去epoch 0
        
        print("=" * 60)


def main():
    """主函数"""
    results_dir = Path(__file__).parent.parent / "results"
    loss_file = results_dir / "training_losses.json"
    
    if not loss_file.exists():
        print(f"错误: 找不到loss文件 {loss_file}")
        print("请先运行 trainer.py 进行训练")
        return
    
    visualizer = LossVisualizer(str(loss_file))
    visualizer.print_statistics()
    visualizer.plot_loss_curves()


if __name__ == "__main__":
    main()

