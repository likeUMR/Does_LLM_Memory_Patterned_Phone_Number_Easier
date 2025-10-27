"""
数据生成模块 - 生成3种不同模式的电话号码数据集
"""

import random
import json
from pathlib import Path
from typing import List, Dict


class PhoneNumberGenerator:
    """电话号码生成器"""
    
    def __init__(self, num_samples: int = 100):
        """
        初始化生成器
        
        Args:
            num_samples: 每个组生成的电话号码数量
        """
        self.num_samples = num_samples
    
    def generate_random_phone(self) -> str:
        """
        生成随机11位电话号码 (GroupA)
        
        Returns:
            随机11位数字字符串
        """
        return ''.join([str(random.randint(0, 9)) for _ in range(11)])
    
    def generate_symmetric_phone(self) -> str:
        """
        生成反对称电话号码 (GroupB)
        前5位和后5位对称，中间1位随机
        
        Returns:
            反对称11位数字字符串
        """
        # 生成前5位
        front = ''.join([str(random.randint(0, 9)) for _ in range(5)])
        # 中间1位随机
        middle = str(random.randint(0, 9))
        # 后5位与前5位对称
        back = front[::-1]
        return front + middle + back
    
    def generate_repetitive_phone(self) -> str:
        """
        生成高度重复电话号码 (GroupC)
        前3个数字相同，中间4个数字相同，后4个数字相同
        
        Returns:
            高度重复的11位数字字符串
        """
        digit1 = str(random.randint(0, 9))
        digit2 = str(random.randint(0, 9))
        digit3 = str(random.randint(0, 9))
        return digit1 * 3 + digit2 * 4 + digit3 * 4
    
    def generate_group_a(self) -> List[str]:
        """生成GroupA数据"""
        return [self.generate_random_phone() for _ in range(self.num_samples)]
    
    def generate_group_b(self) -> List[str]:
        """生成GroupB数据"""
        return [self.generate_symmetric_phone() for _ in range(self.num_samples)]
    
    def generate_group_c(self) -> List[str]:
        """生成GroupC数据"""
        return [self.generate_repetitive_phone() for _ in range(self.num_samples)]
    
    def format_dataset(self, phones: List[str], group_name: str) -> List[Dict[str, str]]:
        """
        将电话号码格式化为训练数据集
        
        Args:
            phones: 电话号码列表
            group_name: 组名称
            
        Returns:
            格式化的数据集
        """
        dataset = []
        for i, phone in enumerate(phones, start=1):
            text = f"第{i}个电话号码是：{phone}"
            dataset.append({
                "text": text,
                "group": group_name,
                "index": i
            })
        return dataset
    
    def generate_all_datasets(self) -> Dict[str, List[Dict[str, str]]]:
        """
        生成所有数据集
        
        Returns:
            包含3个组的数据集字典
        """
        group_a_phones = self.generate_group_a()
        group_b_phones = self.generate_group_b()
        group_c_phones = self.generate_group_c()
        
        datasets = {
            "GroupA": self.format_dataset(group_a_phones, "GroupA"),
            "GroupB": self.format_dataset(group_b_phones, "GroupB"),
            "GroupC": self.format_dataset(group_c_phones, "GroupC")
        }
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, List[Dict[str, str]]], output_dir: Path):
        """
        保存数据集到JSON文件
        
        Args:
            datasets: 数据集字典
            output_dir: 输出目录路径
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for group_name, data in datasets.items():
            file_path = output_dir / f"{group_name.lower()}_dataset.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存 {group_name} 数据集到 {file_path}")


def main():
    """主函数"""
    generator = PhoneNumberGenerator(num_samples=100)
    datasets = generator.generate_all_datasets()
    
    # 保存到data目录
    output_dir = Path(__file__).parent.parent / "data"
    generator.save_datasets(datasets, output_dir)
    
    # 打印统计信息
    print("\n数据集统计:")
    for group_name, data in datasets.items():
        print(f"{group_name}: {len(data)} 条数据")
        print(f"  示例: {data[0]['text']}")


if __name__ == "__main__":
    main()

