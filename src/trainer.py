"""
训练模块 - 实现SFT训练逻辑
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


class PhoneNumberDataset(Dataset):
    """电话号码数据集类"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            编码后的样本
        """
        text = self.texts[idx]
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class SSTrainer:
    """SFT训练器"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "models",
        lr: float = 2e-4,
        epochs: int = 10,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 128
    ):
        """
        初始化训练器
        
        Args:
            model_path: 基础模型路径
            output_dir: 输出目录
            lr: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        # 初始化tokenizer和model
        print(f"加载模型和分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置Lora - 增强版本
        # 增加rank以提升模型表达能力，扩展到FFN层以覆盖更多参数
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # 从8增加到32，参数容量提升给16倍
            lora_alpha=64,  # alpha = 2 * r，保持良好训练稳定性
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention层
                "gate_proj", "up_proj", "down_proj"       # FFN层（新增）
            ]
        )
        
        # 用于记录loss
        self.training_losses = []
    
    def load_dataset(self, dataset_path: str) -> PhoneNumberDataset:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集JSON文件路径
            
        Returns:
            电话号码数据集对象
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        return PhoneNumberDataset(texts, self.tokenizer, self.max_length)
    
    def train_group(self, group_name: str, dataset_path: str):
        """
        训练单个组的数据
        
        Args:
            group_name: 组名称
            dataset_path: 数据集路径
        """
        print(f"\n开始训练 {group_name}...")
        
        # 加载数据集
        dataset = self.load_dataset(dataset_path)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,  # 使用dtype替代torch_dtype
            device_map="auto",
            trust_remote_code=True
        )
        
        # 应用LoRA
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        
        # 设置训练参数 - 开启每epoch评估
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / group_name),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            dataloader_num_workers=0,  # Windows兼容性
            eval_strategy="epoch",  # 每个epoch后评估
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建训练器 - 使用train_dataset作为eval_dataset（实验数据集一致）
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # 评估也使用同一数据集
            data_collator=data_collator,
        )
        
        # 训练前评估：记录epoch 0的loss（未训练基线）
        print("正在评估未训练模型的loss...")
        initial_loss = trainer.evaluate()
        initial_loss_value = initial_loss.get('eval_loss', None)
        if initial_loss_value is None:
            # 如果没有eval_loss，手动计算一次
            trainer.model.eval()
            total_loss = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                inputs = {k: v.unsqueeze(0).to(trainer.model.device) for k, v in sample.items()}
                with torch.no_grad():
                    outputs = trainer.model(**inputs)
                    total_loss += outputs.loss.item()
            initial_loss_value = total_loss / len(dataset)
            trainer.model.train()
        
        print(f"Epoch 0 (未训练) Loss: {initial_loss_value:.4f}")
        
        # 训练
        trainer.train()
        
        # 保存模型
        model.save_pretrained(str(self.output_dir / group_name))
        self.tokenizer.save_pretrained(str(self.output_dir / group_name))
        
        # 记录每个epoch的评估loss（不是训练loss）
        logs = trainer.state.log_history
        epoch_losses = []
        
        # 将初始loss添加到开头（epoch 0）
        epoch_losses.append(initial_loss_value)
        
        # 提取每个epoch的评估loss（eval_loss）
        for log in logs:
            if 'eval_loss' in log:
                epoch_losses.append(log['eval_loss'])
        
        print(f"记录到 {len(epoch_losses)} 个epoch的loss值")
        
        self.training_losses.append({
            'group': group_name,
            'losses': epoch_losses
        })
        
        print(f"{group_name} 训练完成!")
    
    def train_all_groups(self, dataset_dir: str):
        """
        训练所有组
        
        Args:
            dataset_dir: 数据集目录
        """
        dataset_dir = Path(dataset_dir)
        
        groups = ['GroupA', 'GroupB', 'GroupC']
        for group in groups:
            dataset_path = dataset_dir / f"{group.lower()}_dataset.json"
            if dataset_path.exists():
                self.train_group(group, str(dataset_path))
            else:
                print(f"警告: 找不到数据集 {dataset_path}")
    
    def save_losses(self, output_path: str):
        """
        保存训练loss
        
        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_losses, f, ensure_ascii=False, indent=2)
        
        print(f"\n训练loss已保存到: {output_path}")


def main():
    """主函数"""
    # 配置路径
    model_path = Path(__file__).parent.parent / "Qwen3-1___7B"  # 使用非FP8版本
    dataset_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    results_dir = Path(__file__).parent.parent / "results"
    
    # 检查模型路径
    if not model_path.exists():
        print(f"错误: 找不到模型路径 {model_path}")
        return
    
    # 检查数据集
    if not dataset_dir.exists():
        print(f"错误: 找不到数据集目录 {dataset_dir}")
        print("请先运行 data_generator.py 生成数据集")
        return
    
    # 创建训练器
    trainer = SSTrainer(
        model_path=str(model_path),
        output_dir=str(models_dir),
        lr=2e-4,
        epochs=10,
        batch_size=4,
        gradient_accumulation_steps=4
    )
    
    # 训练所有组
    trainer.train_all_groups(str(dataset_dir))
    
    # 保存loss
    trainer.save_losses(str(results_dir / "training_losses.json"))


if __name__ == "__main__":
    main()

