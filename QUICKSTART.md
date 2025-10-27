# 快速开始指南

## 前置条件

1. 已安装 Python 3.11+
2. 已安装 Poetry
3. 已安装 CUDA Toolkit
4. 显卡显存 >= 13GB

## 安装步骤

### 1. 安装依赖

```bash
poetry install
```

或使用 pip:

```bash
pip install -r requirements.txt
```

### 2. 验证环境

运行示例代码验证环境是否配置正确：

```bash
python examples/example_usage.py
```

## 运行实验

### 完整实验流程（推荐）

```bash
python run_experiment.py
```

这将自动完成：
- 生成3种类型的电话号码数据集
- 训练3个微调模型
- 生成loss对比图

### 分步运行

如果你想分步运行：

**步骤1: 生成数据**
```bash
python -m src.data_generator
```

**步骤2: 训练模型**
```bash
python -m src.trainer
```

**步骤3: 可视化结果**
```bash
python -m src.visualizer
```

## 输出结果

实验完成后，你会得到：

1. **data/** - 数据集文件
   - `groupa_dataset.json` - 随机号码数据集
   - `groupb_dataset.json` - 反对称号码数据集
   - `groupc_dataset.json` - 高度重复号码数据集

2. **models/** - 训练好的模型
   - `GroupA/` - GroupA微调模型
   - `GroupB/` - GroupB微调模型
   - `GroupC/` - GroupC微调模型

3. **results/** - 实验结果
   - `training_losses.json` - 训练loss数据
   - `loss_curves.png` - loss对比曲线图

## 常见问题

### Q: 训练时间太长怎么办？
A: 可以减少训练轮数或样本数量，修改 `run_experiment.py` 中的参数。

### Q: 显存不足怎么办？
A: 减小 `batch_size` 或增加 `gradient_accumulation_steps`。

### Q: 如何调整学习率？
A: 修改 `trainer.py` 中的 `lr` 参数，或使用配置文件 `config.py`。

### Q: 想要更多样本怎么办？
A: 修改 `data_generator.py` 中的 `num_samples` 参数（默认100）。

## 下一步

查看 `README.md` 了解更多详细信息。

