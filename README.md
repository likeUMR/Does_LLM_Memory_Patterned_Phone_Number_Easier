# 电话号码记忆实验

## 项目简介

本项目研究大语言模型（LLM）对于不同模式的电话号码的记忆能力差异。通过对比随机号码、反对称号码和高度重复号码的学习效果，探究模型对规律性数据的记忆偏好。

## 核心问题

**大模型记有规律的电话号码快，还是没有规律的电话号码快？**

## 实验设计

### 1. 数据生成
生成3种Group的100个电话号码：
- **GroupA**: 随机的11位号码
- **GroupB**: 反对称的11位号码（前5位和后5位对称）
- **GroupC**: 高度重复的11位号码（前3个数字相同，中间4个数字相同，后4个数字相同）

数据格式：`第{i}个电话号码是：{xxx}`

### 2. 模型训练
- 基础模型：Qwen3-1.7B-FP8
- 优化器：Adam-8bit
- 训练方法：使用LoRA进行参数高效微调
  - LoRA rank: 32（高容量配置）
  - LoRA alpha: 64
  - Target modules: Attention层 + FFN层（7个模块）
  - 可训练参数：~9.8M（约0.57%）
- 学习率：2e-4
- 训练轮数：10 epochs
- 批次大小：4（梯度累积4步）
- 预期显存占用：~9.5GB（16GB显存下）

### 3. 结果可视化
在一张折线图中绘制3条epoch-loss曲线，对比不同模式的训练效果。

## 环境要求

- Python >= 3.11
- Windows 11
- NVIDIA GPU
- CUDA Toolkit
- 显存：13GB+

## 安装依赖

使用Poetry管理依赖：

```bash
poetry install
```

## 使用方法

### 方式1: 运行完整实验（推荐）

```bash
python run_experiment.py
```

这将自动执行：
1. 生成数据集
2. 训练模型
3. 可视化结果

### 方式2: 分步运行

#### 步骤1: 生成数据集
```bash
python -m src.data_generator
```

#### 步骤2: 训练模型
```bash
python -m src.trainer
```

#### 步骤3: 可视化结果
```bash
python -m src.visualizer
```

## 项目结构

```
Which_Phone_Number_Does_LLM_mem_Better/
├── src/                          # 源代码目录
│   ├── __init__.py              # 包初始化
│   ├── data_generator.py        # 数据生成模块
│   ├── trainer.py               # 训练模块
│   └── visualizer.py            # 可视化模块
├── data/                         # 数据集目录
│   ├── groupa_dataset.json      # GroupA数据集
│   ├── groupb_dataset.json      # GroupB数据集
│   └── groupc_dataset.json      # GroupC数据集
├── models/                       # 训练好的模型
│   ├── GroupA/                  # GroupA模型
│   ├── GroupB/                  # GroupB模型
│   └── GroupC/                  # GroupC模型
├── results/                      # 实验结果
│   ├── training_losses.json     # 训练loss数据
│   └── loss_curves.png          # loss曲线图
├── Qwen3-1___7B-FP8/            # 基础模型目录
├── run_experiment.py             # 主运行脚本
├── pyproject.toml               # Poetry配置文件
├── 实验设计.md                   # 实验设计文档
└── README.md                    # 本文件
```

## 预期结果

实验将生成：
1. 三种类型的电话号码数据集
2. 三个微调后的模型（分别对应三种数据模式）
3. 一个loss对比曲线图

通过分析loss曲线，可以观察哪种模式的电话号码更容易被模型学习和记忆。

## 注意事项

1. 训练过程可能需要较长时间，请耐心等待
2. 确保GPU显存充足（至少13GB）
3. 数据集和模型会占用较多磁盘空间
4. 请先自行下载模型文件夹中的safetensor文件

## 许可证

本项目仅供学习和研究使用。

## 作者

Li Kehang (2353146641@qq.com)

