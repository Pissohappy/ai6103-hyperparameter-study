# AI6103 Hyperparameter Study

深度学习课程作业：研究超参数对 EfficientNet-B0 的影响。

## 项目结构

```
ai6103-code/
├── config.py          # 配置文件
├── data.py            # 数据加载和预处理
├── model.py           # 模型定义
├── train.py           # 训练工具
├── utils.py           # 绘图和工具函数
├── experiments.py     # 实验运行器
├── download_data.py   # 下载数据集
├── run_all.py         # 主运行脚本
├── requirements.txt   # 依赖
├── data/              # 数据集目录 (下载后)
└── outputs/           # 实验结果输出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 需要先配置 Kaggle API: https://www.kaggle.com/docs/api
python download_data.py
```

### 3. 运行实验

```bash
# 运行所有实验 (耗时较长)
python run_all.py

# 只运行某个 section
python run_all.py --section 3
python run_all.py --section 4
python run_all.py --section 5
python run_all.py --section 6
```

## 实验说明

| Section | 内容 | Epochs | 实验数量 |
|---------|------|--------|----------|
| 2 | 数据预处理 | - | - |
| 3 | 学习率实验 | 15 | 3 |
| 4 | 学习率调度 | 300 | 2 |
| 5 | 权重衰减 | 300 | 2 |
| 6 | Mixup | 300 | 2 |

## 输出文件

每个实验会生成：
- `*.json` - 训练历史数据
- `*.png` - 训练曲线图

## 注意事项

- 使用 **SGD 优化器** (momentum=0.9)，不用 Adam
- 图片 resize 到 **100×100**
- 所有实验都会记录 train/val 的 loss 和 accuracy
