# MNIST Hidden Width Study

本项目研究单隐藏层多层感知机在 MNIST 上的隐藏层宽度、分类准确率、理论计算量和 CPU 推理延迟之间的关系。

模型结构固定为：

```text
784 -> h -> 10
```

其中 `h` 是隐藏层神经元数量。实验测试 `4、8、16、32、64、128、256` 七种宽度，并使用三个随机种子独立训练。

## 主要结果

| 隐藏层宽度 | 测试准确率 平均值加减标准差 | 参数量 | MACs | CPU 中位延迟 |
|---:|---:|---:|---:|---:|
| 4 | 84.72% +/- 4.06% | 3,190 | 3,176 | 9.875 us |
| 8 | 92.62% +/- 0.06% | 6,370 | 6,352 | 9.916 us |
| 16 | 95.15% +/- 0.29% | 12,730 | 12,704 | 9.916 us |
| 32 | 96.95% +/- 0.08% | 25,450 | 25,408 | 10.166 us |
| 64 | 97.38% +/- 0.09% | 50,890 | 50,816 | 10.333 us |
| 128 | 97.79% +/- 0.12% | 101,770 | 101,632 | 10.917 us |
| 256 | 97.88% +/- 0.19% | 203,530 | 203,264 | 11.750 us |

以 `h=256` 为当前候选集合中的参考模型：

- `h=64` 的平均准确率下降 0.50 个百分点，参数量和 MACs 减少约 75%，CPU 中位延迟降低约 12.06%。
- `h=32` 的平均准确率下降 0.93 个百分点，参数量和 MACs 减少约 87.5%，CPU 中位延迟降低约 13.48%。
- `h=4` 的随机种子间标准差达到 4.06 个百分点，说明极窄网络对初始化和优化路径较敏感。
- 理论 MACs 的下降没有同比转化为真实延迟下降。对这种微型模型，PyTorch 调用、算子启动和内存访问等固定开销占比较高。

这些结论是当前数据划分、模型结构、训练配置、设备和候选宽度下的经验结果，不构成对 MNIST 最少神经元数量的理论证明。

## 项目结构

```text
mnist-width-study/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── check_env.py
├── model.py
├── data_utils.py
├── train.py
├── run_sweep.py
├── analyze.py
├── benchmark.py
├── analyze_tradeoff.py
├── scripts/
│   └── reproduce.sh
├── tests/
│   └── test_results.py
├── results/
│   ├── raw_results.csv
│   ├── summary_results.csv
│   ├── latency_results.csv
│   └── combined_results.csv
├── figures/
│   ├── width_accuracy.png
│   └── accuracy_latency.png
└── paper/
    ├── MNIST宽度准确率与延迟实证研究.docx
    └── MNIST宽度准确率与延迟实证研究.pdf
```

`data/` 和 `checkpoints/` 会在运行时自动生成，不纳入版本控制。

## 实验环境

原始实验环境：

- MacBook Neo
- Apple A18 Pro
- 8 GB 统一内存
- macOS 26.6.2
- Python 3.11.16
- PyTorch 2.14.0
- 训练设备 MPS
- 延迟测试设备 CPU 单线程

## 安装

### 使用 Conda

```bash
conda env create -f environment.yml
conda activate mnist-study
```

### 使用已有 Python 3.11 环境

```bash
python -m pip install -r requirements.txt
```

检查环境：

```bash
python check_env.py
```

## 运行单个实验

以下命令训练一个隐藏层宽度为 32、随机种子为 42 的模型：

```bash
python train.py --hidden-size 32 --seed 42 --epochs 30 --patience 5
```

模型检查点保存在 `checkpoints/`。

## 复现宽度扫描

先运行宽度 8 至 256 的三个随机种子实验：

```bash
python run_sweep.py \
  --widths 8 16 32 64 128 256 \
  --seeds 42 43 44 \
  --epochs 30 \
  --patience 5
```

宽度 4 的模型收敛较慢，诊断实验允许最多 60 轮，仍使用验证损失连续 5 轮不改善的早停规则：

```bash
python run_sweep.py \
  --widths 4 \
  --seeds 42 43 44 \
  --epochs 60 \
  --patience 5
```

也可以运行：

```bash
bash scripts/reproduce.sh
```

## 生成结果和图表

生成准确率汇总和宽度曲线：

```bash
python analyze.py
```

测量 CPU 单样本推理延迟：

```bash
python benchmark.py
```

生成准确率与延迟综合结果：

```bash
python analyze_tradeoff.py
```

校验原始结果、资源公式与报告中的关键工作点：

```bash
python -m unittest discover -s tests
```

## 实验约定

- MNIST 原始训练集固定划分为 54,000 张训练样本和 6,000 张验证样本。
- 数据划分种子固定为 2026，训练种子使用 42、43、44。
- 输入仅进行张量转换和标准化，不使用数据增强。
- 优化器为 Adam，学习率为 0.001，batch size 为 128。
- 以验证损失最低的检查点进行测试。
- CPU 延迟测试使用 batch size 1、单线程、1000 次预热和 10000 次正式测量。

## 解释限制

- 当前分析只覆盖 MNIST 和单隐藏层 MLP，不能直接推广到卷积网络或复杂数据集。
- 三个随机种子能够发现明显不稳定性，但不足以给出精确置信区间。
- 当前的 0.5 和 1.0 个百分点阈值是测试结果的描述性比较。正式模型选择应使用验证集，测试集只用于最终评估。
- 延迟结果只适用于本项目的 PyTorch eager、CPU、batch size 1 测试条件，不包含数据读取和预处理。
- 本项目没有提出新的网络结构或训练方法，适合作为可复现预实验和后续动态宽度研究的起点。

## 公开说明

项目默认按私有研究仓库使用。公开前应补充作者信息并选择合适的软件许可证，同时检查目标会议对匿名、预印本和代码公开的要求。
