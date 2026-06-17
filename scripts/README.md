# 实验脚本目录

本目录包含用于运行 ARIMA 与 STGNN 时间序列预测对比实验的脚本和工具。

## 目录结构

```
scripts/
├── config.py                 # 实验配置中心
├── run_experiments.py       # 主实验执行脚本
├── README.md                # 本文件
└── analysis/                # 分析工具包
    ├── __init__.py
    ├── series_characteristics.py  # 序列特征分析
    ├── compare_models.py          # 模型对比分析
    ├── visualizations.py          # 可视化工具
    └── reports.py                 # 报告生成
```

## 快速开始

### 1. 运行完整实验流程

```bash
# 运行所有阶段（生成序列、运行实验、分析结果、生成可视化）
python scripts/run_experiments.py --phase all
```

### 2. 分阶段运行

```bash
# 仅生成时间序列数据
python scripts/run_experiments.py --phase generate

# 仅运行模型实验（需要先生成序列）
python scripts/run_experiments.py --phase run

# 仅分析结果
python scripts/run_experiments.py --phase analyze

# 仅生成可视化
python scripts/run_experiments.py --phase visualize
```

## 实验阶段说明

### Phase 1: 生成实验时间序列

从月度流动网络数据中选择代表性时间序列，保存到 `datasets/experiment_series/` 目录。

三种序列选择策略：
- **HighWeightSelector**: 选择总流量最大的边（最多300条）
- **HubNodeSelector**: 选择中心节点周边的边（最多300条）
- **CommunitySelector**: 选择社区内部的边（最多300条）

同时计算每条序列的特征（趋势性、季节性、波动性等）。

### Phase 2: 运行对比实验

对每种选择策略获取的序列，分别运行 ARIMA 和 STGNN 模型：

**ARIMA 模型**:
- 尝试多种参数组合: (1,1,1), (2,1,2), (1,1,2), (2,1,1)
- 为每条序列选择最优参数

**STGNN 模型**:
- 空间卷积: GCN
- 时序建模: GRU
- 隐藏层: 32维，2层

评估指标: MAE, RMSE, MAPE, R²

### Phase 3: 分析结果

自动生成分析报告：
- JSON 格式: `ckpt/experiment_report.json`
- Markdown 格式: `ckpt/experiment_report.md`
- CSV 表格: `ckpt/model_comparison_summary.csv`

分析内容包括：
- 整体性能对比
- 按选择器类型分组对比
- 按序列特征分组对比
- 最优模型统计

### Phase 4: 生成可视化

生成以下图表：
- `model_comparison_mae.png`: 按选择器分组的 MAE 对比
- `model_comparison_mape.png`: 按选择器分组的 MAPE 对比
- `series_characteristics.png`: 序列特征分布
- `predictability_analysis.png`: 可预测性与特征关系

## 配置文件

实验配置集中在 `config.py` 中，主要配置项包括：

### 数据配置
```python
DATA_CONFIG = {
    "date_range": {"start": "2010-01", "end": "2020-12"},
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
}
```

### 模型配置
```python
MODEL_CONFIG = {
    "arima": {
        "orders": [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)]
    },
    "stgnn": {
        "spatial_types": ["gcn", "chebyshev"],
        "temporal_types": ["gru", "lstm"],
        "hidden_dims": [32, 64],
        "num_layers": [2, 3]
    }
}
```

## 实验结果

所有实验结果保存在 `ckpt/` 目录：

```
ckpt/
├── experiment.log              # 实验日志
├── models/                     # 保存的模型
├── predictions/                # 预测结果
├── metrics/
│   ├── experiment_results.json # 详细结果
│   ├── experiment_results.csv  # CSV格式结果
│   └── experiment_report.json  # 分析报告
└── plots/                      # 可视化图表
    ├── model_comparison_mae.png
    ├── model_comparison_mape.png
    ├── series_characteristics.png
    └── predictability_analysis.png
```

## 使用分析工具包

### 分析序列特征

```python
from scripts.analysis import SeriesAnalyzer

analyzer = SeriesAnalyzer(period=12)

# 分析单个序列
characteristics = analyzer.analyze_all(series_array)

# 分析序列集合
all_characteristics = analyzer.analyze_series_collection(series_dict)
```

### 对比模型性能

```python
from scripts.analysis import ModelComparator

comparator = ModelComparator().load_results('ckpt/metrics/experiment_results.csv')

# 按指标对比
comparison = comparator.compare_by_metric('mae')

# 按选择器类型对比
by_selector = comparator.compare_by_series_type()

# 生成汇总表
summary = comparator.generate_summary_table()
```

### 生成可视化

```python
from scripts.analysis import (
    plot_model_comparison,
    plot_series_characteristics,
    plot_predictability_analysis
)

# 模型对比图
fig = plot_model_comparison(results_df, metric='mae')

# 序列特征分布
fig = plot_series_characteristics(characteristics)

# 可预测性分析
fig = plot_predictability_analysis(results_df, characteristics)
```

### 生成报告

```python
from scripts.analysis import ReportGenerator

generator = ReportGenerator('ckpt/metrics/experiment_results.csv')

# 生成所有格式报告
generator.generate_all_reports('ckpt/', formats=['json', 'markdown', 'csv'])
```

## 自定义实验

如需修改实验配置，可以：

1. 直接修改 `config.py` 中的配置项
2. 创建自定义配置文件并通过 `--config` 参数加载

```bash
python scripts/run_experiments.py --config my_config.json --phase all
```

## 注意事项

1. **内存限制**: STGNN 模型在序列数量过多时可能消耗大量内存，已设置上限为50个节点
2. **运行时间**: 完整实验可能需要较长时间（数小时），建议分阶段运行
3. **CUDA**: STGNN 训练会自动检测并使用 GPU，如无 GPU 则使用 CPU
4. **复现性**: 实验设置了随机种子（42），确保结果可复现
