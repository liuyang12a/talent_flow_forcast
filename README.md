# Talent Flow - 人才流动网络预测

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)]()

以原始稀疏人才流动网络为研究对象，按「**池化（Pooling）+ 预测（Forecasting）**」两阶段松耦合 pipeline 构建预测实验：池化阶段将高维稀疏网络稠密化为低维 OD 矩阵序列，预测阶段在短时序 OD 矩阵上预测未来网络动态。

> 📐 架构详情见 [`doc/architecture.md`](doc/architecture.md)

## 特性

- 🔌 **两阶段松耦合**：Pooling 与 Forecasting 通过规范数据契约（`ODMatrixSeries`）连接，任一阶段可独立替换
- 🧩 **插件式注册**：`@POOLER_REGISTRY.register(...)` / `@FORECASTER_REGISTRY.register(...)`，新增方法零侵入核心代码
- 📊 **统一评估**：池化内在质量（稠密化/重建误差/谱保持/模块度）+ 预测精度（MAE/RMSE/方向准确率）+ 概率校准 + 显著性检验
- 🗂️ **清晰边界**：每个子包职责单一，依赖严格单向，可维护可扩展
- 🧪 **测试覆盖**：33 个测试覆盖契约、评估、各方法、pipeline 集成

## 快速开始

### 环境准备

```bash
# 安装 uv（现代 Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/Mac
# 或见 https://docs.astral.sh/uv/

# 同步依赖（含可选依赖组）
uv sync                 # 核心
uv sync --extra dl      # +torch/statsmodels (STGNN/ARIMA)
uv sync --extra community  # +python-louvain (Louvain 池化)
uv sync --all-extras    # 全部
```

### 运行

```bash
# 1. 预处理原始简历数据 -> 月度 FlowNetwork（仅需一次）
uv run python -c "from talent_flow.data.preprocess import build_and_save_monthly_flow_networks; build_and_save_monthly_flow_networks('datasets/profiles_jobs_new.jsonl.gz', 'datasets/flow_networks')"

# 2. 完整两阶段 pipeline（YAML 配置驱动）
uv run python scripts/run_pipeline.py --config scripts/configs/default.yaml

# 3. 分阶段运行（解耦调试）
uv run python scripts/run_pooling.py --pooler core_periphery --n-core 50 --start 2010-01 --end 2019-12
uv run python scripts/run_forecast.py --pooled datasets/pooled/core_periphery --forecaster dmd --rank 20

# 4. 池化×预测全交叉实验
uv run python scripts/run_cross_experiment.py --poolers truncation core_periphery --forecasters naive dmd dfm

# 测试
uv run pytest -q
```

## 项目结构

```
talent_flow/              # 主包
├── core/                 # FlowNetwork + 数据契约 + 注册表 (最底层)
├── data/                 # 数据加载/预处理/FlowNetworkStore
├── pooling/              # 池化阶段 (5 种可插拔方法)
├── forecasting/          # 预测阶段 (5 种可插拔方法)
├── evaluation/           # 统一评估体系 (池化+预测双层)
├── pipeline/             # 两阶段编排 + 持久化
├── viz/                  # 可视化
└── utils/                # config/io/seeding/logging
scripts/                  # 瘦层 CLI + YAML 配置
tests/                    # 测试
datasets/flow_networks/   # 月度 FlowNetwork .pkl (输入)
doc/                      # 文档 (architecture.md / refactoring_plan.md)
```

## 已实现的方法

### 池化（5 种）

| 方法 | 注册名 | 特点 |
|------|--------|------|
| 核心-边缘分解 | `core_periphery` | **论文创新**，Hub 独立 + 长尾按属性聚合 |
| Louvain 社群 | `louvain` | 拓扑聚类，高内聚低耦合 |
| 语义聚合 | `semantic` | 按公司属性（行业/地理）硬编码 |
| 核心截断 | `truncation` | Top-N 活跃节点 |
| 稠密子图 | `dense_subgraph` | 适配旧版三阶段算法 |

### 预测（5 种）

| 方法 | 注册名 | 特点 |
|------|--------|------|
| 动态模态分解 | `dmd` | **推荐首选**，专为短时序高维设计 |
| 动态因子模型 | `dfm` | PCA 因子 + 岭回归 VAR，大 N 小 T |
| 逐对 ARIMA | `arima` | 统计基线 |
| STGNN | `stgnn` | 时空图神经网络（深度学习上限探索） |
| 朴素 | `naive` | 持续预测 / 历史均值 |

## 扩展指南

新增一个池化或预测方法只需：

1. 在 `talent_flow/pooling/` 或 `talent_flow/forecasting/` 下新建 `.py`
2. 继承 `BasePooler` / `BaseForecaster`，实现抽象方法
3. 加 `@<REGISTRY>.register("name")` 装饰器
4. 在包 `__init__.py` 加一行 `from . import my_module`

无需改动任何核心代码或脚本。详见 [`doc/architecture.md`](doc/architecture.md#6-扩展指南)。

## 数据格式

- **原始输入**：`datasets/profiles_jobs_new.jsonl.gz`（gzipped JSONL，500万美国劳工简历）
- **月度网络**：`datasets/flow_networks/YYYY-MM.pkl`（`FlowNetwork` 对象）
- **池化产出**：`datasets/pooled/<pooler>/`（od_matrix.npy + assignment.npz + metadata.json）
- **预测产出**：`ckpt/forecasts/<forecaster>/`（predictions.npy + ground_truth.npy + metrics.json）

> 注：legacy pickle 的 `flow_network.FlowNetwork` 已通过模块别名重定向到 `talent_flow.core.flow_network`，旧 .pkl 可直接读取。

## 配置

实验配置通过 YAML 管理（`scripts/configs/default.yaml`），支持纯配置驱动：

```yaml
pooling:
  name: core_periphery
  params: { n_core: 50, core_method: degree }
forecasting:
  name: dmd
  params: { input_len: 12, output_len: 1, rank: 20 }
split: { train_ratio: 0.7, val_ratio: 0.15, test_ratio: 0.15 }
evaluation: { metrics: [mae, rmse, mape, directional_accuracy] }
```

## 许可证

MIT License - 见 [LICENSE](LICENSE)
