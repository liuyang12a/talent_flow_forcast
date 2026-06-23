# CLAUDE.md - Talent Flow Forecast

## 项目概述

人才流动网络预测研究项目。以原始稀疏人才流动网络（月度公司间跳槽关系）为研究对象，按「**池化（Pooling）+ 预测（Forecasting）**」两阶段松耦合 pipeline 构建预测实验：

1. **Pooling 阶段**：将高维稀疏网络稠密化为低维 OD 矩阵序列 `[T, K, K]`，解决空间稀疏性与时间不对齐
2. **Forecasting 阶段**：在短时序（T≈120）OD 矩阵上预测未来网络动态
3. **统一评估**：池化质量 + 预测精度双层评估体系

> 详细架构见 [`doc/architecture.md`](doc/architecture.md)，改造历程见 [`doc/refactoring_plan.md`](doc/refactoring_plan.md)。

## 技术栈

### 核心依赖
- **Python**: 3.11+（PyTorch 不支持 3.14）
- **包管理**: `uv`
- **数据处理**: numpy, pandas, scipy, scikit-learn
- **深度学习**: torch 2.5.1+cu121 (CUDA 12.1)
- **统计模型**: statsmodels (ARIMA)
- **图算法**: networkx, python-louvain
- **可视化**: matplotlib
- **测试**: pytest

### 硬件环境
- **GPU**: NVIDIA RTX 4060 Laptop GPU
- **CUDA**: 12.1+
- **内存**: 建议 16GB+

## 项目结构

```
talent_flow/              # 主包（两阶段 pipeline）
├── core/                 # FlowNetwork + 数据契约 + 注册表 (最底层)
├── data/                 # 数据加载/预处理/FlowNetworkStore
├── pooling/              # 池化阶段 (5 种可插拔方法)
├── forecasting/          # 预测阶段 (5 种可插拔方法)
├── evaluation/           # 统一评估体系 (池化+预测双层)
├── pipeline/             # 两阶段编排 + 持久化
├── viz/                  # 可视化
└── utils/                # config/io/seeding/logging
scripts/                  # 瘦层 CLI (run_pipeline/run_pooling/run_forecast/run_cross_experiment)
tests/                    # 33 tests
datasets/flow_networks/   # 月度 FlowNetwork .pkl (preprocess 产出, 输入)
```

## 核心数据流

```
FlowNetwork dict ──pool()──> ODMatrixSeries [T,K,K] ──fit/predict──> ForecastResult
                                     │                                    │
                              (talent_flow.core.contracts)         evaluation
```

- **契约**：`talent_flow/core/contracts.py` 中的 dataclass（ODMatrixSeries / PoolingResult / ForecastResult）是两阶段唯一耦合点
- **注册表**：`@POOLER_REGISTRY.register("name")` / `@FORECASTER_REGISTRY.register("name")`，新增方法零侵入
- **依赖单向**：core ← data/utils ← pooling/forecasting/evaluation ← pipeline ← scripts

## 已实现的方法

### 池化 (POOLER_REGISTRY)
- `core_periphery`：核心-边缘分解（论文创新，Hub 独立 + 长尾聚合）
- `louvain`：社群发现聚合
- `semantic`：按公司属性（行业/地理）聚合
- `truncation`：核心子图截断（Top-N 活跃节点）
- `dense_subgraph`：适配旧版三阶段稠密子图算法

### 预测 (FORECASTER_REGISTRY)
- `dmd`：动态模态分解（推荐首选，专为短时序高维）
- `dfm`：动态因子模型（PCA 因子 + 岭回归 VAR，大 N 小 T）
- `arima`：逐 OD 对 ARIMA（统计基线）
- `stgnn`：时空图神经网络（深度学习上限探索）
- `naive`：持续预测 / 历史均值（朴素基线）

## 运行方式

```bash
# 完整 pipeline（YAML 配置驱动）
python scripts/run_pipeline.py --config scripts/configs/default.yaml

# 分阶段运行（解耦调试）
python scripts/run_pooling.py --pooler core_periphery --n-core 50 --start 2010-01 --end 2019-12
python scripts/run_forecast.py --pooled datasets/pooled/core_periphery --forecaster dmd --rank 20

# 池化×预测全交叉实验
python scripts/run_cross_experiment.py --poolers truncation core_periphery --forecasters naive dmd dfm

# 测试
python -m pytest tests/ -q
```

## 数据说明

- **原始数据**：`datasets/profiles_jobs_new.jsonl.gz`（500万美国劳工简历）
- **月度网络**：`datasets/flow_networks/YYYY-MM.pkl`（FlowNetwork 对象，由 `talent_flow.data.preprocess` 产出）
- **池化产出**：`datasets/pooled/<pooler_name>/`
- **实验产出**：`ckpt/<experiment_name>/`

> 注意：legacy pickle 的 `flow_network.FlowNetwork` 已通过模块别名重定向到 `talent_flow.core.flow_network`，旧 .pkl 可直接读取。

## 扩展指南

新增方法只需：
1. 在对应包（`pooling/` 或 `forecasting/`）下新建 `.py` 文件
2. 继承 `BasePooler` / `BaseForecaster`，实现抽象方法
3. 加 `@<REGISTRY>.register("name")` 装饰器
4. 在包 `__init__.py` 加一行 `from . import my_module`

无需改动任何核心代码或脚本。

## 已知问题

- PyTorch 不支持 Python 3.14，需用 3.11
- 大规模网络（N>50万）池化时，谱保持度评估自动降级（采样/跳过），避免 O(N³) 特征分解
- STGNN 在 T=120 短时序上需强正则 + 早停，不作主基线
