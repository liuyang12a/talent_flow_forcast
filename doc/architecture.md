# 架构说明

> 本文档描述 talent_flow 项目的两阶段松耦合 pipeline 架构。

## 1. 总体架构

项目以「池化（Pooling）+ 预测（Forecasting）」两阶段松耦合 pipeline 为核心：

```
原始稀疏网络序列 {G_1, ..., G_T}  (FlowNetwork .pkl)
        │
        ▼  ① 统一输入
┌───────────────────────────────────────┐
│  Pooling 阶段 (可插拔)                  │
│  - CorePeriphery / Louvain / Semantic  │
│  - Truncation / DenseSubgraph          │
│  统一输出: ODMatrixSeries [T, K, K]     │
│  + AssignmentMatrix S + 质量指标        │
└───────────────────────────────────────┘
        │  ② 规范契约 (talent_flow.core.contracts)
        ▼
┌───────────────────────────────────────┐
│  Forecasting 阶段 (可插拔)              │
│  - DMD / DFM / ARIMA / STGNN / Naive   │
│  统一输出: ForecastResult [h, K, K]     │
└───────────────────────────────────────┘
        │  ③ 统一评估
        ▼
┌───────────────────────────────────────┐
│  Evaluation 框架                        │
│  - 池化质量 (内在: 稠密化/重建/谱/模块度) │
│  - 预测精度 (外在: MAE/RMSE/方向准确率)   │
│  - 概率校准 / 显著性检验                 │
└───────────────────────────────────────┘
```

## 2. 核心设计原则

### 2.1 契约式设计

两阶段通过 `talent_flow/core/contracts.py` 中的 dataclass 契约连接，而非对象引用：

- `ODMatrixSeries` `[T, K, K]`：池化输出 / 预测输入
- `AssignmentMatrix` `S ∈ R^{N×K}`：原始节点→超节点映射
- `PoolingResult` / `ForecastResult`：阶段完整产出

这样任一阶段可独立替换，不影响另一阶段。

### 2.2 插件注册表

通过装饰器自动注册，新增方法零侵入核心代码：

```python
@POOLER_REGISTRY.register("core_periphery")
class CorePeripheryPooler(BasePooler): ...

@FORECASTER_REGISTRY.register("dmd")
class DMDForecaster(BaseForecaster): ...

# 使用
pooler = POOLER_REGISTRY.build("core_periphery", n_core=50)
```

新增方法只需：写一个新文件 + 加装饰器，无需改动 `__init__.py` 或脚本。

### 2.3 严格单向依赖

```
core (无依赖)
  ↑
data, utils (依赖 core)
  ↑
pooling, forecasting, evaluation (依赖 core/data/utils)
  ↑
pipeline (依赖 pooling/forecasting/evaluation)
  ↑
viz, scripts (依赖 pipeline)
```

`pooling/` 与 `forecasting/` 互不依赖——这是松耦合的核心。

## 3. 目录结构

```
talent_flow_forcast/
├── talent_flow/                    # 主包
│   ├── core/                       # 数据结构 + 契约 + 注册表 (最底层)
│   │   ├── flow_network.py         #   FlowNetwork 稀疏图
│   │   ├── contracts.py            #   ODMatrixSeries/PoolingResult/ForecastResult
│   │   └── registry.py             #   POOLER/FORECASTER 注册表
│   ├── data/                       # 数据加载与预处理
│   │   ├── loader.py               #   原始 JSONL 流式读取
│   │   ├── preprocess.py           #   生成月度 FlowNetwork .pkl
│   │   ├── flow_network_store.py   #   .pkl 加载/缓存/聚合
│   │   └── company_directory.py    #   公司属性 (行业/地理)
│   ├── pooling/                    # 池化阶段 (可插拔)
│   │   ├── base.py                 #   BasePooler (子类只需 build_assignment)
│   │   ├── core_periphery.py       #   核心-边缘分解 (论文创新)
│   │   ├── community.py            #   Louvain 社群
│   │   ├── semantic.py             #   属性聚合
│   │   ├── truncation.py           #   核心子图截断
│   │   ├── dense_subgraph.py       #   适配旧版稠密子图
│   │   └── legacy_dense_subgraph.py
│   ├── forecasting/                # 预测阶段 (可插拔)
│   │   ├── base.py                 #   BaseForecaster
│   │   ├── windowing.py            #   统一滑动窗口/划分
│   │   ├── naive.py                #   朴素基线
│   │   ├── dmd.py                  #   动态模态分解 (推荐首选)
│   │   ├── factor.py               #   动态因子模型 (DFM)
│   │   ├── arima.py                #   逐对 ARIMA
│   │   ├── stgnn.py                #   时空图神经网络
│   │   └── layers.py               #   图卷积层
│   ├── evaluation/                 # 统一评估体系
│   │   ├── metrics.py              #   基础指标 (统一 target,prediction 顺序)
│   │   ├── pooling_eval.py         #   池化内在质量
│   │   ├── forecast_eval.py        #   预测精度 + 核心/边缘分层
│   │   ├── probabilistic.py        #   PICP/PINAW/CRPS
│   │   ├── significance.py         #   配对t/Wilcoxon
│   │   └── report.py               #   EvaluationReport/对比表
│   ├── pipeline/                   # 两阶段编排
│   │   ├── pipeline.py             #   PoolingForecastPipeline
│   │   └── persistence.py          #   PoolingResult/ForecastResult 存取
│   ├── viz/                        # 可视化
│   └── utils/                      # config/io/seeding/logging
├── scripts/                        # 瘦层 CLI
│   ├── run_pooling.py              #   单独运行池化
│   ├── run_forecast.py             #   单独运行预测
│   ├── run_pipeline.py             #   完整 pipeline (YAML 驱动)
│   ├── run_cross_experiment.py     #   池化×预测全交叉
│   └── configs/default.yaml        #   默认配置
├── tests/                          # 测试 (33 tests)
├── datasets/                       # 数据 (不入版本控制)
├── ckpt/                           # 实验产出 (不入版本控制)
└── doc/                            # 文档
```

## 4. 关键接口

### 4.1 BasePooler

子类只需实现 `build_assignment()`，通用 `pool()` 流程（稀疏聚合 `S^T A S` + 质量评估）共享：

```python
class CorePeripheryPooler(BasePooler):
    def build_assignment(self, networks) -> AssignmentMatrix:
        # 1. 识别核心节点 (k-core 或度数 Top-N)
        # 2. 边缘节点按属性聚合
        # 3. 构建分配矩阵 S
        ...
```

### 4.2 BaseForecaster

子类实现 `fit()` / `predict()`，消费 `ODMatrixSeries`：

```python
class DMDForecaster(BaseForecaster):
    def fit(self, od_series, val_series=None): ...
    def predict(self, od_series) -> ForecastResult: ...
```

### 4.3 Pipeline

```python
pipeline = PoolingForecastPipeline.from_config(cfg_dict)
result = pipeline.run(networks, metrics=["mae", "rmse"])
# result.pooling / result.forecast / result.metrics
```

## 5. 运行方式

### 完整 pipeline (YAML 配置驱动)

```bash
python scripts/run_pipeline.py --config scripts/configs/default.yaml
```

### 分阶段运行（解耦调试）

```bash
# 阶段1: 池化
# 1. 核心-边缘分解（论文创新，Hub独立+长尾聚合）
python scripts/run_pooling.py --pooler core_periphery --n-core 50 --start 2010-01 --end 2019-12

# 2. Louvain 社群发现（拓扑聚类）
python scripts/run_pooling.py --pooler louvain --start 2010-01 --end 2019-12

# 3. 语义聚合（按公司属性：行业/地理）
python scripts/run_pooling.py --pooler semantic --start 2010-01 --end 2019-12

# 4. 核心截断（Top-N 活跃节点）
python scripts/run_pooling.py --pooler truncation --n-core 50 --start 2010-01 --end 2019-12

# 5. 稠密子图（适配旧版三阶段算法）
python scripts/run_pooling.py --pooler dense_subgraph --start 2010-01 --end 2019-12


# 阶段2: 预测（读已池化数据）
python scripts/run_forecast.py --pooled datasets/pooled/core_periphery \
    --forecaster dmd --rank 20 --out ckpt/forecasts/dmd
```

### 全交叉实验

```bash
python scripts/run_cross_experiment.py \
    --poolers truncation core_periphery \
    --forecasters naive dmd dfm \
    --start 2015-01 --end 2017-12
```

## 6. 扩展指南

### 新增一个池化方法

1. 在 `talent_flow/pooling/` 下新建 `my_pooler.py`
2. 继承 `BasePooler`，实现 `build_assignment()`
3. 加 `@POOLER_REGISTRY.register("my_pooler")` 装饰器
4. 在 `pooling/__init__.py` 加一行 `from . import my_pooler`
5. 无需改动任何其他文件

### 新增一个预测方法

同理在 `talent_flow/forecasting/` 下新建文件，继承 `BaseForecaster`，用 `@FORECASTER_REGISTRY.register(...)` 注册。

## 7. 已实现的方法

### 池化 (5 种)
| 方法 | 名称 | 特点 |
|------|------|------|
| 核心-边缘分解 | `core_periphery` | 论文创新，Hub独立+长尾聚合 |
| Louvain 社群 | `louvain` | 拓扑聚类 |
| 语义聚合 | `semantic` | 按属性硬编码 (业务baseline) |
| 核心截断 | `truncation` | Top-N 活跃节点 |
| 稠密子图 | `dense_subgraph` | 适配旧版三阶段算法 |

### 预测 (5 种)
| 方法 | 名称 | 特点 |
|------|------|------|
| 动态模态分解 | `dmd` | 推荐首选，专为短时序高维 |
| 动态因子模型 | `dfm` | PCA因子+岭VAR，大N小T |
| 逐对 ARIMA | `arima` | 统计基线 |
| STGNN | `stgnn` | 深度学习上限探索 |
| 朴素 | `naive` | 持续/历史均值基线 |

## 8. 测试

```bash
python -m pytest tests/ -q   # 33 tests
```

覆盖：契约一致性、评估指标、各池化方法、各预测方法、pipeline 集成、持久化往返。
