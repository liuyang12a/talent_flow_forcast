# talent_flow_forcast 项目改造实施方案

> **目标**：将现有实验项目重构为「池化（Pooling）+ 预测（Forecasting）」两阶段松耦合 pipeline，以原始稀疏人才流动网络为研究对象，构建可扩展、可维护的预测实验框架。
>
> **设计原则**：
> 1. 两阶段松耦合，上下游数据接口简洁规范
> 2. 统一评估体系优先（池化 + 预测双层）
> 3. 边界清晰的目录结构，保持可维护性与可扩展性
>
> **重构策略**：渐进重构——保留成熟底层（FlowNetwork 数据结构、算法内核），迁移到新的分层架构，复用 DenseSubgraphExtractor 作为一种 pooling 实现。
>
> **池化统一输出**：节点中心 OD 矩阵序列 `[T, K, K]`
>
> 最后更新：2026-06

---

## 目录

1. [现状诊断与改造目标](#1-现状诊断与改造目标)
2. [核心设计：两阶段松耦合架构](#2-核心设计两阶段松耦合架构)
3. [目录结构设计](#3-目录结构设计)
4. [关键数据接口规范](#4-关键数据接口规范)
5. [统一评估体系设计](#5-统一评估体系设计)
6. [模块详细设计](#6-模块详细设计)
7. [迁移映射：旧代码 → 新代码](#7-迁移映射旧代码--新代码)
8. [实施路线图](#8-实施路线图)
9. [配置与依赖管理](#9-配置与依赖管理)
10. [风险与注意事项](#10-风险与注意事项)

---

## 1. 现状诊断与改造目标

### 1.1 现状诊断

通过深度探索，现有项目存在以下关键问题：

| 问题 | 表现 | 影响 |
|------|------|------|
| **范式不匹配** | 现有 selectors / DenseSubgraphExtractor 输出 edge-centric `[T, E]` 边序列，与论文主路线（节点中心 OD 矩阵 `[T, K, K]`）不一致 | 无法直接支持 DMD/张量分解等需要完整网络结构的方法 |
| **两阶段耦合** | pooling 与 forecasting 逻辑混在 `scripts/run_experiments.py` 的 4 个 phase 中，无清晰接口 | 更换任一阶段方法需改动多处 |
| **无统一评估** | `metrics.py` 仅 MAE/RMSE/MAPE 等基础回归指标，无池化质量评估，无概率预测校准 | 无法公平对比不同池化/预测方法 |
| **无模型注册** | 模型无 registry，新增模型需改 `__init__.py` 和脚本 | 扩展性差 |
| **接口不一致** | 指标函数参数顺序混乱（`mae(pred, target)` vs `calculate_metrics(target, pred)`）；DL 模型 `save/load` 与统计模型不对称 | 易踩坑 |
| **遗留代码** | `src/ts_forecast_framework.py`、`src/run_demo.py`、`src/visualize_results.py` 是旧的独立框架，与新架构无关 | 维护负担 |
| **dense_core 输出不可用** | 24672 条边 → 2.4GB 邻接矩阵，无法直接用于 STGNN | 需重新设计池化产出 |
| **STGNN 潜在 bug** | `_train_loop` 结尾 `self.model = self.model`（no-op），`self.model` 保持 None，`save()` 调用 `self.model.state_dict()` 会崩溃 | 需修复 |
| **文档漂移** | CLAUDE.md / project_summary.md / usage_guide.md 使用过时的 `demo.*` 导入路径 | 误导 |
| **依赖缺失声明** | scipy/seaborn/pmdarima 被使用但未在 pyproject.toml 声明 | 环境不一致 |

### 1.2 改造目标

```
原始稀疏网络序列 {G_1, ..., G_T}  (FlowNetwork .pkl)
        │
        ▼  ① 统一输入
┌───────────────────────────────────────┐
│  Pooling 阶段 (可插拔)                  │
│  - CorePeriphery / Louvain / Semantic  │
│  - 复用 DenseSubgraphExtractor          │
│  统一输出: ODMatrixSeries [T, K, K]     │
│  + PoolingArtifact (分配矩阵 S, 元数据) │
└───────────────────────────────────────┘
        │  ② 规范接口
        ▼
┌───────────────────────────────────────┐
│  Forecasting 阶段 (可插拔)              │
│  - DMD / DFM+BVAR / DeepAR / Moirai    │
│  - 复用 ARIMA / STGNN                   │
│  统一输出: ForecastResult               │
└───────────────────────────────────────┘
        │  ③ 统一评估
        ▼
┌───────────────────────────────────────┐
│  Evaluation 框架 (优先建设)             │
│  - 池化质量评估 (内在)                  │
│  - 预测精度评估 (外在)                  │
│  - 概率校准 / 显著性检验                │
└───────────────────────────────────────┘
```

**三个核心目标**：
1. **松耦合**：Pooling 和 Forecasting 通过规范的 `ODMatrixSeries` 接口连接，任一阶段可独立替换
2. **统一评估**：所有池化方法和预测方法都接入同一套评估体系
3. **清晰边界**：每个包职责单一，可通过新增文件扩展，无需改动核心

---

## 2. 核心设计：两阶段松耦合架构

### 2.1 设计哲学

采用**契约式设计（Design by Contract）**：两阶段通过明确的数据契约（dataclass + 类型注解）连接，而非直接的对象引用。这样：

- Pooling 阶段只需保证输出符合 `ODMatrixSeries` 契约
- Forecasting 阶段只需保证能消费 `ODMatrixSeries` 并产出 `ForecastResult`
- 评估框架对两者都适用

### 2.2 数据流契约

```
FlowNetwork (dict[str, FlowNetwork])
    ↓ BasePooler.pool()
PoolingResult
    ├── od_series: ODMatrixSeries       # [T, K, K] 主输出
    │   ├── matrix: np.ndarray          # [T, K, K]
    │   ├── timestamps: list[str]
    │   ├── supernode_ids: list
    │   └── metadata: dict
    ├── assignment: AssignmentMatrix    # S: [N, K] 原始节点→超节点
    ├── quality: PoolingQualityMetrics  # 内在质量
    └── config: dict
    ↓ (持久化到 datasets/pooled/)
ODMatrixSeries (从磁盘加载)
    ↓ BaseForecaster.fit() / predict()
ForecastResult
    ├── predictions: np.ndarray         # [h, K, K]
    ├── ground_truth: np.ndarray        # [h, K, K]
    ├── prediction_intervals: dict|None # 概率预测
    ├── metadata: dict
    └── metrics: dict
    ↓ Evaluator.evaluate()
EvaluationReport
```

### 2.3 插件式注册机制

采用**注册表模式（Registry Pattern）**，通过装饰器自动注册：

```python
# 注册池化方法
@POOLER_REGISTRY.register("core_periphery")
class CorePeripheryPooler(BasePooler):
    ...

# 注册预测方法
@FORECASTER_REGISTRY.register("dmd")
class DMDForecaster(BaseForecaster):
    ...

# 使用
pooler = POOLER_REGISTRY.build("core_periphery", **config)
forecaster = FORECASTER_REGISTRY.build("dmd", **config)
```

新增方法只需：写一个新文件 + 加装饰器，无需改动任何核心代码或 `__init__.py`。

---

## 3. 目录结构设计

### 3.1 新目录结构

```
talent_flow_forcast/
├── talent_flow/                    # ← 主包（替代顶层散落文件）
│   ├── __init__.py
│   │
│   ├── core/                       # 核心数据结构与契约（最底层，无依赖）
│   │   ├── __init__.py
│   │   ├── flow_network.py         # ← 迁移自顶层 flow_network.py
│   │   ├── contracts.py            # ★ 两阶段数据契约 (ODMatrixSeries 等)
│   │   └── registry.py             # ★ 插件注册表 (POOLER/FORECASTER/EVALUATOR)
│   │
│   ├── data/                       # 数据加载与原始网络处理
│   │   ├── __init__.py
│   │   ├── loader.py               # ← 迁移自 data_loader.py (JobRecord, DataLoader)
│   │   ├── preprocess.py           # ← 迁移自 preprocess.py (生成 flow_networks/*.pkl)
│   │   ├── flow_network_store.py   # ★ FlowNetwork 的磁盘加载/索引/缓存
│   │   └── company_directory.py    # ← 迁移自 statistic.py (公司属性: 行业/地理)
│   │
│   ├── pooling/                    # ★ 池化阶段（可插拔）
│   │   ├── __init__.py
│   │   ├── base.py                 # BasePooler ABC + PoolingResult 契约
│   │   ├── assignment.py           # ★ AssignmentMatrix 构建/验证工具
│   │   ├── core_periphery.py       # ★ 核心-边缘分解池化器
│   │   ├── community.py            # ★ 社群发现池化器 (Louvain/Leiden)
│   │   ├── semantic.py             # ★ 语义/属性聚合池化器 (行业/地理)
│   │   ├── truncation.py           # ★ 核心子图截断 (现有 HighWeight 思路)
│   │   ├── dense_subgraph.py       # ← 迁移自 src/data/dense_subgraph.py (适配新接口)
│   │   └── quality.py              # ★ 池化质量指标 (谱保持/模块度/稠密化/重建误差)
│   │
│   ├── forecasting/                # ★ 预测阶段（可插拔）
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseForecaster ABC + ForecastResult 契约
│   │   ├── windowing.py            # ★ 滑动窗口/数据划分工具 (统一)
│   │   ├── naive.py                # ★ 持续预测/历史均值 (朴素基线)
│   │   ├── arima.py                # ← 迁移自 src/models/statistical/arima.py
│   │   ├── bsts.py                 # ★ 贝叶斯结构时间序列
│   │   ├── bvar.py                 # ★ 贝叶斯 VAR / FAVAR
│   │   ├── factor.py               # ★ 动态因子模型 (DFM)
│   │   ├── dmd.py                  # ★ 动态模态分解
│   │   ├── stgnn.py                # ← 迁移自 src/models/deep_learning/stgnn.py (修复 bug)
│   │   ├── deepar.py               # ★ DeepAR (跨学习)
│   │   ├── foundation.py           # ★ 时间序列基础模型 (Moirai/Chronos 零样本)
│   │   └── reconciliation.py       # ★ MinT 时间层次协调 (可叠加层)
│   │
│   ├── evaluation/                 # ★ 统一评估体系（优先建设）
│   │   ├── __init__.py
│   │   ├── metrics.py              # ← 迁移+扩展自 src/utils/metrics.py (统一参数顺序)
│   │   ├── pooling_eval.py         # ★ 池化质量评估器
│   │   ├── forecast_eval.py        # ★ 预测精度评估器
│   │   ├── probabilistic.py        # ★ 概率预测校准 (PICP/PINAW/校准曲线)
│   │   ├── significance.py         # ★ 统计显著性检验 (配对t/Wilcoxon)
│   │   └── report.py               # ★ EvaluationReport 生成/对比表
│   │
│   ├── pipeline/                   # ★ 两阶段编排
│   │   ├── __init__.py
│   │   ├── pipeline.py             # ★ PoolingForecastPipeline 编排器
│   │   ├── stages.py               # ★ 阶段抽象 (可单独运行某阶段)
│   │   └── persistence.py          # ★ 持久化 (PoolingResult/ForecastResult 存取)
│   │
│   ├── viz/                        # 可视化（从 scripts/analysis 抽出）
│   │   ├── __init__.py
│   │   ├── pooling_viz.py          # 池化结果可视化
│   │   ├── forecast_viz.py         # 预测结果可视化
│   │   └── comparison_viz.py       # 方法对比可视化 (雷达图/帕累托)
│   │
│   └── utils/                      # 通用工具
│       ├── __init__.py
│       ├── config.py               # ★ 配置管理 (dataclass + YAML)
│       ├── io.py                   # 磁盘 I/O 工具
│       ├── seeding.py              # 随机种子管理
│       └── logging.py              # 日志配置
│
├── scripts/                        # 启动脚本（瘦层，只做 CLI 封装）
│   ├── run_pooling.py              # ★ 单独运行池化阶段
│   ├── run_forecast.py             # ★ 单独运行预测阶段
│   ├── run_pipeline.py             # ★ 运行完整两阶段 pipeline
│   ├── run_evaluation.py           # ★ 运行评估对比
│   ├── run_cross_experiment.py     # ★ 池化×预测全交叉实验
│   └── configs/                    # ★ 实验配置 (YAML)
│       ├── default.yaml
│       ├── core_periphery_dmd.yaml
│       └── cross_experiment.yaml
│
├── datasets/                       # 数据（不入版本控制）
│   ├── profiles_jobs_new.jsonl.gz  # 原始简历数据
│   ├── flow_networks/              # ★ preprocess 产出的月度 .pkl (输入)
│   ├── pooled/                     # ★ 池化产出 (各策略子目录)
│   │   ├── core_periphery/
│   │   ├── louvain/
│   │   └── semantic_industry/
│   └── company_attributes/         # 公司属性表 (行业/地理/规模)
│
├── ckpt/                           # 实验产出（不入版本控制）
│   ├── forecasts/                  # 预测结果
│   ├── evaluations/                # 评估报告
│   ├── models/                     # 训练好的模型
│   ├── plots/                      # 可视化
│   └── logs/                       # 运行日志
│
├── tests/                          # ★ 测试（新增）
│   ├── test_contracts.py           # 契约一致性测试
│   ├── test_pooling.py             # 池化方法测试
│   ├── test_forecasting.py         # 预测方法测试
│   └── test_evaluation.py          # 评估指标测试
│
├── doc/                            # 文档
│   ├── refactoring_plan.md         # ★ 本文档
│   ├── architecture.md             # ★ 新架构说明（待写）
│   ├── ... (现有文档保留)
│
├── pyproject.toml                  # 依赖管理
├── README.md
└── CLAUDE.md
```

### 3.2 设计要点说明

**为什么用 `talent_flow/` 顶层包？**
- 现有 `src/` 命名模糊，且与 Python 社区惯例（src layout）不完全一致
- 统一为 `talent_flow` 包后，导入路径清晰：`from talent_flow.pooling import CorePeripheryPooler`
- 顶层散落文件（`flow_network.py`、`data_loader.py` 等）归入包内对应子模块

**包依赖方向（严格单向）**：
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
- `core/` 是最底层，只定义数据结构和契约，不依赖任何其他子包
- `pooling/` 和 `forecasting/` 互不依赖（松耦合的核心）
- `evaluation/` 可独立评估任一阶段

**每个子包的 `base.py`**：
- 定义该包的抽象基类（ABC）和契约 dataclass
- 新增方法只需继承 base 并注册，无需改动其他文件

**`scripts/` 保持瘦层**：
- 只做命令行参数解析 + 调用 `talent_flow.pipeline`
- 不含业务逻辑（消除现有 `run_experiments.py` 中重复的 `prepare_stgnn_data`/`create_windows` 等工具函数）

---

## 4. 关键数据接口规范

### 4.1 核心契约（`talent_flow/core/contracts.py`）

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ODMatrixSeries:
    """池化阶段的统一输出 / 预测阶段的统一输入。
    
    表示 K 个超节点之间，T 个时间步的 OD 流动矩阵序列。
    """
    matrix: np.ndarray              # shape [T, K, K], dtype float32
                                    # matrix[t, i, j] = 时刻 t 从超节点 i 到 j 的流动量
    timestamps: list[str]           # 长度 T, "YYYY-MM" 格式
    supernode_ids: list             # 长度 K, 超节点标识 (可为 int 或 str)
    metadata: dict = field(default_factory=dict)
                                    # 含: pooler_name, K, T, supernode_labels 等

    def __post_init__(self):
        T, K1, K2 = self.matrix.shape
        assert K1 == K2, "OD 矩阵必须为方阵"
        assert len(self.timestamps) == T
        assert len(self.supernode_ids) == K1

    @property
    def T(self) -> int: return self.matrix.shape[0]
    @property
    def K(self) -> int: return self.matrix.shape[1]


@dataclass
class AssignmentMatrix:
    """原始节点到超节点的分配矩阵 S ∈ R^{N×K}。用于反池化与质量评估。"""
    S: np.ndarray                   # [N, K], 硬分配每行一个 1，软分配为概率
    original_node_ids: list         # 长度 N
    supernode_ids: list             # 长度 K
    is_soft: bool = False


@dataclass
class PoolingResult:
    """池化阶段的完整产出。"""
    od_series: ODMatrixSeries       # 主输出
    assignment: AssignmentMatrix    # 原始→超节点映射
    quality: "PoolingQualityMetrics"  # 内在质量指标
    config: dict                    # 池化配置快照
    pooler_name: str


@dataclass
class ForecastResult:
    """预测阶段的统一输出。"""
    predictions: np.ndarray         # [h, K, K] 点预测
    ground_truth: np.ndarray        # [h, K, K] 真实值
    prediction_intervals: Optional[dict] = None
                                    # {"lower": [h,K,K], "upper": [h,K,K], "level": 0.9}
    timestamps: Optional[list[str]] = None
    forecaster_name: str = ""
    metadata: dict = field(default_factory=dict)
```

### 4.2 Pooler 接口（`talent_flow/pooling/base.py`）

```python
from abc import ABC, abstractmethod
from typing import Dict
from talent_flow.core import FlowNetwork, contracts

class BasePooler(ABC):
    """所有池化方法的基类。"""
    
    name: str = "base"
    
    def __init__(self, **config):
        self.config = config
    
    @abstractmethod
    def build_assignment(self, networks: Dict[str, FlowNetwork]) -> contracts.AssignmentMatrix:
        """核心方法：构建原始节点→超节点的分配矩阵 S。
        
        这是池化的本质。不同的池化方法只在此处不同。
        分配矩阵必须是时间不变的（基于时间聚合图或静态属性计算）。
        """
        ...
    
    def pool(self, networks: Dict[str, FlowNetwork]) -> contracts.PoolingResult:
        """通用流程：build_assignment → 聚合 OD 矩阵 → 质量评估。
        
        子类通常不需要重写此方法，只需实现 build_assignment。
        """
        assignment = self.build_assignment(networks)
        od_series = self._aggregate(networks, assignment)
        quality = self._evaluate_quality(networks, assignment, od_series)
        return contracts.PoolingResult(
            od_series=od_series, assignment=assignment,
            quality=quality, config=self.config, pooler_name=self.name
        )
    
    def _aggregate(self, networks, assignment) -> contracts.ODMatrixSeries:
        """通过 S^T A_t S 聚合每个时间步。通用实现。"""
        S = assignment.S
        K = S.shape[1]
        T = len(networks)
        matrix = np.zeros((T, K, K), dtype=np.float32)
        timestamps = []
        for t, (ts, net) in enumerate(sorted(networks.items())):
            A, _ = net.to_adjacency_matrix(node_order=assignment.original_node_ids)
            matrix[t] = S.T @ A @ S
            timestamps.append(ts)
        return contracts.ODMatrixSeries(
            matrix=matrix, timestamps=timestamps,
            supernode_ids=assignment.supernode_ids,
            metadata={"pooler_name": self.name}
        )
    
    def _evaluate_quality(self, networks, assignment, od_series) -> "PoolingQualityMetrics":
        """委托给 evaluation.pooling_eval。"""
        from talent_flow.evaluation import PoolingQualityEvaluator
        return PoolingQualityEvaluator().evaluate(networks, assignment, od_series)
```

**关键设计**：`build_assignment` 是唯一需要子类实现的方法。所有池化方法共享 `pool()` 流程（聚合 + 评估），保证一致性。

### 4.3 Forecaster 接口（`talent_flow/forecasting/base.py`）

```python
class BaseForecaster(ABC):
    """所有预测方法的基类。"""
    
    name: str = "base"
    
    def __init__(self, input_len: int, output_len: int, **config):
        self.input_len = input_len
        self.output_len = output_len
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, od_series: contracts.ODMatrixSeries,
            val_series: Optional[contracts.ODMatrixSeries] = None) -> "BaseForecaster":
        """从 ODMatrixSeries 学习。"""
        ...
    
    @abstractmethod
    def predict(self, od_series: contracts.ODMatrixSeries) -> contracts.ForecastResult:
        """预测未来 output_len 步。"""
        ...
    
    def evaluate(self, test_series: contracts.ODMatrixSeries,
                 metrics: Optional[list[str]] = None) -> dict:
        """通用评估流程：predict → metrics。委托给 evaluation。"""
        from talent_flow.evaluation import ForecastEvaluator
        result = self.predict(test_series)
        return ForecastEvaluator().evaluate(result, metrics)
    
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseForecaster": ...
```

**关键设计**：
- Forecaster 直接消费 `ODMatrixSeries`（池化产出），无需关心池化细节
- 每个方法自行决定如何从 `[T, K, K]` 中学习（展平为面板、提取因子、或逐边预测）
- `evaluate` 通用，保证所有方法用同一套指标

---

## 5. 统一评估体系设计

这是改造的**优先建设项**。评估分两层，所有方法都接入。

### 5.1 评估体系总览

```
evaluation/
├── metrics.py          # 基础指标库 (统一参数顺序: target, prediction)
├── pooling_eval.py     # ★ 池化质量评估
├── forecast_eval.py    # ★ 预测精度评估
├── probabilistic.py    # ★ 概率校准
├── significance.py     # ★ 显著性检验
└── report.py           # ★ 报告生成
```

### 5.2 基础指标库（`metrics.py`）—— 统一参数顺序

**修复现有 bug**：所有指标统一为 `metric(target, prediction)` 顺序。

```python
"""基础回归指标库。所有函数统一参数顺序: (target, prediction)。"""
import numpy as np

def mae(target, prediction, eps=1e-8):
    return np.mean(np.abs(target - prediction))

def rmse(target, prediction, eps=1e-8):
    return np.sqrt(np.mean((target - prediction) ** 2))

def mape(target, prediction, eps=1e-8):
    mask = np.abs(target) > eps
    return np.mean(np.abs(target[mask] - prediction[mask]) / np.abs(target[mask]))

def directional_accuracy(target, prediction, prev):
    """方向准确率: 预测的增减方向是否正确。"""
    true_dir = np.sign(target - prev)
    pred_dir = np.sign(prediction - prev)
    return np.mean(true_dir == pred_dir)

def crps(target, prediction_mean, prediction_std):
    """连续分级概率分数 (概率预测)。"""
    from scipy.stats import norm
    z = (target - prediction_mean) / (prediction_std + 1e-8)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    return prediction_std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

METRIC_REGISTRY = {
    "mae": mae, "rmse": rmse, "mape": mape,
    "directional_accuracy": directional_accuracy, "crps": crps,
    # wape, smape, r2, correlation ...
}

def calculate_metrics(target, prediction, metrics=None, **kwargs):
    """统一入口。target 在前, prediction 在后。"""
    metrics = metrics or ["mae", "rmse", "mape"]
    return {m: METRIC_REGISTRY[m](target, prediction, **kwargs) for m in metrics}
```

### 5.3 池化质量评估（`pooling_eval.py`）

对齐 `pooling_evaluation_framework.md` 的内在评估维度。

```python
@dataclass
class PoolingQualityMetrics:
    """池化内在质量指标。"""
    # 稠密化效果
    original_density: float         # 原始图密度
    pooled_density: float           # 池化后密度
    density_improvement_ratio: float
    zero_reduction: float           # 零元素比例下降
    
    # 信息保留
    reconstruction_error: float     # ||A - S A' S^T||_F / ||A||_F
    
    # 结构保持
    spectral_error: float           # 拉普拉斯特征值相对误差 (前k个)
    modularity: float               # 模块度
    
    # 聚类质量
    cluster_homogeneity: float      # 簇内行业/地理一致性
    
    # 规模
    original_N: int
    pooled_K: int
    compression_ratio: float


class PoolingQualityEvaluator:
    def evaluate(self, networks, assignment, od_series, k_eigen=10) -> PoolingQualityMetrics:
        """计算所有池化质量指标。"""
        ...
```

### 5.4 预测精度评估（`forecast_eval.py`）

```python
class ForecastEvaluator:
    """预测精度评估器。支持分层评估。"""
    
    DEFAULT_METRICS = ["mae", "rmse", "mape", "directional_accuracy"]
    
    def evaluate(self, result: ForecastResult,
                 metrics: Optional[list[str]] = None,
                 core_mask: Optional[np.ndarray] = None) -> dict:
        """评估预测结果。
        
        Args:
            result: ForecastResult
            metrics: 指标列表
            core_mask: [K] 布尔数组, 标记哪些超节点是核心节点。
                       若提供, 则分别评估 core/periphery。
        """
        metrics = metrics or self.DEFAULT_METRICS
        report = {}
        # 整体评估
        report["overall"] = calculate_metrics(
            result.ground_truth, result.predictions, metrics)
        # 分层评估 (核心 vs 边缘)
        if core_mask is not None:
            report["core"] = self._eval_subset(result, metrics, core_mask, core=True)
            report["periphery"] = self._eval_subset(result, metrics, core_mask, core=False)
        # 概率评估
        if result.prediction_intervals is not None:
            report["probabilistic"] = ProbabilisticEvaluator().evaluate(
                result.ground_truth, result.prediction_intervals)
        return report
```

### 5.5 概率校准与显著性检验

```python
# probabilistic.py
class ProbabilisticEvaluator:
    def evaluate(self, target, intervals) -> dict:
        return {
            "picp": self._picp(target, intervals),      # 预测区间覆盖率
            "pinaw": self._pinaw(target, intervals),     # 预测区间归一化平均宽度
        }

# significance.py
class SignificanceTester:
    def paired_t_test(self, results_a, results_b) -> dict:
        """配对 t 检验。"""
    def wilcoxon(self, results_a, results_b) -> dict:
        """Wilcoxon 符号秩检验。"""
```

### 5.6 报告生成（`report.py`）

```python
@dataclass
class EvaluationReport:
    method_name: str
    pooling_quality: Optional[dict]    # 若是池化评估
    forecast_metrics: Optional[dict]   # 若是预测评估
    metadata: dict

class ReportGenerator:
    def generate_comparison_table(self, reports: list[EvaluationReport]) -> pd.DataFrame:
        """生成方法对比表 (Markdown/LaTeX)。"""
    def generate_radar_plot(self, reports, metrics, path): ...
    def generate_pareto_plot(self, reports, metric_x, metric_y, path): ...
```

---

## 6. 模块详细设计

### 6.1 Pooling 方法实现（`talent_flow/pooling/`）

每个池化方法只需实现 `build_assignment`：

**`core_periphery.py`**（论文核心创新）：
```python
@POOLER_REGISTRY.register("core_periphery")
class CorePeripheryPooler(BasePooler):
    name = "core_periphery"
    
    def __init__(self, n_core=50, edge_aggregation="industry", 
                 core_method="k_core", k_core_threshold=5, **kwargs):
        super().__init__(**kwargs)
        ...
    
    def build_assignment(self, networks):
        # 1. 构建时间聚合图
        agg_net = merge_networks(list(networks.values()))
        # 2. 识别核心节点 (k-core 或度数 Top-N)
        core_nodes = self._identify_core(agg_net)
        # 3. 边缘节点按属性聚合
        edge_clusters = self._cluster_periphery(agg_net, core_nodes)
        # 4. 构建 S 矩阵
        return self._build_S(agg_net, core_nodes, edge_clusters)
```

**`community.py`**（复用现有 `CommunitySelector` 逻辑）：
```python
@POOLER_REGISTRY.register("louvain")
class LouvainPooler(BasePooler):
    """基于 Louvain 社群发现的池化。复用现有 community 检测逻辑。"""
```

**`semantic.py`**（业务 baseline）：
```python
@POOLER_REGISTRY.register("semantic_industry")
class SemanticPooler(BasePooler):
    """按公司属性(行业/地理)硬编码聚合。最简单 baseline。"""
```

**`dense_subgraph.py`**（迁移现有 DenseSubgraphExtractor）：
```python
@POOLER_REGISTRY.register("dense_subgraph")
class DenseSubgraphPooler(BasePooler):
    """适配现有 DenseSubgraphExtractor 到新接口。
    通过 to_node_centric 转换其 edge-centric 输出。"""
```

### 6.2 Forecasting 方法实现（`talent_flow/forecasting/`）

**`dmd.py`**（推荐基线）：
```python
@FORECASTER_REGISTRY.register("dmd")
class DMDForecaster(BaseForecaster):
    name = "dmd"
    
    def fit(self, od_series, val_series=None):
        # 展平 [T,K,K] → [K*K, T] 快照矩阵
        X = od_series.matrix.reshape(od_series.T, -1).T  # [K², T]
        # DMD 拟合: X' ≈ A X
        self.A_, self.modes_, self.eigenvalues_ = self._fit_dmd(X, rank=self.config["rank"])
        self.is_fitted = True
        return self
    
    def predict(self, od_series):
        # 模态外推: x_{T+h} = Σ b_j λ_j^h w_j
        preds = self._extrapolate(self.output_len)
        return ForecastResult(predictions=preds.reshape(self.output_len, od_series.K, od_series.K),
                              ground_truth=..., forecaster_name=self.name)
```

**`factor.py` + `bvar.py`**（FAVAR 路线）：
```python
@FORECASTER_REGISTRY.register("dfm_bvar")
class DFMBVARForecaster(BaseForecaster):
    """动态因子模型 + 因子上 BVAR。路线 A。"""
    def fit(self, od_series, val_series=None):
        # 1. 展平为面板 [T, K²]
        panel = od_series.matrix.reshape(od_series.T, -1)
        # 2. PCA 提取 r 个因子
        self.factors_, self.loadings_ = self._extract_factors(panel, r=self.config["r"])
        # 3. 在因子上拟合 BVAR
        self.bvar_ = self._fit_bvar(self.factors_)
```

**`arima.py`**（迁移现有 ARIMA）：
```python
@FORECASTER_REGISTRY.register("arima")
class ARIMAForecaster(BaseForecaster):
    """迁移现有 ARIMAModel, 适配 ODMatrixSeries 输入。逐 OD 对拟合。"""
```

**`stgnn.py`**（迁移现有 STGNN，修复 bug）：
```python
@FORECASTER_REGISTRY.register("stgnn")
class STGNNForecaster(BaseForecaster):
    """迁移现有 STGNNModel。
    修复: _train_loop 中 self.model = self (而非 no-op)。
    适配: 从 ODMatrixSeries 构建 [batch, input_len, K, K] 输入。"""
```

### 6.3 Pipeline 编排（`talent_flow/pipeline/`）

```python
class PoolingForecastPipeline:
    """两阶段编排器。"""
    
    def __init__(self, pooler: BasePooler, forecaster: BaseForecaster,
                 evaluator: Optional[ForecastEvaluator] = None):
        self.pooler = pooler
        self.forecaster = forecaster
        self.evaluator = evaluator or ForecastEvaluator()
    
    def run(self, networks: Dict[str, FlowNetwork],
            train_split: float = 0.7) -> dict:
        # 阶段 1: 池化
        pooling_result = self.pooler.pool(networks)
        
        # 阶段 2: 划分 + 预测
        train_series, test_series = self._split(pooling_result.od_series, train_split)
        self.forecaster.fit(train_series)
        forecast_result = self.forecaster.predict(train_series)
        
        # 评估
        metrics = self.evaluator.evaluate(forecast_result)
        return {"pooling": pooling_result, "forecast": forecast_result, "metrics": metrics}
    
    @staticmethod
    def from_config(config: dict) -> "PoolingForecastPipeline":
        """从配置字典构建 (用于脚本化实验)。"""
        pooler = POOLER_REGISTRY.build(config["pooling"]["name"], **config["pooling"]["params"])
        forecaster = FORECASTER_REGISTRY.build(config["forecasting"]["name"], 
                                                **config["forecasting"]["params"])
        return PoolingForecastPipeline(pooler, forecaster)
```

**关键设计**：
- Pipeline 只是编排，不含业务逻辑
- 两阶段可单独运行（`run_pooling.py` / `run_forecast.py`）
- `from_config` 支持纯配置驱动实验

### 6.4 持久化（`pipeline/persistence.py`）

```python
class PoolingResultStore:
    """PoolingResult 的磁盘存取。"""
    def save(self, result: PoolingResult, path: str):
        # matrix.npy + assignment.npz + quality.json + metadata.json
    def load(self, path: str) -> PoolingResult: ...

class ForecastResultStore:
    def save(self, result: ForecastResult, path: str): ...
    def load(self, path: str) -> ForecastResult: ...
```

这样池化和预测可以**分别运行、中间落盘**，进一步解耦。

---

## 7. 迁移映射：旧代码 → 新代码

| 现有文件 | 去向 | 处理方式 |
|---------|------|---------|
| `flow_network.py` (FlowNetwork) | `talent_flow/core/flow_network.py` | **直接迁移**，成熟稳定 |
| `data_loader.py` (JobRecord, DataLoader) | `talent_flow/data/loader.py` | 直接迁移 |
| `preprocess.py` | `talent_flow/data/preprocess.py` | 直接迁移 |
| `statistic.py` (CompanyDirectory) | `talent_flow/data/company_directory.py` | 迁移，作为语义池化的属性来源 |
| `src/data/dense_subgraph.py` | `talent_flow/pooling/dense_subgraph.py` | 适配新 `BasePooler` 接口 |
| `src/data/selectors.py` | `talent_flow/pooling/truncation.py` 等 | HighWeightSelector → truncation 池化器；CommunitySelector → community 池化器 |
| `src/models/base_model.py` | `talent_flow/forecasting/base.py` | 重构为 `BaseForecaster`，统一 save/load |
| `src/models/statistical/arima.py` | `talent_flow/forecasting/arima.py` | 适配 `ODMatrixSeries` 输入 |
| `src/models/deep_learning/stgnn.py` | `talent_flow/forecasting/stgnn.py` | **修复 `self.model=self` bug**，适配新接口 |
| `src/models/deep_learning/layers.py` | `talent_flow/forecasting/layers.py` | 迁移（STGNN 依赖） |
| `src/utils/metrics.py` | `talent_flow/evaluation/metrics.py` | **统一参数顺序**，扩展指标 |
| `src/data/transforms.py` | `talent_flow/forecasting/windowing.py` | SlidingWindow 等迁入；scalers 视情况 |
| `scripts/config.py` | `talent_flow/utils/config.py` + `scripts/configs/*.yaml` | 配置改 YAML + dataclass |
| `scripts/run_experiments.py` | 拆分: `scripts/run_pipeline.py` + `talent_flow/pipeline/` | 拆解 4 phase，逻辑下沉 |
| `scripts/analysis/*` | `talent_flow/evaluation/` + `talent_flow/viz/` | 分析逻辑→evaluation，绘图→viz |
| `src/ts_forecast_framework.py` | **删除** | 遗留框架，弃用 |
| `src/run_demo.py` | **删除** | 依赖遗留框架 |
| `src/visualize_results.py` | **删除** | 依赖遗留框架 |
| `src/runners/` (空) | **删除** | 从未实现 |

---

## 8. 实施路线图

采用**评估优先、自底向上**的渐进策略，分 5 个阶段。每个阶段可独立验证。

### 阶段 0：骨架搭建（1-2 天）

**目标**：建立目录结构，迁移最底层代码，确保可导入。

- [ ] 创建 `talent_flow/` 包及子包结构（含 `__init__.py`）
- [ ] 迁移 `flow_network.py` → `talent_flow/core/flow_network.py`
- [ ] 编写 `talent_flow/core/contracts.py`（数据契约）
- [ ] 编写 `talent_flow/core/registry.py`（注册表）
- [ ] 迁移 `data_loader.py`、`preprocess.py`、`statistic.py`
- [ ] 更新 `pyproject.toml`（补全 scipy/seaborn 依赖，包名指向 `talent_flow`）
- [ ] 验证：`from talent_flow.core import FlowNetwork, ODMatrixSeries` 可用

### 阶段 1：统一评估体系（2-3 天）⭐ 优先

**目标**：在任何池化/预测方法实现前，先把评估框架搭好。

- [ ] `evaluation/metrics.py`：统一参数顺序的基础指标库
- [ ] `evaluation/pooling_eval.py`：PoolingQualityMetrics + PoolingQualityEvaluator
- [ ] `evaluation/forecast_eval.py`：ForecastEvaluator（含分层评估）
- [ ] `evaluation/probabilistic.py`：PICP/PINAW/CRPS
- [ ] `evaluation/significance.py`：配对 t / Wilcoxon
- [ ] `evaluation/report.py`：EvaluationReport + ReportGenerator
- [ ] `tests/test_evaluation.py`：用合成数据验证指标正确性
- [ ] **验证**：对合成 OD 矩阵运行完整评估流程

### 阶段 2：Pooling 阶段（3-4 天）

**目标**：实现多个可插拔池化方法，产出 `ODMatrixSeries`。

- [ ] `pooling/base.py`：BasePooler + 通用 `pool()` 流程
- [ ] `pooling/assignment.py`：AssignmentMatrix 工具
- [ ] `pooling/semantic.py`：语义聚合（最简单，先做作为 baseline）
- [ ] `pooling/community.py`：Louvain 社群池化（复用现有逻辑）
- [ ] `pooling/core_periphery.py`：**核心-边缘分解**（论文创新）
- [ ] `pooling/truncation.py`：核心子图截断（迁移 HighWeight）
- [ ] `pooling/dense_subgraph.py`：适配现有 DenseSubgraphExtractor
- [ ] `pipeline/persistence.py`：PoolingResult 存取
- [ ] `scripts/run_pooling.py`：单独运行池化的 CLI
- [ ] **验证**：4 种池化方法均产出合法 `ODMatrixSeries`，质量指标可计算

### 阶段 3：Forecasting 阶段（4-5 天）

**目标**：实现多个可插拔预测方法，消费 `ODMatrixSeries`。

- [ ] `forecasting/base.py`：BaseForecaster + ForecastResult
- [ ] `forecasting/windowing.py`：统一滑动窗口/划分工具
- [ ] `forecasting/naive.py`：持续预测/历史均值（朴素基线）
- [ ] `forecasting/dmd.py`：**DMD**（推荐首选基线）
- [ ] `forecasting/factor.py` + `bvar.py`：**DFM+BVAR**（FAVAR 路线）
- [ ] `forecasting/arima.py`：迁移现有 ARIMA
- [ ] `forecasting/stgnn.py`：迁移 STGNN（**修复 bug**）
- [ ] `forecasting/deepar.py`：DeepAR 跨学习
- [ ] `forecasting/foundation.py`：Moirai/Chronos 零样本
- [ ] `forecasting/reconciliation.py`：MinT 协调层
- [ ] `scripts/run_forecast.py`：单独运行预测的 CLI
- [ ] **验证**：各预测方法在固定池化产出上可 fit/predict/evaluate

### 阶段 4：Pipeline 编排与实验（2-3 天）

**目标**：串联两阶段，运行实验。

- [ ] `pipeline/pipeline.py`：PoolingForecastPipeline
- [ ] `pipeline/stages.py`：单阶段运行支持
- [ ] `scripts/run_pipeline.py`：完整 pipeline CLI
- [ ] `scripts/run_cross_experiment.py`：池化×预测全交叉实验
- [ ] `scripts/configs/*.yaml`：实验配置
- [ ] `viz/`：可视化（迁移+扩展 scripts/analysis 绘图）
- [ ] **验证**：完整 pipeline 端到端运行，产出评估报告

### 阶段 5：清理与文档（1-2 天）

- [ ] 删除遗留代码（`ts_forecast_framework.py`、`run_demo.py` 等）
- [ ] 删除/归档旧 `src/` 目录
- [ ] 更新 `README.md`、`CLAUDE.md`（修复 `demo.*` 路径漂移）
- [ ] 编写 `doc/architecture.md`（新架构说明）
- [ ] 补全 `tests/`
- [ ] 更新 `.gitignore`（datasets/、ckpt/ 不入版本控制）

---

## 9. 配置与依赖管理

### 9.1 配置改 YAML + dataclass

```yaml
# scripts/configs/core_periphery_dmd.yaml
pooling:
  name: core_periphery
  params:
    n_core: 50
    core_method: k_core
    k_core_threshold: 5
    edge_aggregation: industry

forecasting:
  name: dmd
  params:
    rank: 20

data:
  flow_networks_dir: datasets/flow_networks
  start_date: "2010-01"
  end_date: "2020-12"

split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

evaluation:
  metrics: [mae, rmse, mape, directional_accuracy]
  core_periphery_split: true
  significance_test: wilcoxon
  n_seeds: 5
```

```python
# talent_flow/utils/config.py
@dataclass
class ExperimentConfig:
    pooling: MethodConfig
    forecasting: MethodConfig
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig": ...
```

### 9.2 依赖补全（`pyproject.toml`）

```toml
[project]
name = "talent-flow"
version = "0.3.0"
dependencies = [
    "numpy>=2.0.0", "pandas>=2.0.0", "scipy>=1.11.0",
    "scikit-learn>=1.6.0", "networkx>=3.0", "matplotlib>=3.9.0",
    "tqdm>=4.68.0", "pyyaml>=6.0",
]

[project.optional-dependencies]
dl = ["torch>=2.0.0", "statsmodels>=0.14.0"]
community = ["python-louvain>=0.16"]
foundation = ["transformers>=4.30.0", "gluonts>=0.14.0"]  # Moirai/Chronos
bayesian = ["pymc>=5.0", "statsmodels>=0.14.0"]            # BVAR/BSTS
dev = ["pytest>=7.0", "ruff>=0.1.0", "seaborn>=0.13.0"]
```

**关键**：按方法族拆分可选依赖，避免安装负担（基础实验只需核心依赖）。

---

## 10. 风险与注意事项

### 10.1 技术风险

| 风险 | 应对 |
|------|------|
| DFM/BVAR 需 `statsmodels`/`pymc`，实现复杂 | 阶段 3 中先做 DMD（最简单），DFM+BVAR 可后续迭代 |
| 基础模型（Moirai）依赖重，API 可能变化 | 放在可选依赖 `[foundation]`，先做零样本包装，失败可降级 |
| `S^T A S` 聚合在 N 很大时内存高 | 分批处理；或先在时间聚合图上算 S，再逐月聚合 |
| STGNN 在 K=70、T=120 上仍可能过拟合 | 作为深度学习上限探索，强正则 + 早停；不作为主基线 |

### 10.2 迁移注意事项

1. **不破坏现有数据**：`datasets/flow_networks/` 是 preprocess.py 产出的原始输入，迁移 preprocess.py 后需保证输出格式不变（仍是 `FlowNetwork` .pkl）
2. **保留 DenseSubgraphExtractor 算法内核**：它有完整的稠密子图三阶段算法，只需适配新接口（通过 `to_node_centric` 转换输出）
3. **FlowNetwork 是契约核心**：所有池化方法都依赖 `net.to_adjacency_matrix()`，迁移时确保此方法行为不变
4. **修复 STGNN bug**：`_train_loop` 结尾必须是 `self.model = self`（让 save() 可用）
5. **统一指标参数顺序**：迁移时注意 `calculate_metrics` 调用处都要改为 `(target, prediction)`

### 10.3 可维护性保障

1. **每个子包有 `base.py`**：新增方法只继承 base + 注册，不改其他文件
2. **契约在 `core/contracts.py` 集中定义**：接口变更只在一处
3. **配置驱动**：实验通过 YAML 配置，无需改代码即可换方法
4. **测试覆盖**：`tests/test_contracts.py` 验证所有方法产出符合契约
5. **依赖单向**：严格遵守包依赖方向，避免循环

---

## 附录：关键决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 池化统一输出 | 节点中心 OD 矩阵 `[T,K,K]` | 与论文主路线（核心-边缘/网络级预测）一致 |
| 重构策略 | 渐进重构 | 保留成熟底层，降低风险 |
| 评估范围 | 预测+池化双层 | 对齐 pooling_evaluation_framework.md |
| 包结构 | `talent_flow/` 顶层包 | 清晰导入路径，统一管理 |
| 注册机制 | 装饰器注册表 | 新增方法零侵入 |
| 配置 | YAML + dataclass | 实验可配置化，无需改代码 |
| 持久化 | PoolingResult/ForecastResult 落盘 | 两阶段可分别运行，进一步解耦 |

---

*本方案为 talent_flow_forcast 项目改造的详细实施计划。遵循「评估优先、自底向上、渐进重构」原则，确保两阶段松耦合、统一评估、目录清晰。*
