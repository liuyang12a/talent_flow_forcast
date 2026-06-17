# CLAUDE.md - Talent Flow Forecast

## 项目概述

人才流动网络分析工具 - 从招聘数据中提取企业员工流动网络，并提供完整的时间序列预测框架。

项目包含两个主要部分：
1. **数据预处理管道**：从原始招聘数据构建月度员工流动网络
2. **时间序列预测框架**：对流动网络中的边（公司间跳槽关系）进行预测

## 技术栈

### 核心依赖
- **Python**: 3.11+（推荐）
- **包管理**: `uv`（现代Python包管理器）
- **数据处理**: `numpy`, `pandas`
- **机器学习**: `scikit-learn`
- **深度学习**: `torch` 2.5.1+cu121 (CUDA 12.1)
- **统计模型**: `statsmodels` (ARIMA)
- **图算法**: `networkx`, `python-louvain` (社区检测)
- **可视化**: `matplotlib`

### 硬件环境
- **GPU**: NVIDIA RTX 4060 Laptop GPU
- **CUDA**: 12.1+
- **内存**: 建议 16GB+

## 核心功能

### 1. 数据预处理 (preprocess.py)
- 从JSONL招聘数据解析员工职业轨迹
- 识别公司间跳槽事件
- 构建月度FlowNetwork对象
- 输出：datasets/flow_networks/*.pkl

### 2. 时间序列预测框架 (src/)

#### 数据模块 (src/data/)
- **BaseDataset**: 抽象基类，支持多模态时序数据
- **FlowNetworkDataset**: FlowNetwork专用数据集
- **Transforms**: 数据变换（ZScore、MinMax、差分、滑动窗口、时间特征编码）
- **Selectors**: 典型序列选择器
  - HighWeightSelector: 高权重边选择
  - HubNodeSelector: 中心节点周边边选择
  - CommunitySelector: 社区内边选择（Louvain算法）

#### 模型模块 (src/models/)
- **BaseTimeSeriesModel**: 时序模型统一接口
- **BaseStatisticalModel**: 统计模型基类
- **BaseDeepLearningModel**: PyTorch深度学习基类
- **ARIMA**: 自回归积分滑动平均模型
- **STGNN**: 空间-时间图神经网络
  - 空间卷积: GCN、Chebyshev多项式
  - 时序建模: GRU、LSTM、TCN、Attention

#### 评估指标 (src/utils/)
MAE, MSE, RMSE, MAPE, WAPE, SMAPE, R², Correlation

## 目录结构

```
talent_flow_forcast/
├── datasets/                      # 数据目录（git忽略）
│   ├── flow_networks/            # 月度网络数据(.pkl)
│   └── profiles_jobs_new.jsonl.gz # 原始招聘数据
├── src/                          # 时间序列预测框架
│   ├── data/                     # 数据模块（代码）
│   │   ├── base_dataset.py
│   │   ├── flow_network_dataset.py
│   │   ├── transforms.py
│   │   └── selectors.py
│   ├── models/                   # 模型模块
│   │   ├── base_model.py
│   │   ├── statistical/
│   │   │   └── arima.py
│   │   └── deep_learning/
│   │       ├── layers.py
│   │       └── stgnn.py
│   ├── utils/
│   │   └── metrics.py
│   ├── experiments/
│   │   ├── run_arima.py
│   │   ├── run_stgnn.py
│   │   └── test_framework.py
│   ├── ts_forecast_framework.py  # 旧版预测框架
│   ├── run_demo.py
│   └── README.md
├── doc/                           # 文档
│   ├── milestones/progress.md
│   ├── project_summary.md
│   └── usage_guide.md
├── preprocess.py                  # 数据预处理
├── data_loader.py                 # 数据加载工具
├── statistic.py                   # 统计分析
├── flow_network.py                # FlowNetwork类
├── pyproject.toml
└── CLAUDE.md                      # 本文件
```

## 开发里程碑

### ✅ 已完成
- [x] 数据预处理管道（月度FlowNetwork构建）
- [x] 深度学习环境配置（CUDA + PyTorch）
- [x] 数据模块重构（基类、FlowNetwork数据集、变换器）
- [x] 典型序列选择器（高权重、中心节点、社区）
- [x] 模型基类设计（统一接口）
- [x] ARIMA模型实现
- [x] STGNN模型实现（GCN+GRU）
- [x] 评估指标系统
- [x] 实验脚本
- [x] 目录重命名（data/ → datasets/）
- [x] ARIMA vs STGNN对比实验框架（scripts/目录）
- [x] 实验结果分析工具包
- [x] 高权重边序列实验（50条边）

### 📋 待开发/优化
- [ ] STGNN训练循环优化（当前在base_model中）
- [ ] 更多基线模型（Prophet、N-BEATS、Informer）
- [ ] 超参数自动调优（Optuna集成）
- [ ] 模型集成/堆叠
- [ ] 可视化工具增强
- [ ] 结果持久化（MLflow/Wandb）
- [ ] 多步预测策略（递归 vs 直接 vs MIMO）
- [ ] 图结构学习（自适应邻接矩阵）

## 使用指南

### 环境设置

```bash
# 1. 克隆项目并进入目录
cd talent_flow_forcast

# 2. 创建Python 3.11虚拟环境
uv venv .venv-torch --python 3.11

# 3. 激活环境
.venv-torch\Scripts\activate  # Windows

# 4. 安装依赖
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install statsmodels networkx python-louvain
```

### 快速开始

#### 1. 运行框架测试
```bash
python src/experiments/test_framework.py
```

#### 2. 运行ARIMA实验
```bash
python src/experiments/run_arima.py
```

#### 3. 运行STGNN实验
```bash
python src/experiments/run_stgnn.py
```

#### 4. 运行对比实验（scripts目录）
```bash
# 运行完整实验流程
python scripts/run_experiments.py --phase all

# 或分阶段运行
python scripts/run_experiments.py --phase generate  # 生成时间序列
python scripts/run_experiments.py --phase run       # 运行模型实验
python scripts/run_experiments.py --phase analyze   # 分析结果
python scripts/run_experiments.py --phase visualize # 生成可视化

# 快速实验（使用预生成数据）
python scripts/run_quick_experiment.py
```

### 代码示例

#### 加载数据并选择边
```python
from src.data import FlowNetworkDataLoader, HighWeightSelector

loader = FlowNetworkDataLoader(
    data_dir="datasets/flow_networks",
    start_date="2017-01",
    end_date="2020-12"
)
networks = loader.load_networks()

selector = HighWeightSelector(top_k=50)
edges = selector.select(networks)
```

#### 使用ARIMA
```python
from src.models.statistical import ARIMAModel

model = ARIMAModel(input_len=6, output_len=1, order=(2, 1, 2))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 使用STGNN
```python
from src.models.deep_learning import STGNNModel

model = STGNNModel(
    input_len=12,
    output_len=3,
    num_nodes=50,
    adjacency_matrix=adj_matrix,
    hidden_dim=64,
    num_layers=2,
    device="auto"
)
model.fit(X_train, y_train, X_val, y_val, epochs=100)
predictions = model.predict(X_test)
```

## 关键设计决策

### 1. 数据与模型分离
- `src/data/`: 纯数据加载和处理代码，与模型无关
- `src/models/`: 模型实现，通过基类统一接口
- 支持多种预测任务（单变量、多变量、空间-时间）

### 2. 典型序列选择策略
- **高权重边**: 基于总流量，识别最重要的流动关系
- **中心节点边**: 基于度中心性，识别关键企业的流动
- **社区边**: 基于Louvain社区检测，识别内聚群体

### 3. 模型架构
- **统计模型**: 单序列独立建模，适合稀疏数据
- **STGNN**: 联合建模空间（图卷积）和时间（RNN/Attention）依赖

### 4. 目录命名
- `datasets/`: 存放数据文件（git忽略）
- `src/data/`: 数据模块源代码（git跟踪）

## 注意事项

1. **Python版本**: 使用Python 3.11，PyTorch暂不支持3.14
2. **GPU内存**: STGNN训练时如OOM，可减少num_nodes或hidden_dim
3. **数据路径**: 确保datasets/flow_networks/存在月度pkl文件
4. **CUDA**: Windows上可能需要安装Visual C++ Redistributable

## 参考资源

- BasicTS框架参考: `D:/experiments/tsf/basicts`
- PyTorch文档: https://pytorch.org/docs/
- Statsmodels ARIMA: https://www.statsmodels.org/stable/tsa.html

## 最后更新

2024-06-17: 完成STGNN实现，目录重命名为datasets/
