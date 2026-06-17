# 时间序列预测框架使用指南

## 项目概述

本项目扩展了原有的人才流动网络分析工具，新增了完整的时间序列预测框架，支持：
- 数据与模型分离的模块化设计
- 典型时间序列选择（高权重边、中心节点、社区结构）
- 统计模型（ARIMA）和深度学习模型（STGNN）
- 基于PyTorch的CUDA加速

## 环境配置

### 快速开始

项目已配置好Python 3.11虚拟环境，包含PyTorch CUDA支持：

```bash
# 激活深度学习环境（Windows）
.venv-torch\Scripts\activate

# 验证CUDA可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 安装依赖

```bash
# 基础依赖
uv pip install -e .

# 深度学习依赖（已安装）
uv pip install torch statsmodels networkx python-louvain
```

## 核心功能

### 1. 数据模块

#### 加载数据

```python
from demo.data import FlowNetworkDataLoader

loader = FlowNetworkDataLoader(
    data_dir="datasets/flow_networks",
    start_date="2017-01",
    end_date="2020-12"
)

networks = loader.load_networks()
print(f"Loaded {len(networks)} monthly networks")
```

#### 选择典型序列

```python
from demo.data import HighWeightSelector, HubNodeSelector, CommunitySelector

# 选择高权重边
selector = HighWeightSelector(top_k=50, min_months=6)
edges = selector.select(networks)

# 选择中心节点周围的边
hub_selector = HubNodeSelector(
    hub_threshold=10,
    max_edges_per_hub=20
)
hub_edges = hub_selector.select(networks)

# 基于社区检测选择
comm_selector = CommunitySelector(
    resolution=1.0,
    min_community_size=5,
    edge_selection="internal"
)
comm_edges = comm_selector.select(networks)
```

#### 创建数据集

```python
from demo.data.transforms import ZScoreScaler

scaler = ZScoreScaler()
train_ds, val_ds, test_ds = loader.create_datasets(
    edges=edges,
    input_len=6,      # 输入序列长度（历史月数）
    output_len=1,     # 输出序列长度（预测月数）
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    overlap=True,     # 是否使用重叠窗口
    scaler=scaler
)

# 访问数据
sample = train_ds[0]
print(sample['inputs'].shape)   # [6, num_edges]
print(sample['target'].shape)   # [1, num_edges]
```

### 2. 模型使用

#### ARIMA模型

```python
from demo.models.statistical import ARIMAModel

# 创建模型
model = ARIMAModel(
    input_len=6,
    output_len=1,
    order=(2, 1, 2),  # (p, d, q)参数
    name="ARIMA_Flow"
)

# 准备数据（滑动窗口）
import numpy as np
X_train, y_train = [], []
for i in range(len(train_ds)):
    sample = train_ds[i]
    X_train.append(sample['inputs'])
    y_train.append(sample['target'])
X_train = np.array(X_train)
y_train = np.array(y_train)

# 训练
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 评估
metrics = model.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")
```

#### STGNN模型

```python
from demo.models.deep_learning import STGNNModel
import numpy as np

# 构建邻接矩阵（基于边之间的连接关系）
def build_adjacency_matrix(edges):
    n = len(edges)
    adj = np.eye(n)
    for i, (src_i, tgt_i) in enumerate(edges):
        for j, (src_j, tgt_j) in enumerate(edges):
            if i != j and (src_i == src_j or tgt_i == tgt_j):
                adj[i, j] = 1.0
    return adj

adj_matrix = build_adjacency_matrix(edges)

# 创建模型
model = STGNNModel(
    input_len=12,
    output_len=3,
    num_nodes=len(edges),
    adjacency_matrix=adj_matrix,
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    spatial_type="gcn",      # 空间卷积类型: gcn, chebyshev
    temporal_type="gru",     # 时序模型类型: gru, lstm, tcn, attention
    dropout=0.1,
    device="auto"            # 自动选择GPU/CPU
)

# 准备数据 [batch, time, nodes, features]
def prepare_data(dataset):
    X_list, y_list = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        X = sample['inputs'][:, :, np.newaxis]   # [time, nodes, 1]
        y = sample['target'][:, :, np.newaxis]
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list)

X_train, y_train = prepare_data(train_ds)
X_val, y_val = prepare_data(val_ds)

# 训练
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=100,
    batch_size=8,
    learning_rate=0.001,
    early_stopping_patience=20
)

# 保存模型
model.save("demo/output/stgnn_model.pt")

# 加载模型
loaded_model = STGNNModel.load("demo/output/stgnn_model.pt")
```

### 3. 运行实验

#### ARIMA实验

```bash
python demo/experiments/run_arima.py
```

输出：
- `demo/output/arima_results.npz` - 预测结果和指标

#### STGNN实验

```bash
python demo/experiments/run_stgnn.py
```

输出：
- `demo/output/stgnn_results.npz` - 预测结果
- `demo/output/stgnn_model.pt` - 训练好的模型

## 架构说明

### 数据流

```
Flow Networks (.pkl files)
    ↓
FlowNetworkDataLoader
    ↓
Selector (HighWeight/HubNode/Community)
    ↓
FlowNetworkDataset
    ↓
Transforms (Scaler, SlidingWindow)
    ↓
Model (ARIMA/STGNN)
    ↓
Metrics & Visualization
```

### 扩展新模型

要添加新的预测模型，继承基类：

```python
from demo.models.base_model import BaseStatisticalModel

class MyModel(BaseStatisticalModel):
    def __init__(self, input_len, output_len, **kwargs):
        super().__init__(input_len, output_len, "MyModel", **kwargs)
        # 初始化模型

    def _fit_impl(self, X, y, **kwargs):
        # 实现训练逻辑
        pass

    def predict(self, X, **kwargs):
        # 实现预测逻辑
        return predictions
```

## 常见问题

### Q: PyTorch CUDA不可用？

确保使用正确的虚拟环境：
```bash
.venv-torch\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 如何处理内存不足？

1. 减少batch_size
2. 减少num_nodes（选择更少的边）
3. 使用更小的hidden_dim

### Q: 如何选择ARIMA的(p,d,q)参数？

```python
from demo.models.statistical import AutoARIMAModel

# 自动选择最优参数
model = AutoARIMAModel(input_len=6, output_len=1)
model.fit(X_train, y_train)
```

## 参考

- BasicTS框架参考：`D:/experiments/tsf/basicts`
- PyTorch文档：https://pytorch.org/docs/
- Statsmodels ARIMA：https://www.statsmodels.org/stable/tsa.html
