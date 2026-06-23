# 时空双稠密子图提取：从稀硫人才流动网络构建可预测张量

## 摘要

人才流动网络在空间和时间两个维度上均呈现显著稀硫性：节点度服从幂律分布，大量边仅活跃于少数月份。这种稀硫性使得时间序列预测模型（如 STGNN）难以有效学习。本文提出一种三阶段贪心算法，从原始时序图中提取时空双稠密子图，在最小化结构信息损失的前提下，构建适合时间序列建模的稠密张量。算法以边为中心（edge-centric）输出完整流向信息，通过统一接口支持向节点视图的柔性派生。

---

## 1. 问题定义

### 1.1 原始数据形式

设存在时间序列图 $\mathcal{G} = \{G_1, G_2, \ldots, G_T\}$，其中 $T = 132$ 为月度时间步。每个时刻的图 $G_t = (V_t, E_t, W_t)$ 是一个有向加权网络：

- $V_t$：该月发生人才流动的公司集合
- $E_t \subseteq V_t \times V_t$：有向边集合，边 $(u, v)$ 表示员工从公司 $u$ 跳槽至公司 $v$
- $W_t: E_t \to \mathbb{Z}^+$：边权重函数，$W_t(u,v)$ 为当月从 $u$ 流向 $v$ 的员工数

聚合网络定义为 $G_{\text{agg}} = \sum_{t=1}^{T} G_t$，即对所有时间步的权重求和。

### 1.2 稀硫性的形式化刻画

**空间稀硫性**。聚合网络 $G_{\text{agg}}$ 的节点度数服从幂律分布：

$$P(d) \propto d^{-\alpha}, \quad \alpha > 1$$

其中 $d(v) = d_{\text{in}}(v) + d_{\text{out}}(v)$ 为节点 $v$ 的加权总度数。这意味着少数公司（枢纽）参与了绝大部分人才流动，而大量节点仅有零星连接。

**时间稀硫性**。对于边 $e = (u,v)$，定义其活动指示函数：

$$\mathbb{1}_t(e) = \begin{cases} 1 & \text{if } W_t(u,v) > 0 \\ 0 & \text{otherwise} \end{cases}$$

边 $e$ 的活动比例为：

$$\rho(e) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}_t(e)$$

经验观察表明，多数边的 $\rho(e)$ 较低（$\rho < 0.3$），即边在大部分月份权重为零。

### 1.3 优化目标

给定原始时序网络 $\mathcal{G}$，寻找：

1. 核心公司节点子集 $V^* \subseteq \bigcup_{t} V_t$，满足：
   - **空间稠密**：$V^*$ 内部连接密度 $\delta_{\text{spatial}}(V^*)$ 高
   - **信息保留**：$V^*$ 的流量覆盖率 $\gamma(V^*) \geq \tau_{\text{cover}}$

2. 核心边子集 $E^* \subseteq \{(u,v) \mid u,v \in V^*\}$，满足：
   - **时间稠密**：$\forall e \in E^*,\; \rho(e) \geq \tau_{\rho}$
   - **时间连续**：$\forall e \in E^*,\; g_{\max}(e) \leq \tau_{\text{gap}}$

最终输出为一个稠密边-中心张量 $\mathbf{X} \in \mathbb{R}^{T \times |E^*| \times 1}$ 及其邻接矩阵 $\mathbf{A} \in \mathbb{R}^{|E^*| \times |E^*|}$。

---

## 2. 方法

### 2.1 整体架构

算法采用三阶段管道，与输出格式解耦：

```
阶段 A: 聚合统计 → 计算节点/边时空特征
阶段 B: 空间核心提取 → 贪心选择覆盖最大流量的节点子集
阶段 C: 时间稠密过滤 → 剔除活跃比例低或缺口过大的边
阶段 E: 张量构建 → BaseTensorBuilder 接口，可插拔实现
```

阶段 A–C 由 `DenseSubgraphExtractor` 统一执行，阶段 E 委托给可注入的 `BaseTensorBuilder`。

### 2.2 阶段 A：聚合统计量计算

#### 节点统计量

设 $V = \bigcup_{t=1}^{T} V_t$ 为全部出现过的公司集合。对每个节点 $v \in V$：

**加权总流量**（聚合网络中的入度+出度）：

$$\Phi(v) = \sum_{t=1}^{T} \left[ d_{\text{out}}^{(t)}(v) + d_{\text{in}}^{(t)}(v) \right]$$

其中 $d_{\text{out}}^{(t)}(v) = \sum_{u} W_t(v,u)$，$d_{\text{in}}^{(t)}(v) = \sum_{u} W_t(u,v)$。

**时间活跃度**：节点 $v$ 在多少个月份中有至少一条相连边（无论是入边还是出边）：

$$\alpha(v) = \frac{1}{T} \left| \left\{ t \mid \exists u: W_t(v,u) > 0 \text{ or } W_t(u,v) > 0 \right\} \right|$$

**综合节点得分**：

$$S_{\text{node}}(v) = \Phi(v) \times \alpha(v)$$

该得分同时奖励高流量和高时间活跃度的节点，抑制流量大但仅出现在少数月份的异常节点。

#### 边统计量

对每条出现过至少一次的边 $e = (u,v)$：

**聚合权重**：

$$\bar{W}(e) = \sum_{t=1}^{T} W_t(u,v)$$

**活动比例**：

$$\rho(e) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\left[ W_t(u,v) > 0 \right]$$

**最大连续零值间隔**：设时间序列 $\{W_t(e)\}_{t=1}^{T}$，其零值游程为 $\{r_1, r_2, \ldots\}$（以月数为单位）：

$$g_{\max}(e) = \max_k r_k$$

**综合时间得分**：

$$S_{\text{time}}(e) = \rho(e) \times \left(1 - \frac{g_{\max}(e)}{T}\right)$$

当 $\rho=1$ 且 $g_{\max}=0$ 时 $S_{\text{time}} = 1$（最佳）；当 $\rho \to 0$ 或 $g_{\max} \to T$ 时 $S_{\text{time}} \to 0$（最差）。

### 2.3 阶段 B：空间核心提取

提供三种可选策略，默认使用 Flow-Preserving Core。

#### 策略 B1：流保留核心（Flow-Preserving Core）— 默认

按 $S_{\text{node}}(v)$ 降序排列节点。贪心添加节点，累计流量覆盖率为：

$$\gamma(V_k) = \frac{\sum_{v \in V_k} \Phi(v)}{\sum_{v \in V} \Phi(v)}$$

其中 $V_k$ 为得分前 $k$ 名的节点集合。选择最小的 $k$ 使得：

$$\gamma(V_k) \geq \tau_{\text{cover}} \quad \text{或} \quad k = N_{\max}$$

参数 $\tau_{\text{cover}}$（如 0.80）为最低覆盖率阈值，$N_{\max}$（如 200）为最大节点数上限。同时要求 $k \geq N_{\min}$（如 20）。

**性质**：该策略保证输出节点集覆盖了总流量的至少 $\tau_{\text{cover}}$ 比例，是最直接的"最小化信息损失"方法。

#### 策略 B2：K-Core 分解

在聚合网络 $G_{\text{agg}}$ 的无向化版本上计算 k-core 分解。k-core 定义为反复删除度数小于 $k$ 的节点后剩余的子图：

$$\text{k-core}(G) = \{ v \in V \mid \deg_{\text{剩余}}(v) \geq k \}$$

从最大 $k$ 开始递减，直到 k-core 的大小 $|\text{k-core}| \in [N_{\min}, N_{\max}]$。

**性质**：保证每个节点至少有 $k$ 个邻居，空间密度有下界保障；但可能遗漏流量大而度数略低于 $k$ 的枢纽。

#### 策略 B3：贪心最大密度子图

维护候选边池（初始为 $S_{\text{time}}$ 最高的种子边），迭代选择邻接边加入子图。每次选择使子图平均密度增量最大的边：

$$\Delta \delta(e) = \delta(S \cup \{e\}) - \delta(S)$$

其中子图密度 $\delta(S) = |E_S| / (|V_S| \cdot (|V_S|-1))$。当 $\max \Delta \delta < \epsilon$ 时停止。

**性质**：局部密度最大化，适合需要极稠密子图的场景；计算复杂度较高。

### 2.4 阶段 C：时间稠密性过滤

对核心节点间存在的所有边进行筛选。边 $e$ 通过过滤当且仅当：

$$\rho(e) \geq \tau_{\rho} \quad \land \quad g_{\max}(e) \leq \tau_{\text{gap}}$$

或等价地使用综合得分：

$$S_{\text{time}}(e) \geq \tau_{\text{time}}$$

其中 $\tau_{\rho}$（默认 0.30）、$\tau_{\text{gap}}$（默认 12 个月）、$\tau_{\text{time}}$（默认 0.25）为可配置阈值。

### 2.5 阶段 E：张量构建与统一接口

#### 统一接口设计

```python
class BaseTensorBuilder(ABC):
    def build(networks, nodes, edges, timestamps) -> (tensor, adjacency, metadata)
```

`DenseSubgraphExtractor` 接收一个 `BaseTensorBuilder` 实例，`extract()` 的最后一步委托给 builder。切换输出格式仅需替换 builder，核心算法（阶段 A–C）完全不变。

#### 实现 E2：EdgeCentricTensorBuilder（默认）

**输出**：

$$\mathbf{X} \in \mathbb{R}^{T \times |E^*| \times 1}, \quad X_{t,i,0} = W_t(e_i)$$

即每时刻每条边的原始权重。

**邻接矩阵** $\mathbf{A} \in \mathbb{R}^{|E^*| \times |E^*|}$ 基于边的端点共享关系构建（线图变换）：

$$A_{i,j} = \begin{cases} 1 & \text{if } e_i \text{ 与 } e_j \text{ 共享至少一个端点} \\ 0 & \text{otherwise} \end{cases}$$

其中共享端点的判断条件为：$\text{src}(e_i) = \text{src}(e_j) \lor \text{src}(e_i) = \text{tgt}(e_j) \lor \text{tgt}(e_i) = \text{src}(e_j) \lor \text{tgt}(e_i) = \text{tgt}(e_j)$。

**信息保留度**：★★★★★ — 完整保留"哪个公司流向哪个公司"的配对信息。

**STGNN 适配**：`num_nodes = |E*|`，每个节点代表一条公司间流动边，节点信号为月度流量的时间序列。

#### 实现 E1：NodeCentricTensorBuilder（派生视图）

**输出**：$\mathbf{X} \in \mathbb{R}^{T \times |V^*| \times C}$，其中 $C$ 为特征通道数。

特征可为以下任意组合：

- 净流：$\phi_{\text{net}}(v,t) = d_{\text{out}}^{(t)}(v) - d_{\text{in}}^{(t)}(v)$
- 入流：$\phi_{\text{in}}(v,t) = d_{\text{in}}^{(t)}(v)$
- 出流：$\phi_{\text{out}}(v,t) = d_{\text{out}}^{(t)}(v)$
- 总流：$\phi_{\text{total}}(v,t) = d_{\text{out}}^{(t)}(v) + d_{\text{in}}^{(t)}(v)$

**邻接矩阵** $\mathbf{A} \in \mathbb{R}^{|V^*| \times |V^*|}$ 基于聚合网络中的实际流动：

$$A_{u,v} = \begin{cases} \bar{W}(u,v) & \text{if } \bar{W}(u,v) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**信息保留度**：★★★☆☆ — 丢失了边级配对信息，但可通过 `to_node_centric()` 从 E2 结果无损失派生。

#### 两种实现的关系

E1 可从 E2 的输出通过纯函数派生：

```
EdgeCentricBuilder  →  tensor_e [T, E, 1] + adj_e [E, E]
         │
         │  to_node_centric(tensor_e, edges, nodes):
         │    对每个 v∈V*, 每个 t:
         │      φ_*(v,t) = Σ{W_t(u,v) for u where (u,v)∈E*}
         │               + Σ{W_t(v,u) for u where (v,u)∈E*}
         │    以适当方向聚合
         │
         └──→  tensor_n [T, N, C] + adj_n [N, N]
```

E1 → E2 的逆方向不可行（信息已损失），因此默认使用 E2。

---

## 3. 质量评估指标

提取完成后，计算以下指标以验证子图质量。

### 3.1 空间指标

| 指标 | 公式 | 含义 |
|------|------|------|
| 节点数 | $N^* = |V^*|$ | 核心规模 |
| 边数 | $E^* = |E^*|$ | 有向边数 |
| 空间密度 | $\delta = \frac{E^*}{N^* (N^* - 1)}$ | 有向图密度 |
| 流量覆盖率 | $\gamma = \frac{\sum_{e \in E^*} \bar{W}(e)}{\sum_{e \in E_{\text{all}}} \bar{W}(e)}$ | 保留的流量比例 |
| 平均度数 | $\bar{d} = \frac{1}{N^*} \sum_{v \in V^*} d(v)$ | 每节点的平均连接数 |

期望：$N^*$ 适中（20–200），$\delta$ 较高（>0.01），$\gamma$ 较高（>0.70）。

### 3.2 时间指标

| 指标 | 公式 | 含义 |
|------|------|------|
| 平均活动率 | $\bar{\rho} = \frac{1}{E^*} \sum_{e \in E^*} \rho(e)$ | 边的平均非零月份比例 |
| 中位活动率 | $\rho_{50} = \text{median}\{\rho(e)\}$ | 鲁棒中心趋势 |
| 低活动边比例 | $f_{\text{low}} = \frac{|\{e: \rho(e) < 0.3\}|}{E^*}$ | 仍存在稀硫边的比例 |
| 平均最大缺口 | $\bar{g}_{\max} = \frac{1}{E^*} \sum g_{\max}(e)$ | 平均最长连续零值月数 |

期望：$\bar{\rho}$ 较高（>0.4），$f_{\text{low}}$ 较低（<0.2），$\bar{g}_{\max}$ 较小（<12）。

### 3.3 综合得分

$$\text{density\_score} = \delta \times \bar{\rho} \times \gamma$$

该得分平衡了空间密实度（$\delta$）、时间持续性（$\bar{\rho}$）和信息覆盖率（$\gamma$）。理论上界为 1（全连接、每月非零、完全覆盖），实际期望 >0.01。

---

## 4. 参数配置

### 4.1 参数总览

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| 最大节点数 | $N_{\max}$ | 200 | 核心公司数上限 |
| 最小节点数 | $N_{\min}$ | 20 | 核心公司数下限 |
| 目标覆盖率 | $\tau_{\text{cover}}$ | 0.80 | 流量覆盖率最低要求 |
| 最小活动率 | $\tau_{\rho}$ | 0.30 | 边非零月份比例下限 |
| 最大允许缺口 | $\tau_{\text{gap}}$ | 12 | 最长连续零值月数上限 |
| 空间策略 | — | `flow_core` | 阶段B的策略选择 |
| 张量类型 | — | `edge_centric` | 输出格式 |

### 4.2 推荐参数组合

**场景 1：高质量稠密核心（保守）**

$$N_{\max}=50,\; \tau_{\text{cover}}=0.60,\; \tau_{\rho}=0.50,\; \tau_{\text{gap}}=6$$

适用于 STGNN 初始实验，边活动率 >50%，最大缺口仅半年。

**场景 2：平衡覆盖与稠密（推荐）**

$$N_{\max}=200,\; \tau_{\text{cover}}=0.80,\; \tau_{\rho}=0.30,\; \tau_{\text{gap}}=12$$

适用于常规时间序列建模，覆盖80%流量，允许1年的偶发缺口。

**场景 3：高覆盖容忍稀硫（激进）**

$$N_{\max}=500,\; \tau_{\text{cover}}=0.90,\; \tau_{\rho}=0.10,\; \tau_{\text{gap}}=24$$

适用于 ARIMA 逐边建模（对稀硫容忍度较高）。

---

## 5. 算法复杂度

设 $T$ 为时间步数，$V$ 为总节点数，$E_{\text{total}}$ 为所有时间步的去重边总数。

| 阶段 | 时间复杂度 | 说明 |
|------|-----------|------|
| A：聚合统计 | $O(T \cdot E_{\max})$ | 遍历全部月度网络的所有边 |
| B1：流保留核心 | $O(V \log V)$ | 节点排序 |
| B2：K-Core | $O(V + E_{\text{agg}})$ | 标准 k-core 分解 |
| C：时间过滤 | $O(E_{\text{core}} \cdot T)$ | 对每条核心边扫描时间维 |
| E：张量构建 | $O(T \cdot E^*)$ | 填充张量矩阵 |

整体复杂度主要由阶段A主导，在 $T=132$、边数约万级的规模下可在一分钟内完成。

---

## 6. 实现文件对应

| 模块 | 职责 |
|------|------|
| `src/data/dense_subgraph.py` | `DenseSubgraphExtractor`、`BaseTensorBuilder`、两种 Builder 实现 |
| `scripts/config.py` | `dense_core` 选择器配置 |
| `scripts/run_experiments.py` | `phase1_generate_series()` 集成 |
| `scripts/analysis/network_diagnostics.py` | 原始网络稀硫性诊断，辅助参数调优 |

---

## 参考文献

1. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.* — Louvain 算法。
2. Seidman, S. B. (1983). Network structure and minimum degree. *Social Networks*, 5(3), 269–287. — K-core 分解。
3. Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks. *IJCAI 2018*. — STGNN 架框参考。

---

*最后更新：2026-06-18*
