# Talent Flow - 人才流动网络分析工具

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

从招聘数据中提取企业员工流动网络，构建时间窗口化的流量网络分析。

## 功能特性

- 📊 **数据加载**: 高效读取 gzipped JSONL 格式的招聘数据
- 🔄 **流动网络**: 识别员工在不同公司间的流动路径
- 📈 **统计分析**: 生成流动网络的统计报告
- ⏱️ **时间窗口**: 支持按时间窗口切分和分析数据

## 快速开始

### 前置要求

- [uv](https://docs.astral.sh/uv/getting-started/installation/) - 快速安装: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 一键运行

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/talent-flow.git
cd talent-flow

# 2. 使用 uv 一键运行（自动创建虚拟环境并安装依赖）
uv run python preprocess.py

# 或在虚拟环境中运行
uv run --python 3.11 python statistic.py
```

### 常用命令

```bash
# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 运行主程序
uv run python preprocess.py

# 运行统计模块
uv run python statistic.py

# 格式化代码
uv run black *.py

# 代码检查
uv run ruff check *.py

# 运行测试
uv run pytest
```

## 项目结构

```
talent-flow/
├── data/               # 数据目录（大型数据文件不提交到Git）
├── cache/              # 缓存目录
├── data_loader.py      # 数据加载模块
├── flow_network.py     # 流动网络模块
├── preprocess.py       # 预处理主程序
├── statistic.py        # 统计分析模块
├── pyproject.toml      # 项目配置和依赖
└── README.md           # 项目说明
```

## 模块说明

### data_loader.py
数据加载模块，提供从 gzipped JSONL 文件高效流式读取招聘数据的功能。

### flow_network.py
流动网络核心模块，定义网络结构和节点/边的操作方法。

### preprocess.py
预处理主程序，从原始数据中提取员工流动网络。

### statistic.py
统计分析模块，生成流动网络的统计报告和可视化数据。

## 数据格式

输入数据应为 gzipped JSONL 格式，每行包含一个招聘记录的 JSON 对象：

```json
{
  "company": "公司名称",
  "employee_id": "员工ID",
  "start_date": "2020-01",
  "end_date": "2023-06",
  "position": "职位"
}
```

## 配置说明

项目使用 `pyproject.toml` 进行配置：

- **依赖管理**: 纯 Python 标准库，无第三方依赖
- **Python 版本**: >= 3.9
- **开发工具**: pytest, black, ruff

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目主页: https://github.com/yourusername/talent-flow
- 问题反馈: https://github.com/yourusername/talent-flow/issues
