#!/usr/bin/env python3
"""
Talent Flow 一键运行脚本

用法:
    uv run python run.py [模块名]

示例:
    uv run python run.py preprocess    # 运行预处理模块
    uv run python run.py statistic     # 运行统计模块
    uv run python run.py all           # 运行完整流程
"""

import sys
import subprocess
from pathlib import Path


def run_module(module_name: str):
    """运行指定的 Python 模块"""
    module_map = {
        "preprocess": "preprocess.py",
        "statistic": "statistic.py",
        "data": "data_loader.py",
        "flow": "flow_network.py",
    }

    if module_name in module_map:
        script = module_map[module_name]
        print(f"🚀 正在运行 {script}...")
        subprocess.run([sys.executable, script], check=True)
    else:
        print(f"❌ 未知模块: {module_name}")
        print(f"可用模块: {', '.join(module_map.keys())}")
        sys.exit(1)


def run_all():
    """运行完整流程"""
    print("🎯 运行完整流程...")

    # 检查数据目录
    data_dir = Path("data")
    if not data_dir.exists():
        print("⚠️  创建数据目录...")
        data_dir.mkdir(exist_ok=True)

    # 运行预处理
    print("\n📊 步骤 1: 数据预处理")
    run_module("preprocess")

    # 运行统计
    print("\n📈 步骤 2: 统计分析")
    run_module("statistic")

    print("\n✅ 完整流程运行完成！")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n可用命令:")
        print("  preprocess  - 运行预处理模块")
        print("  statistic   - 运行统计模块")
        print("  all         - 运行完整流程")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "all":
        run_all()
    else:
        run_module(command)


if __name__ == "__main__":
    main()
