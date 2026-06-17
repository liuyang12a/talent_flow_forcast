#!/usr/bin/env python3
"""Generate analysis from experiment results."""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import CKPT_DIR, SERIES_OUTPUT_DIR

# Load results
df = pd.read_csv(CKPT_DIR / 'metrics' / 'experiment_results.csv')
print(f"Loaded {len(df)} results")
print(f"Models: {df['model_type'].unique()}")
print(f"Series: {df['series_id'].nunique()}")

# Summary statistics
print("\n=== Model Performance Summary ===")
summary = df.groupby('model_type')[['metric_mae', 'metric_rmse', 'metric_mape', 'metric_r2']].agg(['mean', 'std', 'median'])
print(summary)

# Compare ARIMA vs STGNN
print("\n=== ARIMA vs STGNN Comparison ===")
arima_data = df[df['model_type'] == 'arima']
stgnn_data = df[df['model_type'] == 'stgnn']

# Find common series
common_series = set(arima_data['series_id']) & set(stgnn_data['series_id'])
print(f"Common series: {len(common_series)}")

if len(common_series) > 0:
    comparison = []
    for series_id in common_series:
        arima_mae = arima_data[arima_data['series_id'] == series_id]['metric_mae'].values[0]
        stgnn_mae = stgnn_data[stgnn_data['series_id'] == series_id]['metric_mae'].values[0]
        comparison.append({
            'series_id': series_id,
            'arima_mae': arima_mae,
            'stgnn_mae': stgnn_mae,
            'diff': arima_mae - stgnn_mae,
            'improvement': (arima_mae - stgnn_mae) / arima_mae * 100 if arima_mae != 0 else 0,
            'winner': 'stgnn' if stgnn_mae < arima_mae else 'arima'
        })

    comp_df = pd.DataFrame(comparison)
    print(f"\nMAE Comparison:")
    print(f"  ARIMA mean: {comp_df['arima_mae'].mean():.4f}")
    print(f"  STGNN mean: {comp_df['stgnn_mae'].mean():.4f}")
    print(f"  Mean improvement: {comp_df['improvement'].mean():.1f}%")
    print(f"  STGNN wins: {(comp_df['winner'] == 'stgnn').sum()}")
    print(f"  ARIMA wins: {(comp_df['winner'] == 'arima').sum()}")

    # Visualizations
    plots_dir = CKPT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Box plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['metric_mae', 'metric_rmse', 'metric_mape']
    titles = ['MAE', 'RMSE', 'MAPE']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        data_to_plot = [df[df['model_type'] == 'arima'][metric].dropna(),
                       df[df['model_type'] == 'stgnn'][metric].dropna()]
        axes[idx].boxplot(data_to_plot)
        axes[idx].set_xticklabels(['ARIMA', 'STGNN'])
        axes[idx].set_title(f'{title} Comparison')
        axes[idx].set_ylabel(title)

    plt.tight_layout()
    plt.savefig(plots_dir / 'model_comparison_boxplot.png', dpi=150)
    print(f"\nSaved boxplot to {plots_dir / 'model_comparison_boxplot.png'}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(comp_df['arima_mae'], comp_df['stgnn_mae'], alpha=0.6)
    max_val = max(comp_df['arima_mae'].max(), comp_df['stgnn_mae'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal performance')
    ax.set_xlabel('ARIMA MAE')
    ax.set_ylabel('STGNN MAE')
    ax.set_title('ARIMA vs STGNN Performance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'model_comparison_scatter.png', dpi=150)
    print(f"Saved scatter plot to {plots_dir / 'model_comparison_scatter.png'}")

# Generate markdown report
report_lines = []
report_lines.append("# 时间序列预测对比实验报告\n")
report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

report_lines.append("## 实验概况\n")
report_lines.append(f"- **总结果数**: {len(df)}")
report_lines.append(f"- **唯一序列数**: {df['series_id'].nunique()}")
report_lines.append(f"- **测试模型**: {', '.join(df['model_type'].unique())}\n")

report_lines.append("## 整体性能对比\n")
report_lines.append("| 指标 | ARIMA 均值 | STGNN 均值 | 差异 |")
report_lines.append("|------|-----------|-----------|------|")

for metric, name in [('metric_mae', 'MAE'), ('metric_rmse', 'RMSE'), ('metric_mape', 'MAPE')]:
    arima_mean = df[df['model_type'] == 'arima'][metric].mean()
    stgnn_mean = df[df['model_type'] == 'stgnn'][metric].mean()
    diff = stgnn_mean - arima_mean
    report_lines.append(f"| {name} | {arima_mean:.4f} | {stgnn_mean:.4f} | {diff:+.4f} |")

if len(common_series) > 0:
    report_lines.append("\n## 模型对比详情\n")
    report_lines.append(f"- **共同评估序列数**: {len(common_series)}")
    report_lines.append(f"- **ARIMA 胜出**: {(comp_df['winner'] == 'arima').sum()}")
    report_lines.append(f"- **STGNN 胜出**: {(comp_df['winner'] == 'stgnn').sum()}")
    report_lines.append(f"- **平均改进率**: {comp_df['improvement'].mean():.1f}%\n")

report_lines.append("## 结论\n")
if len(common_series) > 0:
    if comp_df['improvement'].mean() > 0:
        report_lines.append(f"- STGNN 平均比 ARIMA 提升了 {comp_df['improvement'].mean():.1f}%")
    else:
        report_lines.append(f"- ARIMA 平均比 STGNN 提升了 {abs(comp_df['improvement'].mean()):.1f}%")

    if (comp_df['winner'] == 'stgnn').sum() > (comp_df['winner'] == 'arima').sum():
        report_lines.append(f"- STGNN 在 {(comp_df['winner'] == 'stgnn').sum()}/{len(common_series)} 的序列上表现更优")
    else:
        report_lines.append(f"- ARIMA 在 {(comp_df['winner'] == 'arima').sum()}/{len(common_series)} 的序列上表现更优")
else:
    report_lines.append("- 需要更多对比数据来生成结论")

# Save report
report_path = CKPT_DIR / 'experiment_report_final.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\nSaved report to {report_path}")
print("\n=== Report Preview ===")
print('\n'.join(report_lines[:30]))
