"""
Report generation module.

This module provides tools for generating comprehensive experiment reports
in various formats (JSON, Markdown, HTML).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .compare_models import ModelComparator
from .series_characteristics import SeriesAnalyzer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generator for experiment reports.

    Creates comprehensive reports summarizing experiment results,
    model comparisons, and analysis insights.
    """

    def __init__(self, results_path: Optional[Path] = None):
        """
        Initialize the report generator.

        Args:
            results_path: Optional path to results file
        """
        self.results = None
        self.comparator = None

        if results_path:
            self.load_results(results_path)

    def load_results(self, results_path: Path) -> "ReportGenerator":
        """
        Load experiment results.

        Args:
            results_path: Path to results file

        Returns:
            Self for method chaining
        """
        self.comparator = ModelComparator().load_results(results_path)
        self.results = self.comparator.results
        return self

    def generate_summary(self) -> Dict[str, any]:
        """
        Generate experiment summary.

        Returns:
            Summary dictionary
        """
        if self.results is None:
            raise ValueError("No results loaded.")

        summary = {
            "experiment_info": {
                "generated_at": datetime.now().isoformat(),
                "total_results": len(self.results),
                "unique_series": self.results['series_id'].nunique() if 'series_id' in self.results.columns else 0,
                "models_tested": self.results['model_type'].unique().tolist() if 'model_type' in self.results.columns else [],
            },
            "overall_comparison": {},
            "by_selector_type": {},
            "by_characteristic": {},
            "best_model_summary": {}
        }

        # Overall comparison
        for metric in ['mae', 'rmse', 'mape', 'r2']:
            if metric in self.results.columns:
                summary["overall_comparison"][metric] = self.comparator.compare_by_metric(metric)

        # By selector type
        if 'selector_type' in self.results.columns:
            summary["by_selector_type"] = self.comparator.compare_by_series_type()

        # By characteristic
        for char in ['volume', 'volatility', 'trend', 'seasonality']:
            col_name = f'char_{char}'
            if col_name in self.results.columns:
                summary["by_characteristic"][char] = self.comparator.compare_by_characteristic(char)

        # Best model summary
        try:
            best_models = self.comparator.get_best_model_per_series()
            if len(best_models) > 0:
                best_counts = best_models['best_model'].value_counts()
                summary["best_model_summary"] = {
                    "total_series": len(best_models),
                    "model_counts": best_counts.to_dict(),
                    "model_percentages": (best_counts / len(best_models) * 100).to_dict()
                }
        except Exception as e:
            logger.warning(f"Could not generate best model summary: {e}")

        return summary

    def generate_json_report(self, output_path: Path) -> None:
        """
        Generate JSON format report.

        Args:
            output_path: Path to save the report
        """
        summary = self.generate_summary()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"JSON report saved to {output_path}")

    def generate_markdown_report(self, output_path: Path) -> None:
        """
        Generate Markdown format report.

        Args:
            output_path: Path to save the report
        """
        summary = self.generate_summary()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append("# 时间序列预测对比实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Experiment Info
        lines.append("## 实验概况")
        lines.append("")
        info = summary["experiment_info"]
        lines.append(f"- **总结果数**: {info['total_results']}")
        lines.append(f"- **唯一序列数**: {info['unique_series']}")
        lines.append(f"- **测试模型**: {', '.join(info['models_tested'])}")
        lines.append("")

        # Overall Comparison
        lines.append("## 整体性能对比")
        lines.append("")

        if summary["overall_comparison"]:
            # Create comparison table
            lines.append("| 指标 | ARIMA 均值 | STGNN 均值 | 改进率 | 显著性 |")
            lines.append("|------|-----------|-----------|--------|--------|")

            for metric, comp in summary["overall_comparison"].items():
                if 'error' not in comp:
                    arima_mean = comp['arima_mean']
                    stgnn_mean = comp['stgnn_mean']
                    improvement = comp['improvement_pct']
                    significant = "是" if comp['statistical_test']['is_significant'] else "否"
                    lines.append(f"| {metric.upper()} | {arima_mean:.4f} | {stgnn_mean:.4f} | {improvement:+.1f}% | {significant} |")

            lines.append("")

            # Win counts
            lines.append("### 胜负统计")
            lines.append("")
            for metric, comp in summary["overall_comparison"].items():
                if 'win_counts' in comp:
                    wins = comp['win_counts']
                    lines.append(f"**{metric.upper()}**:")
                    lines.append(f"- ARIMA 胜出: {wins['arima']} 次")
                    lines.append(f"- STGNN 胜出: {wins['stgnn']} 次")
                    lines.append(f"- 平局: {wins['ties']} 次")
                    lines.append("")

        # By Selector Type
        if summary["by_selector_type"]:
            lines.append("## 按序列选择器分类对比")
            lines.append("")

            for metric, selectors in summary["by_selector_type"].items():
                lines.append(f"### {metric.upper()}")
                lines.append("")
                lines.append("| 选择器类型 | ARIMA 均值 | STGNN 均值 | 改进率 |")
                lines.append("|-----------|-----------|-----------|--------|")

                for selector, comp in selectors.items():
                    if 'error' not in comp:
                        arima_mean = comp['arima_mean']
                        stgnn_mean = comp['stgnn_mean']
                        improvement = comp['improvement_pct']
                        lines.append(f"| {selector} | {arima_mean:.4f} | {stgnn_mean:.4f} | {improvement:+.1f}% |")

                lines.append("")

        # Best Model Summary
        if summary["best_model_summary"]:
            lines.append("## 最优模型统计")
            lines.append("")
            best_summary = summary["best_model_summary"]
            lines.append(f"- **总序列数**: {best_summary['total_series']}")
            lines.append("")
            lines.append("| 模型 | 胜出次数 | 占比 |")
            lines.append("|------|---------|------|")

            for model, count in best_summary["model_counts"].items():
                pct = best_summary["model_percentages"][model]
                lines.append(f"| {model.upper()} | {count} | {pct:.1f}% |")

            lines.append("")

        # Conclusions
        lines.append("## 结论与洞察")
        lines.append("")

        # Auto-generate insights
        insights = self._generate_insights(summary)
        for insight in insights:
            lines.append(f"- {insight}")
        lines.append("")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Markdown report saved to {output_path}")

    def _generate_insights(self, summary: Dict) -> List[str]:
        """
        Generate automatic insights from summary.

        Args:
            summary: Summary dictionary

        Returns:
            List of insight strings
        """
        insights = []

        # Overall performance insight
        if "overall_comparison" in summary and "mae" in summary["overall_comparison"]:
            mae_comp = summary["overall_comparison"]["mae"]
            if 'improvement_pct' in mae_comp:
                improvement = mae_comp['improvement_pct']
                if improvement > 5:
                    insights.append(f"STGNN 在 MAE 指标上比 ARIMA 提升了 {improvement:.1f}%")
                elif improvement < -5:
                    insights.append(f"ARIMA 在 MAE 指标上比 STGNN 提升了 {abs(improvement):.1f}%")
                else:
                    insights.append("ARIMA 和 STGNN 在 MAE 指标上表现相近")

        # Best model insight
        if "best_model_summary" in summary and summary["best_model_summary"]:
            best_counts = summary["best_model_summary"].get("model_counts", {})
            if best_counts:
                best_model = max(best_counts, key=best_counts.get)
                best_count = best_counts[best_model]
                total = summary["best_model_summary"]["total_series"]
                insights.append(f"{best_model.upper()} 在 {best_count}/{total} 的序列上表现最优")

        # Selector type insights
        if "by_selector_type" in summary and "mae" in summary["by_selector_type"]:
            selector_comps = summary["by_selector_type"]["mae"]
            for selector, comp in selector_comps.items():
                if 'improvement_pct' in comp:
                    improvement = comp['improvement_pct']
                    if abs(improvement) > 10:
                        better = "STGNN" if improvement > 0 else "ARIMA"
                        insights.append(f"在 {selector} 类型序列上，{better} 表现显著更好")

        return insights if insights else ["需要更多数据来生成深入洞察"]

    def generate_csv_tables(self, output_dir: Path) -> List[Path]:
        """
        Generate CSV tables for detailed analysis.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        # Summary table
        try:
            summary_table = self.comparator.generate_summary_table()
            summary_path = output_dir / "model_comparison_summary.csv"
            summary_table.to_csv(summary_path, index=False)
            saved_paths.append(summary_path)
        except Exception as e:
            logger.warning(f"Could not generate summary table: {e}")

        # Best model per series
        try:
            best_models = self.comparator.get_best_model_per_series()
            best_path = output_dir / "best_model_per_series.csv"
            best_models.to_csv(best_path, index=False)
            saved_paths.append(best_path)
        except Exception as e:
            logger.warning(f"Could not generate best model table: {e}")

        # Raw results (sample)
        if self.results is not None:
            sample_path = output_dir / "results_sample.csv"
            self.results.head(1000).to_csv(sample_path, index=False)
            saved_paths.append(sample_path)

        logger.info(f"Generated {len(saved_paths)} CSV tables")
        return saved_paths

    def generate_all_reports(
        self,
        output_dir: Path,
        formats: List[str] = ['json', 'markdown', 'csv']
    ) -> Dict[str, Path]:
        """
        Generate all report formats.

        Args:
            output_dir: Directory to save reports
            formats: List of formats to generate

        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = {}

        if 'json' in formats:
            json_path = output_dir / "experiment_report.json"
            self.generate_json_report(json_path)
            generated['json'] = json_path

        if 'markdown' in formats:
            md_path = output_dir / "experiment_report.md"
            self.generate_markdown_report(md_path)
            generated['markdown'] = md_path

        if 'csv' in formats:
            csv_paths = self.generate_csv_tables(output_dir)
            if csv_paths:
                generated['csv'] = csv_paths

        return generated


def generate_experiment_report(
    results_path: Path,
    output_dir: Path,
    formats: List[str] = ['json', 'markdown', 'csv']
) -> Dict[str, Path]:
    """
    Convenience function to generate all experiment reports.

    Args:
        results_path: Path to results file
        output_dir: Directory to save reports
        formats: List of formats to generate

    Returns:
        Dictionary mapping format to file path
    """
    generator = ReportGenerator(results_path)
    return generator.generate_all_reports(output_dir, formats)
