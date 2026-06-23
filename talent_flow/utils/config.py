#!/usr/bin/env python3
"""Experiment configuration via dataclasses + YAML.

An :class:`ExperimentConfig` fully describes one pooling+forecasting experiment
and can be (de)serialized to YAML, enabling config-driven experiments without
code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class MethodConfig:
    """A named method + its parameters."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    flow_networks_dir: str = "datasets/flow_networks"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    company_attributes_path: Optional[str] = None


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    input_len: int = 12
    output_len: int = 1


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["mae", "rmse", "mape", "directional_accuracy"]
    )
    core_periphery_split: bool = False
    significance_test: Optional[str] = None  # "wilcoxon" | "paired_t"
    n_seeds: int = 1


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    pooling: MethodConfig = field(
        default_factory=lambda: MethodConfig("semantic_industry")
    )
    forecasting: MethodConfig = field(
        default_factory=lambda: MethodConfig("naive")
    )
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "ckpt"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML serialization")
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "default"),
            pooling=MethodConfig(**d["pooling"]),
            forecasting=MethodConfig(**d["forecasting"]),
            data=DataConfig(**d.get("data", {})),
            split=SplitConfig(**d.get("split", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
            output_dir=d.get("output_dir", "ckpt"),
            seed=d.get("seed", 42),
        )

    @classmethod
    def from_yaml(cls, text: str) -> "ExperimentConfig":
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))
