"""
Experiment Configuration Module

This module centralizes all configuration settings for the ARIMA vs STGNN
comparison experiments.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "flow_networks"
SERIES_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "experiment_series"
CKPT_DIR = PROJECT_ROOT / "ckpt"

# Ensure directories exist
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SERIES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(CKPT_DIR / "models").mkdir(exist_ok=True)
(CKPT_DIR / "predictions").mkdir(exist_ok=True)
(CKPT_DIR / "metrics").mkdir(exist_ok=True)
(CKPT_DIR / "plots").mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "date_range": {"start": "2010-01", "end": "2020-12"},
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
}

# Sequence selection configuration
SELECTOR_CONFIG = {
    "high_weight": {
        "top_k": 300,
        "min_months": 6,
        "exclude_self_loops": True
    },
    "hub_nodes": {
        "hub_threshold": 10,
        "max_edges_per_hub": 50,
        "degree_type": "total",
        "selection_mode": "both",
        "max_total": 300  # Upper limit
    },
    "communities": {
        "resolution": 1.0,
        "min_community_size": 10,  # Increased to reduce number of communities
        "max_communities": 5,      # Reduced
        "edge_selection": "internal",
        "max_total": 300  # Upper limit
    }
}

# Model configurations
MODEL_CONFIG = {
    "arima": {
        "orders": [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)],
        "seasonal_orders": [None],  # Can add seasonal orders like (1, 1, 1, 12)
        "use_auto_arima": True,
        "auto_arima_config": {
            "seasonal": False,
            "stepwise": True,
            "suppress_warnings": True,
            "max_p": 5,
            "max_q": 5,
            "max_d": 2
        }
    },
    "stgnn": {
        "spatial_types": ["gcn", "chebyshev"],
        "temporal_types": ["gru", "lstm"],
        "hidden_dims": [32, 64],
        "num_layers": [2, 3],
        "dropout": 0.1,
        "output_type": "direct",
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 0.001,
            "early_stopping_patience": 20
        }
    }
}

# Prediction task settings
PREDICTION_SETTINGS = [
    {"input_len": 6, "output_len": 1, "name": "short_term"},
    {"input_len": 12, "output_len": 3, "name": "medium_term"},
    {"input_len": 12, "output_len": 6, "name": "long_term"}
]

# Experiment matrix
EXPERIMENT_MATRIX = {
    "experiment_1": {
        "name": "基础性能对比",
        "description": "ARIMA vs STGNN在所有序列上的基础性能对比",
        "series": "all",
        "models": ["arima", "stgnn"],
        "settings": [{"input_len": 12, "output_len": 1}],
        "metrics": ["mae", "rmse", "mape", "r2"]
    },
    "experiment_2": {
        "name": "序列类型对比",
        "description": "不同选择策略获取的序列的可预测性差异",
        "series_by_selector": ["high_weight", "hub_nodes", "communities"],
        "models": ["arima", "stgnn"],
        "settings": [{"input_len": 12, "output_len": 1}],
        "metrics": ["mae", "rmse", "mape", "r2"]
    },
    "experiment_3": {
        "name": "序列特征对比",
        "description": "不同特征序列的可预测性差异分析",
        "series_by_characteristic": {
            "volume": ["high", "medium", "low"],
            "volatility": ["high", "medium", "low"],
            "trend": ["strong", "weak"],
            "seasonality": ["strong", "weak"]
        },
        "models": ["arima", "stgnn"],
        "settings": [{"input_len": 12, "output_len": 1}],
        "metrics": ["mae", "rmse", "mape"]
    },
    "experiment_4": {
        "name": "预测长度对比",
        "description": "不同预测长度下模型性能变化",
        "series": "all",
        "models": ["arima", "stgnn"],
        "settings": [
            {"input_len": 6, "output_len": 1},
            {"input_len": 12, "output_len": 3},
            {"input_len": 12, "output_len": 6}
        ],
        "metrics": ["mae", "rmse", "mape"]
    }
}

# Series characteristic thresholds
CHARACTERISTIC_THRESHOLDS = {
    "volume": {
        "high_percentile": 80,
        "low_percentile": 20
    },
    "volatility": {
        "high_cv": 0.5,
        "low_cv": 0.2
    },
    "trend": {
        "adf_pvalue": 0.05
    },
    "seasonality": {
        "strong_threshold": 0.3
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": CKPT_DIR / "experiment.log"
}

# Random seed for reproducibility
RANDOM_SEED = 42


def get_config(config_type: str = "all") -> Dict[str, Any]:
    """
    Get configuration dictionary.

    Args:
        config_type: Type of config to return ("all", "data", "models", "experiments")

    Returns:
        Configuration dictionary
    """
    configs = {
        "all": {
            "data": DATA_CONFIG,
            "selectors": SELECTOR_CONFIG,
            "models": MODEL_CONFIG,
            "prediction": PREDICTION_SETTINGS,
            "experiments": EXPERIMENT_MATRIX,
            "thresholds": CHARACTERISTIC_THRESHOLDS,
            "paths": {
                "data_dir": str(DATA_DIR),
                "series_output_dir": str(SERIES_OUTPUT_DIR),
                "ckpt_dir": str(CKPT_DIR)
            }
        },
        "data": DATA_CONFIG,
        "models": MODEL_CONFIG,
        "experiments": EXPERIMENT_MATRIX
    }
    return configs.get(config_type, configs["all"])


def print_config():
    """Print current configuration settings."""
    import json

    config = get_config("all")
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(json.dumps(config, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    print_config()
