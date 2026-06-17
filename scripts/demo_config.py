"""
Demo Configuration Module for Experiment Debugging

This module provides a separate configuration for demo/debugging runs,
ensuring that demo data and results are isolated from production experiments.

Usage:
    from scripts.demo_config import DEMO_CONFIG, get_demo_config

    # Demo run with isolated output
    config = get_demo_config()
    # Use config['paths']['ckpt_dir'] for output
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import shutil

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "flow_networks"

# Demo-specific output directories (isolated from production ckpt/)
DEMO_CKPT_DIR = PROJECT_ROOT / "demo_output"
DEMO_SERIES_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "demo_experiment_series"

# Ensure demo directories exist
def init_demo_directories():
    """Initialize demo output directories (clean slate)."""
    # Remove existing demo output for clean state
    if DEMO_CKPT_DIR.exists():
        shutil.rmtree(DEMO_CKPT_DIR)

    DEMO_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_SERIES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_CKPT_DIR / "models").mkdir(exist_ok=True)
    (DEMO_CKPT_DIR / "predictions").mkdir(exist_ok=True)
    (DEMO_CKPT_DIR / "metrics").mkdir(exist_ok=True)
    (DEMO_CKPT_DIR / "plots").mkdir(exist_ok=True)

    return DEMO_CKPT_DIR


# Demo data configuration (smaller scale for faster debugging)
DEMO_DATA_CONFIG = {
    "date_range": {"start": "2017-01", "end": "2020-12"},
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
}

# Demo sequence selection configuration (reduced for quick testing)
DEMO_SELECTOR_CONFIG = {
    "high_weight": {
        "top_k": 30,  # Reduced from 300 for faster runs
        "min_months": 6,
        "exclude_self_loops": True
    },
    "hub_nodes": {
        "hub_threshold": 10,
        "max_edges_per_hub": 20,  # Reduced from 50
        "degree_type": "total",
        "selection_mode": "both",
        "max_total": 50  # Reduced from 300
    },
    "communities": {
        "resolution": 1.0,
        "min_community_size": 10,
        "max_communities": 3,  # Reduced from 5
        "edge_selection": "internal",
        "max_total": 50  # Reduced from 300
    }
}

# Demo model configurations (lighter for faster execution)
DEMO_MODEL_CONFIG = {
    "arima": {
        "orders": [(1, 1, 1), (2, 1, 2)],  # Reduced set
        "seasonal_orders": [None],
        "use_auto_arima": False,  # Disabled for speed
        "auto_arima_config": {
            "seasonal": False,
            "stepwise": True,
            "suppress_warnings": True,
            "max_p": 3,  # Reduced from 5
            "max_q": 3,  # Reduced from 5
            "max_d": 2
        }
    },
    "stgnn": {
        "spatial_types": ["gcn"],  # Single option for demo
        "temporal_types": ["gru"],  # Single option for demo
        "hidden_dims": [32],  # Single smaller option
        "num_layers": [2],  # Single option
        "dropout": 0.1,
        "output_type": "direct",
        "training": {
            "epochs": 20,  # Reduced from 100
            "batch_size": 8,
            "learning_rate": 0.001,
            "early_stopping_patience": 5  # Reduced from 20
        }
    }
}

# Demo prediction task settings (simpler tasks)
DEMO_PREDICTION_SETTINGS = [
    {"input_len": 6, "output_len": 1, "name": "short_term"},
    # Removed medium_term and long_term for demo
]

# Demo experiment matrix (simplified)
DEMO_EXPERIMENT_MATRIX = {
    "demo_experiment": {
        "name": "Demo Performance Test",
        "description": "Quick ARIMA vs STGNN test on small dataset",
        "series": "high_weight",  # Only one selector
        "models": ["arima", "stgnn"],
        "settings": [{"input_len": 6, "output_len": 1}],
        "metrics": ["mae", "rmse", "mape"],
        "max_series": 10  # Limit number of series for demo
    }
}

# Demo logging configuration
DEMO_LOGGING_CONFIG = {
    "level": "DEBUG",  # More verbose for debugging
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": DEMO_CKPT_DIR / "demo_experiment.log"
}

# Demo random seed
DEMO_RANDOM_SEED = 42


def get_demo_config(config_type: str = "all") -> Dict[str, Any]:
    """
    Get demo configuration dictionary.

    Args:
        config_type: Type of config to return ("all", "data", "models", "experiments")

    Returns:
        Configuration dictionary for demo runs
    """
    # Ensure directories are initialized
    init_demo_directories()

    configs = {
        "all": {
            "data": DEMO_DATA_CONFIG,
            "selectors": DEMO_SELECTOR_CONFIG,
            "models": DEMO_MODEL_CONFIG,
            "prediction": DEMO_PREDICTION_SETTINGS,
            "experiments": DEMO_EXPERIMENT_MATRIX,
            "paths": {
                "data_dir": str(DATA_DIR),
                "series_output_dir": str(DEMO_SERIES_OUTPUT_DIR),
                "ckpt_dir": str(DEMO_CKPT_DIR)
            },
            "logging": DEMO_LOGGING_CONFIG,
            "random_seed": DEMO_RANDOM_SEED
        },
        "data": DEMO_DATA_CONFIG,
        "models": DEMO_MODEL_CONFIG,
        "experiments": DEMO_EXPERIMENT_MATRIX,
        "paths": {
            "data_dir": str(DATA_DIR),
            "series_output_dir": str(DEMO_SERIES_OUTPUT_DIR),
            "ckpt_dir": str(DEMO_CKPT_DIR)
        }
    }
    return configs.get(config_type, configs["all"])


def print_demo_config():
    """Print current demo configuration settings."""
    import json

    config = get_demo_config("all")
    print("=" * 60)
    print("Demo Experiment Configuration (Isolated from Production)")
    print("=" * 60)
    print(f"Checkpoint Dir: {config['paths']['ckpt_dir']}")
    print(f"Series Output Dir: {config['paths']['series_output_dir']}")
    print("-" * 60)
    print(json.dumps(config, indent=2, default=str))
    print("=" * 60)


class DemoExperimentPaths:
    """Helper class for managing demo experiment paths."""

    def __init__(self):
        self.ckpt_dir = DEMO_CKPT_DIR
        self.series_dir = DEMO_SERIES_OUTPUT_DIR

    def get_model_path(self, model_name: str) -> Path:
        """Get path for saving a model."""
        return self.ckpt_dir / "models" / f"{model_name}.pkl"

    def get_predictions_path(self, experiment_name: str) -> Path:
        """Get path for saving predictions."""
        return self.ckpt_dir / "predictions" / f"{experiment_name}.npz"

    def get_metrics_path(self, filename: str = "demo_results.json") -> Path:
        """Get path for saving metrics."""
        return self.ckpt_dir / "metrics" / filename

    def get_plot_path(self, plot_name: str) -> Path:
        """Get path for saving plots."""
        return self.ckpt_dir / "plots" / f"{plot_name}.png"

    def get_series_path(self, selector_name: str) -> Path:
        """Get path for saving/loading series data."""
        return self.series_dir / selector_name

    def clean(self):
        """Clean all demo output directories."""
        if self.ckpt_dir.exists():
            shutil.rmtree(self.ckpt_dir)
        init_demo_directories()
        print(f"Demo output cleaned: {self.ckpt_dir}")


# Convenience references
DEMO_CONFIG = get_demo_config("all")
DEMO_PATHS = DemoExperimentPaths()

if __name__ == "__main__":
    print_demo_config()
    print("\nDemo paths initialized:")
    print(f"  CKPT: {DEMO_CKPT_DIR}")
    print(f"  Series: {DEMO_SERIES_OUTPUT_DIR}")
    print("\nThese are isolated from production ckpt/ directory.")
