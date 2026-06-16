#!/usr/bin/env python3
"""
Quick test script to verify all modules are working correctly.
"""

import sys
from pathlib import Path
# Add project root to path (works from any directory)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from demo.data import (
            BaseDataset, DatasetConfig, TimeSeriesDataset, SpatialTemporalDataset,
            FlowNetworkDataset, FlowNetworkDataLoader,
            ZScoreScaler, MinMaxScaler, SlidingWindowTransform, TimeFeatureEncoder,
            HighWeightSelector, HubNodeSelector, CommunitySelector
        )
        print("  [OK] Data module imports successful")
    except Exception as e:
        print(f"  [FAIL] Data module import failed: {e}")
        return False

    try:
        from demo.models import (
            BaseTimeSeriesModel, BaseStatisticalModel, BaseDeepLearningModel
        )
        print("  [OK] Model base imports successful")
    except Exception as e:
        print(f"  [FAIL] Model base import failed: {e}")
        return False

    try:
        from demo.models.statistical import ARIMAModel
        print("  [OK] ARIMA model import successful")
    except Exception as e:
        print(f"  [FAIL] ARIMA model import failed: {e}")
        return False

    try:
        import torch
        from demo.models.deep_learning import STGNNModel
        print("  [OK] STGNN model import successful")
    except Exception as e:
        print(f"  [FAIL] STGNN model import failed: {e}")
        return False

    try:
        from demo.utils import calculate_metrics, mae, rmse, mape
        print("  [OK] Utils imports successful")
    except Exception as e:
        print(f"  [FAIL] Utils import failed: {e}")
        return False

    return True


def test_data_module():
    """Test data module functionality."""
    print("\nTesting data module...")

    import numpy as np
    from demo.data import DatasetConfig, TimeSeriesDataset, ZScoreScaler

    try:
        # Create dummy data
        data = np.random.randn(100, 5)

        # Test dataset
        config = DatasetConfig(input_len=6, output_len=1, overlap=True)
        dataset = TimeSeriesDataset(data, config, mode="train")

        assert len(dataset) > 0, "Dataset should have samples"

        sample = dataset[0]
        assert 'inputs' in sample, "Sample should have inputs"
        assert 'target' in sample, "Sample should have target"
        assert sample['inputs'].shape[0] == 6, "Input length should match config"

        print("  [OK] TimeSeriesDataset works correctly")
    except Exception as e:
        print(f"  [FAIL] TimeSeriesDataset test failed: {e}")
        return False

    try:
        # Test scaler
        scaler = ZScoreScaler()
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)

        assert np.allclose(data, recovered, atol=1e-6), "Scaler inverse should recover original"

        print("  [OK] ZScoreScaler works correctly")
    except Exception as e:
        print(f"  [FAIL] ZScoreScaler test failed: {e}")
        return False

    return True


def test_arima():
    """Test ARIMA model."""
    print("\nTesting ARIMA model...")

    import numpy as np
    from demo.models.statistical import ARIMAModel

    try:
        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 50
        X = np.random.randn(n_samples, 6)
        y = np.random.randn(n_samples, 1)

        model = ARIMAModel(input_len=6, output_len=1, order=(1, 0, 1))
        model.fit(X, y)

        predictions = model.predict(X[:5])
        assert predictions.shape == (5, 1), "Predictions shape should match"

        metrics = model.evaluate(X[:10], y[:10])
        assert 'rmse' in metrics, "Metrics should include RMSE"

        print("  [OK] ARIMA model works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] ARIMA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stgnn():
    """Test STGNN model."""
    print("\nTesting STGNN model...")

    import torch
    import numpy as np
    from demo.models.deep_learning import STGNNModel

    try:
        # Create small test case
        num_nodes = 5
        adj = np.eye(num_nodes) + np.random.rand(num_nodes, num_nodes) * 0.5
        adj = (adj + adj.T) > 0  # Make symmetric
        adj = adj.astype(np.float32)

        # Create model
        model = STGNNModel(
            input_len=6,
            output_len=3,
            num_nodes=num_nodes,
            adjacency_matrix=adj,
            input_dim=1,
            hidden_dim=16,
            num_layers=1,
            device="cpu"  # Use CPU for testing
        )

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 6, num_nodes, 1)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3, num_nodes, 1), f"Output shape mismatch: {output.shape}"

        print("  [OK] STGNN model works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] STGNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics module."""
    print("\nTesting metrics...")

    import numpy as np
    from demo.utils import calculate_metrics

    try:
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.1, 2.1, 2.9, 4.2])

        metrics = calculate_metrics(target, pred)

        assert 'mae' in metrics, "Should have MAE"
        assert 'rmse' in metrics, "Should have RMSE"
        assert metrics['mae'] > 0, "MAE should be positive"

        print("  [OK] Metrics calculation works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Metrics test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA available: {device_name}")
        else:
            print("  [WARN] CUDA not available (CPU only)")

        return True
    except Exception as e:
        print(f"  [FAIL] CUDA test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Framework Tests")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Data Module", test_data_module()))
    results.append(("ARIMA Model", test_arima()))
    results.append(("STGNN Model", test_stgnn()))
    results.append(("Metrics", test_metrics()))
    results.append(("CUDA", test_cuda()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)
    if all_passed:
        print("All tests passed! [OK]")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
