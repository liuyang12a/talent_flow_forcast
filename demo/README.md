# Time Series Forecasting Framework for Talent Flow Networks

This framework provides a complete solution for forecasting monthly employee flow between companies based on historical network data.

## Overview

The forecasting framework consists of four main components:

1. **FlowDataLoader**: Loads and preprocesses monthly flow network data
2. **FeatureEngineer**: Creates lag, rolling, and time-based features
3. **FlowForecaster**: Machine learning models for time series prediction
4. **ForecastingPipeline**: End-to-end pipeline for training and evaluation

## Quick Start

### 1. Prepare Data

Place your monthly flow network files in `data/flow_networks/` with naming convention:
```
flow_2023-01.json.gz
flow_2023-02.json.gz
flow_2023-03.json.gz
...
```

Each file should contain flow network data in JSON format:
```json
{
  "nodes": [...],
  "edges": [
    {
      "source": "Company A",
      "target": "Company B",
      "count": 5,
      "weight": 1.0
    },
    ...
  ]
}
```

### 2. Run Forecasting

```bash
# Basic usage
uv run python demo/ts_forecast_framework.py

# Or import and customize
uv run python -c "from demo.ts_forecast_framework import ForecastingPipeline; ..."
```

## Configuration

### Model Options

- `ridge`: Ridge Regression (default, fast and stable)
- `lasso`: Lasso Regression (feature selection)
- `rf`: Random Forest (handles non-linear patterns)
- `gb`: Gradient Boosting (best accuracy, slower)

### Hyperparameters

```python
pipeline = ForecastingPipeline(
    data_dir="data/flow_networks",
    lookback_window=6,      # Use 6 months of history
    forecast_horizon=1,     # Predict 1 month ahead
    model_type="ridge"      # Model selection
)
```

## Architecture

### Data Flow

```
Monthly Flow Data
       ↓
FlowDataLoader
       ↓
Time Series Construction
       ↓
FeatureEngineer
       ↓
[ Lag Features | Rolling Stats | Time Features ]
       ↓
FlowForecaster (ML Model)
       ↓
Predictions + Evaluation Metrics
```

### Feature Engineering

**Lag Features**: Historical flow counts (t-1, t-2, ..., t-n)

**Rolling Statistics**:
- Rolling mean (3-month, 6-month windows)
- Rolling standard deviation

**Time Features**:
- Year, Month, Quarter
- Season indicators (Spring, Summer, Autumn, Winter)

## Output

Results are saved to `demo/output/`:

- `metrics.json`: Evaluation metrics per company pair
- `predictions.json`: Actual vs predicted values

## Advanced Usage

### Custom Feature Engineering

```python
from demo.ts_forecast_framework import FeatureEngineer

fe = FeatureEngineer(lookback_window=12)
X, y, timestamps = fe.prepare_dataset(
    flow_metrics,
    use_time_features=True,
    use_rolling=True
)
```

### Training Custom Model

```python
from demo.ts_forecast_framework import FlowForecaster

forecaster = FlowForecaster(model_type="rf", n_estimators=200)
forecaster.fit(X_train, y_train)
metrics = forecaster.evaluate(X_test, y_test)
```

### Batch Evaluation

```python
results = pipeline.run(
    test_size=3,           # Last 3 months for testing
    min_observations=12,   # Require at least 12 months data
    top_k=50              # Only evaluate top 50 flows
)
```

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Extending the Framework

### Add Custom Model

```python
class MyCustomForecaster(FlowForecaster):
    def _init_model(self):
        from sklearn.svm import SVR
        self.model = SVR(kernel='rbf')
```

### Add New Features

Extend `FeatureEngineer.create_rolling_features()` to add:
- Exponential moving averages
- Trend indicators
- Fourier terms for seasonality

## Notes

- Framework uses only scikit-learn (no deep learning dependencies)
- All models are retrained per company pair (local models)
- Zero-padding is used for missing months
- Minimum observations threshold prevents overfitting on short series
