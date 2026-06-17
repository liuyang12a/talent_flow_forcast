"""
Base model classes for time series forecasting.

This module defines abstract base classes that all forecasting models
must implement, ensuring consistent interfaces across different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
import pickle
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for all time series forecasting models.

    This class defines the common interface that all models must implement,
    including fit, predict, evaluate, save, and load methods.

    Attributes:
        input_len: Length of input sequence (history)
        output_len: Length of output sequence (forecast horizon)
        is_fitted: Whether the model has been fitted
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        name: str = "BaseModel",
        **kwargs
    ):
        """
        Initialize the model.

        Args:
            input_len: Length of input sequence (number of historical time steps)
            output_len: Length of output sequence (number of future time steps to predict)
            name: Model name for identification
            **kwargs: Additional model-specific parameters
        """
        self.input_len = input_len
        self.output_len = output_len
        self.name = name
        self.is_fitted = False
        self.config = kwargs

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseTimeSeriesModel":
        """
        Train the model on the given data.

        Args:
            X: Input features [n_samples, input_len, n_features] or [n_samples, input_len]
            y: Target values [n_samples, output_len] or [n_samples, output_len, n_features]
            X_val: Optional validation inputs for early stopping
            y_val: Optional validation targets for early stopping
            **kwargs: Additional training parameters

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for the given inputs.

        Args:
            X: Input features [n_samples, input_len, ...]
            **kwargs: Additional prediction parameters

        Returns:
            Predictions [n_samples, output_len, ...]
        """
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            X: Input features
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of metric names to values
        """
        from src.utils.metrics import calculate_metrics

        predictions = self.predict(X, **kwargs)
        return calculate_metrics(y, predictions, metrics)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseTimeSeriesModel":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'name': self.name,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'is_fitted': self.is_fitted,
            **self.config
        }


class BaseStatisticalModel(BaseTimeSeriesModel):
    """
    Base class for statistical forecasting models.

    Statistical models include ARIMA, Exponential Smoothing, etc.
    These models typically work on univariate or low-dimensional data
    and don't require gradient-based optimization.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        name: str = "StatisticalModel",
        **kwargs
    ):
        super().__init__(input_len, output_len, name, **kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseStatisticalModel":
        """
        Train the statistical model.

        For statistical models, X typically contains the historical values
        and y contains the future values to predict.
        """
        self._fit_impl(X, y, **kwargs)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit_impl(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Implementation-specific training logic."""
        pass


class BaseDeepLearningModel(BaseTimeSeriesModel, nn.Module):
    """
    Base class for deep learning forecasting models.

    Deep learning models use neural networks and typically require
    gradient-based optimization. They can handle high-dimensional
    spatial-temporal data.

    This class is designed to work with PyTorch models.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        name: str = "DeepLearningModel",
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize deep learning model.

        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence
            name: Model name
            device: Device to use ('cpu', 'cuda', or 'auto')
            **kwargs: Additional parameters
        """
        BaseTimeSeriesModel.__init__(self, input_len, output_len, name, **kwargs)
        nn.Module.__init__(self)

        # Set device
        if device == "auto":
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import torch
            self.device = torch.device(device)

        self.model = None  # PyTorch model, to be implemented by subclasses

    def _to_tensor(self, data: np.ndarray) -> "torch.Tensor":
        """Convert numpy array to PyTorch tensor on model device."""
        import torch
        return torch.from_numpy(data).float().to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        **kwargs
    ) -> "BaseDeepLearningModel":
        """
        Train the deep learning model.

        Args:
            X: Training inputs [n_samples, input_len, ...]
            y: Training targets [n_samples, output_len, ...]
            X_val: Validation inputs
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Epochs to wait before early stopping
            **kwargs: Additional training parameters
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Create data loaders
        train_dataset = TensorDataset(
            self._to_tensor(X),
            self._to_tensor(y)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                self._to_tensor(X_val),
                self._to_tensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None

        # Training loop
        self._train_loop(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            **kwargs
        )

        self.is_fitted = True
        return self

    @abstractmethod
    def _train_loop(
        self,
        train_loader: "DataLoader",
        val_loader: Optional["DataLoader"],
        epochs: int,
        learning_rate: float,
        early_stopping_patience: int,
        **kwargs
    ) -> None:
        """Implementation-specific training loop."""
        pass

    def predict(self, X: np.ndarray, batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            X: Input features
            batch_size: Batch size for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predictions as numpy array
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Use self.model if set, otherwise use self (for models that are their own nn.Module)
        model = self.model if self.model is not None else self
        model.eval()

        dataset = TensorDataset(self._to_tensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0]
                pred = model(batch_x)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights and configuration."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.get_config(),
            'model_class': self.__class__.__name__,
        }

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "BaseDeepLearningModel":
        """Load model from saved weights."""
        import torch

        checkpoint = torch.load(path, map_location='cpu')

        # Create new instance with saved config
        config = checkpoint['config']
        model = cls(**config, **kwargs)

        # Load weights
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.is_fitted = True

        logger.info(f"Model loaded from {path}")
        return model
