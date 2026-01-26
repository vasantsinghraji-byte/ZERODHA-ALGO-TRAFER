# -*- coding: utf-8 -*-
"""
ML Engine - Machine Learning Model Management
==============================================
Core ML engine for model training, prediction, and lifecycle management.

Classes:
    - BaseMLModel: Abstract base class for all ML models
    - RandomForestModel: Random Forest implementation
    - XGBoostModel: XGBoost implementation
    - LSTMModel: LSTM neural network implementation
    - MLEngine: Central engine for managing ML models
    - ModelMetrics: Performance metrics container
    - PredictionResult: Prediction output container
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    CUSTOM = "custom"


class TaskType(Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            **self.custom_metrics
        }


@dataclass
class PredictionResult:
    """Container for model predictions"""
    symbol: str
    timestamp: datetime
    prediction: Union[float, int, str]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    features_used: Optional[List[str]] = None
    model_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'features_used': self.features_used,
            'model_version': self.model_version,
            'metadata': self.metadata
        }


class BaseMLModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(
        self,
        name: str,
        model_type: ModelType,
        task_type: TaskType = TaskType.CLASSIFICATION,
        version: str = "1.0.0",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.model_type = model_type
        self.task_type = task_type
        self.version = version
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.metrics = ModelMetrics()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self._logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X)

        if self.task_type == TaskType.CLASSIFICATION and SKLEARN_AVAILABLE:
            self.metrics.accuracy = accuracy_score(y, predictions)
            self.metrics.precision = precision_score(y, predictions, average='weighted', zero_division=0)
            self.metrics.recall = recall_score(y, predictions, average='weighted', zero_division=0)
            self.metrics.f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        elif self.task_type == TaskType.REGRESSION:
            self.metrics.mse = float(np.mean((y - predictions) ** 2))
            self.metrics.mae = float(np.mean(np.abs(y - predictions)))
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.metrics.r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        self.updated_at = datetime.now()
        return self.metrics

    def save(self, path: str) -> None:
        """Save model to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'name': self.name,
                'version': self.version,
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'is_trained': self.is_trained
            }, f)
        self._logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.name = data.get('name', self.name)
            self.version = data.get('version', self.version)
            self.hyperparameters = data.get('hyperparameters', {})
            self.feature_names = data.get('feature_names', [])
            self.metrics = data.get('metrics', ModelMetrics())
            self.is_trained = data.get('is_trained', False)
        self._logger.info(f"Model loaded from {path}")


class RandomForestModel(BaseMLModel):
    """Random Forest model implementation"""

    def __init__(
        self,
        name: str = "random_forest",
        task_type: TaskType = TaskType.CLASSIFICATION,
        version: str = "1.0.0",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            model_type=ModelType.RANDOM_FOREST,
            task_type=task_type,
            version=version,
            hyperparameters=hyperparameters or {}
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        if not SKLEARN_AVAILABLE:
            self._logger.warning("scikit-learn not available, model will not function")
            return

        if task_type == TaskType.CLASSIFICATION:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                **self.hyperparameters
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                **self.hyperparameters
            )

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train the random forest model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestModel")

        self.model.fit(X, y)
        self.is_trained = True
        self.updated_at = datetime.now()
        self._logger.info(f"Model trained on {len(y)} samples")
        return self.evaluate(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self.task_type == TaskType.REGRESSION:
            return self.predict(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances"""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


class XGBoostModel(BaseMLModel):
    """XGBoost model implementation"""

    def __init__(
        self,
        name: str = "xgboost",
        task_type: TaskType = TaskType.CLASSIFICATION,
        version: str = "1.0.0",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            model_type=ModelType.XGBOOST,
            task_type=task_type,
            version=version,
            hyperparameters=hyperparameters or {}
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        if not XGBOOST_AVAILABLE:
            self._logger.warning("xgboost not available, model will not function")
            return

        objective = 'binary:logistic' if task_type == TaskType.CLASSIFICATION else 'reg:squarederror'
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            **self.hyperparameters
        ) if task_type == TaskType.CLASSIFICATION else xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            **self.hyperparameters
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train the XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostModel")

        self.model.fit(X, y)
        self.is_trained = True
        self.updated_at = datetime.now()
        self._logger.info(f"Model trained on {len(y)} samples")
        return self.evaluate(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if self.task_type == TaskType.REGRESSION:
            return self.predict(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances"""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


class LSTMModel(BaseMLModel):
    """LSTM neural network model (stub - requires TensorFlow/PyTorch)"""

    def __init__(
        self,
        name: str = "lstm",
        task_type: TaskType = TaskType.CLASSIFICATION,
        version: str = "1.0.0",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            model_type=ModelType.LSTM,
            task_type=task_type,
            version=version,
            hyperparameters=hyperparameters or {}
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._logger.warning("LSTM model is a stub - requires TensorFlow/PyTorch for full functionality")

    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train the LSTM model (stub)"""
        self._logger.warning("LSTM training is not implemented - using stub")
        self.is_trained = True
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (stub)"""
        self._logger.warning("LSTM prediction is not implemented - returning zeros")
        return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (stub)"""
        self._logger.warning("LSTM predict_proba is not implemented - returning uniform")
        return np.full((len(X), 2), 0.5)


class MLEngine:
    """
    Central ML Engine for model lifecycle management.

    Manages model training, prediction, versioning, and deployment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models: Dict[str, BaseMLModel] = {}
        self.active_model: Optional[str] = None
        self._logger = logging.getLogger(__name__)
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the ML engine"""
        self._logger.info("Initializing ML Engine")
        self._initialized = True
        return True

    def register_model(self, model: BaseMLModel) -> None:
        """Register a model with the engine"""
        self.models[model.name] = model
        self._logger.info(f"Registered model: {model.name} (v{model.version})")

    def get_model(self, name: str) -> Optional[BaseMLModel]:
        """Get a registered model by name"""
        return self.models.get(name)

    def set_active_model(self, name: str) -> bool:
        """Set the active model for predictions"""
        if name not in self.models:
            self._logger.error(f"Model {name} not found")
            return False
        self.active_model = name
        self._logger.info(f"Active model set to: {name}")
        return True

    def train_model(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Optional[ModelMetrics]:
        """Train a registered model"""
        model = self.get_model(name)
        if not model:
            self._logger.error(f"Model {name} not found")
            return None

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train
        model.train(X_train, y_train)

        # Evaluate on validation set
        metrics = model.evaluate(X_val, y_val)
        self._logger.info(f"Model {name} trained - Accuracy: {metrics.accuracy:.4f}")

        return metrics

    def predict(
        self,
        symbol: str,
        features: np.ndarray,
        model_name: Optional[str] = None
    ) -> Optional[PredictionResult]:
        """Make a prediction using the specified or active model"""
        name = model_name or self.active_model
        if not name:
            self._logger.error("No model specified and no active model set")
            return None

        model = self.get_model(name)
        if not model:
            self._logger.error(f"Model {name} not found")
            return None

        if not model.is_trained:
            self._logger.error(f"Model {name} is not trained")
            return None

        prediction = model.predict(features)

        # Get confidence
        confidence = 1.0
        probabilities = None
        if model.task_type == TaskType.CLASSIFICATION:
            proba = model.predict_proba(features)
            if len(proba.shape) > 1:
                confidence = float(np.max(proba[0]))
                probabilities = {str(i): float(p) for i, p in enumerate(proba[0])}

        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction=prediction[0] if len(prediction) == 1 else prediction.tolist(),
            confidence=confidence,
            probabilities=probabilities,
            features_used=model.feature_names,
            model_version=model.version
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [
            {
                'name': m.name,
                'type': m.model_type.value,
                'version': m.version,
                'is_trained': m.is_trained,
                'created_at': m.created_at.isoformat()
            }
            for m in self.models.values()
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'initialized': self._initialized,
            'total_models': len(self.models),
            'active_model': self.active_model,
            'trained_models': sum(1 for m in self.models.values() if m.is_trained),
            'sklearn_available': SKLEARN_AVAILABLE,
            'xgboost_available': XGBOOST_AVAILABLE
        }


# Global instance
_ml_engine: Optional[MLEngine] = None


def get_ml_engine() -> MLEngine:
    """Get or create the global ML engine instance"""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLEngine()
    return _ml_engine


def set_ml_engine(engine: MLEngine) -> None:
    """Set the global ML engine instance"""
    global _ml_engine
    _ml_engine = engine


__all__ = [
    'BaseMLModel',
    'RandomForestModel',
    'XGBoostModel',
    'LSTMModel',
    'MLEngine',
    'ModelMetrics',
    'PredictionResult',
    'ModelType',
    'TaskType',
    'get_ml_engine',
    'set_ml_engine',
    'SKLEARN_AVAILABLE',
    'XGBOOST_AVAILABLE'
]
