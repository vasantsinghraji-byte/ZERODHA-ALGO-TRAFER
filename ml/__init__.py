"""
Machine Learning Integration Package
Advanced ML capabilities for trading

Features:
- Predictive modeling (Random Forest, XGBoost, LSTM)
- Automated feature engineering (100+ features)
- Sentiment analysis (news, social media)
- Anomaly detection (statistical, multivariate, time series)
- Reinforcement learning (Q-Learning, DQN)
- Pattern classification (candlestick, chart patterns)
"""

from .ml_engine import (
    BaseMLModel,
    RandomForestModel,
    XGBoostModel,
    MLEngine,
    ModelMetrics,
    PredictionResult
)

# Use fixed LSTM implementation
try:
    from .lstm_fixed import LSTMModelFixed as LSTMModel
except ImportError:
    # Fallback to original if fixed version has issues
    from .ml_engine import LSTMModel

from .feature_engineer import (
    AutoFeatureEngineer,
    FeatureSet
)

from .sentiment_analyzer import (
    SentimentAnalyzer,
    TextPreprocessor,
    SentimentLexicon,
    SentimentScore,
    AggregatedSentiment
)

from .anomaly_detector import (
    AnomalyDetector,
    StatisticalAnomalyDetector,
    IsolationForestDetector,
    TimeSeriesAnomalyDetector,
    Anomaly,
    AnomalyType,
    AnomalySeverity,
    AnomalyDetectionResult
)

from .reinforcement_learning import (
    TradingEnvironment,
    QLearningAgent,
    DQNAgent,
    RLTrainer,
    Action,
    TradingState,
    Episode
)

from .pattern_classifier import (
    PatternClassifier,
    CandlestickPatternFeatures,
    ChartPatternFeatures,
    Pattern,
    PatternType,
    PatternSignal,
    PatternClassificationResult
)

__all__ = [
    # ML Engine
    'BaseMLModel',
    'RandomForestModel',
    'XGBoostModel',
    'LSTMModel',
    'MLEngine',
    'ModelMetrics',
    'PredictionResult',

    # Feature Engineering
    'AutoFeatureEngineer',
    'FeatureSet',

    # Sentiment Analysis
    'SentimentAnalyzer',
    'TextPreprocessor',
    'SentimentLexicon',
    'SentimentScore',
    'AggregatedSentiment',

    # Anomaly Detection
    'AnomalyDetector',
    'StatisticalAnomalyDetector',
    'IsolationForestDetector',
    'TimeSeriesAnomalyDetector',
    'Anomaly',
    'AnomalyType',
    'AnomalySeverity',
    'AnomalyDetectionResult',

    # Reinforcement Learning
    'TradingEnvironment',
    'QLearningAgent',
    'DQNAgent',
    'RLTrainer',
    'Action',
    'TradingState',
    'Episode',

    # Pattern Classification
    'PatternClassifier',
    'CandlestickPatternFeatures',
    'ChartPatternFeatures',
    'Pattern',
    'PatternType',
    'PatternSignal',
    'PatternClassificationResult'
]

__version__ = '1.0.0'
__author__ = 'Zerodha Algo Trader'
