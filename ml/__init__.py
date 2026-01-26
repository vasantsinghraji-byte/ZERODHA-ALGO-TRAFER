# -*- coding: utf-8 -*-
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

import logging

logger = logging.getLogger(__name__)

# Track available modules
_AVAILABLE_MODULES = []

# ============================================================================
# Core ML Engine (required)
# ============================================================================
try:
    from .ml_engine import (
        BaseMLModel,
        RandomForestModel,
        XGBoostModel,
        LSTMModel,
        MLEngine,
        ModelMetrics,
        PredictionResult,
        ModelType,
        TaskType,
        get_ml_engine,
        set_ml_engine,
    )
    _AVAILABLE_MODULES.append('ml_engine')
except ImportError as e:
    logger.warning(f"ML engine not available: {e}")
    # Provide stubs
    BaseMLModel = None
    RandomForestModel = None
    XGBoostModel = None
    LSTMModel = None
    MLEngine = None
    ModelMetrics = None
    PredictionResult = None
    ModelType = None
    TaskType = None
    get_ml_engine = None
    set_ml_engine = None

# ============================================================================
# Feature Store (Phase 11)
# ============================================================================
try:
    from .feature_store import (
        # Feature calculation
        FeatureCalculator,
        FeatureSet,
        FeatureDefinition,
        FeatureCategory,
        FeatureFrequency,
        FeatureValue,
        calculate_features,
        get_feature_definitions,
        get_feature_calculator,
        # Feature storage
        FeatureStore,
        FeatureStoreConfig,
        FeatureCache,
        StorageAdapter,
        MemoryStorageAdapter,
        RedisStorageAdapter,
        FileStorageAdapter,
        StorageBackend,
        get_feature_store,
        set_feature_store,
        put_features,
        get_features,
        get_historical_features,
        # Versioning
        FeatureVersionManager,
        FeatureVersionInfo,
        FeatureLineage,
        FeatureSchema,
        FeatureChange,
        TransformationStep,
        SemanticVersion,
        VersionStatus,
        ChangeType,
        get_version_manager,
        set_version_manager,
        register_feature,
        get_feature_version,
        validate_feature,
        # Pipeline
        FeaturePipeline,
        FeatureGenerator,
        TechnicalFeatureGenerator,
        VolatilityFeatureGenerator,
        MomentumFeatureGenerator,
        FeatureImportanceTracker,
        DriftDetector as FeatureDriftDetector,
        FeatureImportance,
        ImportanceReport,
        DriftResult,
        DriftReport,
        ImportanceMethod,
        DriftType,
        DriftMethod,
        get_feature_pipeline,
        set_feature_pipeline,
        generate_features,
        compute_importance,
        detect_drift,
    )
    _AVAILABLE_MODULES.append('feature_store')
except ImportError as e:
    logger.warning(f"Feature store not available: {e}")
    FeatureCalculator = None
    FeatureSet = None
    FeatureStore = None
    FeaturePipeline = None
    FeatureVersionManager = None

# ============================================================================
# Drift Detector (Phase 11)
# ============================================================================
try:
    from .drift_detector import (
        DriftDetector,
        DriftConfig,
        DriftAnalysis,
        DriftStatus,
        DriftSeverity,
        get_drift_detector,
        set_drift_detector,
    )
    _AVAILABLE_MODULES.append('drift_detector')
except ImportError as e:
    logger.warning(f"Drift detector not available: {e}")
    DriftDetector = None

# ============================================================================
# Model Registry (Phase 11)
# ============================================================================
try:
    from .model_registry import (
        ModelRegistry,
        ModelVersion,
        ModelInfo,
        ModelStage,
        get_model_registry,
        set_model_registry,
    )
    _AVAILABLE_MODULES.append('model_registry')
except ImportError as e:
    logger.warning(f"Model registry not available: {e}")
    ModelRegistry = None

# ============================================================================
# Optional: Feature Engineering (legacy)
# ============================================================================
try:
    from .feature_engineer import (
        AutoFeatureEngineer,
        FeatureSet as LegacyFeatureSet,
    )
    _AVAILABLE_MODULES.append('feature_engineer')
except ImportError:
    # Not critical - feature_store provides this functionality
    AutoFeatureEngineer = None
    LegacyFeatureSet = None

# ============================================================================
# Optional: Sentiment Analysis
# ============================================================================
try:
    from .sentiment_analyzer import (
        SentimentAnalyzer,
        TextPreprocessor,
        SentimentLexicon,
        SentimentScore,
        AggregatedSentiment
    )
    _AVAILABLE_MODULES.append('sentiment_analyzer')
except ImportError:
    SentimentAnalyzer = None
    TextPreprocessor = None
    SentimentLexicon = None
    SentimentScore = None
    AggregatedSentiment = None

# ============================================================================
# Optional: Anomaly Detection
# ============================================================================
try:
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
    _AVAILABLE_MODULES.append('anomaly_detector')
except ImportError:
    AnomalyDetector = None
    StatisticalAnomalyDetector = None
    IsolationForestDetector = None
    TimeSeriesAnomalyDetector = None
    Anomaly = None
    AnomalyType = None
    AnomalySeverity = None
    AnomalyDetectionResult = None

# ============================================================================
# Optional: Reinforcement Learning
# ============================================================================
try:
    from .reinforcement_learning import (
        TradingEnvironment,
        QLearningAgent,
        DQNAgent,
        RLTrainer,
        Action,
        TradingState,
        Episode
    )
    _AVAILABLE_MODULES.append('reinforcement_learning')
except ImportError:
    TradingEnvironment = None
    QLearningAgent = None
    DQNAgent = None
    RLTrainer = None
    Action = None
    TradingState = None
    Episode = None

# ============================================================================
# Optional: Pattern Classification
# ============================================================================
try:
    from .pattern_classifier import (
        PatternClassifier,
        CandlestickPatternFeatures,
        ChartPatternFeatures,
        Pattern,
        PatternType,
        PatternSignal,
        PatternClassificationResult
    )
    _AVAILABLE_MODULES.append('pattern_classifier')
except ImportError:
    PatternClassifier = None
    CandlestickPatternFeatures = None
    ChartPatternFeatures = None
    Pattern = None
    PatternType = None
    PatternSignal = None
    PatternClassificationResult = None


def get_available_modules():
    """Return list of available ML modules"""
    return _AVAILABLE_MODULES.copy()


__all__ = [
    # Module discovery
    'get_available_modules',

    # ML Engine
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

    # Feature Store
    'FeatureCalculator',
    'FeatureSet',
    'FeatureDefinition',
    'FeatureCategory',
    'FeatureFrequency',
    'FeatureValue',
    'calculate_features',
    'get_feature_definitions',
    'get_feature_calculator',
    'FeatureStore',
    'FeatureStoreConfig',
    'FeatureCache',
    'StorageAdapter',
    'MemoryStorageAdapter',
    'RedisStorageAdapter',
    'FileStorageAdapter',
    'StorageBackend',
    'get_feature_store',
    'set_feature_store',
    'put_features',
    'get_features',
    'get_historical_features',
    'FeatureVersionManager',
    'FeatureVersionInfo',
    'FeatureLineage',
    'FeatureSchema',
    'FeatureChange',
    'TransformationStep',
    'SemanticVersion',
    'VersionStatus',
    'ChangeType',
    'get_version_manager',
    'set_version_manager',
    'register_feature',
    'get_feature_version',
    'validate_feature',
    'FeaturePipeline',
    'FeatureGenerator',
    'TechnicalFeatureGenerator',
    'VolatilityFeatureGenerator',
    'MomentumFeatureGenerator',
    'FeatureImportanceTracker',
    'FeatureDriftDetector',
    'FeatureImportance',
    'ImportanceReport',
    'DriftResult',
    'DriftReport',
    'ImportanceMethod',
    'DriftType',
    'DriftMethod',
    'get_feature_pipeline',
    'set_feature_pipeline',
    'generate_features',
    'compute_importance',
    'detect_drift',

    # Drift Detector
    'DriftDetector',
    'DriftConfig',
    'DriftAnalysis',
    'DriftStatus',
    'DriftSeverity',
    'get_drift_detector',
    'set_drift_detector',

    # Model Registry
    'ModelRegistry',
    'ModelVersion',
    'ModelInfo',
    'ModelStage',
    'get_model_registry',
    'set_model_registry',

    # Legacy Feature Engineering
    'AutoFeatureEngineer',

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
    'PatternClassificationResult',
]

__version__ = '2.0.0'
__author__ = 'Zerodha Algo Trader'
