# -*- coding: utf-8 -*-
"""
Feature Store - Centralized ML Feature Repository
=================================================
Production-grade feature store for ML trading models.

Modules:
- features: Feature definitions and calculations
- store: Feature storage and retrieval
- versioning: Version control and lineage tracking

Example:
    >>> from ml.feature_store import (
    ...     FeatureStore, FeatureCalculator, FeatureVersionManager
    ... )
    >>>
    >>> # Calculate features
    >>> calc = FeatureCalculator()
    >>> df_with_features = calc.calculate_all(ohlcv_df)
    >>>
    >>> # Store features
    >>> store = FeatureStore()
    >>> store.put("RELIANCE", {"rsi_14": 65.5, "volatility": 0.023})
    >>>
    >>> # Version management
    >>> manager = FeatureVersionManager()
    >>> manager.register_version("rsi_14", "1.0.0", description="14-period RSI")
"""

from .features import (
    # Core classes
    FeatureCalculator,
    FeatureSet,
    FeatureValue,
    FeatureDefinition,
    # Enums
    FeatureCategory,
    FeatureFrequency,
    # Functions
    get_feature_calculator,
    calculate_features,
    get_feature_definitions,
)

from .store import (
    # Core classes
    FeatureStore,
    FeatureStoreConfig,
    FeatureCache,
    # Storage adapters
    StorageAdapter,
    MemoryStorageAdapter,
    RedisStorageAdapter,
    FileStorageAdapter,
    # Enums
    StorageBackend,
    # Functions
    get_feature_store,
    set_feature_store,
    put_features,
    get_features,
    get_historical_features,
)

from .versioning import (
    # Core classes
    FeatureVersionManager,
    FeatureVersionInfo,
    FeatureLineage,
    FeatureSchema,
    FeatureChange,
    TransformationStep,
    SemanticVersion,
    # Enums
    VersionStatus,
    ChangeType,
    # Functions
    get_version_manager,
    set_version_manager,
    register_feature,
    get_feature_version,
    validate_feature,
)

from .pipeline import (
    # Core classes
    FeaturePipeline,
    FeatureGenerator,
    TechnicalFeatureGenerator,
    VolatilityFeatureGenerator,
    MomentumFeatureGenerator,
    FeatureImportanceTracker,
    DriftDetector,
    # Data classes
    FeatureImportance,
    ImportanceReport,
    DriftResult,
    DriftReport,
    # Enums
    ImportanceMethod,
    DriftType,
    DriftMethod,
    # Functions
    get_feature_pipeline,
    set_feature_pipeline,
    generate_features,
    compute_importance,
    detect_drift,
)

__all__ = [
    # Features
    'FeatureCalculator',
    'FeatureSet',
    'FeatureValue',
    'FeatureDefinition',
    'FeatureCategory',
    'FeatureFrequency',
    'get_feature_calculator',
    'calculate_features',
    'get_feature_definitions',
    # Store
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
    # Versioning
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
    # Pipeline
    'FeaturePipeline',
    'FeatureGenerator',
    'TechnicalFeatureGenerator',
    'VolatilityFeatureGenerator',
    'MomentumFeatureGenerator',
    'FeatureImportanceTracker',
    'DriftDetector',
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
]
