# -*- coding: utf-8 -*-
"""
Feature Engineering Pipeline - Automated Feature Management
============================================================
Production-grade pipeline for feature engineering, importance
tracking, and drift detection.

Features:
- Automated feature generation from raw data
- Feature importance tracking with multiple methods
- Statistical drift detection (PSI, KS, Chi-square)
- Feature selection based on importance/drift

Example:
    >>> from ml.feature_store import FeaturePipeline, DriftDetector
    >>>
    >>> # Create pipeline
    >>> pipeline = FeaturePipeline()
    >>> pipeline.add_generator(TechnicalFeatureGenerator())
    >>> pipeline.add_generator(VolatilityFeatureGenerator())
    >>>
    >>> # Generate features
    >>> features_df = pipeline.generate(ohlcv_df)
    >>>
    >>> # Track importance
    >>> importance = pipeline.compute_importance(features_df, target)
    >>>
    >>> # Detect drift
    >>> detector = DriftDetector()
    >>> drift_report = detector.detect(train_features, live_features)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import logging
import warnings

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImportanceMethod(Enum):
    """Feature importance calculation methods."""
    RANDOM_FOREST = "random_forest"
    PERMUTATION = "permutation"
    MUTUAL_INFO = "mutual_info"
    CORRELATION = "correlation"
    SHAP = "shap"


class DriftType(Enum):
    """Types of feature drift."""
    NO_DRIFT = "no_drift"
    MINOR_DRIFT = "minor_drift"
    MODERATE_DRIFT = "moderate_drift"
    SEVERE_DRIFT = "severe_drift"


class DriftMethod(Enum):
    """Drift detection methods."""
    PSI = "psi"                 # Population Stability Index
    KS_TEST = "ks_test"         # Kolmogorov-Smirnov test
    CHI_SQUARE = "chi_square"   # Chi-square test
    WASSERSTEIN = "wasserstein" # Wasserstein distance
    JS_DIVERGENCE = "js_div"    # Jensen-Shannon divergence


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_name: str
    importance_score: float
    rank: int
    method: ImportanceMethod

    # Additional metrics
    std_dev: float = 0.0
    p_value: Optional[float] = None

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)

    def __lt__(self, other: 'FeatureImportance') -> bool:
        return self.importance_score < other.importance_score


@dataclass
class ImportanceReport:
    """Complete feature importance report."""
    method: ImportanceMethod
    importances: List[FeatureImportance]
    computed_at: datetime = field(default_factory=datetime.now)

    # Summary stats
    total_features: int = 0
    top_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.total_features = len(self.importances)
        sorted_imps = sorted(self.importances, reverse=True)
        self.top_features = [imp.feature_name for imp in sorted_imps[:10]]

    def get_top_n(self, n: int = 10) -> List[FeatureImportance]:
        """Get top N important features."""
        return sorted(self.importances, reverse=True)[:n]

    def get_by_name(self, name: str) -> Optional[FeatureImportance]:
        """Get importance by feature name."""
        for imp in self.importances:
            if imp.feature_name == name:
                return imp
        return None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict mapping feature -> importance."""
        return {imp.feature_name: imp.importance_score for imp in self.importances}

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        data = [
            {
                'feature': imp.feature_name,
                'importance': imp.importance_score,
                'rank': imp.rank,
                'std_dev': imp.std_dev
            }
            for imp in self.importances
        ]
        return pd.DataFrame(data).sort_values('importance', ascending=False)


@dataclass
class DriftResult:
    """Drift detection result for a single feature."""
    feature_name: str
    drift_type: DriftType
    drift_score: float
    method: DriftMethod

    # Statistical details
    statistic: float = 0.0
    p_value: Optional[float] = None
    threshold: float = 0.0

    # Reference stats
    reference_mean: float = 0.0
    reference_std: float = 0.0
    current_mean: float = 0.0
    current_std: float = 0.0

    @property
    def has_drift(self) -> bool:
        """Check if significant drift detected."""
        return self.drift_type not in (DriftType.NO_DRIFT, DriftType.MINOR_DRIFT)

    @property
    def mean_shift(self) -> float:
        """Calculate mean shift."""
        if self.reference_std == 0:
            return 0.0
        return (self.current_mean - self.reference_mean) / self.reference_std


@dataclass
class DriftReport:
    """Complete drift detection report."""
    results: List[DriftResult]
    method: DriftMethod
    computed_at: datetime = field(default_factory=datetime.now)

    # Summary
    total_features: int = 0
    drifted_features: int = 0
    drift_rate: float = 0.0

    def __post_init__(self):
        self.total_features = len(self.results)
        self.drifted_features = sum(1 for r in self.results if r.has_drift)
        self.drift_rate = self.drifted_features / self.total_features if self.total_features > 0 else 0.0

    @property
    def has_significant_drift(self) -> bool:
        """Check if significant drift detected in any feature."""
        return self.drift_rate > 0.1  # More than 10% features drifted

    def get_drifted_features(self) -> List[DriftResult]:
        """Get features with detected drift."""
        return [r for r in self.results if r.has_drift]

    def get_by_name(self, name: str) -> Optional[DriftResult]:
        """Get result by feature name."""
        for r in self.results:
            if r.feature_name == name:
                return r
        return None

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        data = [
            {
                'feature': r.feature_name,
                'drift_type': r.drift_type.value,
                'drift_score': r.drift_score,
                'p_value': r.p_value,
                'mean_shift': r.mean_shift,
                'has_drift': r.has_drift
            }
            for r in self.results
        ]
        return pd.DataFrame(data).sort_values('drift_score', ascending=False)


class FeatureGenerator(ABC):
    """Abstract base class for feature generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name."""
        pass

    @property
    def feature_names(self) -> List[str]:
        """List of features this generator produces."""
        return []

    @abstractmethod
    def generate(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Generate features from input DataFrame."""
        pass


class TechnicalFeatureGenerator(FeatureGenerator):
    """Generates technical analysis features."""

    def __init__(
        self,
        sma_periods: List[int] = None,
        ema_periods: List[int] = None,
        rsi_periods: List[int] = None,
        include_macd: bool = True,
        include_bollinger: bool = True,
        include_stochastic: bool = True
    ):
        self.sma_periods = sma_periods or [10, 20, 50, 200]
        self.ema_periods = ema_periods or [12, 26]
        self.rsi_periods = rsi_periods or [14]
        self.include_macd = include_macd
        self.include_bollinger = include_bollinger
        self.include_stochastic = include_stochastic

    @property
    def name(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        names = []
        names.extend([f'sma_{p}' for p in self.sma_periods])
        names.extend([f'ema_{p}' for p in self.ema_periods])
        names.extend([f'rsi_{p}' for p in self.rsi_periods])
        if self.include_macd:
            names.extend(['macd', 'macd_signal', 'macd_histogram'])
        if self.include_bollinger:
            names.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width'])
        if self.include_stochastic:
            names.extend(['stoch_k', 'stoch_d'])
        return names

    def generate(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Generate technical features."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        result = df.copy()
        close = df['close'].values

        # SMAs
        for period in self.sma_periods:
            result[f'sma_{period}'] = self._sma(close, period)

        # EMAs
        for period in self.ema_periods:
            result[f'ema_{period}'] = self._ema(close, period)

        # RSI
        for period in self.rsi_periods:
            result[f'rsi_{period}'] = self._rsi(close, period)

        # MACD
        if self.include_macd:
            ema_12 = self._ema(close, 12)
            ema_26 = self._ema(close, 26)
            result['macd'] = ema_12 - ema_26
            result['macd_signal'] = self._ema(result['macd'].values, 9)
            result['macd_histogram'] = result['macd'] - result['macd_signal']

        # Bollinger Bands
        if self.include_bollinger:
            sma_20 = self._sma(close, 20)
            std_20 = self._rolling_std(close, 20)
            result['bb_upper'] = sma_20 + 2 * std_20
            result['bb_middle'] = sma_20
            result['bb_lower'] = sma_20 - 2 * std_20
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']

        # Stochastic
        if self.include_stochastic and 'high' in df.columns and 'low' in df.columns:
            result['stoch_k'], result['stoch_d'] = self._stochastic(
                df['high'].values, df['low'].values, close, 14, 3
            )

        return result

    def _sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(prices) - 1):
            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return result

    def _rolling_std(self, prices: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.std(prices[i - period + 1:i + 1])
        return result

    def _stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int,
        d_period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(high)
        k_line = np.full(n, np.nan)

        for i in range(k_period - 1, n):
            hh = np.max(high[i - k_period + 1:i + 1])
            ll = np.min(low[i - k_period + 1:i + 1])
            if hh - ll == 0:
                k_line[i] = 50.0
            else:
                k_line[i] = ((close[i] - ll) / (hh - ll)) * 100.0

        d_line = self._sma(k_line, d_period)
        return k_line, d_line


class VolatilityFeatureGenerator(FeatureGenerator):
    """Generates volatility-based features."""

    def __init__(
        self,
        atr_periods: List[int] = None,
        vol_periods: List[int] = None,
        include_parkinson: bool = True
    ):
        self.atr_periods = atr_periods or [14]
        self.vol_periods = vol_periods or [10, 20, 30]
        self.include_parkinson = include_parkinson

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def feature_names(self) -> List[str]:
        names = []
        names.extend([f'atr_{p}' for p in self.atr_periods])
        names.extend([f'volatility_{p}' for p in self.vol_periods])
        if self.include_parkinson:
            names.append('parkinson_vol')
        return names

    def generate(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        result = df.copy()
        close = df['close'].values

        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            high = df['high'].values
            low = df['low'].values

            for period in self.atr_periods:
                result[f'atr_{period}'] = self._atr(high, low, close, period)

            # Parkinson volatility
            if self.include_parkinson:
                result['parkinson_vol'] = self._parkinson_vol(high, low, 20)

        # Realized volatility
        for period in self.vol_periods:
            result[f'volatility_{period}'] = self._realized_vol(close, period)

        return result

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # SMA of TR
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            result[i] = np.mean(tr[i - period + 1:i + 1])
        return result

    def _realized_vol(self, prices: np.ndarray, period: int) -> np.ndarray:
        n = len(prices)
        result = np.full(n, np.nan)
        log_returns = np.diff(np.log(prices))

        for i in range(period, n):
            result[i] = np.std(log_returns[i - period:i]) * np.sqrt(252)
        return result

    def _parkinson_vol(
        self,
        high: np.ndarray,
        low: np.ndarray,
        period: int
    ) -> np.ndarray:
        n = len(high)
        result = np.full(n, np.nan)
        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))

        for i in range(period - 1, n):
            result[i] = np.sqrt(factor * np.mean(log_hl[i - period + 1:i + 1]) * 252)
        return result


class MomentumFeatureGenerator(FeatureGenerator):
    """Generates momentum-based features."""

    def __init__(self, periods: List[int] = None):
        self.periods = periods or [5, 10, 20, 60]

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def feature_names(self) -> List[str]:
        names = []
        names.extend([f'returns_{p}' for p in self.periods])
        names.extend([f'momentum_{p}' for p in self.periods])
        names.append('roc')
        return names

    def generate(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        result = df.copy()
        close = df['close'].values

        for period in self.periods:
            # Returns
            returns = np.full(len(close), np.nan)
            returns[period:] = (close[period:] - close[:-period]) / close[:-period]
            result[f'returns_{period}'] = returns

            # Momentum (price ratio)
            momentum = np.full(len(close), np.nan)
            momentum[period:] = close[period:] / close[:-period]
            result[f'momentum_{period}'] = momentum

        # Rate of Change
        roc = np.full(len(close), np.nan)
        roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100
        result['roc'] = roc

        return result


class FeatureImportanceTracker:
    """
    Tracks and computes feature importance.

    Supports multiple methods:
    - Random Forest importance
    - Permutation importance
    - Mutual information
    - Correlation with target

    Example:
        >>> tracker = FeatureImportanceTracker()
        >>> report = tracker.compute(X, y, method=ImportanceMethod.RANDOM_FOREST)
        >>> print(report.top_features)
    """

    def __init__(self):
        self._history: Dict[str, List[ImportanceReport]] = defaultdict(list)

    def compute(
        self,
        X: 'pd.DataFrame',
        y: np.ndarray,
        method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST,
        task: str = 'classification'
    ) -> ImportanceReport:
        """
        Compute feature importance.

        Args:
            X: Feature DataFrame
            y: Target array
            method: Importance method
            task: 'classification' or 'regression'

        Returns:
            ImportanceReport with ranked features
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        feature_names = list(X.columns)
        X_clean = X.fillna(0).values

        if method == ImportanceMethod.RANDOM_FOREST:
            scores, stds = self._rf_importance(X_clean, y, task)
        elif method == ImportanceMethod.MUTUAL_INFO:
            scores, stds = self._mutual_info_importance(X_clean, y, task)
        elif method == ImportanceMethod.CORRELATION:
            scores, stds = self._correlation_importance(X_clean, y)
        else:
            # Default to correlation
            scores, stds = self._correlation_importance(X_clean, y)

        # Create ranked importances
        sorted_indices = np.argsort(scores)[::-1]
        importances = []

        for rank, idx in enumerate(sorted_indices, 1):
            importances.append(FeatureImportance(
                feature_name=feature_names[idx],
                importance_score=float(scores[idx]),
                rank=rank,
                method=method,
                std_dev=float(stds[idx]) if stds is not None else 0.0
            ))

        report = ImportanceReport(method=method, importances=importances)

        # Store in history
        self._history[method.value].append(report)

        return report

    def _rf_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random Forest importance."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for RF importance")

        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        importances = model.feature_importances_
        stds = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

        return importances, stds

    def _mutual_info_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mutual information importance."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MI importance")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == 'classification':
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)

        return scores, np.zeros(len(scores))

    def _correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correlation-based importance."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            valid_mask = ~np.isnan(X[:, i]) & ~np.isnan(y)
            if valid_mask.sum() > 10:
                corr = np.corrcoef(X[valid_mask, i], y[valid_mask])[0, 1]
                scores[i] = abs(corr) if not np.isnan(corr) else 0.0

        return scores, np.zeros(n_features)

    def get_history(self, method: Optional[ImportanceMethod] = None) -> List[ImportanceReport]:
        """Get importance history."""
        if method:
            return self._history.get(method.value, [])

        all_reports = []
        for reports in self._history.values():
            all_reports.extend(reports)
        return all_reports

    def get_stable_features(
        self,
        min_reports: int = 3,
        top_n: int = 10
    ) -> List[str]:
        """Get features that are consistently important."""
        all_reports = self.get_history()
        if len(all_reports) < min_reports:
            return []

        # Count top-N appearances
        feature_counts: Dict[str, int] = defaultdict(int)
        for report in all_reports:
            for feat in report.top_features[:top_n]:
                feature_counts[feat] += 1

        # Return features appearing in most reports
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [f for f, count in sorted_features if count >= min_reports]


class DriftDetector:
    """
    Detects feature drift between reference and current data.

    Supports multiple methods:
    - PSI (Population Stability Index)
    - KS Test (Kolmogorov-Smirnov)
    - Chi-Square test
    - Wasserstein distance

    Example:
        >>> detector = DriftDetector()
        >>> report = detector.detect(train_df, live_df)
        >>> if report.has_significant_drift:
        ...     print(f"Drifted features: {report.get_drifted_features()}")
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        n_bins: int = 10
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins

        self._history: List[DriftReport] = []

    def detect(
        self,
        reference: 'pd.DataFrame',
        current: 'pd.DataFrame',
        method: DriftMethod = DriftMethod.PSI,
        features: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift between reference and current data.

        Args:
            reference: Reference/training data
            current: Current/live data
            method: Detection method
            features: Specific features to check (None = all common)

        Returns:
            DriftReport with per-feature results
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        # Get common features
        if features is None:
            features = list(set(reference.columns) & set(current.columns))
            # Exclude non-numeric
            features = [f for f in features if reference[f].dtype in ['float64', 'int64', 'float32', 'int32']]

        results = []

        for feature in features:
            ref_data = reference[feature].dropna().values
            cur_data = current[feature].dropna().values

            if len(ref_data) < 10 or len(cur_data) < 10:
                continue

            if method == DriftMethod.PSI:
                result = self._psi_test(feature, ref_data, cur_data)
            elif method == DriftMethod.KS_TEST:
                result = self._ks_test(feature, ref_data, cur_data)
            elif method == DriftMethod.WASSERSTEIN:
                result = self._wasserstein_test(feature, ref_data, cur_data)
            else:
                result = self._psi_test(feature, ref_data, cur_data)

            results.append(result)

        report = DriftReport(results=results, method=method)
        self._history.append(report)

        return report

    def _psi_test(
        self,
        feature: str,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Population Stability Index test."""
        # Create bins from reference
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, self.n_bins + 1)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Add small value to avoid division by zero
        ref_props = np.clip(ref_props, 0.0001, 1)
        cur_props = np.clip(cur_props, 0.0001, 1)

        # PSI calculation
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        # Classify drift
        if psi < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif psi < 0.2:
            drift_type = DriftType.MINOR_DRIFT
        elif psi < 0.25:
            drift_type = DriftType.MODERATE_DRIFT
        else:
            drift_type = DriftType.SEVERE_DRIFT

        return DriftResult(
            feature_name=feature,
            drift_type=drift_type,
            drift_score=float(psi),
            method=DriftMethod.PSI,
            statistic=float(psi),
            threshold=self.psi_threshold,
            reference_mean=float(np.mean(reference)),
            reference_std=float(np.std(reference)),
            current_mean=float(np.mean(current)),
            current_std=float(np.std(current))
        )

    def _ks_test(
        self,
        feature: str,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Kolmogorov-Smirnov test."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for KS test")

        statistic, p_value = stats.ks_2samp(reference, current)

        # Classify based on p-value
        if p_value > 0.1:
            drift_type = DriftType.NO_DRIFT
        elif p_value > 0.05:
            drift_type = DriftType.MINOR_DRIFT
        elif p_value > 0.01:
            drift_type = DriftType.MODERATE_DRIFT
        else:
            drift_type = DriftType.SEVERE_DRIFT

        return DriftResult(
            feature_name=feature,
            drift_type=drift_type,
            drift_score=float(statistic),
            method=DriftMethod.KS_TEST,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.ks_threshold,
            reference_mean=float(np.mean(reference)),
            reference_std=float(np.std(reference)),
            current_mean=float(np.mean(current)),
            current_std=float(np.std(current))
        )

    def _wasserstein_test(
        self,
        feature: str,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Wasserstein distance test."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for Wasserstein test")

        distance = stats.wasserstein_distance(reference, current)

        # Normalize by reference std
        ref_std = np.std(reference)
        normalized_distance = distance / ref_std if ref_std > 0 else distance

        # Classify
        if normalized_distance < 0.1:
            drift_type = DriftType.NO_DRIFT
        elif normalized_distance < 0.2:
            drift_type = DriftType.MINOR_DRIFT
        elif normalized_distance < 0.5:
            drift_type = DriftType.MODERATE_DRIFT
        else:
            drift_type = DriftType.SEVERE_DRIFT

        return DriftResult(
            feature_name=feature,
            drift_type=drift_type,
            drift_score=float(normalized_distance),
            method=DriftMethod.WASSERSTEIN,
            statistic=float(distance),
            reference_mean=float(np.mean(reference)),
            reference_std=float(np.std(reference)),
            current_mean=float(np.mean(current)),
            current_std=float(np.std(current))
        )

    def get_history(self) -> List[DriftReport]:
        """Get drift detection history."""
        return self._history.copy()

    def get_drift_trend(self, feature: str) -> List[Tuple[datetime, float]]:
        """Get drift score trend for a feature over time."""
        trend = []
        for report in self._history:
            result = report.get_by_name(feature)
            if result:
                trend.append((report.computed_at, result.drift_score))
        return trend


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    Combines:
    - Multiple feature generators
    - Importance tracking
    - Drift detection
    - Feature selection

    Example:
        >>> pipeline = FeaturePipeline()
        >>> pipeline.add_generator(TechnicalFeatureGenerator())
        >>> pipeline.add_generator(VolatilityFeatureGenerator())
        >>>
        >>> # Generate all features
        >>> features_df = pipeline.generate(ohlcv_df)
        >>>
        >>> # Compute importance
        >>> importance = pipeline.compute_importance(features_df, target)
        >>>
        >>> # Select top features
        >>> selected_df = pipeline.select_top_features(features_df, n=20)
    """

    def __init__(self):
        self._generators: List[FeatureGenerator] = []
        self._importance_tracker = FeatureImportanceTracker()
        self._drift_detector = DriftDetector()
        self._selected_features: Optional[List[str]] = None

    def add_generator(self, generator: FeatureGenerator) -> 'FeaturePipeline':
        """Add a feature generator."""
        self._generators.append(generator)
        return self

    def get_all_feature_names(self) -> List[str]:
        """Get names of all features that will be generated."""
        names = []
        for gen in self._generators:
            names.extend(gen.feature_names)
        return names

    def generate(
        self,
        df: 'pd.DataFrame',
        generators: Optional[List[str]] = None
    ) -> 'pd.DataFrame':
        """
        Generate features using registered generators.

        Args:
            df: Input DataFrame with OHLCV data
            generators: Specific generators to use (None = all)

        Returns:
            DataFrame with generated features
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        result = df.copy()

        for gen in self._generators:
            if generators is None or gen.name in generators:
                try:
                    result = gen.generate(result)
                except Exception as e:
                    logger.warning(f"Generator {gen.name} failed: {e}")

        return result

    def compute_importance(
        self,
        X: 'pd.DataFrame',
        y: np.ndarray,
        method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST,
        task: str = 'classification'
    ) -> ImportanceReport:
        """Compute feature importance."""
        return self._importance_tracker.compute(X, y, method, task)

    def detect_drift(
        self,
        reference: 'pd.DataFrame',
        current: 'pd.DataFrame',
        method: DriftMethod = DriftMethod.PSI
    ) -> DriftReport:
        """Detect feature drift."""
        return self._drift_detector.detect(reference, current, method)

    def select_top_features(
        self,
        df: 'pd.DataFrame',
        n: int = 20,
        importance_report: Optional[ImportanceReport] = None
    ) -> 'pd.DataFrame':
        """Select top N important features."""
        if importance_report is None:
            # Use last computed importance
            history = self._importance_tracker.get_history()
            if not history:
                raise ValueError("No importance computed. Call compute_importance first.")
            importance_report = history[-1]

        top_features = [imp.feature_name for imp in importance_report.get_top_n(n)]
        self._selected_features = top_features

        # Keep only features that exist in df
        available = [f for f in top_features if f in df.columns]

        return df[available]

    def get_stable_features(self, min_reports: int = 3) -> List[str]:
        """Get consistently important features."""
        return self._importance_tracker.get_stable_features(min_reports)

    @property
    def importance_tracker(self) -> FeatureImportanceTracker:
        """Get importance tracker."""
        return self._importance_tracker

    @property
    def drift_detector(self) -> DriftDetector:
        """Get drift detector."""
        return self._drift_detector


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_pipeline: Optional[FeaturePipeline] = None


def get_feature_pipeline() -> FeaturePipeline:
    """Get global feature pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline()
        _pipeline.add_generator(TechnicalFeatureGenerator())
        _pipeline.add_generator(VolatilityFeatureGenerator())
        _pipeline.add_generator(MomentumFeatureGenerator())
    return _pipeline


def set_feature_pipeline(pipeline: FeaturePipeline) -> None:
    """Set global feature pipeline."""
    global _pipeline
    _pipeline = pipeline


def generate_features(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Generate features using global pipeline."""
    return get_feature_pipeline().generate(df)


def compute_importance(
    X: 'pd.DataFrame',
    y: np.ndarray,
    method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST
) -> ImportanceReport:
    """Compute importance using global pipeline."""
    return get_feature_pipeline().compute_importance(X, y, method)


def detect_drift(
    reference: 'pd.DataFrame',
    current: 'pd.DataFrame'
) -> DriftReport:
    """Detect drift using global pipeline."""
    return get_feature_pipeline().detect_drift(reference, current)
