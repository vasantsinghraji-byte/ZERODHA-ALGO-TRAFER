# -*- coding: utf-8 -*-
"""
Feature Definitions - Pre-calculated Trading Features
======================================================
Defines and calculates features for ML models.

Features included:
- Technical indicators (RSI, MACD, Bollinger, etc.)
- Volatility metrics (ATR, realized vol, Parkinson)
- Order flow indicators (OFI, VWAP deviation, volume profile)
- Market microstructure (spread, depth imbalance)

Example:
    >>> from ml.feature_store import FeatureCalculator, FeatureSet
    >>>
    >>> calc = FeatureCalculator()
    >>> features = calc.calculate_all(df)
    >>> print(features.columns)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime, timedelta
import numpy as np
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Try to use Numba-optimized functions if available
try:
    from core.infrastructure import (
        fast_sma, fast_ema, fast_rsi, fast_macd,
        fast_bollinger, fast_atr, fast_stochastic
    )
    FAST_FUNCTIONS_AVAILABLE = True
except ImportError:
    FAST_FUNCTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Feature categories."""
    PRICE = "price"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    ORDER_FLOW = "order_flow"
    MICROSTRUCTURE = "microstructure"
    TIME = "time"
    CUSTOM = "custom"


class FeatureFrequency(Enum):
    """Feature calculation frequency."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    DAY = "1d"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    category: FeatureCategory
    description: str = ""

    # Calculation settings
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    # Data requirements
    required_columns: List[str] = field(default_factory=list)
    lookback_periods: int = 1

    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    # Output type
    output_type: str = "float64"  # float64, int64, bool, category

    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureValue:
    """A calculated feature value."""
    name: str
    value: Any
    timestamp: datetime
    symbol: str = ""
    version: str = "1.0.0"

    # Metadata
    calculation_time_ms: float = 0.0
    is_valid: bool = True
    error_message: str = ""


class FeatureSet:
    """
    Collection of features for a symbol.

    Example:
        >>> features = FeatureSet("RELIANCE")
        >>> features.add("rsi_14", 65.5)
        >>> features.add("volatility_20", 0.023)
        >>> df = features.to_dataframe()
    """

    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.timestamp = datetime.now()
        self._features: Dict[str, FeatureValue] = {}

    def add(
        self,
        name: str,
        value: Any,
        version: str = "1.0.0",
        is_valid: bool = True
    ) -> None:
        """Add a feature value."""
        self._features[name] = FeatureValue(
            name=name,
            value=value,
            timestamp=self.timestamp,
            symbol=self.symbol,
            version=version,
            is_valid=is_valid
        )

    def get(self, name: str) -> Optional[Any]:
        """Get feature value by name."""
        fv = self._features.get(name)
        return fv.value if fv else None

    def get_all(self) -> Dict[str, Any]:
        """Get all feature values as dict."""
        return {name: fv.value for name, fv in self._features.items()}

    @property
    def names(self) -> List[str]:
        """Get all feature names."""
        return list(self._features.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'features': self.get_all()
        }

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for to_dataframe()")

        data = self.get_all()
        data['symbol'] = self.symbol
        data['timestamp'] = self.timestamp
        return pd.DataFrame([data])

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureSet({self.symbol}, {len(self)} features)"


class FeatureCalculator:
    """
    Calculates trading features from OHLCV data.

    Uses Numba-optimized functions when available for speed.

    Example:
        >>> calc = FeatureCalculator()
        >>>
        >>> # Calculate all features
        >>> df_with_features = calc.calculate_all(ohlcv_df)
        >>>
        >>> # Calculate specific features
        >>> rsi = calc.calculate_rsi(close_prices, period=14)
    """

    def __init__(self, use_fast_functions: bool = True):
        self.use_fast = use_fast_functions and FAST_FUNCTIONS_AVAILABLE

        # Feature registry
        self._features: Dict[str, FeatureDefinition] = {}
        self._register_default_features()

    def _register_default_features(self) -> None:
        """Register default feature definitions."""
        # Technical indicators
        self.register_feature(FeatureDefinition(
            name="sma_20",
            category=FeatureCategory.TECHNICAL,
            description="20-period Simple Moving Average",
            parameters={"period": 20},
            required_columns=["close"],
            lookback_periods=20
        ))

        self.register_feature(FeatureDefinition(
            name="ema_12",
            category=FeatureCategory.TECHNICAL,
            description="12-period Exponential Moving Average",
            parameters={"period": 12},
            required_columns=["close"],
            lookback_periods=12
        ))

        self.register_feature(FeatureDefinition(
            name="rsi_14",
            category=FeatureCategory.TECHNICAL,
            description="14-period Relative Strength Index",
            parameters={"period": 14},
            required_columns=["close"],
            lookback_periods=15
        ))

        self.register_feature(FeatureDefinition(
            name="macd",
            category=FeatureCategory.TECHNICAL,
            description="MACD Line",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            required_columns=["close"],
            lookback_periods=35
        ))

        self.register_feature(FeatureDefinition(
            name="bb_upper",
            category=FeatureCategory.TECHNICAL,
            description="Bollinger Band Upper",
            parameters={"period": 20, "std_dev": 2.0},
            required_columns=["close"],
            lookback_periods=20
        ))

        # Volatility features
        self.register_feature(FeatureDefinition(
            name="atr_14",
            category=FeatureCategory.VOLATILITY,
            description="14-period Average True Range",
            parameters={"period": 14},
            required_columns=["high", "low", "close"],
            lookback_periods=15
        ))

        self.register_feature(FeatureDefinition(
            name="volatility_20",
            category=FeatureCategory.VOLATILITY,
            description="20-period Realized Volatility",
            parameters={"period": 20},
            required_columns=["close"],
            lookback_periods=21
        ))

        # Volume features
        self.register_feature(FeatureDefinition(
            name="volume_sma_20",
            category=FeatureCategory.VOLUME,
            description="20-period Volume SMA",
            parameters={"period": 20},
            required_columns=["volume"],
            lookback_periods=20
        ))

        self.register_feature(FeatureDefinition(
            name="volume_ratio",
            category=FeatureCategory.VOLUME,
            description="Volume vs Average Ratio",
            parameters={"period": 20},
            required_columns=["volume"],
            lookback_periods=20
        ))

    def register_feature(self, definition: FeatureDefinition) -> None:
        """Register a feature definition."""
        self._features[definition.name] = definition

    def get_feature_definitions(self) -> List[FeatureDefinition]:
        """Get all registered feature definitions."""
        return list(self._features.values())

    # ==================== TECHNICAL INDICATORS ====================

    def calculate_sma(
        self,
        prices: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if self.use_fast:
            return fast_sma(prices, period)

        prices = np.asarray(prices, dtype=np.float64)
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def calculate_ema(
        self,
        prices: np.ndarray,
        period: int = 12
    ) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if self.use_fast:
            return fast_ema(prices, period)

        prices = np.asarray(prices, dtype=np.float64)
        result = np.full(len(prices), np.nan)
        multiplier = 2.0 / (period + 1)

        result[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def calculate_rsi(
        self,
        prices: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate Relative Strength Index."""
        if self.use_fast:
            return fast_rsi(prices, period)

        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)
        result = np.full(n, np.nan)

        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, n - 1):
            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return result

    def calculate_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple:
        """
        Calculate MACD.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if self.use_fast:
            return fast_macd(prices, fast_period, slow_period, signal_period)

        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if self.use_fast:
            return fast_bollinger(prices, period, std_dev)

        prices = np.asarray(prices, dtype=np.float64)
        middle = self.calculate_sma(prices, period)

        n = len(prices)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = prices[i - period + 1:i + 1]
            std = np.std(window)
            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std

        return upper, middle, lower

    def calculate_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple:
        """
        Calculate Stochastic Oscillator.

        Returns:
            Tuple of (k_line, d_line)
        """
        if self.use_fast:
            return fast_stochastic(high, low, close, k_period, d_period)

        n = len(high)
        k_line = np.full(n, np.nan)

        for i in range(k_period - 1, n):
            hh = np.max(high[i - k_period + 1:i + 1])
            ll = np.min(low[i - k_period + 1:i + 1])
            if hh - ll == 0:
                k_line[i] = 50.0
            else:
                k_line[i] = ((close[i] - ll) / (hh - ll)) * 100.0

        d_line = self.calculate_sma(k_line, d_period)
        return k_line, d_line

    # ==================== VOLATILITY FEATURES ====================

    def calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate Average True Range."""
        if self.use_fast:
            return fast_atr(high, low, close, period)

        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        return self.calculate_sma(tr, period)

    def calculate_realized_volatility(
        self,
        prices: np.ndarray,
        period: int = 20,
        annualize: bool = True
    ) -> np.ndarray:
        """
        Calculate Realized Volatility.

        Uses log returns and standard deviation.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)
        result = np.full(n, np.nan)

        # Calculate log returns
        log_returns = np.diff(np.log(prices))

        for i in range(period, n):
            window = log_returns[i - period:i]
            vol = np.std(window)
            if annualize:
                vol *= np.sqrt(252)  # Annualize
            result[i] = vol

        return result

    def calculate_parkinson_volatility(
        self,
        high: np.ndarray,
        low: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """
        Calculate Parkinson Volatility.

        Uses high-low range, more efficient than close-to-close.
        """
        n = len(high)
        result = np.full(n, np.nan)

        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))

        for i in range(period - 1, n):
            window = log_hl[i - period + 1:i + 1]
            result[i] = np.sqrt(factor * np.mean(window) * 252)

        return result

    # ==================== VOLUME FEATURES ====================

    def calculate_vwap(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)

        # Avoid division by zero
        cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)

        return cumulative_tp_vol / cumulative_vol

    def calculate_volume_ratio(
        self,
        volume: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Calculate volume relative to average."""
        volume = np.asarray(volume, dtype=np.float64)
        avg_volume = self.calculate_sma(volume, period)

        # Avoid division by zero
        avg_volume = np.where(avg_volume == 0, 1, avg_volume)

        return volume / avg_volume

    def calculate_obv(
        self,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """Calculate On-Balance Volume."""
        n = len(close)
        obv = np.zeros(n)
        obv[0] = volume[0]

        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    # ==================== ORDER FLOW FEATURES ====================

    def calculate_order_flow_imbalance(
        self,
        bid_volume: np.ndarray,
        ask_volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Order Flow Imbalance (OFI).

        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        """
        total = bid_volume + ask_volume
        total = np.where(total == 0, 1, total)
        return (bid_volume - ask_volume) / total

    def calculate_vwap_deviation(
        self,
        close: np.ndarray,
        vwap: np.ndarray
    ) -> np.ndarray:
        """Calculate price deviation from VWAP."""
        vwap = np.where(vwap == 0, 1, vwap)
        return (close - vwap) / vwap * 100

    def calculate_trade_flow(
        self,
        price: np.ndarray,
        volume: np.ndarray,
        bid: np.ndarray,
        ask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate trade flow direction.

        Classifies trades as buy or sell based on price relative to bid/ask.
        """
        mid = (bid + ask) / 2

        # Above mid = likely buy, below mid = likely sell
        direction = np.where(price >= mid, 1, -1)

        return direction * volume

    # ==================== TIME FEATURES ====================

    def calculate_time_features(
        self,
        timestamps: Union[np.ndarray, List[datetime]]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate time-based features.

        Returns dict with:
        - hour: Hour of day (0-23)
        - minute: Minute of hour (0-59)
        - day_of_week: Day of week (0-6, Mon=0)
        - is_market_open: Near market open (9:15-10:00)
        - is_market_close: Near market close (15:00-15:30)
        """
        n = len(timestamps)

        hour = np.zeros(n, dtype=np.int32)
        minute = np.zeros(n, dtype=np.int32)
        day_of_week = np.zeros(n, dtype=np.int32)
        is_open = np.zeros(n, dtype=np.bool_)
        is_close = np.zeros(n, dtype=np.bool_)

        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                hour[i] = ts.hour
                minute[i] = ts.minute
                day_of_week[i] = ts.weekday()

                # Market open: 9:15 - 10:00
                is_open[i] = (ts.hour == 9 and ts.minute >= 15) or (ts.hour == 10 and ts.minute == 0)

                # Market close: 15:00 - 15:30
                is_close[i] = ts.hour == 15 and ts.minute <= 30

        return {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_market_open': is_open,
            'is_market_close': is_close
        }

    # ==================== ALL FEATURES ====================

    def calculate_all(
        self,
        df: 'pd.DataFrame',
        features: Optional[List[str]] = None
    ) -> 'pd.DataFrame':
        """
        Calculate all features for a DataFrame.

        Args:
            df: DataFrame with OHLCV columns
            features: List of feature names to calculate (None = all)

        Returns:
            DataFrame with original data plus features
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for calculate_all()")

        result = df.copy()

        # Extract arrays
        close = df['close'].values if 'close' in df.columns else None
        high = df['high'].values if 'high' in df.columns else None
        low = df['low'].values if 'low' in df.columns else None
        volume = df['volume'].values if 'volume' in df.columns else None

        if close is None:
            raise ValueError("DataFrame must have 'close' column")

        # Technical indicators
        if features is None or 'sma_20' in features:
            result['sma_20'] = self.calculate_sma(close, 20)

        if features is None or 'sma_50' in features:
            result['sma_50'] = self.calculate_sma(close, 50)

        if features is None or 'ema_12' in features:
            result['ema_12'] = self.calculate_ema(close, 12)

        if features is None or 'ema_26' in features:
            result['ema_26'] = self.calculate_ema(close, 26)

        if features is None or 'rsi_14' in features:
            result['rsi_14'] = self.calculate_rsi(close, 14)

        if features is None or 'macd' in features:
            macd, signal, hist = self.calculate_macd(close)
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_histogram'] = hist

        if features is None or 'bb_upper' in features:
            upper, middle, lower = self.calculate_bollinger_bands(close)
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower

        if high is not None and low is not None:
            if features is None or 'stoch_k' in features:
                k, d = self.calculate_stochastic(high, low, close)
                result['stoch_k'] = k
                result['stoch_d'] = d

            if features is None or 'atr_14' in features:
                result['atr_14'] = self.calculate_atr(high, low, close, 14)

            if features is None or 'parkinson_vol' in features:
                result['parkinson_vol'] = self.calculate_parkinson_volatility(high, low)

        # Volatility
        if features is None or 'volatility_20' in features:
            result['volatility_20'] = self.calculate_realized_volatility(close, 20)

        # Volume features
        if volume is not None:
            if features is None or 'volume_ratio' in features:
                result['volume_ratio'] = self.calculate_volume_ratio(volume, 20)

            if features is None or 'obv' in features:
                result['obv'] = self.calculate_obv(close, volume)

            if high is not None and low is not None:
                if features is None or 'vwap' in features:
                    result['vwap'] = self.calculate_vwap(high, low, close, volume)
                    result['vwap_deviation'] = self.calculate_vwap_deviation(close, result['vwap'].values)

        # Price features
        if features is None or 'returns' in features:
            result['returns'] = np.concatenate([[np.nan], np.diff(close) / close[:-1]])

        if features is None or 'log_returns' in features:
            result['log_returns'] = np.concatenate([[np.nan], np.diff(np.log(close))])

        # Momentum
        if features is None or 'momentum_10' in features:
            result['momentum_10'] = close / np.roll(close, 10) - 1
            result['momentum_10'][:10] = np.nan

        return result

    def calculate_for_symbol(
        self,
        df: 'pd.DataFrame',
        symbol: str
    ) -> FeatureSet:
        """
        Calculate features and return as FeatureSet.

        Useful for real-time feature calculation.
        """
        df_with_features = self.calculate_all(df)

        # Get last row values
        last_row = df_with_features.iloc[-1]

        feature_set = FeatureSet(symbol)

        # Add all feature columns (excluding OHLCV)
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'date'}

        for col in df_with_features.columns:
            if col.lower() not in ohlcv_cols:
                value = last_row[col]
                if pd.notna(value):
                    feature_set.add(col, float(value))

        return feature_set


# Convenience functions
_calculator: Optional[FeatureCalculator] = None


def get_feature_calculator() -> FeatureCalculator:
    """Get global feature calculator."""
    global _calculator
    if _calculator is None:
        _calculator = FeatureCalculator()
    return _calculator


def calculate_features(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Calculate all features using global calculator."""
    return get_feature_calculator().calculate_all(df)


def get_feature_definitions() -> List[FeatureDefinition]:
    """Get all registered feature definitions."""
    return get_feature_calculator().get_feature_definitions()
