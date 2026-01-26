# -*- coding: utf-8 -*-
"""
Correlation Analysis Module - Know Your Asset Relationships!
=============================================================
Calculate and monitor correlations between assets to manage
diversification and avoid concentration risk.

High correlation = assets move together = less diversification!

Example:
    >>> from core.risk import CorrelationAnalyzer
    >>>
    >>> analyzer = CorrelationAnalyzer(lookback=60)
    >>> matrix = analyzer.calculate_correlation_matrix(prices_df)
    >>>
    >>> # Find highly correlated pairs
    >>> high_corr = matrix.get_high_correlations(threshold=0.7)
    >>> print(f"Warning: {len(high_corr)} highly correlated pairs!")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    # Lookback periods
    default_lookback: int = 60          # Days for correlation calculation
    min_lookback: int = 20              # Minimum required data points

    # Thresholds
    high_correlation_threshold: float = 0.7     # Above this = highly correlated
    low_correlation_threshold: float = -0.5     # Below this = negatively correlated

    # Rolling correlation
    rolling_window: int = 30            # Window for rolling correlation
    update_frequency: int = 1           # Days between recalculations

    # Method
    method: str = "pearson"             # pearson, spearman, or kendall


@dataclass
class CorrelationPair:
    """A pair of correlated assets."""
    symbol1: str
    symbol2: str
    correlation: float
    lookback_days: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_high(self) -> bool:
        """Check if correlation is high (>0.7)."""
        return self.correlation > 0.7

    @property
    def is_negative(self) -> bool:
        """Check if correlation is negative."""
        return self.correlation < 0

    def __str__(self) -> str:
        return f"{self.symbol1}-{self.symbol2}: {self.correlation:.2f}"


class CorrelationMatrix:
    """
    Correlation matrix with analysis utilities.

    Wraps a pandas correlation matrix with additional functionality.
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        lookback_days: int,
        timestamp: Optional[datetime] = None
    ):
        self.matrix = matrix
        self.lookback_days = lookback_days
        self.timestamp = timestamp or datetime.now()
        self.symbols = list(matrix.columns)

    def get(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return np.nan
        return self.matrix.loc[symbol1, symbol2]

    def get_correlations_for(self, symbol: str) -> pd.Series:
        """Get all correlations for a symbol."""
        if symbol not in self.symbols:
            return pd.Series()
        return self.matrix[symbol].drop(symbol)

    def get_high_correlations(
        self,
        threshold: float = 0.7
    ) -> List[CorrelationPair]:
        """Find all pairs with correlation above threshold."""
        pairs = []
        n = len(self.symbols)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    pairs.append(CorrelationPair(
                        symbol1=self.symbols[i],
                        symbol2=self.symbols[j],
                        correlation=corr,
                        lookback_days=self.lookback_days,
                        timestamp=self.timestamp
                    ))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs

    def get_low_correlations(
        self,
        threshold: float = 0.3
    ) -> List[CorrelationPair]:
        """Find pairs with low correlation (good for diversification)."""
        pairs = []
        n = len(self.symbols)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix.iloc[i, j]
                if abs(corr) <= threshold:
                    pairs.append(CorrelationPair(
                        symbol1=self.symbols[i],
                        symbol2=self.symbols[j],
                        correlation=corr,
                        lookback_days=self.lookback_days,
                        timestamp=self.timestamp
                    ))

        pairs.sort(key=lambda p: abs(p.correlation))
        return pairs

    def get_negative_correlations(self) -> List[CorrelationPair]:
        """Find negatively correlated pairs (hedging opportunities)."""
        pairs = []
        n = len(self.symbols)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix.iloc[i, j]
                if corr < 0:
                    pairs.append(CorrelationPair(
                        symbol1=self.symbols[i],
                        symbol2=self.symbols[j],
                        correlation=corr,
                        lookback_days=self.lookback_days,
                        timestamp=self.timestamp
                    ))

        pairs.sort(key=lambda p: p.correlation)
        return pairs

    def get_most_correlated(self, symbol: str, n: int = 5) -> List[CorrelationPair]:
        """Get n most correlated symbols to a given symbol."""
        if symbol not in self.symbols:
            return []

        correlations = self.get_correlations_for(symbol)
        top = correlations.abs().nlargest(n)

        pairs = []
        for other_symbol in top.index:
            pairs.append(CorrelationPair(
                symbol1=symbol,
                symbol2=other_symbol,
                correlation=correlations[other_symbol],
                lookback_days=self.lookback_days,
                timestamp=self.timestamp
            ))

        return pairs

    def get_least_correlated(self, symbol: str, n: int = 5) -> List[CorrelationPair]:
        """Get n least correlated symbols (best for diversification)."""
        if symbol not in self.symbols:
            return []

        correlations = self.get_correlations_for(symbol)
        bottom = correlations.abs().nsmallest(n)

        pairs = []
        for other_symbol in bottom.index:
            pairs.append(CorrelationPair(
                symbol1=symbol,
                symbol2=other_symbol,
                correlation=correlations[other_symbol],
                lookback_days=self.lookback_days,
                timestamp=self.timestamp
            ))

        return pairs

    def average_correlation(self) -> float:
        """Calculate average correlation across all pairs."""
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones(self.matrix.shape), k=1).astype(bool)
        upper_values = self.matrix.values[mask]
        return np.nanmean(upper_values)

    def to_heatmap_data(self) -> Dict[str, Any]:
        """Convert to format suitable for heatmap visualization."""
        return {
            'symbols': self.symbols,
            'values': self.matrix.values.tolist(),
            'lookback': self.lookback_days,
            'timestamp': self.timestamp.isoformat()
        }

    def summary(self) -> str:
        """Generate text summary of correlation matrix."""
        high = self.get_high_correlations(0.7)
        negative = self.get_negative_correlations()

        lines = [
            f"Correlation Matrix Summary ({self.timestamp.strftime('%Y-%m-%d')})",
            f"Symbols: {len(self.symbols)}",
            f"Lookback: {self.lookback_days} days",
            f"Average Correlation: {self.average_correlation():.2f}",
            f"High Correlations (>0.7): {len(high)}",
            f"Negative Correlations: {len(negative)}",
        ]

        if high:
            lines.append("\nTop High Correlations:")
            for p in high[:5]:
                lines.append(f"  {p}")

        if negative:
            lines.append("\nNegative Correlations (hedge opportunities):")
            for p in negative[:3]:
                lines.append(f"  {p}")

        return "\n".join(lines)


@dataclass
class RollingCorrelation:
    """Rolling correlation between two assets over time."""
    symbol1: str
    symbol2: str
    correlations: pd.Series       # Time series of correlations
    window: int
    current: float = 0.0

    def get_current(self) -> float:
        """Get most recent correlation."""
        return self.correlations.iloc[-1] if len(self.correlations) > 0 else np.nan

    def get_average(self) -> float:
        """Get average correlation over the period."""
        return self.correlations.mean()

    def get_stability(self) -> float:
        """
        Get correlation stability (inverse of std dev).
        Higher = more stable relationship.
        """
        std = self.correlations.std()
        return 1.0 / (1.0 + std) if std > 0 else 1.0

    def is_strengthening(self, lookback: int = 10) -> bool:
        """Check if correlation is increasing."""
        if len(self.correlations) < lookback:
            return False
        recent = self.correlations.iloc[-lookback:]
        return recent.iloc[-1] > recent.iloc[0]


class CorrelationAnalyzer:
    """
    Correlation analysis engine.

    Calculates and monitors correlations between assets.
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._cache: Dict[str, CorrelationMatrix] = {}
        self._rolling_cache: Dict[Tuple[str, str], RollingCorrelation] = {}

    def calculate_correlation_matrix(
        self,
        prices: pd.DataFrame,
        lookback: Optional[int] = None,
        method: Optional[str] = None
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix from price data.

        Args:
            prices: DataFrame with symbols as columns, dates as index
            lookback: Number of days to use (default: config.default_lookback)
            method: Correlation method (pearson, spearman, kendall)

        Returns:
            CorrelationMatrix with analysis utilities
        """
        lookback = lookback or self.config.default_lookback
        method = method or self.config.method

        # Use last N days
        if len(prices) > lookback:
            prices = prices.iloc[-lookback:]

        if len(prices) < self.config.min_lookback:
            logger.warning(f"Insufficient data: {len(prices)} < {self.config.min_lookback}")

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr(method=method)

        return CorrelationMatrix(
            matrix=corr_matrix,
            lookback_days=len(returns),
            timestamp=datetime.now()
        )

    def calculate_pairwise(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        lookback: Optional[int] = None,
        method: Optional[str] = None
    ) -> float:
        """
        Calculate correlation between two price series.

        Args:
            prices1: First price series
            prices2: Second price series
            lookback: Days to use
            method: Correlation method

        Returns:
            Correlation coefficient
        """
        lookback = lookback or self.config.default_lookback
        method = method or self.config.method

        # Align and trim
        df = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna()

        if len(df) > lookback:
            df = df.iloc[-lookback:]

        if len(df) < self.config.min_lookback:
            return np.nan

        # Calculate returns
        returns1 = df['p1'].pct_change().dropna()
        returns2 = df['p2'].pct_change().dropna()

        return returns1.corr(returns2, method=method)

    def calculate_rolling_correlation(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        window: Optional[int] = None
    ) -> RollingCorrelation:
        """
        Calculate rolling correlation between two assets.

        Args:
            prices1: First price series
            prices2: Second price series
            window: Rolling window size

        Returns:
            RollingCorrelation with time series
        """
        window = window or self.config.rolling_window

        # Align prices
        df = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna()

        # Calculate returns
        returns1 = df['p1'].pct_change()
        returns2 = df['p2'].pct_change()

        # Rolling correlation
        rolling_corr = returns1.rolling(window).corr(returns2)

        return RollingCorrelation(
            symbol1=prices1.name if hasattr(prices1, 'name') else 'asset1',
            symbol2=prices2.name if hasattr(prices2, 'name') else 'asset2',
            correlations=rolling_corr.dropna(),
            window=window,
            current=rolling_corr.iloc[-1] if len(rolling_corr) > 0 else np.nan
        )

    def find_correlated_pairs(
        self,
        prices: pd.DataFrame,
        threshold: float = 0.7,
        lookback: Optional[int] = None
    ) -> List[CorrelationPair]:
        """
        Find all pairs with correlation above threshold.

        Args:
            prices: Price DataFrame
            threshold: Correlation threshold
            lookback: Days to analyze

        Returns:
            List of highly correlated pairs
        """
        matrix = self.calculate_correlation_matrix(prices, lookback)
        return matrix.get_high_correlations(threshold)

    def check_new_position_correlation(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        prices: pd.DataFrame,
        threshold: float = 0.7
    ) -> Tuple[bool, List[CorrelationPair]]:
        """
        Check if adding a new position would create correlation risk.

        Args:
            new_symbol: Symbol to potentially add
            existing_symbols: Current portfolio symbols
            prices: Price data
            threshold: Maximum allowed correlation

        Returns:
            (is_safe, list of high correlations)
        """
        if new_symbol not in prices.columns:
            logger.warning(f"Symbol {new_symbol} not in price data")
            return True, []

        high_correlations = []
        new_prices = prices[new_symbol]

        for symbol in existing_symbols:
            if symbol not in prices.columns:
                continue

            corr = self.calculate_pairwise(new_prices, prices[symbol])

            if abs(corr) >= threshold:
                high_correlations.append(CorrelationPair(
                    symbol1=new_symbol,
                    symbol2=symbol,
                    correlation=corr,
                    lookback_days=self.config.default_lookback
                ))

        is_safe = len(high_correlations) == 0
        return is_safe, high_correlations

    def get_diversification_score(
        self,
        symbols: List[str],
        prices: pd.DataFrame
    ) -> float:
        """
        Calculate diversification score for a set of symbols.

        Score of 1.0 = perfectly uncorrelated (best diversification)
        Score of 0.0 = perfectly correlated (no diversification)

        Returns:
            Diversification score between 0 and 1
        """
        if len(symbols) < 2:
            return 1.0

        # Filter to available symbols
        available = [s for s in symbols if s in prices.columns]
        if len(available) < 2:
            return 1.0

        matrix = self.calculate_correlation_matrix(prices[available])
        avg_corr = matrix.average_correlation()

        # Convert correlation to diversification score
        # avg_corr of 1.0 -> score of 0.0
        # avg_corr of 0.0 -> score of 1.0
        # avg_corr of -1.0 -> score of 1.0 (negative correlation = good)
        return max(0, 1 - abs(avg_corr))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_correlation(
    prices: pd.DataFrame,
    lookback: int = 60
) -> CorrelationMatrix:
    """Quick correlation matrix calculation."""
    analyzer = CorrelationAnalyzer()
    return analyzer.calculate_correlation_matrix(prices, lookback)


def calculate_rolling_correlation(
    prices1: pd.Series,
    prices2: pd.Series,
    window: int = 30
) -> RollingCorrelation:
    """Quick rolling correlation calculation."""
    analyzer = CorrelationAnalyzer()
    return analyzer.calculate_rolling_correlation(prices1, prices2, window)


def find_high_correlations(
    prices: pd.DataFrame,
    threshold: float = 0.7
) -> List[CorrelationPair]:
    """Find highly correlated pairs quickly."""
    analyzer = CorrelationAnalyzer()
    return analyzer.find_correlated_pairs(prices, threshold)
