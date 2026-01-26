# -*- coding: utf-8 -*-
"""
Hot Path Optimization - Critical Path Performance
=================================================
Optimizes the critical tick-to-order path for minimal latency.

The "Hot Path" is: Market Tick → Signal Generation → Order Execution
This module provides:
- Profiling tools to identify bottlenecks
- Numba JIT-compiled fast calculations
- Optimized data structures
- Performance monitoring

Example:
    >>> from core.infrastructure import HotPathProfiler, fast_sma, fast_ema
    >>>
    >>> # Profile a function
    >>> profiler = HotPathProfiler()
    >>> with profiler.profile("tick_processing"):
    ...     process_tick(tick)
    >>> profiler.report()
    >>>
    >>> # Use optimized calculations
    >>> sma = fast_sma(prices, period=20)
    >>> ema = fast_ema(prices, period=12)
"""

import cProfile
import pstats
import io
import time
import functools
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from contextlib import contextmanager
import logging

# Try to import Numba for JIT compilation
try:
    from numba import jit, njit, prange
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback - create no-op decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not callable(args[0]) else args[0]

    def njit(*args, **kwargs):
        return jit(*args, **kwargs)

    def prange(*args):
        return range(*args)

    import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# PROFILING UTILITIES
# ============================================================

@dataclass
class ProfileResult:
    """Result of profiling a code section."""
    name: str
    total_time_ms: float
    call_count: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_fast(self) -> bool:
        """Check if execution is fast enough (<1ms p95)."""
        return self.p95_time_ms < 1.0

    @property
    def is_acceptable(self) -> bool:
        """Check if execution is acceptable (<5ms p95)."""
        return self.p95_time_ms < 5.0


@dataclass
class HotPathMetrics:
    """Metrics for the entire hot path."""
    tick_to_signal_ms: float = 0.0
    signal_to_order_ms: float = 0.0
    order_to_ack_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Component breakdown
    components: Dict[str, float] = field(default_factory=dict)

    # Thresholds
    target_total_ms: float = 10.0

    @property
    def is_within_target(self) -> bool:
        """Check if total latency is within target."""
        return self.total_latency_ms <= self.target_total_ms

    @property
    def slowest_component(self) -> Tuple[str, float]:
        """Get the slowest component."""
        if not self.components:
            return ("unknown", 0.0)
        return max(self.components.items(), key=lambda x: x[1])


class HotPathProfiler:
    """
    Profiler for hot path optimization.

    Tracks execution times of critical code sections and provides
    detailed analysis and recommendations.

    Example:
        >>> profiler = HotPathProfiler()
        >>>
        >>> # Manual timing
        >>> with profiler.profile("calculate_signal"):
        ...     signal = strategy.calculate(tick)
        >>>
        >>> # Decorator
        >>> @profiler.track("order_submission")
        ... def submit_order(order):
        ...     return broker.place_order(order)
        >>>
        >>> # Get report
        >>> profiler.report()
    """

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._timings: Dict[str, deque] = {}
        self._call_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

        # cProfile for detailed analysis
        self._profiler: Optional[cProfile.Profile] = None
        self._profiling_active = False

    @contextmanager
    def profile(self, name: str):
        """
        Context manager to profile a code section.

        Example:
            >>> with profiler.profile("tick_processing"):
            ...     process_tick(tick)
        """
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_ms = elapsed_ns / 1_000_000

            with self._lock:
                if name not in self._timings:
                    self._timings[name] = deque(maxlen=self.max_samples)
                    self._call_counts[name] = 0

                self._timings[name].append(elapsed_ms)
                self._call_counts[name] += 1

    def track(self, name: str) -> Callable:
        """
        Decorator to track function execution time.

        Example:
            >>> @profiler.track("calculate_ema")
            ... def calculate_ema(prices, period):
            ...     return ema_result
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def start_detailed_profile(self) -> None:
        """Start detailed cProfile profiling."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._profiling_active = True

    def stop_detailed_profile(self) -> str:
        """Stop detailed profiling and return stats."""
        if not self._profiler:
            return "No profiler active"

        self._profiler.disable()
        self._profiling_active = False

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(30)  # Top 30 functions

        return stream.getvalue()

    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get profiling result for a named section."""
        with self._lock:
            if name not in self._timings or not self._timings[name]:
                return None

            timings = list(self._timings[name])
            sorted_timings = sorted(timings)
            n = len(sorted_timings)

            return ProfileResult(
                name=name,
                total_time_ms=sum(timings),
                call_count=self._call_counts.get(name, 0),
                avg_time_ms=sum(timings) / n,
                min_time_ms=sorted_timings[0],
                max_time_ms=sorted_timings[-1],
                p50_time_ms=sorted_timings[n // 2],
                p95_time_ms=sorted_timings[int(n * 0.95)] if n >= 20 else sorted_timings[-1],
                p99_time_ms=sorted_timings[int(n * 0.99)] if n >= 100 else sorted_timings[-1]
            )

    def get_all_results(self) -> Dict[str, ProfileResult]:
        """Get all profiling results."""
        results = {}
        with self._lock:
            for name in self._timings.keys():
                result = self.get_result(name)
                if result:
                    results[name] = result
        return results

    def get_hot_path_metrics(self) -> HotPathMetrics:
        """Get aggregated hot path metrics."""
        results = self.get_all_results()

        components = {name: r.avg_time_ms for name, r in results.items()}

        # Try to identify standard hot path components
        tick_to_signal = sum(
            r.avg_time_ms for name, r in results.items()
            if any(k in name.lower() for k in ['tick', 'signal', 'indicator', 'calculate'])
        )

        signal_to_order = sum(
            r.avg_time_ms for name, r in results.items()
            if any(k in name.lower() for k in ['order', 'submit', 'execute'])
        )

        total = sum(r.avg_time_ms for r in results.values())

        return HotPathMetrics(
            tick_to_signal_ms=tick_to_signal,
            signal_to_order_ms=signal_to_order,
            total_latency_ms=total,
            components=components
        )

    def report(self) -> str:
        """Generate a performance report."""
        results = self.get_all_results()

        if not results:
            return "No profiling data collected yet."

        lines = [
            "=" * 60,
            "HOT PATH PERFORMANCE REPORT",
            "=" * 60,
            ""
        ]

        # Sort by average time (slowest first)
        sorted_results = sorted(
            results.values(),
            key=lambda r: r.avg_time_ms,
            reverse=True
        )

        lines.append(f"{'Component':<30} {'Avg(ms)':<10} {'P95(ms)':<10} {'Calls':<10}")
        lines.append("-" * 60)

        for r in sorted_results:
            status = "OK" if r.is_fast else ("SLOW" if not r.is_acceptable else "WARN")
            lines.append(
                f"{r.name:<30} {r.avg_time_ms:<10.3f} {r.p95_time_ms:<10.3f} {r.call_count:<10} [{status}]"
            )

        # Summary
        metrics = self.get_hot_path_metrics()
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Total Hot Path Latency: {metrics.total_latency_ms:.3f} ms")
        lines.append(f"Target: {metrics.target_total_ms:.1f} ms")
        lines.append(f"Status: {'PASS' if metrics.is_within_target else 'FAIL'}")

        if not metrics.is_within_target:
            slowest_name, slowest_time = metrics.slowest_component
            lines.append(f"Bottleneck: {slowest_name} ({slowest_time:.3f} ms)")

        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._timings.clear()
            self._call_counts.clear()

    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on profiling data."""
        suggestions = []
        results = self.get_all_results()

        for name, r in results.items():
            if r.p95_time_ms > 5.0:
                suggestions.append(
                    f"[CRITICAL] '{name}' is slow ({r.p95_time_ms:.1f}ms p95). "
                    f"Consider Numba JIT or caching."
                )
            elif r.p95_time_ms > 1.0:
                suggestions.append(
                    f"[WARNING] '{name}' could be faster ({r.p95_time_ms:.1f}ms p95). "
                    f"Review algorithm complexity."
                )

            # High call count with moderate time
            if r.call_count > 1000 and r.avg_time_ms > 0.1:
                suggestions.append(
                    f"[INFO] '{name}' called {r.call_count} times. "
                    f"Consider batching or caching results."
                )

        return suggestions


# ============================================================
# NUMBA JIT OPTIMIZED FUNCTIONS
# ============================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _fast_sma_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Simple Moving Average."""
        n = len(prices)
        result = np.empty(n, dtype=np.float64)
        result[:period-1] = np.nan

        # First SMA
        window_sum = 0.0
        for i in range(period):
            window_sum += prices[i]
        result[period-1] = window_sum / period

        # Subsequent SMAs using sliding window
        for i in range(period, n):
            window_sum = window_sum - prices[i-period] + prices[i]
            result[i] = window_sum / period

        return result

    @njit(cache=True, fastmath=True)
    def _fast_ema_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Exponential Moving Average."""
        n = len(prices)
        result = np.empty(n, dtype=np.float64)
        result[:period-1] = np.nan

        # First EMA is SMA
        sma = 0.0
        for i in range(period):
            sma += prices[i]
        sma /= period
        result[period-1] = sma

        # EMA multiplier
        multiplier = 2.0 / (period + 1)

        # Calculate EMA
        for i in range(period, n):
            result[i] = (prices[i] - result[i-1]) * multiplier + result[i-1]

        return result

    @njit(cache=True, fastmath=True)
    def _fast_rsi_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Relative Strength Index."""
        n = len(prices)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Calculate price changes
        gains = np.empty(n-1, dtype=np.float64)
        losses = np.empty(n-1, dtype=np.float64)

        for i in range(n-1):
            change = prices[i+1] - prices[i]
            if change > 0:
                gains[i] = change
                losses[i] = 0.0
            else:
                gains[i] = 0.0
                losses[i] = -change

        # First average
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(period):
            avg_gain += gains[i]
            avg_loss += losses[i]
        avg_gain /= period
        avg_loss /= period

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # Subsequent RSI using smoothed averages
        for i in range(period, n-1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i+1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i+1] = 100.0 - (100.0 / (1.0 + rs))

        return result

    @njit(cache=True, fastmath=True)
    def _fast_macd_impl(prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized MACD calculation."""
        n = len(prices)

        # Calculate EMAs
        ema_fast = _fast_ema_impl(prices, fast)
        ema_slow = _fast_ema_impl(prices, slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        # Need to handle NaN values
        valid_start = slow - 1
        macd_valid = macd_line[valid_start:]
        signal_line_valid = _fast_ema_impl(macd_valid, signal)

        signal_line = np.empty(n, dtype=np.float64)
        signal_line[:valid_start] = np.nan
        signal_line[valid_start:] = signal_line_valid

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @njit(cache=True, fastmath=True)
    def _fast_bollinger_impl(prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized Bollinger Bands."""
        n = len(prices)
        middle = _fast_sma_impl(prices, period)

        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        upper[:period-1] = np.nan
        lower[:period-1] = np.nan

        for i in range(period-1, n):
            # Calculate standard deviation for window
            window_sum = 0.0
            window_sum_sq = 0.0
            for j in range(i - period + 1, i + 1):
                window_sum += prices[j]
                window_sum_sq += prices[j] * prices[j]

            mean = window_sum / period
            variance = (window_sum_sq / period) - (mean * mean)
            std = np.sqrt(max(variance, 0.0))

            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std

        return upper, middle, lower

    @njit(cache=True, fastmath=True)
    def _fast_atr_impl(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Average True Range."""
        n = len(high)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Calculate True Range
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # First ATR
        atr_sum = 0.0
        for i in range(period):
            atr_sum += tr[i]
        result[period-1] = atr_sum / period

        # Subsequent ATR using smoothing
        for i in range(period, n):
            result[i] = (result[i-1] * (period - 1) + tr[i]) / period

        return result

    @njit(cache=True, fastmath=True)
    def _fast_stochastic_impl(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized Stochastic Oscillator."""
        n = len(high)
        k_line = np.empty(n, dtype=np.float64)
        k_line[:k_period-1] = np.nan

        # Calculate %K
        for i in range(k_period-1, n):
            highest_high = high[i]
            lowest_low = low[i]
            for j in range(i - k_period + 1, i):
                if high[j] > highest_high:
                    highest_high = high[j]
                if low[j] < lowest_low:
                    lowest_low = low[j]

            range_hl = highest_high - lowest_low
            if range_hl == 0:
                k_line[i] = 50.0
            else:
                k_line[i] = ((close[i] - lowest_low) / range_hl) * 100.0

        # Calculate %D (SMA of %K)
        d_line = _fast_sma_impl(k_line, d_period)

        return k_line, d_line

    @njit(cache=True, fastmath=True, parallel=True)
    def _fast_correlation_matrix(returns: np.ndarray) -> np.ndarray:
        """Numba-optimized correlation matrix calculation."""
        n_assets = returns.shape[1]
        corr = np.empty((n_assets, n_assets), dtype=np.float64)

        for i in prange(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    # Calculate correlation
                    x = returns[:, i]
                    y = returns[:, j]
                    n = len(x)

                    sum_x = 0.0
                    sum_y = 0.0
                    sum_xy = 0.0
                    sum_x2 = 0.0
                    sum_y2 = 0.0

                    for k in range(n):
                        sum_x += x[k]
                        sum_y += y[k]
                        sum_xy += x[k] * y[k]
                        sum_x2 += x[k] * x[k]
                        sum_y2 += y[k] * y[k]

                    num = n * sum_xy - sum_x * sum_y
                    den = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

                    if den == 0:
                        corr[i, j] = 0.0
                    else:
                        corr[i, j] = num / den

                    corr[j, i] = corr[i, j]

        return corr

else:
    # Fallback implementations without Numba
    def _fast_sma_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback SMA implementation."""
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _fast_ema_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback EMA implementation."""
        result = np.full(len(prices), np.nan)
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _fast_rsi_impl(prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback RSI implementation."""
        result = np.full(len(prices), np.nan)
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(prices)):
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))

            if i < len(gains):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return result

    def _fast_macd_impl(prices, fast, slow, signal):
        macd_line = _fast_ema_impl(prices, fast) - _fast_ema_impl(prices, slow)
        signal_line = _fast_ema_impl(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _fast_bollinger_impl(prices, period, std_dev):
        middle = _fast_sma_impl(prices, period)
        rolling_std = np.array([
            np.std(prices[max(0, i-period+1):i+1]) if i >= period-1 else np.nan
            for i in range(len(prices))
        ])
        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std
        return upper, middle, lower

    def _fast_atr_impl(high, low, close, period):
        n = len(high)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        result = _fast_sma_impl(tr, period)
        return result

    def _fast_stochastic_impl(high, low, close, k_period, d_period):
        n = len(high)
        k_line = np.full(n, np.nan)
        for i in range(k_period - 1, n):
            hh = np.max(high[i - k_period + 1:i + 1])
            ll = np.min(low[i - k_period + 1:i + 1])
            if hh - ll == 0:
                k_line[i] = 50.0
            else:
                k_line[i] = ((close[i] - ll) / (hh - ll)) * 100.0
        d_line = _fast_sma_impl(k_line, d_period)
        return k_line, d_line

    def _fast_correlation_matrix(returns):
        return np.corrcoef(returns.T)


# ============================================================
# PUBLIC API - FAST FUNCTIONS
# ============================================================

def fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast Simple Moving Average.

    Uses Numba JIT compilation for ~10-50x speedup on large arrays.

    Args:
        prices: Price array
        period: SMA period

    Returns:
        SMA values (NaN for initial period-1 values)
    """
    prices = np.asarray(prices, dtype=np.float64)
    return _fast_sma_impl(prices, period)


def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast Exponential Moving Average.

    Args:
        prices: Price array
        period: EMA period

    Returns:
        EMA values
    """
    prices = np.asarray(prices, dtype=np.float64)
    return _fast_ema_impl(prices, period)


def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Fast Relative Strength Index.

    Args:
        prices: Price array
        period: RSI period (default 14)

    Returns:
        RSI values (0-100)
    """
    prices = np.asarray(prices, dtype=np.float64)
    return _fast_rsi_impl(prices, period)


def fast_macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast MACD calculation.

    Args:
        prices: Price array
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    prices = np.asarray(prices, dtype=np.float64)
    return _fast_macd_impl(prices, fast_period, slow_period, signal_period)


def fast_bollinger(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast Bollinger Bands.

    Args:
        prices: Price array
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    prices = np.asarray(prices, dtype=np.float64)
    return _fast_bollinger_impl(prices, period, std_dev)


def fast_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Fast Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        ATR values
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _fast_atr_impl(high, low, close, period)


def fast_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)

    Returns:
        Tuple of (k_line, d_line)
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _fast_stochastic_impl(high, low, close, k_period, d_period)


def fast_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Fast correlation matrix calculation.

    Uses parallel processing for large matrices.

    Args:
        returns: 2D array of returns (rows=time, cols=assets)

    Returns:
        Correlation matrix
    """
    returns = np.asarray(returns, dtype=np.float64)
    return _fast_correlation_matrix(returns)


# ============================================================
# HOT PATH TICK PROCESSOR
# ============================================================

@dataclass
class OptimizedTick:
    """Optimized tick data structure for hot path."""
    __slots__ = ['symbol', 'price', 'volume', 'timestamp_ns', 'bid', 'ask']

    symbol: str
    price: float
    volume: int
    timestamp_ns: int
    bid: float
    ask: float


class HotPathTickProcessor:
    """
    Optimized tick processor for the hot path.

    Minimizes allocations and uses pre-allocated buffers.

    Example:
        >>> processor = HotPathTickProcessor(buffer_size=1000)
        >>> processor.process_tick(tick)
        >>> signal = processor.get_signal()
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        sma_period: int = 20,
        ema_period: int = 12,
        rsi_period: int = 14
    ):
        self.buffer_size = buffer_size
        self.sma_period = sma_period
        self.ema_period = ema_period
        self.rsi_period = rsi_period

        # Pre-allocated price buffer
        self._prices = np.zeros(buffer_size, dtype=np.float64)
        self._volumes = np.zeros(buffer_size, dtype=np.int64)
        self._index = 0
        self._count = 0

        # Cached indicators
        self._last_sma = 0.0
        self._last_ema = 0.0
        self._last_rsi = 50.0

        # EMA state for incremental update
        self._ema_multiplier = 2.0 / (ema_period + 1)

    def process_tick(self, price: float, volume: int = 0) -> None:
        """
        Process a single tick with minimal overhead.

        Updates internal state and indicators incrementally.
        """
        # Store in circular buffer
        self._prices[self._index] = price
        self._volumes[self._index] = volume
        self._index = (self._index + 1) % self.buffer_size
        self._count = min(self._count + 1, self.buffer_size)

        # Update EMA incrementally (O(1) operation)
        if self._count == 1:
            self._last_ema = price
        else:
            self._last_ema = (price - self._last_ema) * self._ema_multiplier + self._last_ema

    def get_prices(self, n: Optional[int] = None) -> np.ndarray:
        """Get last n prices from buffer."""
        if n is None:
            n = self._count

        n = min(n, self._count)

        if self._count < self.buffer_size:
            return self._prices[:self._count][-n:]
        else:
            # Handle circular buffer wrap
            end_idx = self._index
            start_idx = (end_idx - n) % self.buffer_size

            if start_idx < end_idx:
                return self._prices[start_idx:end_idx]
            else:
                return np.concatenate([
                    self._prices[start_idx:],
                    self._prices[:end_idx]
                ])

    def get_sma(self) -> float:
        """Get current SMA value."""
        if self._count < self.sma_period:
            return np.nan

        prices = self.get_prices(self.sma_period)
        self._last_sma = np.mean(prices)
        return self._last_sma

    def get_ema(self) -> float:
        """Get current EMA value (incrementally updated)."""
        return self._last_ema

    def get_rsi(self) -> float:
        """Get current RSI value."""
        if self._count < self.rsi_period + 1:
            return 50.0  # Neutral

        prices = self.get_prices(self.rsi_period + 1)
        rsi_values = fast_rsi(prices, self.rsi_period)
        self._last_rsi = rsi_values[-1]
        return self._last_rsi

    def get_signal(self) -> int:
        """
        Get trading signal based on indicators.

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        if self._count < self.sma_period:
            return 0

        current_price = self._prices[(self._index - 1) % self.buffer_size]
        sma = self.get_sma()
        ema = self.get_ema()

        # Simple signal logic
        if current_price > sma and current_price > ema:
            return 1  # Buy
        elif current_price < sma and current_price < ema:
            return -1  # Sell
        else:
            return 0  # Hold


# ============================================================
# CONFIGURATION AND GLOBALS
# ============================================================

@dataclass
class HotPathConfig:
    """Configuration for hot path optimization."""
    # Profiling
    enable_profiling: bool = True
    profile_sample_rate: float = 0.1  # Profile 10% of calls

    # Buffer sizes
    tick_buffer_size: int = 1000
    indicator_cache_size: int = 100

    # Numba settings
    use_numba: bool = NUMBA_AVAILABLE
    numba_parallel: bool = True

    # Thresholds
    target_tick_latency_ms: float = 1.0
    target_signal_latency_ms: float = 5.0
    target_order_latency_ms: float = 10.0


# Global profiler instance
_profiler: Optional[HotPathProfiler] = None


def get_profiler() -> HotPathProfiler:
    """Get global hot path profiler."""
    global _profiler
    if _profiler is None:
        _profiler = HotPathProfiler()
    return _profiler


def set_profiler(profiler: HotPathProfiler) -> None:
    """Set global hot path profiler."""
    global _profiler
    _profiler = profiler


def profile(name: str):
    """Decorator to profile a function using global profiler."""
    return get_profiler().track(name)


def is_numba_available() -> bool:
    """Check if Numba JIT is available."""
    return NUMBA_AVAILABLE


def benchmark_fast_functions(n_prices: int = 10000) -> Dict[str, float]:
    """
    Benchmark fast functions vs standard implementations.

    Returns dict of function names to speedup factors.
    """
    import time

    prices = np.random.randn(n_prices).cumsum() + 100
    high = prices + np.abs(np.random.randn(n_prices))
    low = prices - np.abs(np.random.randn(n_prices))
    close = prices

    results = {}

    # Benchmark SMA
    start = time.perf_counter()
    for _ in range(100):
        fast_sma(prices, 20)
    fast_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        np.convolve(prices, np.ones(20)/20, mode='valid')
    slow_time = time.perf_counter() - start

    results['sma'] = slow_time / fast_time if fast_time > 0 else 1.0

    # Benchmark EMA
    start = time.perf_counter()
    for _ in range(100):
        fast_ema(prices, 12)
    fast_time = time.perf_counter() - start
    results['ema'] = 1.0  # No direct comparison

    # Benchmark RSI
    start = time.perf_counter()
    for _ in range(100):
        fast_rsi(prices, 14)
    fast_time = time.perf_counter() - start
    results['rsi_time_ms'] = fast_time * 10  # Per call in ms

    return results
