# -*- coding: utf-8 -*-
"""
Vectorized Backtesting Engine - Speed Demon Mode!
==================================================
Ultra-fast backtesting using NumPy vectorization and Numba JIT.

Why vectorized?
- 100-1000x faster than bar-by-bar iteration
- Perfect for parameter optimization
- Enables Monte Carlo simulations

Example:
    >>> from backtest.vectorized import VectorizedBacktester, fast_sma
    >>>
    >>> # Fast indicator calculation
    >>> sma = fast_sma(close_prices, period=20)
    >>>
    >>> # Vectorized backtest
    >>> vbt = VectorizedBacktester(data)
    >>> result = vbt.run_ma_crossover(fast=10, slow=50)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd

# Try to import numba for JIT compilation
try:
    from numba import jit, prange, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA-OPTIMIZED INDICATORS
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _sma_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Simple Moving Average."""
        n = len(close)
        result = np.empty(n, dtype=np.float64)
        result[:period-1] = np.nan

        # First SMA value
        result[period-1] = np.mean(close[:period])

        # Rolling calculation
        for i in range(period, n):
            result[i] = result[i-1] + (close[i] - close[i-period]) / period

        return result

    @jit(nopython=True, cache=True)
    def _ema_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized Exponential Moving Average."""
        n = len(close)
        result = np.empty(n, dtype=np.float64)
        result[:period-1] = np.nan

        alpha = 2.0 / (period + 1)

        # First EMA is SMA
        result[period-1] = np.mean(close[:period])

        # EMA calculation
        for i in range(period, n):
            result[i] = alpha * close[i] + (1 - alpha) * result[i-1]

        return result

    @jit(nopython=True, cache=True)
    def _rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-optimized RSI calculation."""
        n = len(close)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Calculate price changes
        deltas = np.empty(n, dtype=np.float64)
        deltas[0] = 0.0
        for i in range(1, n):
            deltas[i] = close[i] - close[i-1]

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # First average
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # Rolling RSI
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))

        return result

    @jit(nopython=True, cache=True)
    def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-optimized ATR calculation."""
        n = len(close)
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

        # First ATR is simple average
        result[period-1] = np.mean(tr[:period])

        # Smoothed ATR
        for i in range(period, n):
            result[i] = (result[i-1] * (period - 1) + tr[i]) / period

        return result

    @jit(nopython=True, cache=True)
    def _bollinger_bands_numba(
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized Bollinger Bands."""
        n = len(close)
        middle = np.empty(n, dtype=np.float64)
        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)

        middle[:period-1] = np.nan
        upper[:period-1] = np.nan
        lower[:period-1] = np.nan

        for i in range(period - 1, n):
            window = close[i-period+1:i+1]
            mean = np.mean(window)
            std = np.std(window)

            middle[i] = mean
            upper[i] = mean + std_dev * std
            lower[i] = mean - std_dev * std

        return middle, upper, lower

    @jit(nopython=True, cache=True)
    def _macd_numba(
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized MACD calculation."""
        ema_fast = _ema_numba(close, fast)
        ema_slow = _ema_numba(close, slow)

        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        n = len(close)
        signal_line = np.empty(n, dtype=np.float64)
        signal_line[:slow+signal-2] = np.nan

        alpha = 2.0 / (signal + 1)

        # Find first valid MACD value
        start_idx = slow - 1
        signal_line[start_idx + signal - 1] = np.nanmean(macd_line[start_idx:start_idx+signal])

        for i in range(start_idx + signal, n):
            signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i-1]

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @jit(nopython=True, parallel=True, cache=True)
    def _crossover_signals_numba(fast_ma: np.ndarray, slow_ma: np.ndarray) -> np.ndarray:
        """
        Numba-optimized crossover signal detection.
        Returns: 1 for buy (golden cross), -1 for sell (death cross), 0 for hold
        """
        n = len(fast_ma)
        signals = np.zeros(n, dtype=np.int64)

        for i in prange(1, n):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                continue
            if np.isnan(fast_ma[i-1]) or np.isnan(slow_ma[i-1]):
                continue

            # Golden cross (fast crosses above slow)
            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                signals[i] = 1
            # Death cross (fast crosses below slow)
            elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                signals[i] = -1

        return signals

else:
    # Fallback implementations without Numba
    def _sma_numba(close: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(close).rolling(period).mean().values

    def _ema_numba(close: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(close).ewm(span=period, adjust=False).mean().values

    def _rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return (100 - 100 / (1 + rs)).values

    def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().values

    def _bollinger_bands_numba(close: np.ndarray, period: int = 20, std_dev: float = 2.0):
        s = pd.Series(close)
        middle = s.rolling(period).mean().values
        std = s.rolling(period).std().values
        return middle, middle + std_dev * std, middle - std_dev * std

    def _macd_numba(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        s = pd.Series(close)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.values, signal_line.values, (macd - signal_line).values

    def _crossover_signals_numba(fast_ma: np.ndarray, slow_ma: np.ndarray) -> np.ndarray:
        signals = np.zeros(len(fast_ma), dtype=np.int64)
        for i in range(1, len(fast_ma)):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                continue
            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                signals[i] = 1
            elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                signals[i] = -1
        return signals


# =============================================================================
# PUBLIC INDICATOR FUNCTIONS
# =============================================================================

def fast_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Fast SMA calculation using Numba."""
    return _sma_numba(np.asarray(close, dtype=np.float64), period)


def fast_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Fast EMA calculation using Numba."""
    return _ema_numba(np.asarray(close, dtype=np.float64), period)


def fast_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Fast RSI calculation using Numba."""
    return _rsi_numba(np.asarray(close, dtype=np.float64), period)


def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Fast ATR calculation using Numba."""
    return _atr_numba(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        period
    )


def fast_bollinger(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast Bollinger Bands calculation using Numba."""
    return _bollinger_bands_numba(np.asarray(close, dtype=np.float64), period, std_dev)


def fast_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast MACD calculation using Numba."""
    return _macd_numba(np.asarray(close, dtype=np.float64), fast, slow, signal)


# =============================================================================
# VECTORIZED BACKTEST RESULTS
# =============================================================================

@dataclass
class VectorizedResult:
    """Results from vectorized backtesting."""
    strategy_name: str
    symbol: str
    params: Dict[str, Any]

    # Performance
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Trade stats
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float

    # Equity curve
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Timing
    execution_time_ms: float = 0.0


# =============================================================================
# VECTORIZED BACKTESTER
# =============================================================================

class VectorizedBacktester:
    """
    Ultra-Fast Vectorized Backtester.

    Uses NumPy operations and Numba JIT for maximum speed.
    Perfect for parameter optimization and Monte Carlo simulations.

    Example:
        >>> vbt = VectorizedBacktester(data)
        >>> result = vbt.run_ma_crossover(fast=10, slow=50)
        >>> print(f"Return: {result.total_return:.2%}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005    # 0.05% slippage
    ):
        """
        Initialize vectorized backtester.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            initial_capital: Starting capital
            commission_pct: Commission as percentage (0.001 = 0.1%)
            slippage_pct: Slippage as percentage
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Extract arrays for vectorized operations
        self.open = data['open'].values.astype(np.float64)
        self.high = data['high'].values.astype(np.float64)
        self.low = data['low'].values.astype(np.float64)
        self.close = data['close'].values.astype(np.float64)
        self.volume = data['volume'].values.astype(np.float64) if 'volume' in data else np.ones(len(data))

        if isinstance(data.index, pd.DatetimeIndex):
            self.dates = data.index
        else:
            self.dates = pd.to_datetime(data.index)

        self.n = len(self.close)

        logger.info(f"VectorizedBacktester initialized with {self.n} bars. "
                   f"Numba available: {NUMBA_AVAILABLE}")

    def _calculate_metrics(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        strategy_name: str,
        params: Dict[str, Any]
    ) -> VectorizedResult:
        """Calculate performance metrics from signals."""
        start_time = time.time()

        # Calculate returns
        position = np.zeros(self.n, dtype=np.float64)
        current_pos = 0

        for i in range(self.n):
            if signals[i] == 1:  # Buy
                current_pos = 1
            elif signals[i] == -1:  # Sell
                current_pos = 0
            position[i] = current_pos

        # Shift position to avoid look-ahead bias
        position = np.roll(position, 1)
        position[0] = 0

        # Calculate strategy returns
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.insert(price_returns, 0, 0)

        strategy_returns = position * price_returns

        # Apply transaction costs
        trades = np.diff(position)
        trades = np.insert(trades, 0, 0)
        trade_costs = np.abs(trades) * (self.commission_pct + self.slippage_pct)
        strategy_returns -= trade_costs

        # Equity curve
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)

        # Calculate metrics
        total_return = equity[-1] / self.initial_capital - 1

        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # Trade statistics
        trade_indices = np.where(trades != 0)[0]
        total_trades = len(trade_indices) // 2  # Entry + exit = 1 trade

        # Win rate and profit factor
        trade_returns = []
        i = 0
        while i < len(trade_indices) - 1:
            if trades[trade_indices[i]] > 0:  # Entry
                entry_idx = trade_indices[i]
                # Find exit
                for j in range(i + 1, len(trade_indices)):
                    if trades[trade_indices[j]] < 0:  # Exit
                        exit_idx = trade_indices[j]
                        trade_return = (prices[exit_idx] / prices[entry_idx]) - 1
                        trade_return -= 2 * (self.commission_pct + self.slippage_pct)
                        trade_returns.append(trade_return)
                        i = j
                        break
            i += 1

        trade_returns = np.array(trade_returns) if trade_returns else np.array([0.0])

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        avg_trade = np.mean(trade_returns) if len(trade_returns) > 0 else 0

        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        execution_time = (time.time() - start_time) * 1000

        return VectorizedResult(
            strategy_name=strategy_name,
            symbol="",
            params=params,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity,
            returns=strategy_returns,
            execution_time_ms=execution_time
        )

    def run_ma_crossover(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        use_ema: bool = False
    ) -> VectorizedResult:
        """
        Run MA crossover strategy.

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            use_ema: Use EMA instead of SMA

        Returns:
            VectorizedResult with performance metrics
        """
        if use_ema:
            fast_ma = fast_ema(self.close, fast_period)
            slow_ma = fast_ema(self.close, slow_period)
        else:
            fast_ma = fast_sma(self.close, fast_period)
            slow_ma = fast_sma(self.close, slow_period)

        signals = _crossover_signals_numba(fast_ma, slow_ma)

        result = self._calculate_metrics(
            signals, self.close,
            strategy_name="MA_Crossover",
            params={'fast': fast_period, 'slow': slow_period, 'use_ema': use_ema}
        )

        return result

    def run_rsi_strategy(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ) -> VectorizedResult:
        """
        Run RSI mean-reversion strategy.

        Buy when RSI < oversold, Sell when RSI > overbought.
        """
        rsi = fast_rsi(self.close, period)

        signals = np.zeros(self.n, dtype=np.int64)

        for i in range(1, self.n):
            if np.isnan(rsi[i]):
                continue
            # Buy signal
            if rsi[i] < oversold and rsi[i-1] >= oversold:
                signals[i] = 1
            # Sell signal
            elif rsi[i] > overbought and rsi[i-1] <= overbought:
                signals[i] = -1

        return self._calculate_metrics(
            signals, self.close,
            strategy_name="RSI",
            params={'period': period, 'oversold': oversold, 'overbought': overbought}
        )

    def run_bollinger_strategy(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> VectorizedResult:
        """
        Run Bollinger Bands mean-reversion strategy.

        Buy when price touches lower band, Sell when price touches upper band.
        """
        middle, upper, lower = fast_bollinger(self.close, period, std_dev)

        signals = np.zeros(self.n, dtype=np.int64)

        for i in range(1, self.n):
            if np.isnan(lower[i]):
                continue
            # Buy when price touches lower band
            if self.close[i] <= lower[i] and self.close[i-1] > lower[i-1]:
                signals[i] = 1
            # Sell when price touches upper band
            elif self.close[i] >= upper[i] and self.close[i-1] < upper[i-1]:
                signals[i] = -1

        return self._calculate_metrics(
            signals, self.close,
            strategy_name="Bollinger",
            params={'period': period, 'std_dev': std_dev}
        )

    def run_macd_strategy(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> VectorizedResult:
        """
        Run MACD crossover strategy.

        Buy when MACD crosses above signal, Sell when MACD crosses below.
        """
        macd_line, signal_line, histogram = fast_macd(self.close, fast, slow, signal)

        signals = _crossover_signals_numba(macd_line, signal_line)

        return self._calculate_metrics(
            signals, self.close,
            strategy_name="MACD",
            params={'fast': fast, 'slow': slow, 'signal': signal}
        )

    def run_custom_signals(
        self,
        signals: np.ndarray,
        strategy_name: str = "Custom",
        params: Optional[Dict[str, Any]] = None
    ) -> VectorizedResult:
        """
        Run backtest with custom signals.

        Args:
            signals: Array of signals (1=buy, -1=sell, 0=hold)
            strategy_name: Name for the strategy
            params: Strategy parameters

        Returns:
            VectorizedResult
        """
        return self._calculate_metrics(
            signals, self.close,
            strategy_name=strategy_name,
            params=params or {}
        )

    def optimize_ma_crossover(
        self,
        fast_range: range = range(5, 30, 5),
        slow_range: range = range(30, 200, 10),
        use_ema: bool = False
    ) -> Tuple[Dict[str, Any], VectorizedResult]:
        """
        Optimize MA crossover parameters.

        Returns best parameters and result.
        """
        best_result = None
        best_params = None
        best_sharpe = float('-inf')

        for fast in fast_range:
            for slow in slow_range:
                if fast >= slow:
                    continue

                result = self.run_ma_crossover(fast, slow, use_ema)

                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_result = result
                    best_params = {'fast': fast, 'slow': slow, 'use_ema': use_ema}

        return best_params, best_result


# =============================================================================
# BENCHMARKING
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark comparison."""
    name: str
    execution_time_ms: float
    iterations: int
    avg_time_ms: float
    data_points: int
    speedup: float = 1.0  # Compared to baseline


class Benchmark:
    """
    Performance benchmarking for vectorized vs standard calculations.

    Example:
        >>> benchmark = Benchmark(data)
        >>> results = benchmark.run_all()
        >>> benchmark.print_report(results)
    """

    def __init__(self, data: pd.DataFrame, iterations: int = 100):
        self.data = data
        self.close = data['close'].values.astype(np.float64)
        self.high = data['high'].values.astype(np.float64)
        self.low = data['low'].values.astype(np.float64)
        self.iterations = iterations
        self.n = len(self.close)

    def _time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a function over multiple iterations."""
        # Warm-up
        func(*args, **kwargs)

        start = time.time()
        for _ in range(self.iterations):
            func(*args, **kwargs)
        end = time.time()

        return (end - start) * 1000  # ms

    def benchmark_sma(self, period: int = 20) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark SMA calculation."""
        # Pandas (baseline)
        def pandas_sma():
            return pd.Series(self.close).rolling(period).mean().values

        # Vectorized (Numba)
        def numba_sma():
            return fast_sma(self.close, period)

        pandas_time = self._time_function(pandas_sma)
        numba_time = self._time_function(numba_sma)

        pandas_result = BenchmarkResult(
            name="SMA (Pandas)",
            execution_time_ms=pandas_time,
            iterations=self.iterations,
            avg_time_ms=pandas_time / self.iterations,
            data_points=self.n,
            speedup=1.0
        )

        numba_result = BenchmarkResult(
            name="SMA (Numba)",
            execution_time_ms=numba_time,
            iterations=self.iterations,
            avg_time_ms=numba_time / self.iterations,
            data_points=self.n,
            speedup=pandas_time / numba_time if numba_time > 0 else 0
        )

        return pandas_result, numba_result

    def benchmark_rsi(self, period: int = 14) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark RSI calculation."""
        # Pandas
        def pandas_rsi():
            delta = pd.Series(self.close).diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            return (100 - 100 / (1 + rs)).values

        # Numba
        def numba_rsi():
            return fast_rsi(self.close, period)

        pandas_time = self._time_function(pandas_rsi)
        numba_time = self._time_function(numba_rsi)

        pandas_result = BenchmarkResult(
            name="RSI (Pandas)",
            execution_time_ms=pandas_time,
            iterations=self.iterations,
            avg_time_ms=pandas_time / self.iterations,
            data_points=self.n,
            speedup=1.0
        )

        numba_result = BenchmarkResult(
            name="RSI (Numba)",
            execution_time_ms=numba_time,
            iterations=self.iterations,
            avg_time_ms=numba_time / self.iterations,
            data_points=self.n,
            speedup=pandas_time / numba_time if numba_time > 0 else 0
        )

        return pandas_result, numba_result

    def benchmark_backtest(self) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark full backtest."""
        from .engine import Backtester
        from strategies import get_strategy

        # Standard backtester
        def standard_backtest():
            bt = Backtester(initial_capital=100000)
            strategy = get_strategy('sma_crossover')
            return bt.run(self.data, strategy, "TEST")

        # Vectorized backtester
        def vectorized_backtest():
            vbt = VectorizedBacktester(self.data)
            return vbt.run_ma_crossover(fast=10, slow=50)

        # Fewer iterations for full backtest
        iterations = min(10, self.iterations)

        start = time.time()
        for _ in range(iterations):
            try:
                standard_backtest()
            except Exception:
                pass
        standard_time = (time.time() - start) * 1000

        start = time.time()
        for _ in range(iterations):
            vectorized_backtest()
        vectorized_time = (time.time() - start) * 1000

        standard_result = BenchmarkResult(
            name="Backtest (Standard)",
            execution_time_ms=standard_time,
            iterations=iterations,
            avg_time_ms=standard_time / iterations,
            data_points=self.n,
            speedup=1.0
        )

        vectorized_result = BenchmarkResult(
            name="Backtest (Vectorized)",
            execution_time_ms=vectorized_time,
            iterations=iterations,
            avg_time_ms=vectorized_time / iterations,
            data_points=self.n,
            speedup=standard_time / vectorized_time if vectorized_time > 0 else 0
        )

        return standard_result, vectorized_result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []

        print("Running SMA benchmark...")
        pandas_sma, numba_sma = self.benchmark_sma()
        results.extend([pandas_sma, numba_sma])

        print("Running RSI benchmark...")
        pandas_rsi, numba_rsi = self.benchmark_rsi()
        results.extend([pandas_rsi, numba_rsi])

        print("Running backtest benchmark...")
        standard_bt, vectorized_bt = self.benchmark_backtest()
        results.extend([standard_bt, vectorized_bt])

        return results

    @staticmethod
    def print_report(results: List[BenchmarkResult]) -> None:
        """Print benchmark report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Name':<25} {'Avg Time (ms)':<15} {'Speedup':<10} {'Data Points':<12}")
        print("-" * 70)

        for r in results:
            speedup_str = f"{r.speedup:.1f}x" if r.speedup > 1 else "baseline"
            print(f"{r.name:<25} {r.avg_time_ms:<15.3f} {speedup_str:<10} {r.data_points:<12}")

        print("=" * 70)
        print(f"Numba available: {NUMBA_AVAILABLE}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_vectorized_backtest(
    data: pd.DataFrame,
    strategy: str = "ma_crossover",
    **kwargs
) -> VectorizedResult:
    """
    Quick vectorized backtest.

    Args:
        data: OHLCV data
        strategy: Strategy name (ma_crossover, rsi, bollinger, macd)
        **kwargs: Strategy parameters

    Returns:
        VectorizedResult
    """
    vbt = VectorizedBacktester(data)

    if strategy == "ma_crossover":
        return vbt.run_ma_crossover(**kwargs)
    elif strategy == "rsi":
        return vbt.run_rsi_strategy(**kwargs)
    elif strategy == "bollinger":
        return vbt.run_bollinger_strategy(**kwargs)
    elif strategy == "macd":
        return vbt.run_macd_strategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_benchmark(data: pd.DataFrame, iterations: int = 100) -> List[BenchmarkResult]:
    """Run performance benchmark."""
    benchmark = Benchmark(data, iterations)
    results = benchmark.run_all()
    Benchmark.print_report(results)
    return results
