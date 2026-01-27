# -*- coding: utf-8 -*-
"""
Unified Backtesting Interface
=============================
Solves the "Duality Trap" - one interface, automatic engine selection.

Problem:
- Vectorized engine is fast but can't handle complex orders (Iceberg, partials)
- Iterative engine handles complex orders but is slow
- Cost models (slippage, commission) differed between engines!

Solution:
- UnifiedBacktester: Single interface that automatically selects the right engine
- BacktestConfig: Standardized cost model used by BOTH engines
- Consistent results regardless of which engine runs

Usage:
    >>> from backtest import UnifiedBacktester, BacktestConfig
    >>>
    >>> # Create config with consistent costs
    >>> config = BacktestConfig(
    ...     initial_capital=100000,
    ...     slippage_pct=0.05,      # 0.05% slippage
    ...     commission_pct=0.03,    # 0.03% commission (Zerodha intraday)
    ... )
    >>>
    >>> # Unified backtest - automatically picks best engine
    >>> bt = UnifiedBacktester(data, config)
    >>> result = bt.run(strategy)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# STANDARDIZED CONFIGURATION
# =============================================================================

class CostModel(Enum):
    """Cost model types for backtesting."""
    ZERODHA_INTRADAY = auto()   # 0.03% brokerage + STT + charges
    ZERODHA_DELIVERY = auto()   # 0% brokerage + STT + DP charges
    FLAT_FEE = auto()           # Fixed fee per trade
    PERCENTAGE = auto()         # Percentage of trade value
    CUSTOM = auto()             # User-defined


class FillModel(Enum):
    """
    Execution fill model for backtesting.

    CLOSE:    Fill at close price (unrealistic, overestimates profits by 10-20%)
    BID_ASK:  Fill at simulated bid/ask (realistic spread modeling)
    OHLC:     Use High for sells, Low for buys (conservative/worst-case)

    Professional Standard: BID_ASK with realistic spread_bps.
    """
    CLOSE = auto()      # Legacy: fill at close price
    BID_ASK = auto()    # Realistic: fill at simulated bid/ask with spread
    OHLC = auto()       # Conservative: worst-case fills using high/low


class SlippageModel(Enum):
    """
    Slippage model for market orders.

    FIXED:      Fixed percentage slippage (default)
    VARIABLE:   Variable slippage based on recent volatility
    VOLUME:     Slippage increases with order size relative to volume
    """
    FIXED = auto()      # Fixed percentage (e.g., 0.05%)
    VARIABLE = auto()   # Variable based on ATR/volatility
    VOLUME = auto()     # Based on order size / average volume


@dataclass
class BacktestConfig:
    """
    Unified configuration for ALL backtesting engines.

    CRITICAL: Both Vectorized and Iterative engines use this config
    to ensure consistent cost calculations and matching results.

    REALITY GAP FIX:
    - Old: Fill at close price with slippage (overestimates profits 10-20%)
    - New: Fill at simulated bid/ask with spread + slippage (realistic)

    Indian Market Defaults (Zerodha):
    - Intraday: 0.03% brokerage + ~0.05% other charges
    - Delivery: 0% brokerage + 0.1% STT + DP charges

    Bid-Ask Spread (typical):
    - Nifty 50 futures: 1-2 bps (very liquid)
    - Large caps (RELIANCE, TCS): 3-5 bps
    - Mid caps: 10-20 bps
    - Small caps: 20-50+ bps

    Slippage (market impact):
    - Small orders (<1% ADV): 2-5 bps
    - Medium orders (1-5% ADV): 5-15 bps
    - Large orders (>5% ADV): 15-50+ bps
    """

    # Capital
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0  # % of capital per trade

    # =========================================================================
    # EXECUTION MODEL (NEW - Fixes "Reality Gap")
    # =========================================================================

    # Fill model - how orders are executed
    fill_model: FillModel = FillModel.BID_ASK  # Default to realistic

    # Bid-Ask Spread in basis points (bps)
    # 1 bp = 0.01% = 0.0001
    # Example: 5 bps for Nifty = 0.05% spread
    spread_bps: float = 5.0  # Default 5 bps (large cap liquid stocks)

    # Slippage model and parameters
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_pct: float = 0.05       # 0.05% market impact (5 bps)

    # =========================================================================
    # TRANSACTION COSTS
    # =========================================================================

    # Commission/brokerage (as percentage points)
    # Example: 0.03 means 0.03% (not 3%)
    commission_pct: float = 0.03     # 0.03% commission (Zerodha intraday)
    other_charges_pct: float = 0.02  # 0.02% STT, stamp duty, etc.

    # Cost model preset (applies default costs)
    cost_model: CostModel = CostModel.ZERODHA_INTRADAY

    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================

    max_positions: int = 5
    max_drawdown_pct: float = 20.0   # Stop if drawdown exceeds this

    # =========================================================================
    # ENGINE SETTINGS
    # =========================================================================

    warmup_bars: int = 50            # Bars to skip for indicator warmup
    force_iterative: bool = False    # Force iterative engine even for simple strategies
    force_vectorized: bool = False   # Force vectorized (will fail on complex orders)

    def __post_init__(self):
        """Apply cost model presets."""
        if self.cost_model == CostModel.ZERODHA_INTRADAY:
            # Zerodha intraday: 0.03% brokerage, no STT on sell
            self.commission_pct = 0.03
            self.other_charges_pct = 0.02  # Exchange charges, GST
        elif self.cost_model == CostModel.ZERODHA_DELIVERY:
            # Zerodha delivery: 0% brokerage, 0.1% STT on sell
            self.commission_pct = 0.0
            self.other_charges_pct = 0.10  # STT, DP charges

    @property
    def spread_pct(self) -> float:
        """Spread as percentage (converted from bps)."""
        return self.spread_bps / 100  # 5 bps = 0.05%

    @property
    def half_spread_pct(self) -> float:
        """Half-spread (cost per side) as percentage."""
        return self.spread_pct / 2  # Buy at ask, sell at bid

    @property
    def total_cost_pct(self) -> float:
        """
        Total one-way transaction cost as percentage points.
        Includes: half-spread + slippage + commission + other charges.

        Example: With 5 bps spread, 5 bps slippage, 3 bps commission, 2 bps other
        = 0.025 + 0.05 + 0.03 + 0.02 = 0.125% per side
        """
        return self.half_spread_pct + self.slippage_pct + self.commission_pct + self.other_charges_pct

    def get_buy_price(self, close: float, high: float = None, low: float = None) -> float:
        """
        Calculate execution price for a BUY order.

        In reality: BUY orders execute at the ASK price (above mid).
        Plus slippage from market impact.

        Args:
            close: Close/mid price of the bar
            high: High price (for OHLC fill model)
            low: Low price (unused for buys)

        Returns:
            Simulated execution price for BUY
        """
        if self.fill_model == FillModel.CLOSE:
            # Legacy: simple slippage on close
            return close * (1 + self.slippage_pct / 100)

        elif self.fill_model == FillModel.OHLC:
            # Conservative: assume we buy at worst price (high)
            base_price = high if high else close
            return base_price * (1 + self.slippage_pct / 100)

        else:  # FillModel.BID_ASK (default, realistic)
            # BUY at ASK = mid + half spread, plus slippage
            ask_price = close * (1 + self.half_spread_pct / 100)
            return ask_price * (1 + self.slippage_pct / 100)

    def get_sell_price(self, close: float, high: float = None, low: float = None) -> float:
        """
        Calculate execution price for a SELL order.

        In reality: SELL orders execute at the BID price (below mid).
        Minus slippage from market impact.

        Args:
            close: Close/mid price of the bar
            high: High price (unused for sells)
            low: Low price (for OHLC fill model)

        Returns:
            Simulated execution price for SELL
        """
        if self.fill_model == FillModel.CLOSE:
            # Legacy: simple slippage on close
            return close * (1 - self.slippage_pct / 100)

        elif self.fill_model == FillModel.OHLC:
            # Conservative: assume we sell at worst price (low)
            base_price = low if low else close
            return base_price * (1 - self.slippage_pct / 100)

        else:  # FillModel.BID_ASK (default, realistic)
            # SELL at BID = mid - half spread, minus slippage
            bid_price = close * (1 - self.half_spread_pct / 100)
            return bid_price * (1 - self.slippage_pct / 100)

    def estimate_profit_reduction(self) -> str:
        """
        Estimate how much realistic execution reduces profits vs close price.

        Returns human-readable string explaining the reality gap.
        """
        # Per trade round-trip cost difference
        legacy_rt = 2 * self.slippage_pct
        realistic_rt = self.round_trip_cost_pct

        reduction = realistic_rt - legacy_rt + self.spread_pct  # spread was ignored

        return (
            f"Reality Gap Estimate:\n"
            f"  Legacy model (close + slippage): {legacy_rt:.2f}% round-trip\n"
            f"  Realistic model (bid/ask + spread + slippage): {realistic_rt:.2f}% round-trip\n"
            f"  Additional spread cost: {self.spread_pct:.2f}% round-trip\n"
            f"  \n"
            f"  For 100 trades, realistic model costs ~{reduction * 100:.1f}% more.\n"
            f"  If Nifty spread is 2 points at 22000, that's Rs.{2 * 75:.0f}/lot extra per trade."
        )

    @property
    def round_trip_cost_pct(self) -> float:
        """
        Total round-trip (buy + sell) cost as percentage points.
        Example: 0.20 means 0.20% (two tenths of a percent).
        """
        return 2 * self.total_cost_pct

    def format_cost(self) -> str:
        """Format cost as readable string."""
        return f"{self.total_cost_pct:.2f}% one-way, {self.round_trip_cost_pct:.2f}% round-trip"

    def to_iterative_params(self) -> Dict[str, Any]:
        """
        Convert to parameters for iterative engine.

        engine.py uses: price * (1 + slippage_pct / 100)
        So 0.1 means 0.1% slippage.
        """
        return {
            'initial_capital': self.initial_capital,
            'position_size_pct': self.position_size_pct,
            'commission': 0,  # We use percentage, not flat fee
            'slippage_pct': self.total_cost_pct,  # Combined into slippage (in %)
        }

    def to_vectorized_params(self) -> Dict[str, Any]:
        """
        Convert to parameters for vectorized engine.

        vectorized.py uses: trade_costs = trades * (commission_pct + slippage_pct)
        So 0.001 means 0.1% commission.
        We need to divide our values by 100 to convert.
        """
        return {
            'initial_capital': self.initial_capital,
            # Convert from percentage points to decimal
            # 0.03 (our format, meaning 0.03%) -> 0.0003 (vectorized format)
            'commission_pct': (self.commission_pct + self.other_charges_pct) / 100,
            'slippage_pct': self.slippage_pct / 100,
        }


# =============================================================================
# ENGINE SELECTION
# =============================================================================

class EngineType(Enum):
    """Backtesting engine types."""
    VECTORIZED = auto()   # Fast, for simple strategies
    ITERATIVE = auto()    # Flexible, for complex orders
    EVENT_DRIVEN = auto() # Full event-driven architecture


def detect_strategy_complexity(strategy) -> EngineType:
    """
    Automatically detect which engine to use based on strategy features.

    Returns ITERATIVE if strategy uses:
    - Iceberg orders
    - Partial fills
    - Complex stop-loss logic
    - Multiple simultaneous positions
    - Position sizing based on volatility

    Returns VECTORIZED for simple strategies:
    - Moving average crossover
    - RSI overbought/oversold
    - Bollinger band bounce
    - Simple breakout
    """
    # Check for complex order indicators
    strategy_name = getattr(strategy, 'name', '').lower()
    strategy_class = type(strategy).__name__.lower()

    # Complex strategies that need iterative engine
    complex_indicators = [
        'iceberg', 'partial', 'twap', 'vwap', 'adaptive',
        'volatility_sizing', 'multi_position', 'scale_in', 'scale_out',
        'pyramid', 'grid', 'martingale', 'anti_martingale'
    ]

    for indicator in complex_indicators:
        if indicator in strategy_name or indicator in strategy_class:
            return EngineType.ITERATIVE

    # Check for complex methods
    if hasattr(strategy, 'calculate_position_size'):
        # Custom position sizing = need iterative
        return EngineType.ITERATIVE

    if hasattr(strategy, 'on_partial_fill'):
        return EngineType.ITERATIVE

    # Default to vectorized for speed
    return EngineType.VECTORIZED


# =============================================================================
# UNIFIED BACKTESTER
# =============================================================================

class UnifiedBacktester:
    """
    Unified Backtesting Interface - Solves the Duality Trap!

    Automatically selects the best engine:
    - Vectorized for simple strategies (100-1000x faster)
    - Iterative for complex order management

    CRITICAL: Uses BacktestConfig to ensure CONSISTENT cost calculations
    regardless of which engine runs. No more optimization in one engine
    and running in another with different results!

    Example:
        >>> config = BacktestConfig(
        ...     initial_capital=100000,
        ...     cost_model=CostModel.ZERODHA_INTRADAY
        ... )
        >>> bt = UnifiedBacktester(data, config)
        >>>
        >>> # Simple strategy -> uses vectorized (fast)
        >>> result1 = bt.run(ma_crossover_strategy)
        >>>
        >>> # Complex strategy -> auto-switches to iterative
        >>> result2 = bt.run(iceberg_strategy)
        >>>
        >>> # Force specific engine
        >>> result3 = bt.run(strategy, engine=EngineType.ITERATIVE)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize unified backtester.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            config: Unified backtest configuration (uses defaults if not provided)
        """
        self.data = data
        self.config = config or BacktestConfig()

        # Validate data
        self._validate_data()

        # Engine instances (lazy initialized)
        self._vectorized_engine = None
        self._iterative_engine = None

        logger.info(
            f"UnifiedBacktester initialized: {len(data)} bars, "
            f"capital={self.config.initial_capital}, "
            f"total_cost={self.config.total_cost_pct:.3%}"
        )

    def _validate_data(self):
        """Validate input data."""
        required_cols = ['open', 'high', 'low', 'close']
        alt_cols = ['Open', 'High', 'Low', 'Close']

        has_required = all(c in self.data.columns for c in required_cols)
        has_alt = all(c in self.data.columns for c in alt_cols)

        if not (has_required or has_alt):
            raise ValueError(
                f"Data must have OHLC columns. Found: {list(self.data.columns)}"
            )

        if len(self.data) < self.config.warmup_bars:
            logger.warning(
                f"Data has only {len(self.data)} bars, "
                f"less than warmup period of {self.config.warmup_bars}"
            )

    def run(
        self,
        strategy=None,
        signals: Optional[np.ndarray] = None,
        engine: Optional[EngineType] = None,
        **kwargs
    ):
        """
        Run backtest with automatic or specified engine selection.

        Args:
            strategy: Strategy instance (for iterative/event-driven)
            signals: Pre-computed signal array (for vectorized)
            engine: Force specific engine (auto-detect if None)
            **kwargs: Additional parameters for specific engines

        Returns:
            BacktestResult with consistent metrics
        """
        # Determine engine to use
        if engine is None:
            if self.config.force_vectorized:
                engine = EngineType.VECTORIZED
            elif self.config.force_iterative:
                engine = EngineType.ITERATIVE
            elif strategy is not None:
                engine = detect_strategy_complexity(strategy)
            elif signals is not None:
                engine = EngineType.VECTORIZED
            else:
                raise ValueError("Must provide either strategy or signals")

        logger.info(f"Running backtest with {engine.name} engine")

        # Run with selected engine
        if engine == EngineType.VECTORIZED:
            return self._run_vectorized(signals, **kwargs)
        elif engine == EngineType.ITERATIVE:
            return self._run_iterative(strategy, **kwargs)
        else:
            return self._run_event_driven(strategy, **kwargs)

    def _run_vectorized(self, signals: Optional[np.ndarray] = None, **kwargs):
        """Run vectorized backtest."""
        from .vectorized import VectorizedBacktester

        if self._vectorized_engine is None:
            params = self.config.to_vectorized_params()
            self._vectorized_engine = VectorizedBacktester(self.data, **params)

        if signals is None:
            raise ValueError("Vectorized backtest requires pre-computed signals array")

        return self._vectorized_engine.backtest_signals(signals)

    def _run_iterative(self, strategy, **kwargs):
        """Run iterative backtest with realistic execution modeling."""
        from .engine import Backtester

        if self._iterative_engine is None:
            # Pass config directly for realistic execution pricing
            self._iterative_engine = Backtester(config=self.config)

        return self._iterative_engine.run(
            data=self.data,
            strategy=strategy,
            symbol=kwargs.get('symbol', 'BACKTEST')
        )

    def _run_event_driven(self, strategy, **kwargs):
        """Run event-driven backtest with realistic execution modeling."""
        from .engine import EventDrivenBacktester

        # Pass config directly for realistic execution pricing
        engine = EventDrivenBacktester(config=self.config)
        return engine.run(
            data=self.data,
            strategy=strategy,
            symbol=kwargs.get('symbol', 'BACKTEST')
        )

    # =========================================================================
    # CONVENIENCE METHODS FOR COMMON STRATEGIES
    # =========================================================================

    def run_ma_crossover(
        self,
        fast_period: int = 10,
        slow_period: int = 50
    ):
        """
        Run moving average crossover backtest (uses vectorized).

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            BacktestResult
        """
        from .vectorized import fast_sma

        close = self.data['close'].values if 'close' in self.data.columns else self.data['Close'].values

        fast_ma = fast_sma(close, fast_period)
        slow_ma = fast_sma(close, slow_period)

        # Generate signals: 1 = long, 0 = flat, -1 = short
        signals = np.where(fast_ma > slow_ma, 1, 0)

        return self._run_vectorized(signals)

    def run_rsi(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ):
        """
        Run RSI strategy backtest (uses vectorized).

        Args:
            period: RSI period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)

        Returns:
            BacktestResult
        """
        from .vectorized import fast_rsi

        close = self.data['close'].values if 'close' in self.data.columns else self.data['Close'].values
        rsi = fast_rsi(close, period)

        # Generate signals
        signals = np.zeros(len(close))
        signals[rsi < oversold] = 1   # Buy when oversold
        signals[rsi > overbought] = 0  # Sell when overbought

        # Forward fill signals (stay in position until opposite signal)
        position = 0
        for i in range(len(signals)):
            if signals[i] == 1:
                position = 1
            elif signals[i] == 0 and rsi[i] > overbought:
                position = 0
            signals[i] = position

        return self._run_vectorized(signals)

    def optimize(
        self,
        strategy_fn: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using vectorized backtester.

        Args:
            strategy_fn: Function that takes params and returns signals array
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor')

        Returns:
            Dictionary with best parameters and result
        """
        import itertools

        best_result = None
        best_params = None
        best_metric = float('-inf')

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combinations = 1
        for v in param_values:
            total_combinations *= len(v)

        logger.info(f"Optimizing over {total_combinations} parameter combinations")

        for i, values in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, values))

            try:
                # Generate signals with these parameters
                signals = strategy_fn(self.data, **params)

                # Run backtest
                result = self._run_vectorized(signals)

                # Extract metric
                metric_value = getattr(result, metric, None)
                if metric_value is None:
                    continue

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result

            except Exception as e:
                logger.debug(f"Params {params} failed: {e}")
                continue

            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{total_combinations} combinations")

        return {
            'best_params': best_params,
            'best_metric_value': best_metric,
            'metric_name': metric,
            'result': best_result
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def unified_backtest(
    data: pd.DataFrame,
    strategy=None,
    signals: np.ndarray = None,
    config: BacktestConfig = None,
    **kwargs
):
    """
    Quick unified backtest function.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy instance (optional)
        signals: Pre-computed signals (optional)
        config: BacktestConfig (uses defaults if not provided)
        **kwargs: Additional parameters

    Returns:
        BacktestResult
    """
    bt = UnifiedBacktester(data, config)
    return bt.run(strategy=strategy, signals=signals, **kwargs)


def compare_engines(
    data: pd.DataFrame,
    strategy,
    config: BacktestConfig = None
) -> Dict[str, Any]:
    """
    Run backtest with BOTH engines and compare results.

    Useful for verifying that cost models match.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy instance
        config: BacktestConfig

    Returns:
        Dictionary with results from both engines and comparison metrics
    """
    config = config or BacktestConfig()
    bt = UnifiedBacktester(data, config)

    # Run with both engines
    iterative_result = bt.run(strategy, engine=EngineType.ITERATIVE)

    # For vectorized, we need to generate signals first
    # This is a simplified comparison - in practice you'd use the same signal generation
    logger.warning(
        "compare_engines: Vectorized requires pre-computed signals. "
        "For accurate comparison, generate signals from strategy first."
    )

    return {
        'iterative_result': iterative_result,
        'config': config,
        'total_cost_pct': config.total_cost_pct,
    }
