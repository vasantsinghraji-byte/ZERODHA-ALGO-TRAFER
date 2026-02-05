# -*- coding: utf-8 -*-
"""
Walk-Forward Optimization Engine - The Ultimate Strategy Tester!
================================================================
Avoid overfitting by testing on truly out-of-sample data.

Walk-Forward Analysis splits your data into multiple windows:
- In-Sample (IS): Optimize parameters on this data
- Out-of-Sample (OOS): Test optimized parameters here

The OOS results show how the strategy would REALLY perform!

Example:
    >>> from backtest.wfo import WalkForwardOptimizer, WFOConfig
    >>>
    >>> wfo = WalkForwardOptimizer(
    ...     data=historical_data,
    ...     strategy_class=MyStrategy,
    ...     param_grid={'fast_period': [10, 20], 'slow_period': [50, 100]}
    ... )
    >>> results = wfo.run()
    >>> print(results.summary())
"""

import logging
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Type, Callable, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

from strategies.base import Strategy
from .engine import Backtester, BacktestResult, Trade

logger = logging.getLogger(__name__)


class ObjectiveFunction(Enum):
    """Optimization objectives."""
    SHARPE_RATIO = "sharpe"
    TOTAL_RETURN = "return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    CALMAR_RATIO = "calmar"
    SORTINO_RATIO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize
    CUSTOM = "custom"


@dataclass
class WFOConfig:
    """Configuration for Walk-Forward Optimization."""
    # Window settings
    in_sample_days: int = 252           # ~1 year trading days
    out_of_sample_days: int = 63        # ~3 months trading days
    step_days: int = 63                 # How much to advance each window

    # Anchor settings
    anchored: bool = False              # If True, IS window always starts from beginning

    # Optimization settings
    objective: ObjectiveFunction = ObjectiveFunction.SHARPE_RATIO
    custom_objective: Optional[Callable[[BacktestResult], float]] = None
    minimize: bool = False              # If True, minimize objective (e.g., drawdown)

    # Constraints
    min_trades: int = 10                # Minimum trades required in IS period
    min_win_rate: float = 0.0           # Minimum win rate constraint
    max_drawdown_pct: float = 100.0     # Maximum drawdown constraint

    # Execution settings
    parallel: bool = True               # Run optimizations in parallel
    max_workers: int = 4                # Max parallel workers
    verbose: bool = True                # Print progress

    # Capital
    initial_capital: float = 100000


@dataclass
class WindowPeriod:
    """A single walk-forward window."""
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime

    @property
    def is_days(self) -> int:
        return (self.is_end - self.is_start).days

    @property
    def oos_days(self) -> int:
        return (self.oos_end - self.oos_start).days


@dataclass
class OptimizationResult:
    """Result of optimizing a single window."""
    window: WindowPeriod
    best_params: Dict[str, Any]
    is_result: BacktestResult
    oos_result: BacktestResult
    objective_value: float
    all_param_results: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)


@dataclass
class WFOResult:
    """Complete Walk-Forward Optimization results."""
    strategy_name: str
    symbol: str
    config: WFOConfig
    param_grid: Dict[str, List[Any]]

    # Window results
    windows: List[OptimizationResult] = field(default_factory=list)

    # Aggregated OOS results
    oos_trades: List[Trade] = field(default_factory=list)
    oos_equity_curve: List[float] = field(default_factory=list)
    oos_dates: List[datetime] = field(default_factory=list)

    # Aggregated metrics
    total_oos_return: float = 0.0
    avg_oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_max_drawdown: float = 0.0

    # Robustness metrics
    efficiency_ratio: float = 0.0       # OOS Sharpe / IS Sharpe
    consistency_score: float = 0.0      # % of profitable OOS windows
    parameter_stability: float = 0.0    # How stable are optimal params

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "WALK-FORWARD OPTIMIZATION RESULTS",
            "=" * 60,
            f"Strategy: {self.strategy_name}",
            f"Symbol: {self.symbol}",
            f"Windows: {len(self.windows)}",
            "",
            "OUT-OF-SAMPLE PERFORMANCE (What really matters!)",
            "-" * 40,
            f"Total Return: {self.total_oos_return:.2%}",
            f"Avg Sharpe: {self.avg_oos_sharpe:.2f}",
            f"Win Rate: {self.oos_win_rate:.1%}",
            f"Profit Factor: {self.oos_profit_factor:.2f}",
            f"Max Drawdown: {self.oos_max_drawdown:.2%}",
            "",
            "ROBUSTNESS METRICS",
            "-" * 40,
            f"Efficiency Ratio: {self.efficiency_ratio:.2f} (OOS/IS performance)",
            f"Consistency: {self.consistency_score:.1%} profitable windows",
            f"Parameter Stability: {self.parameter_stability:.1%}",
            "",
            "WINDOW DETAILS",
            "-" * 40,
        ]

        for i, window in enumerate(self.windows):
            lines.append(
                f"Window {i+1}: IS {window.is_result.return_pct:+.1%} -> "
                f"OOS {window.oos_result.return_pct:+.1%} | "
                f"Params: {window.best_params}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.

    The gold standard for strategy validation! Tests your strategy
    on truly out-of-sample data to avoid curve-fitting.

    How it works:
    1. Split data into overlapping windows
    2. For each window:
       - Optimize parameters on in-sample data
       - Test best parameters on out-of-sample data
    3. Aggregate all OOS results for realistic performance

    Example:
        >>> wfo = WalkForwardOptimizer(
        ...     data=df,
        ...     strategy_class=MACrossover,
        ...     param_grid={
        ...         'fast_period': [5, 10, 15, 20],
        ...         'slow_period': [30, 50, 100]
        ...     }
        ... )
        >>> result = wfo.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        config: Optional[WFOConfig] = None,
        symbol: str = "UNKNOWN"
    ):
        """
        Initialize Walk-Forward Optimizer.

        Args:
            data: Historical OHLCV data with datetime index
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameter names to lists of values
            config: WFO configuration
            symbol: Trading symbol
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.config = config or WFOConfig()
        self.symbol = symbol

        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                self.data = data.set_index('date')
            else:
                raise ValueError("Data must have datetime index or 'date' column")

        self.data = self.data.sort_index()

        # Generate windows
        self.windows = self._generate_windows()

        logger.info(
            f"WFO initialized: {len(self.windows)} windows, "
            f"{self._count_param_combinations()} parameter combinations"
        )

    def _generate_windows(self) -> List[WindowPeriod]:
        """Generate walk-forward windows."""
        windows = []

        start_date = self.data.index.min()
        end_date = self.data.index.max()

        # Calculate window boundaries
        is_delta = timedelta(days=self.config.in_sample_days)
        oos_delta = timedelta(days=self.config.out_of_sample_days)
        step_delta = timedelta(days=self.config.step_days)

        window_id = 0
        current_start = start_date

        while True:
            if self.config.anchored:
                is_start = start_date  # Always start from beginning
            else:
                is_start = current_start

            is_end = is_start + is_delta
            oos_start = is_end
            oos_end = oos_start + oos_delta

            # Check if we've gone past the data
            if oos_end > end_date:
                # Try to fit one more window with available data
                if oos_start < end_date:
                    oos_end = end_date
                    windows.append(WindowPeriod(
                        window_id=window_id,
                        is_start=is_start,
                        is_end=is_end,
                        oos_start=oos_start,
                        oos_end=oos_end
                    ))
                break

            windows.append(WindowPeriod(
                window_id=window_id,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end
            ))

            window_id += 1
            current_start += step_delta

        return windows

    def _count_param_combinations(self) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in self.param_grid.values():
            count *= len(values)
        return count

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _get_objective_value(self, result: BacktestResult) -> float:
        """Calculate objective function value from backtest result."""
        if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
            return result.sharpe_ratio
        elif self.config.objective == ObjectiveFunction.TOTAL_RETURN:
            return result.return_pct
        elif self.config.objective == ObjectiveFunction.PROFIT_FACTOR:
            return result.profit_factor
        elif self.config.objective == ObjectiveFunction.WIN_RATE:
            return result.win_rate
        elif self.config.objective == ObjectiveFunction.CALMAR_RATIO:
            if result.max_drawdown_pct > 0:
                return result.return_pct / result.max_drawdown_pct
            return 0.0
        elif self.config.objective == ObjectiveFunction.SORTINO_RATIO:
            return self._calculate_sortino(result)
        elif self.config.objective == ObjectiveFunction.MAX_DRAWDOWN:
            return -result.max_drawdown_pct  # Negative because we want to minimize
        elif self.config.objective == ObjectiveFunction.CUSTOM:
            if self.config.custom_objective:
                return self.config.custom_objective(result)
            return 0.0
        else:
            return result.sharpe_ratio

    def _calculate_sortino(self, result: BacktestResult) -> float:
        """Calculate Sortino ratio from equity curve.

        MATH FIX: Calculate downside deviation over the ENTIRE period,
        treating positive returns as 0. This maintains the correct sample size
        and produces industry-standard Sortino ratios.
        """
        if len(result.equity_curve) < 2:
            return 0.0

        returns = pd.Series(result.equity_curve).pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # Correct downside deviation: zero out positive returns, keep full sample size
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0

        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return 0.0

        return (returns.mean() * 252) / downside_std

    def _passes_constraints(self, result: BacktestResult) -> bool:
        """Check if result passes all constraints."""
        if result.total_trades < self.config.min_trades:
            return False
        if result.win_rate < self.config.min_win_rate:
            return False
        if result.max_drawdown_pct > self.config.max_drawdown_pct:
            return False
        return True

    def _run_backtest(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> BacktestResult:
        """Run a single backtest with given parameters."""
        # Create strategy with parameters
        strategy = self.strategy_class(**params)

        # Run backtest
        backtester = Backtester(
            initial_capital=self.config.initial_capital,
            position_size_pct=10.0
        )

        result = backtester.run(
            data=data,
            strategy=strategy,
            symbol=self.symbol
        )

        return result

    def _optimize_window(self, window: WindowPeriod) -> OptimizationResult:
        """Optimize parameters for a single window."""
        # Get in-sample data
        is_data = self.data[window.is_start:window.is_end]
        oos_data = self.data[window.oos_start:window.oos_end]

        if len(is_data) < 20 or len(oos_data) < 5:
            raise ValueError(f"Insufficient data for window {window.window_id}")

        # Get all parameter combinations
        param_combinations = self._get_param_combinations()

        # Run backtests for all combinations
        all_results: List[Tuple[Dict[str, Any], float, BacktestResult]] = []

        for params in param_combinations:
            try:
                result = self._run_backtest(is_data, params)

                if self._passes_constraints(result):
                    obj_value = self._get_objective_value(result)
                    all_results.append((params, obj_value, result))

            except Exception as e:
                logger.debug(f"Backtest failed for {params}: {e}")
                continue

        if not all_results:
            # Use first param combination as fallback
            params = param_combinations[0]
            result = self._run_backtest(is_data, params)
            obj_value = self._get_objective_value(result)
            all_results.append((params, obj_value, result))

        # Find best parameters
        if self.config.minimize:
            best = min(all_results, key=lambda x: x[1])
        else:
            best = max(all_results, key=lambda x: x[1])

        best_params, best_obj, is_result = best

        # Run OOS backtest with best parameters
        oos_result = self._run_backtest(oos_data, best_params)

        return OptimizationResult(
            window=window,
            best_params=best_params,
            is_result=is_result,
            oos_result=oos_result,
            objective_value=best_obj,
            all_param_results=[(p, v) for p, v, _ in all_results]
        )

    def run(self) -> WFOResult:
        """
        Run complete Walk-Forward Optimization.

        Returns:
            WFOResult with all window results and aggregated metrics
        """
        logger.info(f"Starting WFO with {len(self.windows)} windows...")

        window_results: List[OptimizationResult] = []

        if self.config.parallel and len(self.windows) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._optimize_window, w): w
                    for w in self.windows
                }

                for future in as_completed(futures):
                    window = futures[future]
                    try:
                        result = future.result()
                        window_results.append(result)

                        if self.config.verbose:
                            print(f"Window {window.window_id + 1}/{len(self.windows)}: "
                                  f"IS {result.is_result.return_pct:+.1%} -> "
                                  f"OOS {result.oos_result.return_pct:+.1%}")

                    except Exception as e:
                        logger.error(f"Window {window.window_id} failed: {e}")
        else:
            # Sequential execution
            for i, window in enumerate(self.windows):
                try:
                    result = self._optimize_window(window)
                    window_results.append(result)

                    if self.config.verbose:
                        print(f"Window {i + 1}/{len(self.windows)}: "
                              f"IS {result.is_result.return_pct:+.1%} -> "
                              f"OOS {result.oos_result.return_pct:+.1%}")

                except Exception as e:
                    logger.error(f"Window {i} failed: {e}")

        # Sort by window ID
        window_results.sort(key=lambda x: x.window.window_id)

        # Aggregate results
        result = self._aggregate_results(window_results)

        logger.info("WFO complete!")
        return result

    def _aggregate_results(self, window_results: List[OptimizationResult]) -> WFOResult:
        """Aggregate results from all windows."""
        result = WFOResult(
            strategy_name=self.strategy_class.__name__,
            symbol=self.symbol,
            config=self.config,
            param_grid=self.param_grid,
            windows=window_results
        )

        if not window_results:
            return result

        # Collect all OOS trades
        for wr in window_results:
            result.oos_trades.extend(wr.oos_result.trades)

        # Build combined equity curve
        capital = self.config.initial_capital
        for wr in window_results:
            if wr.oos_result.equity_curve:
                # Scale equity curve to current capital
                first_equity = wr.oos_result.equity_curve[0]
                if first_equity > 0:
                    scale = capital / first_equity
                    scaled = [e * scale for e in wr.oos_result.equity_curve]
                    result.oos_equity_curve.extend(scaled)
                    capital = scaled[-1]

            if wr.oos_result.dates:
                result.oos_dates.extend(wr.oos_result.dates)

        # Calculate aggregated metrics
        oos_returns = [wr.oos_result.return_pct for wr in window_results]
        is_returns = [wr.is_result.return_pct for wr in window_results]
        oos_sharpes = [wr.oos_result.sharpe_ratio for wr in window_results]
        is_sharpes = [wr.is_result.sharpe_ratio for wr in window_results]

        # Total OOS return (compounded)
        cumulative_return = 1.0
        for ret in oos_returns:
            cumulative_return *= (1 + ret)
        result.total_oos_return = cumulative_return - 1

        # Average metrics
        result.avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        # Win rate across all OOS trades
        if result.oos_trades:
            winners = sum(1 for t in result.oos_trades if t.profit_loss > 0)
            result.oos_win_rate = winners / len(result.oos_trades)

        # Profit factor across all OOS trades
        total_profit = sum(t.profit_loss for t in result.oos_trades if t.profit_loss > 0)
        total_loss = abs(sum(t.profit_loss for t in result.oos_trades if t.profit_loss < 0))
        result.oos_profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        # Max drawdown from combined equity curve
        if result.oos_equity_curve:
            result.oos_max_drawdown = self._calculate_max_drawdown(result.oos_equity_curve)

        # Robustness metrics
        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        if avg_is_sharpe > 0:
            result.efficiency_ratio = result.avg_oos_sharpe / avg_is_sharpe
        else:
            result.efficiency_ratio = 0.0

        # Consistency: % of profitable OOS windows
        profitable_windows = sum(1 for r in oos_returns if r > 0)
        result.consistency_score = profitable_windows / len(oos_returns) if oos_returns else 0.0

        # Parameter stability: how often are the same parameters chosen?
        result.parameter_stability = self._calculate_param_stability(window_results)

        return result

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_param_stability(self, window_results: List[OptimizationResult]) -> float:
        """Calculate how stable the optimal parameters are across windows."""
        if len(window_results) < 2:
            return 1.0

        # Count how often each parameter value appears
        param_counts: Dict[str, Dict[Any, int]] = {}

        for wr in window_results:
            for param, value in wr.best_params.items():
                if param not in param_counts:
                    param_counts[param] = {}
                if value not in param_counts[param]:
                    param_counts[param][value] = 0
                param_counts[param][value] += 1

        # Calculate stability for each parameter
        stabilities = []
        n_windows = len(window_results)

        for param, counts in param_counts.items():
            # Most common value count
            max_count = max(counts.values())
            stabilities.append(max_count / n_windows)

        return np.mean(stabilities) if stabilities else 0.0

    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get the most frequently selected optimal parameters."""
        result = self.run()

        if not result.windows:
            return {}

        # Count parameter occurrences
        param_counts: Dict[str, Dict[Any, int]] = {}

        for wr in result.windows:
            for param, value in wr.best_params.items():
                if param not in param_counts:
                    param_counts[param] = {}
                if value not in param_counts[param]:
                    param_counts[param][value] = 0
                param_counts[param][value] += 1

        # Select most common value for each parameter
        optimal = {}
        for param, counts in param_counts.items():
            optimal[param] = max(counts.keys(), key=lambda v: counts[v])

        return optimal


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def walk_forward_optimize(
    data: pd.DataFrame,
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    symbol: str = "UNKNOWN",
    is_days: int = 252,
    oos_days: int = 63,
    objective: str = "sharpe"
) -> WFOResult:
    """
    Quick walk-forward optimization.

    Args:
        data: Historical OHLCV data
        strategy_class: Strategy class to optimize
        param_grid: Parameter grid
        symbol: Trading symbol
        is_days: In-sample period days
        oos_days: Out-of-sample period days
        objective: Optimization objective (sharpe, return, profit_factor)

    Returns:
        WFOResult with optimization results
    """
    obj_map = {
        'sharpe': ObjectiveFunction.SHARPE_RATIO,
        'return': ObjectiveFunction.TOTAL_RETURN,
        'profit_factor': ObjectiveFunction.PROFIT_FACTOR,
        'win_rate': ObjectiveFunction.WIN_RATE,
        'calmar': ObjectiveFunction.CALMAR_RATIO
    }

    config = WFOConfig(
        in_sample_days=is_days,
        out_of_sample_days=oos_days,
        objective=obj_map.get(objective, ObjectiveFunction.SHARPE_RATIO)
    )

    wfo = WalkForwardOptimizer(
        data=data,
        strategy_class=strategy_class,
        param_grid=param_grid,
        config=config,
        symbol=symbol
    )

    return wfo.run()


def print_wfo_report(result: WFOResult) -> None:
    """Print WFO result summary."""
    print(result.summary())


# =============================================================================
# ANCHORED WALK-FORWARD
# =============================================================================

class AnchoredWalkForward(WalkForwardOptimizer):
    """
    Anchored Walk-Forward Optimization.

    In anchored WFO, the in-sample period always starts from the beginning
    of the data. This gives more data for optimization in later windows.

    Example:
        Window 1: IS = Days 1-252, OOS = Days 253-315
        Window 2: IS = Days 1-315, OOS = Days 316-378
        Window 3: IS = Days 1-378, OOS = Days 379-441
        ...
    """

    def __init__(self, *args, **kwargs):
        # Force anchored mode
        if 'config' in kwargs:
            kwargs['config'].anchored = True
        else:
            kwargs['config'] = WFOConfig(anchored=True)

        super().__init__(*args, **kwargs)


# =============================================================================
# COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# =============================================================================

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Advanced cross-validation that:
    - Uses all possible train/test combinations
    - Purges overlapping data to prevent leakage
    - More statistically robust than simple WFO

    Reference: Marcos Lopez de Prado - Advances in Financial ML
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        symbol: str = "UNKNOWN"
    ):
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.symbol = symbol

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                self.data = data.set_index('date')
            else:
                raise ValueError("Data must have datetime index")

    def _generate_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate purged train/test splits."""
        splits = []
        n = len(self.data)
        fold_size = n // self.n_splits

        for test_fold in range(self.n_splits):
            # Test indices
            test_start = test_fold * fold_size
            test_end = (test_fold + 1) * fold_size if test_fold < self.n_splits - 1 else n

            # Train indices (all folds except test)
            train_indices = []
            for fold in range(self.n_splits):
                if fold != test_fold:
                    fold_start = fold * fold_size
                    fold_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
                    train_indices.extend(range(fold_start, fold_end))

            # Apply purging and embargo
            purged_train = self._apply_purging(
                train_indices,
                test_start,
                test_end
            )

            train_data = self.data.iloc[purged_train]
            test_data = self.data.iloc[test_start:test_end]

            splits.append((train_data, test_data))

        return splits

    def _apply_purging(
        self,
        train_indices: List[int],
        test_start: int,
        test_end: int
    ) -> List[int]:
        """Remove train samples that are too close to test period."""
        purge_window = self.purge_days + self.embargo_days

        purged = [
            i for i in train_indices
            if i < test_start - purge_window or i >= test_end + purge_window
        ]

        return purged

    def run(self, objective: ObjectiveFunction = ObjectiveFunction.SHARPE_RATIO) -> Dict[str, Any]:
        """Run combinatorial purged cross-validation."""
        splits = self._generate_splits()

        all_oos_results = []
        best_overall_params = None
        best_overall_score = float('-inf')

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Running fold {i+1}/{self.n_splits}")

            # Find best params on training data
            best_params, best_score = self._optimize_fold(train_data, objective)

            # Evaluate on test data
            strategy = self.strategy_class(**best_params)
            backtester = Backtester(initial_capital=100000)
            test_result = backtester.run(test_data, strategy, self.symbol)

            all_oos_results.append({
                'fold': i,
                'params': best_params,
                'train_score': best_score,
                'test_return': test_result.return_pct,
                'test_sharpe': test_result.sharpe_ratio
            })

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_params = best_params

        # Aggregate results
        avg_oos_return = np.mean([r['test_return'] for r in all_oos_results])
        avg_oos_sharpe = np.mean([r['test_sharpe'] for r in all_oos_results])

        return {
            'best_params': best_overall_params,
            'avg_oos_return': avg_oos_return,
            'avg_oos_sharpe': avg_oos_sharpe,
            'fold_results': all_oos_results,
            'n_splits': self.n_splits
        }

    def _optimize_fold(
        self,
        train_data: pd.DataFrame,
        objective: ObjectiveFunction
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize parameters on training fold."""
        best_params = None
        best_score = float('-inf')

        for params in self._get_param_combinations():
            try:
                strategy = self.strategy_class(**params)
                backtester = Backtester(initial_capital=100000)
                result = backtester.run(train_data, strategy, self.symbol)

                if objective == ObjectiveFunction.SHARPE_RATIO:
                    score = result.sharpe_ratio
                elif objective == ObjectiveFunction.TOTAL_RETURN:
                    score = result.return_pct
                else:
                    score = result.sharpe_ratio

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception:
                continue

        return best_params or {}, best_score

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations
