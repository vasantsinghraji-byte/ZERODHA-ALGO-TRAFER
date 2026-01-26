# -*- coding: utf-8 -*-
"""
Backtesting Module
==================
Test your strategies on historical data before risking real money!

RECOMMENDED: Use UnifiedBacktester for consistent results!

The Problem (Duality Trap):
- VectorizedBacktester is fast but can't handle complex orders
- Backtester (iterative) handles complex orders but is slow
- Cost models differed between engines - optimization results wouldn't match!

The Solution:
- UnifiedBacktester: Single interface, automatic engine selection
- BacktestConfig: Standardized cost model used by ALL engines
- Consistent results regardless of which engine runs

Usage:
    >>> from backtest import UnifiedBacktester, BacktestConfig, CostModel
    >>>
    >>> config = BacktestConfig(
    ...     initial_capital=100000,
    ...     cost_model=CostModel.ZERODHA_INTRADAY
    ... )
    >>> bt = UnifiedBacktester(data, config)
    >>> result = bt.run(strategy)

Engines available:
- UnifiedBacktester: RECOMMENDED - auto-selects best engine
- Backtester: Legacy bar-by-bar engine (backward compatible)
- EventDrivenBacktester: Event-driven engine (unified architecture)
- VectorizedBacktester: Ultra-fast Numba-optimized backtester
- WalkForwardOptimizer: Walk-forward optimization for robust parameter tuning
"""

# =============================================================================
# UNIFIED INTERFACE (RECOMMENDED)
# =============================================================================
from .unified import (
    # Main classes
    UnifiedBacktester,
    BacktestConfig,
    # Enums
    CostModel,
    EngineType,
    # Utilities
    detect_strategy_complexity,
    unified_backtest,
    compare_engines,
)

# =============================================================================
# INDIVIDUAL ENGINES (for advanced users)
# =============================================================================
from .engine import (
    # Data classes
    Trade,
    BacktestResult,
    # Legacy backtester
    Backtester,
    quick_backtest,
    print_backtest_report,
    # Event-driven backtester
    EventDrivenBacktester,
    event_driven_backtest,
)
from .metrics import calculate_metrics, print_report
from .wfo import (
    # Walk-Forward Optimization
    WalkForwardOptimizer,
    WFOConfig,
    WFOResult,
    ObjectiveFunction,
    WindowPeriod,
    OptimizationResult,
    AnchoredWalkForward,
    CombinatorialPurgedCV,
    walk_forward_optimize,
    print_wfo_report,
)
from .vectorized import (
    # Vectorized backtester
    VectorizedBacktester,
    VectorizedResult,
    # Fast indicators
    fast_sma,
    fast_ema,
    fast_rsi,
    fast_atr,
    fast_bollinger,
    fast_macd,
    # Benchmarking
    Benchmark,
    BenchmarkResult,
    quick_vectorized_backtest,
    run_benchmark,
    NUMBA_AVAILABLE,
)

__all__ = [
    # ==========================================================================
    # UNIFIED INTERFACE (RECOMMENDED)
    # ==========================================================================
    'UnifiedBacktester',
    'BacktestConfig',
    'CostModel',
    'EngineType',
    'detect_strategy_complexity',
    'unified_backtest',
    'compare_engines',

    # ==========================================================================
    # INDIVIDUAL ENGINES
    # ==========================================================================
    # Data classes
    'Trade',
    'BacktestResult',
    # Legacy backtester
    'Backtester',
    'quick_backtest',
    'print_backtest_report',
    # Event-driven backtester
    'EventDrivenBacktester',
    'event_driven_backtest',
    # Metrics
    'calculate_metrics',
    'print_report',
    # Walk-Forward Optimization
    'WalkForwardOptimizer',
    'WFOConfig',
    'WFOResult',
    'ObjectiveFunction',
    'WindowPeriod',
    'OptimizationResult',
    'AnchoredWalkForward',
    'CombinatorialPurgedCV',
    'walk_forward_optimize',
    'print_wfo_report',
    # Vectorized backtester
    'VectorizedBacktester',
    'VectorizedResult',
    'fast_sma',
    'fast_ema',
    'fast_rsi',
    'fast_atr',
    'fast_bollinger',
    'fast_macd',
    'Benchmark',
    'BenchmarkResult',
    'quick_vectorized_backtest',
    'run_benchmark',
    'NUMBA_AVAILABLE',
]
