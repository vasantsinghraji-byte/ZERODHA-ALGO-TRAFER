# -*- coding: utf-8 -*-
"""
Backtesting Module
==================
Test your strategies on historical data before risking real money!

Engines available:
- Backtester: Legacy bar-by-bar engine (backward compatible)
- EventDrivenBacktester: New event-driven engine (unified architecture)
- VectorizedBacktester: Ultra-fast Numba-optimized backtester
- WalkForwardOptimizer: Walk-forward optimization for robust parameter tuning
"""

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
