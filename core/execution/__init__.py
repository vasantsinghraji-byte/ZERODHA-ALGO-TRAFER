# -*- coding: utf-8 -*-
"""
Algorithmic Execution Module - Smart Order Execution!
=======================================================
Execute large orders with minimal market impact.

Algorithms:
- TWAP: Time-Weighted Average Price - spread orders evenly over time
- VWAP: Volume-Weighted Average Price - follow volume patterns
- Iceberg: Hide large orders by showing only small visible quantities

Example:
    >>> from core.execution import TWAPExecutor, VWAPExecutor, IcebergExecutor
    >>>
    >>> # TWAP - execute 1000 shares over 30 minutes
    >>> twap = TWAPExecutor(broker)
    >>> twap.execute("RELIANCE", quantity=1000, duration_minutes=30)
    >>>
    >>> # VWAP - follow volume profile
    >>> vwap = VWAPExecutor(broker, volume_profile)
    >>> vwap.execute("TCS", quantity=500, duration_minutes=60)
    >>>
    >>> # Iceberg - hide 900 of 1000 shares
    >>> iceberg = IcebergExecutor(broker)
    >>> iceberg.execute("HDFCBANK", total_qty=1000, visible_qty=100)
"""

from .twap import (
    TWAPExecutor,
    TWAPConfig,
    TWAPOrder,
    TWAPSlice,
    TWAPResult,
)

from .vwap import (
    VWAPExecutor,
    VWAPConfig,
    VWAPOrder,
    VolumeProfile,
    VWAPResult,
)

from .iceberg import (
    IcebergExecutor,
    IcebergConfig,
    IcebergOrder,
    IcebergResult,
)

from .slippage_model import (
    # Models
    SlippageModel,
    FixedSlippage,
    LinearSlippage,
    SquareRootImpact,
    VolumeDependentSlippage,
    OrderBookSlippage,
    # Simulator
    FillSimulator,
    # Data classes
    SlippageConfig,
    SlippageResult,
    FillResult,
    SlippageType,
    FillType,
    # Functions
    calculate_slippage,
    simulate_fill,
    estimate_execution_cost,
)

from .smart_router import (
    # Core classes
    SmartRouter,
    SmartRouterConfig,
    QuoteProvider,
    DefaultQuoteProvider,
    # Enums
    Exchange,
    RoutingStrategy,
    OrderSide,
    # Data classes
    ExchangeQuote,
    TransactionCosts,
    ExchangeLatency,
    RoutingDecision,
    SplitResult,
    # Pre-configured costs
    NSE_COSTS,
    BSE_COSTS,
    # Functions
    get_smart_router,
    set_smart_router,
    route_order,
    compare_prices,
    best_exchange_for,
)

from .liquidity_aggregator import (
    # Core classes
    LiquidityAggregator,
    LiquidityAggregatorConfig,
    OrderBookProvider,
    # Enums
    LiquidityTier,
    ExecutionUrgency,
    # Data classes
    OrderBookLevel,
    AggregatedLevel,
    OrderBook,
    AggregatedOrderBook,
    LiquidityMetrics,
    ExecutionStep,
    ExecutionPlan,
    # Functions
    get_liquidity_aggregator,
    set_liquidity_aggregator,
    get_aggregated_book,
    analyze_liquidity,
    plan_execution,
)

__all__ = [
    # TWAP
    'TWAPExecutor',
    'TWAPConfig',
    'TWAPOrder',
    'TWAPSlice',
    'TWAPResult',
    # VWAP
    'VWAPExecutor',
    'VWAPConfig',
    'VWAPOrder',
    'VolumeProfile',
    'VWAPResult',
    # Iceberg
    'IcebergExecutor',
    'IcebergConfig',
    'IcebergOrder',
    'IcebergResult',
    # Slippage Models
    'SlippageModel',
    'FixedSlippage',
    'LinearSlippage',
    'SquareRootImpact',
    'VolumeDependentSlippage',
    'OrderBookSlippage',
    'FillSimulator',
    'SlippageConfig',
    'SlippageResult',
    'FillResult',
    'SlippageType',
    'FillType',
    'calculate_slippage',
    'simulate_fill',
    'estimate_execution_cost',
    # Smart Router
    'SmartRouter',
    'SmartRouterConfig',
    'QuoteProvider',
    'DefaultQuoteProvider',
    'Exchange',
    'RoutingStrategy',
    'OrderSide',
    'ExchangeQuote',
    'TransactionCosts',
    'ExchangeLatency',
    'RoutingDecision',
    'SplitResult',
    'NSE_COSTS',
    'BSE_COSTS',
    'get_smart_router',
    'set_smart_router',
    'route_order',
    'compare_prices',
    'best_exchange_for',
    # Liquidity Aggregator
    'LiquidityAggregator',
    'LiquidityAggregatorConfig',
    'OrderBookProvider',
    'LiquidityTier',
    'ExecutionUrgency',
    'OrderBookLevel',
    'AggregatedLevel',
    'OrderBook',
    'AggregatedOrderBook',
    'LiquidityMetrics',
    'ExecutionStep',
    'ExecutionPlan',
    'get_liquidity_aggregator',
    'set_liquidity_aggregator',
    'get_aggregated_book',
    'analyze_liquidity',
    'plan_execution',
]
