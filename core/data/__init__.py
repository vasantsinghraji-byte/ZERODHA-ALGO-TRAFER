"""
Unified Data Layer for Event-Driven Trading Engine.

This module provides data sources that emit events to the EventBus.
All sources implement the same interface, allowing the trading engine
to use THE SAME CODE for backtest, live trading, and replay.

Data Sources:
    - HistoricalDataSource: Replay historical OHLCV bars for backtesting
    - LiveDataSource: Stream real-time ticks from WebSocket
    - ReplayDataSource: Replay recorded live sessions for debugging
    - SimulatedLiveSource: Generate synthetic ticks for testing

Usage:
    from core.data import (
        DataSourceConfig,
        HistoricalDataSource,
        LiveDataSource,
        ReplayDataSource,
    )
    from core.events import get_event_bus

    # For backtesting
    config = DataSourceConfig(
        symbols=['RELIANCE', 'TCS'],
        timeframe='1d',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
    )
    bus = get_event_bus()
    source = HistoricalDataSource(config, bus, data={'RELIANCE': df})
    source.start()  # Emits BarEvents to bus

    # For live trading
    config = DataSourceConfig(symbols=['RELIANCE'], emit_ticks=True)
    source = LiveDataSource(config, bus, api_key='...', access_token='...')
    source.start_async()  # Streams TickEvents to bus

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       DATA SOURCES                              │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
    │  │ Historical  │ │    Live     │ │   Replay    │               │
    │  │ DataFrame   │ │  WebSocket  │ │   File      │               │
    │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘               │
    │         │               │               │                       │
    │         ▼               ▼               ▼                       │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │              UNIFIED DATASOURCE INTERFACE               │   │
    │  │         emit_events() → BarEvent | TickEvent            │   │
    │  └───────────────────────────┬─────────────────────────────┘   │
    │                              │                                  │
    │                              ▼                                  │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │                      EVENT BUS                          │   │
    │  │              Same events, same handlers!                │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
"""

# Base classes and configuration
from .source import (
    DataSource,
    DataSourceConfig,
    DataSourceMode,
    DataSourceState,
    DataSourceStats,
    create_bar_event,
    create_tick_event,
    dataframe_to_bar_events,
)

# Data source implementations
from .historical import (
    HistoricalDataSource,
    MultiTimeframeHistoricalSource,
)

from .live import (
    LiveDataSource,
    SimulatedLiveSource,
    BarAggregator,
)

from .replay import (
    ReplayDataSource,
    EventRecorder,
    RecordedEvent,
)

# TimescaleDB adapter
from .timescale import (
    # Connection
    TimescaleConnection,
    TimescaleConfig,
    # Schema
    SchemaManager,
    # Repositories
    TickRepository,
    BarRepository,
    QueryBuilder,
    # Data classes
    Tick,
    Bar,
    Timeframe,
    # Convenience
    create_timescale_connection,
    initialize_database,
    TIMESCALE_AVAILABLE,
)

# Migration tools
from .migration import (
    MigrationManager,
    MigrationConfig,
    MigrationResult,
    MigrationStatus,
    MigrationType,
    ValidationResult,
    SQLiteReader,
    migrate_sqlite_to_timescale,
    validate_migration,
)

# Tick filtering
from .tick_filter import (
    TickFilter,
    TickFilterConfig,
    SymbolFilterConfig,
    TickFilterStats,
    FilterReason,
    RejectedTick,
    get_tick_filter,
    set_tick_filter,
)

# Corporate actions
from .corporate_actions import (
    CorporateActionHandler,
    CorporateActionConfig,
    CorporateAction,
    CorporateActionType,
    AdjustmentRecord,
    AdjustmentType,
    get_corporate_action_handler,
    set_corporate_action_handler,
)

# Order book processing
from .orderbook import (
    OrderBook,
    OrderBookManager,
    OrderBookSnapshot,
    OrderBookMetrics,
    OrderBookConfig,
    PriceLevel,
    DepthLevel,
    get_orderbook_manager,
    set_orderbook_manager,
)

# Historical data adjustment
from .historical_adjustment import (
    HistoricalDataAdjuster,
    HistoricalAdjustmentConfig,
    AdjustmentMode,
    DividendHandling,
    DividendReinvestment,
    adjust_for_backtest,
    simulate_drip,
    get_historical_adjuster,
    set_historical_adjuster,
)


__all__ = [
    # Base classes
    'DataSource',
    'DataSourceConfig',
    'DataSourceMode',
    'DataSourceState',
    'DataSourceStats',

    # Helper functions
    'create_bar_event',
    'create_tick_event',
    'dataframe_to_bar_events',

    # Historical (Backtest)
    'HistoricalDataSource',
    'MultiTimeframeHistoricalSource',

    # Live Trading
    'LiveDataSource',
    'SimulatedLiveSource',
    'BarAggregator',

    # Replay (Debug)
    'ReplayDataSource',
    'EventRecorder',
    'RecordedEvent',

    # TimescaleDB
    'TimescaleConnection',
    'TimescaleConfig',
    'SchemaManager',
    'TickRepository',
    'BarRepository',
    'QueryBuilder',
    'Tick',
    'Bar',
    'Timeframe',
    'create_timescale_connection',
    'initialize_database',
    'TIMESCALE_AVAILABLE',

    # Migration
    'MigrationManager',
    'MigrationConfig',
    'MigrationResult',
    'MigrationStatus',
    'MigrationType',
    'ValidationResult',
    'SQLiteReader',
    'migrate_sqlite_to_timescale',
    'validate_migration',

    # Tick Filter
    'TickFilter',
    'TickFilterConfig',
    'SymbolFilterConfig',
    'TickFilterStats',
    'FilterReason',
    'RejectedTick',
    'get_tick_filter',
    'set_tick_filter',

    # Corporate Actions
    'CorporateActionHandler',
    'CorporateActionConfig',
    'CorporateAction',
    'CorporateActionType',
    'AdjustmentRecord',
    'AdjustmentType',
    'get_corporate_action_handler',
    'set_corporate_action_handler',

    # Order Book
    'OrderBook',
    'OrderBookManager',
    'OrderBookSnapshot',
    'OrderBookMetrics',
    'OrderBookConfig',
    'PriceLevel',
    'DepthLevel',
    'get_orderbook_manager',
    'set_orderbook_manager',

    # Historical Data Adjustment
    'HistoricalDataAdjuster',
    'HistoricalAdjustmentConfig',
    'AdjustmentMode',
    'DividendHandling',
    'DividendReinvestment',
    'adjust_for_backtest',
    'simulate_drip',
    'get_historical_adjuster',
    'set_historical_adjuster',
]
