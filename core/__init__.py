"""
Core Trading Engine Module - The Brain!
=======================================
Contains all the core trading logic.

Components:
- Broker: Zerodha API connection
- DataManager: Historical & live data
- OrderManager: Buy/sell orders
- PositionManager: Track holdings
- TradingEngine: Main autopilot
- RiskManager: Safety controls
- Bootstrap: Component initialization & validation
"""

from .broker import ZerodhaBroker, Quote, Order, Position
from .data_manager import DataManager
from .order_manager import OrderManager, OrderStatus, Side, OrderType
from .position_manager import PositionManager, Position as ManagedPosition
from .trading_engine import (
    TradingEngine,
    TradingMode,
    EngineConfig,
    create_paper_engine,
    # Event-driven engine
    EventDrivenLiveEngine,
    create_event_driven_paper_engine,
    create_event_driven_live_engine,
)
from .live_feed import LiveFeed, SimulatedFeed, Tick

# Risk management
from .risk_manager import (
    RiskManager,
    RiskConfig,
    RiskMetrics,
    StopLossType,
    calculate_risk_reward,
    calculate_position_size_fixed
)

# Portfolio risk
from .risk import (
    # Correlation
    CorrelationAnalyzer,
    CorrelationMatrix,
    CorrelationConfig,
    calculate_correlation,
    find_high_correlations,
    # Portfolio
    PortfolioRiskManager,
    PortfolioRiskConfig,
    SectorExposure,
    PortfolioRiskReport,
    check_correlation_risk,
    check_sector_exposure,
    # VaR
    VaRCalculator,
    VaRResult,
    calculate_var,
    # Beta
    BetaCalculator,
    BetaResult,
    calculate_beta,
    # Drawdown
    DrawdownTracker,
    DrawdownMetrics,
    calculate_max_drawdown,
)

# State persistence
from .state import (
    StrategyState,
    StrategyStateManager,
    CheckpointManager,
    CheckpointConfig,
    get_state_manager,
    get_checkpoint_manager,
    # Recovery
    RecoveryManager,
    RecoveryConfig,
    RecoveryResult,
    SessionTracker,
    get_recovery_manager,
    # Reconciliation
    ReconciliationManager,
    ReconciliationConfig,
    ReconciliationReport,
    reconcile_positions,
    check_position_sync,
)

# Infrastructure utilities
from .infrastructure import (
    # Rate Limiter
    RateLimiter,
    RateLimiterConfig,
    EndpointConfig,
    RateLimitExceeded,
    APIEndpoint,
    # Latency Monitor
    LatencyMonitor,
    LatencyMonitorConfig,
    LatencyThresholds,
    LatencyType,
    LatencyStats,
    LatencyAlert,
    AlertSeverity,
    get_latency_monitor,
    set_latency_monitor,
    # Kill Switch
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
    KillSwitchEvent,
    TriggerReason,
    get_kill_switch,
    set_kill_switch,
    trigger_emergency_stop,
)

# Execution algorithms
from .execution import (
    TWAPExecutor,
    TWAPConfig,
    TWAPOrder,
    TWAPResult,
    VWAPExecutor,
    VWAPConfig,
    VolumeProfile,
    VWAPResult,
    IcebergExecutor,
    IcebergConfig,
    IcebergOrder,
    IcebergResult,
    # Slippage models
    SlippageModel,
    SquareRootImpact,
    FillSimulator,
    SlippageConfig,
    SlippageResult,
    FillResult,
    calculate_slippage,
    simulate_fill,
)

__all__ = [
    # Broker
    'ZerodhaBroker',
    'Quote',
    'Order',
    'Position',

    # Data
    'DataManager',
    'LiveFeed',
    'SimulatedFeed',
    'Tick',

    # Orders
    'OrderManager',
    'OrderStatus',
    'Side',
    'OrderType',

    # Positions
    'PositionManager',
    'ManagedPosition',

    # Engine
    'TradingEngine',
    'TradingMode',
    'EngineConfig',
    'create_paper_engine',
    # Event-driven engine
    'EventDrivenLiveEngine',
    'create_event_driven_paper_engine',
    'create_event_driven_live_engine',

    # Risk
    'RiskManager',
    'RiskConfig',
    'RiskMetrics',
    'StopLossType',
    'calculate_risk_reward',
    'calculate_position_size_fixed',
    # Portfolio Risk
    'CorrelationAnalyzer',
    'CorrelationMatrix',
    'CorrelationConfig',
    'calculate_correlation',
    'find_high_correlations',
    'PortfolioRiskManager',
    'PortfolioRiskConfig',
    'SectorExposure',
    'PortfolioRiskReport',
    'check_correlation_risk',
    'check_sector_exposure',
    # VaR
    'VaRCalculator',
    'VaRResult',
    'calculate_var',
    # Beta
    'BetaCalculator',
    'BetaResult',
    'calculate_beta',
    # Drawdown
    'DrawdownTracker',
    'DrawdownMetrics',
    'calculate_max_drawdown',

    # State Persistence
    'StrategyState',
    'StrategyStateManager',
    'CheckpointManager',
    'CheckpointConfig',
    'get_state_manager',
    'get_checkpoint_manager',
    # Recovery
    'RecoveryManager',
    'RecoveryConfig',
    'RecoveryResult',
    'SessionTracker',
    'get_recovery_manager',
    # Reconciliation
    'ReconciliationManager',
    'ReconciliationConfig',
    'ReconciliationReport',
    'reconcile_positions',
    'check_position_sync',

    # Execution Algorithms
    'TWAPExecutor',
    'TWAPConfig',
    'TWAPOrder',
    'TWAPResult',
    'VWAPExecutor',
    'VWAPConfig',
    'VolumeProfile',
    'VWAPResult',
    'IcebergExecutor',
    'IcebergConfig',
    'IcebergOrder',
    'IcebergResult',
    # Slippage Models
    'SlippageModel',
    'SquareRootImpact',
    'FillSimulator',
    'SlippageConfig',
    'SlippageResult',
    'FillResult',
    'calculate_slippage',
    'simulate_fill',

    # Infrastructure
    'RateLimiter',
    'RateLimiterConfig',
    'EndpointConfig',
    'RateLimitExceeded',
    'APIEndpoint',
    'LatencyMonitor',
    'LatencyMonitorConfig',
    'LatencyThresholds',
    'LatencyType',
    'LatencyStats',
    'LatencyAlert',
    'AlertSeverity',
    'get_latency_monitor',
    'set_latency_monitor',
    'KillSwitch',
    'KillSwitchConfig',
    'KillSwitchState',
    'KillSwitchEvent',
    'TriggerReason',
    'get_kill_switch',
    'set_kill_switch',
    'trigger_emergency_stop',

    # Bootstrap
    'ComponentRegistry',
    'bootstrap_application',
    'validate_components_only',
]

# Bootstrap (component initialization)
from .bootstrap import (
    ComponentRegistry,
    ComponentInfo,
    ComponentStatus,
    bootstrap_application,
    validate_components_only,
)
