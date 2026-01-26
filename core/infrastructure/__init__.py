"""
Core Infrastructure Module
Provides foundational utilities for the trading system.
"""

from core.infrastructure.rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    EndpointConfig,
    RateLimitExceeded,
    APIEndpoint,
)

from core.infrastructure.latency_monitor import (
    LatencyMonitor,
    LatencyMonitorConfig,
    LatencyThresholds,
    LatencyType,
    LatencyStats,
    LatencyAlert,
    AlertSeverity,
    get_latency_monitor,
    set_latency_monitor,
)

from core.infrastructure.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
    KillSwitchEvent,
    TriggerReason,
    get_kill_switch,
    set_kill_switch,
    trigger_emergency_stop,
)

from core.infrastructure.hot_path import (
    # Profiling
    HotPathProfiler,
    ProfileResult,
    HotPathMetrics,
    HotPathConfig,
    # Fast functions
    fast_sma,
    fast_ema,
    fast_rsi,
    fast_macd,
    fast_bollinger,
    fast_atr,
    fast_stochastic,
    fast_correlation_matrix,
    # Tick processing
    OptimizedTick,
    HotPathTickProcessor,
    # Utilities
    get_profiler,
    set_profiler,
    profile,
    is_numba_available,
    benchmark_fast_functions,
)

from core.infrastructure.connection_pool import (
    # WebSocket
    WebSocketConnection,
    WebSocketManager,
    WebSocketConfig,
    # Connection Pool
    ConnectionPool,
    ConnectionPoolConfig,
    # Co-location
    CoLocationOptimizer,
    CoLocationConfig,
    # Enums
    ConnectionState,
    DataCenter,
    # Data classes
    ConnectionStats,
    # Functions
    get_websocket_manager,
    set_websocket_manager,
    get_connection_pool,
    set_connection_pool,
    get_colocation_optimizer,
    measure_latency,
    analyze_network,
)

from core.infrastructure.flight_recorder import (
    # Main classes
    FlightRecorder,
    MarketReplayer,
    TickRecorder,
    EventRecorder,
    DebugAnalyzer,
    # Data classes
    TickRecord,
    EventRecord,
    RecordingSession,
    # Enums
    CompressionType,
    RecordType,
    PlaybackState,
    # Compression
    Compressor,
    # Functions
    get_flight_recorder,
    set_flight_recorder,
    start_recording,
    stop_recording,
    record_tick,
    record_event,
)

from core.infrastructure.shadow_mode import (
    # Main classes
    ShadowEngine,
    ShadowBroker,
    ShadowStrategy,
    ValidationGate,
    PnLTracker,
    # Data classes
    ShadowOrder,
    ShadowPosition,
    ShadowTrade,
    PerformanceMetrics,
    PerformanceComparison,
    ValidationResult,
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    ShadowMode,
    ValidationStatus,
    # Functions
    get_shadow_engine,
    set_shadow_engine,
    register_shadow,
    shadow_on_tick,
    validate_shadow,
)

from core.infrastructure.ab_testing import (
    # Main classes
    ABTestFramework,
    ABTest,
    ABTestConfig,
    SignificanceTester,
    TrafficAllocator,
    RollbackMonitor,
    # Data classes
    TrafficSchedule,
    VariantMetrics,
    SignificanceResult,
    RollbackEvent,
    # Enums
    TestStatus,
    SignificanceMethod,
    TrafficAllocation,
    RollbackReason,
    # Functions
    get_ab_framework,
    set_ab_framework,
    create_ab_test,
    get_variant,
    record_ab_outcome,
)

from core.infrastructure.audit_trail import (
    # Main classes
    AuditTrail,
    HashChain,
    FileAuditStorage,
    RegulatoryReporter,
    # Data classes
    AuditRecord,
    IntegrityReport,
    # Enums
    AuditEventType,
    ReportFormat,
    IntegrityStatus,
    # Functions
    get_audit_trail,
    set_audit_trail,
    log_trade,
    verify_audit_integrity,
)

from core.infrastructure.risk_compliance import (
    # Main classes
    ComplianceEngine,
    PositionTracker,
    CircuitBreakerMonitor,
    SEBIComplianceRules,
    MarketHoursChecker,
    # Data classes
    PositionLimit,
    PriceBand,
    CircuitBreakerStatus,
    ComplianceResult,
    ComplianceAlert,
    TradingSession,
    # Enums
    ViolationType,
    ComplianceAction,
    CircuitBreakerLevel,
    MarketStatus,
    Exchange,
    # Functions
    get_compliance_engine,
    set_compliance_engine,
    check_compliance,
    is_market_open,
)

from core.infrastructure.integration import (
    # Main class
    InfrastructureManager,
    # Config
    InfrastructureConfig,
    InfrastructureStatus,
    # Functions
    get_infrastructure_manager,
    set_infrastructure_manager,
    initialize_infrastructure,
)

__all__ = [
    # Rate Limiter
    'RateLimiter',
    'RateLimiterConfig',
    'EndpointConfig',
    'RateLimitExceeded',
    'APIEndpoint',
    # Latency Monitor
    'LatencyMonitor',
    'LatencyMonitorConfig',
    'LatencyThresholds',
    'LatencyType',
    'LatencyStats',
    'LatencyAlert',
    'AlertSeverity',
    'get_latency_monitor',
    'set_latency_monitor',
    # Kill Switch
    'KillSwitch',
    'KillSwitchConfig',
    'KillSwitchState',
    'KillSwitchEvent',
    'TriggerReason',
    'get_kill_switch',
    'set_kill_switch',
    'trigger_emergency_stop',
    # Hot Path
    'HotPathProfiler',
    'ProfileResult',
    'HotPathMetrics',
    'HotPathConfig',
    'fast_sma',
    'fast_ema',
    'fast_rsi',
    'fast_macd',
    'fast_bollinger',
    'fast_atr',
    'fast_stochastic',
    'fast_correlation_matrix',
    'OptimizedTick',
    'HotPathTickProcessor',
    'get_profiler',
    'set_profiler',
    'profile',
    'is_numba_available',
    'benchmark_fast_functions',
    # Connection Pool
    'WebSocketConnection',
    'WebSocketManager',
    'WebSocketConfig',
    'ConnectionPool',
    'ConnectionPoolConfig',
    'CoLocationOptimizer',
    'CoLocationConfig',
    'ConnectionState',
    'DataCenter',
    'ConnectionStats',
    'get_websocket_manager',
    'set_websocket_manager',
    'get_connection_pool',
    'set_connection_pool',
    'get_colocation_optimizer',
    'measure_latency',
    'analyze_network',
    # Flight Recorder
    'FlightRecorder',
    'MarketReplayer',
    'TickRecorder',
    'EventRecorder',
    'DebugAnalyzer',
    'TickRecord',
    'EventRecord',
    'RecordingSession',
    'CompressionType',
    'RecordType',
    'PlaybackState',
    'Compressor',
    'get_flight_recorder',
    'set_flight_recorder',
    'start_recording',
    'stop_recording',
    'record_tick',
    'record_event',
    # Shadow Mode
    'ShadowEngine',
    'ShadowBroker',
    'ShadowStrategy',
    'ValidationGate',
    'PnLTracker',
    'ShadowOrder',
    'ShadowPosition',
    'ShadowTrade',
    'PerformanceMetrics',
    'PerformanceComparison',
    'ValidationResult',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'ShadowMode',
    'ValidationStatus',
    'get_shadow_engine',
    'set_shadow_engine',
    'register_shadow',
    'shadow_on_tick',
    'validate_shadow',
    # A/B Testing
    'ABTestFramework',
    'ABTest',
    'ABTestConfig',
    'SignificanceTester',
    'TrafficAllocator',
    'RollbackMonitor',
    'TrafficSchedule',
    'VariantMetrics',
    'SignificanceResult',
    'RollbackEvent',
    'TestStatus',
    'SignificanceMethod',
    'TrafficAllocation',
    'RollbackReason',
    'get_ab_framework',
    'set_ab_framework',
    'create_ab_test',
    'get_variant',
    'record_ab_outcome',
    # Audit Trail
    'AuditTrail',
    'HashChain',
    'FileAuditStorage',
    'RegulatoryReporter',
    'AuditRecord',
    'IntegrityReport',
    'AuditEventType',
    'ReportFormat',
    'IntegrityStatus',
    'get_audit_trail',
    'set_audit_trail',
    'log_trade',
    'verify_audit_integrity',
    # Risk Compliance
    'ComplianceEngine',
    'PositionTracker',
    'CircuitBreakerMonitor',
    'SEBIComplianceRules',
    'MarketHoursChecker',
    'PositionLimit',
    'PriceBand',
    'CircuitBreakerStatus',
    'ComplianceResult',
    'ComplianceAlert',
    'TradingSession',
    'ViolationType',
    'ComplianceAction',
    'CircuitBreakerLevel',
    'MarketStatus',
    'Exchange',
    'get_compliance_engine',
    'set_compliance_engine',
    'check_compliance',
    'is_market_open',
    # Integration
    'InfrastructureManager',
    'InfrastructureConfig',
    'InfrastructureStatus',
    'get_infrastructure_manager',
    'set_infrastructure_manager',
    'initialize_infrastructure',
]
