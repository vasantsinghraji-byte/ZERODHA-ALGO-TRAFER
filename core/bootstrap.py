# -*- coding: utf-8 -*-
"""
Component Bootstrap & Validation
=================================
Unified initialization of all Phase 8-12 components with hard-fail validation.

This module ensures:
1. All blueprint components (Phase 8-12) are loaded at startup
2. Missing required components cause immediate failure (not silent skip)
3. Startup logs confirm each component's status
4. CI/CD pipelines can verify complete initialization

Usage:
    from core.bootstrap import bootstrap_application, ComponentRegistry

    # Initialize everything
    registry = bootstrap_application()

    # Access components
    event_bus = registry.event_bus
    engine = registry.trading_engine
    infrastructure = registry.infrastructure_manager
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ============================================================================
# Component Status
# ============================================================================

class ComponentStatus(Enum):
    """Status of a component during initialization."""
    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    description: str
    module_path: str
    class_name: str
    phase: str  # e.g., "8.1.1", "8.2.1"
    priority: int  # Lower = load first
    required: bool = True
    status: ComponentStatus = ComponentStatus.PENDING
    instance: Any = None
    error: Optional[str] = None
    load_time_ms: float = 0.0


# ============================================================================
# Component Registry
# ============================================================================

class ComponentRegistry:
    """
    Central registry for all application components.

    Manages initialization order, dependencies, and validation of
    all Phase 8-12 components defined in BLUEPRINT.md:
    - Phase 8: Event-Driven Architecture (12 components)
    - Phase 9: Data Superiority (4 components)
    - Phase 10: Execution Alpha (2 components)
    - Phase 11: MLOps (4 components)
    - Phase 12: Infrastructure & Compliance (6 components)
    """

    # All Phase 8-12 components from BLUEPRINT.md
    REQUIRED_COMPONENTS = [
        # Phase 8.1 - Event-Driven Architecture
        ComponentInfo(
            name="event_bus",
            description="Core Event System (Pub/Sub)",
            module_path="core.events",
            class_name="EventBus",
            phase="8.1.1",
            priority=1,
            required=True
        ),
        ComponentInfo(
            name="data_source",
            description="Unified Data Source Interface",
            module_path="core.data",
            class_name="DataSource",
            phase="8.1.2",
            priority=2,
            required=True
        ),
        ComponentInfo(
            name="strategy_base",
            description="Unified Strategy Interface",
            module_path="strategies.base",
            class_name="Strategy",
            phase="8.1.3",
            priority=3,
            required=True
        ),

        # Phase 8.2 - Engine Updates
        ComponentInfo(
            name="backtest_engine",
            description="Event-Driven Backtest Engine",
            module_path="backtest",
            class_name="EventDrivenBacktester",
            phase="8.2.1",
            priority=10,
            required=True
        ),
        ComponentInfo(
            name="live_engine",
            description="Event-Driven Live Trading Engine",
            module_path="core.trading_engine",
            class_name="EventDrivenLiveEngine",
            phase="8.2.2",
            priority=11,
            required=True
        ),

        # Phase 8.3 - State Persistence
        ComponentInfo(
            name="state_manager",
            description="Strategy State Persistence",
            module_path="core.state",
            class_name="StrategyStateManager",
            phase="8.3",
            priority=20,
            required=True
        ),

        # Phase 8.4 - Walk-Forward Optimization
        ComponentInfo(
            name="wfo_optimizer",
            description="Walk-Forward Optimization Engine",
            module_path="backtest.wfo",
            class_name="WalkForwardOptimizer",
            phase="8.4",
            priority=30,
            required=True
        ),

        # Phase 8.5 - Portfolio Risk Management
        ComponentInfo(
            name="portfolio_risk",
            description="Portfolio-Level Risk Manager",
            module_path="core.risk",
            class_name="PortfolioRiskManager",
            phase="8.5",
            priority=40,
            required=True
        ),

        # Phase 8.6 - Algorithmic Execution
        ComponentInfo(
            name="execution_algos",
            description="Algorithmic Execution (TWAP/VWAP/Iceberg)",
            module_path="core.execution",
            class_name="TWAPExecutor",
            phase="8.6",
            priority=50,
            required=True
        ),

        # Phase 8.7 - TimescaleDB Migration
        ComponentInfo(
            name="timescale_db",
            description="TimescaleDB Data Layer",
            module_path="core.data.timescale",
            class_name="TimescaleRepository",
            phase="8.7",
            priority=60,
            required=False  # Optional - requires DB server
        ),

        # Phase 8.8 - Infrastructure & Guardrails
        ComponentInfo(
            name="infrastructure",
            description="Infrastructure Manager (Kill Switch, Rate Limiter, etc.)",
            module_path="core.infrastructure",
            class_name="InfrastructureManager",
            phase="8.8",
            priority=70,
            required=True
        ),

        # Phase 8.9 - Data Integrity
        ComponentInfo(
            name="tick_filter",
            description="Bad Tick Filter & Data Integrity",
            module_path="core.data.tick_filter",
            class_name="TickFilter",
            phase="8.9",
            priority=80,
            required=True
        ),

        # =====================================================================
        # PHASE 9: DATA SUPERIORITY (Information Advantage)
        # =====================================================================

        # Phase 9.1.1 - Order Book Processing
        ComponentInfo(
            name="orderbook",
            description="Order Book Processing & Analysis",
            module_path="core.data.orderbook",
            class_name="OrderBook",
            phase="9.1.1",
            priority=100,
            required=True
        ),

        # Phase 9.1.2 - Market Microstructure Indicators
        ComponentInfo(
            name="microstructure",
            description="Market Microstructure Indicators",
            module_path="indicators.microstructure",
            class_name="MicrostructureAnalyzer",
            phase="9.1.2",
            priority=101,
            required=True
        ),

        # Phase 9.2.1 - News Sentiment Engine (Optional - requires API keys)
        ComponentInfo(
            name="news_sentiment",
            description="News Sentiment Analysis Engine",
            module_path="core.data.news_sentiment",
            class_name="NewsSentimentEngine",
            phase="9.2.1",
            priority=102,
            required=False  # Optional - requires external API
        ),

        # Phase 9.3.1 - Corporate Actions Handler
        ComponentInfo(
            name="corporate_actions",
            description="Corporate Actions Handler (Splits, Dividends)",
            module_path="core.data.corporate_actions",
            class_name="CorporateActionHandler",
            phase="9.3.1",
            priority=103,
            required=True
        ),

        # =====================================================================
        # PHASE 10: EXECUTION ALPHA (Smart Order Routing)
        # =====================================================================

        # Phase 10.1.1 - Smart Order Router
        ComponentInfo(
            name="smart_router",
            description="Smart Order Router (Multi-Exchange)",
            module_path="core.execution.smart_router",
            class_name="SmartRouter",
            phase="10.1.1",
            priority=110,
            required=True
        ),

        # Phase 10.1.2 - Liquidity Aggregation
        ComponentInfo(
            name="liquidity_aggregator",
            description="Liquidity Aggregation Engine",
            module_path="core.execution.liquidity_aggregator",
            class_name="LiquidityAggregator",
            phase="10.1.2",
            priority=111,
            required=True
        ),

        # =====================================================================
        # PHASE 11: MLOPS (Machine Learning Operations)
        # =====================================================================

        # Phase 11.1.1 - Feature Store
        ComponentInfo(
            name="feature_store",
            description="Centralized Feature Repository",
            module_path="ml.feature_store",
            class_name="FeatureStore",
            phase="11.1.1",
            priority=120,
            required=True
        ),

        # Phase 11.1.2 - Feature Pipeline
        ComponentInfo(
            name="feature_pipeline",
            description="Feature Engineering Pipeline",
            module_path="ml.feature_store.pipeline",
            class_name="FeaturePipeline",
            phase="11.1.2",
            priority=121,
            required=True
        ),

        # Phase 11.2.1 - Drift Detection
        ComponentInfo(
            name="drift_detector",
            description="Concept Drift Detection",
            module_path="ml.drift_detector",
            class_name="DriftDetector",
            phase="11.2.1",
            priority=122,
            required=True
        ),

        # Phase 11.2.2 - Model Registry
        ComponentInfo(
            name="model_registry",
            description="Model Registry & Deployment",
            module_path="ml.model_registry",
            class_name="ModelRegistry",
            phase="11.2.2",
            priority=123,
            required=True
        ),

        # =====================================================================
        # PHASE 12: INFRASTRUCTURE & COMPLIANCE
        # =====================================================================

        # Phase 12.1.1 - Flight Recorder
        ComponentInfo(
            name="flight_recorder",
            description="Black Box Recorder (Market Replay)",
            module_path="core.infrastructure.flight_recorder",
            class_name="FlightRecorder",
            phase="12.1.1",
            priority=130,
            required=True
        ),

        # Phase 12.2.1 - Shadow Trading Engine
        ComponentInfo(
            name="shadow_engine",
            description="Shadow Mode (Paper Trading Validation)",
            module_path="core.infrastructure.shadow_mode",
            class_name="ShadowEngine",
            phase="12.2.1",
            priority=131,
            required=True
        ),

        # Phase 12.2.2 - A/B Testing Framework
        ComponentInfo(
            name="ab_testing",
            description="A/B Testing Framework",
            module_path="core.infrastructure.ab_testing",
            class_name="ABTestFramework",
            phase="12.2.2",
            priority=132,
            required=True
        ),

        # Phase 12.3.1 - Audit Trail
        ComponentInfo(
            name="audit_trail",
            description="Trade Audit Trail (Immutable Log)",
            module_path="core.infrastructure.audit_trail",
            class_name="AuditTrail",
            phase="12.3.1",
            priority=133,
            required=True
        ),

        # Phase 12.3.2 - Risk Compliance
        ComponentInfo(
            name="risk_compliance",
            description="Risk Compliance Engine (SEBI Rules)",
            module_path="core.infrastructure.risk_compliance",
            class_name="ComplianceEngine",
            phase="12.3.2",
            priority=134,
            required=True
        ),

        # Phase 12.4.1 - Infrastructure Integration
        ComponentInfo(
            name="infra_integration",
            description="Unified Infrastructure Integration",
            module_path="core.infrastructure.integration",
            class_name="InfrastructureManager",
            phase="12.4.1",
            priority=135,
            required=True
        ),

        # =====================================================================
        # GUARDIAN ANGEL - Account Protection (NEW!)
        # =====================================================================
        ComponentInfo(
            name="guardian_angel",
            description="Account Blowup Prevention (Position Sync & Server-Side SL)",
            module_path="core.guardian",
            class_name="GuardianAngel",
            phase="12.5.1",
            priority=140,
            required=True
        ),
    ]

    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[str, ComponentInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._initialized = False
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # Core instances (set during initialization)
        self._event_bus = None
        self._trading_engine = None
        self._infrastructure_manager = None

        # Register all required components
        for comp in self.REQUIRED_COMPONENTS:
            self._components[comp.name] = ComponentInfo(
                name=comp.name,
                description=comp.description,
                module_path=comp.module_path,
                class_name=comp.class_name,
                phase=comp.phase,
                priority=comp.priority,
                required=comp.required
            )

    @property
    def event_bus(self):
        """Get the initialized EventBus instance."""
        return self._event_bus

    @property
    def trading_engine(self):
        """Get the initialized EventDrivenLiveEngine instance."""
        return self._trading_engine

    @property
    def infrastructure_manager(self):
        """Get the initialized InfrastructureManager instance."""
        return self._infrastructure_manager

    def _setup_watchlist_and_data_source(self):
        """
        Load watchlist configuration and setup data source for the engine.

        This enables dynamic symbol selection without hardcoding.
        """
        try:
            from config.loader import get_active_symbols, get_instrument_tokens

            symbols = get_active_symbols()
            instrument_tokens = get_instrument_tokens()

            if symbols:
                logger.info(f"        Loaded {len(symbols)} symbols from watchlist")

                # Create simulated data source for paper trading
                # Filter to symbols we have tokens for (or use all for simulation)
                self._trading_engine.create_simulated_source(
                    symbols=symbols[:10],  # Limit to first 10 for simulation
                    tick_interval=1.0
                )
                logger.info(f"        Simulated data source created for {min(len(symbols), 10)} symbols")

                # Store watchlist info for UI access
                self._watchlist_symbols = symbols
                self._instrument_tokens = instrument_tokens
            else:
                logger.warning("        No symbols in watchlist - using defaults")
                self._watchlist_symbols = ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"]
                self._instrument_tokens = {}

        except ImportError as e:
            logger.warning(f"        Could not load watchlist config: {e}")
            self._watchlist_symbols = ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"]
            self._instrument_tokens = {}

        except Exception as e:
            logger.error(f"        Watchlist setup error: {e}")
            self._watchlist_symbols = []
            self._instrument_tokens = {}

    @property
    def watchlist_symbols(self) -> list:
        """Get the loaded watchlist symbols."""
        return getattr(self, '_watchlist_symbols', [])

    def validate_imports(self) -> List[ComponentInfo]:
        """
        Validate that all required components can be imported.

        Returns:
            List of components that failed to import

        Raises:
            RuntimeError: If any required component fails to import
        """
        failed = []

        # Sort by priority
        sorted_components = sorted(
            self._components.values(),
            key=lambda c: c.priority
        )

        logger.info("=" * 60)
        logger.info("COMPONENT VALIDATION - Phase 8-12 Blueprint")
        logger.info("=" * 60)

        for comp in sorted_components:
            comp.status = ComponentStatus.LOADING
            start = time.perf_counter()

            try:
                # Attempt import
                module = __import__(comp.module_path, fromlist=[comp.class_name])
                cls = getattr(module, comp.class_name, None)

                if cls is None:
                    raise ImportError(f"Class '{comp.class_name}' not found in '{comp.module_path}'")

                # Store reference (not instance yet)
                comp.status = ComponentStatus.LOADED
                comp.load_time_ms = (time.perf_counter() - start) * 1000
                self._instances[comp.name] = cls

                logger.info(
                    f"  [{comp.phase}] {comp.name:20} - LOADED "
                    f"({comp.load_time_ms:.1f}ms)"
                )

            except Exception as e:
                comp.status = ComponentStatus.FAILED
                comp.error = str(e)
                comp.load_time_ms = (time.perf_counter() - start) * 1000

                if comp.required:
                    logger.error(
                        f"  [{comp.phase}] {comp.name:20} - FAILED: {e}"
                    )
                    failed.append(comp)
                else:
                    logger.warning(
                        f"  [{comp.phase}] {comp.name:20} - SKIPPED (optional): {e}"
                    )
                    comp.status = ComponentStatus.SKIPPED

        logger.info("=" * 60)

        if failed:
            logger.error(f"VALIDATION FAILED: {len(failed)} required component(s) missing")
            for comp in failed:
                logger.error(f"  - {comp.name} ({comp.phase}): {comp.error}")
            raise RuntimeError(
                f"Component validation failed. Missing: {[c.name for c in failed]}"
            )

        logger.info(f"VALIDATION PASSED: All {len(sorted_components)} components verified")
        return failed

    def initialize_runtime(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize runtime instances of core components.

        This creates the actual instances that will be used at runtime:
        - EventBus
        - EventDrivenLiveEngine
        - InfrastructureManager

        Args:
            config: Optional configuration dictionary

        Returns:
            True if initialization successful
        """
        config = config or {}
        self._start_time = datetime.now()

        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNTIME INITIALIZATION")
        logger.info("=" * 60)

        try:
            # 1. Create EventBus (foundation for everything)
            logger.info("  [1/4] Creating EventBus...")
            from core.events import EventBus
            self._event_bus = EventBus(async_mode=False)
            logger.info("        EventBus created (sync mode)")

            # 2. Create Trading Engine
            logger.info("  [2/4] Creating EventDrivenLiveEngine...")
            from core.trading_engine import EventDrivenLiveEngine, EngineConfig, TradingMode

            engine_config = EngineConfig(
                mode=TradingMode.PAPER,
                capital=config.get('initial_capital', 100000.0),
                position_size_pct=config.get('position_size_pct', 10.0),
                max_daily_loss_pct=config.get('max_daily_loss_pct', 5.0),
            )

            self._trading_engine = EventDrivenLiveEngine(
                event_bus=self._event_bus,
                broker=None,  # Paper trading
                config=engine_config
            )
            logger.info("        EventDrivenLiveEngine created (paper mode)")

            # Load watchlist configuration and setup data source
            self._setup_watchlist_and_data_source()

            # 3. Create and initialize Infrastructure Manager
            logger.info("  [3/4] Creating InfrastructureManager...")
            from core.infrastructure import (
                InfrastructureManager,
                InfrastructureConfig,
                initialize_infrastructure
            )

            infra_config = InfrastructureConfig(
                enable_recording=config.get('enable_recording', True),
                enable_shadow=config.get('enable_shadow', False),
                enable_ab_testing=config.get('enable_ab_testing', False),
                enable_audit=config.get('enable_audit', True),
                enable_compliance=config.get('enable_compliance', True),
                enable_kill_switch=config.get('enable_kill_switch', True),
                enable_latency_monitor=config.get('enable_latency_monitor', True),
                max_loss_threshold=config.get('max_loss_threshold', 50000.0),
                max_drawdown_pct=config.get('max_drawdown_pct', 10.0),
            )

            self._infrastructure_manager = InfrastructureManager(
                event_bus=self._event_bus,
                config=infra_config
            )

            if self._infrastructure_manager.initialize():
                self._infrastructure_manager.attach_trading_engine(self._trading_engine)
                logger.info("        InfrastructureManager initialized and attached")
            else:
                raise RuntimeError("InfrastructureManager initialization failed")

            # 4. Set global references
            logger.info("  [4/4] Setting global references...")
            from core.infrastructure import set_infrastructure_manager
            set_infrastructure_manager(self._infrastructure_manager)
            logger.info("        Global references set")

            self._initialized = True
            self._end_time = datetime.now()

            duration = (self._end_time - self._start_time).total_seconds()
            logger.info("=" * 60)
            logger.info(f"INITIALIZATION COMPLETE ({duration:.2f}s)")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"INITIALIZATION FAILED: {e}")
            self._initialized = False
            raise

    def start_services(self) -> bool:
        """
        Start all background services.

        This starts:
        - Infrastructure (recording, monitoring, etc.)
        - Does NOT start trading engine (that's manual)

        Returns:
            True if services started successfully
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize_runtime() first")

        logger.info("")
        logger.info("Starting background services...")

        # Start infrastructure services
        if self._infrastructure_manager:
            if self._infrastructure_manager.start():
                logger.info("  - InfrastructureManager: STARTED")
            else:
                logger.warning("  - InfrastructureManager: FAILED TO START")
                return False

        logger.info("Background services started")
        return True

    def stop_services(self):
        """Stop all background services."""
        logger.info("Stopping background services...")

        if self._trading_engine:
            self._trading_engine.stop()
            logger.info("  - TradingEngine: STOPPED")

        if self._infrastructure_manager:
            self._infrastructure_manager.stop()
            logger.info("  - InfrastructureManager: STOPPED")

        logger.info("Background services stopped")

    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive status report.

        Returns:
            Dict with status of all components
        """
        components_status = {}
        for name, comp in self._components.items():
            components_status[name] = {
                'phase': comp.phase,
                'status': comp.status.value,
                'required': comp.required,
                'error': comp.error,
                'load_time_ms': comp.load_time_ms
            }

        # Calculate phase-level summaries
        phase_summary = {}
        for comp in self._components.values():
            phase_num = comp.phase.split('.')[0]  # e.g., "8" from "8.1.1"
            if phase_num not in phase_summary:
                phase_summary[phase_num] = {'total': 0, 'loaded': 0, 'failed': 0, 'skipped': 0}
            phase_summary[phase_num]['total'] += 1
            if comp.status == ComponentStatus.LOADED:
                phase_summary[phase_num]['loaded'] += 1
            elif comp.status == ComponentStatus.FAILED:
                phase_summary[phase_num]['failed'] += 1
            elif comp.status == ComponentStatus.SKIPPED:
                phase_summary[phase_num]['skipped'] += 1

        return {
            'initialized': self._initialized,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'end_time': self._end_time.isoformat() if self._end_time else None,
            'components': components_status,
            'phase_summary': phase_summary,
            'total_components': len(self._components),
            'loaded': sum(1 for c in self._components.values() if c.status == ComponentStatus.LOADED),
            'failed': sum(1 for c in self._components.values() if c.status == ComponentStatus.FAILED),
            'skipped': sum(1 for c in self._components.values() if c.status == ComponentStatus.SKIPPED),
        }

    def print_status_report(self):
        """Print a formatted status report."""
        report = self.get_status_report()

        phase_names = {
            '8': 'Event-Driven Architecture',
            '9': 'Data Superiority',
            '10': 'Execution Alpha',
            '11': 'MLOps',
            '12': 'Infrastructure & Compliance / Guardian Angel',
        }

        print("\n" + "=" * 60)
        print("COMPONENT STATUS REPORT (Phase 8-12)")
        print("=" * 60)
        print(f"Initialized: {report['initialized']}")
        print(f"Total Components: {report['total_components']}")
        print(f"Loaded: {report['loaded']}")
        print(f"Failed: {report['failed']}")
        print(f"Skipped: {report['skipped']}")
        print("-" * 60)

        # Show phase summary
        print("\nPhase Summary:")
        for phase_num in sorted(report.get('phase_summary', {}).keys()):
            ps = report['phase_summary'][phase_num]
            phase_name = phase_names.get(phase_num, f'Phase {phase_num}')
            pct = (ps['loaded'] / ps['total'] * 100) if ps['total'] > 0 else 0
            print(f"  Phase {phase_num}: {phase_name}")
            print(f"          {ps['loaded']}/{ps['total']} loaded ({pct:.0f}%)")

        print("-" * 60)
        print("\nComponent Details:")

        # Group by phase
        current_phase = None
        sorted_items = sorted(report['components'].items(), key=lambda x: x[1]['phase'])

        for name, status in sorted_items:
            phase_num = status['phase'].split('.')[0]
            if phase_num != current_phase:
                current_phase = phase_num
                phase_name = phase_names.get(phase_num, f'Phase {phase_num}')
                print(f"\n  --- Phase {phase_num}: {phase_name} ---")

            status_str = status['status'].upper()
            if status['status'] == 'loaded':
                status_str = f"\033[92m{status_str}\033[0m"  # Green
            elif status['status'] == 'failed':
                status_str = f"\033[91m{status_str}\033[0m"  # Red
            elif status['status'] == 'skipped':
                status_str = f"\033[93m{status_str}\033[0m"  # Yellow

            required = "*" if status['required'] else " "
            print(f"    [{status['phase']:6}]{required} {name:25} {status_str}")
            if status['error']:
                print(f"               Error: {status['error']}")

        print("\n" + "=" * 60)
        print("* = Required component")
        print()


# ============================================================================
# Bootstrap Function
# ============================================================================

def bootstrap_application(config: Optional[Dict[str, Any]] = None) -> ComponentRegistry:
    """
    Bootstrap the entire application.

    This is the single entry point that:
    1. Validates all 12 components can be imported
    2. Initializes runtime instances
    3. Starts background services

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized ComponentRegistry

    Raises:
        RuntimeError: If any required component fails to load

    Example:
        >>> registry = bootstrap_application({'initial_capital': 200000})
        >>> engine = registry.trading_engine
        >>> engine.add_strategy(my_strategy, "RELIANCE")
        >>> engine.start()
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALGOTRADER PRO - BOOTSTRAP")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Create registry
    registry = ComponentRegistry()

    # Step 1: Validate all imports
    registry.validate_imports()

    # Step 2: Initialize runtime
    registry.initialize_runtime(config)

    # Step 3: Start services
    registry.start_services()

    logger.info("")
    logger.info("Application ready.")
    logger.info("=" * 60)

    return registry


def validate_components_only() -> bool:
    """
    Validate components without starting the application.

    Useful for CI/CD pipelines to verify all components are importable.

    Returns:
        True if all required components are valid

    Raises:
        RuntimeError: If any required component fails
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    registry = ComponentRegistry()
    registry.validate_imports()
    return True


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ComponentRegistry',
    'ComponentInfo',
    'ComponentStatus',
    'bootstrap_application',
    'validate_components_only',
]
