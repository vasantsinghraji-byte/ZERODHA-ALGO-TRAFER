# -*- coding: utf-8 -*-
"""
Infrastructure Integration - Wiring Everything Together.

This module provides a unified InfrastructureManager that initializes and
coordinates all infrastructure components with the trading engine:

- Flight Recorder: Records all ticks and events for replay
- Shadow Mode: Runs paper strategies in parallel with live
- A/B Testing: Routes traffic between strategy variants
- Audit Trail: Maintains immutable trade log with hash chain
- Risk Compliance: Enforces SEBI regulations and position limits
- Kill Switch: Emergency stop capability
- Latency Monitor: Tracks system latency

Usage:
    from core.infrastructure.integration import (
        InfrastructureManager,
        get_infrastructure_manager,
        initialize_infrastructure
    )

    # Initialize with event bus
    manager = InfrastructureManager(event_bus)
    manager.initialize()
    manager.start()

    # Or use convenience function
    manager = initialize_infrastructure(event_bus, trading_engine)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure components."""

    # Flight Recorder
    enable_recording: bool = True
    recording_dir: str = "recordings"
    compress_recordings: bool = True

    # Shadow Mode
    enable_shadow: bool = False
    shadow_capital: float = 100000.0

    # A/B Testing
    enable_ab_testing: bool = False
    default_traffic_control: float = 0.1  # 10% to treatment

    # Audit Trail
    enable_audit: bool = True
    audit_dir: str = "audit_logs"

    # Risk Compliance
    enable_compliance: bool = True
    max_position_value: float = 500000.0
    max_daily_loss_pct: float = 5.0

    # Kill Switch
    enable_kill_switch: bool = True
    max_loss_threshold: float = 50000.0
    max_drawdown_pct: float = 10.0

    # Latency Monitoring
    enable_latency_monitor: bool = True
    latency_warn_ms: float = 100.0
    latency_critical_ms: float = 500.0


class InfrastructureStatus(Enum):
    """Status of infrastructure manager."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# Infrastructure Manager
# ============================================================================

class InfrastructureManager:
    """
    Unified manager for all infrastructure components.

    Coordinates initialization, event subscriptions, and lifecycle
    management for:
    - Flight Recorder (market replay)
    - Shadow Mode (paper trading validation)
    - A/B Testing (strategy comparison)
    - Audit Trail (compliance logging)
    - Risk Compliance (SEBI regulations)
    - Kill Switch (emergency stop)
    - Latency Monitor (performance tracking)

    Example:
        >>> from core.events import EventBus
        >>> from core.infrastructure.integration import InfrastructureManager
        >>>
        >>> bus = EventBus()
        >>> manager = InfrastructureManager(bus)
        >>> manager.initialize()
        >>> manager.start()
        >>>
        >>> # Get component status
        >>> status = manager.get_status()
        >>> print(f"Recording: {status['flight_recorder']['recording']}")
    """

    def __init__(
        self,
        event_bus=None,
        config: Optional[InfrastructureConfig] = None
    ):
        """
        Initialize infrastructure manager.

        Args:
            event_bus: EventBus instance (created if not provided)
            config: Infrastructure configuration
        """
        self.config = config or InfrastructureConfig()
        self._event_bus = event_bus

        # Components (lazy initialized)
        self._flight_recorder = None
        self._shadow_engine = None
        self._ab_framework = None
        self._audit_trail = None
        self._compliance_engine = None
        self._kill_switch = None
        self._latency_monitor = None

        # State
        self._status = InfrastructureStatus.UNINITIALIZED
        self._lock = threading.RLock()
        self._handler_names: List[str] = []

        # Trading engine reference
        self._trading_engine = None

        # Callbacks
        self._on_kill_switch_triggered: Optional[Callable] = None
        self._on_compliance_violation: Optional[Callable] = None
        self._on_latency_alert: Optional[Callable] = None

        logger.info("InfrastructureManager created")

    @property
    def event_bus(self):
        """Get or create event bus."""
        if self._event_bus is None:
            from core.events import EventBus
            self._event_bus = EventBus()
        return self._event_bus

    @property
    def status(self) -> InfrastructureStatus:
        """Get current status."""
        with self._lock:
            return self._status

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize all infrastructure components.

        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._status not in (InfrastructureStatus.UNINITIALIZED,
                                     InfrastructureStatus.STOPPED):
                logger.warning(f"Cannot initialize from status: {self._status}")
                return False

            self._status = InfrastructureStatus.INITIALIZING

        try:
            # Initialize components based on config
            if self.config.enable_recording:
                self._init_flight_recorder()

            if self.config.enable_shadow:
                self._init_shadow_engine()

            if self.config.enable_ab_testing:
                self._init_ab_framework()

            if self.config.enable_audit:
                self._init_audit_trail()

            if self.config.enable_compliance:
                self._init_compliance_engine()

            if self.config.enable_kill_switch:
                self._init_kill_switch()

            if self.config.enable_latency_monitor:
                self._init_latency_monitor()

            with self._lock:
                self._status = InfrastructureStatus.READY

            logger.info("Infrastructure initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Infrastructure initialization failed: {e}")
            with self._lock:
                self._status = InfrastructureStatus.ERROR
            return False

    def _init_flight_recorder(self):
        """Initialize flight recorder."""
        from core.infrastructure.flight_recorder import (
            FlightRecorder,
            CompressionType
        )

        compression = CompressionType.LZ4 if self.config.compress_recordings else CompressionType.NONE
        self._flight_recorder = FlightRecorder(
            base_dir=self.config.recording_dir,
            compression=compression
        )
        logger.debug("Flight recorder initialized")

    def _init_shadow_engine(self):
        """Initialize shadow trading engine."""
        from core.infrastructure.shadow_mode import ShadowEngine

        self._shadow_engine = ShadowEngine(
            initial_capital=self.config.shadow_capital
        )
        logger.debug("Shadow engine initialized")

    def _init_ab_framework(self):
        """Initialize A/B testing framework."""
        from core.infrastructure.ab_testing import ABTestFramework

        self._ab_framework = ABTestFramework()
        logger.debug("A/B testing framework initialized")

    def _init_audit_trail(self):
        """Initialize audit trail."""
        from core.infrastructure.audit_trail import (
            AuditTrail,
            FileAuditStorage
        )

        storage = FileAuditStorage(base_dir=self.config.audit_dir)
        self._audit_trail = AuditTrail(storage=storage)
        logger.debug("Audit trail initialized")

    def _init_compliance_engine(self):
        """Initialize compliance engine."""
        from core.infrastructure.risk_compliance import ComplianceEngine

        self._compliance_engine = ComplianceEngine(
            max_order_value=self.config.max_position_value,
            max_daily_turnover=self.config.max_position_value * 10,
            enable_market_hours_check=True,
            enable_circuit_breaker_check=True,
            enable_sebi_rules=True
        )
        logger.debug("Compliance engine initialized")

    def _init_kill_switch(self):
        """Initialize kill switch."""
        from core.infrastructure.kill_switch import (
            KillSwitch,
            KillSwitchConfig
        )

        config = KillSwitchConfig(
            max_loss=self.config.max_loss_threshold,
            max_drawdown_pct=self.config.max_drawdown_pct
        )
        self._kill_switch = KillSwitch(config=config)
        logger.debug("Kill switch initialized")

    def _init_latency_monitor(self):
        """Initialize latency monitor."""
        from core.infrastructure.latency_monitor import (
            LatencyMonitor,
            LatencyThresholds
        )

        thresholds = LatencyThresholds(
            warn_ms=self.config.latency_warn_ms,
            critical_ms=self.config.latency_critical_ms
        )
        self._latency_monitor = LatencyMonitor(thresholds=thresholds)
        logger.debug("Latency monitor initialized")

    # =========================================================================
    # Start/Stop
    # =========================================================================

    def start(self) -> bool:
        """
        Start infrastructure components and subscribe to events.

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._status != InfrastructureStatus.READY:
                logger.warning(f"Cannot start from status: {self._status}")
                return False

        try:
            # Subscribe to events
            self._subscribe_events()

            # Start recording if enabled
            if self._flight_recorder:
                session = self._flight_recorder.start_recording()
                logger.info(f"Recording started: {session.session_id}")

            with self._lock:
                self._status = InfrastructureStatus.RUNNING

            logger.info("Infrastructure started")
            return True

        except Exception as e:
            logger.error(f"Infrastructure start failed: {e}")
            with self._lock:
                self._status = InfrastructureStatus.ERROR
            return False

    def stop(self) -> bool:
        """
        Stop infrastructure components and unsubscribe from events.

        Returns:
            True if stopped successfully
        """
        try:
            # Unsubscribe from events
            self._unsubscribe_events()

            # Stop recording
            if self._flight_recorder:
                self._flight_recorder.stop_recording()

            with self._lock:
                self._status = InfrastructureStatus.STOPPED

            logger.info("Infrastructure stopped")
            return True

        except Exception as e:
            logger.error(f"Infrastructure stop failed: {e}")
            return False

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _subscribe_events(self):
        """Subscribe to trading events."""
        from core.events import EventType

        bus = self.event_bus
        self._handler_names = []

        # Tick events - for recording and latency
        name = bus.subscribe(
            EventType.TICK,
            self._on_tick_event,
            priority=10,  # High priority for recording
            name="infra_tick"
        )
        self._handler_names.append(name)

        # Bar events - for recording
        name = bus.subscribe(
            EventType.BAR,
            self._on_bar_event,
            priority=10,
            name="infra_bar"
        )
        self._handler_names.append(name)

        # Order events - for audit and compliance
        name = bus.subscribe(
            EventType.ORDER_SUBMITTED,
            self._on_order_submitted,
            priority=5,  # Very high - compliance check
            name="infra_order_submit"
        )
        self._handler_names.append(name)

        # Fill events - for audit and shadow
        name = bus.subscribe(
            EventType.ORDER_FILLED,
            self._on_order_filled,
            priority=20,
            name="infra_fill"
        )
        self._handler_names.append(name)

        # Position events - for compliance and kill switch
        name = bus.subscribe(
            EventType.POSITION_OPENED,
            self._on_position_opened,
            priority=30,
            name="infra_pos_open"
        )
        self._handler_names.append(name)

        name = bus.subscribe(
            EventType.POSITION_CLOSED,
            self._on_position_closed,
            priority=30,
            name="infra_pos_close"
        )
        self._handler_names.append(name)

        # Signal events - for A/B testing
        name = bus.subscribe(
            EventType.SIGNAL_GENERATED,
            self._on_signal_generated,
            priority=15,
            name="infra_signal"
        )
        self._handler_names.append(name)

        logger.debug(f"Subscribed to {len(self._handler_names)} event types")

    def _unsubscribe_events(self):
        """Unsubscribe from all events."""
        bus = self.event_bus
        for name in self._handler_names:
            bus.unsubscribe(name)
        self._handler_names.clear()

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_tick_event(self, event):
        """Handle tick events."""
        from core.events.events import TickEvent

        if not isinstance(event, TickEvent):
            return

        # Record tick
        if self._flight_recorder:
            self._flight_recorder.record_tick(
                timestamp=event.timestamp,
                symbol=event.symbol,
                price=event.last_price,
                volume=event.volume,
                bid=getattr(event, 'bid', 0.0),
                ask=getattr(event, 'ask', 0.0)
            )

        # Track latency
        if self._latency_monitor:
            latency_ms = (datetime.now() - event.timestamp).total_seconds() * 1000
            self._latency_monitor.record("tick_processing", latency_ms)

        # Forward to shadow engine
        if self._shadow_engine:
            self._shadow_engine.on_tick(
                symbol=event.symbol,
                price=event.last_price,
                timestamp=event.timestamp
            )

    def _on_bar_event(self, event):
        """Handle bar events."""
        from core.events.events import BarEvent

        if not isinstance(event, BarEvent):
            return

        # Record bar as event
        if self._flight_recorder:
            self._flight_recorder.record_event(
                event_type="bar",
                data={
                    "symbol": event.symbol,
                    "open": event.open,
                    "high": event.high,
                    "low": event.low,
                    "close": event.close,
                    "volume": event.volume,
                    "timeframe": event.timeframe
                },
                timestamp=event.timestamp
            )

    def _on_order_submitted(self, event):
        """Handle order submission - compliance check."""
        from core.events.events import OrderEvent

        if not isinstance(event, OrderEvent):
            return

        # Compliance check
        if self._compliance_engine:
            from core.infrastructure.risk_compliance import ComplianceAction

            result = self._compliance_engine.check_order_compliance(
                symbol=event.symbol,
                side=event.side.value.lower(),
                quantity=event.quantity,
                price=event.price
            )

            if result.action == ComplianceAction.BLOCK:
                logger.warning(f"Order BLOCKED by compliance: {result.reason}")

                if self._on_compliance_violation:
                    self._on_compliance_violation(result)

                # Could emit rejection event here
                return

            elif result.action == ComplianceAction.REDUCE:
                logger.warning(f"Order REDUCED by compliance: {result.reason}")

        # Audit logging
        if self._audit_trail:
            from core.infrastructure.audit_trail import AuditEventType

            self._audit_trail.log_event(
                event_type=AuditEventType.ORDER_SUBMITTED,
                data={
                    "order_id": event.order_id,
                    "symbol": event.symbol,
                    "side": event.side.value,
                    "quantity": event.quantity,
                    "price": event.price,
                    "strategy": event.strategy_name
                }
            )

    def _on_order_filled(self, event):
        """Handle order fills - audit and shadow."""
        from core.events.events import FillEvent

        if not isinstance(event, FillEvent):
            return

        # Audit logging
        if self._audit_trail:
            from core.infrastructure.audit_trail import AuditEventType

            self._audit_trail.log_event(
                event_type=AuditEventType.ORDER_FILLED,
                data={
                    "order_id": event.order_id,
                    "symbol": event.symbol,
                    "side": event.side.value,
                    "quantity": event.quantity,
                    "price": event.price,
                    "strategy": event.strategy_name
                }
            )

        # Record in flight recorder
        if self._flight_recorder:
            self._flight_recorder.record_event(
                event_type="fill",
                data={
                    "order_id": event.order_id,
                    "symbol": event.symbol,
                    "side": event.side.value,
                    "quantity": event.quantity,
                    "price": event.price
                },
                timestamp=event.timestamp
            )

        # Update kill switch P&L tracking
        if self._kill_switch and hasattr(event, 'realized_pnl'):
            self._kill_switch.update_pnl(event.realized_pnl)

            if self._kill_switch.is_triggered:
                logger.critical("KILL SWITCH TRIGGERED!")
                if self._on_kill_switch_triggered:
                    self._on_kill_switch_triggered(self._kill_switch.state)

    def _on_position_opened(self, event):
        """Handle position opened."""
        from core.events.events import PositionEvent

        if not isinstance(event, PositionEvent):
            return

        # Track position in compliance
        if self._compliance_engine:
            self._compliance_engine.position_tracker.update_position(
                symbol=event.symbol,
                quantity=event.quantity,
                value=event.quantity * event.entry_price
            )

        # Audit logging
        if self._audit_trail:
            from core.infrastructure.audit_trail import AuditEventType

            self._audit_trail.log_event(
                event_type=AuditEventType.POSITION_OPENED,
                data={
                    "symbol": event.symbol,
                    "quantity": event.quantity,
                    "entry_price": event.entry_price,
                    "stop_loss": event.stop_loss,
                    "target": event.target,
                    "strategy": event.strategy_name
                }
            )

    def _on_position_closed(self, event):
        """Handle position closed."""
        from core.events.events import PositionEvent

        if not isinstance(event, PositionEvent):
            return

        pnl = getattr(event, 'realized_pnl', 0.0)

        # Update compliance tracking
        if self._compliance_engine:
            self._compliance_engine.position_tracker.close_position(event.symbol)

        # Update kill switch
        if self._kill_switch:
            self._kill_switch.update_pnl(pnl)

            if self._kill_switch.is_triggered:
                logger.critical("KILL SWITCH TRIGGERED!")
                if self._on_kill_switch_triggered:
                    self._on_kill_switch_triggered(self._kill_switch.state)

        # Audit logging
        if self._audit_trail:
            from core.infrastructure.audit_trail import AuditEventType

            self._audit_trail.log_event(
                event_type=AuditEventType.POSITION_CLOSED,
                data={
                    "symbol": event.symbol,
                    "quantity": event.quantity,
                    "entry_price": event.entry_price,
                    "exit_price": event.current_price,
                    "realized_pnl": pnl,
                    "strategy": event.strategy_name
                }
            )

    def _on_signal_generated(self, event):
        """Handle signal events - A/B testing routing."""
        from core.events.events import SignalEvent

        if not isinstance(event, SignalEvent):
            return

        # A/B test routing
        if self._ab_framework:
            # Get variant for this strategy
            variant = self._ab_framework.get_variant(
                test_id=f"strategy_{event.strategy_name}",
                user_id=event.symbol  # Use symbol as user for consistent routing
            )

            if variant:
                # Record outcome metric
                self._ab_framework.record_metric(
                    test_id=f"strategy_{event.strategy_name}",
                    metric_name="signals_generated",
                    value=1.0,
                    variant=variant
                )

        # Record in flight recorder
        if self._flight_recorder:
            self._flight_recorder.record_event(
                event_type="signal",
                data={
                    "strategy": event.strategy_name,
                    "symbol": event.symbol,
                    "signal_type": event.signal_type.value,
                    "price": event.price,
                    "confidence": event.confidence
                },
                timestamp=event.timestamp
            )

    # =========================================================================
    # Trading Engine Integration
    # =========================================================================

    def attach_trading_engine(self, engine):
        """
        Attach to a trading engine instance.

        Args:
            engine: TradingEngine or EventDrivenLiveEngine instance
        """
        self._trading_engine = engine

        # Set up kill switch callback
        if self._kill_switch:
            def on_kill_switch(state):
                logger.critical("Kill switch triggered - stopping engine!")
                if self._trading_engine:
                    self._trading_engine.stop()
                if self._on_kill_switch_triggered:
                    self._on_kill_switch_triggered(state)

            self._on_kill_switch_triggered = on_kill_switch

        logger.info("Attached to trading engine")

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def flight_recorder(self):
        """Get flight recorder instance."""
        return self._flight_recorder

    @property
    def shadow_engine(self):
        """Get shadow engine instance."""
        return self._shadow_engine

    @property
    def ab_framework(self):
        """Get A/B testing framework."""
        return self._ab_framework

    @property
    def audit_trail(self):
        """Get audit trail instance."""
        return self._audit_trail

    @property
    def compliance_engine(self):
        """Get compliance engine instance."""
        return self._compliance_engine

    @property
    def kill_switch(self):
        """Get kill switch instance."""
        return self._kill_switch

    @property
    def latency_monitor(self):
        """Get latency monitor instance."""
        return self._latency_monitor

    # =========================================================================
    # Status & Reporting
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all infrastructure components.

        Returns:
            Dict with status of each component
        """
        status = {
            "status": self._status.value,
            "timestamp": datetime.now().isoformat()
        }

        if self._flight_recorder:
            session = self._flight_recorder.current_session
            status["flight_recorder"] = {
                "enabled": True,
                "recording": session is not None,
                "session_id": session.session_id if session else None,
                "tick_count": self._flight_recorder.tick_recorder.tick_count if session else 0
            }
        else:
            status["flight_recorder"] = {"enabled": False}

        if self._shadow_engine:
            metrics = self._shadow_engine.pnl_tracker.get_metrics()
            status["shadow_engine"] = {
                "enabled": True,
                "strategies": len(self._shadow_engine._strategies),
                "total_pnl": metrics.total_pnl,
                "total_trades": metrics.total_trades
            }
        else:
            status["shadow_engine"] = {"enabled": False}

        if self._ab_framework:
            status["ab_testing"] = {
                "enabled": True,
                "active_tests": len(self._ab_framework.list_tests())
            }
        else:
            status["ab_testing"] = {"enabled": False}

        if self._audit_trail:
            status["audit_trail"] = {
                "enabled": True,
                "record_count": self._audit_trail.record_count,
                "integrity_valid": self._audit_trail.verify_integrity()
            }
        else:
            status["audit_trail"] = {"enabled": False}

        if self._compliance_engine:
            status["compliance"] = {
                "enabled": True,
                "checks_performed": 0,  # Compliance engine doesn't track this
                "violations": 0
            }
        else:
            status["compliance"] = {"enabled": False}

        if self._kill_switch:
            state = self._kill_switch.state
            status["kill_switch"] = {
                "enabled": True,
                "triggered": self._kill_switch.is_triggered,
                "current_pnl": state.current_pnl,
                "current_drawdown": state.current_drawdown_pct
            }
        else:
            status["kill_switch"] = {"enabled": False}

        if self._latency_monitor:
            stats = self._latency_monitor.get_stats()
            status["latency_monitor"] = {
                "enabled": True,
                "avg_latency_ms": stats.get("tick_processing", {}).get("avg", 0),
                "max_latency_ms": stats.get("tick_processing", {}).get("max", 0)
            }
        else:
            status["latency_monitor"] = {"enabled": False}

        return status

    def get_audit_report(self, start_time: datetime = None, end_time: datetime = None) -> Dict:
        """Generate audit report for time range."""
        if not self._audit_trail:
            return {"error": "Audit trail not enabled"}

        return self._audit_trail.generate_report(
            start_time=start_time,
            end_time=end_time
        )

    def get_compliance_report(self) -> Dict:
        """Generate compliance report."""
        if not self._compliance_engine:
            return {"error": "Compliance engine not enabled"}

        return self._compliance_engine.generate_report()


# ============================================================================
# Global Instance
# ============================================================================

_global_manager: Optional[InfrastructureManager] = None
_global_lock = threading.Lock()


def get_infrastructure_manager() -> Optional[InfrastructureManager]:
    """Get the global infrastructure manager instance."""
    return _global_manager


def set_infrastructure_manager(manager: InfrastructureManager):
    """Set the global infrastructure manager instance."""
    global _global_manager
    with _global_lock:
        _global_manager = manager


def initialize_infrastructure(
    event_bus=None,
    trading_engine=None,
    config: Optional[InfrastructureConfig] = None
) -> InfrastructureManager:
    """
    Convenience function to initialize and start infrastructure.

    Args:
        event_bus: EventBus instance
        trading_engine: Trading engine to attach
        config: Infrastructure configuration

    Returns:
        Initialized and running InfrastructureManager

    Example:
        >>> from core.events import EventBus
        >>> from core.trading_engine import EventDrivenLiveEngine
        >>>
        >>> bus = EventBus()
        >>> engine = EventDrivenLiveEngine(bus)
        >>>
        >>> manager = initialize_infrastructure(bus, engine)
        >>> # Infrastructure is now running and integrated
    """
    manager = InfrastructureManager(event_bus, config)

    if manager.initialize():
        if trading_engine:
            manager.attach_trading_engine(trading_engine)

        manager.start()
        set_infrastructure_manager(manager)

        return manager
    else:
        raise RuntimeError("Infrastructure initialization failed")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'InfrastructureManager',
    'InfrastructureConfig',
    'InfrastructureStatus',
    'get_infrastructure_manager',
    'set_infrastructure_manager',
    'initialize_infrastructure',
]
