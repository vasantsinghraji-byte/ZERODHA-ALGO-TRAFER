"""
Kill Switch (Panic Button) for Trading System
Emergency stop mechanism to halt all trading activity immediately.

Provides multiple trigger mechanisms:
- Manual trigger via API call
- Keyboard shortcut (Ctrl+Shift+K)
- Automatic drawdown breach detection
- EventBus integration for system-wide shutdown
"""

import atexit
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    """Kill switch states"""
    ARMED = "armed"           # Ready to trigger
    TRIGGERED = "triggered"   # Emergency stop active
    DISARMED = "disarmed"     # Not active


class TriggerReason(Enum):
    """Reasons for kill switch activation"""
    MANUAL = "manual"                     # User triggered
    KEYBOARD = "keyboard"                 # Keyboard shortcut
    MAX_DRAWDOWN = "max_drawdown"         # Drawdown limit breach
    MAX_LOSS = "max_loss"                 # Daily loss limit
    API_ERROR = "api_error"               # Critical API failure
    CONNECTION_LOST = "connection_lost"   # Broker disconnected
    EXTERNAL = "external"                 # External system trigger


@dataclass
class KillSwitchEvent:
    """Record of a kill switch activation"""
    timestamp: datetime
    reason: TriggerReason
    message: str
    orders_cancelled: int = 0
    positions_closed: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KillSwitchConfig:
    """Configuration for the kill switch"""
    # Drawdown triggers
    max_drawdown_percent: float = 5.0     # Max portfolio drawdown before trigger
    max_daily_loss: float = 10000.0       # Max daily loss in currency

    # Behavior options
    close_positions_on_trigger: bool = False   # Close all positions at market
    cancel_orders_on_trigger: bool = True      # Cancel all open orders
    disable_eventbus_on_trigger: bool = True   # Stop event processing

    # Keyboard shortcut
    enable_keyboard_trigger: bool = True
    keyboard_shortcut: str = "ctrl+shift+k"   # Shortcut to trigger

    # Monitoring
    monitor_interval_seconds: float = 1.0      # How often to check triggers

    # Recovery
    require_manual_reset: bool = True          # Must manually reset after trigger
    cooldown_seconds: float = 60.0             # Min time before re-arm


class KillSwitch:
    """
    Emergency stop mechanism for the trading system.

    The Kill Switch provides a "panic button" to immediately halt all trading
    activity when things go wrong. It bypasses normal event processing for
    instant response.

    Features:
    - Cancel all open orders immediately
    - Optionally close all positions at market
    - Disable EventBus to stop new order flow
    - Multiple trigger mechanisms (manual, keyboard, drawdown)
    - Thread-safe, designed for concurrent access
    - Integrates with EventBus for system-wide alerts

    Example:
        kill_switch = KillSwitch(
            broker=broker,
            order_manager=order_manager,
            position_manager=position_manager
        )
        kill_switch.arm()

        # Manual trigger
        kill_switch.trigger(TriggerReason.MANUAL, "User panic")

        # Or via keyboard: Ctrl+Shift+K

        # Check status
        if kill_switch.is_triggered:
            print("EMERGENCY STOP ACTIVE")
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        broker: Optional[Any] = None,
        order_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        event_bus: Optional[Any] = None
    ):
        """
        Args:
            config: Kill switch configuration
            broker: ZerodhaBroker instance
            order_manager: OrderManager instance
            position_manager: PositionManager instance
            event_bus: EventBus instance (uses global if None)
        """
        self._config = config or KillSwitchConfig()
        self._broker = broker
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._event_bus = event_bus

        # State
        self._state = KillSwitchState.DISARMED
        self._state_lock = threading.RLock()
        self._trigger_time: Optional[datetime] = None
        self._trigger_reason: Optional[TriggerReason] = None

        # History
        self._events: List[KillSwitchEvent] = []
        self._events_lock = threading.Lock()

        # Monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._peak_value: float = 0.0
        self._daily_start_value: float = 0.0
        self._daily_start_time: Optional[datetime] = None

        # Keyboard listener
        self._keyboard_thread: Optional[threading.Thread] = None
        self._keyboard_running = False

        # Callbacks
        self._callbacks: List[Callable[[KillSwitchEvent], None]] = []

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info("KillSwitch initialized")

    def _get_event_bus(self):
        """Get EventBus instance"""
        if self._event_bus is not None:
            return self._event_bus
        try:
            from core.events.event_bus import get_event_bus
            return get_event_bus()
        except ImportError:
            return None

    # =========================================================================
    # State Management
    # =========================================================================

    @property
    def state(self) -> KillSwitchState:
        """Current kill switch state"""
        with self._state_lock:
            return self._state

    @property
    def is_armed(self) -> bool:
        """Check if kill switch is armed"""
        return self.state == KillSwitchState.ARMED

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch has been triggered"""
        return self.state == KillSwitchState.TRIGGERED

    @property
    def trigger_reason(self) -> Optional[TriggerReason]:
        """Get the reason for trigger"""
        with self._state_lock:
            return self._trigger_reason

    def arm(self) -> bool:
        """
        Arm the kill switch.

        Returns:
            True if armed successfully
        """
        with self._state_lock:
            if self._state == KillSwitchState.TRIGGERED:
                if self._config.require_manual_reset:
                    logger.warning("Cannot arm: requires manual reset after trigger")
                    return False

                # Check cooldown
                if self._trigger_time:
                    elapsed = (datetime.now() - self._trigger_time).total_seconds()
                    if elapsed < self._config.cooldown_seconds:
                        logger.warning(
                            f"Cannot arm: cooldown active ({self._config.cooldown_seconds - elapsed:.0f}s remaining)"
                        )
                        return False

            self._state = KillSwitchState.ARMED
            self._trigger_reason = None

        # Start monitoring
        self._start_monitoring()

        # Start keyboard listener
        if self._config.enable_keyboard_trigger:
            self._start_keyboard_listener()

        # Initialize daily tracking
        self._reset_daily_tracking()

        logger.info("Kill switch ARMED")
        return True

    def disarm(self) -> bool:
        """
        Disarm the kill switch.

        Returns:
            True if disarmed successfully
        """
        with self._state_lock:
            self._state = KillSwitchState.DISARMED
            self._trigger_reason = None

        # Stop monitoring
        self._stop_monitoring()
        self._stop_keyboard_listener()

        logger.info("Kill switch DISARMED")
        return True

    def reset(self) -> bool:
        """
        Reset kill switch after trigger.

        Returns:
            True if reset successfully
        """
        with self._state_lock:
            if self._state != KillSwitchState.TRIGGERED:
                logger.warning("Cannot reset: not in triggered state")
                return False

            # Check cooldown
            if self._trigger_time:
                elapsed = (datetime.now() - self._trigger_time).total_seconds()
                if elapsed < self._config.cooldown_seconds:
                    logger.warning(
                        f"Cannot reset: cooldown active ({self._config.cooldown_seconds - elapsed:.0f}s remaining)"
                    )
                    return False

            self._state = KillSwitchState.DISARMED
            self._trigger_reason = None
            self._trigger_time = None

        logger.info("Kill switch RESET")
        return True

    # =========================================================================
    # Trigger Mechanism
    # =========================================================================

    def trigger(
        self,
        reason: TriggerReason = TriggerReason.MANUAL,
        message: str = "",
        context: Optional[Dict] = None
    ) -> KillSwitchEvent:
        """
        TRIGGER THE KILL SWITCH - EMERGENCY STOP!

        This method bypasses normal event processing for immediate effect.
        All operations are performed synchronously for fastest response.

        Args:
            reason: Why the kill switch was triggered
            message: Human-readable description
            context: Additional context data

        Returns:
            KillSwitchEvent with details of actions taken
        """
        trigger_start = time.perf_counter()

        with self._state_lock:
            if self._state == KillSwitchState.TRIGGERED:
                logger.warning("Kill switch already triggered")
                # Return last event
                with self._events_lock:
                    return self._events[-1] if self._events else KillSwitchEvent(
                        timestamp=datetime.now(),
                        reason=reason,
                        message="Already triggered"
                    )

            self._state = KillSwitchState.TRIGGERED
            self._trigger_time = datetime.now()
            self._trigger_reason = reason

        # Log immediately
        logger.critical(f"🚨 KILL SWITCH TRIGGERED: {reason.value} - {message}")

        # Create event record
        event = KillSwitchEvent(
            timestamp=datetime.now(),
            reason=reason,
            message=message or f"Kill switch triggered: {reason.value}",
            context=context or {}
        )

        # Execute emergency procedures (bypassing event queue)
        try:
            # 1. Disable EventBus FIRST to stop new orders
            if self._config.disable_eventbus_on_trigger:
                self._disable_event_bus()

            # 2. Cancel all open orders
            if self._config.cancel_orders_on_trigger:
                event.orders_cancelled = self._cancel_all_orders()

            # 3. Optionally close all positions
            if self._config.close_positions_on_trigger:
                event.positions_closed = self._close_all_positions()

        except Exception as e:
            logger.error(f"Error during kill switch execution: {e}", exc_info=True)
            event.context['execution_error'] = str(e)

        # Record timing
        elapsed_ms = (time.perf_counter() - trigger_start) * 1000
        event.context['execution_time_ms'] = elapsed_ms

        # Store event
        with self._events_lock:
            self._events.append(event)

        # Publish to EventBus (if still available)
        self._publish_kill_switch_event(event)

        # Call registered callbacks
        self._notify_callbacks(event)

        logger.critical(
            f"🚨 EMERGENCY STOP COMPLETE in {elapsed_ms:.1f}ms - "
            f"Orders cancelled: {event.orders_cancelled}, "
            f"Positions closed: {event.positions_closed}"
        )

        return event

    def _disable_event_bus(self) -> None:
        """Disable EventBus processing"""
        event_bus = self._get_event_bus()
        if event_bus:
            try:
                event_bus.pause()
                logger.info("EventBus paused")
            except Exception as e:
                logger.error(f"Failed to pause EventBus: {e}")

    def _cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of orders cancelled."""
        cancelled = 0

        # Via OrderManager
        if self._order_manager:
            try:
                open_orders = self._order_manager.get_open_orders()
                for order in open_orders:
                    try:
                        if self._order_manager.cancel_order(order.id):
                            cancelled += 1
                            logger.info(f"Cancelled order: {order.id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order.id}: {e}")
            except Exception as e:
                logger.error(f"Failed to get open orders: {e}")

        # Also try direct broker cancellation
        if self._broker and hasattr(self._broker, 'get_orders'):
            try:
                broker_orders = self._broker.get_orders()
                for order in broker_orders:
                    if order.status in ['OPEN', 'PENDING', 'TRIGGER PENDING']:
                        try:
                            if self._broker.cancel_order(order.order_id):
                                cancelled += 1
                                logger.info(f"Cancelled broker order: {order.order_id}")
                        except Exception as e:
                            logger.error(f"Failed to cancel broker order: {e}")
            except Exception as e:
                logger.error(f"Failed to get broker orders: {e}")

        return cancelled

    def _close_all_positions(self) -> int:
        """Close all positions at market. Returns count of positions closed."""
        closed = 0

        if not self._position_manager:
            logger.warning("No position manager - cannot close positions")
            return 0

        try:
            positions = self._position_manager.get_all_positions()

            for pos in positions:
                try:
                    if pos.quantity > 0:
                        # Close long position by selling
                        if self._broker:
                            order_id = self._broker.sell(
                                symbol=pos.symbol,
                                quantity=pos.quantity,
                                exchange=pos.exchange
                            )
                            if order_id:
                                closed += 1
                                logger.info(f"Closed position: {pos.symbol} (sold {pos.quantity})")
                        else:
                            # Paper trading close
                            self._position_manager.close_position(pos.symbol, pos.last_price)
                            closed += 1

                    elif pos.quantity < 0:
                        # Close short position by buying
                        if self._broker:
                            order_id = self._broker.buy(
                                symbol=pos.symbol,
                                quantity=abs(pos.quantity),
                                exchange=pos.exchange
                            )
                            if order_id:
                                closed += 1
                                logger.info(f"Closed position: {pos.symbol} (bought {abs(pos.quantity)})")

                except Exception as e:
                    logger.error(f"Failed to close position {pos.symbol}: {e}")

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")

        return closed

    def _publish_kill_switch_event(self, event: KillSwitchEvent) -> None:
        """Publish kill switch event to EventBus"""
        event_bus = self._get_event_bus()
        if not event_bus:
            return

        try:
            from core.events.events import ErrorEvent, EventType

            error_event = ErrorEvent(
                event_type=EventType.ERROR,
                error_code="KILL_SWITCH_TRIGGERED",
                message=event.message,
                data={
                    'reason': event.reason.value,
                    'orders_cancelled': event.orders_cancelled,
                    'positions_closed': event.positions_closed,
                    'context': event.context
                },
                source="kill_switch"
            )

            # Force publish even if paused (bypass the pause)
            event_bus._deliver(error_event)

        except Exception as e:
            logger.error(f"Failed to publish kill switch event: {e}")

    # =========================================================================
    # Monitoring
    # =========================================================================

    def _start_monitoring(self) -> None:
        """Start the drawdown monitoring thread"""
        if self._monitor_running:
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="KillSwitch-Monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.debug("Kill switch monitoring started")

    def _stop_monitoring(self) -> None:
        """Stop the monitoring thread"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitor_running and self._state == KillSwitchState.ARMED:
            try:
                self._check_triggers()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self._config.monitor_interval_seconds)

    def _check_triggers(self) -> None:
        """Check automatic trigger conditions"""
        if not self._position_manager:
            return

        try:
            # Get current portfolio value
            current_value = self._position_manager.get_portfolio_value()

            # Update peak value
            if current_value > self._peak_value:
                self._peak_value = current_value

            # Check max drawdown
            if self._peak_value > 0:
                drawdown_pct = ((self._peak_value - current_value) / self._peak_value) * 100

                if drawdown_pct >= self._config.max_drawdown_percent:
                    self.trigger(
                        reason=TriggerReason.MAX_DRAWDOWN,
                        message=f"Max drawdown breached: {drawdown_pct:.1f}% (limit: {self._config.max_drawdown_percent}%)",
                        context={
                            'drawdown_percent': drawdown_pct,
                            'peak_value': self._peak_value,
                            'current_value': current_value
                        }
                    )
                    return

            # Check daily loss
            if self._daily_start_value > 0:
                daily_loss = self._daily_start_value - current_value

                if daily_loss >= self._config.max_daily_loss:
                    self.trigger(
                        reason=TriggerReason.MAX_LOSS,
                        message=f"Max daily loss breached: Rs.{daily_loss:,.0f} (limit: Rs.{self._config.max_daily_loss:,.0f})",
                        context={
                            'daily_loss': daily_loss,
                            'start_value': self._daily_start_value,
                            'current_value': current_value
                        }
                    )
                    return

        except Exception as e:
            logger.error(f"Error checking triggers: {e}")

    def _reset_daily_tracking(self) -> None:
        """Reset daily tracking values"""
        if self._position_manager:
            self._daily_start_value = self._position_manager.get_portfolio_value()
            self._peak_value = self._daily_start_value
        else:
            self._daily_start_value = 0.0
            self._peak_value = 0.0
        self._daily_start_time = datetime.now()

    def update_portfolio_value(self, value: float) -> None:
        """
        Update current portfolio value for monitoring.

        Call this method with current portfolio value for accurate
        drawdown tracking when position_manager is not available.

        Args:
            value: Current portfolio value
        """
        if value > self._peak_value:
            self._peak_value = value

    # =========================================================================
    # Keyboard Trigger
    # =========================================================================

    def _start_keyboard_listener(self) -> None:
        """Start keyboard shortcut listener"""
        if self._keyboard_running:
            return

        # Try to use keyboard library (optional dependency)
        try:
            import keyboard
            self._keyboard_running = True

            def on_hotkey():
                if self._state == KillSwitchState.ARMED:
                    logger.warning("Keyboard trigger detected!")
                    self.trigger(
                        reason=TriggerReason.KEYBOARD,
                        message=f"Triggered via keyboard shortcut ({self._config.keyboard_shortcut})"
                    )

            keyboard.add_hotkey(self._config.keyboard_shortcut, on_hotkey)
            logger.info(f"Keyboard trigger enabled: {self._config.keyboard_shortcut}")

        except ImportError:
            logger.debug("keyboard library not available - keyboard trigger disabled")
            self._keyboard_running = False
        except Exception as e:
            logger.warning(f"Failed to setup keyboard trigger: {e}")
            self._keyboard_running = False

    def _stop_keyboard_listener(self) -> None:
        """Stop keyboard listener"""
        if not self._keyboard_running:
            return

        try:
            import keyboard
            keyboard.remove_hotkey(self._config.keyboard_shortcut)
        except Exception:
            pass

        self._keyboard_running = False

    # =========================================================================
    # Signal Handlers
    # =========================================================================

    def register_signal_handlers(self) -> None:
        """
        Register OS signal handlers for emergency stop.

        SIGTERM and SIGINT will trigger the kill switch.
        """
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received signal {sig_name}")
            self.trigger(
                reason=TriggerReason.EXTERNAL,
                message=f"Triggered by signal: {sig_name}"
            )

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Signal handlers registered (SIGTERM, SIGINT)")
        except Exception as e:
            logger.warning(f"Failed to register signal handlers: {e}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_callback(self, callback: Callable[[KillSwitchEvent], None]) -> None:
        """Register a callback for kill switch events"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[KillSwitchEvent], None]) -> bool:
        """Remove a callback"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_callbacks(self, event: KillSwitchEvent) -> None:
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # =========================================================================
    # Status & History
    # =========================================================================

    def get_events(self) -> List[KillSwitchEvent]:
        """Get history of kill switch events"""
        with self._events_lock:
            return self._events.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        with self._state_lock:
            return {
                'state': self._state.value,
                'is_armed': self._state == KillSwitchState.ARMED,
                'is_triggered': self._state == KillSwitchState.TRIGGERED,
                'trigger_reason': self._trigger_reason.value if self._trigger_reason else None,
                'trigger_time': self._trigger_time.isoformat() if self._trigger_time else None,
                'events_count': len(self._events),
                'config': {
                    'max_drawdown_percent': self._config.max_drawdown_percent,
                    'max_daily_loss': self._config.max_daily_loss,
                    'close_positions': self._config.close_positions_on_trigger,
                    'keyboard_enabled': self._config.enable_keyboard_trigger,
                }
            }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _cleanup(self) -> None:
        """Cleanup on exit"""
        self._stop_monitoring()
        self._stop_keyboard_listener()


# =============================================================================
# Global Instance
# =============================================================================

_global_kill_switch: Optional[KillSwitch] = None
_global_kill_switch_lock = threading.Lock()


def get_kill_switch() -> KillSwitch:
    """Get the global kill switch instance"""
    global _global_kill_switch
    if _global_kill_switch is None:
        with _global_kill_switch_lock:
            if _global_kill_switch is None:
                _global_kill_switch = KillSwitch()
    return _global_kill_switch


def set_kill_switch(kill_switch: KillSwitch) -> None:
    """Set the global kill switch instance"""
    global _global_kill_switch
    with _global_kill_switch_lock:
        _global_kill_switch = kill_switch


def trigger_emergency_stop(
    reason: TriggerReason = TriggerReason.MANUAL,
    message: str = "Emergency stop"
) -> KillSwitchEvent:
    """
    Convenience function to trigger emergency stop.

    Can be called from anywhere in the codebase.
    """
    return get_kill_switch().trigger(reason, message)
