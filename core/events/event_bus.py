"""
Event Bus - Central Pub/Sub Message Broker for Event-Driven Trading Engine.

The Event Bus is the "central nervous system" of the trading engine.
All components communicate through events:
- Data sources PUBLISH price events (ticks, bars)
- Strategies SUBSCRIBE to price events and PUBLISH signals
- Order manager SUBSCRIBES to signals and PUBLISHES order events
- Risk manager SUBSCRIBES to everything and can PUBLISH risk alerts

This allows the SAME code to run in both backtest and live modes.
The only difference is the SOURCE of events.

Features:
- Thread-safe pub/sub
- Synchronous and asynchronous handlers
- Event filtering and routing
- Event history for debugging/replay
- Performance metrics
"""

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from .events import Event, EventType
from .handlers import HandlerRegistry, HandlerRegistration, EventHandler
from .queue import EventQueue, EventPriority


logger = logging.getLogger(__name__)


@dataclass
class EventBusStats:
    """Statistics for event bus performance monitoring."""
    events_published: int = 0
    events_delivered: int = 0
    events_dropped: int = 0
    handler_errors: int = 0
    avg_delivery_time_ms: float = 0.0
    max_delivery_time_ms: float = 0.0
    last_event_time: Optional[datetime] = None
    events_by_type: Dict[str, int] = field(default_factory=dict)


class EventBus:
    """
    Central Event Bus for the trading engine.

    The bus receives events from publishers and routes them to subscribers.
    It supports both synchronous (blocking) and asynchronous (queued) modes.

    Synchronous Mode (default for backtest):
        - Events are delivered immediately in the same thread
        - Handlers block until complete
        - Guarantees event order

    Asynchronous Mode (for live trading):
        - Events are queued for processing
        - Separate consumer thread processes events
        - Non-blocking for publishers
        - Better for high-frequency data

    Usage:
        bus = EventBus()

        # Subscribe to events
        bus.subscribe(EventType.TICK, handle_tick)
        bus.subscribe([EventType.ORDER_FILLED, EventType.ORDER_REJECTED], handle_order)

        # Publish events
        bus.publish(tick_event)
        bus.publish(signal_event)

        # For async mode
        bus.start()  # Start consumer thread
        bus.stop()   # Stop consumer thread
    """

    def __init__(
        self,
        async_mode: bool = False,
        queue_size: int = 10000,
        history_size: int = 1000,
        name: str = "main"
    ):
        """
        Initialize event bus.

        Args:
            async_mode: If True, use queued async delivery
            queue_size: Size of event queue (for async mode)
            history_size: Number of events to keep in history
            name: Bus name (for logging)
        """
        self.name = name
        self.async_mode = async_mode

        # Handler registry
        self._registry = HandlerRegistry()

        # Event queue (for async mode)
        self._queue = EventQueue(max_size=queue_size)

        # Event history (ring buffer)
        self._history: deque = deque(maxlen=history_size)
        self._history_lock = threading.Lock()

        # Thread management
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Statistics
        self._stats = EventBusStats()
        self._stats_lock = threading.Lock()

        # Timing
        self._delivery_times: deque = deque(maxlen=100)

        # Pause/resume for debugging
        self._paused = False
        self._pause_condition = threading.Condition()

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def subscribe(
        self,
        event_types: Union[EventType, List[EventType], None],
        handler: EventHandler,
        priority: int = 100,
        name: str = "",
        group: str = ""
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_types: Event type(s) to subscribe to. None = all events.
            handler: Function to call when event occurs
            priority: Handler priority (lower = called first)
            name: Handler name (for unsubscribe)
            group: Handler group (for bulk unsubscribe)

        Returns:
            Handler name (use for unsubscribe)

        Example:
            # Subscribe to single event type
            bus.subscribe(EventType.TICK, self.on_tick)

            # Subscribe to multiple types
            bus.subscribe([EventType.TICK, EventType.BAR], self.on_price)

            # Subscribe to all events
            bus.subscribe(None, self.on_any_event)
        """
        if isinstance(event_types, EventType):
            types_list = [event_types]
        elif event_types is None:
            types_list = None
        else:
            types_list = event_types

        return self._registry.register(
            handler=handler,
            event_types=types_list,
            priority=priority,
            name=name,
            group=group
        )

    def unsubscribe(self, name: str) -> bool:
        """
        Unsubscribe a handler by name.

        Args:
            name: Handler name (returned from subscribe)

        Returns:
            True if handler was found and removed
        """
        return self._registry.unregister(name)

    def unsubscribe_group(self, group: str) -> int:
        """
        Unsubscribe all handlers in a group.

        Args:
            group: Group name

        Returns:
            Number of handlers removed
        """
        return self._registry.unregister_group(group)

    def unsubscribe_all(self):
        """Remove all subscribers."""
        self._registry.clear()

    # =========================================================================
    # Event Publishing
    # =========================================================================

    def publish(self, event: Event, priority: Optional[EventPriority] = None) -> bool:
        """
        Publish an event to all subscribers.

        In sync mode: Delivers immediately, blocks until all handlers complete.
        In async mode: Queues event for later delivery.

        Args:
            event: The event to publish
            priority: Optional priority override

        Returns:
            True if event was published (sync) or queued (async)
        """
        # Record in history
        with self._history_lock:
            self._history.append(event)

        # Update stats
        with self._stats_lock:
            self._stats.events_published += 1
            self._stats.last_event_time = datetime.now()
            type_name = event.event_type.name
            self._stats.events_by_type[type_name] = self._stats.events_by_type.get(type_name, 0) + 1

        if self.async_mode:
            # Queue for async processing
            return self._queue.put(event, priority)
        else:
            # Deliver synchronously
            self._deliver(event)
            return True

    def emit(self, event: Event, priority: Optional[EventPriority] = None) -> bool:
        """Alias for publish."""
        return self.publish(event, priority)

    def _deliver(self, event: Event):
        """
        Deliver event to all matching handlers.

        Called directly in sync mode, or by consumer thread in async mode.
        """
        # Check if paused
        if self._paused:
            with self._pause_condition:
                while self._paused:
                    self._pause_condition.wait()

        start_time = time.perf_counter()

        handlers = self._registry.get_handlers(event)

        for reg in handlers:
            handler = reg.get_handler()
            if handler is None:
                continue

            try:
                if reg.is_async:
                    # Run async handler in event loop
                    self._run_async_handler(handler, event)
                else:
                    handler(event)

                with self._stats_lock:
                    self._stats.events_delivered += 1

            except Exception as e:
                logger.error(f"Handler {reg.name} error for {event.event_type}: {e}", exc_info=True)
                with self._stats_lock:
                    self._stats.handler_errors += 1

        # Track timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._delivery_times.append(elapsed_ms)

        with self._stats_lock:
            self._stats.avg_delivery_time_ms = sum(self._delivery_times) / len(self._delivery_times)
            self._stats.max_delivery_time_ms = max(self._stats.max_delivery_time_ms, elapsed_ms)

    def _run_async_handler(self, handler: Callable, event: Event):
        """Run an async handler."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Schedule in running loop
            asyncio.ensure_future(handler(event))
        else:
            # Run in new loop
            loop.run_until_complete(handler(event))

    # =========================================================================
    # Async Mode Control
    # =========================================================================

    def start(self):
        """Start the event consumer thread (for async mode)."""
        if self._running:
            return

        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name=f"EventBus-{self.name}",
            daemon=True
        )
        self._consumer_thread.start()
        logger.info(f"EventBus '{self.name}' started in async mode")

    def stop(self, timeout: float = 5.0):
        """Stop the event consumer thread."""
        if not self._running:
            return

        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=timeout)
            self._consumer_thread = None
        logger.info(f"EventBus '{self.name}' stopped")

    def _consumer_loop(self):
        """Main loop for async event processing."""
        while self._running:
            event = self._queue.get(block=True, timeout=0.1)
            if event:
                self._deliver(event)
                self._queue.task_done()

    # =========================================================================
    # Debugging & Control
    # =========================================================================

    def pause(self):
        """Pause event delivery (for debugging)."""
        self._paused = True
        logger.info(f"EventBus '{self.name}' paused")

    def resume(self):
        """Resume event delivery."""
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        logger.info(f"EventBus '{self.name}' resumed")

    @contextmanager
    def paused(self):
        """Context manager for pausing event delivery."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def get_history(self, count: int = 100, event_type: Optional[EventType] = None) -> List[Event]:
        """
        Get recent event history.

        Args:
            count: Max events to return
            event_type: Filter by event type

        Returns:
            List of recent events
        """
        with self._history_lock:
            if event_type:
                events = [e for e in self._history if e.event_type == event_type]
            else:
                events = list(self._history)
            return events[-count:]

    def clear_history(self):
        """Clear event history."""
        with self._history_lock:
            self._history.clear()

    def get_stats(self) -> EventBusStats:
        """Get event bus statistics."""
        with self._stats_lock:
            return EventBusStats(
                events_published=self._stats.events_published,
                events_delivered=self._stats.events_delivered,
                events_dropped=self._stats.events_dropped,
                handler_errors=self._stats.handler_errors,
                avg_delivery_time_ms=self._stats.avg_delivery_time_ms,
                max_delivery_time_ms=self._stats.max_delivery_time_ms,
                last_event_time=self._stats.last_event_time,
                events_by_type=dict(self._stats.events_by_type)
            )

    def reset_stats(self):
        """Reset statistics."""
        with self._stats_lock:
            self._stats = EventBusStats()
            self._delivery_times.clear()

    def list_subscribers(self) -> List[Dict[str, Any]]:
        """List all registered subscribers."""
        return self._registry.list_handlers()

    @property
    def subscriber_count(self) -> int:
        """Get total subscriber count."""
        return self._registry.get_handler_count()

    @property
    def queue_size(self) -> int:
        """Get current queue size (async mode)."""
        return self._queue.size

    @property
    def is_running(self) -> bool:
        """Check if bus is running (async mode)."""
        return self._running


# =============================================================================
# Global Event Bus Instance
# =============================================================================

_global_bus: Optional[EventBus] = None
_global_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.

    Creates a new bus if one doesn't exist.
    Thread-safe singleton pattern.
    """
    global _global_bus
    if _global_bus is None:
        with _global_bus_lock:
            if _global_bus is None:
                _global_bus = EventBus(name="global")
    return _global_bus


def set_event_bus(bus: EventBus):
    """Set the global event bus instance."""
    global _global_bus
    with _global_bus_lock:
        _global_bus = bus


def reset_event_bus():
    """Reset the global event bus (for testing)."""
    global _global_bus
    with _global_bus_lock:
        if _global_bus:
            _global_bus.stop()
            _global_bus.unsubscribe_all()
        _global_bus = None


# =============================================================================
# Convenience Functions
# =============================================================================

def subscribe(
    event_types: Union[EventType, List[EventType], None],
    handler: EventHandler,
    priority: int = 100,
    name: str = ""
) -> str:
    """Subscribe to events on the global bus."""
    return get_event_bus().subscribe(event_types, handler, priority, name)


def unsubscribe(name: str) -> bool:
    """Unsubscribe from the global bus."""
    return get_event_bus().unsubscribe(name)


def publish(event: Event) -> bool:
    """Publish event to the global bus."""
    return get_event_bus().publish(event)


def emit(event: Event) -> bool:
    """Alias for publish."""
    return publish(event)
