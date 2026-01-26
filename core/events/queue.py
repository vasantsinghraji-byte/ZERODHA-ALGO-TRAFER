"""
Thread-Safe Event Queue for Event-Driven Trading Engine.

This module provides a thread-safe queue for buffering events between
producers (data sources) and consumers (strategy engine, order manager).

Features:
- Thread-safe with proper locking
- Priority support for urgent events (risk alerts)
- Batch retrieval for efficiency
- Backpressure handling when queue is full
"""

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import List, Optional, Tuple

from .events import Event, EventType


class EventPriority(IntEnum):
    """
    Event priority levels.

    Lower number = higher priority.
    Risk events and system events get priority over regular market data.
    """
    CRITICAL = 0    # System errors, risk breaches
    HIGH = 1        # Stop loss, target hits
    NORMAL = 2      # Signals, order events
    LOW = 3         # Market data (ticks, bars)


@dataclass
class PrioritizedEvent:
    """Wrapper for event with priority and sequence number."""
    priority: EventPriority
    sequence: int
    event: Event

    def __lt__(self, other: 'PrioritizedEvent') -> bool:
        """Compare by priority first, then by sequence (FIFO within priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence


class EventQueue:
    """
    Thread-safe priority queue for events.

    Events are processed in priority order:
    1. CRITICAL: Risk breaches, errors (process immediately)
    2. HIGH: Stop loss/target hits (process before signals)
    3. NORMAL: Signals, order events (standard processing)
    4. LOW: Market data (can be batched)

    Usage:
        queue = EventQueue(max_size=10000)

        # Producer thread
        queue.put(tick_event, priority=EventPriority.LOW)
        queue.put(risk_event, priority=EventPriority.CRITICAL)

        # Consumer thread
        event = queue.get(timeout=1.0)
        queue.task_done()
    """

    # Default priority mapping by event type
    DEFAULT_PRIORITIES = {
        EventType.ERROR: EventPriority.CRITICAL,
        EventType.RISK_LIMIT_BREACH: EventPriority.CRITICAL,
        EventType.DAILY_LOSS_LIMIT: EventPriority.CRITICAL,

        EventType.STOP_LOSS_HIT: EventPriority.HIGH,
        EventType.TARGET_HIT: EventPriority.HIGH,

        EventType.SIGNAL_GENERATED: EventPriority.NORMAL,
        EventType.ORDER_SUBMITTED: EventPriority.NORMAL,
        EventType.ORDER_FILLED: EventPriority.NORMAL,
        EventType.ORDER_REJECTED: EventPriority.NORMAL,
        EventType.POSITION_OPENED: EventPriority.NORMAL,
        EventType.POSITION_CLOSED: EventPriority.NORMAL,

        EventType.TICK: EventPriority.LOW,
        EventType.BAR: EventPriority.LOW,
        EventType.PRICE_UPDATE: EventPriority.LOW,
        EventType.HEARTBEAT: EventPriority.LOW,
    }

    def __init__(self, max_size: int = 10000):
        """
        Initialize event queue.

        Args:
            max_size: Maximum queue size. 0 = unlimited.
        """
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # Statistics
        self._events_received = 0
        self._events_processed = 0
        self._events_dropped = 0
        self._last_event_time: Optional[datetime] = None

    def _get_next_sequence(self) -> int:
        """Get next sequence number (thread-safe)."""
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _get_priority(self, event: Event, priority: Optional[EventPriority] = None) -> EventPriority:
        """Get priority for event, using default if not specified."""
        if priority is not None:
            return priority
        return self.DEFAULT_PRIORITIES.get(event.event_type, EventPriority.NORMAL)

    def put(
        self,
        event: Event,
        priority: Optional[EventPriority] = None,
        block: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Add event to queue.

        Args:
            event: The event to queue
            priority: Optional priority override
            block: Whether to block if queue is full
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            True if event was queued, False if dropped
        """
        prio = self._get_priority(event, priority)
        seq = self._get_next_sequence()
        prioritized = PrioritizedEvent(priority=prio, sequence=seq, event=event)

        try:
            self._queue.put(prioritized, block=block, timeout=timeout)
            with self._stats_lock:
                self._events_received += 1
                self._last_event_time = datetime.now()
            return True
        except queue.Full:
            with self._stats_lock:
                self._events_dropped += 1
            return False

    def put_nowait(self, event: Event, priority: Optional[EventPriority] = None) -> bool:
        """Add event without blocking. Returns False if queue is full."""
        return self.put(event, priority, block=False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Event]:
        """
        Get next event from queue.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            Next event or None if timeout/empty
        """
        try:
            prioritized: PrioritizedEvent = self._queue.get(block=block, timeout=timeout)
            return prioritized.event
        except queue.Empty:
            return None

    def get_nowait(self) -> Optional[Event]:
        """Get next event without blocking. Returns None if empty."""
        return self.get(block=False)

    def get_batch(self, max_count: int = 100, timeout: float = 0.0) -> List[Event]:
        """
        Get a batch of events from queue.

        Useful for processing multiple market data events at once.

        Args:
            max_count: Maximum number of events to retrieve
            timeout: Timeout for first event (subsequent are non-blocking)

        Returns:
            List of events (may be empty if timeout)
        """
        events = []

        # Get first event with timeout
        first = self.get(block=True, timeout=timeout if timeout > 0 else 0.001)
        if first:
            events.append(first)

        # Get remaining events without blocking
        while len(events) < max_count:
            event = self.get_nowait()
            if event is None:
                break
            events.append(event)

        return events

    def task_done(self):
        """Mark current task as done (for join() synchronization)."""
        self._queue.task_done()
        with self._stats_lock:
            self._events_processed += 1

    def join(self):
        """Block until all events have been processed."""
        self._queue.join()

    def clear(self):
        """Clear all events from queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    @property
    def size(self) -> int:
        """Current number of events in queue."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._stats_lock:
            return {
                'current_size': self.size,
                'events_received': self._events_received,
                'events_processed': self._events_processed,
                'events_dropped': self._events_dropped,
                'last_event_time': self._last_event_time,
                'pending': self._events_received - self._events_processed
            }


class MultiEventQueue:
    """
    Multiple queues for different event categories.

    Separates high-frequency market data from lower-frequency control events.
    This prevents market data floods from delaying risk events.

    Usage:
        mq = MultiEventQueue()

        # Market data goes to data queue
        mq.put_data(tick_event)

        # Risk events go to control queue
        mq.put_control(risk_event)

        # Consumer checks control queue first
        event = mq.get_control_nowait() or mq.get_data_nowait()
    """

    def __init__(self, data_queue_size: int = 50000, control_queue_size: int = 1000):
        """
        Initialize multi-queue.

        Args:
            data_queue_size: Size of market data queue
            control_queue_size: Size of control/signal queue
        """
        self.data_queue = EventQueue(max_size=data_queue_size)
        self.control_queue = EventQueue(max_size=control_queue_size)

    def put_data(self, event: Event) -> bool:
        """Put market data event (ticks, bars)."""
        return self.data_queue.put_nowait(event, EventPriority.LOW)

    def put_control(self, event: Event, priority: Optional[EventPriority] = None) -> bool:
        """Put control event (signals, orders, risk)."""
        return self.control_queue.put(event, priority)

    def put(self, event: Event, priority: Optional[EventPriority] = None) -> bool:
        """
        Auto-route event to appropriate queue based on type.
        """
        if event.event_type in (EventType.TICK, EventType.BAR, EventType.PRICE_UPDATE):
            return self.put_data(event)
        return self.put_control(event, priority)

    def get_control_nowait(self) -> Optional[Event]:
        """Get control event without blocking."""
        return self.control_queue.get_nowait()

    def get_data_nowait(self) -> Optional[Event]:
        """Get data event without blocking."""
        return self.data_queue.get_nowait()

    def get_next(self, timeout: float = 0.1) -> Optional[Event]:
        """
        Get next event, prioritizing control queue.

        Checks control queue first, then data queue.
        """
        # Always check control queue first
        event = self.control_queue.get_nowait()
        if event:
            return event

        # Then check data queue
        return self.data_queue.get(block=True, timeout=timeout)

    def get_stats(self) -> dict:
        """Get statistics for both queues."""
        return {
            'data_queue': self.data_queue.get_stats(),
            'control_queue': self.control_queue.get_stats()
        }
