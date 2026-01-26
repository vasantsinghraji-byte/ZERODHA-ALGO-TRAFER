"""
Event Type Definitions for Event-Driven Trading Engine.

This module defines all event types used in the unified trading system.
Events are the common language between backtest and live trading modes.

The engine doesn't know if it's "live" or "testing" - it just receives Events.
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional
import uuid


class EventType(Enum):
    """All event types in the system."""

    # Market Data Events
    TICK = auto()           # Real-time tick update
    BAR = auto()            # OHLCV bar completed
    PRICE_UPDATE = auto()   # Generic price update (tick or bar)

    # Order Events
    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_REJECTED = auto()
    ORDER_CANCELLED = auto()
    ORDER_FILLED = auto()
    ORDER_PARTIALLY_FILLED = auto()

    # Position Events
    POSITION_OPENED = auto()
    POSITION_UPDATED = auto()
    POSITION_CLOSED = auto()

    # Signal Events
    SIGNAL_GENERATED = auto()

    # Risk Events
    STOP_LOSS_HIT = auto()
    TARGET_HIT = auto()
    RISK_LIMIT_BREACH = auto()
    DAILY_LOSS_LIMIT = auto()

    # System Events
    MARKET_OPEN = auto()
    MARKET_CLOSE = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    HEARTBEAT = auto()
    ERROR = auto()


class Side(Enum):
    """Order/Position side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"


@dataclass
class Event(ABC):
    """
    Base class for all events.

    Every event has:
    - event_id: Unique identifier
    - event_type: Type of event
    - timestamp: When the event occurred
    - source: Where the event originated (backtest/live/replay)
    """
    # All fields have defaults to allow subclass fields without defaults
    event_type: EventType = field(default=None)  # Set by subclass in __post_init__
    timestamp: datetime = field(default=None)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "unknown"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# Market Data Events
# =============================================================================

@dataclass
class TickEvent(Event):
    """
    Real-time tick update from market.

    Used in live trading when WebSocket pushes price updates.
    """
    symbol: str = ""
    instrument_token: int = 0
    last_price: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    open_interest: int = 0

    def __post_init__(self):
        self.event_type = EventType.TICK
        super().__post_init__()


@dataclass
class BarEvent(Event):
    """
    OHLCV bar completed event.

    Used in backtesting when iterating over historical data.
    Also used in live trading when a candle closes.
    """
    symbol: str = ""
    timeframe: str = "1m"  # 1m, 5m, 15m, 1h, 1d
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    bar_index: int = 0  # Position in the data series

    def __post_init__(self):
        self.event_type = EventType.BAR
        super().__post_init__()

    @property
    def last_price(self) -> float:
        """Alias for close price - makes it compatible with TickEvent interface."""
        return self.close


@dataclass
class PriceUpdateEvent(Event):
    """
    Generic price update event - the UNIFIED interface.

    This is the primary event that strategies should listen to.
    Works for both ticks (live) and bars (backtest).

    The strategy doesn't care if it's a tick or bar - it just sees a price update.
    """
    symbol: str = ""
    price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    bar_index: int = -1  # -1 for live ticks
    is_bar_close: bool = False  # True when a bar has closed

    def __post_init__(self):
        self.event_type = EventType.PRICE_UPDATE
        super().__post_init__()

    @classmethod
    def from_tick(cls, tick: TickEvent) -> 'PriceUpdateEvent':
        """Create PriceUpdateEvent from a TickEvent."""
        return cls(
            symbol=tick.symbol,
            price=tick.last_price,
            high=tick.last_price,
            low=tick.last_price,
            volume=tick.volume,
            bar_index=-1,
            is_bar_close=False,
            timestamp=tick.timestamp,
            source=tick.source
        )

    @classmethod
    def from_bar(cls, bar: BarEvent) -> 'PriceUpdateEvent':
        """Create PriceUpdateEvent from a BarEvent."""
        return cls(
            symbol=bar.symbol,
            price=bar.close,
            high=bar.high,
            low=bar.low,
            volume=bar.volume,
            bar_index=bar.bar_index,
            is_bar_close=True,
            timestamp=bar.timestamp,
            source=bar.source
        )


# =============================================================================
# Signal Events
# =============================================================================

class SignalType(Enum):
    """Signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class SignalEvent(Event):
    """
    Trading signal generated by a strategy.

    This is what strategies emit when they detect an opportunity.
    """
    signal_type: SignalType = SignalType.HOLD
    symbol: str = ""
    price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 0
    confidence: float = 0.5
    reason: str = ""
    strategy_name: str = ""

    def __post_init__(self):
        self.event_type = EventType.SIGNAL_GENERATED
        super().__post_init__()


# =============================================================================
# Order Events
# =============================================================================

@dataclass
class OrderEvent(Event):
    """
    Order lifecycle event.

    Emitted when orders are submitted, filled, cancelled, etc.
    """
    order_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    order_type: str = "MARKET"  # MARKET, LIMIT, SL, SL-M
    price: float = 0.0
    trigger_price: float = 0.0
    filled_quantity: int = 0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: str = ""
    strategy_name: str = ""

    def __post_init__(self):
        # Set event_type based on status if not already set
        if self.event_type is None:
            status_to_event = {
                OrderStatus.PENDING: EventType.ORDER_SUBMITTED,
                OrderStatus.SUBMITTED: EventType.ORDER_SUBMITTED,
                OrderStatus.ACCEPTED: EventType.ORDER_ACCEPTED,
                OrderStatus.REJECTED: EventType.ORDER_REJECTED,
                OrderStatus.CANCELLED: EventType.ORDER_CANCELLED,
                OrderStatus.FILLED: EventType.ORDER_FILLED,
                OrderStatus.PARTIALLY_FILLED: EventType.ORDER_PARTIALLY_FILLED,
            }
            self.event_type = status_to_event.get(self.status, EventType.ORDER_SUBMITTED)
        super().__post_init__()


@dataclass
class FillEvent(Event):
    """
    Order fill event - subset of OrderEvent for convenience.

    Emitted specifically when an order is filled (fully or partially).
    """
    order_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    strategy_name: str = ""

    def __post_init__(self):
        self.event_type = EventType.ORDER_FILLED
        super().__post_init__()


# =============================================================================
# Position Events
# =============================================================================

@dataclass
class PositionEvent(Event):
    """
    Position change event.

    Emitted when positions are opened, updated, or closed.
    """
    symbol: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    strategy_name: str = ""

    def __post_init__(self):
        if self.event_type is None:
            self.event_type = EventType.POSITION_UPDATED
        super().__post_init__()


# =============================================================================
# Risk Events
# =============================================================================

@dataclass
class RiskEvent(Event):
    """
    Risk-related event.

    Emitted when risk limits are breached, stops are hit, etc.
    """
    symbol: str = ""
    risk_type: str = ""  # stop_loss, target, daily_limit, correlation
    message: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    action_taken: str = ""  # closed_position, blocked_order, etc.

    def __post_init__(self):
        if self.event_type is None:
            self.event_type = EventType.RISK_LIMIT_BREACH
        super().__post_init__()


@dataclass
class StopLossEvent(RiskEvent):
    """Stop loss triggered event."""

    def __post_init__(self):
        self.event_type = EventType.STOP_LOSS_HIT
        self.risk_type = "stop_loss"
        super().__post_init__()


@dataclass
class TargetHitEvent(RiskEvent):
    """Target price hit event."""

    def __post_init__(self):
        self.event_type = EventType.TARGET_HIT
        self.risk_type = "target"
        super().__post_init__()


# =============================================================================
# System Events
# =============================================================================

@dataclass
class SystemEvent(Event):
    """
    System-level event.

    Market open/close, session events, errors, etc.
    """
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.event_type is None:
            self.event_type = EventType.SESSION_START
        super().__post_init__()


@dataclass
class MarketOpenEvent(SystemEvent):
    """Market open event."""

    def __post_init__(self):
        self.event_type = EventType.MARKET_OPEN
        self.message = "Market opened"
        super().__post_init__()


@dataclass
class MarketCloseEvent(SystemEvent):
    """Market close event."""

    def __post_init__(self):
        self.event_type = EventType.MARKET_CLOSE
        self.message = "Market closed"
        super().__post_init__()


@dataclass
class ErrorEvent(SystemEvent):
    """Error event."""
    error_code: str = ""
    exception: Optional[Exception] = None

    def __post_init__(self):
        self.event_type = EventType.ERROR
        super().__post_init__()


@dataclass
class HeartbeatEvent(SystemEvent):
    """Heartbeat event for health monitoring."""
    sequence: int = 0

    def __post_init__(self):
        self.event_type = EventType.HEARTBEAT
        self.message = f"Heartbeat #{self.sequence}"
        super().__post_init__()
