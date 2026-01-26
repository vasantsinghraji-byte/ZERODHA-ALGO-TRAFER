"""
Event-Driven Architecture for AlgoTrader Pro.

This module provides the unified event system that powers both backtesting
and live trading with THE SAME CODE. The engine doesn't know if it's
"live" or "testing" - it just receives and processes Events.

Core Components:
    - Event: Base class for all events
    - EventBus: Central pub/sub message broker
    - EventQueue: Thread-safe priority queue
    - HandlerRegistry: Event handler management

Event Types:
    - Market Data: TickEvent, BarEvent, PriceUpdateEvent
    - Signals: SignalEvent
    - Orders: OrderEvent, FillEvent
    - Positions: PositionEvent
    - Risk: RiskEvent, StopLossEvent, TargetHitEvent
    - System: MarketOpenEvent, MarketCloseEvent, ErrorEvent

Usage:
    from core.events import (
        EventBus, EventType,
        TickEvent, BarEvent, PriceUpdateEvent,
        SignalEvent, SignalType,
        get_event_bus, subscribe, publish
    )

    # Get the global event bus
    bus = get_event_bus()

    # Subscribe to events
    bus.subscribe(EventType.PRICE_UPDATE, handle_price)
    bus.subscribe([EventType.TICK, EventType.BAR], handle_market_data)

    # Publish events
    bus.publish(PriceUpdateEvent(symbol="RELIANCE", price=2450.50))

    # Using decorators
    class MyStrategy:
        @on_event(EventType.PRICE_UPDATE)
        def on_price(self, event):
            # Process price update
            pass

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    📨 EVENT BUS (Central Nervous System)        │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Events: PriceUpdate | OrderFill | PositionChange |     │   │
    │  │          RiskAlert | MarketOpen | MarketClose           │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                              │                                  │
    │              ┌───────────────┼───────────────┐                  │
    │              ▼               ▼               ▼                  │
    │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
    │  │ DATA SOURCE  │ │   STRATEGY   │ │  EXECUTION   │            │
    │  │ (Agnostic)   │ │   ENGINE     │ │   ENGINE     │            │
    │  │              │ │              │ │              │            │
    │  │ - Backtest   │ │ Same code    │ │ - Paper      │            │
    │  │ - Live Feed  │ │ for both!    │ │ - Live       │            │
    │  │ - Replay     │ │              │ │ - Simulated  │            │
    │  └──────────────┘ └──────────────┘ └──────────────┘            │
    └─────────────────────────────────────────────────────────────────┘
"""

# Event Types
from .events import (
    # Base
    Event,
    EventType,
    Side,
    OrderStatus,
    SignalType,

    # Market Data Events
    TickEvent,
    BarEvent,
    PriceUpdateEvent,

    # Signal Events
    SignalEvent,

    # Order Events
    OrderEvent,
    FillEvent,

    # Position Events
    PositionEvent,

    # Risk Events
    RiskEvent,
    StopLossEvent,
    TargetHitEvent,

    # System Events
    SystemEvent,
    MarketOpenEvent,
    MarketCloseEvent,
    ErrorEvent,
    HeartbeatEvent,
)

# Event Bus
from .event_bus import (
    EventBus,
    EventBusStats,
    get_event_bus,
    set_event_bus,
    reset_event_bus,
    subscribe,
    unsubscribe,
    publish,
    emit,
)

# Event Queue
from .queue import (
    EventQueue,
    EventPriority,
    MultiEventQueue,
)

# Handler Registry
from .handlers import (
    HandlerRegistry,
    HandlerRegistration,
    on_event,
    collect_handlers,
    register_object_handlers,
)


__all__ = [
    # Base Event Types
    'Event',
    'EventType',
    'Side',
    'OrderStatus',
    'SignalType',

    # Market Data Events
    'TickEvent',
    'BarEvent',
    'PriceUpdateEvent',

    # Signal Events
    'SignalEvent',

    # Order Events
    'OrderEvent',
    'FillEvent',

    # Position Events
    'PositionEvent',

    # Risk Events
    'RiskEvent',
    'StopLossEvent',
    'TargetHitEvent',

    # System Events
    'SystemEvent',
    'MarketOpenEvent',
    'MarketCloseEvent',
    'ErrorEvent',
    'HeartbeatEvent',

    # Event Bus
    'EventBus',
    'EventBusStats',
    'get_event_bus',
    'set_event_bus',
    'reset_event_bus',
    'subscribe',
    'unsubscribe',
    'publish',
    'emit',

    # Event Queue
    'EventQueue',
    'EventPriority',
    'MultiEventQueue',

    # Handler Registry
    'HandlerRegistry',
    'HandlerRegistration',
    'on_event',
    'collect_handlers',
    'register_object_handlers',
]
