"""
Base Strategy Class - Unified Event-Driven Interface

All strategies follow this pattern. The unified interface allows the same
strategy code to work seamlessly in both backtest and live trading modes.

The engine doesn't know if it's "live" or "testing" - it just sends Events.
Your strategy doesn't care either - it just reacts to Events.

Think of this as a recipe template - each strategy fills in the details!
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
import pandas as pd

# Import event types for type hints
if TYPE_CHECKING:
    from core.events.events import (
        Event, BarEvent, TickEvent, PriceUpdateEvent,
        FillEvent, PositionEvent, SignalEvent
    )


class SignalType(Enum):
    """What the strategy is telling you to do"""
    BUY = "BUY"           # Buy now!
    SELL = "SELL"         # Sell now!
    HOLD = "HOLD"         # Wait and watch
    EXIT = "EXIT"         # Close your position
    EXIT_LONG = "EXIT_LONG"   # Close long position
    EXIT_SHORT = "EXIT_SHORT" # Close short position


class RiskLevel(Enum):
    """How risky is this strategy?"""
    LOW = "LOW"           # 🐢 Safe, small gains
    MEDIUM = "MEDIUM"     # ⚡ Balanced
    HIGH = "HIGH"         # 🚀 Risky, big potential


@dataclass
class Signal:
    """
    A trading signal from a strategy.

    This is what the strategy tells you to do.
    """
    signal_type: SignalType    # BUY, SELL, HOLD, or EXIT
    symbol: str                # Which stock
    price: float               # Current price
    stop_loss: float = 0       # Where to cut losses
    target: float = 0          # Where to take profits
    confidence: float = 0.5    # How sure (0 to 1)
    reason: str = ""           # Why this signal
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def emoji(self) -> str:
        """Get emoji for the signal"""
        return {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡",
            SignalType.EXIT: "🚪",
        }.get(self.signal_type, "❓")


class Strategy(ABC):
    """
    Base class for all trading strategies - Unified Event-Driven Interface.

    Every strategy must implement:
    1. name - What's it called?
    2. description - What does it do?
    3. analyze() - Look at data and decide what to do (legacy interface)

    Event-Driven Methods (optional, for new strategies):
    - on_bar() - Called when a new bar completes
    - on_tick() - Called on each tick (live trading)
    - on_price() - Unified price update (works for both bar/tick)
    - on_fill() - Called when an order is filled
    - on_position() - Called when position changes

    The engine doesn't care if it's backtest or live - it just sends events.
    Your strategy doesn't care either - it just reacts to events.
    """

    def __init__(self):
        self._signals: List[Signal] = []
        self._event_bus = None  # Set by engine when registering strategy
        self._positions: Dict[str, Any] = {}  # Current positions by symbol
        self._bars: Dict[str, pd.DataFrame] = {}  # Historical bars by symbol
        self._last_prices: Dict[str, float] = {}  # Last known prices

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Simple description a 5th grader can understand"""
        pass

    @property
    def risk_level(self) -> RiskLevel:
        """How risky is this strategy?"""
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        """Emoji representing this strategy"""
        return "📊"

    # =========================================================================
    # Legacy Interface (backward compatible)
    # =========================================================================

    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze the data and generate a signal.

        This is the LEGACY interface - still works for backward compatibility.
        New strategies should prefer on_bar() or on_price() methods.

        Args:
            data: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            symbol: Stock symbol

        Returns:
            Signal telling what to do
        """
        pass

    # =========================================================================
    # Event-Driven Interface (new unified approach)
    # =========================================================================

    def on_event(self, event: 'Event') -> Optional[Signal]:
        """
        Unified event entry point - routes events to specific handlers.

        This is called by the engine for EVERY event. Override specific
        handlers (on_bar, on_price, etc.) instead of this method.

        Args:
            event: Any event from the event bus

        Returns:
            Optional Signal if the event triggers a trading decision
        """
        from core.events.events import (
            EventType, BarEvent, TickEvent, PriceUpdateEvent,
            FillEvent, PositionEvent
        )

        if event.event_type == EventType.BAR:
            return self.on_bar(event)
        elif event.event_type == EventType.TICK:
            return self.on_tick(event)
        elif event.event_type == EventType.PRICE_UPDATE:
            return self.on_price(event)
        elif event.event_type == EventType.ORDER_FILLED:
            return self.on_fill(event)
        elif event.event_type in (EventType.POSITION_OPENED,
                                   EventType.POSITION_UPDATED,
                                   EventType.POSITION_CLOSED):
            return self.on_position(event)

        return None

    def on_bar(self, bar: 'BarEvent') -> Optional[Signal]:
        """
        Called when a new bar (candle) completes.

        This is the primary method for bar-based strategies.
        Default implementation builds bar history and calls analyze().

        Args:
            bar: The completed bar event

        Returns:
            Optional Signal if bar triggers a trading decision
        """
        # Build bar history for this symbol
        self._update_bar_history(bar)

        # Update last price
        self._last_prices[bar.symbol] = bar.close

        # Call legacy analyze() with accumulated data
        if bar.symbol in self._bars and len(self._bars[bar.symbol]) > 0:
            return self.analyze(self._bars[bar.symbol], bar.symbol)

        return None

    def on_tick(self, tick: 'TickEvent') -> Optional[Signal]:
        """
        Called on each tick update (real-time price).

        Override this for tick-based strategies (scalping, HFT).
        Default implementation just updates last price.

        Args:
            tick: The tick event

        Returns:
            Optional Signal if tick triggers a trading decision
        """
        self._last_prices[tick.symbol] = tick.last_price
        return None

    def on_price(self, price: 'PriceUpdateEvent') -> Optional[Signal]:
        """
        Unified price update handler - works for both bars and ticks.

        This is the RECOMMENDED method to override for new strategies.
        The engine converts bars/ticks to PriceUpdateEvent so you get
        a consistent interface regardless of backtest or live mode.

        Args:
            price: The price update event

        Returns:
            Optional Signal if price triggers a trading decision
        """
        self._last_prices[price.symbol] = price.price
        return None

    def on_fill(self, fill: 'FillEvent') -> Optional[Signal]:
        """
        Called when an order is filled.

        Override this to react to fills (e.g., set trailing stops,
        scale in/out of positions, etc.)

        Args:
            fill: The fill event

        Returns:
            Optional Signal for follow-up action
        """
        return None

    def on_position(self, position: 'PositionEvent') -> Optional[Signal]:
        """
        Called when position changes.

        Override this to react to position updates (e.g., adjust
        risk parameters, log P&L, etc.)

        Args:
            position: The position event

        Returns:
            Optional Signal for position management
        """
        return None

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _update_bar_history(self, bar: 'BarEvent') -> None:
        """Add bar to historical data for this symbol."""
        symbol = bar.symbol

        # Create new row
        new_row = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }])

        if symbol not in self._bars:
            self._bars[symbol] = new_row
        else:
            self._bars[symbol] = pd.concat(
                [self._bars[symbol], new_row],
                ignore_index=True
            )

    def set_event_bus(self, bus) -> None:
        """Set the event bus for emitting signals."""
        self._event_bus = bus

    def emit_signal(self, signal: Signal) -> None:
        """
        Emit a signal to the event bus.

        This converts the legacy Signal to SignalEvent and publishes it.
        """
        if self._event_bus is None:
            self._signals.append(signal)
            return

        from core.events.events import SignalEvent, SignalType as EventSignalType

        # Map local SignalType to event SignalType
        signal_type_map = {
            SignalType.BUY: EventSignalType.BUY,
            SignalType.SELL: EventSignalType.SELL,
            SignalType.HOLD: EventSignalType.HOLD,
            SignalType.EXIT: EventSignalType.EXIT,
        }

        event_signal = SignalEvent(
            signal_type=signal_type_map.get(signal.signal_type, EventSignalType.HOLD),
            symbol=signal.symbol,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            reason=signal.reason,
            strategy_name=self.name,
            timestamp=signal.timestamp
        )

        self._event_bus.publish(event_signal)

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for a symbol."""
        return self._last_prices.get(symbol)

    def get_bars(self, symbol: str, count: int = None) -> Optional[pd.DataFrame]:
        """Get historical bars for a symbol."""
        if symbol not in self._bars:
            return None
        bars = self._bars[symbol]
        if count is not None:
            return bars.tail(count)
        return bars

    # =========================================================================
    # Risk Calculations
    # =========================================================================

    def calculate_stop_loss(self, price: float, is_buy: bool,
                           risk_percent: float = 2.0) -> float:
        """
        Calculate where to place stop-loss.

        Args:
            price: Entry price
            is_buy: True for buy, False for sell
            risk_percent: How much to risk (default 2%)

        Returns:
            Stop-loss price
        """
        if is_buy:
            return price * (1 - risk_percent / 100)
        else:
            return price * (1 + risk_percent / 100)

    def calculate_target(self, price: float, is_buy: bool,
                        reward_ratio: float = 2.0) -> float:
        """
        Calculate profit target.

        Args:
            price: Entry price
            is_buy: True for buy, False for sell
            reward_ratio: Risk:Reward ratio (default 1:2)

        Returns:
            Target price
        """
        risk = abs(price - self.calculate_stop_loss(price, is_buy)) * reward_ratio
        if is_buy:
            return price + risk
        else:
            return price - risk

    # =========================================================================
    # Parameter Management
    # =========================================================================

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters for display"""
        return {}

    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters"""
        pass

    def reset(self) -> None:
        """Reset strategy state - called between backtest runs."""
        self._signals.clear()
        self._positions.clear()
        self._bars.clear()
        self._last_prices.clear()

    def __str__(self) -> str:
        return f"{self.emoji} {self.name}"


# =============================================================================
# Legacy Strategy Adapter
# =============================================================================

class LegacyStrategyAdapter:
    """
    Adapter that wraps legacy strategies to work with the event-driven system.

    Use this to integrate existing strategies that only implement analyze()
    into the new event-driven engine without modifying the original code.

    Example:
        # Old strategy
        class MyOldStrategy(Strategy):
            def analyze(self, data, symbol):
                ...

        # Wrap it for event-driven use
        old_strategy = MyOldStrategy()
        adapted = LegacyStrategyAdapter(old_strategy, event_bus)

        # Now it responds to events
        adapted.on_bar(bar_event)
    """

    def __init__(self, strategy: Strategy, event_bus=None, warmup_bars: int = 50):
        """
        Initialize the adapter.

        Args:
            strategy: The legacy strategy to wrap
            event_bus: Optional event bus for emitting signals
            warmup_bars: Number of bars to collect before calling analyze()
        """
        self.strategy = strategy
        self._event_bus = event_bus
        self.warmup_bars = warmup_bars
        self._bars: Dict[str, pd.DataFrame] = {}
        self._last_signals: Dict[str, Signal] = {}

        # Forward strategy properties
        self.name = strategy.name
        self.description = strategy.description
        self.risk_level = strategy.risk_level
        self.emoji = strategy.emoji

    def on_event(self, event: 'Event') -> Optional[Signal]:
        """Route events to appropriate handlers."""
        from core.events.events import EventType

        if event.event_type == EventType.BAR:
            return self.on_bar(event)
        elif event.event_type == EventType.PRICE_UPDATE:
            return self.on_price(event)

        return None

    def on_bar(self, bar: 'BarEvent') -> Optional[Signal]:
        """
        Handle bar event by building history and calling analyze().

        Args:
            bar: The bar event

        Returns:
            Signal from the legacy strategy, if any
        """
        symbol = bar.symbol

        # Build bar DataFrame
        new_row = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }])

        if symbol not in self._bars:
            self._bars[symbol] = new_row
        else:
            self._bars[symbol] = pd.concat(
                [self._bars[symbol], new_row],
                ignore_index=True
            )

        # Check if we have enough bars
        if len(self._bars[symbol]) < self.warmup_bars:
            return None

        # Call legacy analyze()
        try:
            signal = self.strategy.analyze(self._bars[symbol], symbol)
            self._last_signals[symbol] = signal

            # Emit to event bus if connected
            if self._event_bus is not None and signal.signal_type != SignalType.HOLD:
                self._emit_signal_event(signal)

            return signal

        except Exception as e:
            # Log error but don't crash
            print(f"Error in legacy strategy {self.name}: {e}")
            return None

    def on_price(self, price: 'PriceUpdateEvent') -> Optional[Signal]:
        """
        Handle price update by converting to bar if it's a bar close.

        Args:
            price: The price update event

        Returns:
            Signal if bar triggers analyze()
        """
        if price.is_bar_close:
            # Create a BarEvent from PriceUpdateEvent
            from core.events.events import BarEvent

            bar = BarEvent(
                symbol=price.symbol,
                open=price.price,  # Approximate - we don't have full OHLC
                high=price.high,
                low=price.low,
                close=price.price,
                volume=price.volume,
                bar_index=price.bar_index,
                timestamp=price.timestamp,
                source=price.source
            )
            return self.on_bar(bar)

        return None

    def _emit_signal_event(self, signal: Signal) -> None:
        """Convert Signal to SignalEvent and publish."""
        from core.events.events import SignalEvent, SignalType as EventSignalType

        signal_type_map = {
            SignalType.BUY: EventSignalType.BUY,
            SignalType.SELL: EventSignalType.SELL,
            SignalType.HOLD: EventSignalType.HOLD,
            SignalType.EXIT: EventSignalType.EXIT,
        }

        event_signal = SignalEvent(
            signal_type=signal_type_map.get(signal.signal_type, EventSignalType.HOLD),
            symbol=signal.symbol,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            reason=signal.reason,
            strategy_name=self.name,
            timestamp=signal.timestamp
        )

        self._event_bus.publish(event_signal)

    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        """Get the last signal for a symbol."""
        return self._last_signals.get(symbol)

    def reset(self) -> None:
        """Reset adapter state."""
        self._bars.clear()
        self._last_signals.clear()
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()

    def __str__(self) -> str:
        return f"Adapted: {self.strategy}"


# =============================================================================
# Event-Driven Strategy Base (Alternative to Strategy)
# =============================================================================

class EventStrategy(ABC):
    """
    Pure event-driven strategy base class.

    Use this for NEW strategies that are designed from the ground up
    for the event-driven system. Unlike Strategy, this doesn't require
    implementing analyze() - you implement on_price() or on_bar() instead.

    Example:
        class MyEventStrategy(EventStrategy):
            name = "My Strategy"
            description = "A modern event-driven strategy"

            def on_bar(self, bar):
                if bar.close > bar.open:
                    return self.create_signal(SignalType.BUY, bar.symbol, bar.close)
                return None
    """

    def __init__(self):
        self._event_bus = None
        self._positions: Dict[str, Any] = {}
        self._bars: Dict[str, pd.DataFrame] = {}
        self._last_prices: Dict[str, float] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""
        pass

    @property
    def risk_level(self) -> RiskLevel:
        """Risk level of the strategy."""
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        """Emoji for the strategy."""
        return "📊"

    # =========================================================================
    # Event Handlers - Override these
    # =========================================================================

    def on_event(self, event: 'Event') -> Optional[Signal]:
        """Main event router."""
        from core.events.events import EventType

        if event.event_type == EventType.BAR:
            self._update_bar_history(event)
            return self.on_bar(event)
        elif event.event_type == EventType.TICK:
            self._last_prices[event.symbol] = event.last_price
            return self.on_tick(event)
        elif event.event_type == EventType.PRICE_UPDATE:
            self._last_prices[event.symbol] = event.price
            return self.on_price(event)
        elif event.event_type == EventType.ORDER_FILLED:
            return self.on_fill(event)
        elif event.event_type in (EventType.POSITION_OPENED,
                                   EventType.POSITION_UPDATED,
                                   EventType.POSITION_CLOSED):
            return self.on_position(event)

        return None

    def on_bar(self, bar: 'BarEvent') -> Optional[Signal]:
        """Handle bar event. Override in subclass."""
        return None

    def on_tick(self, tick: 'TickEvent') -> Optional[Signal]:
        """Handle tick event. Override in subclass."""
        return None

    def on_price(self, price: 'PriceUpdateEvent') -> Optional[Signal]:
        """Handle unified price event. Override in subclass."""
        return None

    def on_fill(self, fill: 'FillEvent') -> Optional[Signal]:
        """Handle fill event. Override in subclass."""
        return None

    def on_position(self, position: 'PositionEvent') -> Optional[Signal]:
        """Handle position event. Override in subclass."""
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def create_signal(
        self,
        signal_type: SignalType,
        symbol: str,
        price: float,
        stop_loss: float = 0,
        target: float = 0,
        confidence: float = 0.5,
        reason: str = ""
    ) -> Signal:
        """Helper to create a Signal object."""
        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            price=price,
            stop_loss=stop_loss,
            target=target,
            confidence=confidence,
            reason=reason
        )

    def _update_bar_history(self, bar: 'BarEvent') -> None:
        """Add bar to historical data."""
        symbol = bar.symbol
        new_row = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }])

        if symbol not in self._bars:
            self._bars[symbol] = new_row
        else:
            self._bars[symbol] = pd.concat(
                [self._bars[symbol], new_row],
                ignore_index=True
            )

        self._last_prices[symbol] = bar.close

    def set_event_bus(self, bus) -> None:
        """Set the event bus."""
        self._event_bus = bus

    def emit_signal(self, signal: Signal) -> None:
        """Emit a signal to the event bus."""
        if self._event_bus is None:
            return

        from core.events.events import SignalEvent, SignalType as EventSignalType

        signal_type_map = {
            SignalType.BUY: EventSignalType.BUY,
            SignalType.SELL: EventSignalType.SELL,
            SignalType.HOLD: EventSignalType.HOLD,
            SignalType.EXIT: EventSignalType.EXIT,
            SignalType.EXIT_LONG: EventSignalType.EXIT_LONG,
            SignalType.EXIT_SHORT: EventSignalType.EXIT_SHORT,
        }

        event_signal = SignalEvent(
            signal_type=signal_type_map.get(signal.signal_type, EventSignalType.HOLD),
            symbol=signal.symbol,
            price=signal.price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            reason=signal.reason,
            strategy_name=self.name,
            timestamp=signal.timestamp
        )

        self._event_bus.publish(event_signal)

    def get_bars(self, symbol: str, count: int = None) -> Optional[pd.DataFrame]:
        """Get historical bars for a symbol."""
        if symbol not in self._bars:
            return None
        bars = self._bars[symbol]
        if count is not None:
            return bars.tail(count)
        return bars

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price."""
        return self._last_prices.get(symbol)

    def reset(self) -> None:
        """Reset strategy state."""
        self._positions.clear()
        self._bars.clear()
        self._last_prices.clear()

    def __str__(self) -> str:
        return f"{self.emoji} {self.name}"
