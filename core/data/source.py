"""
Unified Data Source Interface for Event-Driven Trading Engine.

This module provides the abstract interface that ALL data sources must implement.
The key insight: the trading engine doesn't care WHERE data comes from.
It just receives events.

Data Sources:
- HistoricalDataSource: Replay historical bars for backtesting
- LiveDataSource: Stream real-time ticks from WebSocket
- ReplayDataSource: Replay recorded live sessions for debugging

The engine uses the SAME code for all modes - only the data source changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Union,
)
import threading
import logging

import pandas as pd

from core.events import (
    EventBus,
    BarEvent,
    TickEvent,
    PriceUpdateEvent,
    MarketOpenEvent,
    MarketCloseEvent,
)


logger = logging.getLogger(__name__)


class DataSourceMode(Enum):
    """Operating mode for data source."""
    BACKTEST = auto()      # Historical data replay
    LIVE = auto()          # Real-time WebSocket
    PAPER = auto()         # Live data, simulated execution
    REPLAY = auto()        # Replay recorded session


class DataSourceState(Enum):
    """State of the data source."""
    IDLE = auto()          # Not started
    CONNECTING = auto()    # Connecting to data source
    RUNNING = auto()       # Actively emitting events
    PAUSED = auto()        # Temporarily paused
    STOPPED = auto()       # Stopped
    ERROR = auto()         # Error state


@dataclass
class DataSourceConfig:
    """
    Configuration for data sources.

    Attributes:
        symbols: List of symbols to subscribe to
        timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d)
        start_date: Start date for historical data
        end_date: End date for historical data
        emit_bars: Whether to emit BarEvents
        emit_ticks: Whether to emit TickEvents (live only)
        warmup_bars: Number of bars to load before start for indicator warmup
        speed_multiplier: Playback speed (1.0 = realtime, 0 = as fast as possible)
    """
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    emit_bars: bool = True
    emit_ticks: bool = False
    warmup_bars: int = 50
    speed_multiplier: float = 0.0  # 0 = no delay (backtest mode)

    def __post_init__(self):
        # Default date range if not specified
        if self.end_date is None:
            self.end_date = datetime.now()
        if self.start_date is None:
            self.start_date = self.end_date - timedelta(days=365)


@dataclass
class DataSourceStats:
    """Statistics for data source performance."""
    events_emitted: int = 0
    bars_processed: int = 0
    ticks_processed: int = 0
    start_time: Optional[datetime] = None
    last_event_time: Optional[datetime] = None
    errors: int = 0


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    A DataSource is responsible for:
    1. Loading/connecting to market data
    2. Converting data to events (BarEvent, TickEvent)
    3. Emitting events to the EventBus

    The trading engine doesn't know or care what type of DataSource is used.
    This is the key to having the SAME strategy code work in backtest and live.

    Usage:
        # For backtesting
        source = HistoricalDataSource(config, event_bus)
        source.start()  # Emits all historical bars as events

        # For live trading
        source = LiveDataSource(config, event_bus)
        source.start()  # Connects to WebSocket and streams ticks

        # Same event handlers work for both!
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        mode: DataSourceMode = DataSourceMode.BACKTEST
    ):
        """
        Initialize data source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            mode: Operating mode
        """
        self.config = config
        self.event_bus = event_bus
        self.mode = mode

        # State
        self._state = DataSourceState.IDLE
        self._state_lock = threading.Lock()

        # Statistics
        self._stats = DataSourceStats()

        # Control
        self._running = False
        self._paused = False

        # Callbacks (alternative to EventBus)
        self._on_bar: Optional[Callable[[BarEvent], None]] = None
        self._on_tick: Optional[Callable[[TickEvent], None]] = None
        self._on_price: Optional[Callable[[PriceUpdateEvent], None]] = None

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.

        For historical: Load data from files/database
        For live: Connect to WebSocket

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the data source."""
        pass

    @abstractmethod
    def _emit_events(self) -> Generator[Union[BarEvent, TickEvent], None, None]:
        """
        Generate events from the data source.

        This is the core method that produces events.
        Subclasses implement this to yield BarEvent or TickEvent.

        Yields:
            BarEvent or TickEvent
        """
        pass

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """
        Start emitting events.

        Connects to data source and begins emitting events.
        For backtest: Runs through all historical data
        For live: Starts streaming and runs until stopped
        """
        if self._state == DataSourceState.RUNNING:
            logger.warning("DataSource already running")
            return

        logger.info(f"Starting {self.__class__.__name__} in {self.mode.name} mode")

        # Connect
        if not self.connect():
            self._set_state(DataSourceState.ERROR)
            raise RuntimeError("Failed to connect to data source")

        self._set_state(DataSourceState.RUNNING)
        self._running = True
        self._stats.start_time = datetime.now()

        # Emit market open event
        if self.event_bus:
            self.event_bus.emit(MarketOpenEvent(source=self.mode.name.lower()))

        try:
            # Emit all events
            for event in self._emit_events():
                if not self._running:
                    break

                # Handle pause
                while self._paused and self._running:
                    threading.Event().wait(0.1)

                # Emit event
                self._dispatch_event(event)

        except Exception as e:
            logger.error(f"Error in data source: {e}", exc_info=True)
            self._stats.errors += 1
            self._set_state(DataSourceState.ERROR)
            raise

        finally:
            # Emit market close event
            if self.event_bus:
                self.event_bus.emit(MarketCloseEvent(source=self.mode.name.lower()))

            self._running = False
            self._set_state(DataSourceState.STOPPED)
            self.disconnect()

        logger.info(f"DataSource stopped. Stats: {self._stats}")

    def start_async(self) -> threading.Thread:
        """
        Start emitting events in a background thread.

        Returns:
            Thread running the data source
        """
        thread = threading.Thread(
            target=self.start,
            name=f"DataSource-{self.mode.name}",
            daemon=True
        )
        thread.start()
        return thread

    def stop(self):
        """Stop emitting events."""
        logger.info("Stopping DataSource...")
        self._running = False

    def pause(self):
        """Pause event emission."""
        self._paused = True
        self._set_state(DataSourceState.PAUSED)

    def resume(self):
        """Resume event emission."""
        self._paused = False
        self._set_state(DataSourceState.RUNNING)

    # =========================================================================
    # Event Dispatch
    # =========================================================================

    def _dispatch_event(self, event: Union[BarEvent, TickEvent]):
        """Dispatch event to bus and callbacks."""
        self._stats.events_emitted += 1
        self._stats.last_event_time = event.timestamp

        if isinstance(event, BarEvent):
            self._stats.bars_processed += 1
        elif isinstance(event, TickEvent):
            self._stats.ticks_processed += 1

        # Emit to EventBus
        if self.event_bus:
            self.event_bus.emit(event)

            # Also emit as PriceUpdateEvent for unified handling
            if isinstance(event, BarEvent):
                price_event = PriceUpdateEvent.from_bar(event)
                self.event_bus.emit(price_event)
            elif isinstance(event, TickEvent):
                price_event = PriceUpdateEvent.from_tick(event)
                self.event_bus.emit(price_event)

        # Call direct callbacks
        if isinstance(event, BarEvent) and self._on_bar:
            self._on_bar(event)
        elif isinstance(event, TickEvent) and self._on_tick:
            self._on_tick(event)

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_bar(self, callback: Callable[[BarEvent], None]):
        """Register callback for bar events."""
        self._on_bar = callback

    def on_tick(self, callback: Callable[[TickEvent], None]):
        """Register callback for tick events."""
        self._on_tick = callback

    def on_price(self, callback: Callable[[PriceUpdateEvent], None]):
        """Register callback for price update events."""
        self._on_price = callback

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, state: DataSourceState):
        """Set state thread-safely."""
        with self._state_lock:
            self._state = state

    @property
    def state(self) -> DataSourceState:
        """Get current state."""
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Check if source is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if source is paused."""
        return self._paused

    @property
    def stats(self) -> DataSourceStats:
        """Get statistics."""
        return self._stats

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return self.config.symbols.copy()

    def add_symbol(self, symbol: str):
        """Add symbol to subscription list."""
        if symbol not in self.config.symbols:
            self.config.symbols.append(symbol)

    def remove_symbol(self, symbol: str):
        """Remove symbol from subscription list."""
        if symbol in self.config.symbols:
            self.config.symbols.remove(symbol)


# =============================================================================
# Helper Functions
# =============================================================================

def create_bar_event(
    symbol: str,
    timestamp: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    timeframe: str = "1d",
    bar_index: int = 0,
    source: str = "unknown"
) -> BarEvent:
    """
    Helper to create a BarEvent from OHLCV data.

    Args:
        symbol: Stock symbol
        timestamp: Bar timestamp
        open_: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume
        timeframe: Bar timeframe
        bar_index: Index in data series
        source: Data source name

    Returns:
        BarEvent
    """
    return BarEvent(
        symbol=symbol,
        timestamp=timestamp,
        timeframe=timeframe,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_index=bar_index,
        source=source
    )


def create_tick_event(
    symbol: str,
    last_price: float,
    volume: int = 0,
    bid: float = 0.0,
    ask: float = 0.0,
    timestamp: Optional[datetime] = None,
    source: str = "unknown"
) -> TickEvent:
    """
    Helper to create a TickEvent.

    Args:
        symbol: Stock symbol
        last_price: Last traded price
        volume: Volume
        bid: Bid price
        ask: Ask price
        timestamp: Tick timestamp
        source: Data source name

    Returns:
        TickEvent
    """
    return TickEvent(
        symbol=symbol,
        last_price=last_price,
        volume=volume,
        bid=bid,
        ask=ask,
        timestamp=timestamp or datetime.now(),
        source=source
    )


def dataframe_to_bar_events(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1d",
    source: str = "historical"
) -> Generator[BarEvent, None, None]:
    """
    Convert a pandas DataFrame to BarEvents.

    Expects DataFrame with columns: open, high, low, close, volume
    and a datetime index.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        timeframe: Bar timeframe
        source: Data source name

    Yields:
        BarEvent for each row
    """
    # Normalize column names
    df = df.copy()
    df.columns = df.columns.str.lower()

    for idx, (timestamp, row) in enumerate(df.iterrows()):
        yield BarEvent(
            symbol=symbol,
            timestamp=timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp),
            timeframe=timeframe,
            open=float(row.get('open', row.get('o', 0))),
            high=float(row.get('high', row.get('h', 0))),
            low=float(row.get('low', row.get('l', 0))),
            close=float(row.get('close', row.get('c', 0))),
            volume=int(row.get('volume', row.get('v', 0))),
            bar_index=idx,
            source=source
        )
