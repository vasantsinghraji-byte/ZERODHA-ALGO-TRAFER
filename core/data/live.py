"""
Live Data Source for Real-Time Trading.

Converts WebSocket ticks to events for the event-driven trading engine.
Wraps the existing LiveFeed and adds:
- TickEvent emission
- Bar aggregation from ticks
- Event bus integration

The key insight: Live ticks are converted to the SAME event types
as historical data, so strategies use identical code.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, Generator, List, Optional, Union

from core.events import BarEvent, TickEvent, EventBus, PriceUpdateEvent
from .source import (
    DataSource,
    DataSourceConfig,
    DataSourceMode,
    DataSourceState,
)


logger = logging.getLogger(__name__)


@dataclass
class BarAggregator:
    """
    Aggregates ticks into OHLCV bars.

    Collects ticks within a time window and produces a BarEvent
    when the window closes.
    """
    symbol: str
    timeframe: str
    current_bar_start: Optional[datetime] = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    bar_index: int = 0

    def add_tick(self, price: float, volume: int, timestamp: datetime) -> Optional[BarEvent]:
        """
        Add a tick to the current bar.

        Args:
            price: Tick price
            volume: Tick volume
            timestamp: Tick timestamp

        Returns:
            BarEvent if bar is complete, None otherwise
        """
        bar_start = self._get_bar_start(timestamp)

        # Check if we need to close the current bar
        completed_bar = None
        if self.current_bar_start is not None and bar_start > self.current_bar_start:
            # Bar is complete - create event
            completed_bar = self._create_bar_event()
            # Reset for new bar
            self._reset()
            self.bar_index += 1

        # Start new bar if needed
        if self.current_bar_start is None:
            self.current_bar_start = bar_start
            self.open = price
            self.high = price
            self.low = price

        # Update OHLCV
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1

        return completed_bar

    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the bar containing this timestamp."""
        if self.timeframe == '1m':
            return timestamp.replace(second=0, microsecond=0)
        elif self.timeframe == '5m':
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif self.timeframe == '15m':
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif self.timeframe == '30m':
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif self.timeframe == '1h':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif self.timeframe == '1d':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default to 1 minute
            return timestamp.replace(second=0, microsecond=0)

    def _create_bar_event(self) -> BarEvent:
        """Create a BarEvent from current aggregated data."""
        return BarEvent(
            symbol=self.symbol,
            timestamp=self.current_bar_start,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bar_index=self.bar_index,
            source="live"
        )

    def _reset(self):
        """Reset aggregator for new bar."""
        self.current_bar_start = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0
        self.tick_count = 0

    def flush(self) -> Optional[BarEvent]:
        """Flush current bar (force close)."""
        if self.current_bar_start is not None and self.tick_count > 0:
            bar = self._create_bar_event()
            self._reset()
            self.bar_index += 1
            return bar
        return None


class LiveDataSource(DataSource):
    """
    Live data source using WebSocket.

    Wraps the existing LiveFeed and converts ticks to events.
    Optionally aggregates ticks into bars.

    Usage:
        config = DataSourceConfig(
            symbols=['RELIANCE', 'TCS'],
            timeframe='1m',
            emit_ticks=True,
            emit_bars=True
        )

        source = LiveDataSource(
            config,
            event_bus,
            api_key='your_api_key',
            access_token='your_access_token'
        )

        # Start in background (non-blocking)
        thread = source.start_async()

        # Or start blocking
        source.start()
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        api_key: str = "",
        access_token: str = "",
        live_feed: Optional[object] = None,  # Existing LiveFeed instance
        symbol_to_token: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize live data source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            api_key: Zerodha API key
            access_token: Zerodha access token
            live_feed: Existing LiveFeed instance (optional)
            symbol_to_token: Mapping of symbol to instrument token
        """
        super().__init__(config, event_bus, DataSourceMode.LIVE)

        self._api_key = api_key
        self._access_token = access_token
        self._live_feed = live_feed
        self._symbol_to_token = symbol_to_token or {}
        self._token_to_symbol: Dict[int, str] = {}

        # Bar aggregators for each symbol
        self._aggregators: Dict[str, BarAggregator] = {}

        # Event queue for thread-safe tick handling
        self._tick_queue: List[TickEvent] = []
        self._tick_lock = threading.Lock()

        # Control flags
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to WebSocket.

        Returns:
            True if connection successful
        """
        logger.info("Connecting to live feed...")

        try:
            # Create LiveFeed if not provided
            if self._live_feed is None:
                from core.live_feed import LiveFeed
                self._live_feed = LiveFeed(
                    api_key=self._api_key,
                    access_token=self._access_token,
                    debug=False
                )

            # Build reverse token map
            self._token_to_symbol = {v: k for k, v in self._symbol_to_token.items()}

            # Initialize aggregators
            for symbol in self.config.symbols:
                self._aggregators[symbol] = BarAggregator(
                    symbol=symbol,
                    timeframe=self.config.timeframe
                )

            # Set up tick callback
            original_on_tick = self._live_feed.on_tick
            self._live_feed.on_tick = self._handle_tick

            # Connect
            if hasattr(self._live_feed, 'connect'):
                result = self._live_feed.connect()
                if not result:
                    logger.error("Failed to connect to WebSocket")
                    return False

            # Subscribe to symbols
            if self._symbol_to_token:
                tokens = [self._symbol_to_token[s] for s in self.config.symbols if s in self._symbol_to_token]
                if hasattr(self._live_feed, 'subscribe'):
                    self._live_feed.subscribe(tokens)

            self._connected = True
            logger.info(f"Connected to live feed for {len(self.config.symbols)} symbols")
            return True

        except ImportError:
            logger.warning("LiveFeed not available - using simulation mode")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to live feed: {e}", exc_info=True)
            return False

    def disconnect(self):
        """Disconnect from WebSocket."""
        if self._live_feed and hasattr(self._live_feed, 'disconnect'):
            # Flush any pending bars
            for symbol, aggregator in self._aggregators.items():
                bar = aggregator.flush()
                if bar:
                    self._dispatch_event(bar)

            self._live_feed.disconnect()
            self._connected = False
            logger.info("Disconnected from live feed")

    def _emit_events(self) -> Generator[Union[BarEvent, TickEvent], None, None]:
        """
        Generate events from live feed.

        Runs continuously until stopped, yielding tick and bar events.
        """
        logger.info("Starting live event emission...")

        while self._running:
            # Process queued ticks
            ticks_to_process = []
            with self._tick_lock:
                if self._tick_queue:
                    ticks_to_process = self._tick_queue.copy()
                    self._tick_queue.clear()

            for tick_event in ticks_to_process:
                # Emit tick event if configured
                if self.config.emit_ticks:
                    yield tick_event

                # Aggregate into bars if configured
                if self.config.emit_bars and tick_event.symbol in self._aggregators:
                    aggregator = self._aggregators[tick_event.symbol]
                    bar_event = aggregator.add_tick(
                        price=tick_event.last_price,
                        volume=tick_event.volume,
                        timestamp=tick_event.timestamp
                    )
                    if bar_event:
                        yield bar_event

            # Small sleep to prevent busy loop
            if not ticks_to_process:
                time.sleep(0.01)

    def _handle_tick(self, tick):
        """
        Handle incoming tick from LiveFeed.

        Converts LiveFeed Tick to TickEvent and queues it.
        """
        try:
            # Get symbol from token
            symbol = getattr(tick, 'symbol', '')
            if not symbol and hasattr(tick, 'instrument_token'):
                symbol = self._token_to_symbol.get(tick.instrument_token, '')

            if not symbol:
                return

            # Create TickEvent
            tick_event = TickEvent(
                symbol=symbol,
                instrument_token=getattr(tick, 'instrument_token', 0),
                last_price=getattr(tick, 'last_price', 0.0),
                volume=getattr(tick, 'volume', 0),
                bid=getattr(tick, 'buy_quantity', 0),  # Simplified
                ask=getattr(tick, 'sell_quantity', 0),  # Simplified
                timestamp=getattr(tick, 'timestamp', datetime.now()),
                source="live"
            )

            # Queue tick
            with self._tick_lock:
                self._tick_queue.append(tick_event)

        except Exception as e:
            logger.error(f"Error handling tick: {e}")

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def subscribe(self, symbols: List[str]):
        """Subscribe to additional symbols."""
        for symbol in symbols:
            if symbol not in self.config.symbols:
                self.config.symbols.append(symbol)

            if symbol not in self._aggregators:
                self._aggregators[symbol] = BarAggregator(
                    symbol=symbol,
                    timeframe=self.config.timeframe
                )

        # Subscribe in LiveFeed
        if self._live_feed and self._symbol_to_token:
            tokens = [self._symbol_to_token[s] for s in symbols if s in self._symbol_to_token]
            if tokens and hasattr(self._live_feed, 'subscribe'):
                self._live_feed.subscribe(tokens)

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self.config.symbols:
                self.config.symbols.remove(symbol)

            if symbol in self._aggregators:
                # Flush and remove aggregator
                bar = self._aggregators[symbol].flush()
                if bar:
                    self._dispatch_event(bar)
                del self._aggregators[symbol]

        # Unsubscribe in LiveFeed
        if self._live_feed and self._symbol_to_token:
            tokens = [self._symbol_to_token[s] for s in symbols if s in self._symbol_to_token]
            if tokens and hasattr(self._live_feed, 'unsubscribe'):
                self._live_feed.unsubscribe(tokens)

    def get_latest_tick(self, symbol: str) -> Optional[TickEvent]:
        """Get the latest tick for a symbol."""
        if self._live_feed and hasattr(self._live_feed, 'get_latest_tick'):
            tick = self._live_feed.get_latest_tick(symbol)
            if tick:
                return TickEvent(
                    symbol=symbol,
                    last_price=tick.last_price,
                    timestamp=tick.timestamp,
                    source="live"
                )
        return None

    def set_symbol_mapping(self, symbol_to_token: Dict[str, int]):
        """Set symbol to token mapping."""
        self._symbol_to_token = symbol_to_token
        self._token_to_symbol = {v: k for k, v in symbol_to_token.items()}

    @property
    def is_connected(self) -> bool:
        """Check if connected to WebSocket."""
        return self._connected


class SimulatedLiveSource(LiveDataSource):
    """
    Simulated live data source for testing.

    Generates synthetic tick data for testing live trading logic
    without connecting to a real data feed.
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        base_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.001,
        tick_interval: float = 1.0,
    ):
        """
        Initialize simulated source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            base_prices: Starting prices for each symbol
            volatility: Price volatility (std dev as fraction of price)
            tick_interval: Seconds between ticks
        """
        super().__init__(config, event_bus)
        self._base_prices = base_prices or {}
        self._current_prices: Dict[str, float] = {}
        self._volatility = volatility
        self._tick_interval = tick_interval
        self.mode = DataSourceMode.PAPER

    def connect(self) -> bool:
        """Initialize simulated connection."""
        import random

        # Initialize prices
        for symbol in self.config.symbols:
            if symbol in self._base_prices:
                self._current_prices[symbol] = self._base_prices[symbol]
            else:
                # Random price between 100 and 5000
                self._current_prices[symbol] = random.uniform(100, 5000)

            self._aggregators[symbol] = BarAggregator(
                symbol=symbol,
                timeframe=self.config.timeframe
            )

        self._connected = True
        logger.info(f"Simulated live source connected for {len(self.config.symbols)} symbols")
        return True

    def _emit_events(self) -> Generator[Union[BarEvent, TickEvent], None, None]:
        """Generate simulated tick events."""
        import random

        logger.info("Starting simulated tick generation...")

        while self._running:
            for symbol in self.config.symbols:
                # Generate random price movement
                price = self._current_prices[symbol]
                change = random.gauss(0, price * self._volatility)
                new_price = max(0.01, price + change)
                self._current_prices[symbol] = new_price

                # Create tick event
                tick_event = TickEvent(
                    symbol=symbol,
                    last_price=new_price,
                    volume=random.randint(1, 1000),
                    timestamp=datetime.now(),
                    source="simulated"
                )

                # Emit tick if configured
                if self.config.emit_ticks:
                    yield tick_event

                # Aggregate into bars
                if self.config.emit_bars and symbol in self._aggregators:
                    bar_event = self._aggregators[symbol].add_tick(
                        price=new_price,
                        volume=tick_event.volume,
                        timestamp=tick_event.timestamp
                    )
                    if bar_event:
                        yield bar_event

            time.sleep(self._tick_interval)
