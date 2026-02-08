# -*- coding: utf-8 -*-
"""
Live Data Feed - Real-Time Stock Prices!
=========================================
Gets live prices as they happen, like watching a scoreboard.

Uses Zerodha's WebSocket to stream live data.
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

try:
    from kiteconnect import KiteTicker
except ImportError:
    KiteTicker = None


# ============== DATA CLASSES ==============

@dataclass
class Tick:
    """A single price update (tick)"""
    instrument_token: int
    symbol: str = ""
    last_price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0
    # WARNING: Mutable defaults in dataclasses must use field(default_factory=...)
    # Using `ohlc: Dict = {}` would share ONE dict across ALL Tick instances!
    ohlc: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_green(self) -> bool:
        """Is the price up?"""
        return self.change >= 0

    def __str__(self):
        arrow = "^" if self.is_green else "v"
        return f"{self.symbol}: Rs.{self.last_price:.2f} {arrow}{abs(self.change_percent):.2f}%"


# ============== LIVE FEED ==============

class LiveFeed:
    """
    Real-time price feed using WebSocket.

    Think of it like a radio that broadcasts prices live!

    Usage:
        feed = LiveFeed(api_key, access_token)
        feed.subscribe(["RELIANCE", "TCS"])
        feed.on_tick = my_callback_function
        feed.start()
    """

    def __init__(
        self,
        api_key: str = "",
        access_token: str = "",
        debug: bool = False
    ):
        """
        Initialize Live Feed.

        Args:
            api_key: Zerodha API key
            access_token: Zerodha access token
            debug: Enable debug logging
        """
        self.api_key = api_key
        self.access_token = access_token
        self.debug = debug

        # State
        self._ticker: Optional[Any] = None
        self._connected = False
        self._subscribed_tokens: List[int] = []
        self._symbol_map: Dict[int, str] = {}  # token -> symbol

        # Latest ticks (rolling buffer)
        self._latest_ticks: Dict[int, Tick] = {}
        self._tick_history: Dict[int, deque] = {}  # Keep last 100 ticks per symbol
        self._history_size = 100

        # Callbacks
        self.on_tick: Optional[Callable[[Tick], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Thread lock
        self._lock = threading.Lock()

    # ============== CONNECTION ==============

    def connect(self) -> bool:
        """
        Connect to Zerodha WebSocket.

        Returns:
            True if connection initiated
        """
        if not KiteTicker:
            print("KiteTicker not available. Install kiteconnect.")
            return False

        if not self.api_key or not self.access_token:
            print("API key and access token required!")
            return False

        try:
            self._ticker = KiteTicker(self.api_key, self.access_token)

            # Set up callbacks
            self._ticker.on_ticks = self._handle_ticks
            self._ticker.on_connect = self._handle_connect
            self._ticker.on_close = self._handle_disconnect
            self._ticker.on_error = self._handle_error

            # Start in background thread
            self._ticker.connect(threaded=True)

            print("Connecting to live feed...")
            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """
        Disconnect from WebSocket.

        Properly cleans up all resources to prevent memory leaks:
        - Unsubscribes from all tokens
        - Clears callbacks to release references
        - Waits for background thread to finish
        - Clears internal tick data
        """
        if self._ticker:
            try:
                # Unsubscribe all before closing to prevent orphan subscriptions
                if self._subscribed_tokens:
                    try:
                        self._ticker.unsubscribe(self._subscribed_tokens)
                    except Exception:
                        pass  # Best effort - connection may already be closing

                # Clear callbacks to prevent memory leaks from circular references
                self._ticker.on_ticks = None
                self._ticker.on_connect = None
                self._ticker.on_close = None
                self._ticker.on_error = None

                # Close connection with proper close code
                self._ticker.close(code=1000, reason="Normal closure")

                # Wait for WebSocket thread to finish (with timeout to prevent hanging)
                if hasattr(self._ticker, '_ws_thread') and self._ticker._ws_thread:
                    self._ticker._ws_thread.join(timeout=5)

            except Exception as e:
                if self.debug:
                    print(f"Error during disconnect: {e}")
            finally:
                self._ticker = None
                self._connected = False

        # Clear internal state to free memory
        with self._lock:
            self._latest_ticks.clear()
            self._tick_history.clear()
            self._subscribed_tokens.clear()
            self._symbol_map.clear()

        print("Disconnected from live feed")

    def start(self) -> bool:
        """Start the live feed (alias for connect)"""
        return self.connect()

    def stop(self):
        """Stop the live feed (alias for disconnect)"""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected

    # ============== SUBSCRIPTION ==============

    def subscribe(self, tokens: List[int], symbols: Optional[Dict[int, str]] = None):
        """
        Subscribe to instruments.

        Args:
            tokens: List of instrument tokens
            symbols: Optional dict mapping token -> symbol name
        """
        self._subscribed_tokens.extend(tokens)

        if symbols:
            self._symbol_map.update(symbols)

        # Initialize tick history
        for token in tokens:
            if token not in self._tick_history:
                self._tick_history[token] = deque(maxlen=self._history_size)

        if self._connected and self._ticker:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, tokens)
            print(f"Subscribed to {len(tokens)} instruments")

    def unsubscribe(self, tokens: List[int]):
        """Unsubscribe from instruments"""
        for token in tokens:
            if token in self._subscribed_tokens:
                self._subscribed_tokens.remove(token)

        if self._connected and self._ticker:
            self._ticker.unsubscribe(tokens)
            print(f"Unsubscribed from {len(tokens)} instruments")

    # ============== DATA ACCESS ==============

    def get_tick(self, token: int) -> Optional[Tick]:
        """
        Get latest tick for an instrument.

        Args:
            token: Instrument token

        Returns:
            Latest Tick or None
        """
        with self._lock:
            return self._latest_ticks.get(token)

    def get_price(self, token: int) -> float:
        """Get latest price for an instrument"""
        tick = self.get_tick(token)
        return tick.last_price if tick else 0.0

    def get_all_ticks(self) -> Dict[int, Tick]:
        """Get all latest ticks"""
        with self._lock:
            return self._latest_ticks.copy()

    def get_tick_history(self, token: int) -> List[Tick]:
        """Get tick history for an instrument"""
        with self._lock:
            if token in self._tick_history:
                return list(self._tick_history[token])
            return []

    # ============== INTERNAL HANDLERS ==============

    def _handle_ticks(self, ws, ticks: List[Dict]):
        """Handle incoming ticks"""
        for tick_data in ticks:
            try:
                token = tick_data.get('instrument_token', 0)
                symbol = self._symbol_map.get(token, str(token))

                # Get OHLC data
                ohlc = tick_data.get('ohlc', {})
                last_price = tick_data.get('last_price', 0)

                # Validate price data to prevent division by zero
                if last_price <= 0:
                    if self.debug:
                        print(f"Invalid last_price {last_price} for token {token}, skipping tick")
                    continue

                prev_close = ohlc.get('close', 0)
                if prev_close <= 0:
                    prev_close = last_price  # Use current price as fallback

                # Calculate change - safe now since both prices are validated
                change = last_price - prev_close
                if prev_close > 0:
                    change_pct = (change / prev_close * 100)
                else:
                    change_pct = 0.0

                tick = Tick(
                    instrument_token=token,
                    symbol=symbol,
                    last_price=last_price,
                    change=change,
                    change_percent=change_pct,
                    volume=tick_data.get('volume', 0),
                    buy_quantity=tick_data.get('buy_quantity', 0),
                    sell_quantity=tick_data.get('sell_quantity', 0),
                    ohlc=ohlc,
                    timestamp=datetime.now()
                )

                with self._lock:
                    self._latest_ticks[token] = tick
                    self._tick_history[token].append(tick)

                # Call user callback
                if self.on_tick:
                    self.on_tick(tick)

                if self.debug:
                    print(f"Tick: {tick}")

            except Exception as e:
                if self.debug:
                    print(f"Error processing tick: {e}")

    def _handle_connect(self, ws, response):
        """Handle connection"""
        self._connected = True
        print("Connected to live feed!")

        # Subscribe to instruments
        if self._subscribed_tokens:
            self._ticker.subscribe(self._subscribed_tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, self._subscribed_tokens)

        if self.on_connect:
            self.on_connect()

    def _handle_disconnect(self, ws, code, reason):
        """Handle disconnection"""
        self._connected = False
        print(f"Disconnected from live feed: {reason}")

        if self.on_disconnect:
            self.on_disconnect()

    def _handle_error(self, ws, code, reason):
        """Handle errors"""
        print(f"Live feed error: {reason}")

        if self.on_error:
            self.on_error(reason)


# ============== SIMULATED FEED FOR TESTING ==============

class SimulatedFeed:
    """
    Simulated live feed for testing without real connection.

    Generates fake price movements for testing strategies.
    """

    def __init__(self, symbols: List[str], base_prices: Optional[Dict[str, float]] = None):
        """
        Initialize simulated feed.

        Args:
            symbols: List of symbols to simulate
            base_prices: Starting prices (default: 1000 for all)
        """
        self.symbols = symbols
        self.base_prices = base_prices or {s: 1000.0 for s in symbols}
        self._current_prices = self.base_prices.copy()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_tick: Optional[Callable[[Tick], None]] = None

    def start(self, interval: float = 1.0):
        """
        Start generating simulated ticks.

        Args:
            interval: Seconds between ticks
        """
        self._running = True
        self._thread = threading.Thread(target=self._generate_ticks, args=(interval,))
        self._thread.daemon = True
        self._thread.start()
        print(f"Simulated feed started for {len(self.symbols)} symbols")

    def stop(self):
        """Stop the simulated feed"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("Simulated feed stopped")

    def _generate_ticks(self, interval: float):
        """Generate fake ticks"""
        import random

        while self._running:
            for i, symbol in enumerate(self.symbols):
                # Random walk
                change_pct = random.gauss(0, 0.5)  # 0.5% volatility
                old_price = self._current_prices[symbol]
                new_price = old_price * (1 + change_pct / 100)
                self._current_prices[symbol] = new_price

                tick = Tick(
                    instrument_token=i + 1,
                    symbol=symbol,
                    last_price=new_price,
                    change=new_price - old_price,
                    change_percent=change_pct,
                    volume=random.randint(1000, 10000),
                    ohlc={
                        'open': self.base_prices[symbol],
                        'high': max(new_price, self._current_prices[symbol] * 1.02),
                        'low': min(new_price, self._current_prices[symbol] * 0.98),
                        'close': old_price
                    },
                    timestamp=datetime.now()
                )

                if self.on_tick:
                    self.on_tick(tick)

            time.sleep(interval)

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return self._current_prices.get(symbol, 0)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("LIVE FEED - Test (Simulated)")
    print("=" * 50)

    def on_tick(tick: Tick):
        print(f"  {tick}")

    # Test simulated feed
    feed = SimulatedFeed(["RELIANCE", "TCS", "INFY"])
    feed.on_tick = on_tick

    print("\nStarting simulated feed (5 seconds)...")
    feed.start(interval=1.0)

    time.sleep(5)

    feed.stop()

    print("\n" + "=" * 50)
    print("Live Feed ready!")
    print("=" * 50)
