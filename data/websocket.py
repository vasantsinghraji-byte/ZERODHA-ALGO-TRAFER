import logging
import threading
from typing import Dict, Set, Optional, Callable, List
from datetime import datetime
import time

from kiteconnect import KiteTicker

from config.config import settings
from .processor import TickProcessor
from .models import Tick, MarketDepth, DepthItem

logger = logging.getLogger(__name__)

# Subscription modes for Zerodha WebSocket
# MODE_LTP: Last Traded Price only (minimal data)
# MODE_QUOTE: Quote data including market depth (Level 2)
# MODE_FULL: Full tick data including depth + more fields
SUBSCRIPTION_MODE = "MODE_QUOTE"  # Use MODE_QUOTE for Level 2 depth

class MarketDataStream:
    """WebSocket manager for real-time market data streaming"""
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kws: Optional[KiteTicker] = None
        self.processor = TickProcessor()
        
        # Connection state
        self.subscribed_tokens: Set[int] = set()
        self.is_connected = False
        self.reconnect_count = 0
        self.max_reconnects = 5
        
        # Callbacks
        self.callbacks: Dict[str, Callable] = {}
        
        # Threading
        self.connection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    def _setup_handlers(self) -> None:
        """Setup WebSocket event handlers"""
        if not self.kws:
            return
            
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_message = self._on_message
        self.kws.on_reconnect = self._on_reconnect
        self.kws.on_noreconnect = self._on_noreconnect
        self.kws.on_order_update = self._on_order_update

    def _on_connect(self, ws, response) -> None:
        """Called when WebSocket connects successfully"""
        logger.info("WebSocket connected successfully")
        self.is_connected = True
        self.reconnect_count = 0

        # Subscribe to pending tokens with MODE_QUOTE for Level 2 depth
        if self.subscribed_tokens:
            token_list = list(self.subscribed_tokens)
            logger.info(f"Subscribing to {len(token_list)} instruments with {SUBSCRIPTION_MODE}")
            self.kws.subscribe(token_list)
            # Use MODE_QUOTE for order book depth (Level 2 data)
            # MODE_QUOTE includes: LTP, OHLC, volume, buy/sell qty, and market depth (5 levels)
            self.kws.set_mode(self.kws.MODE_QUOTE, token_list)

    def _on_close(self, ws, code, reason) -> None:
        """Called when WebSocket connection closes"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        self.is_connected = False

    def _on_error(self, ws, code, reason) -> None:
        """Called on WebSocket error"""
        logger.error(f"WebSocket error: {code} - {reason}")
        
    def _on_message(self, ws, ticks) -> None:
        """Called when tick data is received"""
        try:
            for tick_data in ticks:
                tick = self._parse_tick(tick_data)
                if tick:
                    # Process tick asynchronously
                    self.processor.process_tick(tick)
                    
                    # Notify registered callbacks
                    for callback_name, callback in self.callbacks.items():
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"Callback {callback_name} error: {e}")
                            
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")

    def _on_reconnect(self, ws, attempts_count) -> None:
        """Called when reconnection attempt is made"""
        logger.info(f"Reconnecting... Attempt {attempts_count}")
        self.reconnect_count = attempts_count

    def _on_noreconnect(self, ws) -> None:
        """Called when max reconnection attempts reached"""
        logger.error("Max reconnection attempts reached. WebSocket stopped.")
        self.is_connected = False

    def _on_order_update(self, ws, data) -> None:
        """Called on order update"""
        logger.info(f"Order update: {data}")

    def _parse_tick(self, tick_data: Dict) -> Optional[Tick]:
        """Parse raw tick data into Tick model with Level 2 depth"""
        try:
            ohlc = tick_data.get('ohlc', {})

            # Parse market depth (Level 2 order book)
            depth = None
            raw_depth = tick_data.get('depth', {})
            if raw_depth:
                buy_depth = [
                    DepthItem(
                        price=level.get('price', 0),
                        quantity=level.get('quantity', 0),
                        orders=level.get('orders', 0)
                    )
                    for level in raw_depth.get('buy', [])
                ]
                sell_depth = [
                    DepthItem(
                        price=level.get('price', 0),
                        quantity=level.get('quantity', 0),
                        orders=level.get('orders', 0)
                    )
                    for level in raw_depth.get('sell', [])
                ]
                depth = MarketDepth(buy=buy_depth, sell=sell_depth)

            return Tick(
                instrument_token=tick_data['instrument_token'],
                timestamp=datetime.fromtimestamp(tick_data['exchange_timestamp'].timestamp()),
                last_price=tick_data['last_price'],
                volume=tick_data.get('volume', 0),
                buy_quantity=tick_data.get('buy_quantity', 0),
                sell_quantity=tick_data.get('sell_quantity', 0),
                open=ohlc.get('open'),
                high=ohlc.get('high'),
                low=ohlc.get('low'),
                close=ohlc.get('close'),
                depth=depth,
                average_price=tick_data.get('average_price'),
                change=tick_data.get('change'),
                last_traded_quantity=tick_data.get('last_traded_quantity'),
                oi=tick_data.get('oi'),
                oi_day_high=tick_data.get('oi_day_high'),
                oi_day_low=tick_data.get('oi_day_low'),
            )
        except Exception as e:
            logger.error(f"Error parsing tick: {e}")
            return None

    def subscribe(self, instrument_tokens: List[int]) -> None:
        """Subscribe to instrument tokens with Level 2 depth"""
        new_tokens = set(instrument_tokens) - self.subscribed_tokens
        if new_tokens:
            self.subscribed_tokens.update(new_tokens)
            if self.is_connected and self.kws:
                token_list = list(new_tokens)
                self.kws.subscribe(token_list)
                # Use MODE_QUOTE for order book depth (Level 2 data)
                self.kws.set_mode(self.kws.MODE_QUOTE, token_list)
                logger.info(f"Subscribed to {len(token_list)} instruments with Level 2 depth")

    def unsubscribe(self, instrument_tokens: List[int]) -> None:
        """Unsubscribe from instrument tokens"""
        tokens_to_remove = set(instrument_tokens) & self.subscribed_tokens
        if tokens_to_remove:
            self.subscribed_tokens.difference_update(tokens_to_remove)
            if self.is_connected and self.kws:
                self.kws.unsubscribe(list(tokens_to_remove))
                logger.info(f"Unsubscribed from {len(tokens_to_remove)} instruments")

    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback for tick events"""
        self.callbacks[name] = callback
        logger.info(f"Registered callback: {name}")

    def unregister_callback(self, name: str) -> None:
        """Unregister a callback"""
        if name in self.callbacks:
            del self.callbacks[name]
            logger.info(f"Unregistered callback: {name}")

    def start(self) -> None:
        """Start WebSocket connection"""
        if self.is_connected:
            logger.warning("WebSocket already connected")
            return
            
        try:
            self.kws = KiteTicker(self.api_key, self.access_token)
            self._setup_handlers()
            
            logger.info("Starting WebSocket connection...")
            self.kws.connect(threaded=True)
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if not self.is_connected:
                raise ConnectionError("Failed to establish WebSocket connection")
                
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            raise

    def stop(self) -> None:
        """Stop WebSocket connection"""
        try:
            if self.kws:
                self.kws.close()
            self.is_connected = False
            self._stop_event.set()
            logger.info("WebSocket stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {e}")

    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            "connected": self.is_connected,
            "subscribed_tokens": len(self.subscribed_tokens),
            "reconnect_count": self.reconnect_count,
            "callbacks": list(self.callbacks.keys())
        }
