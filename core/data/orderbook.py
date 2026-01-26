"""
Order Book Processing for Level 2 Market Data.

Handles real-time order book snapshots with bid/ask aggregation,
imbalance calculations, and memory-efficient storage.

Level 2 data provides market depth beyond just the best bid/ask,
revealing supply/demand dynamics and potential support/resistance.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DepthLevel(Enum):
    """Supported order book depth levels"""
    LEVEL_5 = 5
    LEVEL_10 = 10
    LEVEL_20 = 20


class Side(Enum):
    """Order book side"""
    BID = "bid"
    ASK = "ask"


@dataclass
class PriceLevel:
    """
    A single price level in the order book.

    Attributes:
        price: Price at this level
        quantity: Total quantity at this price
        orders: Number of orders at this level (if available)
        timestamp: When this level was last updated
    """
    price: float
    quantity: int
    orders: int = 0
    timestamp: Optional[datetime] = None

    @property
    def value(self) -> float:
        """Total value at this price level"""
        return self.price * self.quantity

    def __eq__(self, other):
        if isinstance(other, PriceLevel):
            return self.price == other.price
        return False

    def __hash__(self):
        return hash(self.price)


@dataclass
class OrderBookSnapshot:
    """
    Complete order book snapshot at a point in time.

    Attributes:
        symbol: Trading symbol
        timestamp: Snapshot timestamp
        bids: List of bid levels (best bid first)
        asks: List of ask levels (best ask first)
        exchange_timestamp: Timestamp from exchange (if available)
    """
    symbol: str
    timestamp: datetime
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    exchange_timestamp: Optional[datetime] = None
    sequence: int = 0

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Best (highest) bid price level"""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Best (lowest) ask price level"""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float:
        """Mid-point between best bid and ask"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0

    @property
    def spread_percent(self) -> float:
        """Spread as percentage of mid price"""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0

    @property
    def total_bid_quantity(self) -> int:
        """Total quantity on bid side"""
        return sum(level.quantity for level in self.bids)

    @property
    def total_ask_quantity(self) -> int:
        """Total quantity on ask side"""
        return sum(level.quantity for level in self.asks)

    @property
    def total_bid_value(self) -> float:
        """Total value on bid side"""
        return sum(level.value for level in self.bids)

    @property
    def total_ask_value(self) -> float:
        """Total value on ask side"""
        return sum(level.value for level in self.asks)

    @property
    def depth(self) -> int:
        """Number of price levels available"""
        return max(len(self.bids), len(self.asks))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bids': [{'price': l.price, 'quantity': l.quantity, 'orders': l.orders} for l in self.bids],
            'asks': [{'price': l.price, 'quantity': l.quantity, 'orders': l.orders} for l in self.asks],
            'mid_price': self.mid_price,
            'spread': self.spread,
            'sequence': self.sequence
        }


@dataclass
class OrderBookMetrics:
    """Calculated metrics from order book"""
    symbol: str
    timestamp: datetime

    # Basic metrics
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0          # Spread in basis points

    # Imbalance metrics
    obi: float = 0.0                  # Order Book Imbalance (-1 to 1)
    obi_weighted: float = 0.0         # Volume-weighted OBI
    bid_ask_ratio: float = 0.0        # Bid volume / Ask volume

    # Depth metrics
    bid_depth: int = 0                # Total bid quantity
    ask_depth: int = 0                # Total ask quantity
    bid_value: float = 0.0            # Total bid value
    ask_value: float = 0.0            # Total ask value

    # Pressure indicators
    buy_pressure: float = 0.0         # Normalized buy pressure
    sell_pressure: float = 0.0        # Normalized sell pressure

    # Micro-structure
    bid_slope: float = 0.0            # Price impact slope for bids
    ask_slope: float = 0.0            # Price impact slope for asks
    resilience: float = 0.0           # Book resilience estimate


@dataclass
class OrderBookConfig:
    """Configuration for order book processing"""
    # Depth settings
    max_depth: DepthLevel = DepthLevel.LEVEL_20
    aggregation_levels: int = 5       # Levels to use for aggregation

    # History settings
    history_size: int = 1000          # Snapshots to keep in memory
    metrics_history_size: int = 500   # Metrics to keep

    # OBI calculation
    obi_levels: int = 5               # Levels to use for OBI
    obi_weighted: bool = True         # Use volume-weighted OBI

    # Performance
    update_metrics_on_snapshot: bool = True
    emit_events: bool = True


class OrderBook:
    """
    Real-time order book for a single symbol.

    Maintains current state and history of order book snapshots
    with efficient memory usage via circular buffers.

    Example:
        book = OrderBook("RELIANCE")

        # Update with new snapshot
        book.update(bids=[(2500.0, 100), (2499.5, 200)],
                   asks=[(2500.5, 150), (2501.0, 100)])

        # Get metrics
        metrics = book.get_metrics()
        print(f"OBI: {metrics.obi:.2f}")

        # Get aggregated view
        agg = book.aggregate_levels(5)
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[OrderBookConfig] = None
    ):
        """
        Args:
            symbol: Trading symbol
            config: Order book configuration
        """
        self.symbol = symbol
        self._config = config or OrderBookConfig()
        self._lock = threading.RLock()

        # Current state
        self._current: Optional[OrderBookSnapshot] = None
        self._sequence = 0

        # History (circular buffers)
        self._snapshot_history: deque = deque(maxlen=self._config.history_size)
        self._metrics_history: deque = deque(maxlen=self._config.metrics_history_size)

        # Cached metrics
        self._cached_metrics: Optional[OrderBookMetrics] = None
        self._metrics_dirty = True

        # Callbacks
        self._update_callbacks: List[Callable[[OrderBookSnapshot], None]] = []

        logger.debug(f"OrderBook initialized for {symbol}")

    def update(
        self,
        bids: List[Tuple[float, int, int]],
        asks: List[Tuple[float, int, int]],
        timestamp: Optional[datetime] = None,
        exchange_timestamp: Optional[datetime] = None
    ) -> OrderBookSnapshot:
        """
        Update order book with new data.

        Args:
            bids: List of (price, quantity, orders) tuples, best first
            asks: List of (price, quantity, orders) tuples, best first
            timestamp: Local timestamp (default: now)
            exchange_timestamp: Exchange timestamp if available

        Returns:
            The created snapshot
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            self._sequence += 1

            # Create price levels
            bid_levels = [
                PriceLevel(price=p, quantity=q, orders=o if len(b) > 2 else 0, timestamp=timestamp)
                for b in bids
                for p, q, o in [b if len(b) >= 3 else (*b, 0)]
            ][:self._config.max_depth.value]

            ask_levels = [
                PriceLevel(price=p, quantity=q, orders=o if len(a) > 2 else 0, timestamp=timestamp)
                for a in asks
                for p, q, o in [a if len(a) >= 3 else (*a, 0)]
            ][:self._config.max_depth.value]

            # Create snapshot
            snapshot = OrderBookSnapshot(
                symbol=self.symbol,
                timestamp=timestamp,
                bids=bid_levels,
                asks=ask_levels,
                exchange_timestamp=exchange_timestamp,
                sequence=self._sequence
            )

            # Update current state
            self._current = snapshot
            self._snapshot_history.append(snapshot)
            self._metrics_dirty = True

            # Calculate and store metrics if configured
            if self._config.update_metrics_on_snapshot:
                metrics = self._calculate_metrics(snapshot)
                self._cached_metrics = metrics
                self._metrics_history.append(metrics)
                self._metrics_dirty = False

        # Notify callbacks
        for callback in self._update_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Update callback error: {e}")

        return snapshot

    def update_from_dict(self, data: Dict[str, Any]) -> OrderBookSnapshot:
        """
        Update from dictionary format (common for API responses).

        Expected format:
        {
            'bids': [[price, qty, orders], ...],
            'asks': [[price, qty, orders], ...],
            'timestamp': '2024-01-01T10:00:00'
        }
        """
        bids = [tuple(b) for b in data.get('bids', [])]
        asks = [tuple(a) for a in data.get('asks', [])]

        timestamp = None
        if 'timestamp' in data:
            if isinstance(data['timestamp'], str):
                timestamp = datetime.fromisoformat(data['timestamp'])
            elif isinstance(data['timestamp'], datetime):
                timestamp = data['timestamp']

        exchange_ts = None
        if 'exchange_timestamp' in data:
            if isinstance(data['exchange_timestamp'], str):
                exchange_ts = datetime.fromisoformat(data['exchange_timestamp'])

        return self.update(bids, asks, timestamp, exchange_ts)

    @property
    def current(self) -> Optional[OrderBookSnapshot]:
        """Current order book snapshot"""
        with self._lock:
            return self._current

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Best bid price level"""
        with self._lock:
            return self._current.best_bid if self._current else None

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Best ask price level"""
        with self._lock:
            return self._current.best_ask if self._current else None

    @property
    def mid_price(self) -> float:
        """Current mid price"""
        with self._lock:
            return self._current.mid_price if self._current else 0.0

    @property
    def spread(self) -> float:
        """Current spread"""
        with self._lock:
            return self._current.spread if self._current else 0.0

    # =========================================================================
    # Metrics Calculation
    # =========================================================================

    def get_metrics(self, recalculate: bool = False) -> Optional[OrderBookMetrics]:
        """
        Get current order book metrics.

        Args:
            recalculate: Force recalculation even if cached

        Returns:
            OrderBookMetrics or None if no data
        """
        with self._lock:
            if self._current is None:
                return None

            if recalculate or self._metrics_dirty or self._cached_metrics is None:
                self._cached_metrics = self._calculate_metrics(self._current)
                self._metrics_dirty = False

            return self._cached_metrics

    def _calculate_metrics(self, snapshot: OrderBookSnapshot) -> OrderBookMetrics:
        """Calculate all metrics from a snapshot"""
        metrics = OrderBookMetrics(
            symbol=self.symbol,
            timestamp=snapshot.timestamp
        )

        if not snapshot.bids or not snapshot.asks:
            return metrics

        # Basic metrics
        metrics.mid_price = snapshot.mid_price
        metrics.spread = snapshot.spread
        metrics.spread_bps = (snapshot.spread / metrics.mid_price * 10000) if metrics.mid_price > 0 else 0

        # Depth metrics
        metrics.bid_depth = snapshot.total_bid_quantity
        metrics.ask_depth = snapshot.total_ask_quantity
        metrics.bid_value = snapshot.total_bid_value
        metrics.ask_value = snapshot.total_ask_value

        # Bid/Ask ratio
        if metrics.ask_depth > 0:
            metrics.bid_ask_ratio = metrics.bid_depth / metrics.ask_depth
        else:
            metrics.bid_ask_ratio = float('inf') if metrics.bid_depth > 0 else 1.0

        # Order Book Imbalance (OBI)
        obi_levels = min(self._config.obi_levels, len(snapshot.bids), len(snapshot.asks))

        if obi_levels > 0:
            if self._config.obi_weighted:
                # Volume-weighted OBI
                bid_vol = sum(snapshot.bids[i].quantity for i in range(obi_levels))
                ask_vol = sum(snapshot.asks[i].quantity for i in range(obi_levels))
            else:
                # Simple count
                bid_vol = obi_levels
                ask_vol = obi_levels

            total = bid_vol + ask_vol
            if total > 0:
                metrics.obi = (bid_vol - ask_vol) / total
                metrics.obi_weighted = metrics.obi

        # Buy/Sell pressure (normalized)
        total_depth = metrics.bid_depth + metrics.ask_depth
        if total_depth > 0:
            metrics.buy_pressure = metrics.bid_depth / total_depth
            metrics.sell_pressure = metrics.ask_depth / total_depth

        # Price impact slopes (simplified)
        if len(snapshot.bids) >= 2:
            price_drop = snapshot.bids[0].price - snapshot.bids[-1].price
            qty_sum = sum(l.quantity for l in snapshot.bids)
            if price_drop > 0:
                metrics.bid_slope = qty_sum / price_drop

        if len(snapshot.asks) >= 2:
            price_rise = snapshot.asks[-1].price - snapshot.asks[0].price
            qty_sum = sum(l.quantity for l in snapshot.asks)
            if price_rise > 0:
                metrics.ask_slope = qty_sum / price_rise

        # Resilience estimate (simplified: average of slopes)
        if metrics.bid_slope > 0 and metrics.ask_slope > 0:
            metrics.resilience = (metrics.bid_slope + metrics.ask_slope) / 2

        return metrics

    def calculate_obi(self, levels: int = 5) -> float:
        """
        Calculate Order Book Imbalance.

        OBI = (BidVolume - AskVolume) / (BidVolume + AskVolume)

        Range: -1 (all asks) to +1 (all bids)
        Positive OBI suggests buying pressure, negative suggests selling.

        Args:
            levels: Number of levels to include

        Returns:
            OBI value between -1 and 1
        """
        with self._lock:
            if self._current is None:
                return 0.0

            bids = self._current.bids[:levels]
            asks = self._current.asks[:levels]

            bid_vol = sum(l.quantity for l in bids)
            ask_vol = sum(l.quantity for l in asks)

            total = bid_vol + ask_vol
            if total == 0:
                return 0.0

            return (bid_vol - ask_vol) / total

    def calculate_vwap_impact(self, side: Side, quantity: int) -> Tuple[float, float]:
        """
        Calculate VWAP and price impact for a hypothetical order.

        Args:
            side: BID (buy) or ASK (sell)
            quantity: Order quantity

        Returns:
            Tuple of (vwap, price_impact_percent)
        """
        with self._lock:
            if self._current is None:
                return 0.0, 0.0

            levels = self._current.asks if side == Side.BID else self._current.bids
            if not levels:
                return 0.0, 0.0

            remaining = quantity
            total_cost = 0.0
            filled = 0

            for level in levels:
                fill_qty = min(remaining, level.quantity)
                total_cost += fill_qty * level.price
                filled += fill_qty
                remaining -= fill_qty

                if remaining <= 0:
                    break

            if filled == 0:
                return 0.0, 0.0

            vwap = total_cost / filled
            best_price = levels[0].price
            impact = ((vwap - best_price) / best_price) * 100

            # For sells, impact is negative (price drops)
            if side == Side.ASK:
                impact = -impact

            return vwap, abs(impact)

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate_levels(self, num_levels: int = 5) -> Dict[str, List[Dict]]:
        """
        Get aggregated view of top N levels.

        Args:
            num_levels: Number of levels to return

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        with self._lock:
            if self._current is None:
                return {'bids': [], 'asks': []}

            bids = [
                {'price': l.price, 'quantity': l.quantity, 'orders': l.orders, 'value': l.value}
                for l in self._current.bids[:num_levels]
            ]

            asks = [
                {'price': l.price, 'quantity': l.quantity, 'orders': l.orders, 'value': l.value}
                for l in self._current.asks[:num_levels]
            ]

            return {'bids': bids, 'asks': asks}

    def get_depth_at_price(self, price: float, tolerance: float = 0.01) -> Tuple[int, int]:
        """
        Get bid and ask depth at a specific price level.

        Args:
            price: Target price
            tolerance: Price tolerance (fraction)

        Returns:
            Tuple of (bid_qty, ask_qty) at or near the price
        """
        with self._lock:
            if self._current is None:
                return 0, 0

            bid_qty = 0
            ask_qty = 0
            price_range = price * tolerance

            for level in self._current.bids:
                if abs(level.price - price) <= price_range:
                    bid_qty += level.quantity

            for level in self._current.asks:
                if abs(level.price - price) <= price_range:
                    ask_qty += level.quantity

            return bid_qty, ask_qty

    def get_cumulative_depth(self, side: Side, price_pct: float = 1.0) -> int:
        """
        Get cumulative depth within price percentage from best.

        Args:
            side: BID or ASK
            price_pct: Percentage from best price to include

        Returns:
            Cumulative quantity
        """
        with self._lock:
            if self._current is None:
                return 0

            if side == Side.BID:
                levels = self._current.bids
                if not levels:
                    return 0
                best = levels[0].price
                threshold = best * (1 - price_pct / 100)
                return sum(l.quantity for l in levels if l.price >= threshold)
            else:
                levels = self._current.asks
                if not levels:
                    return 0
                best = levels[0].price
                threshold = best * (1 + price_pct / 100)
                return sum(l.quantity for l in levels if l.price <= threshold)

    # =========================================================================
    # History Access
    # =========================================================================

    def get_history(self, count: int = 100) -> List[OrderBookSnapshot]:
        """Get recent snapshot history"""
        with self._lock:
            return list(self._snapshot_history)[-count:]

    def get_metrics_history(self, count: int = 100) -> List[OrderBookMetrics]:
        """Get recent metrics history"""
        with self._lock:
            return list(self._metrics_history)[-count:]

    def get_obi_series(self, count: int = 100) -> List[Tuple[datetime, float]]:
        """Get OBI time series"""
        with self._lock:
            return [
                (m.timestamp, m.obi)
                for m in list(self._metrics_history)[-count:]
            ]

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_update(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Register callback for order book updates"""
        self._update_callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> bool:
        """Remove update callback"""
        try:
            self._update_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear_history(self) -> None:
        """Clear snapshot and metrics history"""
        with self._lock:
            self._snapshot_history.clear()
            self._metrics_history.clear()

    def summary(self) -> str:
        """Get formatted summary"""
        with self._lock:
            if self._current is None:
                return f"OrderBook({self.symbol}): No data"

            metrics = self.get_metrics()
            lines = [
                f"OrderBook({self.symbol}):",
                f"  Mid: {self._current.mid_price:.2f}",
                f"  Spread: {self._current.spread:.2f} ({self._current.spread_percent:.3f}%)",
                f"  Depth: {self._current.depth} levels",
                f"  Bid Volume: {self._current.total_bid_quantity:,}",
                f"  Ask Volume: {self._current.total_ask_quantity:,}",
            ]

            if metrics:
                lines.extend([
                    f"  OBI: {metrics.obi:+.3f}",
                    f"  Buy Pressure: {metrics.buy_pressure:.1%}",
                ])

            return "\n".join(lines)


class OrderBookManager:
    """
    Manager for multiple order books.

    Provides centralized access to order books for all symbols
    with memory-efficient pooling.

    Example:
        manager = OrderBookManager()

        # Update order book
        manager.update("RELIANCE", bids, asks)

        # Get OBI for a symbol
        obi = manager.get_obi("RELIANCE")

        # Get all symbols with data
        symbols = manager.symbols
    """

    def __init__(self, config: Optional[OrderBookConfig] = None):
        """
        Args:
            config: Default configuration for order books
        """
        self._config = config or OrderBookConfig()
        self._books: Dict[str, OrderBook] = {}
        self._lock = threading.RLock()

        # Global callbacks
        self._update_callbacks: List[Callable[[str, OrderBookSnapshot], None]] = []

        logger.info("OrderBookManager initialized")

    def get_book(self, symbol: str) -> OrderBook:
        """
        Get or create order book for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            OrderBook instance
        """
        with self._lock:
            if symbol not in self._books:
                self._books[symbol] = OrderBook(symbol, self._config)
                # Wire up callback
                self._books[symbol].on_update(
                    lambda snap, s=symbol: self._on_book_update(s, snap)
                )
            return self._books[symbol]

    def update(
        self,
        symbol: str,
        bids: List[Tuple[float, int, int]],
        asks: List[Tuple[float, int, int]],
        timestamp: Optional[datetime] = None
    ) -> OrderBookSnapshot:
        """Update order book for a symbol"""
        book = self.get_book(symbol)
        return book.update(bids, asks, timestamp)

    def get_metrics(self, symbol: str) -> Optional[OrderBookMetrics]:
        """Get metrics for a symbol"""
        with self._lock:
            if symbol in self._books:
                return self._books[symbol].get_metrics()
        return None

    def get_obi(self, symbol: str, levels: int = 5) -> float:
        """Get OBI for a symbol"""
        with self._lock:
            if symbol in self._books:
                return self._books[symbol].calculate_obi(levels)
        return 0.0

    def get_all_obi(self, levels: int = 5) -> Dict[str, float]:
        """Get OBI for all symbols"""
        with self._lock:
            return {
                symbol: book.calculate_obi(levels)
                for symbol, book in self._books.items()
            }

    @property
    def symbols(self) -> List[str]:
        """List of symbols with order books"""
        with self._lock:
            return list(self._books.keys())

    def on_update(self, callback: Callable[[str, OrderBookSnapshot], None]) -> None:
        """Register callback for any order book update"""
        self._update_callbacks.append(callback)

    def _on_book_update(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """Internal handler for book updates"""
        for callback in self._update_callbacks:
            try:
                callback(symbol, snapshot)
            except Exception as e:
                logger.error(f"Manager callback error: {e}")

    def remove_book(self, symbol: str) -> bool:
        """Remove order book for a symbol"""
        with self._lock:
            if symbol in self._books:
                del self._books[symbol]
                return True
        return False

    def clear_all(self) -> None:
        """Clear all order books"""
        with self._lock:
            self._books.clear()

    def summary(self) -> str:
        """Get summary of all order books"""
        with self._lock:
            lines = [f"OrderBookManager: {len(self._books)} symbols"]
            for symbol, book in self._books.items():
                metrics = book.get_metrics()
                if metrics:
                    lines.append(
                        f"  {symbol}: mid={book.mid_price:.2f}, "
                        f"obi={metrics.obi:+.3f}, spread={book.spread:.2f}"
                    )
            return "\n".join(lines)


# =============================================================================
# Global Instance
# =============================================================================

_global_manager: Optional[OrderBookManager] = None
_global_manager_lock = threading.Lock()


def get_orderbook_manager() -> OrderBookManager:
    """Get the global order book manager instance"""
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = OrderBookManager()
    return _global_manager


def set_orderbook_manager(manager: OrderBookManager) -> None:
    """Set the global order book manager instance"""
    global _global_manager
    with _global_manager_lock:
        _global_manager = manager
