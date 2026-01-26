"""
Market Microstructure Indicators.

Advanced indicators derived from Level 2 order book data and trade flow.
These indicators reveal supply/demand dynamics not visible in price alone.

Microstructure indicators help identify:
- Order flow imbalances predicting short-term price moves
- Spread dynamics indicating liquidity conditions
- Queue position for execution optimization
- Trade classification (buyer vs seller initiated)
"""

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Classified trade direction"""
    BUY = "buy"           # Buyer initiated (hit the ask)
    SELL = "sell"         # Seller initiated (hit the bid)
    NEUTRAL = "neutral"   # Cannot determine


class SpreadRegime(Enum):
    """Market spread regime classification"""
    TIGHT = "tight"       # Below average spread
    NORMAL = "normal"     # Average spread
    WIDE = "wide"         # Above average spread
    STRESSED = "stressed" # Abnormally wide spread


@dataclass
class Trade:
    """
    A single trade execution.

    Attributes:
        timestamp: Trade timestamp
        symbol: Trading symbol
        price: Execution price
        quantity: Trade size
        direction: Classified direction (buy/sell/neutral)
        is_block: Whether this is a block trade
    """
    timestamp: datetime
    symbol: str
    price: float
    quantity: int
    direction: TradeDirection = TradeDirection.NEUTRAL
    is_block: bool = False
    exchange_order_id: str = ""

    @property
    def value(self) -> float:
        """Trade value"""
        return self.price * self.quantity


@dataclass
class SpreadSnapshot:
    """Bid-ask spread snapshot"""
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    spread: float
    spread_bps: float        # Spread in basis points
    regime: SpreadRegime = SpreadRegime.NORMAL


@dataclass
class MicrostructureMetrics:
    """
    Comprehensive microstructure metrics.

    All metrics calculated over configurable time windows.
    """
    symbol: str
    timestamp: datetime

    # Order Book Imbalance
    obi: float = 0.0                  # Current OBI (-1 to 1)
    obi_ema: float = 0.0              # Smoothed OBI
    obi_momentum: float = 0.0         # OBI change rate

    # Spread Dynamics
    spread_current: float = 0.0        # Current spread
    spread_avg: float = 0.0            # Average spread
    spread_volatility: float = 0.0     # Spread volatility
    spread_regime: SpreadRegime = SpreadRegime.NORMAL
    spread_zscore: float = 0.0         # Spread z-score

    # Trade Flow Imbalance
    tfi: float = 0.0                   # Trade Flow Imbalance (-1 to 1)
    tfi_volume: float = 0.0            # Volume-weighted TFI
    buy_volume: int = 0                # Buyer-initiated volume
    sell_volume: int = 0               # Seller-initiated volume
    trade_count: int = 0               # Trade count in window

    # Volume-Weighted Metrics
    vwap: float = 0.0                  # VWAP in window
    vwbam: float = 0.0                 # Volume-Weighted Bid-Ask Midpoint
    microprice: float = 0.0            # Size-weighted microprice

    # Queue Position
    queue_position_bid: int = 0        # Estimated queue position (bid)
    queue_position_ask: int = 0        # Estimated queue position (ask)
    time_to_fill_bid: float = 0.0      # Estimated time to fill (seconds)
    time_to_fill_ask: float = 0.0      # Estimated time to fill (seconds)

    # Liquidity Metrics
    market_depth: float = 0.0          # Total visible liquidity
    bid_depth: int = 0                 # Bid side depth
    ask_depth: int = 0                 # Ask side depth
    resilience: float = 0.0            # Book resilience


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure analysis"""
    # Window sizes
    obi_window: int = 100              # Ticks for OBI calculation
    spread_window: int = 500           # Ticks for spread analysis
    trade_flow_window: int = 100       # Trades for TFI
    vwap_window: int = 200             # Trades for VWAP

    # EMA parameters
    obi_ema_span: int = 20             # OBI EMA span
    spread_ema_span: int = 50          # Spread EMA span

    # Trade classification
    tick_rule_enabled: bool = True     # Use tick rule for classification
    quote_rule_enabled: bool = True    # Use quote rule for classification
    block_trade_threshold: int = 10000 # Min size for block trade

    # Spread regime thresholds
    tight_spread_percentile: float = 25.0
    wide_spread_percentile: float = 75.0
    stressed_spread_multiplier: float = 3.0

    # Queue estimation
    avg_order_size: int = 100          # Average order size assumption
    fill_rate_per_second: float = 50.0 # Estimated fills per second


class TradeFlowAnalyzer:
    """
    Analyzes trade flow to determine buyer vs seller initiated trades.

    Uses multiple classification methods:
    1. Tick Rule: Compare to previous trade price
    2. Quote Rule: Compare to bid-ask midpoint
    3. Lee-Ready: Combines tick and quote rules
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self._config = config or MicrostructureConfig()
        self._lock = threading.Lock()

        # Trade history
        self._trades: deque = deque(maxlen=self._config.trade_flow_window)
        self._last_price: Dict[str, float] = {}
        self._last_mid: Dict[str, float] = {}

        # Aggregates
        self._buy_volume: Dict[str, int] = {}
        self._sell_volume: Dict[str, int] = {}
        self._trade_count: Dict[str, int] = {}

    def classify_trade(
        self,
        symbol: str,
        price: float,
        quantity: int,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> Trade:
        """
        Classify a trade as buyer or seller initiated.

        Args:
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            bid: Current bid price (for quote rule)
            ask: Current ask price (for quote rule)
            timestamp: Trade timestamp

        Returns:
            Trade object with classified direction
        """
        timestamp = timestamp or datetime.now()
        direction = TradeDirection.NEUTRAL

        with self._lock:
            # Quote Rule (Lee-Ready)
            if self._config.quote_rule_enabled and bid is not None and ask is not None:
                mid = (bid + ask) / 2
                if price > mid:
                    direction = TradeDirection.BUY
                elif price < mid:
                    direction = TradeDirection.SELL
                else:
                    # At midpoint, use tick rule
                    direction = self._apply_tick_rule(symbol, price)
                self._last_mid[symbol] = mid
            # Tick Rule only
            elif self._config.tick_rule_enabled:
                direction = self._apply_tick_rule(symbol, price)

            # Update last price
            self._last_price[symbol] = price

            # Determine if block trade
            is_block = quantity >= self._config.block_trade_threshold

            # Create trade object
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                quantity=quantity,
                direction=direction,
                is_block=is_block
            )

            # Store and update aggregates
            self._trades.append(trade)
            self._update_aggregates(trade)

            return trade

    def _apply_tick_rule(self, symbol: str, price: float) -> TradeDirection:
        """Apply tick rule for trade classification"""
        last_price = self._last_price.get(symbol)
        if last_price is None:
            return TradeDirection.NEUTRAL

        if price > last_price:
            return TradeDirection.BUY
        elif price < last_price:
            return TradeDirection.SELL
        else:
            return TradeDirection.NEUTRAL

    def _update_aggregates(self, trade: Trade) -> None:
        """Update volume aggregates"""
        symbol = trade.symbol

        if symbol not in self._buy_volume:
            self._buy_volume[symbol] = 0
            self._sell_volume[symbol] = 0
            self._trade_count[symbol] = 0

        self._trade_count[symbol] += 1

        if trade.direction == TradeDirection.BUY:
            self._buy_volume[symbol] += trade.quantity
        elif trade.direction == TradeDirection.SELL:
            self._sell_volume[symbol] += trade.quantity

    def get_trade_flow_imbalance(self, symbol: str) -> float:
        """
        Calculate Trade Flow Imbalance (TFI).

        TFI = (BuyVolume - SellVolume) / (BuyVolume + SellVolume)

        Range: -1 (all sells) to +1 (all buys)

        Args:
            symbol: Trading symbol

        Returns:
            TFI value
        """
        with self._lock:
            buy_vol = self._buy_volume.get(symbol, 0)
            sell_vol = self._sell_volume.get(symbol, 0)

            total = buy_vol + sell_vol
            if total == 0:
                return 0.0

            return (buy_vol - sell_vol) / total

    def get_volume_breakdown(self, symbol: str) -> Dict[str, int]:
        """Get buy/sell volume breakdown"""
        with self._lock:
            return {
                'buy_volume': self._buy_volume.get(symbol, 0),
                'sell_volume': self._sell_volume.get(symbol, 0),
                'trade_count': self._trade_count.get(symbol, 0)
            }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset aggregates"""
        with self._lock:
            if symbol:
                self._buy_volume[symbol] = 0
                self._sell_volume[symbol] = 0
                self._trade_count[symbol] = 0
            else:
                self._buy_volume.clear()
                self._sell_volume.clear()
                self._trade_count.clear()
                self._trades.clear()


class SpreadAnalyzer:
    """
    Analyzes bid-ask spread dynamics.

    Tracks spread over time to identify:
    - Current spread regime (tight/normal/wide/stressed)
    - Spread volatility and trends
    - Anomalous spread conditions
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self._config = config or MicrostructureConfig()
        self._lock = threading.Lock()

        # Spread history
        self._spreads: Dict[str, deque] = {}

        # Statistics
        self._spread_sum: Dict[str, float] = {}
        self._spread_sq_sum: Dict[str, float] = {}
        self._spread_count: Dict[str, int] = {}

        # EMA
        self._spread_ema: Dict[str, float] = {}
        self._ema_alpha = 2.0 / (self._config.spread_ema_span + 1)

    def update(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None
    ) -> SpreadSnapshot:
        """
        Update spread analysis with new quote.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            timestamp: Quote timestamp

        Returns:
            SpreadSnapshot with current analysis
        """
        timestamp = timestamp or datetime.now()

        if bid <= 0 or ask <= 0 or ask < bid:
            logger.warning(f"Invalid quote for {symbol}: bid={bid}, ask={ask}")
            return SpreadSnapshot(
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2 if bid > 0 and ask > 0 else 0,
                spread=0,
                spread_bps=0
            )

        mid = (bid + ask) / 2
        spread = ask - bid
        spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        with self._lock:
            # Initialize if needed
            if symbol not in self._spreads:
                self._spreads[symbol] = deque(maxlen=self._config.spread_window)
                self._spread_sum[symbol] = 0.0
                self._spread_sq_sum[symbol] = 0.0
                self._spread_count[symbol] = 0
                self._spread_ema[symbol] = spread_bps

            # Update history
            spreads = self._spreads[symbol]

            # Remove oldest if at capacity
            if len(spreads) >= self._config.spread_window:
                old = spreads[0]
                self._spread_sum[symbol] -= old.spread_bps
                self._spread_sq_sum[symbol] -= old.spread_bps ** 2

            # Calculate statistics
            self._spread_sum[symbol] += spread_bps
            self._spread_sq_sum[symbol] += spread_bps ** 2
            self._spread_count[symbol] = min(
                self._spread_count[symbol] + 1,
                self._config.spread_window
            )

            n = self._spread_count[symbol]
            avg = self._spread_sum[symbol] / n
            variance = (self._spread_sq_sum[symbol] / n) - (avg ** 2)
            std = math.sqrt(max(0, variance))

            # Update EMA
            self._spread_ema[symbol] = (
                self._ema_alpha * spread_bps +
                (1 - self._ema_alpha) * self._spread_ema[symbol]
            )

            # Calculate z-score
            zscore = (spread_bps - avg) / std if std > 0 else 0

            # Determine regime
            regime = self._classify_regime(spread_bps, avg, std)

            snapshot = SpreadSnapshot(
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                spread_bps=spread_bps,
                regime=regime
            )

            spreads.append(snapshot)

            return snapshot

    def _classify_regime(
        self,
        spread_bps: float,
        avg: float,
        std: float
    ) -> SpreadRegime:
        """Classify current spread regime"""
        if std == 0:
            return SpreadRegime.NORMAL

        zscore = (spread_bps - avg) / std

        # Stressed: extremely wide
        if zscore > self._config.stressed_spread_multiplier:
            return SpreadRegime.STRESSED

        # Wide: above 75th percentile (roughly z > 0.67)
        if spread_bps > avg + 0.67 * std:
            return SpreadRegime.WIDE

        # Tight: below 25th percentile (roughly z < -0.67)
        if spread_bps < avg - 0.67 * std:
            return SpreadRegime.TIGHT

        return SpreadRegime.NORMAL

    def get_spread_stats(self, symbol: str) -> Dict[str, float]:
        """Get spread statistics"""
        with self._lock:
            if symbol not in self._spread_count or self._spread_count[symbol] == 0:
                return {'avg': 0, 'std': 0, 'ema': 0, 'count': 0}

            n = self._spread_count[symbol]
            avg = self._spread_sum[symbol] / n
            variance = (self._spread_sq_sum[symbol] / n) - (avg ** 2)
            std = math.sqrt(max(0, variance))

            return {
                'avg': avg,
                'std': std,
                'ema': self._spread_ema.get(symbol, 0),
                'count': n
            }

    def get_current_regime(self, symbol: str) -> SpreadRegime:
        """Get current spread regime"""
        with self._lock:
            spreads = self._spreads.get(symbol)
            if spreads and len(spreads) > 0:
                return spreads[-1].regime
            return SpreadRegime.NORMAL


class QueuePositionEstimator:
    """
    Estimates queue position for limit orders.

    Helps with execution optimization by estimating:
    - Current position in queue at a price level
    - Expected time to fill
    - Queue dynamics (additions, cancellations)
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self._config = config or MicrostructureConfig()
        self._lock = threading.Lock()

        # Queue state tracking
        self._queue_sizes: Dict[str, Dict[str, int]] = {}  # symbol -> {price: size}
        self._fill_rates: Dict[str, float] = {}            # symbol -> fills/second
        self._last_update: Dict[str, datetime] = {}

    def update_queue(
        self,
        symbol: str,
        price: float,
        size: int,
        side: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update queue size at a price level.

        Args:
            symbol: Trading symbol
            price: Price level
            size: Current queue size
            side: 'bid' or 'ask'
            timestamp: Update timestamp
        """
        timestamp = timestamp or datetime.now()
        key = f"{side}_{price}"

        with self._lock:
            if symbol not in self._queue_sizes:
                self._queue_sizes[symbol] = {}

            self._queue_sizes[symbol][key] = size
            self._last_update[symbol] = timestamp

    def update_from_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, int]],
        asks: List[Tuple[float, int]]
    ) -> None:
        """
        Update queue from order book snapshot.

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks
        """
        with self._lock:
            if symbol not in self._queue_sizes:
                self._queue_sizes[symbol] = {}

            # Clear old data
            self._queue_sizes[symbol] = {}

            # Add bids
            for price, qty in bids:
                self._queue_sizes[symbol][f"bid_{price}"] = qty

            # Add asks
            for price, qty in asks:
                self._queue_sizes[symbol][f"ask_{price}"] = qty

            self._last_update[symbol] = datetime.now()

    def estimate_position(
        self,
        symbol: str,
        price: float,
        side: str,
        order_size: int
    ) -> Tuple[int, float]:
        """
        Estimate queue position and time to fill.

        Assumes FIFO queue and uses average order size assumption.

        Args:
            symbol: Trading symbol
            price: Limit price
            side: 'bid' or 'ask'
            order_size: Order quantity

        Returns:
            Tuple of (queue_position, estimated_time_to_fill_seconds)
        """
        with self._lock:
            key = f"{side}_{price}"
            queue_size = self._queue_sizes.get(symbol, {}).get(key, 0)

            # Estimate position (assuming we're at back of queue)
            # Use average order size to estimate number of orders
            avg_size = self._config.avg_order_size
            estimated_orders = queue_size / avg_size if avg_size > 0 else 0

            # Position is total queue in front of us
            position = queue_size

            # Estimate time to fill
            fill_rate = self._fill_rates.get(
                symbol,
                self._config.fill_rate_per_second
            )

            if fill_rate > 0:
                time_to_fill = (queue_size + order_size / 2) / fill_rate
            else:
                time_to_fill = float('inf')

            return position, time_to_fill

    def update_fill_rate(
        self,
        symbol: str,
        fills_observed: int,
        time_window_seconds: float
    ) -> None:
        """Update observed fill rate for better estimation"""
        if time_window_seconds > 0:
            with self._lock:
                self._fill_rates[symbol] = fills_observed / time_window_seconds


class MicrostructureAnalyzer:
    """
    Comprehensive market microstructure analyzer.

    Combines order book analysis, trade flow, spread dynamics,
    and queue estimation into unified metrics.

    Example:
        analyzer = MicrostructureAnalyzer()

        # Update with order book
        analyzer.update_orderbook(
            "RELIANCE",
            bids=[(2500, 1000), (2499.5, 2000)],
            asks=[(2500.5, 500), (2501, 1500)]
        )

        # Process trade
        analyzer.process_trade("RELIANCE", 2500.5, 100)

        # Get metrics
        metrics = analyzer.get_metrics("RELIANCE")
        print(f"OBI: {metrics.obi:.3f}, TFI: {metrics.tfi:.3f}")
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        """
        Args:
            config: Analyzer configuration
        """
        self._config = config or MicrostructureConfig()
        self._lock = threading.RLock()

        # Component analyzers
        self._trade_flow = TradeFlowAnalyzer(self._config)
        self._spread = SpreadAnalyzer(self._config)
        self._queue = QueuePositionEstimator(self._config)

        # Order book state
        self._orderbooks: Dict[str, Dict] = {}

        # OBI tracking
        self._obi_history: Dict[str, deque] = {}
        self._obi_ema: Dict[str, float] = {}
        self._obi_ema_alpha = 2.0 / (self._config.obi_ema_span + 1)

        # VWAP tracking
        self._vwap_sum: Dict[str, float] = {}
        self._vwap_volume: Dict[str, int] = {}

        # Metrics cache
        self._cached_metrics: Dict[str, MicrostructureMetrics] = {}

        # Callbacks
        self._metric_callbacks: List[Callable[[MicrostructureMetrics], None]] = []

        logger.info("MicrostructureAnalyzer initialized")

    def update_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, int]],
        asks: List[Tuple[float, int]],
        timestamp: Optional[datetime] = None
    ) -> MicrostructureMetrics:
        """
        Update with new order book data.

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) for bids, best first
            asks: List of (price, quantity) for asks, best first
            timestamp: Update timestamp

        Returns:
            Updated metrics
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            # Store order book state
            self._orderbooks[symbol] = {
                'bids': bids,
                'asks': asks,
                'timestamp': timestamp
            }

            # Update spread analyzer
            if bids and asks:
                self._spread.update(symbol, bids[0][0], asks[0][0], timestamp)

            # Update queue estimator
            self._queue.update_from_orderbook(symbol, bids, asks)

            # Calculate OBI
            obi = self._calculate_obi(bids, asks)
            self._update_obi_ema(symbol, obi)

            # Calculate microprice
            microprice = self._calculate_microprice(bids, asks)

            # Calculate VWBAM
            vwbam = self._calculate_vwbam(bids, asks)

            # Build metrics
            metrics = self._build_metrics(symbol, timestamp)

            # Cache metrics
            self._cached_metrics[symbol] = metrics

            # Notify callbacks
            for callback in self._metric_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Metric callback error: {e}")

            return metrics

    def process_trade(
        self,
        symbol: str,
        price: float,
        quantity: int,
        timestamp: Optional[datetime] = None
    ) -> Trade:
        """
        Process a trade execution.

        Args:
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            timestamp: Trade timestamp

        Returns:
            Classified Trade object
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            # Get current bid/ask for classification
            book = self._orderbooks.get(symbol, {})
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            bid = bids[0][0] if bids else None
            ask = asks[0][0] if asks else None

            # Classify and track trade
            trade = self._trade_flow.classify_trade(
                symbol, price, quantity, bid, ask, timestamp
            )

            # Update VWAP
            if symbol not in self._vwap_sum:
                self._vwap_sum[symbol] = 0.0
                self._vwap_volume[symbol] = 0

            self._vwap_sum[symbol] += price * quantity
            self._vwap_volume[symbol] += quantity

            return trade

    def _calculate_obi(
        self,
        bids: List[Tuple[float, int]],
        asks: List[Tuple[float, int]]
    ) -> float:
        """Calculate Order Book Imbalance"""
        levels = min(self._config.obi_window, len(bids), len(asks))
        if levels == 0:
            return 0.0

        bid_vol = sum(b[1] for b in bids[:levels])
        ask_vol = sum(a[1] for a in asks[:levels])

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def _update_obi_ema(self, symbol: str, obi: float) -> None:
        """Update OBI exponential moving average"""
        if symbol not in self._obi_ema:
            self._obi_ema[symbol] = obi
            self._obi_history[symbol] = deque(maxlen=self._config.obi_window)

        self._obi_ema[symbol] = (
            self._obi_ema_alpha * obi +
            (1 - self._obi_ema_alpha) * self._obi_ema[symbol]
        )
        self._obi_history[symbol].append(obi)

    def _calculate_microprice(
        self,
        bids: List[Tuple[float, int]],
        asks: List[Tuple[float, int]]
    ) -> float:
        """
        Calculate microprice (size-weighted mid price).

        Microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

        This adjusts the mid price toward the side with more size,
        providing a better estimate of fair value.
        """
        if not bids or not asks:
            return 0.0

        bid_price, bid_size = bids[0]
        ask_price, ask_size = asks[0]

        total_size = bid_size + ask_size
        if total_size == 0:
            return (bid_price + ask_price) / 2

        return (bid_price * ask_size + ask_price * bid_size) / total_size

    def _calculate_vwbam(
        self,
        bids: List[Tuple[float, int]],
        asks: List[Tuple[float, int]]
    ) -> float:
        """
        Calculate Volume-Weighted Bid-Ask Midpoint.

        Uses volume across multiple levels to weight the midpoint.
        """
        if not bids or not asks:
            return 0.0

        # Use top 5 levels
        levels = 5
        bid_levels = bids[:levels]
        ask_levels = asks[:levels]

        bid_vwap = sum(p * q for p, q in bid_levels)
        bid_vol = sum(q for _, q in bid_levels)

        ask_vwap = sum(p * q for p, q in ask_levels)
        ask_vol = sum(q for _, q in ask_levels)

        if bid_vol == 0 or ask_vol == 0:
            return 0.0

        bid_avg = bid_vwap / bid_vol
        ask_avg = ask_vwap / ask_vol

        return (bid_avg + ask_avg) / 2

    def _build_metrics(
        self,
        symbol: str,
        timestamp: datetime
    ) -> MicrostructureMetrics:
        """Build comprehensive metrics"""
        metrics = MicrostructureMetrics(symbol=symbol, timestamp=timestamp)

        # Order book metrics
        book = self._orderbooks.get(symbol, {})
        bids = book.get('bids', [])
        asks = book.get('asks', [])

        if bids and asks:
            metrics.obi = self._calculate_obi(bids, asks)
            metrics.obi_ema = self._obi_ema.get(symbol, 0)
            metrics.microprice = self._calculate_microprice(bids, asks)
            metrics.vwbam = self._calculate_vwbam(bids, asks)

            # OBI momentum
            history = self._obi_history.get(symbol, deque())
            if len(history) >= 2:
                metrics.obi_momentum = history[-1] - history[-2]

            # Depth
            metrics.bid_depth = sum(q for _, q in bids)
            metrics.ask_depth = sum(q for _, q in asks)
            metrics.market_depth = metrics.bid_depth + metrics.ask_depth

        # Spread metrics
        spread_stats = self._spread.get_spread_stats(symbol)
        metrics.spread_avg = spread_stats['avg']
        metrics.spread_volatility = spread_stats['std']
        metrics.spread_regime = self._spread.get_current_regime(symbol)

        spreads = self._spread._spreads.get(symbol)
        if spreads and len(spreads) > 0:
            metrics.spread_current = spreads[-1].spread_bps
            if spread_stats['std'] > 0:
                metrics.spread_zscore = (
                    (metrics.spread_current - spread_stats['avg']) /
                    spread_stats['std']
                )

        # Trade flow metrics
        metrics.tfi = self._trade_flow.get_trade_flow_imbalance(symbol)
        volume_breakdown = self._trade_flow.get_volume_breakdown(symbol)
        metrics.buy_volume = volume_breakdown['buy_volume']
        metrics.sell_volume = volume_breakdown['sell_volume']
        metrics.trade_count = volume_breakdown['trade_count']

        total_vol = metrics.buy_volume + metrics.sell_volume
        if total_vol > 0:
            metrics.tfi_volume = (metrics.buy_volume - metrics.sell_volume) / total_vol

        # VWAP
        if self._vwap_volume.get(symbol, 0) > 0:
            metrics.vwap = self._vwap_sum[symbol] / self._vwap_volume[symbol]

        # Queue position (at best bid/ask)
        if bids:
            pos, time_to_fill = self._queue.estimate_position(
                symbol, bids[0][0], 'bid', 100
            )
            metrics.queue_position_bid = pos
            metrics.time_to_fill_bid = time_to_fill

        if asks:
            pos, time_to_fill = self._queue.estimate_position(
                symbol, asks[0][0], 'ask', 100
            )
            metrics.queue_position_ask = pos
            metrics.time_to_fill_ask = time_to_fill

        return metrics

    def get_metrics(self, symbol: str) -> Optional[MicrostructureMetrics]:
        """Get current metrics for a symbol"""
        with self._lock:
            return self._cached_metrics.get(symbol)

    def get_obi(self, symbol: str) -> float:
        """Get current OBI for a symbol"""
        with self._lock:
            metrics = self._cached_metrics.get(symbol)
            return metrics.obi if metrics else 0.0

    def get_tfi(self, symbol: str) -> float:
        """Get current Trade Flow Imbalance"""
        return self._trade_flow.get_trade_flow_imbalance(symbol)

    def get_spread_regime(self, symbol: str) -> SpreadRegime:
        """Get current spread regime"""
        return self._spread.get_current_regime(symbol)

    def on_metrics(self, callback: Callable[[MicrostructureMetrics], None]) -> None:
        """Register callback for metric updates"""
        self._metric_callbacks.append(callback)

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset analyzer state"""
        with self._lock:
            if symbol:
                self._orderbooks.pop(symbol, None)
                self._obi_history.pop(symbol, None)
                self._obi_ema.pop(symbol, None)
                self._vwap_sum.pop(symbol, None)
                self._vwap_volume.pop(symbol, None)
                self._cached_metrics.pop(symbol, None)
            else:
                self._orderbooks.clear()
                self._obi_history.clear()
                self._obi_ema.clear()
                self._vwap_sum.clear()
                self._vwap_volume.clear()
                self._cached_metrics.clear()

            self._trade_flow.reset(symbol)

    def summary(self, symbol: str) -> str:
        """Get formatted summary for a symbol"""
        metrics = self.get_metrics(symbol)
        if not metrics:
            return f"MicrostructureAnalyzer({symbol}): No data"

        lines = [
            f"Microstructure({symbol}):",
            f"  OBI: {metrics.obi:+.3f} (EMA: {metrics.obi_ema:+.3f})",
            f"  TFI: {metrics.tfi:+.3f} (Vol: {metrics.tfi_volume:+.3f})",
            f"  Spread: {metrics.spread_current:.1f} bps ({metrics.spread_regime.value})",
            f"  Microprice: {metrics.microprice:.2f}",
            f"  Depth: {metrics.bid_depth:,} bid / {metrics.ask_depth:,} ask",
            f"  Trades: {metrics.trade_count} (B:{metrics.buy_volume:,} S:{metrics.sell_volume:,})"
        ]

        return "\n".join(lines)


# =============================================================================
# Global Instance
# =============================================================================

_global_analyzer: Optional[MicrostructureAnalyzer] = None
_global_analyzer_lock = threading.Lock()


def get_microstructure_analyzer() -> MicrostructureAnalyzer:
    """Get the global microstructure analyzer instance"""
    global _global_analyzer
    if _global_analyzer is None:
        with _global_analyzer_lock:
            if _global_analyzer is None:
                _global_analyzer = MicrostructureAnalyzer()
    return _global_analyzer


def set_microstructure_analyzer(analyzer: MicrostructureAnalyzer) -> None:
    """Set the global microstructure analyzer instance"""
    global _global_analyzer
    with _global_analyzer_lock:
        _global_analyzer = analyzer
