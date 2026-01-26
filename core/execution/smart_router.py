# -*- coding: utf-8 -*-
"""
Smart Order Router - Multi-Exchange Order Routing
==================================================
Intelligent order routing across NSE and BSE for optimal execution.

Features:
- Real-time price comparison across exchanges
- Best execution routing
- Order splitting for large orders
- Transaction cost optimization
- Latency-aware routing

Example:
    >>> from core.execution import SmartRouter, Exchange
    >>>
    >>> router = SmartRouter(broker)
    >>>
    >>> # Route to best exchange automatically
    >>> result = router.route_order(
    ...     symbol="RELIANCE",
    ...     quantity=1000,
    ...     side="BUY",
    ...     order_type="LIMIT",
    ...     limit_price=2450.0
    ... )
    >>> print(f"Routed to {result.exchange.value}: {result.execution_price}")
    >>>
    >>> # Split large order across exchanges
    >>> results = router.split_order(
    ...     symbol="HDFCBANK",
    ...     quantity=50000,
    ...     side="BUY"
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import math


class Exchange(Enum):
    """Supported exchanges."""
    NSE = "NSE"
    BSE = "BSE"


class RoutingStrategy(Enum):
    """Order routing strategies."""
    BEST_PRICE = "best_price"           # Route to exchange with best price
    BEST_LIQUIDITY = "best_liquidity"   # Route to exchange with best liquidity
    LOWEST_COST = "lowest_cost"         # Route to minimize total costs
    LOWEST_LATENCY = "lowest_latency"   # Route to fastest exchange
    SMART = "smart"                     # Balanced optimization
    SPLIT = "split"                     # Split across exchanges


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExchangeQuote:
    """Real-time quote from an exchange."""
    exchange: Exchange
    symbol: str
    timestamp: datetime

    # Price data
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0

    # Depth data
    bid_quantity: int = 0
    ask_quantity: int = 0
    total_bid_depth: int = 0
    total_ask_depth: int = 0

    # Volume
    volume: int = 0
    avg_trade_size: int = 0

    # Circuit limits
    upper_circuit: float = 0.0
    lower_circuit: float = 0.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = (self.bid_price + self.ask_price) / 2
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid_price + self.ask_price) / 2

    def get_execution_price(self, side: str, quantity: int) -> float:
        """Estimate execution price for given side and quantity."""
        if side == "BUY":
            # For buy, we hit the ask
            if quantity <= self.ask_quantity:
                return self.ask_price
            # Large order may move price
            depth_ratio = quantity / max(self.total_ask_depth, 1)
            impact = min(depth_ratio * self.spread, self.spread * 3)
            return self.ask_price + impact
        else:
            # For sell, we hit the bid
            if quantity <= self.bid_quantity:
                return self.bid_price
            depth_ratio = quantity / max(self.total_bid_depth, 1)
            impact = min(depth_ratio * self.spread, self.spread * 3)
            return self.bid_price - impact


@dataclass
class TransactionCosts:
    """Transaction cost breakdown for an exchange."""
    exchange: Exchange

    # Regulatory charges (as percentage)
    stt_rate: float = 0.0           # Securities Transaction Tax
    exchange_fee_rate: float = 0.0  # Exchange transaction charges
    sebi_fee_rate: float = 0.0      # SEBI turnover fee
    stamp_duty_rate: float = 0.0    # Stamp duty
    gst_rate: float = 18.0          # GST on brokerage and charges

    # Brokerage
    brokerage_rate: float = 0.0     # Percentage
    brokerage_flat: float = 0.0     # Flat fee per order
    brokerage_max: float = 20.0     # Max brokerage per order

    def calculate_total_cost(
        self,
        price: float,
        quantity: int,
        side: str,
        is_intraday: bool = False
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs.

        Returns dict with cost breakdown.
        """
        turnover = price * quantity

        # STT rates differ by segment and trade type
        if side == "SELL":
            # STT only on sell side for equity delivery
            stt = turnover * (self.stt_rate / 100) if not is_intraday else turnover * 0.00025
        else:
            stt = 0.0 if not is_intraday else turnover * 0.00025

        # Exchange transaction charges
        exchange_charges = turnover * (self.exchange_fee_rate / 100)

        # SEBI fees
        sebi_charges = turnover * (self.sebi_fee_rate / 100)

        # Stamp duty (only on buy side)
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = turnover * (self.stamp_duty_rate / 100)

        # Brokerage
        brokerage_pct = turnover * (self.brokerage_rate / 100)
        brokerage = min(brokerage_pct + self.brokerage_flat, self.brokerage_max)

        # GST on brokerage and exchange charges
        gst = (brokerage + exchange_charges) * (self.gst_rate / 100)

        total = stt + exchange_charges + sebi_charges + stamp_duty + brokerage + gst

        return {
            'stt': round(stt, 2),
            'exchange_charges': round(exchange_charges, 2),
            'sebi_charges': round(sebi_charges, 2),
            'stamp_duty': round(stamp_duty, 2),
            'brokerage': round(brokerage, 2),
            'gst': round(gst, 2),
            'total': round(total, 2),
            'cost_bps': round((total / turnover) * 10000, 2) if turnover > 0 else 0.0
        }


@dataclass
class ExchangeLatency:
    """Latency statistics for an exchange."""
    exchange: Exchange

    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Recent samples
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)

    def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency_ms)
        self.last_updated = datetime.now()
        self._recalculate_stats()

    def _recalculate_stats(self) -> None:
        """Recalculate statistics from samples."""
        if not self.samples:
            return

        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)

        self.avg_latency_ms = sum(sorted_samples) / n
        self.min_latency_ms = sorted_samples[0]
        self.max_latency_ms = sorted_samples[-1]
        self.p50_latency_ms = sorted_samples[n // 2]
        self.p95_latency_ms = sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1]
        self.p99_latency_ms = sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1]

    @property
    def is_healthy(self) -> bool:
        """Check if exchange latency is acceptable."""
        return self.p95_latency_ms < 100  # Under 100ms is healthy


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    symbol: str
    side: str
    quantity: int

    # Routing result
    exchange: Exchange
    strategy_used: RoutingStrategy

    # Price info
    expected_price: float = 0.0
    price_improvement: float = 0.0  # vs worst exchange

    # Cost info
    estimated_costs: Dict[str, float] = field(default_factory=dict)
    total_savings: float = 0.0

    # Timing
    decision_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Metadata
    quotes: Dict[Exchange, ExchangeQuote] = field(default_factory=dict)
    reason: str = ""


@dataclass
class SplitResult:
    """Result of order splitting across exchanges."""
    symbol: str
    side: str
    total_quantity: int

    # Split allocation
    allocations: List[Tuple[Exchange, int, float]] = field(default_factory=list)  # (exchange, qty, price)

    # Aggregated metrics
    weighted_avg_price: float = 0.0
    total_costs: float = 0.0

    # Comparison
    single_exchange_price: float = 0.0  # What it would cost on single exchange
    savings: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SmartRouterConfig:
    """Configuration for smart router."""
    # Strategy preferences
    default_strategy: RoutingStrategy = RoutingStrategy.SMART

    # Thresholds
    min_price_improvement_bps: float = 1.0   # Min improvement to switch exchanges
    max_latency_penalty_ms: float = 50.0      # Max latency diff before penalizing
    liquidity_weight: float = 0.3             # Weight for liquidity in smart routing
    cost_weight: float = 0.3                  # Weight for costs in smart routing

    # Order splitting
    split_enabled: bool = True
    min_split_quantity: int = 1000            # Min quantity to consider splitting
    max_split_exchanges: int = 2              # Max exchanges to split across
    min_exchange_allocation_pct: float = 20.0 # Min % allocation per exchange

    # Safety
    circuit_limit_buffer_pct: float = 1.0     # Stay 1% away from circuit limits
    max_participation_pct: float = 10.0       # Max % of exchange volume

    # Caching
    quote_cache_ms: int = 100                 # Quote cache duration
    latency_update_interval_s: int = 60       # Latency stats update interval

    # Cost assumptions
    assume_intraday: bool = False             # Assume intraday for cost calc


# Default transaction costs for Indian exchanges
NSE_COSTS = TransactionCosts(
    exchange=Exchange.NSE,
    stt_rate=0.1,              # 0.1% on sell (delivery)
    exchange_fee_rate=0.00325, # NSE transaction charges
    sebi_fee_rate=0.0001,      # SEBI turnover fee
    stamp_duty_rate=0.015,     # Stamp duty on buy
    gst_rate=18.0,
    brokerage_rate=0.03,       # 0.03% or flat
    brokerage_flat=0.0,
    brokerage_max=20.0,
)

BSE_COSTS = TransactionCosts(
    exchange=Exchange.BSE,
    stt_rate=0.1,
    exchange_fee_rate=0.00275, # BSE charges slightly lower
    sebi_fee_rate=0.0001,
    stamp_duty_rate=0.015,
    gst_rate=18.0,
    brokerage_rate=0.03,
    brokerage_flat=0.0,
    brokerage_max=20.0,
)


class QuoteProvider(ABC):
    """Abstract interface for getting exchange quotes."""

    @abstractmethod
    def get_quote(self, symbol: str, exchange: Exchange) -> Optional[ExchangeQuote]:
        """Get current quote for symbol on exchange."""
        pass

    @abstractmethod
    def get_quotes(self, symbol: str) -> Dict[Exchange, ExchangeQuote]:
        """Get quotes from all exchanges."""
        pass


class DefaultQuoteProvider(QuoteProvider):
    """
    Default quote provider using broker API.

    Override this with actual broker integration.
    """

    def __init__(self, broker: Any = None):
        self.broker = broker
        self._cache: Dict[str, Tuple[datetime, Dict[Exchange, ExchangeQuote]]] = {}
        self._cache_ttl_ms = 100

    def get_quote(self, symbol: str, exchange: Exchange) -> Optional[ExchangeQuote]:
        """Get quote for specific exchange."""
        quotes = self.get_quotes(symbol)
        return quotes.get(exchange)

    def get_quotes(self, symbol: str) -> Dict[Exchange, ExchangeQuote]:
        """Get quotes from all exchanges."""
        # Check cache
        cache_key = symbol
        if cache_key in self._cache:
            cached_time, cached_quotes = self._cache[cache_key]
            age_ms = (datetime.now() - cached_time).total_seconds() * 1000
            if age_ms < self._cache_ttl_ms:
                return cached_quotes

        quotes = {}

        # Try to get from broker
        if self.broker is not None:
            try:
                # Get NSE quote
                nse_data = self._fetch_broker_quote(symbol, Exchange.NSE)
                if nse_data:
                    quotes[Exchange.NSE] = nse_data

                # Get BSE quote
                bse_data = self._fetch_broker_quote(symbol, Exchange.BSE)
                if bse_data:
                    quotes[Exchange.BSE] = bse_data
            except Exception:
                pass

        # If no broker data, create simulated quotes for testing
        if not quotes:
            quotes = self._create_simulated_quotes(symbol)

        # Cache results
        self._cache[cache_key] = (datetime.now(), quotes)

        return quotes

    def _fetch_broker_quote(self, symbol: str, exchange: Exchange) -> Optional[ExchangeQuote]:
        """Fetch quote from broker API."""
        if self.broker is None:
            return None

        try:
            # This would integrate with actual broker API
            # Example for Zerodha Kite:
            # instrument = f"{exchange.value}:{symbol}"
            # ltp = self.broker.ltp(instrument)
            # quote = self.broker.quote(instrument)

            # Placeholder - override in actual implementation
            return None
        except Exception:
            return None

    def _create_simulated_quotes(self, symbol: str) -> Dict[Exchange, ExchangeQuote]:
        """Create simulated quotes for testing."""
        import random

        base_price = 1000.0 + random.random() * 1000
        spread = base_price * 0.001  # 0.1% spread

        # NSE quote - typically more liquid
        nse_quote = ExchangeQuote(
            exchange=Exchange.NSE,
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=base_price - spread/2,
            ask_price=base_price + spread/2,
            last_price=base_price,
            bid_quantity=random.randint(500, 2000),
            ask_quantity=random.randint(500, 2000),
            total_bid_depth=random.randint(10000, 50000),
            total_ask_depth=random.randint(10000, 50000),
            volume=random.randint(100000, 1000000),
            avg_trade_size=random.randint(50, 200),
        )

        # BSE quote - slightly different prices, less liquid
        bse_offset = random.uniform(-0.002, 0.002) * base_price
        bse_quote = ExchangeQuote(
            exchange=Exchange.BSE,
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=base_price + bse_offset - spread/2,
            ask_price=base_price + bse_offset + spread/2,
            last_price=base_price + bse_offset,
            bid_quantity=random.randint(200, 1000),
            ask_quantity=random.randint(200, 1000),
            total_bid_depth=random.randint(5000, 25000),
            total_ask_depth=random.randint(5000, 25000),
            volume=random.randint(50000, 500000),
            avg_trade_size=random.randint(30, 100),
        )

        return {Exchange.NSE: nse_quote, Exchange.BSE: bse_quote}


class SmartRouter:
    """
    Multi-Exchange Smart Order Router.

    Intelligently routes orders across NSE and BSE for optimal execution
    based on price, liquidity, costs, and latency.

    Example:
        >>> router = SmartRouter(broker)
        >>>
        >>> # Simple routing
        >>> result = router.route_order("RELIANCE", 100, "BUY")
        >>> print(f"Route to {result.exchange.value}")
        >>>
        >>> # Specific strategy
        >>> result = router.route_order(
        ...     "HDFCBANK", 5000, "SELL",
        ...     strategy=RoutingStrategy.LOWEST_COST
        ... )
    """

    def __init__(
        self,
        broker: Any = None,
        quote_provider: Optional[QuoteProvider] = None,
        config: Optional[SmartRouterConfig] = None
    ):
        self.broker = broker
        self.quote_provider = quote_provider or DefaultQuoteProvider(broker)
        self.config = config or SmartRouterConfig()

        # Transaction costs per exchange
        self.costs: Dict[Exchange, TransactionCosts] = {
            Exchange.NSE: NSE_COSTS,
            Exchange.BSE: BSE_COSTS,
        }

        # Latency tracking per exchange
        self.latency: Dict[Exchange, ExchangeLatency] = {
            Exchange.NSE: ExchangeLatency(exchange=Exchange.NSE),
            Exchange.BSE: ExchangeLatency(exchange=Exchange.BSE),
        }

        # Callbacks
        self._on_route_decision: Optional[Callable[[RoutingDecision], None]] = None

        # Thread safety
        self._lock = threading.RLock()

    def set_transaction_costs(self, exchange: Exchange, costs: TransactionCosts) -> None:
        """Update transaction costs for an exchange."""
        with self._lock:
            self.costs[exchange] = costs

    def update_latency(self, exchange: Exchange, latency_ms: float) -> None:
        """Record a latency measurement for an exchange."""
        with self._lock:
            self.latency[exchange].add_sample(latency_ms)

    def on_route_decision(self, callback: Callable[[RoutingDecision], None]) -> None:
        """Set callback for routing decisions."""
        self._on_route_decision = callback

    def route_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "LIMIT",
        limit_price: Optional[float] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> RoutingDecision:
        """
        Route order to optimal exchange.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            quantity: Order quantity
            side: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            limit_price: Limit price (for limit orders)
            strategy: Routing strategy (uses default if not specified)

        Returns:
            RoutingDecision with selected exchange and analysis
        """
        start_time = time.time()
        strategy = strategy or self.config.default_strategy

        # Get quotes from all exchanges
        quotes = self.quote_provider.get_quotes(symbol)

        if not quotes:
            raise ValueError(f"No quotes available for {symbol}")

        # Select exchange based on strategy
        if strategy == RoutingStrategy.BEST_PRICE:
            selected, reason = self._route_by_price(quotes, side, quantity)
        elif strategy == RoutingStrategy.BEST_LIQUIDITY:
            selected, reason = self._route_by_liquidity(quotes, side, quantity)
        elif strategy == RoutingStrategy.LOWEST_COST:
            selected, reason = self._route_by_cost(quotes, side, quantity)
        elif strategy == RoutingStrategy.LOWEST_LATENCY:
            selected, reason = self._route_by_latency(quotes)
        elif strategy == RoutingStrategy.SMART:
            selected, reason = self._route_smart(quotes, side, quantity)
        else:
            # Default to best price
            selected, reason = self._route_by_price(quotes, side, quantity)

        # Calculate metrics
        selected_quote = quotes[selected]
        expected_price = selected_quote.get_execution_price(side, quantity)

        # Calculate costs
        costs = self.costs[selected].calculate_total_cost(
            expected_price, quantity, side,
            is_intraday=self.config.assume_intraday
        )

        # Calculate price improvement vs worst exchange
        worst_price = self._get_worst_price(quotes, side, quantity)
        if side == "BUY":
            price_improvement = worst_price - expected_price
        else:
            price_improvement = expected_price - worst_price

        # Calculate total savings
        worst_costs = self._get_highest_costs(quotes, side, quantity)
        total_savings = price_improvement * quantity + (worst_costs - costs['total'])

        decision = RoutingDecision(
            symbol=symbol,
            side=side,
            quantity=quantity,
            exchange=selected,
            strategy_used=strategy,
            expected_price=expected_price,
            price_improvement=price_improvement,
            estimated_costs=costs,
            total_savings=total_savings,
            decision_time_ms=(time.time() - start_time) * 1000,
            quotes=quotes,
            reason=reason
        )

        # Notify callback
        if self._on_route_decision:
            try:
                self._on_route_decision(decision)
            except Exception:
                pass

        return decision

    def split_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        min_allocation_pct: Optional[float] = None
    ) -> SplitResult:
        """
        Split a large order across multiple exchanges.

        Optimizes allocation based on:
        - Available liquidity at each exchange
        - Price levels at each exchange
        - Transaction costs

        Args:
            symbol: Trading symbol
            quantity: Total order quantity
            side: "BUY" or "SELL"
            min_allocation_pct: Minimum % per exchange (default from config)

        Returns:
            SplitResult with allocation per exchange
        """
        min_pct = min_allocation_pct or self.config.min_exchange_allocation_pct

        # Get quotes
        quotes = self.quote_provider.get_quotes(symbol)

        if len(quotes) < 2:
            # Only one exchange available, no split possible
            exchange = list(quotes.keys())[0]
            quote = quotes[exchange]
            price = quote.get_execution_price(side, quantity)
            costs = self.costs[exchange].calculate_total_cost(price, quantity, side)

            return SplitResult(
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                allocations=[(exchange, quantity, price)],
                weighted_avg_price=price,
                total_costs=costs['total'],
                single_exchange_price=price,
                savings=0.0
            )

        # Calculate optimal split
        allocations = self._calculate_optimal_split(quotes, side, quantity, min_pct)

        # Calculate metrics
        total_value = sum(qty * price for _, qty, price in allocations)
        weighted_avg_price = total_value / quantity if quantity > 0 else 0

        total_costs = sum(
            self.costs[exchange].calculate_total_cost(price, qty, side)['total']
            for exchange, qty, price in allocations
        )

        # Compare to single exchange execution
        single_exchange = Exchange.NSE  # Default to NSE for comparison
        single_quote = quotes.get(single_exchange, list(quotes.values())[0])
        single_price = single_quote.get_execution_price(side, quantity)
        single_costs = self.costs[single_exchange].calculate_total_cost(
            single_price, quantity, side
        )['total']

        # Calculate savings
        if side == "BUY":
            price_savings = (single_price - weighted_avg_price) * quantity
        else:
            price_savings = (weighted_avg_price - single_price) * quantity
        cost_savings = single_costs - total_costs
        total_savings = price_savings + cost_savings

        return SplitResult(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            allocations=allocations,
            weighted_avg_price=weighted_avg_price,
            total_costs=total_costs,
            single_exchange_price=single_price,
            savings=total_savings
        )

    def compare_exchanges(
        self,
        symbol: str,
        quantity: int,
        side: str
    ) -> Dict[Exchange, Dict[str, Any]]:
        """
        Compare execution across all exchanges.

        Returns detailed comparison including price, costs, and liquidity.
        """
        quotes = self.quote_provider.get_quotes(symbol)
        comparison = {}

        for exchange, quote in quotes.items():
            exec_price = quote.get_execution_price(side, quantity)
            costs = self.costs[exchange].calculate_total_cost(exec_price, quantity, side)
            latency = self.latency[exchange]

            # Calculate total cost including slippage
            if side == "BUY":
                available_qty = quote.ask_quantity
                depth = quote.total_ask_depth
            else:
                available_qty = quote.bid_quantity
                depth = quote.total_bid_depth

            comparison[exchange] = {
                'quote': quote,
                'execution_price': exec_price,
                'spread_bps': quote.spread_bps,
                'available_at_touch': available_qty,
                'total_depth': depth,
                'transaction_costs': costs,
                'total_cost': exec_price * quantity + costs['total'],
                'latency_p95_ms': latency.p95_latency_ms,
                'latency_healthy': latency.is_healthy,
            }

        return comparison

    def get_best_exchange(
        self,
        symbol: str,
        side: str
    ) -> Tuple[Exchange, float]:
        """
        Quick lookup for best exchange and price.

        Returns:
            Tuple of (best_exchange, best_price)
        """
        quotes = self.quote_provider.get_quotes(symbol)

        best_exchange = None
        best_price = float('inf') if side == "BUY" else 0.0

        for exchange, quote in quotes.items():
            if side == "BUY":
                price = quote.ask_price
                if price < best_price:
                    best_price = price
                    best_exchange = exchange
            else:
                price = quote.bid_price
                if price > best_price:
                    best_price = price
                    best_exchange = exchange

        return best_exchange, best_price

    # Private routing methods

    def _route_by_price(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> Tuple[Exchange, str]:
        """Route to exchange with best price."""
        best_exchange = None
        best_price = float('inf') if side == "BUY" else 0.0

        for exchange, quote in quotes.items():
            exec_price = quote.get_execution_price(side, quantity)

            if side == "BUY":
                if exec_price < best_price:
                    best_price = exec_price
                    best_exchange = exchange
            else:
                if exec_price > best_price:
                    best_price = exec_price
                    best_exchange = exchange

        other = [e for e in quotes.keys() if e != best_exchange]
        other_price = quotes[other[0]].get_execution_price(side, quantity) if other else best_price
        diff_bps = abs(best_price - other_price) / best_price * 10000 if best_price else 0

        return best_exchange, f"Best price by {diff_bps:.1f} bps"

    def _route_by_liquidity(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> Tuple[Exchange, str]:
        """Route to exchange with best liquidity."""
        best_exchange = None
        best_liquidity = 0

        for exchange, quote in quotes.items():
            if side == "BUY":
                liquidity = quote.total_ask_depth
            else:
                liquidity = quote.total_bid_depth

            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_exchange = exchange

        return best_exchange, f"Best liquidity: {best_liquidity:,} shares"

    def _route_by_cost(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> Tuple[Exchange, str]:
        """Route to exchange with lowest total cost."""
        best_exchange = None
        lowest_cost = float('inf')

        for exchange, quote in quotes.items():
            exec_price = quote.get_execution_price(side, quantity)
            costs = self.costs[exchange].calculate_total_cost(exec_price, quantity, side)
            total_cost = exec_price * quantity + costs['total']

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_exchange = exchange

        return best_exchange, f"Lowest total cost: ₹{lowest_cost:,.2f}"

    def _route_by_latency(
        self,
        quotes: Dict[Exchange, ExchangeQuote]
    ) -> Tuple[Exchange, str]:
        """Route to exchange with lowest latency."""
        best_exchange = None
        lowest_latency = float('inf')

        for exchange in quotes.keys():
            latency = self.latency[exchange].p95_latency_ms
            if latency < lowest_latency:
                lowest_latency = latency
                best_exchange = exchange

        return best_exchange, f"Lowest latency: {lowest_latency:.1f}ms"

    def _route_smart(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> Tuple[Exchange, str]:
        """
        Smart routing using weighted scoring.

        Considers:
        - Price (40% weight)
        - Liquidity (30% weight)
        - Costs (20% weight)
        - Latency (10% weight)
        """
        scores: Dict[Exchange, float] = {}

        # Calculate metrics for each exchange
        metrics = {}
        for exchange, quote in quotes.items():
            exec_price = quote.get_execution_price(side, quantity)
            if side == "BUY":
                liquidity = quote.total_ask_depth
            else:
                liquidity = quote.total_bid_depth
            costs = self.costs[exchange].calculate_total_cost(exec_price, quantity, side)
            latency = self.latency[exchange].p95_latency_ms

            metrics[exchange] = {
                'price': exec_price,
                'liquidity': liquidity,
                'cost_bps': costs['cost_bps'],
                'latency': latency
            }

        # Normalize and score
        prices = [m['price'] for m in metrics.values()]
        liquidities = [m['liquidity'] for m in metrics.values()]
        costs = [m['cost_bps'] for m in metrics.values()]
        latencies = [m['latency'] for m in metrics.values()]

        for exchange, m in metrics.items():
            # Price score (lower is better for buy, higher for sell)
            if side == "BUY":
                price_score = 1 - (m['price'] - min(prices)) / (max(prices) - min(prices) + 0.001)
            else:
                price_score = (m['price'] - min(prices)) / (max(prices) - min(prices) + 0.001)

            # Liquidity score (higher is better)
            liq_range = max(liquidities) - min(liquidities) + 1
            liquidity_score = (m['liquidity'] - min(liquidities)) / liq_range

            # Cost score (lower is better)
            cost_range = max(costs) - min(costs) + 0.001
            cost_score = 1 - (m['cost_bps'] - min(costs)) / cost_range

            # Latency score (lower is better)
            lat_range = max(latencies) - min(latencies) + 0.001
            latency_score = 1 - (m['latency'] - min(latencies)) / lat_range

            # Weighted total
            scores[exchange] = (
                0.4 * price_score +
                0.3 * liquidity_score +
                0.2 * cost_score +
                0.1 * latency_score
            )

        best_exchange = max(scores, key=scores.get)
        return best_exchange, f"Smart score: {scores[best_exchange]:.3f}"

    def _calculate_optimal_split(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int,
        min_allocation_pct: float
    ) -> List[Tuple[Exchange, int, float]]:
        """Calculate optimal order split across exchanges."""
        min_qty = int(quantity * min_allocation_pct / 100)

        # Get available liquidity at each exchange
        liquidities = {}
        for exchange, quote in quotes.items():
            if side == "BUY":
                liquidities[exchange] = quote.total_ask_depth
            else:
                liquidities[exchange] = quote.total_bid_depth

        total_liquidity = sum(liquidities.values())

        # Allocate proportionally to liquidity, respecting minimums
        allocations = []
        remaining = quantity

        # Sort by liquidity (most liquid first)
        sorted_exchanges = sorted(liquidities.keys(), key=lambda x: liquidities[x], reverse=True)

        for i, exchange in enumerate(sorted_exchanges):
            if i >= self.config.max_split_exchanges:
                break

            # Proportional allocation
            liquidity_share = liquidities[exchange] / total_liquidity
            target_qty = int(quantity * liquidity_share)

            # Apply minimum
            target_qty = max(target_qty, min_qty)

            # Don't exceed remaining
            actual_qty = min(target_qty, remaining)

            if actual_qty > 0:
                quote = quotes[exchange]
                price = quote.get_execution_price(side, actual_qty)
                allocations.append((exchange, actual_qty, price))
                remaining -= actual_qty

        # Assign any remaining to most liquid exchange
        if remaining > 0 and allocations:
            exchange, qty, price = allocations[0]
            quote = quotes[exchange]
            new_qty = qty + remaining
            new_price = quote.get_execution_price(side, new_qty)
            allocations[0] = (exchange, new_qty, new_price)

        return allocations

    def _get_worst_price(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> float:
        """Get worst execution price across exchanges."""
        if side == "BUY":
            return max(q.get_execution_price(side, quantity) for q in quotes.values())
        else:
            return min(q.get_execution_price(side, quantity) for q in quotes.values())

    def _get_highest_costs(
        self,
        quotes: Dict[Exchange, ExchangeQuote],
        side: str,
        quantity: int
    ) -> float:
        """Get highest transaction costs across exchanges."""
        highest = 0.0
        for exchange, quote in quotes.items():
            price = quote.get_execution_price(side, quantity)
            costs = self.costs[exchange].calculate_total_cost(price, quantity, side)
            if costs['total'] > highest:
                highest = costs['total']
        return highest


# Convenience functions

_router_instance: Optional[SmartRouter] = None


def get_smart_router() -> Optional[SmartRouter]:
    """Get global smart router instance."""
    return _router_instance


def set_smart_router(router: SmartRouter) -> None:
    """Set global smart router instance."""
    global _router_instance
    _router_instance = router


def route_order(
    symbol: str,
    quantity: int,
    side: str,
    strategy: Optional[RoutingStrategy] = None
) -> RoutingDecision:
    """
    Route order using global router.

    Example:
        >>> result = route_order("RELIANCE", 100, "BUY")
        >>> print(result.exchange)
    """
    router = get_smart_router()
    if router is None:
        router = SmartRouter()
        set_smart_router(router)

    return router.route_order(symbol, quantity, side, strategy=strategy)


def compare_prices(symbol: str, quantity: int, side: str) -> Dict[Exchange, float]:
    """
    Quick price comparison across exchanges.

    Returns dict mapping exchange to expected execution price.
    """
    router = get_smart_router()
    if router is None:
        router = SmartRouter()
        set_smart_router(router)

    quotes = router.quote_provider.get_quotes(symbol)
    return {
        exchange: quote.get_execution_price(side, quantity)
        for exchange, quote in quotes.items()
    }


def best_exchange_for(symbol: str, side: str) -> Tuple[Exchange, float]:
    """
    Get best exchange for a symbol.

    Returns tuple of (exchange, price).
    """
    router = get_smart_router()
    if router is None:
        router = SmartRouter()
        set_smart_router(router)

    return router.get_best_exchange(symbol, side)
