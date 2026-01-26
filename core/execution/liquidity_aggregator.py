# -*- coding: utf-8 -*-
"""
Liquidity Aggregator - Multi-Exchange Order Book Aggregation
=============================================================
Combines order books from multiple exchanges for unified liquidity view
and optimal execution planning.

Features:
- Aggregate order books from NSE and BSE
- Unified depth-of-market view
- Liquidity analysis and scoring
- Optimal execution path planning
- VWAP estimation across exchanges

Example:
    >>> from core.execution import LiquidityAggregator, Exchange
    >>>
    >>> aggregator = LiquidityAggregator(broker)
    >>>
    >>> # Get unified order book
    >>> book = aggregator.get_aggregated_book("RELIANCE")
    >>> print(f"Total bid liquidity: {book.total_bid_quantity:,}")
    >>> print(f"Best bid: {book.best_bid}")
    >>>
    >>> # Plan optimal execution
    >>> plan = aggregator.plan_execution("RELIANCE", 10000, "BUY")
    >>> for step in plan.execution_steps:
    ...     print(f"Execute {step.quantity} @ {step.exchange.value}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict
import threading
import math

from .smart_router import (
    Exchange,
    ExchangeQuote,
    QuoteProvider,
    DefaultQuoteProvider,
    TransactionCosts,
    NSE_COSTS,
    BSE_COSTS,
)


class LiquidityTier(Enum):
    """Liquidity classification tiers."""
    HIGHLY_LIQUID = "highly_liquid"     # Top 100 by volume
    LIQUID = "liquid"                    # Easy to trade
    MODERATE = "moderate"                # Some impact expected
    ILLIQUID = "illiquid"               # Significant impact
    VERY_ILLIQUID = "very_illiquid"     # Hard to trade


class ExecutionUrgency(Enum):
    """Execution urgency levels."""
    IMMEDIATE = "immediate"      # Execute now at any cost
    AGGRESSIVE = "aggressive"    # Fast execution, accept some slippage
    NORMAL = "normal"           # Balance speed and cost
    PASSIVE = "passive"         # Minimize impact, accept delay
    PATIENT = "patient"         # Wait for optimal conditions


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: int
    exchange: Exchange
    order_count: int = 1

    @property
    def value(self) -> float:
        """Total value at this level."""
        return self.price * self.quantity


@dataclass
class AggregatedLevel:
    """Aggregated price level across exchanges."""
    price: float
    total_quantity: int
    contributions: List[Tuple[Exchange, int]] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Total value at this level."""
        return self.price * self.total_quantity

    @property
    def exchange_count(self) -> int:
        """Number of exchanges contributing."""
        return len(self.contributions)

    def get_quantity_at(self, exchange: Exchange) -> int:
        """Get quantity from specific exchange."""
        for ex, qty in self.contributions:
            if ex == exchange:
                return qty
        return 0


@dataclass
class OrderBook:
    """Order book for a single exchange."""
    exchange: Exchange
    symbol: str
    timestamp: datetime

    # Bid side (buy orders) - sorted descending by price
    bids: List[OrderBookLevel] = field(default_factory=list)

    # Ask side (sell orders) - sorted ascending by price
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.spread and self.best_bid:
            mid = (self.best_bid + self.best_ask) / 2
            return (self.spread / mid) * 10000
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def total_bid_quantity(self) -> int:
        """Total quantity on bid side."""
        return sum(level.quantity for level in self.bids)

    @property
    def total_ask_quantity(self) -> int:
        """Total quantity on ask side."""
        return sum(level.quantity for level in self.asks)

    @property
    def bid_depth_value(self) -> float:
        """Total value on bid side."""
        return sum(level.value for level in self.bids)

    @property
    def ask_depth_value(self) -> float:
        """Total value on ask side."""
        return sum(level.value for level in self.asks)

    def get_quantity_at_price(self, price: float, side: str) -> int:
        """Get quantity available at a specific price."""
        levels = self.bids if side == "BUY" else self.asks
        for level in levels:
            if abs(level.price - price) < 0.01:
                return level.quantity
        return 0

    def estimate_fill_price(self, quantity: int, side: str) -> Tuple[float, int]:
        """
        Estimate average fill price for given quantity.

        Returns:
            Tuple of (average_price, filled_quantity)
        """
        levels = self.asks if side == "BUY" else self.bids
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

        avg_price = total_cost / filled if filled > 0 else 0
        return avg_price, filled


@dataclass
class AggregatedOrderBook:
    """Aggregated order book across all exchanges."""
    symbol: str
    timestamp: datetime

    # Aggregated levels
    bids: List[AggregatedLevel] = field(default_factory=list)
    asks: List[AggregatedLevel] = field(default_factory=list)

    # Source books
    source_books: Dict[Exchange, OrderBook] = field(default_factory=dict)

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price across all exchanges."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price across all exchanges."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Aggregated spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.spread and self.best_bid:
            mid = (self.best_bid + self.best_ask) / 2
            return (self.spread / mid) * 10000
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def total_bid_quantity(self) -> int:
        """Total quantity on bid side."""
        return sum(level.total_quantity for level in self.bids)

    @property
    def total_ask_quantity(self) -> int:
        """Total quantity on ask side."""
        return sum(level.total_quantity for level in self.asks)

    @property
    def exchange_count(self) -> int:
        """Number of exchanges in aggregation."""
        return len(self.source_books)

    def get_liquidity_at_distance(self, distance_bps: float, side: str) -> int:
        """Get available liquidity within price distance from best."""
        if side == "BUY":
            if not self.best_ask:
                return 0
            max_price = self.best_ask * (1 + distance_bps / 10000)
            return sum(
                level.total_quantity for level in self.asks
                if level.price <= max_price
            )
        else:
            if not self.best_bid:
                return 0
            min_price = self.best_bid * (1 - distance_bps / 10000)
            return sum(
                level.total_quantity for level in self.bids
                if level.price >= min_price
            )

    def estimate_vwap(self, quantity: int, side: str) -> Tuple[float, Dict[Exchange, int]]:
        """
        Estimate VWAP and exchange allocation for given quantity.

        Returns:
            Tuple of (vwap, allocation_dict)
        """
        levels = self.asks if side == "BUY" else self.bids
        remaining = quantity
        total_cost = 0.0
        allocation: Dict[Exchange, int] = defaultdict(int)

        for level in levels:
            fill_qty = min(remaining, level.total_quantity)
            total_cost += fill_qty * level.price

            # Distribute fill across contributing exchanges proportionally
            level_remaining = fill_qty
            for exchange, ex_qty in level.contributions:
                ex_fill = min(level_remaining, ex_qty)
                allocation[exchange] += ex_fill
                level_remaining -= ex_fill
                if level_remaining <= 0:
                    break

            remaining -= fill_qty
            if remaining <= 0:
                break

        filled = quantity - remaining
        vwap = total_cost / filled if filled > 0 else 0
        return vwap, dict(allocation)


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    symbol: str
    timestamp: datetime

    # Basic metrics
    tier: LiquidityTier = LiquidityTier.MODERATE
    liquidity_score: float = 50.0  # 0-100 score

    # Spread metrics
    spread_bps: float = 0.0
    effective_spread_bps: float = 0.0

    # Depth metrics
    bid_depth_value: float = 0.0
    ask_depth_value: float = 0.0
    total_depth_value: float = 0.0

    # Volume metrics
    avg_trade_size: int = 0
    daily_volume: int = 0
    volume_participation_1pct: int = 0  # Qty to be 1% of volume

    # Impact metrics
    impact_10k_bps: float = 0.0   # Price impact for 10k shares
    impact_50k_bps: float = 0.0   # Price impact for 50k shares
    impact_100k_bps: float = 0.0  # Price impact for 100k shares

    # Exchange breakdown
    exchange_liquidity: Dict[Exchange, float] = field(default_factory=dict)
    dominant_exchange: Optional[Exchange] = None

    @property
    def is_liquid(self) -> bool:
        """Check if symbol is sufficiently liquid."""
        return self.tier in (LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID)


@dataclass
class ExecutionStep:
    """Single step in execution plan."""
    exchange: Exchange
    quantity: int
    expected_price: float
    price_level: int  # Which level in order book

    # Cost estimates
    slippage_bps: float = 0.0
    transaction_cost: float = 0.0

    @property
    def total_value(self) -> float:
        """Total value for this step."""
        return self.quantity * self.expected_price

    @property
    def total_cost(self) -> float:
        """Total cost including slippage and fees."""
        slippage_cost = self.total_value * (self.slippage_bps / 10000)
        return slippage_cost + self.transaction_cost


@dataclass
class ExecutionPlan:
    """Optimal execution plan across exchanges."""
    symbol: str
    side: str
    total_quantity: int
    timestamp: datetime

    # Execution steps
    execution_steps: List[ExecutionStep] = field(default_factory=list)

    # Aggregated metrics
    expected_vwap: float = 0.0
    expected_slippage_bps: float = 0.0
    total_transaction_cost: float = 0.0

    # Comparison
    single_exchange_vwap: float = 0.0  # For comparison
    improvement_bps: float = 0.0

    # Timing
    recommended_urgency: ExecutionUrgency = ExecutionUrgency.NORMAL
    estimated_completion_seconds: float = 0.0

    @property
    def total_value(self) -> float:
        """Total value of execution."""
        return sum(step.total_value for step in self.execution_steps)

    @property
    def total_cost(self) -> float:
        """Total cost including all fees and slippage."""
        return sum(step.total_cost for step in self.execution_steps)

    @property
    def exchange_allocation(self) -> Dict[Exchange, int]:
        """Quantity allocation per exchange."""
        allocation: Dict[Exchange, int] = defaultdict(int)
        for step in self.execution_steps:
            allocation[step.exchange] += step.quantity
        return dict(allocation)

    def get_exchange_summary(self) -> List[Dict[str, Any]]:
        """Get summary by exchange."""
        summary = defaultdict(lambda: {'quantity': 0, 'value': 0.0, 'cost': 0.0})
        for step in self.execution_steps:
            summary[step.exchange]['quantity'] += step.quantity
            summary[step.exchange]['value'] += step.total_value
            summary[step.exchange]['cost'] += step.total_cost

        return [
            {
                'exchange': ex,
                'quantity': data['quantity'],
                'value': data['value'],
                'cost': data['cost'],
                'pct': (data['quantity'] / self.total_quantity * 100) if self.total_quantity else 0
            }
            for ex, data in summary.items()
        ]


@dataclass
class LiquidityAggregatorConfig:
    """Configuration for liquidity aggregator."""
    # Order book depth
    max_levels: int = 20                    # Max price levels to aggregate
    price_grouping_bps: float = 1.0         # Group prices within this range

    # Liquidity thresholds
    highly_liquid_volume: int = 10_000_000  # Daily volume threshold
    liquid_volume: int = 1_000_000
    moderate_volume: int = 100_000
    illiquid_volume: int = 10_000

    # Impact calculation
    impact_qty_tiers: List[int] = field(default_factory=lambda: [10000, 50000, 100000])

    # Execution planning
    max_participation_pct: float = 5.0      # Max % of level to take
    min_step_quantity: int = 100            # Min quantity per step

    # Caching
    cache_ttl_ms: int = 100                 # Order book cache duration


class OrderBookProvider:
    """
    Provider for exchange order books.

    Override this class for actual broker integration.
    """

    def __init__(self, broker: Any = None):
        self.broker = broker

    def get_order_book(self, symbol: str, exchange: Exchange, levels: int = 20) -> Optional[OrderBook]:
        """Get order book for symbol from exchange."""
        # Override in actual implementation
        # This creates simulated data for testing
        return self._create_simulated_book(symbol, exchange, levels)

    def _create_simulated_book(self, symbol: str, exchange: Exchange, levels: int) -> OrderBook:
        """Create simulated order book for testing."""
        import random

        base_price = 1000.0 + random.random() * 1000
        tick_size = 0.05

        # NSE typically more liquid
        base_qty = 500 if exchange == Exchange.NSE else 300

        bids = []
        asks = []

        for i in range(levels):
            # Bid side - decreasing prices
            bid_price = round(base_price - (i + 1) * tick_size, 2)
            bid_qty = int(base_qty * random.uniform(0.5, 2.0) * (1 - i * 0.03))
            bids.append(OrderBookLevel(
                price=bid_price,
                quantity=max(bid_qty, 1),
                exchange=exchange,
                order_count=random.randint(1, 10)
            ))

            # Ask side - increasing prices
            ask_price = round(base_price + (i + 1) * tick_size, 2)
            ask_qty = int(base_qty * random.uniform(0.5, 2.0) * (1 - i * 0.03))
            asks.append(OrderBookLevel(
                price=ask_price,
                quantity=max(ask_qty, 1),
                exchange=exchange,
                order_count=random.randint(1, 10)
            ))

        return OrderBook(
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )


class LiquidityAggregator:
    """
    Multi-Exchange Liquidity Aggregator.

    Combines order books from multiple exchanges to provide:
    - Unified view of available liquidity
    - Optimal execution planning
    - Liquidity analysis and scoring

    Example:
        >>> aggregator = LiquidityAggregator(broker)
        >>> book = aggregator.get_aggregated_book("RELIANCE")
        >>> print(f"Best bid: {book.best_bid}, Best ask: {book.best_ask}")
    """

    def __init__(
        self,
        broker: Any = None,
        book_provider: Optional[OrderBookProvider] = None,
        config: Optional[LiquidityAggregatorConfig] = None
    ):
        self.broker = broker
        self.book_provider = book_provider or OrderBookProvider(broker)
        self.config = config or LiquidityAggregatorConfig()

        # Transaction costs
        self.costs: Dict[Exchange, TransactionCosts] = {
            Exchange.NSE: NSE_COSTS,
            Exchange.BSE: BSE_COSTS,
        }

        # Cache
        self._cache: Dict[str, Tuple[datetime, AggregatedOrderBook]] = {}
        self._lock = threading.RLock()

    def get_order_book(self, symbol: str, exchange: Exchange) -> Optional[OrderBook]:
        """Get order book for single exchange."""
        return self.book_provider.get_order_book(symbol, exchange, self.config.max_levels)

    def get_aggregated_book(self, symbol: str) -> AggregatedOrderBook:
        """
        Get aggregated order book across all exchanges.

        Combines bid/ask levels from NSE and BSE into unified view,
        sorted by best price.
        """
        # Check cache
        cache_key = symbol
        with self._lock:
            if cache_key in self._cache:
                cached_time, cached_book = self._cache[cache_key]
                age_ms = (datetime.now() - cached_time).total_seconds() * 1000
                if age_ms < self.config.cache_ttl_ms:
                    return cached_book

        # Fetch order books from all exchanges
        source_books: Dict[Exchange, OrderBook] = {}
        for exchange in [Exchange.NSE, Exchange.BSE]:
            book = self.get_order_book(symbol, exchange)
            if book:
                source_books[exchange] = book

        # Aggregate bids and asks
        aggregated_bids = self._aggregate_levels(
            [book.bids for book in source_books.values()],
            list(source_books.keys()),
            ascending=False  # Bids sorted descending
        )

        aggregated_asks = self._aggregate_levels(
            [book.asks for book in source_books.values()],
            list(source_books.keys()),
            ascending=True  # Asks sorted ascending
        )

        result = AggregatedOrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=aggregated_bids,
            asks=aggregated_asks,
            source_books=source_books
        )

        # Cache result
        with self._lock:
            self._cache[cache_key] = (datetime.now(), result)

        return result

    def _aggregate_levels(
        self,
        level_lists: List[List[OrderBookLevel]],
        exchanges: List[Exchange],
        ascending: bool
    ) -> List[AggregatedLevel]:
        """Aggregate price levels from multiple sources."""
        # Collect all levels with exchange info
        all_levels: List[Tuple[float, int, Exchange]] = []
        for levels, exchange in zip(level_lists, exchanges):
            for level in levels:
                all_levels.append((level.price, level.quantity, exchange))

        # Group by price (within tolerance)
        tolerance = self.config.price_grouping_bps / 10000
        grouped: Dict[float, List[Tuple[int, Exchange]]] = defaultdict(list)

        for price, qty, exchange in all_levels:
            # Find existing group or create new one
            found_group = None
            for group_price in grouped.keys():
                if abs(price - group_price) / group_price < tolerance:
                    found_group = group_price
                    break

            if found_group:
                grouped[found_group].append((qty, exchange))
            else:
                grouped[price].append((qty, exchange))

        # Create aggregated levels
        aggregated = []
        for price, contributions in grouped.items():
            total_qty = sum(qty for qty, _ in contributions)
            aggregated.append(AggregatedLevel(
                price=price,
                total_quantity=total_qty,
                contributions=[(ex, qty) for qty, ex in contributions]
            ))

        # Sort by price
        aggregated.sort(key=lambda x: x.price, reverse=not ascending)

        return aggregated[:self.config.max_levels]

    def analyze_liquidity(
        self,
        symbol: str,
        daily_volume: Optional[int] = None
    ) -> LiquidityMetrics:
        """
        Analyze liquidity for a symbol.

        Returns comprehensive liquidity metrics including tier classification,
        depth analysis, and impact estimates.
        """
        book = self.get_aggregated_book(symbol)

        # Estimate daily volume if not provided
        if daily_volume is None:
            # Use depth as proxy (crude estimate)
            daily_volume = book.total_bid_quantity * 100

        # Calculate spread metrics
        spread_bps = book.spread_bps or 0.0

        # Calculate depth
        bid_depth = sum(level.price * level.total_quantity for level in book.bids)
        ask_depth = sum(level.price * level.total_quantity for level in book.asks)
        total_depth = bid_depth + ask_depth

        # Calculate impact at different quantities
        impacts = {}
        for qty in self.config.impact_qty_tiers:
            vwap, _ = book.estimate_vwap(qty, "BUY")
            if book.best_ask and vwap:
                impact = (vwap - book.best_ask) / book.best_ask * 10000
                impacts[qty] = max(0, impact)
            else:
                impacts[qty] = 0.0

        # Determine liquidity tier
        if daily_volume >= self.config.highly_liquid_volume:
            tier = LiquidityTier.HIGHLY_LIQUID
        elif daily_volume >= self.config.liquid_volume:
            tier = LiquidityTier.LIQUID
        elif daily_volume >= self.config.moderate_volume:
            tier = LiquidityTier.MODERATE
        elif daily_volume >= self.config.illiquid_volume:
            tier = LiquidityTier.ILLIQUID
        else:
            tier = LiquidityTier.VERY_ILLIQUID

        # Calculate liquidity score (0-100)
        volume_score = min(daily_volume / self.config.highly_liquid_volume * 40, 40)
        spread_score = max(0, 30 - spread_bps * 2)  # Lower spread = higher score
        depth_score = min(total_depth / 10_000_000 * 30, 30)  # Up to 30 points
        liquidity_score = volume_score + spread_score + depth_score

        # Exchange breakdown
        exchange_liquidity = {}
        for exchange, source_book in book.source_books.items():
            exchange_liquidity[exchange] = source_book.total_bid_quantity + source_book.total_ask_quantity

        # Find dominant exchange
        dominant = max(exchange_liquidity, key=exchange_liquidity.get) if exchange_liquidity else None

        return LiquidityMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            tier=tier,
            liquidity_score=round(liquidity_score, 1),
            spread_bps=spread_bps,
            effective_spread_bps=spread_bps * 1.2,  # Estimate with impact
            bid_depth_value=bid_depth,
            ask_depth_value=ask_depth,
            total_depth_value=total_depth,
            avg_trade_size=100,  # Would come from trade data
            daily_volume=daily_volume,
            volume_participation_1pct=daily_volume // 100,
            impact_10k_bps=impacts.get(10000, 0.0),
            impact_50k_bps=impacts.get(50000, 0.0),
            impact_100k_bps=impacts.get(100000, 0.0),
            exchange_liquidity=exchange_liquidity,
            dominant_exchange=dominant
        )

    def plan_execution(
        self,
        symbol: str,
        quantity: int,
        side: str,
        urgency: ExecutionUrgency = ExecutionUrgency.NORMAL
    ) -> ExecutionPlan:
        """
        Create optimal execution plan across exchanges.

        Walks through aggregated order book to find optimal path
        that minimizes total cost including slippage and fees.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: "BUY" or "SELL"
            urgency: How quickly to execute

        Returns:
            ExecutionPlan with detailed steps
        """
        book = self.get_aggregated_book(symbol)
        levels = book.asks if side == "BUY" else book.bids

        if not levels:
            raise ValueError(f"No liquidity available for {symbol}")

        steps: List[ExecutionStep] = []
        remaining = quantity
        reference_price = levels[0].price  # Best price as reference

        for level_idx, level in enumerate(levels):
            if remaining <= 0:
                break

            # Determine how much to take from this level
            max_take = int(level.total_quantity * (self.config.max_participation_pct / 100))
            take_qty = min(remaining, max_take)

            if take_qty < self.config.min_step_quantity and remaining > self.config.min_step_quantity:
                continue

            # Allocate across exchanges at this level
            level_remaining = take_qty
            for exchange, ex_qty in level.contributions:
                if level_remaining <= 0:
                    break

                # Calculate exchange allocation
                ex_take = min(level_remaining, ex_qty)
                if ex_take < self.config.min_step_quantity and level_remaining > self.config.min_step_quantity:
                    continue

                # Calculate costs
                slippage_bps = (level.price - reference_price) / reference_price * 10000 if reference_price else 0
                if side == "SELL":
                    slippage_bps = -slippage_bps

                costs = self.costs[exchange].calculate_total_cost(
                    level.price, ex_take, side
                )

                steps.append(ExecutionStep(
                    exchange=exchange,
                    quantity=ex_take,
                    expected_price=level.price,
                    price_level=level_idx,
                    slippage_bps=abs(slippage_bps),
                    transaction_cost=costs['total']
                ))

                level_remaining -= ex_take

            remaining -= take_qty

        # Handle remaining quantity (if any)
        if remaining > 0 and levels:
            # Allocate to dominant exchange at worst price
            worst_level = levels[-1]
            dominant_ex = max(
                worst_level.contributions,
                key=lambda x: x[1]
            )[0] if worst_level.contributions else Exchange.NSE

            slippage_bps = (worst_level.price - reference_price) / reference_price * 10000 if reference_price else 0
            if side == "SELL":
                slippage_bps = -slippage_bps

            costs = self.costs[dominant_ex].calculate_total_cost(
                worst_level.price, remaining, side
            )

            steps.append(ExecutionStep(
                exchange=dominant_ex,
                quantity=remaining,
                expected_price=worst_level.price,
                price_level=len(levels) - 1,
                slippage_bps=abs(slippage_bps) + 10,  # Extra for uncertainty
                transaction_cost=costs['total']
            ))

        # Calculate aggregated metrics
        total_value = sum(s.total_value for s in steps)
        filled_qty = sum(s.quantity for s in steps)
        expected_vwap = total_value / filled_qty if filled_qty else 0

        total_slippage_cost = sum(
            s.total_value * (s.slippage_bps / 10000) for s in steps
        )
        expected_slippage_bps = (total_slippage_cost / total_value * 10000) if total_value else 0

        total_transaction_cost = sum(s.transaction_cost for s in steps)

        # Compare to single exchange (NSE)
        nse_book = book.source_books.get(Exchange.NSE)
        if nse_book:
            single_vwap, _ = nse_book.estimate_fill_price(quantity, side)
        else:
            single_vwap = expected_vwap

        if side == "BUY":
            improvement_bps = (single_vwap - expected_vwap) / expected_vwap * 10000 if expected_vwap else 0
        else:
            improvement_bps = (expected_vwap - single_vwap) / expected_vwap * 10000 if expected_vwap else 0

        # Estimate completion time based on urgency
        completion_times = {
            ExecutionUrgency.IMMEDIATE: 1.0,
            ExecutionUrgency.AGGRESSIVE: 10.0,
            ExecutionUrgency.NORMAL: 60.0,
            ExecutionUrgency.PASSIVE: 300.0,
            ExecutionUrgency.PATIENT: 900.0,
        }

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            timestamp=datetime.now(),
            execution_steps=steps,
            expected_vwap=expected_vwap,
            expected_slippage_bps=expected_slippage_bps,
            total_transaction_cost=total_transaction_cost,
            single_exchange_vwap=single_vwap,
            improvement_bps=improvement_bps,
            recommended_urgency=urgency,
            estimated_completion_seconds=completion_times[urgency]
        )

    def get_unified_depth(
        self,
        symbol: str,
        levels: int = 5
    ) -> Dict[str, Any]:
        """
        Get simplified unified depth view.

        Returns easy-to-use depth summary for UI display.
        """
        book = self.get_aggregated_book(symbol)

        bid_levels = []
        for i, level in enumerate(book.bids[:levels]):
            bid_levels.append({
                'price': level.price,
                'quantity': level.total_quantity,
                'exchanges': [ex.value for ex, _ in level.contributions]
            })

        ask_levels = []
        for i, level in enumerate(book.asks[:levels]):
            ask_levels.append({
                'price': level.price,
                'quantity': level.total_quantity,
                'exchanges': [ex.value for ex, _ in level.contributions]
            })

        return {
            'symbol': symbol,
            'timestamp': book.timestamp.isoformat(),
            'best_bid': book.best_bid,
            'best_ask': book.best_ask,
            'spread': book.spread,
            'spread_bps': book.spread_bps,
            'bids': bid_levels,
            'asks': ask_levels,
            'total_bid_qty': book.total_bid_quantity,
            'total_ask_qty': book.total_ask_quantity,
            'exchanges': list(book.source_books.keys())
        }

    def estimate_execution_cost(
        self,
        symbol: str,
        quantity: int,
        side: str
    ) -> Dict[str, Any]:
        """
        Quick estimation of execution cost.

        Returns dict with vwap, slippage, and cost breakdown.
        """
        plan = self.plan_execution(symbol, quantity, side)

        return {
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'expected_vwap': plan.expected_vwap,
            'expected_slippage_bps': plan.expected_slippage_bps,
            'transaction_cost': plan.total_transaction_cost,
            'total_cost': plan.total_cost,
            'exchange_allocation': plan.exchange_allocation,
            'improvement_vs_single': plan.improvement_bps
        }


# Convenience functions

_aggregator_instance: Optional[LiquidityAggregator] = None


def get_liquidity_aggregator() -> Optional[LiquidityAggregator]:
    """Get global liquidity aggregator instance."""
    return _aggregator_instance


def set_liquidity_aggregator(aggregator: LiquidityAggregator) -> None:
    """Set global liquidity aggregator instance."""
    global _aggregator_instance
    _aggregator_instance = aggregator


def get_aggregated_book(symbol: str) -> AggregatedOrderBook:
    """Get aggregated order book using global aggregator."""
    aggregator = get_liquidity_aggregator()
    if aggregator is None:
        aggregator = LiquidityAggregator()
        set_liquidity_aggregator(aggregator)
    return aggregator.get_aggregated_book(symbol)


def analyze_liquidity(symbol: str) -> LiquidityMetrics:
    """Analyze liquidity using global aggregator."""
    aggregator = get_liquidity_aggregator()
    if aggregator is None:
        aggregator = LiquidityAggregator()
        set_liquidity_aggregator(aggregator)
    return aggregator.analyze_liquidity(symbol)


def plan_execution(
    symbol: str,
    quantity: int,
    side: str
) -> ExecutionPlan:
    """Plan execution using global aggregator."""
    aggregator = get_liquidity_aggregator()
    if aggregator is None:
        aggregator = LiquidityAggregator()
        set_liquidity_aggregator(aggregator)
    return aggregator.plan_execution(symbol, quantity, side)
