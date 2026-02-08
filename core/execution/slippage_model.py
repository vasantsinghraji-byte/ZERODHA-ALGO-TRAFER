# -*- coding: utf-8 -*-
"""
Slippage Model - Realistic Execution Simulation
================================================
Models market impact and slippage for realistic backtesting.

Models included:
- Fixed: Simple fixed percentage/amount slippage
- Linear: Slippage proportional to order size
- SquareRoot: Almgren-Chriss square root impact model
- VolumeDependent: Slippage based on % of daily volume
- OrderBook: Simulated order book depth model

Example:
    >>> from core.execution import SlippageModel, SquareRootImpact
    >>>
    >>> # Square root market impact
    >>> model = SquareRootImpact(volatility=0.02, daily_volume=1_000_000)
    >>> slippage = model.calculate_slippage(
    ...     price=100.0,
    ...     quantity=10000,
    ...     side='BUY'
    ... )
    >>> print(f"Expected slippage: {slippage.amount:.2f}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import random
import math
from datetime import datetime, time, timezone


class SlippageType(Enum):
    """Type of slippage model."""
    FIXED = "fixed"
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    VOLUME_DEPENDENT = "volume_dependent"
    ORDER_BOOK = "order_book"


class FillType(Enum):
    """How the order was filled."""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class SlippageResult:
    """Result of slippage calculation."""
    # Original order
    original_price: float
    quantity: int
    side: str  # 'BUY' or 'SELL'

    # Slippage components
    spread_cost: float = 0.0        # Bid-ask spread cost
    market_impact: float = 0.0      # Price impact from order size
    timing_cost: float = 0.0        # Cost from execution timing

    # Total slippage
    @property
    def total_slippage(self) -> float:
        """Total slippage in price terms."""
        return self.spread_cost + self.market_impact + self.timing_cost

    @property
    def total_slippage_pct(self) -> float:
        """Total slippage as percentage of price."""
        if self.original_price == 0:
            return 0.0
        return (self.total_slippage / self.original_price) * 100

    @property
    def execution_price(self) -> float:
        """Expected execution price after slippage."""
        if self.side == 'BUY':
            return self.original_price + self.total_slippage
        else:
            return self.original_price - self.total_slippage

    @property
    def total_cost(self) -> float:
        """Total cost impact in rupees."""
        return self.total_slippage * self.quantity


@dataclass
class FillResult:
    """Result of fill simulation."""
    # Order details
    order_price: float
    order_quantity: int
    side: str

    # Fill details
    fill_type: FillType = FillType.FULL
    filled_quantity: int = 0
    average_fill_price: float = 0.0

    # Individual fills (for partial fills)
    fills: List[Tuple[int, float]] = field(default_factory=list)  # (qty, price)

    # Timing
    fill_latency_ms: float = 0.0

    @property
    def unfilled_quantity(self) -> int:
        """Quantity not filled."""
        return self.order_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.order_quantity == 0:
            return 0.0
        return (self.filled_quantity / self.order_quantity) * 100

    @property
    def slippage(self) -> float:
        """Slippage from order price."""
        if self.side == 'BUY':
            return self.average_fill_price - self.order_price
        else:
            return self.order_price - self.average_fill_price

    @property
    def slippage_pct(self) -> float:
        """Slippage as percentage."""
        if self.order_price == 0:
            return 0.0
        return (self.slippage / self.order_price) * 100


@dataclass
class SlippageConfig:
    """Configuration for slippage models."""
    # Spread settings
    default_spread_bps: float = 5.0  # 5 basis points default spread

    # Market impact settings
    impact_coefficient: float = 0.1  # Impact scaling factor

    # Volume settings
    avg_daily_volume: int = 1_000_000
    participation_rate: float = 0.1  # Max 10% of volume

    # Volatility
    daily_volatility: float = 0.02  # 2% daily volatility

    # Fill probability
    fill_probability: float = 0.95
    partial_fill_enabled: bool = True

    # Latency
    min_latency_ms: float = 1.0
    max_latency_ms: float = 50.0

    # Time-of-day effects
    time_of_day_enabled: bool = True


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    def __init__(self, config: Optional[SlippageConfig] = None):
        self.config = config or SlippageConfig()

    @abstractmethod
    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        **kwargs
    ) -> SlippageResult:
        """Calculate expected slippage for an order."""
        pass

    def calculate_spread_cost(self, price: float, side: str) -> float:
        """Calculate bid-ask spread cost (half spread)."""
        spread_pct = self.config.default_spread_bps / 10000
        half_spread = price * spread_pct / 2
        return half_spread

    def get_time_of_day_multiplier(self, current_time: Optional[time] = None) -> float:
        """
        Get slippage multiplier based on time of day.
        Higher slippage at open/close, lower during midday.
        """
        if not self.config.time_of_day_enabled:
            return 1.0

        if current_time is None:
            current_time = datetime.now(tz=timezone.utc).time()

        # Market hours: 9:15 - 15:30
        market_open = time(9, 15)
        market_close = time(15, 30)

        # Pre-market or post-market
        if current_time < market_open or current_time > market_close:
            return 2.0  # Higher slippage outside market hours

        # First 30 minutes - high volatility
        if current_time < time(9, 45):
            return 1.5

        # Last 30 minutes - high volatility
        if current_time > time(15, 0):
            return 1.5

        # Lunch time - lower liquidity
        if time(12, 30) <= current_time <= time(13, 30):
            return 1.2

        # Normal trading hours
        return 1.0


class FixedSlippage(SlippageModel):
    """
    Fixed slippage model.
    Applies a constant slippage regardless of order size.
    """

    def __init__(
        self,
        slippage_pct: float = 0.05,  # 0.05% = 5 bps
        config: Optional[SlippageConfig] = None
    ):
        super().__init__(config)
        self.slippage_pct = slippage_pct

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        **kwargs
    ) -> SlippageResult:
        """Calculate fixed slippage."""
        spread_cost = self.calculate_spread_cost(price, side)
        fixed_impact = price * (self.slippage_pct / 100)

        # Apply time of day multiplier
        current_time = kwargs.get('current_time')
        multiplier = self.get_time_of_day_multiplier(current_time)

        return SlippageResult(
            original_price=price,
            quantity=quantity,
            side=side,
            spread_cost=spread_cost,
            market_impact=fixed_impact * multiplier,
            timing_cost=0.0
        )


class LinearSlippage(SlippageModel):
    """
    Linear slippage model.
    Slippage increases linearly with order size relative to volume.
    """

    def __init__(
        self,
        impact_per_pct_volume: float = 0.1,  # 0.1% slippage per 1% of volume
        config: Optional[SlippageConfig] = None
    ):
        super().__init__(config)
        self.impact_per_pct = impact_per_pct_volume

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        volume: Optional[int] = None,
        **kwargs
    ) -> SlippageResult:
        """Calculate linear slippage based on order size."""
        spread_cost = self.calculate_spread_cost(price, side)

        # Use provided volume or default
        daily_volume = volume or self.config.avg_daily_volume

        # Calculate participation rate
        participation = (quantity / daily_volume) * 100  # as percentage

        # Linear impact
        market_impact = price * (self.impact_per_pct / 100) * participation

        # Apply time of day multiplier
        current_time = kwargs.get('current_time')
        multiplier = self.get_time_of_day_multiplier(current_time)

        return SlippageResult(
            original_price=price,
            quantity=quantity,
            side=side,
            spread_cost=spread_cost,
            market_impact=market_impact * multiplier,
            timing_cost=0.0
        )


class SquareRootImpact(SlippageModel):
    """
    Square Root Market Impact Model (Almgren-Chriss).

    Impact = σ * sqrt(Q / V) * coefficient

    Where:
    - σ = daily volatility
    - Q = order quantity
    - V = daily volume
    - coefficient = impact scaling factor

    This model captures the empirically observed square-root
    relationship between order size and market impact.
    """

    def __init__(
        self,
        volatility: Optional[float] = None,
        daily_volume: Optional[int] = None,
        impact_coefficient: float = 0.1,
        config: Optional[SlippageConfig] = None
    ):
        super().__init__(config)
        self.volatility = volatility or self.config.daily_volatility
        self.daily_volume = daily_volume or self.config.avg_daily_volume
        self.coefficient = impact_coefficient

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        volatility: Optional[float] = None,
        volume: Optional[int] = None,
        **kwargs
    ) -> SlippageResult:
        """Calculate square root market impact."""
        spread_cost = self.calculate_spread_cost(price, side)

        # Use provided or default values
        vol = volatility or self.volatility
        daily_vol = volume or self.daily_volume

        # Prevent division by zero
        if daily_vol == 0:
            daily_vol = 1

        # Square root impact formula
        participation_ratio = quantity / daily_vol
        impact = vol * math.sqrt(participation_ratio) * self.coefficient
        market_impact = price * impact

        # Apply time of day multiplier
        current_time = kwargs.get('current_time')
        multiplier = self.get_time_of_day_multiplier(current_time)

        return SlippageResult(
            original_price=price,
            quantity=quantity,
            side=side,
            spread_cost=spread_cost,
            market_impact=market_impact * multiplier,
            timing_cost=0.0
        )


class VolumeDependentSlippage(SlippageModel):
    """
    Volume-dependent slippage model.

    Different slippage tiers based on participation rate:
    - < 1% of volume: minimal slippage
    - 1-5% of volume: moderate slippage
    - 5-10% of volume: significant slippage
    - > 10% of volume: severe slippage
    """

    def __init__(
        self,
        tier_thresholds: Optional[List[float]] = None,
        tier_impacts: Optional[List[float]] = None,
        config: Optional[SlippageConfig] = None
    ):
        super().__init__(config)

        # Default tiers (participation %)
        self.thresholds = tier_thresholds or [1.0, 5.0, 10.0]
        # Default impacts (slippage %)
        self.impacts = tier_impacts or [0.02, 0.05, 0.1, 0.2]

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        volume: Optional[int] = None,
        **kwargs
    ) -> SlippageResult:
        """Calculate volume-dependent slippage."""
        spread_cost = self.calculate_spread_cost(price, side)

        daily_volume = volume or self.config.avg_daily_volume
        participation_pct = (quantity / daily_volume) * 100 if daily_volume > 0 else 100

        # Find applicable tier
        impact_pct = self.impacts[0]
        for i, threshold in enumerate(self.thresholds):
            if participation_pct > threshold:
                impact_pct = self.impacts[i + 1]

        market_impact = price * (impact_pct / 100)

        # Apply time of day multiplier
        current_time = kwargs.get('current_time')
        multiplier = self.get_time_of_day_multiplier(current_time)

        return SlippageResult(
            original_price=price,
            quantity=quantity,
            side=side,
            spread_cost=spread_cost,
            market_impact=market_impact * multiplier,
            timing_cost=0.0
        )


class OrderBookSlippage(SlippageModel):
    """
    Order book depth simulation model.

    Simulates walking through order book levels to fill large orders.
    Each level has limited quantity at progressively worse prices.
    """

    def __init__(
        self,
        levels: int = 5,
        qty_per_level_pct: float = 2.0,  # Each level has 2% of daily volume
        price_increment_bps: float = 2.0,  # 2 bps between levels
        config: Optional[SlippageConfig] = None
    ):
        super().__init__(config)
        self.levels = levels
        self.qty_per_level_pct = qty_per_level_pct
        self.price_increment_bps = price_increment_bps

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        volume: Optional[int] = None,
        **kwargs
    ) -> SlippageResult:
        """Calculate slippage by simulating order book walk."""
        spread_cost = self.calculate_spread_cost(price, side)

        daily_volume = volume or self.config.avg_daily_volume
        qty_per_level = int(daily_volume * self.qty_per_level_pct / 100)
        price_step = price * (self.price_increment_bps / 10000)

        # Simulate walking through order book
        remaining = quantity
        total_cost = 0.0
        level = 0

        while remaining > 0 and level < self.levels:
            fill_qty = min(remaining, qty_per_level)
            level_price = price + (level * price_step) if side == 'BUY' else price - (level * price_step)
            total_cost += fill_qty * level_price
            remaining -= fill_qty
            level += 1

        # If still remaining, use worst price
        if remaining > 0:
            worst_price = price + (self.levels * price_step) if side == 'BUY' else price - (self.levels * price_step)
            total_cost += remaining * worst_price

        # Calculate average fill price
        avg_price = total_cost / quantity if quantity > 0 else price

        # Market impact is difference from mid price
        market_impact = abs(avg_price - price)

        # Apply time of day multiplier
        current_time = kwargs.get('current_time')
        multiplier = self.get_time_of_day_multiplier(current_time)

        return SlippageResult(
            original_price=price,
            quantity=quantity,
            side=side,
            spread_cost=spread_cost,
            market_impact=market_impact * multiplier,
            timing_cost=0.0
        )


class FillSimulator:
    """
    Simulates realistic order fills.

    Handles:
    - Fill probability based on order type and market conditions
    - Partial fills for large orders
    - Execution latency simulation
    - Price improvement/worsening
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        config: Optional[SlippageConfig] = None
    ):
        self.slippage_model = slippage_model or SquareRootImpact()
        self.config = config or SlippageConfig()

    def simulate_fill(
        self,
        price: float,
        quantity: int,
        side: str,
        order_type: str = 'MARKET',
        limit_price: Optional[float] = None,
        volume: Optional[int] = None,
        **kwargs
    ) -> FillResult:
        """
        Simulate order fill with realistic execution.

        Args:
            price: Current market price
            quantity: Order quantity
            side: 'BUY' or 'SELL'
            order_type: 'MARKET' or 'LIMIT'
            limit_price: Limit price (for limit orders)
            volume: Daily volume for slippage calculation
        """
        result = FillResult(
            order_price=limit_price or price,
            order_quantity=quantity,
            side=side,
            fills=[]
        )

        # Simulate latency
        result.fill_latency_ms = random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        )

        # Check fill probability for limit orders
        if order_type == 'LIMIT' and limit_price:
            fill_prob = self._calculate_limit_fill_probability(
                price, limit_price, side
            )
            if random.random() > fill_prob:
                result.fill_type = FillType.NONE
                return result

        # Calculate slippage
        slippage_result = self.slippage_model.calculate_slippage(
            price=price,
            quantity=quantity,
            side=side,
            volume=volume,
            **kwargs
        )

        # For limit orders, check if fill price is acceptable
        if order_type == 'LIMIT' and limit_price:
            exec_price = slippage_result.execution_price
            if side == 'BUY' and exec_price > limit_price:
                # Would need to pay more than limit
                result.fill_type = FillType.NONE
                return result
            elif side == 'SELL' and exec_price < limit_price:
                # Would receive less than limit
                result.fill_type = FillType.NONE
                return result

        # Simulate partial fills for large orders
        if self.config.partial_fill_enabled:
            fills = self._simulate_partial_fills(
                quantity=quantity,
                base_price=price,
                slippage=slippage_result.total_slippage,
                side=side,
                volume=volume or self.config.avg_daily_volume
            )
            result.fills = fills
            result.filled_quantity = sum(qty for qty, _ in fills)

            if result.filled_quantity < quantity:
                result.fill_type = FillType.PARTIAL
            else:
                result.fill_type = FillType.FULL

            # Calculate average fill price
            total_value = sum(qty * px for qty, px in fills)
            result.average_fill_price = total_value / result.filled_quantity if result.filled_quantity > 0 else 0
        else:
            # Full fill at slippage price
            result.fill_type = FillType.FULL
            result.filled_quantity = quantity
            result.average_fill_price = slippage_result.execution_price
            result.fills = [(quantity, slippage_result.execution_price)]

        return result

    def _calculate_limit_fill_probability(
        self,
        market_price: float,
        limit_price: float,
        side: str
    ) -> float:
        """Calculate probability of limit order getting filled."""
        # Distance from market in basis points
        distance_bps = abs(limit_price - market_price) / market_price * 10000

        if side == 'BUY':
            if limit_price >= market_price:
                # Limit at or above market - high probability
                return min(0.99, 0.9 + distance_bps / 100)
            else:
                # Limit below market - decreasing probability
                return max(0.1, 0.9 - distance_bps / 50)
        else:  # SELL
            if limit_price <= market_price:
                # Limit at or below market - high probability
                return min(0.99, 0.9 + distance_bps / 100)
            else:
                # Limit above market - decreasing probability
                return max(0.1, 0.9 - distance_bps / 50)

    def _simulate_partial_fills(
        self,
        quantity: int,
        base_price: float,
        slippage: float,
        side: str,
        volume: int
    ) -> List[Tuple[int, float]]:
        """Simulate partial fills for large orders."""
        fills = []
        remaining = quantity

        # Determine if order is "large" (>5% of volume)
        participation = quantity / volume if volume > 0 else 1.0

        if participation <= 0.05:
            # Small order - single fill
            exec_price = base_price + slippage if side == 'BUY' else base_price - slippage
            return [(quantity, exec_price)]

        # Large order - multiple partial fills
        num_fills = min(10, max(2, int(participation * 20)))
        avg_fill_size = quantity // num_fills

        cumulative_qty = 0
        for i in range(num_fills):
            # Randomize fill size slightly
            if i == num_fills - 1:
                fill_qty = remaining
            else:
                variation = random.uniform(0.8, 1.2)
                fill_qty = min(int(avg_fill_size * variation), remaining)

            if fill_qty <= 0:
                break

            # Progressive slippage - later fills get worse prices
            progress = cumulative_qty / quantity
            progressive_slippage = slippage * (1 + progress * 0.5)

            # Add some randomness
            random_factor = random.uniform(0.9, 1.1)
            final_slippage = progressive_slippage * random_factor

            exec_price = base_price + final_slippage if side == 'BUY' else base_price - final_slippage

            fills.append((fill_qty, round(exec_price, 2)))
            cumulative_qty += fill_qty
            remaining -= fill_qty

        return fills


# Convenience functions
def calculate_slippage(
    price: float,
    quantity: int,
    side: str,
    model_type: SlippageType = SlippageType.SQUARE_ROOT,
    **kwargs
) -> SlippageResult:
    """
    Calculate slippage using specified model.

    Args:
        price: Current market price
        quantity: Order quantity
        side: 'BUY' or 'SELL'
        model_type: Type of slippage model to use
        **kwargs: Additional parameters for the model

    Returns:
        SlippageResult with slippage breakdown
    """
    model_map = {
        SlippageType.FIXED: FixedSlippage,
        SlippageType.LINEAR: LinearSlippage,
        SlippageType.SQUARE_ROOT: SquareRootImpact,
        SlippageType.VOLUME_DEPENDENT: VolumeDependentSlippage,
        SlippageType.ORDER_BOOK: OrderBookSlippage,
    }

    model_class = model_map.get(model_type, SquareRootImpact)
    model = model_class()

    return model.calculate_slippage(price, quantity, side, **kwargs)


def simulate_fill(
    price: float,
    quantity: int,
    side: str,
    order_type: str = 'MARKET',
    limit_price: Optional[float] = None,
    **kwargs
) -> FillResult:
    """
    Simulate order fill with realistic execution.

    Args:
        price: Current market price
        quantity: Order quantity
        side: 'BUY' or 'SELL'
        order_type: 'MARKET' or 'LIMIT'
        limit_price: Limit price for limit orders
        **kwargs: Additional parameters

    Returns:
        FillResult with fill details
    """
    simulator = FillSimulator()
    return simulator.simulate_fill(
        price=price,
        quantity=quantity,
        side=side,
        order_type=order_type,
        limit_price=limit_price,
        **kwargs
    )


def estimate_execution_cost(
    price: float,
    quantity: int,
    side: str,
    volume: int,
    volatility: float = 0.02
) -> dict:
    """
    Estimate total execution cost breakdown.

    Returns dict with:
    - spread_cost: Bid-ask spread cost
    - market_impact: Price impact from order size
    - total_cost: Total execution cost
    - cost_bps: Cost in basis points
    """
    model = SquareRootImpact(volatility=volatility, daily_volume=volume)
    result = model.calculate_slippage(price, quantity, side, volume=volume)

    return {
        'spread_cost': result.spread_cost * quantity,
        'market_impact': result.market_impact * quantity,
        'total_cost': result.total_cost,
        'cost_bps': result.total_slippage_pct * 100,
        'execution_price': result.execution_price,
    }
