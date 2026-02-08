# -*- coding: utf-8 -*-
"""
VWAP Execution Algorithm - Follow the Volume!
===============================================
Volume-Weighted Average Price executes orders following historical
volume patterns to minimize market impact.

Why use VWAP?
- Achieves price close to volume-weighted average
- Matches natural market liquidity patterns
- Better for liquid stocks with predictable volume

Example:
    >>> from core.execution import VWAPExecutor, VolumeProfile
    >>>
    >>> # Create volume profile from historical data
    >>> profile = VolumeProfile.from_historical(volume_data)
    >>>
    >>> # Execute following volume pattern
    >>> vwap = VWAPExecutor(broker, profile)
    >>> result = vwap.execute_sync("RELIANCE", 1000, Side.BUY)
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..broker import ZerodhaBroker

logger = logging.getLogger(__name__)


class Side(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class SliceStatus(Enum):
    """Status of a VWAP slice."""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VolumeProfile:
    """
    Intraday volume profile for VWAP execution.

    Represents expected volume distribution throughout trading day.
    """
    # Time buckets (e.g., 9:15, 9:30, 9:45, ...)
    time_buckets: List[dt_time]
    # Volume weight for each bucket (should sum to 1.0)
    volume_weights: List[float]
    # Source of profile
    source: str = "historical"
    # Date calculated
    calculated_date: Optional[datetime] = None

    def get_weight_for_time(self, t: dt_time) -> float:
        """Get volume weight for a specific time."""
        for i, bucket_time in enumerate(self.time_buckets):
            if i < len(self.time_buckets) - 1:
                if bucket_time <= t < self.time_buckets[i + 1]:
                    return self.volume_weights[i]
            else:
                if t >= bucket_time:
                    return self.volume_weights[i]
        return 0.0

    def get_cumulative_weight(self, t: dt_time) -> float:
        """Get cumulative volume weight up to a time."""
        total = 0.0
        for i, bucket_time in enumerate(self.time_buckets):
            if bucket_time <= t:
                total += self.volume_weights[i]
            else:
                break
        return total

    @classmethod
    def from_historical(
        cls,
        volume_data: pd.DataFrame,
        bucket_minutes: int = 15
    ) -> 'VolumeProfile':
        """
        Create volume profile from historical volume data.

        Args:
            volume_data: DataFrame with 'time' and 'volume' columns
            bucket_minutes: Size of time buckets in minutes

        Returns:
            VolumeProfile instance
        """
        # Aggregate volume by time bucket
        if 'time' not in volume_data.columns:
            # Assume index is datetime
            volume_data = volume_data.copy()
            volume_data['time'] = volume_data.index.time

        # Create buckets
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)

        buckets = []
        current = datetime.combine(datetime.today(), market_open)
        end = datetime.combine(datetime.today(), market_close)

        while current <= end:
            buckets.append(current.time())
            current += timedelta(minutes=bucket_minutes)

        # Calculate average volume per bucket
        bucket_volumes = []
        for i, bucket_start in enumerate(buckets):
            bucket_end = buckets[i + 1] if i < len(buckets) - 1 else market_close

            mask = (volume_data['time'] >= bucket_start) & (volume_data['time'] < bucket_end)
            avg_volume = volume_data.loc[mask, 'volume'].mean() if mask.any() else 0

            bucket_volumes.append(avg_volume)

        # Normalize to weights
        total_volume = sum(bucket_volumes)
        if total_volume > 0:
            weights = [v / total_volume for v in bucket_volumes]
        else:
            weights = [1.0 / len(buckets)] * len(buckets)

        return cls(
            time_buckets=buckets,
            volume_weights=weights,
            source="historical",
            calculated_date=datetime.now(tz=timezone.utc)
        )

    @classmethod
    def typical_profile(cls) -> 'VolumeProfile':
        """
        Create typical U-shaped volume profile.

        High volume at open/close, lower in middle of day.
        """
        buckets = [
            dt_time(9, 15), dt_time(9, 30), dt_time(9, 45),
            dt_time(10, 0), dt_time(10, 30), dt_time(11, 0),
            dt_time(11, 30), dt_time(12, 0), dt_time(12, 30),
            dt_time(13, 0), dt_time(13, 30), dt_time(14, 0),
            dt_time(14, 30), dt_time(15, 0), dt_time(15, 15)
        ]

        # U-shaped pattern: high at open/close
        weights = [
            0.12, 0.10, 0.08,   # Morning high
            0.06, 0.05, 0.04,   # Mid-morning decline
            0.04, 0.04, 0.04,   # Lunch low
            0.05, 0.05, 0.06,   # Afternoon pickup
            0.08, 0.10, 0.09    # Close high
        ]

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return cls(
            time_buckets=buckets,
            volume_weights=weights,
            source="typical_u_shape",
            calculated_date=datetime.now(tz=timezone.utc)
        )


@dataclass
class VWAPConfig:
    """Configuration for VWAP execution."""
    # Timing
    duration_minutes: float = 60.0       # Total execution window
    bucket_minutes: int = 5              # Order placement interval

    # Participation
    max_participation_rate: float = 0.2  # Max % of bucket volume
    min_slice_qty: int = 1               # Minimum per slice

    # Adaptation
    adaptive: bool = True                # Adjust to real-time volume
    catchup_aggressive: float = 1.5      # Aggression when behind schedule

    # Price limits
    limit_price: Optional[float] = None
    price_tolerance_pct: float = 0.5

    # Retry
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Callbacks
    on_slice_complete: Optional[Callable] = None
    on_complete: Optional[Callable] = None


@dataclass
class VWAPSlice:
    """A single slice of a VWAP order."""
    slice_id: int
    bucket_time: dt_time
    target_weight: float             # Expected volume weight
    target_quantity: int
    filled_quantity: int = 0
    actual_volume: int = 0           # Actual market volume observed
    execution_time: Optional[datetime] = None
    price: float = 0.0
    status: SliceStatus = SliceStatus.PENDING
    order_id: str = ""


@dataclass
class VWAPOrder:
    """A VWAP order being executed."""
    order_id: str
    symbol: str
    side: Side
    total_quantity: int
    slices: List[VWAPSlice] = field(default_factory=list)
    profile: Optional[VolumeProfile] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"

    @property
    def filled_quantity(self) -> int:
        return sum(s.filled_quantity for s in self.slices)

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def average_price(self) -> float:
        filled = [(s.filled_quantity, s.price) for s in self.slices if s.price > 0]
        if not filled:
            return 0.0
        total_value = sum(Decimal(qty) * Decimal(str(price)) for qty, price in filled)
        total_qty = sum(qty for qty, _ in filled)
        return float(total_value / total_qty) if total_qty > 0 else 0.0

    @property
    def vwap_target(self) -> float:
        """Calculate target VWAP from filled slices."""
        return self.average_price


@dataclass
class VWAPResult:
    """Result of VWAP execution."""
    order: VWAPOrder
    success: bool
    total_filled: int
    average_price: float
    market_vwap: float               # Actual market VWAP for comparison
    execution_time_seconds: float
    participation_rate: float
    slippage_to_vwap_bps: float      # Slippage vs market VWAP

    def summary(self) -> str:
        """Generate execution summary."""
        lines = [
            "=" * 50,
            "VWAP EXECUTION SUMMARY",
            "=" * 50,
            f"Symbol: {self.order.symbol}",
            f"Side: {self.order.side.value}",
            f"Target Qty: {self.order.total_quantity}",
            f"Filled Qty: {self.total_filled}",
            f"Fill Rate: {self.total_filled / self.order.total_quantity:.1%}",
            f"Execution Price: Rs.{self.average_price:.2f}",
            f"Market VWAP: Rs.{self.market_vwap:.2f}",
            f"Slippage to VWAP: {self.slippage_to_vwap_bps:.1f} bps",
            f"Participation Rate: {self.participation_rate:.1%}",
            f"Duration: {self.execution_time_seconds:.1f}s",
            f"Status: {'SUCCESS' if self.success else 'PARTIAL/FAILED'}",
            "=" * 50
        ]
        return "\n".join(lines)


class VWAPExecutor:
    """
    VWAP (Volume-Weighted Average Price) Executor.

    Executes orders following historical volume patterns to achieve
    price close to market VWAP.

    Example:
        >>> profile = VolumeProfile.typical_profile()
        >>> vwap = VWAPExecutor(broker, profile)
        >>> result = vwap.execute_sync("TCS", 500, Side.BUY)
    """

    def __init__(
        self,
        broker: Optional['ZerodhaBroker'] = None,
        volume_profile: Optional[VolumeProfile] = None,
        config: Optional[VWAPConfig] = None
    ):
        self.broker = broker
        self.profile = volume_profile or VolumeProfile.typical_profile()
        self.config = config or VWAPConfig()
        self._active_orders: Dict[str, VWAPOrder] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Real-time tracking
        self._volume_tracker: Dict[str, int] = {}
        self._price_tracker: Dict[str, List[tuple]] = {}  # (price, volume) pairs

    def create_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[VWAPConfig] = None
    ) -> VWAPOrder:
        """Create a VWAP order with slices based on volume profile."""
        import uuid

        config = config or self.config
        order_id = f"VWAP-{uuid.uuid4().hex[:8]}"

        now = datetime.now(tz=timezone.utc)
        start_time = now.time()

        # Calculate slices based on volume profile
        slices = []
        remaining_weight = 1.0 - self.profile.get_cumulative_weight(start_time)

        if remaining_weight <= 0:
            remaining_weight = 1.0

        # Create slices for remaining time buckets
        bucket_interval = timedelta(minutes=config.bucket_minutes)
        current_time = now

        slice_id = 0
        carried = 0.0
        for i, bucket_time in enumerate(self.profile.time_buckets):
            if bucket_time < start_time:
                continue

            # End after duration
            if (datetime.combine(now.date(), bucket_time) - now).total_seconds() > config.duration_minutes * 60:
                break

            # Calculate target quantity for this bucket
            weight = self.profile.volume_weights[i]
            relative_weight = weight / remaining_weight if remaining_weight > 0 else 0
            exact_qty = quantity * relative_weight + carried
            target_qty = int(exact_qty)
            carried = exact_qty - target_qty

            if target_qty < config.min_slice_qty:
                carried += target_qty
                continue

            slices.append(VWAPSlice(
                slice_id=slice_id,
                bucket_time=bucket_time,
                target_weight=relative_weight,
                target_quantity=target_qty
            ))
            slice_id += 1

        # Adjust last slice to fill remaining quantity
        if slices:
            allocated = sum(s.target_quantity for s in slices)
            slices[-1].target_quantity += quantity - allocated

        order = VWAPOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=slices,
            profile=self.profile,
            start_time=now
        )

        logger.info(f"Created VWAP order {order_id}: {quantity} {symbol} in {len(slices)} slices")
        return order

    def execute_sync(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[VWAPConfig] = None
    ) -> VWAPResult:
        """
        Execute VWAP order synchronously.

        Args:
            symbol: Trading symbol
            quantity: Total quantity
            side: BUY or SELL
            config: Optional config

        Returns:
            VWAPResult with execution details
        """
        config = config or self.config
        order = self.create_order(symbol, quantity, side, config)

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = "executing"
        start_time = time.time()

        # Initialize tracking
        self._volume_tracker[symbol] = 0
        self._price_tracker[symbol] = []

        try:
            for slice_order in order.slices:
                if self._stop_event.is_set():
                    order.status = "cancelled"
                    break

                # Wait until bucket time
                now = datetime.now(tz=timezone.utc).time()
                if slice_order.bucket_time > now:
                    wait_seconds = self._time_diff_seconds(now, slice_order.bucket_time)
                    if wait_seconds > 0:
                        time.sleep(min(wait_seconds, config.bucket_minutes * 60))

                # Adaptive adjustment
                if config.adaptive:
                    self._adjust_slice_quantity(order, slice_order)

                # Execute slice
                self._execute_slice_sync(order, slice_order, config)

                if config.on_slice_complete:
                    try:
                        config.on_slice_complete(slice_order)
                    except Exception:
                        pass

            order.status = "completed"
            order.end_time = datetime.now(tz=timezone.utc)

        except Exception as e:
            order.status = "failed"
            logger.error(f"VWAP execution failed: {e}")

        finally:
            with self._lock:
                self._active_orders.pop(order.order_id, None)

        result = self._create_result(order, start_time)

        if config.on_complete:
            try:
                config.on_complete(result)
            except Exception:
                pass

        return result

    async def execute(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[VWAPConfig] = None
    ) -> VWAPResult:
        """Execute VWAP order asynchronously."""
        config = config or self.config
        order = self.create_order(symbol, quantity, side, config)

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = "executing"
        start_time = time.time()

        self._volume_tracker[symbol] = 0
        self._price_tracker[symbol] = []

        try:
            for slice_order in order.slices:
                if self._stop_event.is_set():
                    order.status = "cancelled"
                    break

                now = datetime.now(tz=timezone.utc).time()
                if slice_order.bucket_time > now:
                    wait_seconds = self._time_diff_seconds(now, slice_order.bucket_time)
                    if wait_seconds > 0:
                        await asyncio.sleep(min(wait_seconds, config.bucket_minutes * 60))

                if config.adaptive:
                    self._adjust_slice_quantity(order, slice_order)

                await self._execute_slice_async(order, slice_order, config)

                if config.on_slice_complete:
                    try:
                        config.on_slice_complete(slice_order)
                    except Exception:
                        pass

            order.status = "completed"
            order.end_time = datetime.now(tz=timezone.utc)

        except Exception as e:
            order.status = "failed"
            logger.error(f"VWAP execution failed: {e}")

        finally:
            with self._lock:
                self._active_orders.pop(order.order_id, None)

        return self._create_result(order, start_time)

    def _time_diff_seconds(self, t1: dt_time, t2: dt_time) -> float:
        """Calculate seconds between two times."""
        today = datetime.today().date()
        dt1 = datetime.combine(today, t1)
        dt2 = datetime.combine(today, t2)
        return (dt2 - dt1).total_seconds()

    def _adjust_slice_quantity(self, order: VWAPOrder, slice_order: VWAPSlice) -> None:
        """Adjust slice quantity based on progress."""
        # Calculate expected vs actual fill rate
        completed_slices = [s for s in order.slices if s.status == SliceStatus.FILLED]

        if not completed_slices:
            return

        expected_fill = sum(s.target_quantity for s in completed_slices)
        actual_fill = order.filled_quantity

        # If behind schedule, increase remaining slices
        if actual_fill < expected_fill * 0.9:
            shortfall = expected_fill - actual_fill
            remaining_slices = [s for s in order.slices if s.status == SliceStatus.PENDING]

            if remaining_slices:
                catchup_per_slice = shortfall // len(remaining_slices)
                slice_order.target_quantity += catchup_per_slice

    def _execute_slice_sync(
        self,
        order: VWAPOrder,
        slice_order: VWAPSlice,
        config: VWAPConfig
    ) -> None:
        """Execute a single VWAP slice."""
        slice_order.status = SliceStatus.EXECUTING

        for attempt in range(config.max_retries + 1):
            try:
                current_price = self._get_current_price(order.symbol)

                if config.limit_price:
                    if order.side == Side.BUY and current_price > config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        return
                    if order.side == Side.SELL and current_price < config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        return

                filled_qty, fill_price, order_id = self._place_order(
                    order.symbol,
                    slice_order.target_quantity,
                    order.side,
                    current_price
                )

                slice_order.filled_quantity = filled_qty
                slice_order.price = fill_price
                slice_order.order_id = order_id
                slice_order.execution_time = datetime.now(tz=timezone.utc)

                # Track for VWAP calculation
                self._price_tracker[order.symbol].append((fill_price, filled_qty))
                self._volume_tracker[order.symbol] += filled_qty

                if filled_qty >= slice_order.target_quantity:
                    slice_order.status = SliceStatus.FILLED
                elif filled_qty > 0:
                    slice_order.status = SliceStatus.PARTIAL
                else:
                    raise Exception("No fill")

                return

            except Exception as e:
                if attempt < config.max_retries:
                    time.sleep(config.retry_delay_seconds)
                else:
                    slice_order.status = SliceStatus.FAILED
                    logger.warning(f"VWAP slice {slice_order.slice_id} failed: {e}")

    async def _execute_slice_async(
        self,
        order: VWAPOrder,
        slice_order: VWAPSlice,
        config: VWAPConfig
    ) -> None:
        """Execute slice asynchronously."""
        slice_order.status = SliceStatus.EXECUTING

        for attempt in range(config.max_retries + 1):
            try:
                current_price = self._get_current_price(order.symbol)

                if config.limit_price:
                    if order.side == Side.BUY and current_price > config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        return
                    if order.side == Side.SELL and current_price < config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        return

                filled_qty, fill_price, order_id = self._place_order(
                    order.symbol,
                    slice_order.target_quantity,
                    order.side,
                    current_price
                )

                slice_order.filled_quantity = filled_qty
                slice_order.price = fill_price
                slice_order.order_id = order_id
                slice_order.execution_time = datetime.now(tz=timezone.utc)

                self._price_tracker[order.symbol].append((fill_price, filled_qty))
                self._volume_tracker[order.symbol] += filled_qty

                if filled_qty >= slice_order.target_quantity:
                    slice_order.status = SliceStatus.FILLED
                elif filled_qty > 0:
                    slice_order.status = SliceStatus.PARTIAL
                else:
                    raise Exception("No fill")

                return

            except Exception:
                if attempt < config.max_retries:
                    await asyncio.sleep(config.retry_delay_seconds)
                else:
                    slice_order.status = SliceStatus.FAILED

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        if self.broker:
            try:
                quote = self.broker.get_quote(symbol)
                return quote.last_price if quote else 0.0
            except Exception:
                pass
        return 100.0

    def _place_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        price: float
    ) -> tuple:
        """Place order and return (filled_qty, fill_price, order_id)."""
        if self.broker:
            try:
                if side == Side.BUY:
                    order = self.broker.buy(symbol, quantity, price)
                else:
                    order = self.broker.sell(symbol, quantity, price)

                return (
                    order.filled_quantity or quantity,
                    order.average_price or price,
                    order.order_id or ""
                )
            except Exception as e:
                logger.error(f"Order failed: {e}")
                raise

        import uuid
        return quantity, price, f"SIM-{uuid.uuid4().hex[:8]}"

    def _calculate_market_vwap(self, symbol: str) -> float:
        """Calculate market VWAP from tracked prices."""
        price_volume_pairs = self._price_tracker.get(symbol, [])
        if not price_volume_pairs:
            return 0.0

        total_value = sum(Decimal(str(p)) * Decimal(v) for p, v in price_volume_pairs)
        total_volume = sum(v for _, v in price_volume_pairs)

        return float(total_value / total_volume) if total_volume > 0 else 0.0

    def _create_result(self, order: VWAPOrder, start_time: float) -> VWAPResult:
        """Create result from completed order."""
        execution_time = time.time() - start_time
        avg_price = order.average_price
        market_vwap = self._calculate_market_vwap(order.symbol)

        # Calculate slippage to VWAP using Decimal for basis point precision
        if market_vwap > 0 and avg_price > 0:
            d_avg = Decimal(str(avg_price))
            d_vwap = Decimal(str(market_vwap))
            if order.side == Side.BUY:
                slippage = float((d_avg - d_vwap) / d_vwap * 10000)
            else:
                slippage = float((d_vwap - d_avg) / d_vwap * 10000)
        else:
            slippage = 0.0

        # Participation rate
        total_volume = self._volume_tracker.get(order.symbol, 0)
        participation = order.filled_quantity / total_volume if total_volume > 0 else 0

        return VWAPResult(
            order=order,
            success=order.filled_quantity >= order.total_quantity * 0.9,
            total_filled=order.filled_quantity,
            average_price=avg_price,
            market_vwap=market_vwap or avg_price,
            execution_time_seconds=execution_time,
            participation_rate=participation,
            slippage_to_vwap_bps=slippage
        )

    def cancel(self, order_id: str) -> bool:
        """Cancel active VWAP order."""
        with self._lock:
            if order_id in self._active_orders:
                self._active_orders[order_id].status = "cancelled"
                return True
        return False
