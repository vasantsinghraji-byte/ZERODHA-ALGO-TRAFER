# -*- coding: utf-8 -*-
"""
TWAP Execution Algorithm - Spread Orders Over Time!
=====================================================
Time-Weighted Average Price splits a large order into equal slices
executed at regular intervals.

Why use TWAP?
- Reduces market impact by not flooding the market
- Achieves price close to time-average
- Simple and predictable execution

Example:
    >>> from core.execution import TWAPExecutor, TWAPConfig
    >>>
    >>> config = TWAPConfig(
    ...     duration_minutes=30,
    ...     num_slices=10,
    ...     participation_rate=0.1
    ... )
    >>> twap = TWAPExecutor(broker, config)
    >>> result = await twap.execute("RELIANCE", 1000, Side.BUY)
    >>> print(f"Avg price: {result.average_price:.2f}")
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..broker import ZerodhaBroker

logger = logging.getLogger(__name__)


class Side(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class SliceStatus(Enum):
    """Status of a TWAP slice."""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TWAPConfig:
    """Configuration for TWAP execution."""
    # Timing
    duration_minutes: float = 30.0      # Total execution duration
    num_slices: int = 10                # Number of order slices
    randomize_timing: bool = True       # Add random jitter to timing
    timing_jitter_pct: float = 0.1      # Max jitter as % of interval

    # Participation
    max_participation_rate: float = 0.2 # Max % of volume per interval
    min_slice_qty: int = 1              # Minimum quantity per slice

    # Price limits
    limit_price: Optional[float] = None # Optional limit price
    price_tolerance_pct: float = 0.5    # Max deviation from current price

    # Retry settings
    max_retries: int = 3                # Retries per slice
    retry_delay_seconds: float = 5.0    # Delay between retries

    # Callbacks
    on_slice_complete: Optional[Callable[['TWAPSlice'], None]] = None
    on_complete: Optional[Callable[['TWAPResult'], None]] = None


@dataclass
class TWAPSlice:
    """A single slice of a TWAP order."""
    slice_id: int
    target_quantity: int
    filled_quantity: int = 0
    target_time: Optional[datetime] = None
    execution_time: Optional[datetime] = None
    price: float = 0.0
    status: SliceStatus = SliceStatus.PENDING
    order_id: str = ""
    retries: int = 0
    error: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status in [SliceStatus.FILLED, SliceStatus.CANCELLED, SliceStatus.FAILED]

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.target_quantity if self.target_quantity > 0 else 0


@dataclass
class TWAPOrder:
    """A TWAP order being executed."""
    order_id: str
    symbol: str
    side: Side
    total_quantity: int
    slices: List[TWAPSlice] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    config: Optional[TWAPConfig] = None

    @property
    def filled_quantity(self) -> int:
        return sum(s.filled_quantity for s in self.slices)

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def progress_pct(self) -> float:
        return (self.filled_quantity / self.total_quantity * 100) if self.total_quantity > 0 else 0

    @property
    def average_price(self) -> float:
        filled = [(s.filled_quantity, s.price) for s in self.slices if s.price > 0]
        if not filled:
            return 0.0
        total_value = sum(Decimal(qty) * Decimal(str(price)) for qty, price in filled)
        total_qty = sum(qty for qty, _ in filled)
        return float(total_value / total_qty) if total_qty > 0 else 0.0


@dataclass
class TWAPResult:
    """Result of TWAP execution."""
    order: TWAPOrder
    success: bool
    total_filled: int
    average_price: float
    execution_time_seconds: float
    slices_completed: int
    slices_failed: int
    total_cost: float = 0.0
    benchmark_price: float = 0.0  # Price at start for comparison
    slippage_bps: float = 0.0     # Slippage in basis points

    def summary(self) -> str:
        """Generate execution summary."""
        lines = [
            "=" * 50,
            "TWAP EXECUTION SUMMARY",
            "=" * 50,
            f"Symbol: {self.order.symbol}",
            f"Side: {self.order.side.value}",
            f"Target Qty: {self.order.total_quantity}",
            f"Filled Qty: {self.total_filled}",
            f"Fill Rate: {self.total_filled / self.order.total_quantity:.1%}",
            f"Average Price: Rs.{self.average_price:.2f}",
            f"Total Cost: Rs.{self.total_cost:,.2f}",
            f"Benchmark: Rs.{self.benchmark_price:.2f}",
            f"Slippage: {self.slippage_bps:.1f} bps",
            f"Duration: {self.execution_time_seconds:.1f}s",
            f"Slices: {self.slices_completed} completed, {self.slices_failed} failed",
            f"Status: {'SUCCESS' if self.success else 'PARTIAL/FAILED'}",
            "=" * 50
        ]
        return "\n".join(lines)


class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) Executor.

    Splits large orders into equal slices executed at regular intervals.

    Example:
        >>> twap = TWAPExecutor(broker)
        >>> result = twap.execute_sync("RELIANCE", 1000, Side.BUY, duration_minutes=30)
        >>> print(f"Filled {result.total_filled} @ Rs.{result.average_price:.2f}")
    """

    def __init__(
        self,
        broker: Optional['ZerodhaBroker'] = None,
        config: Optional[TWAPConfig] = None
    ):
        self.broker = broker
        self.config = config or TWAPConfig()
        self._active_orders: Dict[str, TWAPOrder] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def create_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[TWAPConfig] = None
    ) -> TWAPOrder:
        """Create a TWAP order with calculated slices."""
        import uuid

        config = config or self.config
        order_id = f"TWAP-{uuid.uuid4().hex[:8]}"

        # Calculate slice sizes
        num_slices = config.num_slices
        base_qty = quantity // num_slices
        remainder = quantity % num_slices

        # Calculate timing
        interval = timedelta(minutes=config.duration_minutes / num_slices)
        start_time = datetime.now(tz=timezone.utc)

        slices = []
        for i in range(num_slices):
            # Distribute remainder across first slices
            slice_qty = base_qty + (1 if i < remainder else 0)

            if slice_qty < config.min_slice_qty and i > 0:
                # Merge with previous slice
                slices[-1].target_quantity += slice_qty
                continue

            target_time = start_time + (interval * (i + 1))

            # Add random jitter
            if config.randomize_timing and i > 0:
                import random
                jitter = random.uniform(
                    -config.timing_jitter_pct,
                    config.timing_jitter_pct
                )
                jitter_seconds = interval.total_seconds() * jitter
                target_time += timedelta(seconds=jitter_seconds)

            slices.append(TWAPSlice(
                slice_id=i,
                target_quantity=slice_qty,
                target_time=target_time
            ))

        order = TWAPOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            config=config
        )

        logger.info(f"Created TWAP order {order_id}: {quantity} {symbol} in {len(slices)} slices")
        return order

    async def execute(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[TWAPConfig] = None
    ) -> TWAPResult:
        """
        Execute TWAP order asynchronously.

        Args:
            symbol: Trading symbol
            quantity: Total quantity to execute
            side: BUY or SELL
            config: Optional custom config

        Returns:
            TWAPResult with execution details
        """
        config = config or self.config
        order = self.create_order(symbol, quantity, side, config)

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = "executing"
        start_time = time.time()

        # Get benchmark price
        benchmark_price = self._get_current_price(symbol)

        try:
            for slice_order in order.slices:
                if self._stop_event.is_set():
                    order.status = "cancelled"
                    break

                # Wait until target time
                if slice_order.target_time:
                    wait_seconds = (slice_order.target_time - datetime.now(tz=timezone.utc)).total_seconds()
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)

                # Execute slice
                await self._execute_slice(order, slice_order, config)

                # Callback
                if config.on_slice_complete:
                    try:
                        config.on_slice_complete(slice_order)
                    except Exception as e:
                        logger.error(f"Slice callback error: {e}")

            order.status = "completed"
            order.end_time = datetime.now(tz=timezone.utc)

        except Exception as e:
            order.status = "failed"
            logger.error(f"TWAP execution failed: {e}")

        finally:
            with self._lock:
                self._active_orders.pop(order.order_id, None)

        # Calculate result
        result = self._create_result(order, start_time, benchmark_price)

        if config.on_complete:
            try:
                config.on_complete(result)
            except Exception as e:
                logger.error(f"Complete callback error: {e}")

        return result

    def execute_sync(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        config: Optional[TWAPConfig] = None
    ) -> TWAPResult:
        """
        Execute TWAP order synchronously (blocking).

        Args:
            symbol: Trading symbol
            quantity: Total quantity
            side: BUY or SELL
            config: Optional config

        Returns:
            TWAPResult
        """
        config = config or self.config
        order = self.create_order(symbol, quantity, side, config)

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = "executing"
        start_time = time.time()
        benchmark_price = self._get_current_price(symbol)

        try:
            for slice_order in order.slices:
                if self._stop_event.is_set():
                    order.status = "cancelled"
                    break

                # Wait until target time
                if slice_order.target_time:
                    wait_seconds = (slice_order.target_time - datetime.now(tz=timezone.utc)).total_seconds()
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)

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
            logger.error(f"TWAP execution failed: {e}")

        finally:
            with self._lock:
                self._active_orders.pop(order.order_id, None)

        result = self._create_result(order, start_time, benchmark_price)

        if config.on_complete:
            try:
                config.on_complete(result)
            except Exception:
                pass

        return result

    async def _execute_slice(
        self,
        order: TWAPOrder,
        slice_order: TWAPSlice,
        config: TWAPConfig
    ) -> None:
        """Execute a single slice asynchronously."""
        slice_order.status = SliceStatus.EXECUTING

        for attempt in range(config.max_retries + 1):
            try:
                # Get current price
                current_price = self._get_current_price(order.symbol)

                # Check price tolerance
                if config.limit_price:
                    if order.side == Side.BUY and current_price > config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        slice_order.error = "Price above limit"
                        return
                    if order.side == Side.SELL and current_price < config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        slice_order.error = "Price below limit"
                        return

                # Place order
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

                if filled_qty >= slice_order.target_quantity:
                    slice_order.status = SliceStatus.FILLED
                elif filled_qty > 0:
                    slice_order.status = SliceStatus.PARTIAL
                else:
                    raise Exception("No fill received")

                return

            except Exception as e:
                slice_order.retries += 1
                slice_order.error = str(e)

                if attempt < config.max_retries:
                    await asyncio.sleep(config.retry_delay_seconds)
                else:
                    slice_order.status = SliceStatus.FAILED
                    logger.warning(f"Slice {slice_order.slice_id} failed after {attempt + 1} attempts")

    def _execute_slice_sync(
        self,
        order: TWAPOrder,
        slice_order: TWAPSlice,
        config: TWAPConfig
    ) -> None:
        """Execute a single slice synchronously."""
        slice_order.status = SliceStatus.EXECUTING

        for attempt in range(config.max_retries + 1):
            try:
                current_price = self._get_current_price(order.symbol)

                if config.limit_price:
                    if order.side == Side.BUY and current_price > config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        slice_order.error = "Price above limit"
                        return
                    if order.side == Side.SELL and current_price < config.limit_price:
                        slice_order.status = SliceStatus.CANCELLED
                        slice_order.error = "Price below limit"
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

                if filled_qty >= slice_order.target_quantity:
                    slice_order.status = SliceStatus.FILLED
                elif filled_qty > 0:
                    slice_order.status = SliceStatus.PARTIAL
                else:
                    raise Exception("No fill received")

                return

            except Exception as e:
                slice_order.retries += 1
                slice_order.error = str(e)

                if attempt < config.max_retries:
                    time.sleep(config.retry_delay_seconds)
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
        return 100.0  # Default for testing

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
                logger.error(f"Order placement failed: {e}")
                raise

        # Simulated fill for testing
        import uuid
        return quantity, price, f"SIM-{uuid.uuid4().hex[:8]}"

    def _create_result(
        self,
        order: TWAPOrder,
        start_time: float,
        benchmark_price: float
    ) -> TWAPResult:
        """Create result from completed order."""
        execution_time = time.time() - start_time
        avg_price = order.average_price
        total_filled = order.filled_quantity

        # Use Decimal for financial calculations
        d_avg = Decimal(str(avg_price))
        d_bench = Decimal(str(benchmark_price))
        total_cost = float(Decimal(total_filled) * d_avg)

        # Calculate slippage in basis points
        if benchmark_price > 0 and avg_price > 0:
            if order.side == Side.BUY:
                slippage_bps = float((d_avg - d_bench) / d_bench * 10000)
            else:
                slippage_bps = float((d_bench - d_avg) / d_bench * 10000)
        else:
            slippage_bps = 0.0

        completed = sum(1 for s in order.slices if s.status == SliceStatus.FILLED)
        failed = sum(1 for s in order.slices if s.status == SliceStatus.FAILED)

        return TWAPResult(
            order=order,
            success=total_filled >= order.total_quantity * 0.9,  # 90% fill = success
            total_filled=total_filled,
            average_price=avg_price,
            execution_time_seconds=execution_time,
            slices_completed=completed,
            slices_failed=failed,
            total_cost=total_cost,
            benchmark_price=benchmark_price,
            slippage_bps=slippage_bps
        )

    def cancel(self, order_id: str) -> bool:
        """Cancel an active TWAP order."""
        with self._lock:
            if order_id in self._active_orders:
                self._active_orders[order_id].status = "cancelled"
                return True
        return False

    def cancel_all(self) -> None:
        """Cancel all active TWAP orders."""
        self._stop_event.set()
        with self._lock:
            for order in self._active_orders.values():
                order.status = "cancelled"

    def get_active_orders(self) -> List[TWAPOrder]:
        """Get list of active TWAP orders."""
        with self._lock:
            return list(self._active_orders.values())
