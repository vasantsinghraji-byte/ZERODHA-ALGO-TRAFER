# -*- coding: utf-8 -*-
"""
Iceberg Execution Algorithm - Hide Your Size!
==============================================

⚠️  REQUIRES BROKER INTEGRATION - CURRENTLY SIMULATION ONLY ⚠️

LIMITATIONS (must be addressed for production use):
1. NO ORDER STATUS MONITORING: Real iceberg needs real-time order updates.
   Current implementation ASSUMES INSTANT FILLS (see _place_order method).
   Zerodha doesn't provide WebSocket order updates - you must poll.

2. SYNCHRONOUS BLOCKING: execute_sync() blocks the thread with time.sleep().
   This will freeze your TradingBot if it's single-threaded.
   Use execute() (async) with proper event loop, OR run in a background thread.

3. NO PARTIAL FILL HANDLING: The broker integration assumes orders fill completely.
   Real markets have partial fills that need proper tracking.

WHAT YOU NEED FOR PRODUCTION:
- Order status polling loop or WebSocket integration
- Proper async architecture (not blocking the main trading loop)
- Error recovery for failed order placements
- Position reconciliation after fills

RECOMMENDED ARCHITECTURE:
- Run IcebergExecutor in a dedicated thread/coroutine
- Use broker's order update callbacks if available
- Implement heartbeat checks for hung orders

Example (SIMULATION ONLY - fills are simulated):
    >>> iceberg = IcebergExecutor(broker, config)
    >>> result = iceberg.execute_sync("HDFCBANK", 1000, Side.BUY, price=1500)
    >>> # Note: In simulation, this returns immediately with fake fills
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..broker import ZerodhaBroker

logger = logging.getLogger(__name__)


class Side(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class IcebergStatus(Enum):
    """Status of iceberg order."""
    PENDING = "pending"
    ACTIVE = "active"
    REFILLING = "refilling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RefillTrigger(Enum):
    """When to refill the visible quantity."""
    ON_FULL_FILL = "full"        # Only when fully filled
    ON_PARTIAL = "partial"        # When any fill occurs
    ON_THRESHOLD = "threshold"    # When below threshold %


@dataclass
class IcebergConfig:
    """Configuration for Iceberg execution."""
    # Visibility
    visible_quantity: int = 100         # Quantity shown to market
    visible_pct: float = 0.0            # Alternative: % of total (if > 0)
    randomize_visible: bool = True      # Randomize visible qty
    visible_variance_pct: float = 0.2   # +/- variance for randomization

    # Refill behavior
    refill_trigger: RefillTrigger = RefillTrigger.ON_FULL_FILL
    refill_threshold_pct: float = 0.3   # Refill when below this % filled
    refill_delay_seconds: float = 0.5   # Delay between refills
    max_refills: int = 1000             # Safety limit

    # Price management
    limit_price: Optional[float] = None
    price_adjust_ticks: int = 0         # Adjust price by N ticks on refill
    tick_size: float = 0.05             # Price tick size

    # Execution
    max_duration_minutes: float = 0     # 0 = no limit
    cancel_on_price_move_pct: float = 0 # Cancel if price moves too much

    # Callbacks
    on_fill: Optional[Callable] = None
    on_refill: Optional[Callable] = None
    on_complete: Optional[Callable] = None


@dataclass
class IcebergFill:
    """A single fill event in an iceberg order."""
    fill_id: int
    quantity: int
    price: float
    timestamp: datetime
    order_id: str = ""
    is_refill: bool = False


@dataclass
class IcebergOrder:
    """An Iceberg order being executed."""
    order_id: str
    symbol: str
    side: Side
    total_quantity: int
    visible_quantity: int
    limit_price: float
    fills: List[IcebergFill] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: IcebergStatus = IcebergStatus.PENDING
    current_order_id: str = ""
    refill_count: int = 0

    @property
    def filled_quantity(self) -> int:
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def average_price(self) -> float:
        total_value = sum(f.quantity * f.price for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_value / total_qty if total_qty > 0 else 0

    @property
    def is_complete(self) -> bool:
        return self.filled_quantity >= self.total_quantity


@dataclass
class IcebergResult:
    """Result of Iceberg execution."""
    order: IcebergOrder
    success: bool
    total_filled: int
    average_price: float
    total_cost: float
    execution_time_seconds: float
    refill_count: int
    fill_count: int
    benchmark_price: float = 0.0
    slippage_bps: float = 0.0

    def summary(self) -> str:
        """Generate execution summary."""
        lines = [
            "=" * 50,
            "ICEBERG EXECUTION SUMMARY",
            "=" * 50,
            f"Symbol: {self.order.symbol}",
            f"Side: {self.order.side.value}",
            f"Total Qty: {self.order.total_quantity}",
            f"Visible Qty: {self.order.visible_quantity}",
            f"Filled Qty: {self.total_filled}",
            f"Fill Rate: {self.total_filled / self.order.total_quantity:.1%}",
            f"Average Price: Rs.{self.average_price:.2f}",
            f"Limit Price: Rs.{self.order.limit_price:.2f}",
            f"Total Cost: Rs.{self.total_cost:,.2f}",
            f"Benchmark: Rs.{self.benchmark_price:.2f}",
            f"Slippage: {self.slippage_bps:.1f} bps",
            f"Refills: {self.refill_count}",
            f"Individual Fills: {self.fill_count}",
            f"Duration: {self.execution_time_seconds:.1f}s",
            f"Status: {'SUCCESS' if self.success else 'PARTIAL/FAILED'}",
            "=" * 50
        ]
        return "\n".join(lines)


class IcebergExecutor:
    """
    Iceberg Order Executor.

    Hides large orders by showing only a small visible portion.
    Automatically refills when the visible portion is filled.

    Example:
        >>> iceberg = IcebergExecutor(broker)
        >>> result = iceberg.execute_sync(
        ...     symbol="RELIANCE",
        ...     total_quantity=1000,
        ...     side=Side.BUY,
        ...     limit_price=2500,
        ...     visible_quantity=100
        ... )
    """

    def __init__(
        self,
        broker: Optional['ZerodhaBroker'] = None,
        config: Optional[IcebergConfig] = None
    ):
        self.broker = broker
        self.config = config or IcebergConfig()
        self._active_orders: Dict[str, IcebergOrder] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def create_order(
        self,
        symbol: str,
        total_quantity: int,
        side: Side,
        limit_price: float,
        visible_quantity: Optional[int] = None,
        config: Optional[IcebergConfig] = None
    ) -> IcebergOrder:
        """Create an iceberg order."""
        import uuid

        config = config or self.config
        order_id = f"ICE-{uuid.uuid4().hex[:8]}"

        # Calculate visible quantity
        if visible_quantity:
            vis_qty = visible_quantity
        elif config.visible_pct > 0:
            vis_qty = int(total_quantity * config.visible_pct)
        else:
            vis_qty = config.visible_quantity

        # Randomize if enabled
        if config.randomize_visible:
            import random
            variance = int(vis_qty * config.visible_variance_pct)
            vis_qty = vis_qty + random.randint(-variance, variance)

        vis_qty = max(1, min(vis_qty, total_quantity))

        order = IcebergOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            visible_quantity=vis_qty,
            limit_price=limit_price,
            start_time=datetime.now()
        )

        logger.info(
            f"Created Iceberg order {order_id}: {total_quantity} {symbol} "
            f"(visible: {vis_qty})"
        )
        return order

    def execute_sync(
        self,
        symbol: str,
        total_quantity: int,
        side: Side,
        limit_price: float,
        visible_quantity: Optional[int] = None,
        config: Optional[IcebergConfig] = None
    ) -> IcebergResult:
        """
        Execute iceberg order synchronously.

        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to fill
            side: BUY or SELL
            limit_price: Limit price for orders
            visible_quantity: Quantity to show (optional)
            config: Optional config override

        Returns:
            IcebergResult with execution details
        """
        config = config or self.config
        order = self.create_order(
            symbol, total_quantity, side, limit_price,
            visible_quantity, config
        )

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = IcebergStatus.ACTIVE
        start_time = time.time()
        benchmark_price = limit_price
        fill_id = 0
        deadline = None

        if config.max_duration_minutes > 0:
            deadline = datetime.now().timestamp() + (config.max_duration_minutes * 60)

        try:
            while order.remaining_quantity > 0:
                if self._stop_event.is_set():
                    order.status = IcebergStatus.CANCELLED
                    break

                # Check deadline
                if deadline and time.time() > deadline:
                    logger.warning(f"Iceberg {order.order_id} timed out")
                    break

                # Check refill limit
                if order.refill_count >= config.max_refills:
                    logger.warning(f"Iceberg {order.order_id} hit refill limit")
                    break

                # Calculate current visible quantity
                current_visible = min(order.visible_quantity, order.remaining_quantity)

                if config.randomize_visible and order.refill_count > 0:
                    import random
                    variance = int(current_visible * config.visible_variance_pct)
                    current_visible = current_visible + random.randint(-variance, variance)
                    current_visible = max(1, min(current_visible, order.remaining_quantity))

                # Get adjusted price
                adjusted_price = self._adjust_price(
                    order.limit_price,
                    order.side,
                    order.refill_count,
                    config
                )

                # Place visible order
                try:
                    filled_qty, fill_price, child_order_id = self._place_order(
                        symbol, current_visible, side, adjusted_price
                    )

                    if filled_qty > 0:
                        fill = IcebergFill(
                            fill_id=fill_id,
                            quantity=filled_qty,
                            price=fill_price,
                            timestamp=datetime.now(),
                            order_id=child_order_id,
                            is_refill=order.refill_count > 0
                        )
                        order.fills.append(fill)
                        fill_id += 1

                        if config.on_fill:
                            try:
                                config.on_fill(fill)
                            except Exception:
                                pass

                        logger.debug(
                            f"Iceberg fill: {filled_qty}@{fill_price:.2f} "
                            f"(total: {order.filled_quantity}/{order.total_quantity})"
                        )

                        # Check if we need to refill
                        if self._should_refill(filled_qty, current_visible, config):
                            order.status = IcebergStatus.REFILLING
                            order.refill_count += 1

                            if config.on_refill:
                                try:
                                    config.on_refill(order)
                                except Exception:
                                    pass

                            # Delay before refill
                            time.sleep(config.refill_delay_seconds)
                            order.status = IcebergStatus.ACTIVE
                        else:
                            # Partial fill - wait a bit
                            time.sleep(0.1)

                    else:
                        # No fill - wait and retry
                        time.sleep(1.0)

                except Exception as e:
                    logger.error(f"Iceberg order placement failed: {e}")
                    time.sleep(1.0)

            order.status = IcebergStatus.COMPLETED if order.is_complete else IcebergStatus.FAILED
            order.end_time = datetime.now()

        except Exception as e:
            order.status = IcebergStatus.FAILED
            logger.error(f"Iceberg execution failed: {e}")

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

    async def execute(
        self,
        symbol: str,
        total_quantity: int,
        side: Side,
        limit_price: float,
        visible_quantity: Optional[int] = None,
        config: Optional[IcebergConfig] = None
    ) -> IcebergResult:
        """Execute iceberg order asynchronously."""
        config = config or self.config
        order = self.create_order(
            symbol, total_quantity, side, limit_price,
            visible_quantity, config
        )

        with self._lock:
            self._active_orders[order.order_id] = order

        order.status = IcebergStatus.ACTIVE
        start_time = time.time()
        benchmark_price = limit_price
        fill_id = 0
        deadline = None

        if config.max_duration_minutes > 0:
            deadline = datetime.now().timestamp() + (config.max_duration_minutes * 60)

        try:
            while order.remaining_quantity > 0:
                if self._stop_event.is_set():
                    order.status = IcebergStatus.CANCELLED
                    break

                if deadline and time.time() > deadline:
                    break

                if order.refill_count >= config.max_refills:
                    break

                current_visible = min(order.visible_quantity, order.remaining_quantity)

                if config.randomize_visible and order.refill_count > 0:
                    import random
                    variance = int(current_visible * config.visible_variance_pct)
                    current_visible = max(1, current_visible + random.randint(-variance, variance))
                    current_visible = min(current_visible, order.remaining_quantity)

                adjusted_price = self._adjust_price(
                    order.limit_price, order.side, order.refill_count, config
                )

                try:
                    filled_qty, fill_price, child_order_id = self._place_order(
                        symbol, current_visible, side, adjusted_price
                    )

                    if filled_qty > 0:
                        fill = IcebergFill(
                            fill_id=fill_id,
                            quantity=filled_qty,
                            price=fill_price,
                            timestamp=datetime.now(),
                            order_id=child_order_id,
                            is_refill=order.refill_count > 0
                        )
                        order.fills.append(fill)
                        fill_id += 1

                        if config.on_fill:
                            try:
                                config.on_fill(fill)
                            except Exception:
                                pass

                        if self._should_refill(filled_qty, current_visible, config):
                            order.status = IcebergStatus.REFILLING
                            order.refill_count += 1

                            if config.on_refill:
                                try:
                                    config.on_refill(order)
                                except Exception:
                                    pass

                            await asyncio.sleep(config.refill_delay_seconds)
                            order.status = IcebergStatus.ACTIVE
                        else:
                            await asyncio.sleep(0.1)
                    else:
                        await asyncio.sleep(1.0)

                except Exception as e:
                    logger.error(f"Order failed: {e}")
                    await asyncio.sleep(1.0)

            order.status = IcebergStatus.COMPLETED if order.is_complete else IcebergStatus.FAILED
            order.end_time = datetime.now()

        except Exception as e:
            order.status = IcebergStatus.FAILED
            logger.error(f"Iceberg failed: {e}")

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

    def _should_refill(
        self,
        filled_qty: int,
        visible_qty: int,
        config: IcebergConfig
    ) -> bool:
        """Determine if we should refill based on trigger type."""
        if config.refill_trigger == RefillTrigger.ON_FULL_FILL:
            return filled_qty >= visible_qty
        elif config.refill_trigger == RefillTrigger.ON_PARTIAL:
            return filled_qty > 0
        elif config.refill_trigger == RefillTrigger.ON_THRESHOLD:
            fill_ratio = filled_qty / visible_qty if visible_qty > 0 else 0
            return fill_ratio >= config.refill_threshold_pct
        return filled_qty >= visible_qty

    def _adjust_price(
        self,
        base_price: float,
        side: Side,
        refill_count: int,
        config: IcebergConfig
    ) -> float:
        """Adjust price based on refill count and configuration."""
        if config.price_adjust_ticks == 0:
            return base_price

        # Adjust more aggressively with each refill
        adjustment = config.price_adjust_ticks * config.tick_size * refill_count

        if side == Side.BUY:
            return base_price + adjustment  # Pay more to fill
        else:
            return base_price - adjustment  # Accept less to fill

    def _place_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        price: float
    ) -> tuple:
        """
        Place order and return (filled_qty, fill_price, order_id).

        ⚠️ WARNING: This method ASSUMES INSTANT FILL!

        Real broker integration would need:
        1. Place order → get order_id
        2. Poll order status until filled/cancelled
        3. Handle partial fills across multiple polls
        4. Timeout handling for stuck orders

        Zerodha's Kite API does NOT provide WebSocket order updates.
        You must poll kite.orders() or kite.order_history(order_id).
        """
        if self.broker:
            import warnings
            warnings.warn(
                "IcebergExecutor._place_order() ASSUMES INSTANT FILLS. "
                "Real Zerodha orders may remain pending. You need to implement "
                "order status polling with kite.order_history(order_id) for production.",
                UserWarning,
                stacklevel=2
            )
            try:
                if side == Side.BUY:
                    order = self.broker.buy(symbol, quantity, price)
                else:
                    order = self.broker.sell(symbol, quantity, price)

                # ⚠️ BUG: This returns immediately without waiting for fill!
                # In production, you need to poll until order.status == 'COMPLETE'
                return (
                    order.filled_quantity or quantity,  # Assumes full fill - WRONG!
                    order.average_price or price,
                    order.order_id or ""
                )
            except Exception as e:
                logger.error(f"Order failed: {e}")
                raise

        # Simulated fill - WARNING: This is not real order execution!
        import uuid
        import warnings

        warnings.warn(
            f"IcebergExecutor using SIMULATED fills for {symbol}. "
            "Orders are NOT being sent to any broker. "
            "Connect a real broker for production use.",
            UserWarning,
            stacklevel=3
        )

        return quantity, price, f"SIM-{uuid.uuid4().hex[:8]}"

    def _create_result(
        self,
        order: IcebergOrder,
        start_time: float,
        benchmark_price: float
    ) -> IcebergResult:
        """Create result from completed order."""
        execution_time = time.time() - start_time
        avg_price = order.average_price
        total_filled = order.filled_quantity
        total_cost = total_filled * avg_price

        # Calculate slippage
        if benchmark_price > 0 and avg_price > 0:
            if order.side == Side.BUY:
                slippage = (avg_price - benchmark_price) / benchmark_price * 10000
            else:
                slippage = (benchmark_price - avg_price) / benchmark_price * 10000
        else:
            slippage = 0

        return IcebergResult(
            order=order,
            success=total_filled >= order.total_quantity * 0.9,
            total_filled=total_filled,
            average_price=avg_price,
            total_cost=total_cost,
            execution_time_seconds=execution_time,
            refill_count=order.refill_count,
            fill_count=len(order.fills),
            benchmark_price=benchmark_price,
            slippage_bps=slippage
        )

    def cancel(self, order_id: str) -> bool:
        """Cancel active iceberg order."""
        with self._lock:
            if order_id in self._active_orders:
                self._active_orders[order_id].status = IcebergStatus.CANCELLED
                return True
        return False

    def cancel_all(self) -> None:
        """Cancel all active iceberg orders."""
        self._stop_event.set()

    def get_active_orders(self) -> List[IcebergOrder]:
        """Get list of active orders."""
        with self._lock:
            return list(self._active_orders.values())
