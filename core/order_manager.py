# -*- coding: utf-8 -*-
"""
Order Manager - Your Trading Assistant!
=======================================
Manages all your buy/sell orders in one place.

Think of it like a shopping cart manager:
- Add orders to cart
- Track what's pending
- See what got filled
- Cancel if you change your mind
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import uuid

# Import Money for precise monetary calculations
try:
    from utils.money import Money, round_to_tick, NSE_TICK_SIZE
    HAS_MONEY = True
except ImportError:
    HAS_MONEY = False

logger = logging.getLogger(__name__)


def _to_money(value: Union[float, int, Decimal, None]) -> Decimal:
    """
    Convert value to Decimal for precise monetary calculations.

    Prevents floating-point errors like:
    - 0.1 + 0.2 = 0.30000000000000004
    - Price rejections from brokers due to invalid tick sizes
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ============== ENUMS ==============

class OrderStatus(Enum):
    """Status of an order"""
    PENDING = "PENDING"       # Waiting to be sent
    PLACED = "PLACED"         # Sent to broker
    OPEN = "OPEN"             # At exchange, waiting
    COMPLETE = "COMPLETE"     # Fully filled
    PARTIAL = "PARTIAL"       # Partially filled
    CANCELLED = "CANCELLED"   # Cancelled by user
    REJECTED = "REJECTED"     # Rejected by broker/exchange
    FAILED = "FAILED"         # Failed to place


class OrderType(Enum):
    """Type of order"""
    MARKET = "MARKET"     # Execute at current price
    LIMIT = "LIMIT"       # Execute at specific price
    SL = "SL"             # Stop loss
    SL_M = "SL-M"         # Stop loss market


class Side(Enum):
    """Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """How long to hold"""
    INTRADAY = "MIS"      # Same day
    DELIVERY = "CNC"      # Keep forever
    MARGIN = "NRML"       # F&O


# ============== DATA CLASSES ==============

@dataclass
class Order:
    """
    An order to buy or sell.

    Like a shopping order with all the details.
    """
    id: str = ""
    broker_order_id: str = ""

    # What to buy/sell
    symbol: str = ""
    exchange: str = "NSE"
    side: Side = Side.BUY
    quantity: int = 0

    # How to execute
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.INTRADAY
    price: float = 0.0           # Limit price
    trigger_price: float = 0.0   # For SL orders

    # Risk management
    stop_loss: float = 0.0
    target: float = 0.0

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    message: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Metadata
    strategy: str = ""
    tag: str = ""

    def __post_init__(self):
        if not self.id:
            # Use full UUID to prevent collision risk in high-volume trading
            # Truncated UUIDs (e.g., [:8]) have significantly reduced entropy
            # and can cause order tracking issues over long database histories
            self.id = str(uuid.uuid4())

    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETE

    @property
    def is_open(self) -> bool:
        return self.status in [OrderStatus.PLACED, OrderStatus.OPEN, OrderStatus.PARTIAL]

    @property
    def value(self) -> Decimal:
        """
        Order value using precise Decimal arithmetic.

        Prevents floating-point errors that could cause incorrect
        margin calculations or balance tracking.
        """
        price = _to_money(self.average_price if self.average_price > 0 else self.price)
        return price * self.quantity

    @property
    def filled_value(self) -> Decimal:
        """
        Value of filled portion using precise Decimal arithmetic.
        """
        return _to_money(self.average_price) * self.filled_quantity

    def __str__(self):
        price_str = f"{_to_money(self.price):.2f}"
        return f"{self.side.value} {self.quantity} {self.symbol} @ {price_str} [{self.status.value}]"


# ============== ORDER MANAGER ==============

class OrderManager:
    """
    Manages all orders in one place!

    Features:
    - Place orders (paper or real)
    - Track order status
    - Cancel orders
    - Get order history
    - SERVER-SIDE STOP LOSS (Account Blowup Prevention!)

    Server-Side SL Feature:
        When a live order fills, we IMMEDIATELY place a Stop Loss Market (SL-M)
        order at the broker level. This means:
        - If Python crashes, your SL is still active at the exchange
        - If internet disconnects, your SL is still active
        - No "zombie positions" that can blow up your account

    Example:
        om = OrderManager(broker)
        order = om.place_order("RELIANCE", Side.BUY, 10, stop_loss=2450)
        # If filled, SL-M order is auto-placed at broker level!
    """

    def __init__(self, broker=None, paper_trading: bool = True, auto_server_sl: bool = True):
        """
        Initialize Order Manager.

        Args:
            broker: ZerodhaBroker instance (for live trading)
            paper_trading: Start in paper trading mode (default True)
            auto_server_sl: Auto-place server-side SL after entry fills (default True)
        """
        self.broker = broker
        self.paper_trading = paper_trading
        self.auto_server_sl = auto_server_sl  # NEW: Auto server-side stop loss

        # Order storage
        self._orders: Dict[str, Order] = {}
        self._order_history: deque = deque(maxlen=10000)

        # Server-side SL tracking: entry_order_id -> sl_order_id
        self._server_sl_orders: Dict[str, str] = {}

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_order_placed': [],
            'on_order_filled': [],
            'on_order_cancelled': [],
            'on_order_rejected': [],
            'on_server_sl_placed': [],  # NEW: When server-side SL is placed
        }

        # Paper trading state - use Decimal for precise balance tracking
        self._paper_balance: Decimal = Decimal("100000.00")
        self._paper_positions: Dict[str, int] = {}

        # Pending order metadata for async fill handling (live mode)
        # Maps order_id -> {stop_loss, target, strategy} for position creation on fill
        self._pending_order_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(f"OrderManager initialized. Paper trading: {paper_trading}, Auto Server SL: {auto_server_sl}")

    # ============== PLACE ORDERS ==============

    def place_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        product: ProductType = ProductType.INTRADAY,
        exchange: str = "NSE",
        strategy: str = "",
        tag: str = ""
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, SL, SL-M
            price: Price for limit orders
            stop_loss: Stop loss price
            target: Target price
            product: INTRADAY or DELIVERY
            exchange: NSE or BSE
            strategy: Strategy name (for tracking)
            tag: Custom tag

        Returns:
            Order object
        """
        # Create order
        order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy,
            tag=tag
        )

        # Route to paper or live
        if self.paper_trading:
            self._execute_paper_order(order)
        else:
            self._execute_live_order(order)

        # Store order
        self._orders[order.id] = order
        self._order_history.append(order)

        return order

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        **kwargs
    ) -> Order:
        """Quick buy order"""
        order_type = OrderType.LIMIT if price > 0 else OrderType.MARKET
        return self.place_order(
            symbol=symbol,
            side=Side.BUY,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            target=target,
            **kwargs
        )

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float = 0.0,
        **kwargs
    ) -> Order:
        """Quick sell order"""
        order_type = OrderType.LIMIT if price > 0 else OrderType.MARKET
        return self.place_order(
            symbol=symbol,
            side=Side.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price,
            **kwargs
        )

    # ============== PAPER TRADING ==============

    def _execute_paper_order(self, order: Order):
        """
        Execute order in paper trading mode using precise Decimal arithmetic.

        Prevents floating-point errors that could cause:
        - Vanishing pennies accumulating over many trades
        - Incorrect balance tracking
        - P&L calculation errors
        """
        # SECURITY: Validate order parameters to prevent balance manipulation
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.message = "Invalid quantity: must be positive"
            logger.error(f"Paper order rejected - invalid quantity: {order.quantity}")
            self._trigger_callback('on_order_rejected', order)
            return

        if order.price < 0:
            order.status = OrderStatus.REJECTED
            order.message = "Invalid price: cannot be negative"
            logger.error(f"Paper order rejected - invalid price: {order.price}")
            self._trigger_callback('on_order_rejected', order)
            return

        # Simulate getting current price - use Decimal for precision
        simulated_price = _to_money(order.price if order.price > 0 else 100.0)

        # SECURITY: Sanity check to prevent integer overflow
        order_value = simulated_price * order.quantity
        if order_value > Decimal("1000000000000"):  # 1 trillion
            order.status = OrderStatus.REJECTED
            order.message = "Order value too large"
            logger.error(f"Paper order rejected - value too large: {order_value}")
            self._trigger_callback('on_order_rejected', order)
            return

        # Check balance for buy orders
        if order.side == Side.BUY:
            required = simulated_price * order.quantity
            if required > self._paper_balance:
                order.status = OrderStatus.REJECTED
                order.message = "Insufficient balance"
                logger.warning(f"Order rejected: {order.message}")
                self._trigger_callback('on_order_rejected', order)
                return

            # Deduct balance using Decimal arithmetic
            self._paper_balance -= required

            # Add to positions
            current_qty = self._paper_positions.get(order.symbol, 0)
            self._paper_positions[order.symbol] = current_qty + order.quantity

        else:  # SELL
            # Check if we have shares
            current_qty = self._paper_positions.get(order.symbol, 0)
            if order.quantity > current_qty:
                order.status = OrderStatus.REJECTED
                order.message = f"Insufficient shares. Have: {current_qty}"
                logger.warning(f"Order rejected: {order.message}")
                self._trigger_callback('on_order_rejected', order)
                return

            # Add to balance using Decimal arithmetic
            self._paper_balance += simulated_price * order.quantity

            # Remove from positions
            self._paper_positions[order.symbol] = current_qty - order.quantity
            if self._paper_positions[order.symbol] == 0:
                del self._paper_positions[order.symbol]

        # Mark as complete
        order.status = OrderStatus.COMPLETE
        order.filled_quantity = order.quantity
        order.average_price = float(simulated_price)  # Store as float for compatibility
        order.filled_at = datetime.now()
        order.broker_order_id = f"PAPER-{order.id}"

        logger.info(f"Paper order filled: {order}")
        self._trigger_callback('on_order_filled', order)

    # ============== LIVE TRADING ==============

    def _execute_live_order(self, order: Order):
        """
        Execute order via real broker.

        IMPORTANT: After entry order fills, we automatically place a
        server-side SL-M order if:
        - auto_server_sl is enabled
        - stop_loss price is provided
        - This is an entry order (BUY for long, SELL for short)
        """
        if not self.broker:
            order.status = OrderStatus.FAILED
            order.message = "No broker connected"
            logger.error("Cannot place live order: No broker")
            return

        if not self.broker.is_connected:
            order.status = OrderStatus.FAILED
            order.message = "Broker not connected"
            logger.error("Cannot place live order: Broker not connected")
            return

        try:
            # Place via broker
            if order.side == Side.BUY:
                broker_id = self.broker.buy(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.price,
                    exchange=order.exchange
                )
            else:
                broker_id = self.broker.sell(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.price,
                    exchange=order.exchange
                )

            if broker_id:
                order.broker_order_id = broker_id
                order.status = OrderStatus.PLACED
                order.placed_at = datetime.now()
                logger.info(f"Order placed: {order.broker_order_id}")

                # Store metadata for async fill handling
                # When this order fills later, we need stop_loss/target/strategy
                self._pending_order_metadata[order.id] = {
                    'stop_loss': order.stop_loss,
                    'target': order.target,
                    'strategy': order.strategy,
                    'side': order.side,
                }

                self._trigger_callback('on_order_placed', order)

                # AUTO SERVER-SIDE STOP LOSS
                # Place SL-M order immediately after entry order
                # This ensures protection even if Python crashes
                if self.auto_server_sl and order.stop_loss > 0:
                    self._place_server_side_sl(order)

            else:
                order.status = OrderStatus.FAILED
                order.message = "Broker returned no order ID"

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.message = str(e)
            logger.error(f"Order failed: {e}")

    def _place_server_side_sl(self, entry_order: Order) -> Optional[str]:
        """
        Place server-side Stop Loss Market order after entry.

        CRITICAL SAFETY FEATURE:
        This order lives at the broker/exchange, NOT in Python.
        If your script crashes, the stop loss is STILL ACTIVE.

        Args:
            entry_order: The entry order that was just placed

        Returns:
            SL order ID if successful, None otherwise
        """
        if not self.broker or not self.broker.is_connected:
            logger.error("Cannot place server-side SL: Broker not connected")
            return None

        if entry_order.stop_loss <= 0:
            logger.warning(f"No stop loss price for {entry_order.symbol}, skipping server-side SL")
            return None

        # Determine SL transaction type (opposite of entry)
        # BUY entry -> SELL SL, SELL entry -> BUY SL
        from core.broker import TransactionType, ProductType as BrokerProductType

        if entry_order.side == Side.BUY:
            sl_transaction = TransactionType.SELL
        else:
            sl_transaction = TransactionType.BUY

        # Map product type
        product_map = {
            ProductType.INTRADAY: BrokerProductType.INTRADAY,
            ProductType.DELIVERY: BrokerProductType.DELIVERY,
            ProductType.MARGIN: BrokerProductType.MARGIN,
        }
        broker_product = product_map.get(entry_order.product, BrokerProductType.INTRADAY)

        try:
            sl_order_id = self.broker.place_stop_loss_order(
                symbol=entry_order.symbol,
                quantity=entry_order.quantity,
                trigger_price=entry_order.stop_loss,
                transaction_type=sl_transaction,
                product=broker_product,
                exchange=entry_order.exchange
            )

            if sl_order_id:
                # Track the SL order for this entry
                self._server_sl_orders[entry_order.id] = sl_order_id
                logger.info(
                    f"SERVER-SIDE SL PLACED: {entry_order.symbol} "
                    f"SL @ {entry_order.stop_loss} (Order ID: {sl_order_id})"
                )
                self._trigger_callback('on_server_sl_placed', {
                    'entry_order': entry_order,
                    'sl_order_id': sl_order_id,
                    'trigger_price': entry_order.stop_loss
                })
                return sl_order_id
            else:
                logger.error(f"Failed to place server-side SL for {entry_order.symbol}")
                return None

        except Exception as e:
            logger.error(f"Error placing server-side SL: {e}")
            return None

    def cancel_server_sl(self, entry_order_id: str) -> bool:
        """
        Cancel the server-side SL order for an entry.

        Call this when:
        - Target is hit (no longer need SL)
        - Manually closing position
        - Modifying stop loss

        Args:
            entry_order_id: The entry order's ID

        Returns:
            True if cancelled successfully
        """
        sl_order_id = self._server_sl_orders.get(entry_order_id)
        if not sl_order_id:
            logger.warning(f"No server-side SL found for order {entry_order_id}")
            return False

        if self.broker and self.broker.cancel_order(sl_order_id):
            del self._server_sl_orders[entry_order_id]
            logger.info(f"Server-side SL cancelled: {sl_order_id}")
            return True
        return False

    def modify_server_sl(self, entry_order_id: str, new_trigger_price: float) -> bool:
        """
        Modify the server-side SL order (for trailing stops).

        Args:
            entry_order_id: The entry order's ID
            new_trigger_price: New stop loss price

        Returns:
            True if modified successfully
        """
        sl_order_id = self._server_sl_orders.get(entry_order_id)
        if not sl_order_id:
            logger.warning(f"No server-side SL found for order {entry_order_id}")
            return False

        if self.broker and self.broker.modify_stop_loss_order(sl_order_id, new_trigger_price):
            logger.info(f"Server-side SL modified: {sl_order_id} -> trigger @ {new_trigger_price}")
            return True
        return False

    def get_server_sl_orders(self) -> Dict[str, str]:
        """Get all server-side SL order mappings (entry_id -> sl_id)."""
        return self._server_sl_orders.copy()

    # ============== ORDER SYNC (LIVE MODE) ==============

    def sync_orders(self) -> List[Order]:
        """
        Sync order statuses from broker (for live trading).

        CRITICAL for live mode: Orders are placed asynchronously and may
        fill at any time. This method polls the broker for status updates
        and triggers callbacks when orders complete.

        Call this periodically in your main trading loop (every 1-5 seconds).

        Returns:
            List of orders that changed status during this sync
        """
        if self.paper_trading or not self.broker:
            return []

        if not self.broker.is_connected:
            logger.warning("Cannot sync orders: Broker not connected")
            return []

        changed_orders = []

        try:
            # Get current order statuses from broker
            broker_orders = self.broker.get_orders()

            # Map broker order IDs to our orders
            broker_order_map = {str(bo.order_id): bo for bo in broker_orders}

            # Check each open order for status changes
            for order in self.get_open_orders():
                if not order.broker_order_id:
                    continue

                broker_order = broker_order_map.get(order.broker_order_id)
                if not broker_order:
                    continue

                # Check if status changed
                old_status = order.status
                new_status = self._map_broker_status(broker_order.status)

                if new_status != old_status:
                    self.update_order_status(
                        order_id=order.id,
                        status=new_status,
                        filled_quantity=broker_order.filled_quantity,
                        average_price=broker_order.average_price,
                        message=broker_order.message
                    )
                    changed_orders.append(order)
                    logger.info(f"Order status changed: {order.symbol} {old_status.value} -> {new_status.value}")

        except Exception as e:
            logger.error(f"Order sync failed: {e}")

        return changed_orders

    def _map_broker_status(self, broker_status: str) -> OrderStatus:
        """Map broker status string to OrderStatus enum."""
        status_map = {
            'COMPLETE': OrderStatus.COMPLETE,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'OPEN': OrderStatus.OPEN,
            'PENDING': OrderStatus.PENDING,
            'TRIGGER PENDING': OrderStatus.OPEN,  # SL orders waiting
            'VALIDATION PENDING': OrderStatus.PLACED,
            'PUT ORDER REQUEST RECEIVED': OrderStatus.PLACED,
            'MODIFY PENDING': OrderStatus.OPEN,
            'CANCEL PENDING': OrderStatus.OPEN,
        }
        return status_map.get(broker_status.upper(), OrderStatus.PLACED)

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: int = 0,
        average_price: float = 0.0,
        message: str = ""
    ) -> bool:
        """
        Update order status and trigger appropriate callbacks.

        This is called by sync_orders() or can be called directly if you
        have your own order update mechanism (e.g., WebSocket postback).

        Args:
            order_id: Internal order ID
            status: New status
            filled_quantity: Quantity filled
            average_price: Average fill price
            message: Status message

        Returns:
            True if order was updated
        """
        if order_id not in self._orders:
            logger.warning(f"Cannot update unknown order: {order_id}")
            return False

        order = self._orders[order_id]
        old_status = order.status

        # Update order fields
        order.status = status
        if filled_quantity > 0:
            order.filled_quantity = filled_quantity
        if average_price > 0:
            order.average_price = average_price
        if message:
            order.message = message

        # Trigger appropriate callbacks based on new status
        if status == OrderStatus.COMPLETE and old_status != OrderStatus.COMPLETE:
            order.filled_at = datetime.now()
            logger.info(f"Order filled: {order}")
            self._trigger_callback('on_order_filled', order)

            # Clean up pending metadata after callback (callback may need it)
            if order_id in self._pending_order_metadata:
                del self._pending_order_metadata[order_id]

        elif status == OrderStatus.CANCELLED and old_status != OrderStatus.CANCELLED:
            logger.info(f"Order cancelled: {order}")
            self._trigger_callback('on_order_cancelled', order)

            # Clean up pending metadata
            if order_id in self._pending_order_metadata:
                del self._pending_order_metadata[order_id]

        elif status == OrderStatus.REJECTED and old_status != OrderStatus.REJECTED:
            logger.warning(f"Order rejected: {order} - {message}")
            self._trigger_callback('on_order_rejected', order)

            # Clean up pending metadata
            if order_id in self._pending_order_metadata:
                del self._pending_order_metadata[order_id]

        return True

    def get_pending_order_metadata(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored metadata for a pending order (stop_loss, target, strategy).

        Used by TradingEngine's on_order_filled callback to create positions
        with the correct risk parameters.

        Args:
            order_id: Internal order ID

        Returns:
            Dict with stop_loss, target, strategy, side or None if not found
        """
        return self._pending_order_metadata.get(order_id)

    # ============== ORDER MANAGEMENT ==============

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled
        """
        if order_id not in self._orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self._orders[order_id]

        if not order.is_open:
            logger.warning(f"Order not cancelable: {order.status}")
            return False

        if self.paper_trading:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Paper order cancelled: {order_id}")
        else:
            if self.broker and order.broker_order_id:
                if self.broker.cancel_order(order.broker_order_id):
                    order.status = OrderStatus.CANCELLED
                else:
                    return False

        self._trigger_callback('on_order_cancelled', order)
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [o for o in self._orders.values() if o.is_open]

    def get_orders(self, symbol: str = None, status: OrderStatus = None) -> List[Order]:
        """
        Get orders with optional filters.

        Args:
            symbol: Filter by symbol
            status: Filter by status

        Returns:
            List of matching orders
        """
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    def get_order_history(self) -> List[Order]:
        """Get all historical orders"""
        return list(self._order_history)

    # ============== PAPER TRADING CONTROL ==============

    def set_paper_trading(self, enabled: bool):
        """Enable/disable paper trading"""
        self.paper_trading = enabled
        mode = "PAPER" if enabled else "LIVE"
        logger.info(f"Trading mode: {mode}")

    def set_paper_balance(self, balance: Union[float, Decimal]):
        """Set paper trading balance using precise Decimal"""
        self._paper_balance = _to_money(balance)
        logger.info(f"Paper balance set to: Rs.{self._paper_balance:,.2f}")

    def get_paper_balance(self) -> Decimal:
        """Get paper trading balance as precise Decimal"""
        return self._paper_balance

    def get_paper_balance_float(self) -> float:
        """Get paper trading balance as float (for API compatibility)"""
        return float(self._paper_balance)

    def get_paper_positions(self) -> Dict[str, int]:
        """Get paper trading positions"""
        return self._paper_positions.copy()

    def reset_paper_trading(self, balance: Union[float, Decimal] = Decimal("100000.00")):
        """Reset paper trading to initial state with precise Decimal balance"""
        self._paper_balance = _to_money(balance)
        self._paper_positions = {}
        self._orders = {}
        logger.info(f"Paper trading reset. Balance: Rs.{self._paper_balance:,.2f}")

    # ============== CALLBACKS ==============

    def on_order_placed(self, callback: Callable):
        """Register callback for order placed"""
        self._callbacks['on_order_placed'].append(callback)

    def on_order_filled(self, callback: Callable):
        """Register callback for order filled"""
        self._callbacks['on_order_filled'].append(callback)

    def on_order_cancelled(self, callback: Callable):
        """Register callback for order cancelled"""
        self._callbacks['on_order_cancelled'].append(callback)

    def on_order_rejected(self, callback: Callable):
        """Register callback for order rejected"""
        self._callbacks['on_order_rejected'].append(callback)

    def on_server_sl_placed(self, callback: Callable):
        """Register callback for server-side SL placed"""
        self._callbacks['on_server_sl_placed'].append(callback)

    def _trigger_callback(self, event: str, order):
        """Trigger callbacks for event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # ============== STATS ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get order statistics"""
        total = len(self._orders)
        complete = len([o for o in self._orders.values() if o.status == OrderStatus.COMPLETE])
        cancelled = len([o for o in self._orders.values() if o.status == OrderStatus.CANCELLED])
        rejected = len([o for o in self._orders.values() if o.status == OrderStatus.REJECTED])

        return {
            'total_orders': total,
            'complete': complete,
            'cancelled': cancelled,
            'rejected': rejected,
            'open': len(self.get_open_orders()),
            'paper_balance': self._paper_balance if self.paper_trading else None,
            'paper_positions': len(self._paper_positions) if self.paper_trading else None
        }


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("ORDER MANAGER - Test")
    print("=" * 50)

    # Create order manager in paper mode
    om = OrderManager(paper_trading=True)
    om.set_paper_balance(100000)

    # Place some orders
    order1 = om.buy("RELIANCE", quantity=10, price=2500)
    print(f"\nOrder 1: {order1}")

    order2 = om.buy("TCS", quantity=5, price=3500)
    print(f"Order 2: {order2}")

    # Check balance
    print(f"\nPaper Balance: Rs.{om.get_paper_balance():,.0f}")
    print(f"Positions: {om.get_paper_positions()}")

    # Sell
    order3 = om.sell("RELIANCE", quantity=5, price=2550)
    print(f"\nOrder 3: {order3}")

    print(f"\nFinal Balance: Rs.{om.get_paper_balance():,.0f}")
    print(f"Final Positions: {om.get_paper_positions()}")

    print(f"\nStats: {om.get_stats()}")

    print("\n" + "=" * 50)
    print("Order Manager ready!")
    print("=" * 50)
