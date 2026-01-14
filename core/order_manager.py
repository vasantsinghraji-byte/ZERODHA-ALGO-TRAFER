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
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


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
            self.id = str(uuid.uuid4())[:8]

    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETE

    @property
    def is_open(self) -> bool:
        return self.status in [OrderStatus.PLACED, OrderStatus.OPEN, OrderStatus.PARTIAL]

    @property
    def value(self) -> float:
        """Order value"""
        price = self.average_price if self.average_price > 0 else self.price
        return price * self.quantity

    @property
    def filled_value(self) -> float:
        """Value of filled portion"""
        return self.average_price * self.filled_quantity

    def __str__(self):
        return f"{self.side.value} {self.quantity} {self.symbol} @ {self.price:.2f} [{self.status.value}]"


# ============== ORDER MANAGER ==============

class OrderManager:
    """
    Manages all orders in one place!

    Features:
    - Place orders (paper or real)
    - Track order status
    - Cancel orders
    - Get order history

    Example:
        om = OrderManager(broker)
        order = om.place_order("RELIANCE", Side.BUY, 10)
        print(f"Order status: {order.status}")
    """

    def __init__(self, broker=None, paper_trading: bool = True):
        """
        Initialize Order Manager.

        Args:
            broker: ZerodhaBroker instance (for live trading)
            paper_trading: Start in paper trading mode (default True)
        """
        self.broker = broker
        self.paper_trading = paper_trading

        # Order storage
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_order_placed': [],
            'on_order_filled': [],
            'on_order_cancelled': [],
            'on_order_rejected': [],
        }

        # Paper trading state
        self._paper_balance = 100000.0
        self._paper_positions: Dict[str, int] = {}

        logger.info(f"OrderManager initialized. Paper trading: {paper_trading}")

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
        """Execute order in paper trading mode"""
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

        # Simulate getting current price
        simulated_price = order.price if order.price > 0 else 100.0  # Default for testing

        # SECURITY: Sanity check to prevent integer overflow
        order_value = simulated_price * order.quantity
        if order_value > 1e12:  # 1 trillion - reasonable max for paper trading
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

            # Deduct balance
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

            # Add to balance
            self._paper_balance += simulated_price * order.quantity

            # Remove from positions
            self._paper_positions[order.symbol] = current_qty - order.quantity
            if self._paper_positions[order.symbol] == 0:
                del self._paper_positions[order.symbol]

        # Mark as complete
        order.status = OrderStatus.COMPLETE
        order.filled_quantity = order.quantity
        order.average_price = simulated_price
        order.filled_at = datetime.now()
        order.broker_order_id = f"PAPER-{order.id}"

        logger.info(f"Paper order filled: {order}")
        self._trigger_callback('on_order_filled', order)

    # ============== LIVE TRADING ==============

    def _execute_live_order(self, order: Order):
        """Execute order via real broker"""
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
                self._trigger_callback('on_order_placed', order)
            else:
                order.status = OrderStatus.FAILED
                order.message = "Broker returned no order ID"

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.message = str(e)
            logger.error(f"Order failed: {e}")

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
        return self._order_history.copy()

    # ============== PAPER TRADING CONTROL ==============

    def set_paper_trading(self, enabled: bool):
        """Enable/disable paper trading"""
        self.paper_trading = enabled
        mode = "PAPER" if enabled else "LIVE"
        logger.info(f"Trading mode: {mode}")

    def set_paper_balance(self, balance: float):
        """Set paper trading balance"""
        self._paper_balance = balance
        logger.info(f"Paper balance set to: Rs.{balance:,.0f}")

    def get_paper_balance(self) -> float:
        """Get paper trading balance"""
        return self._paper_balance

    def get_paper_positions(self) -> Dict[str, int]:
        """Get paper trading positions"""
        return self._paper_positions.copy()

    def reset_paper_trading(self, balance: float = 100000.0):
        """Reset paper trading to initial state"""
        self._paper_balance = balance
        self._paper_positions = {}
        self._orders = {}
        logger.info(f"Paper trading reset. Balance: Rs.{balance:,.0f}")

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

    def _trigger_callback(self, event: str, order: Order):
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
