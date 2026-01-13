# -*- coding: utf-8 -*-
"""
OrderManager - Order execution and tracking (Async)
Handles order placement, modifications, and lifecycle management
"""
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime
import logging
import asyncio
from enum import Enum
from .enhanced_strategy import EnhancedSignal, SignalType

if TYPE_CHECKING:
    from .services import IBrokerService

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class Order:
    """Order representation"""

    def __init__(self, order_id: str, instrument_token: int, symbol: str,
                 transaction_type: str, quantity: int, order_type: str,
                 price: float = 0, trigger_price: float = 0, product: str = "MIS"):
        self.order_id = order_id
        self.instrument_token = instrument_token
        self.symbol = symbol
        self.transaction_type = transaction_type  # BUY or SELL
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.trigger_price = trigger_price
        self.product = product

        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.average_price = 0.0
        self.placed_at = datetime.now()
        self.updated_at = datetime.now()

        # Parent order ID (for bracket orders)
        self.parent_order_id: Optional[str] = None

        # Child orders (SL and target)
        self.sl_order_id: Optional[str] = None
        self.target_order_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        return {
            'order_id': self.order_id,
            'instrument_token': self.instrument_token,
            'symbol': self.symbol,
            'transaction_type': self.transaction_type,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'trigger_price': self.trigger_price,
            'product': self.product,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'placed_at': self.placed_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class OrderManager:
    """
    Manages order execution and tracking (Async)

    Responsibilities:
    - Place orders based on signals (async)
    - Track order status (open, filled, cancelled)
    - Manage bracket orders (entry + SL + target)
    - Modify and cancel orders (async)
    - Provide order history and analytics
    """

    def __init__(self, broker: 'IBrokerService'):
        """
        Initialize OrderManager with broker service

        Args:
            broker: Broker service instance (IBrokerService)
        """
        self.broker = broker

        # Active orders (order_id -> Order)
        self.active_orders: Dict[str, Order] = {}

        # Order history (order_id -> Order)
        self.order_history: Dict[str, Order] = {}

        # Order callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []

        # Instrument positions (instrument_token -> position dict)
        self.positions: Dict[int, Dict] = {}

    async def place_order_from_signal(self, signal: EnhancedSignal, instrument_token: int,
                                       symbol: str, quantity: int = None) -> Optional[str]:
        """
        Place order based on trading signal (async)

        Args:
            signal: Trading signal
            instrument_token: Instrument token
            symbol: Trading symbol
            quantity: Order quantity (optional, uses signal if not provided)

        Returns:
            Order ID or None if failed
        """
        try:
            if signal.signal_type == SignalType.NEUTRAL:
                logger.info("NEUTRAL signal - no order placed")
                return None

            # Determine transaction type
            transaction_type = "BUY" if signal.signal_type == SignalType.LONG else "SELL"

            # Use signal quantity if not provided
            qty = quantity or signal.position_size

            # Place main entry order via broker service (async)
            order_id = await self.broker.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type=transaction_type,
                quantity=qty,
                order_type="MARKET",
                product="MIS",
                variety="regular"
            )

            # Create Order object
            order = Order(
                order_id=order_id,
                instrument_token=instrument_token,
                symbol=symbol,
                transaction_type=transaction_type,
                quantity=qty,
                order_type="MARKET",
                product="MIS"
            )

            # Store in active orders
            self.active_orders[order_id] = order

            logger.info(f"Placed {transaction_type} order for {symbol}: "
                       f"{qty} @ MARKET (Order ID: {order_id})")

            # Place SL and target orders (if signal has them) - run concurrently
            tasks = []
            if signal.stop_loss and signal.entry_price:
                tasks.append(self._place_sl_order(order, signal))
            if signal.take_profit_1 and signal.entry_price:
                tasks.append(self._place_target_order(order, signal))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Notify callbacks
            self._notify_order_callbacks(order)

            return order_id

        except Exception as e:
            logger.error(f"Failed to place order from signal: {e}")
            return None

    async def _place_sl_order(self, parent_order: Order, signal: EnhancedSignal):
        """
        Place stop loss order (async)

        Args:
            parent_order: Parent entry order
            signal: Trading signal
        """
        try:
            # Opposite transaction type for SL
            sl_transaction = "SELL" if parent_order.transaction_type == "BUY" else "BUY"

            sl_order_id = await self.broker.place_order(
                tradingsymbol=parent_order.symbol,
                exchange="NSE",
                transaction_type=sl_transaction,
                quantity=parent_order.quantity,
                order_type="SL-M",
                trigger_price=signal.stop_loss,
                product="MIS",
                variety="regular"
            )

            parent_order.sl_order_id = sl_order_id

            logger.info(f"Placed SL order for {parent_order.symbol}: "
                       f"trigger @ {signal.stop_loss} (Order ID: {sl_order_id})")

        except Exception as e:
            logger.error(f"Failed to place SL order: {e}")

    async def _place_target_order(self, parent_order: Order, signal: EnhancedSignal):
        """
        Place target order (async)

        Args:
            parent_order: Parent entry order
            signal: Trading signal
        """
        try:
            # Opposite transaction type for target
            target_transaction = "SELL" if parent_order.transaction_type == "BUY" else "BUY"

            target_order_id = await self.broker.place_order(
                tradingsymbol=parent_order.symbol,
                exchange="NSE",
                transaction_type=target_transaction,
                quantity=parent_order.quantity,
                order_type="LIMIT",
                price=signal.take_profit_1,
                product="MIS",
                variety="regular"
            )

            parent_order.target_order_id = target_order_id

            logger.info(f"Placed TARGET order for {parent_order.symbol}: "
                       f"limit @ {signal.take_profit_1} (Order ID: {target_order_id})")

        except Exception as e:
            logger.error(f"Failed to place target order: {e}")

    def update_order_status(self, order_id: str, status_data: Dict):
        """
        Update order status from WebSocket or polling

        Args:
            order_id: Order ID
            status_data: Status data from Kite
        """
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]

                # Update status
                status = status_data.get('status', '').upper()
                if status == 'COMPLETE':
                    order.status = OrderStatus.COMPLETE
                elif status == 'CANCELLED':
                    order.status = OrderStatus.CANCELLED
                elif status == 'REJECTED':
                    order.status = OrderStatus.REJECTED
                elif status in ['OPEN', 'TRIGGER PENDING']:
                    order.status = OrderStatus.OPEN

                # Update fill info
                order.filled_quantity = status_data.get('filled_quantity', 0)
                order.average_price = status_data.get('average_price', 0.0)
                order.updated_at = datetime.now()

                # Move to history if completed/cancelled/rejected
                if order.status in [OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    self.order_history[order_id] = order
                    del self.active_orders[order_id]

                logger.info(f"Order {order_id} updated: {order.status.value}")

                # Notify callbacks
                self._notify_order_callbacks(order)

        except Exception as e:
            logger.error(f"Failed to update order status: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (async)

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            await self.broker.cancel_order(order_id)

            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                self.active_orders[order_id].updated_at = datetime.now()

            logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    # Note: modify_order not implemented in current broker interface
    # Can be added if needed in future

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.active_orders.get(order_id) or self.order_history.get(order_id)

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())

    def get_order_history(self) -> List[Order]:
        """Get order history"""
        return list(self.order_history.values())

    def register_order_callback(self, callback: Callable[[Order], None]):
        """Register callback for order updates"""
        if callback not in self.order_callbacks:
            self.order_callbacks.append(callback)
            logger.info(f"Registered order callback: {callback.__name__}")

    def unregister_order_callback(self, callback: Callable[[Order], None]):
        """Unregister order callback"""
        if callback in self.order_callbacks:
            self.order_callbacks.remove(callback)

    def _notify_order_callbacks(self, order: Order):
        """Notify all callbacks of order update"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback {callback.__name__}: {e}")

    async def update_positions(self):
        """Fetch and update current positions (async)"""
        try:
            positions_list = await self.broker.get_positions()

            # Update positions dict
            self.positions.clear()
            for pos in positions_list:
                instrument_token = pos.get('instrument_token')
                if instrument_token:
                    self.positions[instrument_token] = pos

            logger.info(f"Updated positions: {len(self.positions)} active")

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    def get_position(self, instrument_token: int) -> Optional[Dict]:
        """Get position for instrument"""
        return self.positions.get(instrument_token)

    def get_stats(self) -> Dict:
        """Get OrderManager statistics"""
        total_orders = len(self.active_orders) + len(self.order_history)

        completed_orders = len([o for o in self.order_history.values()
                               if o.status == OrderStatus.COMPLETE])

        return {
            'active_orders': len(self.active_orders),
            'total_orders': total_orders,
            'completed_orders': completed_orders,
            'active_positions': len(self.positions),
            'registered_callbacks': len(self.order_callbacks)
        }
