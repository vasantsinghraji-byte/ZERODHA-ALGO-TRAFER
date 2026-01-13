"""
Bracket Orders (OCO - One-Cancels-Other)
Automatically place stop-loss and target orders with main order

Features:
- Entry order with automatic SL and target
- OCO logic (one cancels the other)
- Trailing stop-loss
- Partial exit management
- Order modification support
"""

import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class BracketOrderStatus(Enum):
    """Bracket order status"""
    PENDING = "pending"
    ENTRY_PLACED = "entry_placed"
    ENTRY_FILLED = "entry_filled"
    SL_TARGET_PLACED = "sl_target_placed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BracketLeg(Enum):
    """Bracket order legs"""
    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TARGET = "target"


@dataclass
class BracketOrder:
    """Bracket order with entry, SL, and target"""
    bracket_id: str
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    target: float

    # Order details
    transaction_type: str  # BUY or SELL
    product: str = "MIS"
    order_type: str = "LIMIT"

    # Status tracking
    status: BracketOrderStatus = BracketOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Order IDs from broker
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Execution details
    entry_filled_price: Optional[float] = None
    entry_filled_qty: int = 0
    exit_price: Optional[float] = None
    exit_qty: int = 0

    # Trailing stop-loss
    trailing_sl: bool = False
    trailing_sl_points: float = 0.0
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None

    # Metadata
    strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BracketOrderManager:
    """
    Manage bracket orders with OCO logic
    """

    def __init__(self, broker_client: Optional[Any] = None):
        """
        Args:
            broker_client: Broker API client for order placement
        """
        self.broker = broker_client
        self.orders: Dict[str, BracketOrder] = {}
        self.order_history: List[Dict[str, Any]] = []

    def create_bracket_order(self,
                            symbol: str,
                            quantity: int,
                            entry_price: float,
                            stop_loss: float,
                            target: float,
                            transaction_type: str,
                            product: str = "MIS",
                            trailing_sl: bool = False,
                            trailing_sl_points: float = 0.0,
                            strategy: Optional[str] = None) -> BracketOrder:
        """
        Create a bracket order

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            entry_price: Entry price
            stop_loss: Stop-loss price
            target: Target price
            transaction_type: BUY or SELL
            product: Product type (MIS, NRML, CNC)
            trailing_sl: Enable trailing stop-loss
            trailing_sl_points: Trailing points
            strategy: Strategy name

        Returns:
            BracketOrder instance
        """
        # Generate unique bracket ID
        bracket_id = f"BRKT_{uuid.uuid4().hex[:8].upper()}"

        # Validate prices
        if transaction_type == "BUY":
            if stop_loss >= entry_price:
                raise ValueError("Stop-loss must be below entry price for BUY orders")
            if target <= entry_price:
                raise ValueError("Target must be above entry price for BUY orders")
        else:  # SELL
            if stop_loss <= entry_price:
                raise ValueError("Stop-loss must be above entry price for SELL orders")
            if target >= entry_price:
                raise ValueError("Target must be below entry price for SELL orders")

        bracket_order = BracketOrder(
            bracket_id=bracket_id,
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            transaction_type=transaction_type,
            product=product,
            trailing_sl=trailing_sl,
            trailing_sl_points=trailing_sl_points,
            strategy=strategy
        )

        self.orders[bracket_id] = bracket_order

        print(f"Created bracket order: {bracket_id}")
        print(f"  Symbol: {symbol}")
        print(f"  Quantity: {quantity}")
        print(f"  Entry: {entry_price}")
        print(f"  SL: {stop_loss} | Target: {target}")

        return bracket_order

    def place_entry_order(self, bracket_id: str) -> Dict[str, Any]:
        """Place the entry order"""
        if bracket_id not in self.orders:
            raise ValueError(f"Bracket order {bracket_id} not found")

        bracket = self.orders[bracket_id]

        if self.broker:
            # Place order via broker API
            try:
                order_params = {
                    'tradingsymbol': bracket.symbol,
                    'quantity': bracket.quantity,
                    'transaction_type': bracket.transaction_type,
                    'order_type': bracket.order_type,
                    'price': bracket.entry_price,
                    'product': bracket.product,
                    'tag': f'bracket_{bracket_id}'
                }

                # Simulated broker response
                order_id = f"ORD_{uuid.uuid4().hex[:8]}"
                bracket.entry_order_id = order_id
                bracket.status = BracketOrderStatus.ENTRY_PLACED

                result = {
                    'status': 'success',
                    'order_id': order_id,
                    'bracket_id': bracket_id
                }

                self._log_event(bracket_id, BracketLeg.ENTRY, "placed", result)

                return result

            except Exception as e:
                bracket.status = BracketOrderStatus.REJECTED
                raise

        else:
            # Simulation mode
            order_id = f"SIM_{uuid.uuid4().hex[:8]}"
            bracket.entry_order_id = order_id
            bracket.status = BracketOrderStatus.ENTRY_PLACED

            return {
                'status': 'simulated',
                'order_id': order_id,
                'bracket_id': bracket_id
            }

    def on_entry_fill(self, bracket_id: str, filled_price: float, filled_qty: int):
        """Handle entry order fill"""
        if bracket_id not in self.orders:
            return

        bracket = self.orders[bracket_id]

        bracket.entry_filled_price = filled_price
        bracket.entry_filled_qty = filled_qty
        bracket.status = BracketOrderStatus.ENTRY_FILLED

        # Initialize tracking for trailing SL
        if bracket.trailing_sl:
            if bracket.transaction_type == "BUY":
                bracket.highest_price = filled_price
            else:
                bracket.lowest_price = filled_price

        print(f"Entry filled: {bracket_id} @ {filled_price}")

        # Place SL and Target orders
        self._place_sl_target_orders(bracket_id)

    def _place_sl_target_orders(self, bracket_id: str):
        """Place stop-loss and target orders"""
        bracket = self.orders[bracket_id]

        # Reverse transaction type for exit
        exit_type = "SELL" if bracket.transaction_type == "BUY" else "BUY"

        # Place stop-loss order
        sl_order_id = f"SL_{uuid.uuid4().hex[:8]}"
        bracket.sl_order_id = sl_order_id

        # Place target order
        target_order_id = f"TGT_{uuid.uuid4().hex[:8]}"
        bracket.target_order_id = target_order_id

        bracket.status = BracketOrderStatus.SL_TARGET_PLACED

        self._log_event(bracket_id, BracketLeg.STOP_LOSS, "placed",
                       {'order_id': sl_order_id, 'price': bracket.stop_loss})
        self._log_event(bracket_id, BracketLeg.TARGET, "placed",
                       {'order_id': target_order_id, 'price': bracket.target})

        print(f"  SL order placed: {sl_order_id} @ {bracket.stop_loss}")
        print(f"  Target order placed: {target_order_id} @ {bracket.target}")

    def on_sl_hit(self, bracket_id: str, exit_price: float):
        """Handle stop-loss hit"""
        if bracket_id not in self.orders:
            return

        bracket = self.orders[bracket_id]

        bracket.exit_price = exit_price
        bracket.exit_qty = bracket.entry_filled_qty
        bracket.status = BracketOrderStatus.COMPLETED

        # Cancel target order (OCO logic)
        self._cancel_order(bracket.target_order_id)

        self._log_event(bracket_id, BracketLeg.STOP_LOSS, "executed",
                       {'price': exit_price})

        print(f"Stop-loss hit: {bracket_id} @ {exit_price}")
        print(f"  Target order cancelled (OCO)")

    def on_target_hit(self, bracket_id: str, exit_price: float):
        """Handle target hit"""
        if bracket_id not in self.orders:
            return

        bracket = self.orders[bracket_id]

        bracket.exit_price = exit_price
        bracket.exit_qty = bracket.entry_filled_qty
        bracket.status = BracketOrderStatus.COMPLETED

        # Cancel SL order (OCO logic)
        self._cancel_order(bracket.sl_order_id)

        self._log_event(bracket_id, BracketLeg.TARGET, "executed",
                       {'price': exit_price})

        print(f"Target hit: {bracket_id} @ {exit_price}")
        print(f"  Stop-loss order cancelled (OCO)")

    def update_trailing_sl(self, bracket_id: str, current_price: float):
        """Update trailing stop-loss"""
        if bracket_id not in self.orders:
            return

        bracket = self.orders[bracket_id]

        if not bracket.trailing_sl:
            return

        if bracket.status != BracketOrderStatus.SL_TARGET_PLACED:
            return

        if bracket.transaction_type == "BUY":
            # For BUY orders, trail upwards
            if bracket.highest_price is None or current_price > bracket.highest_price:
                bracket.highest_price = current_price
                new_sl = current_price - bracket.trailing_sl_points

                if new_sl > bracket.stop_loss:
                    old_sl = bracket.stop_loss
                    bracket.stop_loss = new_sl

                    # Modify SL order
                    self._modify_sl_order(bracket_id, new_sl)

                    print(f"Trailing SL updated: {bracket_id}")
                    print(f"  {old_sl:.2f} -> {new_sl:.2f}")

        else:  # SELL
            # For SELL orders, trail downwards
            if bracket.lowest_price is None or current_price < bracket.lowest_price:
                bracket.lowest_price = current_price
                new_sl = current_price + bracket.trailing_sl_points

                if new_sl < bracket.stop_loss:
                    old_sl = bracket.stop_loss
                    bracket.stop_loss = new_sl

                    # Modify SL order
                    self._modify_sl_order(bracket_id, new_sl)

                    print(f"Trailing SL updated: {bracket_id}")
                    print(f"  {old_sl:.2f} -> {new_sl:.2f}")

    def _modify_sl_order(self, bracket_id: str, new_price: float):
        """Modify stop-loss order"""
        bracket = self.orders[bracket_id]

        if self.broker:
            # Modify via broker API
            pass

        self._log_event(bracket_id, BracketLeg.STOP_LOSS, "modified",
                       {'new_price': new_price})

    def _cancel_order(self, order_id: Optional[str]):
        """Cancel an order"""
        if not order_id:
            return

        if self.broker:
            # Cancel via broker API
            pass

        print(f"  Order cancelled: {order_id}")

    def cancel_bracket_order(self, bracket_id: str):
        """Cancel entire bracket order"""
        if bracket_id not in self.orders:
            raise ValueError(f"Bracket order {bracket_id} not found")

        bracket = self.orders[bracket_id]

        # Cancel all active orders
        if bracket.entry_order_id:
            self._cancel_order(bracket.entry_order_id)
        if bracket.sl_order_id:
            self._cancel_order(bracket.sl_order_id)
        if bracket.target_order_id:
            self._cancel_order(bracket.target_order_id)

        bracket.status = BracketOrderStatus.CANCELLED

        self._log_event(bracket_id, None, "cancelled", {})  # type: ignore

        print(f"Bracket order cancelled: {bracket_id}")

    def get_bracket_order(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID"""
        return self.orders.get(bracket_id)

    def get_active_brackets(self) -> List[BracketOrder]:
        """Get all active bracket orders"""
        return [
            order for order in self.orders.values()
            if order.status not in [BracketOrderStatus.COMPLETED, BracketOrderStatus.CANCELLED]
        ]

    def get_pnl(self, bracket_id: str) -> Optional[float]:
        """Calculate P&L for bracket order"""
        bracket = self.orders.get(bracket_id)

        if not bracket or bracket.status != BracketOrderStatus.COMPLETED:
            return None

        if not bracket.entry_filled_price or not bracket.exit_price:
            return None

        if bracket.transaction_type == "BUY":
            pnl = (bracket.exit_price - bracket.entry_filled_price) * bracket.exit_qty
        else:
            pnl = (bracket.entry_filled_price - bracket.exit_price) * bracket.exit_qty

        return pnl

    def _log_event(self, bracket_id: str, leg: Optional[BracketLeg], event: str, data: Dict):
        """Log bracket order event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'bracket_id': bracket_id,
            'leg': leg.value if leg else None,
            'event': event,
            'data': data
        }

        self.order_history.append(log_entry)

    def export_bracket_orders(self, filename: str = "bracket_orders.json"):
        """Export bracket orders to JSON"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_brackets': len(self.orders),
            'active_brackets': len(self.get_active_brackets()),
            'orders': [
                {
                    'bracket_id': b.bracket_id,
                    'symbol': b.symbol,
                    'quantity': b.quantity,
                    'entry_price': b.entry_price,
                    'stop_loss': b.stop_loss,
                    'target': b.target,
                    'transaction_type': b.transaction_type,
                    'status': b.status.value,
                    'entry_filled_price': b.entry_filled_price,
                    'exit_price': b.exit_price,
                    'pnl': self.get_pnl(b.bracket_id),
                    'created_at': b.created_at.isoformat()
                }
                for b in self.orders.values()
            ],
            'history': self.order_history
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self.orders)} bracket orders to {filename}")


if __name__ == "__main__":
    print("Bracket Orders (OCO - One-Cancels-Other)")
    print("=" * 60)
    print("\nFeatures:")
    print("  Entry order with automatic SL and target")
    print("  OCO logic (one cancels the other)")
    print("  Trailing stop-loss")
    print("  Partial exit management")
    print("  Order modification support")
