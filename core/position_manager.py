# -*- coding: utf-8 -*-
"""
Position Manager - Track Your Stock Holdings!
=============================================
Keeps track of all your open positions and their P&L.

Think of it like a scoreboard showing:
- What stocks you own
- How much you paid
- How much they're worth now
- Are you winning or losing?
"""

import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


def _to_money(value: Union[float, int, Decimal, None]) -> Decimal:
    """
    Convert value to Decimal for precise monetary calculations.

    Prevents floating-point errors in P&L calculations that could cause:
    - Vanishing pennies accumulating over many trades
    - Incorrect unrealized/realized P&L
    - Wrong position values
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ============== DATA CLASSES ==============

class PositionSide(Enum):
    """Long or Short position"""
    LONG = "LONG"     # You bought and own
    SHORT = "SHORT"   # You sold first (borrowed)


@dataclass
class Position:
    """
    A stock position you currently hold.

    Like a receipt showing what you bought and how it's doing.
    """
    symbol: str
    exchange: str = "NSE"
    side: PositionSide = PositionSide.LONG

    # Quantities
    quantity: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0

    # Prices
    average_price: float = 0.0
    last_price: float = 0.0

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    pnl_percent: float = 0.0

    # Value
    buy_value: float = 0.0
    current_value: float = 0.0

    # Risk management
    stop_loss: float = 0.0
    target: float = 0.0

    # Metadata
    strategy: str = ""
    # WARNING: Mutable/callable defaults in dataclasses must use field(default_factory=...)
    # Using `opened_at: datetime = datetime.now()` would freeze ONE timestamp for ALL instances!
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_open(self) -> bool:
        """Is position still open?"""
        return self.quantity != 0

    @property
    def is_profitable(self) -> bool:
        """Is position in profit?"""
        return self.unrealized_pnl > 0

    @property
    def emoji(self) -> str:
        """Visual indicator"""
        if self.pnl_percent > 5:
            return "🚀"
        elif self.pnl_percent > 0:
            return "📈"
        elif self.pnl_percent > -5:
            return "📉"
        else:
            return "💥"

    def update_price(self, price: float):
        """
        Update with new price using precise Decimal arithmetic.

        Validates price and uses Decimal to prevent:
        - Division by zero errors
        - Floating-point P&L calculation errors
        - Accumulated precision loss over many updates
        """
        # Validate price - reject invalid values
        if price <= 0:
            logger.warning(f"Invalid price {price} for {self.symbol}, skipping update")
            return

        # Use Decimal for precise P&L calculations
        price_decimal = _to_money(price)
        buy_value_decimal = _to_money(self.buy_value)

        self.last_price = float(price_decimal)
        current_value_decimal = price_decimal * self.quantity
        self.current_value = float(current_value_decimal)

        unrealized_decimal = current_value_decimal - buy_value_decimal
        self.unrealized_pnl = float(unrealized_decimal)

        realized_decimal = _to_money(self.realized_pnl)
        self.total_pnl = float(unrealized_decimal + realized_decimal)

        # Safe division - prevent ZeroDivisionError
        if buy_value_decimal > 0:
            pnl_pct = (unrealized_decimal / buy_value_decimal) * 100
            self.pnl_percent = float(pnl_pct)
        else:
            self.pnl_percent = 0.0
            logger.warning(f"Zero buy_value for {self.symbol}, cannot calculate P&L percent")

        self.updated_at = datetime.now()

    def __str__(self):
        avg_price = _to_money(self.average_price)
        pnl = _to_money(self.unrealized_pnl)
        return f"{self.emoji} {self.symbol}: {self.quantity} @ Rs.{avg_price:.2f} | P&L: Rs.{pnl:+.0f} ({self.pnl_percent:+.1f}%)"


# ============== POSITION MANAGER ==============

class PositionManager:
    """
    Manages all your positions!

    Features:
    - Track open positions
    - Calculate P&L in real-time
    - Monitor stop loss / targets
    - Get portfolio summary

    Example:
        pm = PositionManager()
        pm.add_position("RELIANCE", 10, 2500)
        pm.update_price("RELIANCE", 2550)
        print(pm.get_position("RELIANCE"))
    """

    def __init__(self, broker=None, persistence=None):
        """
        Initialize Position Manager.

        Args:
            broker: ZerodhaBroker instance (for live positions)
            persistence: PersistenceManager for state recovery (optional)
        """
        self.broker = broker
        self._persistence = persistence

        # Thread lock for concurrent access safety
        # RLock allows same thread to acquire multiple times (reentrant)
        # Required because methods like close_position call reduce_position
        self._lock = threading.RLock()

        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []

        # Portfolio totals - use Decimal for precise tracking
        self._total_invested: Decimal = Decimal("0")
        self._total_value: Decimal = Decimal("0")
        self._total_pnl: Decimal = Decimal("0")

        # Load persisted positions on startup
        if self._persistence:
            self._load_persisted_positions()

        logger.info("PositionManager initialized")

    def _load_persisted_positions(self):
        """Load positions from persistence on startup."""
        if not self._persistence:
            return

        try:
            saved_positions = self._persistence.load_positions()
            for pos_data in saved_positions:
                # Reconstruct Position object from saved data
                pos = Position(
                    symbol=pos_data['symbol'],
                    exchange=pos_data.get('exchange', 'NSE'),
                    side=PositionSide(pos_data.get('side', 'LONG')),
                    quantity=pos_data['quantity'],
                    buy_quantity=pos_data.get('buy_quantity', pos_data['quantity']),
                    sell_quantity=pos_data.get('sell_quantity', 0),
                    average_price=pos_data['average_price'],
                    last_price=pos_data.get('last_price', pos_data['average_price']),
                    buy_value=pos_data.get('buy_value', 0),
                    current_value=pos_data.get('current_value', 0),
                    realized_pnl=pos_data.get('realized_pnl', 0),
                    stop_loss=pos_data.get('stop_loss', 0),
                    target=pos_data.get('target', 0),
                    strategy=pos_data.get('strategy', '')
                )
                self._positions[pos.symbol] = pos
                logger.info(f"Restored position: {pos.symbol} {pos.quantity}x @ Rs.{pos.average_price}")

            if saved_positions:
                self._update_totals()
                logger.info(f"Loaded {len(saved_positions)} positions from persistence")

        except Exception as e:
            logger.error(f"Failed to load persisted positions: {e}")

    def _persist_position(self, position: Position):
        """Save position to persistence."""
        if not self._persistence:
            return

        try:
            self._persistence.save_position({
                'symbol': position.symbol,
                'exchange': position.exchange,
                'side': position.side.value,
                'quantity': position.quantity,
                'buy_quantity': position.buy_quantity,
                'sell_quantity': position.sell_quantity,
                'average_price': position.average_price,
                'last_price': position.last_price,
                'buy_value': position.buy_value,
                'current_value': position.current_value,
                'realized_pnl': position.realized_pnl,
                'stop_loss': position.stop_loss,
                'target': position.target,
                'strategy': position.strategy,
                'opened_at': position.opened_at.isoformat() if position.opened_at else datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to persist position {position.symbol}: {e}")

    def _delete_persisted_position(self, symbol: str, position: Optional[Position] = None, exit_price: float = 0):
        """Delete position from persistence and save to closed history."""
        if not self._persistence:
            return

        try:
            # Save to closed positions history first
            if position:
                self._persistence.save_closed_position({
                    'symbol': position.symbol,
                    'exchange': position.exchange,
                    'side': position.side.value,
                    'quantity': position.buy_quantity,
                    'buy_quantity': position.buy_quantity,
                    'average_price': position.average_price,
                    'realized_pnl': position.realized_pnl,
                    'strategy': position.strategy,
                    'opened_at': position.opened_at.isoformat() if position.opened_at else ''
                }, exit_price)

            # Then delete from open positions
            self._persistence.delete_position(symbol)
        except Exception as e:
            logger.error(f"Failed to delete persisted position {symbol}: {e}")

    # ============== ADD/UPDATE POSITIONS ==============

    def add_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        exchange: str = "NSE",
        side: PositionSide = PositionSide.LONG,
        stop_loss: float = 0.0,
        target: float = 0.0,
        strategy: str = ""
    ) -> Position:
        """
        Add a new position or update existing.

        Thread-safe: Uses _lock to protect position modifications.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            exchange: NSE or BSE
            side: LONG or SHORT
            stop_loss: Stop loss price
            target: Target price
            strategy: Strategy name

        Returns:
            Position object
        """
        with self._lock:
            if symbol in self._positions:
                # Update existing position using Decimal for precise average price
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity

                # Use Decimal for precise value calculations
                existing_value = _to_money(pos.buy_value)
                new_value = _to_money(price) * quantity
                total_value = existing_value + new_value

                pos.quantity = total_qty
                pos.buy_quantity += quantity
                pos.buy_value = float(total_value)

                # Precise average price calculation
                if total_qty > 0:
                    pos.average_price = float(total_value / total_qty)
                else:
                    pos.average_price = 0

                if stop_loss > 0:
                    pos.stop_loss = stop_loss
                if target > 0:
                    pos.target = target

                pos.update_price(pos.last_price if pos.last_price > 0 else price)

                # Persist updated position
                self._persist_position(pos)

                logger.info(f"Position updated: {pos}")
                return pos

            else:
                # Create new position using Decimal for precise values
                price_decimal = _to_money(price)
                buy_value = float(price_decimal * quantity)

                pos = Position(
                    symbol=symbol,
                    exchange=exchange,
                    side=side,
                    quantity=quantity,
                    buy_quantity=quantity,
                    average_price=float(price_decimal),
                    last_price=float(price_decimal),
                    buy_value=buy_value,
                    current_value=buy_value,
                    stop_loss=stop_loss or float(price_decimal * Decimal("0.98")),  # Default 2% SL
                    target=target or float(price_decimal * Decimal("1.04")),  # Default 4% target
                    strategy=strategy
                )

                self._positions[symbol] = pos
                self._update_totals()

                # Persist new position
                self._persist_position(pos)

                logger.info(f"Position opened: {pos}")
                return pos

    def reduce_position(
        self,
        symbol: str,
        quantity: int,
        price: float
    ) -> Optional[Position]:
        """
        Reduce or close a position (partial/full sell) using precise Decimal arithmetic.

        Thread-safe: Uses _lock to protect position modifications.

        Args:
            symbol: Stock symbol
            quantity: Shares to sell
            price: Exit price

        Returns:
            Updated position or None if closed
        """
        with self._lock:
            if symbol not in self._positions:
                logger.warning(f"Position not found: {symbol}")
                return None

            pos = self._positions[symbol]

            if quantity > pos.quantity:
                logger.warning(f"Cannot sell more than owned. Have: {pos.quantity}")
                quantity = pos.quantity

            # Calculate realized P&L using Decimal for precision
            exit_price = _to_money(price)
            entry_price = _to_money(pos.average_price)
            exit_value = exit_price * quantity
            entry_value = entry_price * quantity
            realized = exit_value - entry_value

            pos.sell_quantity += quantity
            pos.quantity -= quantity
            pos.realized_pnl = float(_to_money(pos.realized_pnl) + realized)

            if pos.quantity == 0:
                # Position fully closed
                pos.update_price(price)
                self._closed_positions.append(pos)
                del self._positions[symbol]
                pnl_display = _to_money(pos.total_pnl)
                logger.info(f"Position closed: {symbol} | P&L: Rs.{pnl_display:+.0f}")
                self._update_totals()

                # Delete from persistence and save to closed history
                self._delete_persisted_position(symbol, pos, price)

                return None
            else:
                # Position partially closed - recalculate buy_value precisely
                pos.buy_value = float(_to_money(pos.average_price) * pos.quantity)
                pos.update_price(price)
                logger.info(f"Position reduced: {pos}")
                self._update_totals()

                # Update persistence with reduced position
                self._persist_position(pos)

                return pos

    def close_position(self, symbol: str, price: float) -> float:
        """
        Fully close a position.

        Thread-safe: Uses _lock (reentrant, so reduce_position can acquire again).

        Args:
            symbol: Stock symbol
            price: Exit price

        Returns:
            Realized P&L
        """
        with self._lock:
            if symbol not in self._positions:
                return 0.0

            pos = self._positions[symbol]
            quantity = pos.quantity

            self.reduce_position(symbol, quantity, price)

            return pos.realized_pnl

    # ============== UPDATE PRICES ==============

    def update_price(self, symbol: str, price: float):
        """
        Update price for a position.

        Thread-safe: Uses _lock to protect position access.

        Args:
            symbol: Stock symbol
            price: Current market price
        """
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].update_price(price)
                self._update_totals()

    def update_all_prices(self, prices: Dict[str, float]):
        """
        Update prices for multiple positions atomically.

        Thread-safe: Single lock acquisition for all updates.

        Args:
            prices: Dict of symbol -> price
        """
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol].update_price(price)
            self._update_totals()

    def sync_with_broker(self):
        """
        Sync positions with broker (for live trading).

        Thread-safe: Uses _lock via called methods.
        """
        if not self.broker:
            logger.warning("No broker connected")
            return

        try:
            broker_positions = self.broker.get_positions()

            for bp in broker_positions:
                if self.has_position(bp.symbol):
                    self.update_price(bp.symbol, bp.last_price)
                elif bp.quantity != 0:
                    # New position from broker
                    self.add_position(
                        symbol=bp.symbol,
                        quantity=bp.quantity,
                        price=bp.average_price
                    )
                    self.update_price(bp.symbol, bp.last_price)

            logger.info("Positions synced with broker")

        except Exception as e:
            logger.error(f"Sync failed: {e}")

    # ============== GETTERS ==============

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.

        Thread-safe: Uses _lock for consistent read.
        """
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """
        Get all open positions.

        Thread-safe: Returns a copy to prevent iteration errors outside lock.
        """
        with self._lock:
            return list(self._positions.values())

    def get_positions_by_strategy(self, strategy: str) -> List[Position]:
        """
        Get positions for a strategy.

        Thread-safe: Returns a copy to prevent iteration errors outside lock.
        """
        with self._lock:
            return [p for p in self._positions.values() if p.strategy == strategy]

    def get_closed_positions(self) -> List[Position]:
        """
        Get all closed positions.

        Thread-safe: Returns a copy to prevent iteration errors outside lock.
        """
        with self._lock:
            return self._closed_positions.copy()

    def has_position(self, symbol: str) -> bool:
        """
        Check if position exists.

        Thread-safe: Uses _lock for consistent read.
        """
        with self._lock:
            return symbol in self._positions

    # ============== RISK MONITORING ==============

    def check_stop_losses(self, prices: Dict[str, float]) -> List[str]:
        """
        Check which positions hit stop loss.

        Thread-safe: Iterates over a snapshot to prevent dictionary mutation errors.

        Args:
            prices: Current prices

        Returns:
            List of symbols that hit stop loss
        """
        triggered = []

        with self._lock:
            # Iterate over snapshot to prevent "dictionary changed size" errors
            positions_snapshot = list(self._positions.items())

        for symbol, pos in positions_snapshot:
            price = prices.get(symbol, pos.last_price)

            if pos.side == PositionSide.LONG and price <= pos.stop_loss:
                triggered.append(symbol)
                logger.warning(f"STOP LOSS triggered: {symbol} @ Rs.{price:.2f}")
            elif pos.side == PositionSide.SHORT and price >= pos.stop_loss:
                triggered.append(symbol)
                logger.warning(f"STOP LOSS triggered: {symbol} @ Rs.{price:.2f}")

        return triggered

    def check_targets(self, prices: Dict[str, float]) -> List[str]:
        """
        Check which positions hit target.

        Thread-safe: Iterates over a snapshot to prevent dictionary mutation errors.

        Args:
            prices: Current prices

        Returns:
            List of symbols that hit target
        """
        triggered = []

        with self._lock:
            # Iterate over snapshot to prevent "dictionary changed size" errors
            positions_snapshot = list(self._positions.items())

        for symbol, pos in positions_snapshot:
            price = prices.get(symbol, pos.last_price)

            if pos.side == PositionSide.LONG and price >= pos.target:
                triggered.append(symbol)
                logger.info(f"TARGET hit: {symbol} @ Rs.{price:.2f}")
            elif pos.side == PositionSide.SHORT and price <= pos.target:
                triggered.append(symbol)
                logger.info(f"TARGET hit: {symbol} @ Rs.{price:.2f}")

        return triggered

    def set_stop_loss(self, symbol: str, stop_loss: float):
        """
        Set stop loss for a position.

        Thread-safe: Uses _lock to protect position modification.
        """
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].stop_loss = stop_loss
                logger.info(f"Stop loss set: {symbol} @ Rs.{stop_loss:.2f}")

    def set_target(self, symbol: str, target: float):
        """
        Set target for a position.

        Thread-safe: Uses _lock to protect position modification.
        """
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].target = target
                logger.info(f"Target set: {symbol} @ Rs.{target:.2f}")

    # ============== PORTFOLIO SUMMARY ==============

    def _update_totals(self):
        """
        Update portfolio totals using Decimal for precision.

        Note: This is a private method always called from within locked context.
        """
        self._total_invested = sum(
            (_to_money(p.buy_value) for p in self._positions.values()),
            Decimal("0")
        )
        self._total_value = sum(
            (_to_money(p.current_value) for p in self._positions.values()),
            Decimal("0")
        )
        self._total_pnl = sum(
            (_to_money(p.unrealized_pnl) for p in self._positions.values()),
            Decimal("0")
        )

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value as precise Decimal. Thread-safe."""
        with self._lock:
            return self._total_value

    def get_total_pnl(self) -> Decimal:
        """Get total unrealized P&L as precise Decimal. Thread-safe."""
        with self._lock:
            return self._total_pnl

    def get_total_invested(self) -> Decimal:
        """Get total invested amount as precise Decimal. Thread-safe."""
        with self._lock:
            return self._total_invested

    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with precise monetary values.

        Thread-safe: Uses _lock to ensure consistent snapshot.

        Returns:
            Dict with portfolio stats
        """
        with self._lock:
            positions = list(self._positions.values())
            closed_positions = self._closed_positions.copy()
            total_invested = self._total_invested
            total_value = self._total_value
            total_pnl = self._total_pnl

        profitable = [p for p in positions if p.is_profitable]
        losing = [p for p in positions if not p.is_profitable]

        # Calculate realized P&L with precision
        realized_pnl = sum(
            (_to_money(p.realized_pnl) for p in closed_positions),
            Decimal("0")
        )

        # Calculate P&L percent with precision
        if total_invested > 0:
            pnl_percent = float((total_pnl / total_invested) * 100)
        else:
            pnl_percent = 0.0

        return {
            'total_positions': len(positions),
            'profitable_positions': len(profitable),
            'losing_positions': len(losing),
            'total_invested': float(total_invested),
            'current_value': float(total_value),
            'unrealized_pnl': float(total_pnl),
            'realized_pnl': float(realized_pnl),
            'pnl_percent': pnl_percent,
            'best_performer': max(positions, key=lambda p: p.pnl_percent).symbol if positions else None,
            'worst_performer': min(positions, key=lambda p: p.pnl_percent).symbol if positions else None
        }

    def print_portfolio(self):
        """
        Print nicely formatted portfolio with precise monetary values.

        Thread-safe: Uses _lock via get_all_positions and local copies.
        """
        print("\n" + "=" * 70)
        print("PORTFOLIO SUMMARY")
        print("=" * 70)

        # Get snapshot of positions and totals
        with self._lock:
            positions = list(self._positions.values())
            total_invested = self._total_invested
            total_value = self._total_value
            total_pnl = self._total_pnl

        if not positions:
            print("No open positions")
        else:
            print(f"{'Symbol':<12} {'Qty':>8} {'Avg Price':>12} {'LTP':>12} {'P&L':>12} {'%':>8}")
            print("-" * 70)

            for pos in sorted(positions, key=lambda p: p.unrealized_pnl, reverse=True):
                avg = _to_money(pos.average_price)
                ltp = _to_money(pos.last_price)
                pnl = _to_money(pos.unrealized_pnl)
                print(f"{pos.symbol:<12} {pos.quantity:>8} {avg:>12.2f} "
                      f"{ltp:>12.2f} {pnl:>+12.0f} {pos.pnl_percent:>+7.1f}%")

            print("-" * 70)
            pnl_pct = float(total_pnl / total_invested * 100) if total_invested else 0
            print(f"{'TOTAL':<12} {'':<8} {total_invested:>12.0f} "
                  f"{total_value:>12.0f} {total_pnl:>+12.0f} "
                  f"{pnl_pct:>+7.1f}%")

        print("=" * 70)

    # ============== CLEAR ==============

    def clear_all(self):
        """
        Clear all positions (use carefully!).

        Thread-safe: Uses _lock to protect state reset.
        """
        with self._lock:
            self._positions = {}
            self._closed_positions = []
            self._total_invested = Decimal("0")
            self._total_value = Decimal("0")
            self._total_pnl = Decimal("0")

        # Clear persistence as well
        if self._persistence:
            try:
                self._persistence.clear_positions()
            except Exception as e:
                logger.error(f"Failed to clear persisted positions: {e}")

        logger.info("All positions cleared")


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("POSITION MANAGER - Test")
    print("=" * 50)

    pm = PositionManager()

    # Add positions
    pm.add_position("RELIANCE", 10, 2500, stop_loss=2450, target=2600)
    pm.add_position("TCS", 5, 3500)
    pm.add_position("INFY", 20, 1500)

    # Update prices (simulate market movement)
    pm.update_price("RELIANCE", 2550)  # Up 2%
    pm.update_price("TCS", 3400)       # Down 3%
    pm.update_price("INFY", 1525)      # Up 1.7%

    # Print portfolio
    pm.print_portfolio()

    # Print summary
    print("\nSummary:")
    summary = pm.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Check stop losses
    sl_hit = pm.check_stop_losses({
        "RELIANCE": 2550,
        "TCS": 3300,  # Below SL
        "INFY": 1525
    })
    print(f"\nStop losses triggered: {sl_hit}")

    print("\n" + "=" * 50)
    print("Position Manager ready!")
    print("=" * 50)
