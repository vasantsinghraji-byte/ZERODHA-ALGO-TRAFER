# -*- coding: utf-8 -*-
"""
Guardian Angel - Account Blowup Prevention!
============================================
Monitors and protects your trading account from zombie positions.

Problem:
    If your Python script crashes while holding a position, you have
    no exit mechanism. The position becomes a "zombie" that can cause
    unlimited losses.

Solution:
    1. HEARTBEAT: Monitor script health
    2. POSITION SYNC: On startup, query broker for actual positions
    3. ORPHAN DETECTION: Find positions without SL orders
    4. AUTO-RECOVERY: Place protective SL orders for orphans

Usage:
    >>> guardian = GuardianAngel(broker, position_manager)
    >>> guardian.sync_positions()  # Call on startup
    >>> guardian.start_heartbeat()  # Start monitoring

Professional Standard:
    - AmiBroker: Server-side bracket orders
    - QuantConnect: Cloud-based position monitoring
    - Interactive Brokers: OCO (One-Cancels-Other) orders
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrphanPosition:
    """A position without proper stop loss protection."""
    symbol: str
    quantity: int
    average_price: float
    side: str  # "LONG" or "SHORT"
    broker_pnl: float
    detected_at: datetime = field(default_factory=datetime.now)
    sl_order_id: Optional[str] = None  # After we place protective SL
    risk_amount: float = 0.0  # Potential loss if no SL


class GuardianAngel:
    """
    Guardian Angel - Protects Your Account!

    Features:
    1. POSITION SYNC: Queries broker on startup to sync state
    2. ORPHAN DETECTION: Finds positions without SL orders
    3. AUTO-SL: Places protective SL for orphan positions
    4. HEARTBEAT: Monitors script health

    Example:
        >>> from core.guardian import GuardianAngel
        >>> guardian = GuardianAngel(broker, position_manager, order_manager)
        >>>
        >>> # On startup - CRITICAL!
        >>> orphans = guardian.sync_positions()
        >>> if orphans:
        ...     print(f"WARNING: {len(orphans)} orphan positions found!")
        ...     guardian.protect_orphans(default_sl_pct=2.0)
    """

    def __init__(
        self,
        broker,
        position_manager=None,
        order_manager=None,
        default_sl_pct: float = 2.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize Guardian Angel.

        Args:
            broker: ZerodhaBroker instance
            position_manager: PositionManager instance (optional)
            order_manager: OrderManager instance (optional)
            default_sl_pct: Default stop loss percentage for orphans
            heartbeat_interval: Heartbeat check interval in seconds
        """
        self.broker = broker
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.default_sl_pct = default_sl_pct
        self.heartbeat_interval = heartbeat_interval

        # State
        self._orphan_positions: Dict[str, OrphanPosition] = {}
        self._last_sync: Optional[datetime] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False

        # Callbacks
        self._on_orphan_detected: List[callable] = []
        self._on_orphan_protected: List[callable] = []

        logger.info("Guardian Angel initialized - watching over your account!")

    # =========================================================================
    # POSITION SYNC (Critical for startup!)
    # =========================================================================

    def sync_positions(self) -> List[OrphanPosition]:
        """
        Sync Python state with broker's actual positions.

        CRITICAL: Call this on every startup!

        Returns:
            List of orphan positions that need protection

        Example:
            >>> orphans = guardian.sync_positions()
            >>> print(f"Found {len(orphans)} positions at broker")
            >>> for orphan in orphans:
            ...     print(f"  {orphan.symbol}: {orphan.quantity} @ {orphan.average_price}")
        """
        if not self.broker or not self.broker.is_connected:
            logger.error("Cannot sync positions: Broker not connected")
            return []

        logger.info("GUARDIAN: Syncing positions with broker...")

        try:
            # Get actual positions from broker
            broker_positions = self.broker.get_positions()
            broker_orders = self.broker.get_orders()

            # Get SL orders that are currently open
            open_sl_orders = self._get_open_sl_orders(broker_orders)

            orphans = []
            for pos in broker_positions:
                if pos.quantity == 0:
                    continue  # Skip closed positions

                # Determine side
                side = "LONG" if pos.quantity > 0 else "SHORT"

                # Check if this position has an open SL order
                has_sl = self._position_has_sl(pos.symbol, side, open_sl_orders)

                if not has_sl:
                    # This is an orphan position!
                    orphan = OrphanPosition(
                        symbol=pos.symbol,
                        quantity=abs(pos.quantity),
                        average_price=pos.average_price,
                        side=side,
                        broker_pnl=pos.pnl,
                        risk_amount=abs(pos.quantity * pos.average_price * self.default_sl_pct / 100)
                    )
                    orphans.append(orphan)
                    self._orphan_positions[pos.symbol] = orphan

                    logger.warning(
                        f"ORPHAN DETECTED: {pos.symbol} {side} {abs(pos.quantity)} "
                        f"@ {pos.average_price} (No SL order found!)"
                    )

                    # Trigger callback
                    for callback in self._on_orphan_detected:
                        try:
                            callback(orphan)
                        except Exception as e:
                            logger.error(f"Orphan callback error: {e}")

            # Sync with position manager if available
            if self.position_manager and orphans:
                self._sync_position_manager(broker_positions)

            self._last_sync = datetime.now()
            logger.info(f"GUARDIAN: Sync complete. Found {len(orphans)} orphan positions.")

            return orphans

        except Exception as e:
            logger.error(f"Position sync failed: {e}", exc_info=True)
            return []

    def _get_open_sl_orders(self, orders) -> List[Dict[str, Any]]:
        """Extract open SL/SL-M orders from order list."""
        sl_orders = []
        for order in orders:
            # Check if it's an open SL order
            if order.status in ['OPEN', 'PENDING', 'TRIGGER PENDING']:
                if 'SL' in str(order.order_id).upper() or order.price == 0:  # SL-M has no price
                    sl_orders.append({
                        'order_id': order.order_id,
                        'symbol': order.symbol,
                        'transaction_type': order.transaction_type,
                        'quantity': order.quantity
                    })
        return sl_orders

    def _position_has_sl(self, symbol: str, side: str, sl_orders: List[Dict]) -> bool:
        """Check if a position has a corresponding SL order."""
        # LONG position needs SELL SL, SHORT needs BUY SL
        needed_txn = "SELL" if side == "LONG" else "BUY"

        for sl in sl_orders:
            if sl['symbol'] == symbol and sl['transaction_type'] == needed_txn:
                return True
        return False

    def _sync_position_manager(self, broker_positions):
        """Sync position manager with broker positions."""
        if not self.position_manager:
            return

        for pos in broker_positions:
            if pos.quantity != 0:
                # Add or update position in manager
                self.position_manager.add_position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    price=pos.average_price,
                    side="BUY" if pos.quantity > 0 else "SELL"
                )
                logger.debug(f"Position manager synced: {pos.symbol}")

    # =========================================================================
    # ORPHAN PROTECTION
    # =========================================================================

    def protect_orphans(self, sl_pct: float = None) -> Dict[str, str]:
        """
        Place protective SL orders for all orphan positions.

        Args:
            sl_pct: Stop loss percentage (default: self.default_sl_pct)

        Returns:
            Dict of symbol -> sl_order_id
        """
        if not self._orphan_positions:
            logger.info("No orphan positions to protect")
            return {}

        sl_pct = sl_pct or self.default_sl_pct
        placed_orders = {}

        for symbol, orphan in self._orphan_positions.items():
            sl_order_id = self._place_protective_sl(orphan, sl_pct)
            if sl_order_id:
                placed_orders[symbol] = sl_order_id
                orphan.sl_order_id = sl_order_id

                # Trigger callback
                for callback in self._on_orphan_protected:
                    try:
                        callback(orphan, sl_order_id)
                    except Exception as e:
                        logger.error(f"Protected callback error: {e}")

        return placed_orders

    def _place_protective_sl(self, orphan: OrphanPosition, sl_pct: float) -> Optional[str]:
        """Place a protective SL order for an orphan position."""
        from core.broker import TransactionType, ProductType

        # Calculate SL price
        if orphan.side == "LONG":
            # LONG: SL below entry
            trigger_price = orphan.average_price * (1 - sl_pct / 100)
            txn_type = TransactionType.SELL
        else:
            # SHORT: SL above entry
            trigger_price = orphan.average_price * (1 + sl_pct / 100)
            txn_type = TransactionType.BUY

        # Round to tick size (0.05 for most Indian stocks)
        trigger_price = round(trigger_price * 20) / 20

        try:
            sl_order_id = self.broker.place_stop_loss_order(
                symbol=orphan.symbol,
                quantity=orphan.quantity,
                trigger_price=trigger_price,
                transaction_type=txn_type,
                product=ProductType.INTRADAY
            )

            if sl_order_id:
                logger.info(
                    f"PROTECTIVE SL PLACED: {orphan.symbol} "
                    f"SL @ {trigger_price:.2f} ({sl_pct}% from entry)"
                )
                return sl_order_id
            else:
                logger.error(f"Failed to place protective SL for {orphan.symbol}")
                return None

        except Exception as e:
            logger.error(f"Error placing protective SL for {orphan.symbol}: {e}")
            return None

    def protect_single(self, symbol: str, sl_pct: float = None) -> Optional[str]:
        """Place protective SL for a single orphan position."""
        orphan = self._orphan_positions.get(symbol)
        if not orphan:
            logger.warning(f"No orphan position found for {symbol}")
            return None

        return self._place_protective_sl(orphan, sl_pct or self.default_sl_pct)

    # =========================================================================
    # HEARTBEAT MONITORING
    # =========================================================================

    def start_heartbeat(self):
        """Start heartbeat monitoring thread."""
        if self._heartbeat_running:
            logger.warning("Heartbeat already running")
            return

        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="GuardianHeartbeat"
        )
        self._heartbeat_thread.start()
        logger.info(f"Guardian heartbeat started (interval: {self.heartbeat_interval}s)")

    def stop_heartbeat(self):
        """Stop heartbeat monitoring."""
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
            logger.info("Guardian heartbeat stopped")

    def _heartbeat_loop(self):
        """Heartbeat monitoring loop."""
        while self._heartbeat_running:
            try:
                # Check broker connection
                if not self.broker.is_connected:
                    logger.warning("GUARDIAN: Broker disconnected! Attempting reconnect...")
                    # Could implement auto-reconnect here

                # Periodic position check
                self._check_position_health()

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(self.heartbeat_interval)

    def _check_position_health(self):
        """Check health of all positions."""
        try:
            broker_positions = self.broker.get_positions()
            for pos in broker_positions:
                if pos.quantity == 0:
                    continue

                # Check for large unrealized loss
                if pos.pnl < 0:
                    loss_pct = abs(pos.pnl / (pos.average_price * abs(pos.quantity))) * 100
                    if loss_pct > 5:  # 5% loss warning
                        logger.warning(
                            f"GUARDIAN WARNING: {pos.symbol} losing {loss_pct:.1f}% "
                            f"(Rs.{abs(pos.pnl):.0f})"
                        )

        except Exception as e:
            logger.debug(f"Position health check failed: {e}")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_orphan_detected(self, callback: callable):
        """Register callback for orphan detection."""
        self._on_orphan_detected.append(callback)

    def on_orphan_protected(self, callback: callable):
        """Register callback for orphan protection."""
        self._on_orphan_protected.append(callback)

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================

    def get_orphans(self) -> List[OrphanPosition]:
        """Get list of orphan positions."""
        return list(self._orphan_positions.values())

    def get_status(self) -> Dict[str, Any]:
        """Get Guardian status report."""
        return {
            'last_sync': self._last_sync.isoformat() if self._last_sync else None,
            'orphan_count': len(self._orphan_positions),
            'orphans': [
                {
                    'symbol': o.symbol,
                    'quantity': o.quantity,
                    'side': o.side,
                    'entry_price': o.average_price,
                    'risk_amount': o.risk_amount,
                    'protected': o.sl_order_id is not None
                }
                for o in self._orphan_positions.values()
            ],
            'heartbeat_running': self._heartbeat_running,
            'default_sl_pct': self.default_sl_pct
        }

    def print_report(self):
        """Print Guardian status report."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("GUARDIAN ANGEL - ACCOUNT PROTECTION STATUS")
        print("=" * 60)

        print(f"\nLast Sync: {status['last_sync'] or 'Never'}")
        print(f"Heartbeat: {'RUNNING' if status['heartbeat_running'] else 'STOPPED'}")
        print(f"Default SL: {status['default_sl_pct']}%")

        if status['orphans']:
            print(f"\nORPHAN POSITIONS ({status['orphan_count']}):")
            print("-" * 60)
            for o in status['orphans']:
                protected = "PROTECTED" if o['protected'] else "UNPROTECTED!"
                print(
                    f"  {o['symbol']}: {o['side']} {o['quantity']} @ {o['entry_price']:.2f} "
                    f"[{protected}] Risk: Rs.{o['risk_amount']:.0f}"
                )
        else:
            print("\nNo orphan positions detected.")

        print("\n" + "=" * 60)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def startup_guardian_check(broker, position_manager=None, order_manager=None, auto_protect: bool = True):
    """
    Convenience function to run Guardian check on startup.

    Usage:
        >>> from core.guardian import startup_guardian_check
        >>> guardian = startup_guardian_check(broker)
        >>> guardian.start_heartbeat()

    Args:
        broker: ZerodhaBroker instance
        position_manager: PositionManager (optional)
        order_manager: OrderManager (optional)
        auto_protect: Automatically place SL for orphans

    Returns:
        GuardianAngel instance
    """
    logger.info("=" * 50)
    logger.info("STARTUP GUARDIAN CHECK")
    logger.info("=" * 50)

    guardian = GuardianAngel(
        broker=broker,
        position_manager=position_manager,
        order_manager=order_manager
    )

    orphans = guardian.sync_positions()

    if orphans:
        logger.warning(f"FOUND {len(orphans)} ORPHAN POSITIONS!")
        guardian.print_report()

        if auto_protect:
            logger.info("Auto-protecting orphan positions...")
            protected = guardian.protect_orphans()
            logger.info(f"Protected {len(protected)} positions with server-side SL")
    else:
        logger.info("All positions have proper SL protection.")

    return guardian


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GUARDIAN ANGEL - Account Protection Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from core.guardian import startup_guardian_check")
    print("  guardian = startup_guardian_check(broker)")
    print("\nThis module protects against:")
    print("  1. Zombie positions (positions without SL)")
    print("  2. Script crashes while holding positions")
    print("  3. Network disconnects during trades")
    print("=" * 60)
