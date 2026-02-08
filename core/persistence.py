# -*- coding: utf-8 -*-
"""
Persistence Manager - State Recovery After Restart!
====================================================
Saves trading state to SQLite database so nothing is lost on crash/restart.

CRITICAL SAFETY FEATURE:
Without persistence, a restart means:
- Forgotten positions -> Double buying -> Doubled risk
- Reset daily P&L -> Bypass max loss limits -> Account blowup
- Lost order tracking -> Orphan orders at broker

This module prevents "trading amnesia" by persisting:
- Open positions (with stop-loss, target, strategy)
- Closed positions (for P&L history)
- Daily trading stats (P&L, trade count)
- Pending orders (for live mode recovery)
"""

import logging
import sqlite3
import threading
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    SQLite-based persistence for trading state.

    Thread-safe: Uses connection pooling and transactions.

    Usage:
        pm = PersistenceManager("trading_state.db")
        pm.save_position(position_dict)
        positions = pm.load_positions()
    """

    def __init__(self, db_path: str = "data/trading_state.db"):
        """
        Initialize persistence manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"PersistenceManager initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            isolation_level='IMMEDIATE'
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create tables if they don't exist."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Positions table (open positions)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        exchange TEXT DEFAULT 'NSE',
                        side TEXT DEFAULT 'LONG',
                        quantity INTEGER NOT NULL,
                        buy_quantity INTEGER DEFAULT 0,
                        sell_quantity INTEGER DEFAULT 0,
                        average_price REAL NOT NULL,
                        last_price REAL DEFAULT 0,
                        buy_value REAL DEFAULT 0,
                        current_value REAL DEFAULT 0,
                        realized_pnl REAL DEFAULT 0,
                        stop_loss REAL DEFAULT 0,
                        target REAL DEFAULT 0,
                        strategy TEXT DEFAULT '',
                        opened_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Closed positions table (historical)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS closed_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        exchange TEXT DEFAULT 'NSE',
                        side TEXT DEFAULT 'LONG',
                        quantity INTEGER NOT NULL,
                        average_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        realized_pnl REAL NOT NULL,
                        strategy TEXT DEFAULT '',
                        opened_at TEXT NOT NULL,
                        closed_at TEXT NOT NULL
                    )
                """)

                # Daily stats table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        date TEXT PRIMARY KEY,
                        daily_pnl REAL DEFAULT 0,
                        daily_trades INTEGER DEFAULT 0,
                        start_capital REAL DEFAULT 0,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Pending orders table (for live mode recovery)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_orders (
                        order_id TEXT PRIMARY KEY,
                        broker_order_id TEXT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL DEFAULT 0,
                        stop_loss REAL DEFAULT 0,
                        target REAL DEFAULT 0,
                        strategy TEXT DEFAULT '',
                        status TEXT DEFAULT 'PLACED',
                        created_at TEXT NOT NULL
                    )
                """)

                # Create indexes for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_closed_positions_date
                    ON closed_positions(closed_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_closed_positions_symbol
                    ON closed_positions(symbol)
                """)

                logger.info("Database tables initialized")

    # ============== POSITIONS ==============

    def save_position(self, position: Dict[str, Any]):
        """
        Save or update a position.

        Args:
            position: Position dict with symbol, quantity, average_price, etc.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO positions (
                        symbol, exchange, side, quantity, buy_quantity, sell_quantity,
                        average_price, last_price, buy_value, current_value, realized_pnl,
                        stop_loss, target, strategy, opened_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position['symbol'],
                    position.get('exchange', 'NSE'),
                    position.get('side', 'LONG'),
                    position['quantity'],
                    position.get('buy_quantity', 0),
                    position.get('sell_quantity', 0),
                    position['average_price'],
                    position.get('last_price', 0),
                    position.get('buy_value', 0),
                    position.get('current_value', 0),
                    position.get('realized_pnl', 0),
                    position.get('stop_loss', 0),
                    position.get('target', 0),
                    position.get('strategy', ''),
                    position.get('opened_at', datetime.now().isoformat()),
                    datetime.now().isoformat()
                ))
                logger.debug(f"Position saved: {position['symbol']}")

    def delete_position(self, symbol: str):
        """Delete a position (when closed)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                logger.debug(f"Position deleted: {symbol}")

    def load_positions(self) -> List[Dict[str, Any]]:
        """
        Load all open positions from database.

        Returns:
            List of position dicts
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM positions")
                rows = cursor.fetchall()

                positions = []
                for row in rows:
                    positions.append({
                        'symbol': row['symbol'],
                        'exchange': row['exchange'],
                        'side': row['side'],
                        'quantity': row['quantity'],
                        'buy_quantity': row['buy_quantity'],
                        'sell_quantity': row['sell_quantity'],
                        'average_price': row['average_price'],
                        'last_price': row['last_price'],
                        'buy_value': row['buy_value'],
                        'current_value': row['current_value'],
                        'realized_pnl': row['realized_pnl'],
                        'stop_loss': row['stop_loss'],
                        'target': row['target'],
                        'strategy': row['strategy'],
                        'opened_at': row['opened_at'],
                        'updated_at': row['updated_at']
                    })

                logger.info(f"Loaded {len(positions)} positions from database")
                return positions

    def clear_positions(self):
        """Clear all positions (use carefully!)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM positions")
                logger.warning("All positions cleared from database")

    # ============== CLOSED POSITIONS ==============

    def save_closed_position(self, position: Dict[str, Any], exit_price: float):
        """
        Save a closed position to history.

        Args:
            position: The closed position dict
            exit_price: Exit price
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO closed_positions (
                        symbol, exchange, side, quantity, average_price, exit_price,
                        realized_pnl, strategy, opened_at, closed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position['symbol'],
                    position.get('exchange', 'NSE'),
                    position.get('side', 'LONG'),
                    position.get('buy_quantity', position['quantity']),
                    position['average_price'],
                    exit_price,
                    position.get('realized_pnl', 0),
                    position.get('strategy', ''),
                    position.get('opened_at', datetime.now().isoformat()),
                    datetime.now().isoformat()
                ))
                logger.debug(f"Closed position saved: {position['symbol']}")

    def get_closed_positions_today(self) -> List[Dict[str, Any]]:
        """Get all positions closed today."""
        today = date.today().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM closed_positions
                    WHERE date(closed_at) = ?
                    ORDER BY closed_at DESC
                """, (today,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    # ============== DAILY STATS ==============

    def save_daily_stats(
        self,
        daily_pnl: float,
        daily_trades: int,
        start_capital: float = 0
    ):
        """
        Save daily trading statistics.

        Args:
            daily_pnl: Today's P&L
            daily_trades: Number of trades today
            start_capital: Starting capital for loss % calculation
        """
        today = date.today().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_stats (
                        date, daily_pnl, daily_trades, start_capital, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    today,
                    float(daily_pnl),
                    daily_trades,
                    float(start_capital),
                    datetime.now().isoformat()
                ))
                logger.debug(f"Daily stats saved: P&L={daily_pnl}, Trades={daily_trades}")

    def load_daily_stats(self) -> Optional[Dict[str, Any]]:
        """
        Load today's trading statistics.

        Returns:
            Dict with daily_pnl, daily_trades, start_capital or None
        """
        today = date.today().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM daily_stats WHERE date = ?",
                    (today,)
                )
                row = cursor.fetchone()

                if row:
                    stats = {
                        'date': row['date'],
                        'daily_pnl': row['daily_pnl'],
                        'daily_trades': row['daily_trades'],
                        'start_capital': row['start_capital'],
                        'updated_at': row['updated_at']
                    }
                    logger.info(f"Loaded daily stats: P&L={stats['daily_pnl']}, Trades={stats['daily_trades']}")
                    return stats

                logger.info("No daily stats found for today (new trading day)")
                return None

    # ============== PENDING ORDERS ==============

    def save_pending_order(self, order: Dict[str, Any]):
        """Save a pending order for recovery after restart."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO pending_orders (
                        order_id, broker_order_id, symbol, side, quantity,
                        price, stop_loss, target, strategy, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order['order_id'],
                    order.get('broker_order_id', ''),
                    order['symbol'],
                    order['side'],
                    order['quantity'],
                    order.get('price', 0),
                    order.get('stop_loss', 0),
                    order.get('target', 0),
                    order.get('strategy', ''),
                    order.get('status', 'PLACED'),
                    order.get('created_at', datetime.now().isoformat())
                ))
                logger.debug(f"Pending order saved: {order['order_id']}")

    def delete_pending_order(self, order_id: str):
        """Delete pending order (when filled or cancelled)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM pending_orders WHERE order_id = ?",
                    (order_id,)
                )
                logger.debug(f"Pending order deleted: {order_id}")

    def load_pending_orders(self) -> List[Dict[str, Any]]:
        """Load all pending orders for recovery."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pending_orders")
                rows = cursor.fetchall()
                orders = [dict(row) for row in rows]
                if orders:
                    logger.info(f"Loaded {len(orders)} pending orders for recovery")
                return orders

    def clear_pending_orders(self):
        """Clear all pending orders."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pending_orders")

    # ============== UTILITY ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM positions")
                open_positions = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM closed_positions")
                total_closed = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM pending_orders")
                pending_orders = cursor.fetchone()[0]

                return {
                    'open_positions': open_positions,
                    'total_closed_positions': total_closed,
                    'pending_orders': pending_orders,
                    'db_path': str(self.db_path)
                }


# ============== SINGLETON INSTANCE ==============

_persistence_manager: Optional[PersistenceManager] = None


def get_persistence_manager(db_path: str = "data/trading_state.db") -> PersistenceManager:
    """
    Get or create the singleton PersistenceManager.

    Args:
        db_path: Database path (only used on first call)

    Returns:
        PersistenceManager instance
    """
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager(db_path)
    return _persistence_manager


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("PERSISTENCE MANAGER - Test")
    print("=" * 50)

    # Create manager with test database
    pm = PersistenceManager("data/test_trading_state.db")

    # Test position save/load
    test_position = {
        'symbol': 'RELIANCE',
        'quantity': 10,
        'average_price': 2500.0,
        'stop_loss': 2450.0,
        'target': 2600.0,
        'strategy': 'turtle',
        'opened_at': datetime.now().isoformat()
    }

    print("\n1. Saving position...")
    pm.save_position(test_position)

    print("\n2. Loading positions...")
    positions = pm.load_positions()
    print(f"   Found {len(positions)} positions")
    for p in positions:
        print(f"   - {p['symbol']}: {p['quantity']}x @ Rs.{p['average_price']}")

    print("\n3. Saving daily stats...")
    pm.save_daily_stats(daily_pnl=-500.0, daily_trades=3, start_capital=100000)

    print("\n4. Loading daily stats...")
    stats = pm.load_daily_stats()
    if stats:
        print(f"   P&L: Rs.{stats['daily_pnl']}, Trades: {stats['daily_trades']}")

    print("\n5. Database stats...")
    db_stats = pm.get_stats()
    print(f"   {db_stats}")

    print("\n6. Cleaning up test position...")
    pm.delete_position('RELIANCE')

    print("\n" + "=" * 50)
    print("Persistence Manager ready!")
    print("=" * 50)
