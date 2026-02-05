# -*- coding: utf-8 -*-
"""
Database System (SQLite)
========================
Simple database to store your trading history!

Stores:
- Your trades (buys and sells)
- Price history
- Strategy results
- Settings
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# BRITTLE PATH FIX: Use robust path resolution
from utils.paths import find_project_root

# Database location - using robust path resolution
BASE_DIR = find_project_root()
DB_PATH = BASE_DIR / "data" / "trading.db"

_db_connection: Optional[sqlite3.Connection] = None


@dataclass
class Trade:
    """A trade record"""
    id: Optional[int] = None
    symbol: str = ""
    action: str = ""  # BUY or SELL
    quantity: int = 0
    price: float = 0.0
    total_value: float = 0.0
    strategy: str = ""
    timestamp: str = ""
    status: str = "OPEN"
    notes: str = ""


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize database with all tables.

    Args:
        db_path: Custom database path (optional)

    Returns:
        Database connection
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            strategy TEXT,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            profit_loss REAL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target REAL NOT NULL,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Price history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            UNIQUE(symbol, timestamp)
        )
    ''')

    # Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Daily P&L table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            starting_balance REAL NOT NULL,
            ending_balance REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            total_trades INTEGER NOT NULL
        )
    ''')

    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Get database connection (singleton)"""
    global _db_connection

    if _db_connection is None:
        _db_connection = init_db()

    return _db_connection


@contextmanager
def db_session():
    """Context manager for database operations"""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e


def close_db():
    """Close database connection"""
    global _db_connection

    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None


# ============== TRADE FUNCTIONS ==============

def add_trade(trade: Trade) -> int:
    """
    Record a trade.

    Args:
        trade: Trade object

    Returns:
        Trade ID
    """
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, strategy, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.timestamp or datetime.now().isoformat(),
            trade.symbol,
            trade.action,
            trade.quantity,
            trade.price,
            trade.total_value or (trade.quantity * trade.price),
            trade.strategy,
            trade.status,
            trade.notes
        ))
        return cursor.lastrowid


def get_trades(limit: int = 100) -> List[Dict]:
    """Get recent trades"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_todays_pnl() -> float:
    """Calculate today's P&L"""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT SUM(CASE WHEN action = 'SELL' THEN total_value ELSE -total_value END) as pnl
        FROM trades WHERE timestamp LIKE ?
    ''', (f"{today}%",))

    row = cursor.fetchone()
    return row['pnl'] if row and row['pnl'] else 0.0


# ============== STATS ==============

def get_trading_stats() -> Dict[str, Any]:
    """Get trading statistics"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total FROM trades')
    total = cursor.fetchone()['total']

    cursor.execute('SELECT SUM(profit_loss) as pnl FROM trades WHERE profit_loss IS NOT NULL')
    row = cursor.fetchone()
    total_pnl = row['pnl'] if row and row['pnl'] else 0

    return {
        'total_trades': total,
        'total_pnl': total_pnl,
        'todays_pnl': get_todays_pnl()
    }
