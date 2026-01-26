# -*- coding: utf-8 -*-
"""
Database management - Single source of truth
No global state, proper connection lifecycle
"""
import sqlite3
from contextlib import contextmanager
from typing import Optional
from .config import get_settings

_db_connection: Optional[sqlite3.Connection] = None

def init_db() -> sqlite3.Connection:
    """Initialize database with schema"""
    settings = get_settings()
    db_path = settings.data_dir / 'algotrader.db'

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target REAL NOT NULL,
            quantity INTEGER NOT NULL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            profit_loss REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

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
            risk_reward REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
