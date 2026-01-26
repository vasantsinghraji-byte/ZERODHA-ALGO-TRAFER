# -*- coding: utf-8 -*-
"""
Core Service Interfaces and Implementations
Defines the key services that the application depends on
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from kiteconnect import KiteConnect
from zerodha_trader.core.config import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# SERVICE INTERFACES (Abstract Base Classes)
# =============================================================================

class IMarketDataService(ABC):
    """Interface for market data operations"""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to market data feed"""
        pass

    @abstractmethod
    def subscribe(self, instrument_tokens: List[int]) -> None:
        """Subscribe to instrument updates"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from market data feed"""
        pass


class IBrokerService(ABC):
    """Interface for broker operations (orders, positions, etc.)"""

    @abstractmethod
    def place_order(self, **kwargs) -> str:
        """Place an order and return order ID"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get order history"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order"""
        pass


class IDataStorageService(ABC):
    """Interface for data persistence"""

    @abstractmethod
    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Save trade data"""
        pass

    @abstractmethod
    def save_signal(self, signal_data: Dict[str, Any]) -> None:
        """Save signal data"""
        pass

    @abstractmethod
    def get_trade_history(self, **filters) -> List[Dict[str, Any]]:
        """Retrieve trade history"""
        pass


class INotificationService(ABC):
    """Interface for notifications (Telegram, email, etc.)"""

    @abstractmethod
    def send_notification(self, message: str, level: str = "info") -> None:
        """Send a notification"""
        pass


# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

class ZerodhaMarketDataService(IMarketDataService):
    """Zerodha implementation of market data service"""

    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kws = None
        logger.info("ZerodhaMarketDataService initialized")

    def connect(self) -> None:
        """Establish WebSocket connection"""
        from kiteconnect import KiteTicker
        self.kws = KiteTicker(self.api_key, self.access_token)
        logger.info("Market data connection established")

    def subscribe(self, instrument_tokens: List[int]) -> None:
        """Subscribe to instruments"""
        if self.kws:
            self.kws.subscribe(instrument_tokens)
            self.kws.set_mode(self.kws.MODE_FULL, instrument_tokens)
            logger.info(f"Subscribed to {len(instrument_tokens)} instruments")

    def disconnect(self) -> None:
        """Disconnect WebSocket"""
        if self.kws:
            self.kws.close()
            logger.info("Market data connection closed")


class ZerodhaBrokerService(IBrokerService):
    """Zerodha implementation of broker service"""

    def __init__(self, kite: KiteConnect):
        self.kite = kite
        logger.info("ZerodhaBrokerService initialized")

    def place_order(self, **kwargs) -> str:
        """Place order via Kite API"""
        order_id = self.kite.place_order(**kwargs)
        logger.info(f"Order placed: {order_id}")
        return order_id

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        positions = self.kite.positions()
        return positions.get('net', [])

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return self.kite.orders()

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order"""
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        logger.info(f"Order cancelled: {order_id}")


class SQLiteDataStorageService(IDataStorageService):
    """SQLite implementation of data storage"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
        logger.info(f"SQLiteDataStorageService initialized: {db_path}")

    def _init_database(self) -> None:
        """Initialize database schema"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                status TEXT NOT NULL,
                strategy TEXT,
                metadata TEXT
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                target REAL,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Save trade to database"""
        import sqlite3
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (timestamp, symbol, side, quantity, entry_price,
                              exit_price, pnl, status, strategy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get('timestamp'),
            trade_data.get('symbol'),
            trade_data.get('side'),
            trade_data.get('quantity'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('pnl'),
            trade_data.get('status'),
            trade_data.get('strategy'),
            json.dumps(trade_data.get('metadata', {}))
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Trade saved: {trade_data.get('symbol')}")

    def save_signal(self, signal_data: Dict[str, Any]) -> None:
        """Save signal to database"""
        import sqlite3
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (timestamp, symbol, signal_type, confidence,
                               entry_price, stop_loss, target, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data.get('timestamp'),
            signal_data.get('symbol'),
            signal_data.get('signal_type'),
            signal_data.get('confidence'),
            signal_data.get('entry_price'),
            signal_data.get('stop_loss'),
            signal_data.get('target'),
            json.dumps(signal_data.get('metadata', {}))
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Signal saved: {signal_data.get('symbol')}")

    def get_trade_history(self, **filters) -> List[Dict[str, Any]]:
        """Retrieve trade history with optional filters"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM trades"
        params = []

        # Build WHERE clause from filters
        where_clauses = []
        if 'symbol' in filters:
            where_clauses.append("symbol = ?")
            params.append(filters['symbol'])
        if 'status' in filters:
            where_clauses.append("status = ?")
            params.append(filters['status'])

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY timestamp DESC"

        if 'limit' in filters:
            query += f" LIMIT {int(filters['limit'])}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


class ConsoleNotificationService(INotificationService):
    """Console-based notification service (for development)"""

    def __init__(self):
        logger.info("ConsoleNotificationService initialized")

    def send_notification(self, message: str, level: str = "info") -> None:
        """Print notification to console"""
        prefix = {
            'info': '9',
            'warning': '�',
            'error': 'L',
            'success': ''
        }.get(level, '9')

        print(f"{prefix} {message}")
        logger.log(
            logging.INFO if level == 'info' else
            logging.WARNING if level == 'warning' else
            logging.ERROR if level == 'error' else
            logging.INFO,
            message
        )


class TelegramNotificationService(INotificationService):
    """Telegram-based notification service (placeholder)"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

        if self.enabled:
            logger.info("TelegramNotificationService initialized (enabled)")
        else:
            logger.info("TelegramNotificationService initialized (disabled - missing credentials)")

    def send_notification(self, message: str, level: str = "info") -> None:
        """Send notification via Telegram"""
        if not self.enabled:
            logger.debug(f"Telegram disabled, skipping: {message}")
            return

        # TODO: Implement actual Telegram API call
        # For now, just log
        logger.info(f"[TELEGRAM] {level.upper()}: {message}")
        # import requests
        # url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        # requests.post(url, json={'chat_id': self.chat_id, 'text': message})
