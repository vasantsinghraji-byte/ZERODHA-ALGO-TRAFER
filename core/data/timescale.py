# -*- coding: utf-8 -*-
"""
TimescaleDB Adapter - Blazing Fast Time-Series Storage!
========================================================
PostgreSQL extension optimized for time-series data.

Why TimescaleDB over SQLite?
- 100-1000x faster queries on time-series data
- Automatic time-based partitioning (hypertables)
- 90-95% compression
- Continuous aggregates (real-time OHLCV from ticks)
- Built-in retention policies

Components:
- TimescaleConnection: Connection pooling and management
- TickRepository: Store and query tick data
- BarRepository: Store and query OHLCV bars
- SchemaManager: Create hypertables and aggregates

Example:
    >>> from core.data import TimescaleConnection, TickRepository
    >>>
    >>> # Connect to TimescaleDB
    >>> conn = TimescaleConnection(
    ...     host='localhost',
    ...     database='trading',
    ...     user='trader'
    ... )
    >>>
    >>> # Store ticks
    >>> repo = TickRepository(conn)
    >>> repo.insert_ticks(ticks)
    >>>
    >>> # Query with time range
    >>> ticks = repo.get_ticks('RELIANCE', start, end)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Iterator, Tuple
from enum import Enum
import logging
from contextlib import contextmanager

# Optional imports - graceful degradation if not installed
try:
    import psycopg2
    from psycopg2 import pool, sql, extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None
    sql = None
    extras = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes for OHLCV bars."""
    TICK = "tick"
    SEC_1 = "1s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


# Timeframe to PostgreSQL interval mapping
TIMEFRAME_INTERVALS = {
    Timeframe.SEC_1: "1 second",
    Timeframe.MIN_1: "1 minute",
    Timeframe.MIN_5: "5 minutes",
    Timeframe.MIN_15: "15 minutes",
    Timeframe.MIN_30: "30 minutes",
    Timeframe.HOUR_1: "1 hour",
    Timeframe.HOUR_4: "4 hours",
    Timeframe.DAY_1: "1 day",
    Timeframe.WEEK_1: "1 week",
    Timeframe.MONTH_1: "1 month",
}


@dataclass
class TimescaleConfig:
    """Configuration for TimescaleDB connection."""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading"
    user: str = "postgres"
    password: str = ""

    # Connection pool settings
    min_connections: int = 2
    max_connections: int = 10

    # Hypertable settings
    chunk_time_interval: str = "1 day"  # Partition by day
    compression_after: str = "7 days"   # Compress after 7 days
    retention_period: str = "365 days"  # Keep 1 year of data

    # Performance settings
    batch_size: int = 10000  # Rows per batch insert


@dataclass
class Tick:
    """Single tick data point."""
    timestamp: datetime
    symbol: str
    ltp: float  # Last traded price
    volume: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_qty: int = 0
    ask_qty: int = 0
    oi: int = 0  # Open interest


@dataclass
class Bar:
    """OHLCV bar data point."""
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    trades: int = 0  # Number of trades in bar
    vwap: Optional[float] = None  # Volume-weighted average price


class TimescaleConnection:
    """
    Connection pool manager for TimescaleDB.

    Manages a pool of PostgreSQL connections for efficient
    database access without connection overhead.
    """

    def __init__(self, config: Optional[TimescaleConfig] = None):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for TimescaleDB. "
                "Install with: pip install psycopg2-binary"
            )

        self.config = config or TimescaleConfig()
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._connected = False

    def connect(self) -> bool:
        """Initialize connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )
            self._connected = True
            logger.info(f"Connected to TimescaleDB at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False

    def disconnect(self):
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            self._connected = False
            logger.info("Disconnected from TimescaleDB")

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected and self._pool is not None

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database. Call connect() first.")

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Get a cursor from a pooled connection."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute(self, query: str, params: tuple = None) -> int:
        """Execute a query and return row count."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def fetch_one(self, query: str, params: tuple = None) -> Optional[tuple]:
        """Execute query and fetch one row."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute query and fetch all rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def check_timescale_extension(self) -> bool:
        """Check if TimescaleDB extension is installed."""
        try:
            result = self.fetch_one(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            if result:
                logger.info(f"TimescaleDB version: {result[0]}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check TimescaleDB extension: {e}")
            return False


class SchemaManager:
    """
    Manages database schema for TimescaleDB.

    Creates hypertables, indexes, and continuous aggregates
    for optimal time-series performance.
    """

    # SQL for creating the ticks hypertable
    CREATE_TICKS_TABLE = """
    CREATE TABLE IF NOT EXISTS market_ticks (
        time        TIMESTAMPTZ NOT NULL,
        symbol      TEXT NOT NULL,
        ltp         DECIMAL(12,2) NOT NULL,
        volume      BIGINT DEFAULT 0,
        bid         DECIMAL(12,2),
        ask         DECIMAL(12,2),
        bid_qty     INTEGER DEFAULT 0,
        ask_qty     INTEGER DEFAULT 0,
        oi          BIGINT DEFAULT 0
    );
    """

    # SQL for creating the bars hypertable
    CREATE_BARS_TABLE = """
    CREATE TABLE IF NOT EXISTS market_bars (
        time        TIMESTAMPTZ NOT NULL,
        symbol      TEXT NOT NULL,
        timeframe   TEXT NOT NULL,
        open        DECIMAL(12,2) NOT NULL,
        high        DECIMAL(12,2) NOT NULL,
        low         DECIMAL(12,2) NOT NULL,
        close       DECIMAL(12,2) NOT NULL,
        volume      BIGINT DEFAULT 0,
        trades      INTEGER DEFAULT 0,
        vwap        DECIMAL(12,2)
    );
    """

    # SQL for creating the orders table
    CREATE_ORDERS_TABLE = """
    CREATE TABLE IF NOT EXISTS orders (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        order_type      TEXT NOT NULL,
        quantity        INTEGER NOT NULL,
        price           DECIMAL(12,2),
        status          TEXT NOT NULL,
        filled_qty      INTEGER DEFAULT 0,
        avg_price       DECIMAL(12,2),
        strategy_id     TEXT,
        broker_order_id TEXT
    );
    """

    # SQL for creating trades table
    CREATE_TRADES_TABLE = """
    CREATE TABLE IF NOT EXISTS trades (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        quantity        INTEGER NOT NULL,
        price           DECIMAL(12,2) NOT NULL,
        order_id        UUID REFERENCES orders(id),
        strategy_id     TEXT,
        pnl             DECIMAL(12,2),
        fees            DECIMAL(12,2) DEFAULT 0
    );
    """

    def __init__(self, connection: TimescaleConnection):
        self.conn = connection
        self.config = connection.config

    def initialize_schema(self) -> bool:
        """
        Create all tables, hypertables, and indexes.

        Returns True if successful.
        """
        try:
            # Enable TimescaleDB extension
            self._enable_timescale()

            # Create tables
            self._create_tables()

            # Convert to hypertables
            self._create_hypertables()

            # Create indexes
            self._create_indexes()

            # Setup compression
            self._setup_compression()

            # Setup retention
            self._setup_retention()

            logger.info("Database schema initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

    def _enable_timescale(self):
        """Enable TimescaleDB extension if not already enabled."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        logger.info("TimescaleDB extension enabled")

    def _create_tables(self):
        """Create all required tables."""
        self.conn.execute(self.CREATE_TICKS_TABLE)
        self.conn.execute(self.CREATE_BARS_TABLE)
        self.conn.execute(self.CREATE_ORDERS_TABLE)
        self.conn.execute(self.CREATE_TRADES_TABLE)
        logger.info("Tables created")

    def _create_hypertables(self):
        """Convert tables to hypertables for time-series optimization."""
        # Check if already hypertables
        for table in ['market_ticks', 'market_bars']:
            result = self.conn.fetch_one(
                """
                SELECT hypertable_name FROM timescaledb_information.hypertables
                WHERE hypertable_name = %s
                """,
                (table,)
            )

            if not result:
                self.conn.execute(
                    f"""
                    SELECT create_hypertable(
                        '{table}',
                        'time',
                        chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                        if_not_exists => TRUE
                    );
                    """
                )
                logger.info(f"Created hypertable: {table}")

    def _create_indexes(self):
        """Create indexes for common query patterns."""
        indexes = [
            # Ticks indexes
            ("idx_ticks_symbol_time", "market_ticks", "symbol, time DESC"),
            ("idx_ticks_time", "market_ticks", "time DESC"),

            # Bars indexes
            ("idx_bars_symbol_tf_time", "market_bars", "symbol, timeframe, time DESC"),
            ("idx_bars_time", "market_bars", "time DESC"),

            # Orders indexes
            ("idx_orders_symbol", "orders", "symbol"),
            ("idx_orders_status", "orders", "status"),
            ("idx_orders_strategy", "orders", "strategy_id"),

            # Trades indexes
            ("idx_trades_symbol", "trades", "symbol"),
            ("idx_trades_strategy", "trades", "strategy_id"),
        ]

        for idx_name, table, columns in indexes:
            try:
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns});"
                )
            except Exception as e:
                # Index may already exist with different definition
                logger.debug(f"Index {idx_name} creation skipped: {e}")

        logger.info("Indexes created")

    def _setup_compression(self):
        """Enable compression on hypertables."""
        for table in ['market_ticks', 'market_bars']:
            try:
                # Enable compression
                self.conn.execute(
                    f"""
                    ALTER TABLE {table} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol'
                    );
                    """
                )

                # Add compression policy
                self.conn.execute(
                    f"""
                    SELECT add_compression_policy(
                        '{table}',
                        INTERVAL '{self.config.compression_after}',
                        if_not_exists => TRUE
                    );
                    """
                )
            except Exception as e:
                logger.debug(f"Compression setup for {table}: {e}")

        logger.info("Compression policies configured")

    def _setup_retention(self):
        """Setup data retention policies."""
        for table in ['market_ticks', 'market_bars']:
            try:
                self.conn.execute(
                    f"""
                    SELECT add_retention_policy(
                        '{table}',
                        INTERVAL '{self.config.retention_period}',
                        if_not_exists => TRUE
                    );
                    """
                )
            except Exception as e:
                logger.debug(f"Retention setup for {table}: {e}")

        logger.info("Retention policies configured")

    def create_continuous_aggregate(
        self,
        name: str,
        timeframe: Timeframe,
        refresh_interval: str = "1 minute"
    ) -> bool:
        """
        Create a continuous aggregate for real-time OHLCV from ticks.

        Continuous aggregates automatically maintain OHLCV bars
        as new ticks arrive - no manual aggregation needed!
        """
        interval = TIMEFRAME_INTERVALS.get(timeframe, "1 minute")

        try:
            # Drop existing if exists
            self.conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {name} CASCADE;")

            # Create continuous aggregate
            self.conn.execute(
                f"""
                CREATE MATERIALIZED VIEW {name}
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('{interval}', time) AS bucket,
                    symbol,
                    first(ltp, time) AS open,
                    max(ltp) AS high,
                    min(ltp) AS low,
                    last(ltp, time) AS close,
                    sum(volume) AS volume,
                    count(*) AS trades
                FROM market_ticks
                GROUP BY bucket, symbol
                WITH NO DATA;
                """
            )

            # Add refresh policy
            self.conn.execute(
                f"""
                SELECT add_continuous_aggregate_policy(
                    '{name}',
                    start_offset => INTERVAL '1 hour',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '{refresh_interval}',
                    if_not_exists => TRUE
                );
                """
            )

            logger.info(f"Created continuous aggregate: {name} ({timeframe.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to create continuous aggregate {name}: {e}")
            return False

    def create_default_aggregates(self) -> bool:
        """Create standard OHLCV aggregates for common timeframes."""
        aggregates = [
            ("ohlcv_1min", Timeframe.MIN_1, "1 minute"),
            ("ohlcv_5min", Timeframe.MIN_5, "1 minute"),
            ("ohlcv_15min", Timeframe.MIN_15, "5 minutes"),
            ("ohlcv_1hour", Timeframe.HOUR_1, "5 minutes"),
            ("ohlcv_1day", Timeframe.DAY_1, "1 hour"),
        ]

        success = True
        for name, timeframe, refresh in aggregates:
            if not self.create_continuous_aggregate(name, timeframe, refresh):
                success = False

        return success


class TickRepository:
    """
    Repository for tick data operations.

    Provides efficient methods for inserting and querying
    high-frequency tick data.
    """

    def __init__(self, connection: TimescaleConnection):
        self.conn = connection
        self.batch_size = connection.config.batch_size

    def insert_tick(self, tick: Tick) -> bool:
        """Insert a single tick."""
        try:
            self.conn.execute(
                """
                INSERT INTO market_ticks
                (time, symbol, ltp, volume, bid, ask, bid_qty, ask_qty, oi)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    tick.timestamp, tick.symbol, tick.ltp, tick.volume,
                    tick.bid, tick.ask, tick.bid_qty, tick.ask_qty, tick.oi
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert tick: {e}")
            return False

    def insert_ticks(self, ticks: List[Tick]) -> int:
        """
        Batch insert multiple ticks efficiently.

        Uses COPY for maximum performance on large batches.
        Returns number of rows inserted.
        """
        if not ticks:
            return 0

        try:
            with self.conn.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Prepare data
                    data = [
                        (
                            t.timestamp, t.symbol, t.ltp, t.volume,
                            t.bid, t.ask, t.bid_qty, t.ask_qty, t.oi
                        )
                        for t in ticks
                    ]

                    # Use execute_values for fast batch insert
                    extras.execute_values(
                        cursor,
                        """
                        INSERT INTO market_ticks
                        (time, symbol, ltp, volume, bid, ask, bid_qty, ask_qty, oi)
                        VALUES %s
                        """,
                        data,
                        page_size=self.batch_size
                    )

                    return len(ticks)

        except Exception as e:
            logger.error(f"Failed to insert ticks batch: {e}")
            return 0

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Tick]:
        """Query ticks for a symbol within time range."""
        query = """
            SELECT time, symbol, ltp, volume, bid, ask, bid_qty, ask_qty, oi
            FROM market_ticks
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = self.conn.fetch_all(query, (symbol, start, end))

        return [
            Tick(
                timestamp=row[0],
                symbol=row[1],
                ltp=float(row[2]),
                volume=row[3],
                bid=float(row[4]) if row[4] else None,
                ask=float(row[5]) if row[5] else None,
                bid_qty=row[6],
                ask_qty=row[7],
                oi=row[8]
            )
            for row in rows
        ]

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get most recent tick for a symbol."""
        row = self.conn.fetch_one(
            """
            SELECT time, symbol, ltp, volume, bid, ask, bid_qty, ask_qty, oi
            FROM market_ticks
            WHERE symbol = %s
            ORDER BY time DESC
            LIMIT 1
            """,
            (symbol,)
        )

        if row:
            return Tick(
                timestamp=row[0],
                symbol=row[1],
                ltp=float(row[2]),
                volume=row[3],
                bid=float(row[4]) if row[4] else None,
                ask=float(row[5]) if row[5] else None,
                bid_qty=row[6],
                ask_qty=row[7],
                oi=row[8]
            )
        return None

    def get_tick_count(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> int:
        """Count ticks matching criteria."""
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        if start:
            conditions.append("time >= %s")
            params.append(start)
        if end:
            conditions.append("time <= %s")
            params.append(end)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT COUNT(*) FROM market_ticks WHERE {where_clause}"

        result = self.conn.fetch_one(query, tuple(params))
        return result[0] if result else 0

    def get_ticks_dataframe(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ):
        """Get ticks as a pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame operations")

        ticks = self.get_ticks(symbol, start, end)
        return pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'ltp': t.ltp,
                'volume': t.volume,
                'bid': t.bid,
                'ask': t.ask,
            }
            for t in ticks
        ])


class BarRepository:
    """
    Repository for OHLCV bar data operations.

    Supports multiple timeframes and efficient batch operations.
    """

    def __init__(self, connection: TimescaleConnection):
        self.conn = connection
        self.batch_size = connection.config.batch_size

    def insert_bar(self, bar: Bar) -> bool:
        """Insert a single OHLCV bar."""
        try:
            self.conn.execute(
                """
                INSERT INTO market_bars
                (time, symbol, timeframe, open, high, low, close, volume, trades, vwap)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    bar.timestamp, bar.symbol, bar.timeframe,
                    bar.open, bar.high, bar.low, bar.close,
                    bar.volume, bar.trades, bar.vwap
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert bar: {e}")
            return False

    def insert_bars(self, bars: List[Bar]) -> int:
        """Batch insert multiple bars efficiently."""
        if not bars:
            return 0

        try:
            with self.conn.get_connection() as conn:
                with conn.cursor() as cursor:
                    data = [
                        (
                            b.timestamp, b.symbol, b.timeframe,
                            b.open, b.high, b.low, b.close,
                            b.volume, b.trades, b.vwap
                        )
                        for b in bars
                    ]

                    extras.execute_values(
                        cursor,
                        """
                        INSERT INTO market_bars
                        (time, symbol, timeframe, open, high, low, close, volume, trades, vwap)
                        VALUES %s
                        """,
                        data,
                        page_size=self.batch_size
                    )

                    return len(bars)

        except Exception as e:
            logger.error(f"Failed to insert bars batch: {e}")
            return 0

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> List[Bar]:
        """Query bars for a symbol and timeframe within time range."""
        query = """
            SELECT time, symbol, timeframe, open, high, low, close, volume, trades, vwap
            FROM market_bars
            WHERE symbol = %s AND timeframe = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = self.conn.fetch_all(query, (symbol, timeframe, start, end))

        return [
            Bar(
                timestamp=row[0],
                symbol=row[1],
                timeframe=row[2],
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=row[7],
                trades=row[8],
                vwap=float(row[9]) if row[9] else None
            )
            for row in rows
        ]

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """Get most recent bar for a symbol and timeframe."""
        row = self.conn.fetch_one(
            """
            SELECT time, symbol, timeframe, open, high, low, close, volume, trades, vwap
            FROM market_bars
            WHERE symbol = %s AND timeframe = %s
            ORDER BY time DESC
            LIMIT 1
            """,
            (symbol, timeframe)
        )

        if row:
            return Bar(
                timestamp=row[0],
                symbol=row[1],
                timeframe=row[2],
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=row[7],
                trades=row[8],
                vwap=float(row[9]) if row[9] else None
            )
        return None

    def get_bars_from_aggregate(
        self,
        symbol: str,
        aggregate_name: str,
        start: datetime,
        end: datetime
    ) -> List[Bar]:
        """
        Query bars from a continuous aggregate.

        Use this to get real-time OHLCV derived from ticks.
        """
        rows = self.conn.fetch_all(
            f"""
            SELECT bucket, symbol, open, high, low, close, volume, trades
            FROM {aggregate_name}
            WHERE symbol = %s AND bucket >= %s AND bucket <= %s
            ORDER BY bucket ASC
            """,
            (symbol, start, end)
        )

        # Extract timeframe from aggregate name
        timeframe = aggregate_name.replace("ohlcv_", "")

        return [
            Bar(
                timestamp=row[0],
                symbol=row[1],
                timeframe=timeframe,
                open=float(row[2]),
                high=float(row[3]),
                low=float(row[4]),
                close=float(row[5]),
                volume=row[6],
                trades=row[7]
            )
            for row in rows
        ]

    def get_bars_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ):
        """Get bars as a pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame operations")

        bars = self.get_bars(symbol, timeframe, start, end)
        return pd.DataFrame([
            {
                'timestamp': b.timestamp,
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume,
            }
            for b in bars
        ])


class QueryBuilder:
    """
    Query builder for common time-series operations.

    Provides helper methods for constructing efficient
    TimescaleDB-specific queries.
    """

    def __init__(self, connection: TimescaleConnection):
        self.conn = connection

    def time_bucket_query(
        self,
        table: str,
        interval: str,
        symbol: str,
        start: datetime,
        end: datetime,
        aggregations: Dict[str, str]
    ) -> List[tuple]:
        """
        Execute a time_bucket aggregation query.

        Args:
            table: Source table name
            interval: Bucket interval (e.g., '5 minutes')
            symbol: Symbol to filter
            start: Start time
            end: End time
            aggregations: Dict of column_name -> aggregation_function
                         e.g., {'avg_price': 'AVG(ltp)', 'max_price': 'MAX(ltp)'}
        """
        agg_clauses = ", ".join(
            f"{func} AS {name}" for name, func in aggregations.items()
        )

        query = f"""
            SELECT time_bucket('{interval}', time) AS bucket, {agg_clauses}
            FROM {table}
            WHERE symbol = %s AND time >= %s AND time <= %s
            GROUP BY bucket
            ORDER BY bucket ASC
        """

        return self.conn.fetch_all(query, (symbol, start, end))

    def get_symbols(self, table: str = "market_ticks") -> List[str]:
        """Get list of all symbols in table."""
        rows = self.conn.fetch_all(
            f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
        )
        return [row[0] for row in rows]

    def get_date_range(
        self,
        table: str,
        symbol: Optional[str] = None
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get min and max timestamps in table."""
        if symbol:
            row = self.conn.fetch_one(
                f"SELECT MIN(time), MAX(time) FROM {table} WHERE symbol = %s",
                (symbol,)
            )
        else:
            row = self.conn.fetch_one(
                f"SELECT MIN(time), MAX(time) FROM {table}"
            )

        return (row[0], row[1]) if row else (None, None)

    def get_chunk_info(self, table: str) -> List[dict]:
        """Get information about hypertable chunks."""
        rows = self.conn.fetch_all(
            """
            SELECT chunk_schema, chunk_name, range_start, range_end,
                   pg_size_pretty(pg_total_relation_size(format('%I.%I', chunk_schema, chunk_name)))
            FROM timescaledb_information.chunks
            WHERE hypertable_name = %s
            ORDER BY range_start DESC
            """,
            (table,)
        )

        return [
            {
                'schema': row[0],
                'name': row[1],
                'start': row[2],
                'end': row[3],
                'size': row[4]
            }
            for row in rows
        ]

    def get_compression_stats(self, table: str) -> dict:
        """Get compression statistics for hypertable."""
        row = self.conn.fetch_one(
            """
            SELECT
                pg_size_pretty(before_compression_total_bytes) AS before,
                pg_size_pretty(after_compression_total_bytes) AS after,
                ROUND((1 - after_compression_total_bytes::FLOAT / before_compression_total_bytes) * 100, 1) AS ratio
            FROM hypertable_compression_stats(%s)
            """,
            (table,)
        )

        if row:
            return {
                'before': row[0],
                'after': row[1],
                'compression_ratio': row[2]
            }
        return {}


# Convenience functions
def create_timescale_connection(
    host: str = "localhost",
    port: int = 5432,
    database: str = "trading",
    user: str = "postgres",
    password: str = ""
) -> TimescaleConnection:
    """Create and connect to TimescaleDB."""
    config = TimescaleConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    conn = TimescaleConnection(config)
    conn.connect()
    return conn


def initialize_database(connection: TimescaleConnection) -> bool:
    """Initialize database schema with hypertables and aggregates."""
    schema = SchemaManager(connection)
    if schema.initialize_schema():
        return schema.create_default_aggregates()
    return False


# Check for required dependencies
TIMESCALE_AVAILABLE = PSYCOPG2_AVAILABLE
