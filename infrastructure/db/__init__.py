from contextlib import contextmanager
import logging
from typing import Generator, Any, Dict, List
from functools import wraps
import time

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.extensions import connection as pg_conn
from config.config import settings
from infrastructure.cache import cache_result, invalidate_cache

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom database error"""
    pass

class ConnectionPool:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, min_conn: int = 1, max_conn: int = 10) -> None:
        if self._pool is None:
            try:
                self._pool = SimpleConnectionPool(
                    minconn=min_conn,
                    maxconn=max_conn,
                    dsn=settings.DATABASE_URL,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise DatabaseError(f"Connection pool initialization failed: {str(e)}")

    def get_connection(self) -> pg_conn:
        if self._pool is None:
            self.initialize()
        return self._pool.getconn()

    def return_connection(self, conn: pg_conn) -> None:
        self._pool.putconn(conn)

    def close_all(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed")

pool = ConnectionPool()

def db_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for database operation retries"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise DatabaseError(f"Operation failed after {max_retries} retries: {str(e)}")
            raise last_error
        return wrapper
    return decorator

@contextmanager
def get_db_connection() -> Generator[pg_conn, None, None]:
    """Get a database connection from the pool with automatic commit/rollback"""
    conn = None
    try:
        conn = pool.get_connection()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"Database operation failed: {str(e)}") from e
    finally:
        if conn:
            pool.return_connection(conn)

@db_retry()
@cache_result(prefix="query", ttl=300)  # Cache for 5 minutes
def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute a query and return results with caching"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall() if cur.description else []

@db_retry()
def execute_write_query(query: str, params: tuple = None) -> None:
    """Execute a write query and invalidate relevant caches"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            # Invalidate cache based on table name
            table_name = query.split()[2].strip().lower()
            invalidate_cache(f"query:*{table_name}*")

@db_retry()
def execute_batch_query(query: str, params_list: List[tuple]) -> None:
    """Execute a batch query and invalidate relevant caches"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, query, params_list)
            # Invalidate cache based on table name
            table_name = query.split()[2].strip().lower()
            invalidate_cache(f"query:*{table_name}*")

def init_db() -> None:
    """Initialize database connection pool"""
    pool.initialize()

def close_db_pool() -> None:
    """Close all database connections"""
    pool.close_all()
