import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from config.config import settings

logger = logging.getLogger(__name__)

def init_db():
    """Initialize database connection and verify connectivity."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                logger.info(f"Connected to PostgreSQL: {version}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = psycopg2.connect(
            settings.DATABASE_URL,
            cursor_factory=RealDictCursor
        )
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
