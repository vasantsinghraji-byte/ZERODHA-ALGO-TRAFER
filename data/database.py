"""
Database Connection and Session Management
"""

from contextlib import contextmanager
from typing import Generator, Optional, Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from config.loader import get_config
from config.logging_config import get_logger

logger = get_logger(__name__)


class Database:
    """Database connection manager"""

    def __init__(self):
        self.config = get_config()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialize()

    def _initialize(self):
        """Initialize database connection"""
        db_config = self.config.database.postgres

        # Build connection URL using SQLAlchemy URL object
        # This prevents password from being exposed in logs, stack traces,
        # or string representations of the connection URL
        connection_url = URL.create(
            drivername="postgresql",
            username=db_config.user,
            password=db_config.password,  # Password is stored securely, not in URL string
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
        )

        # Create engine
        # Note: hide_parameters=True prevents password from appearing in logging
        self.engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            echo=db_config.echo,
            pool_pre_ping=True,  # Verify connections before using
            hide_parameters=True,  # Prevent sensitive data in logs
        )

        # Add connection event listeners
        @event.listens_for(self.engine, "connect")
        def set_search_path(dbapi_conn, connection_record):
            """Set search path for all connections"""
            cursor = dbapi_conn.cursor()
            cursor.execute("SET search_path TO public, market_data, trading, analytics")
            cursor.close()

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Log connection info without sensitive data (no username/password)
        logger.info(
            f"Database initialized: {db_config.host}:{db_config.port}/{db_config.database} "
            f"(pool_size={db_config.pool_size})"
        )

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations

        Usage:
            with db.session_scope() as session:
                user = session.query(User).first()
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_raw(self, query: str, params: Optional[dict] = None) -> Any:
        """
        Execute parameterized SQL query safely.

        SECURITY: Uses SQLAlchemy's text() with bound parameters to prevent
        SQL injection. All dynamic values MUST be passed via the params dict
        using :param_name placeholders — never concatenate values into the query.

        Args:
            query: SQL query with :param_name placeholders (no string interpolation!)
            params: Dictionary of parameter values (required if query has placeholders)

        Returns:
            Query result

        Raises:
            RuntimeError: If engine not initialized

        Example:
            # CORRECT - parameterized query
            db.execute_raw("SELECT * FROM users WHERE id = :user_id", {"user_id": 123})

            # WRONG - SQL injection vulnerability
            db.execute_raw(f"SELECT * FROM users WHERE id = {user_id}")
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized")

        # Log query execution for audit trail (without parameter values for security)
        logger.debug(f"Executing raw SQL: {query[:100]}{'...' if len(query) > 100 else ''}")

        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result

    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database instance
_db_instance = None


def get_db() -> Database:
    """Get the global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def get_session() -> Session:
    """Get a new database session (convenience function)"""
    return get_db().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Get a transactional session scope (convenience function)"""
    with get_db().session_scope() as session:
        yield session
