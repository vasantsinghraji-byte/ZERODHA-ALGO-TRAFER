# -*- coding: utf-8 -*-
"""
Data Migration - SQLite to TimescaleDB Migration Tool
======================================================
Migrate your historical data from SQLite to TimescaleDB.

Features:
- Batch migration with progress tracking
- Data validation (row counts, checksums)
- Rollback support (backup before migration)
- Resume interrupted migrations
- Dry-run mode for testing

Example:
    >>> from core.data import MigrationManager, MigrationConfig
    >>>
    >>> # Configure migration
    >>> config = MigrationConfig(
    ...     sqlite_path='data/historical.db',
    ...     timescale_host='localhost',
    ...     batch_size=10000
    ... )
    >>>
    >>> # Run migration
    >>> manager = MigrationManager(config)
    >>> result = manager.migrate_all()
    >>> print(f"Migrated {result.rows_migrated} rows")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Iterator
from enum import Enum
from pathlib import Path
import logging
import json
import hashlib
import shutil

# SQLite is always available in Python
import sqlite3

# Optional imports
try:
    from .timescale import (
        TimescaleConnection,
        TimescaleConfig,
        TickRepository,
        BarRepository,
        Tick,
        Bar,
        SchemaManager,
    )
    TIMESCALE_IMPORT_OK = True
except ImportError:
    TIMESCALE_IMPORT_OK = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Status of a migration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationType(Enum):
    """Type of data being migrated."""
    TICKS = "ticks"
    BARS = "bars"
    ORDERS = "orders"
    TRADES = "trades"


@dataclass
class MigrationConfig:
    """Configuration for data migration."""
    # SQLite source
    sqlite_path: str = "data/historical.db"

    # TimescaleDB target
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_database: str = "trading"
    timescale_user: str = "postgres"
    timescale_password: str = ""

    # Migration settings
    batch_size: int = 10000        # Rows per batch
    validate_data: bool = True      # Validate after migration
    create_backup: bool = True      # Backup SQLite before migration
    dry_run: bool = False           # Don't write, just validate

    # Progress tracking
    checkpoint_file: str = "data/migration_checkpoint.json"
    log_file: str = "data/migration.log"

    # Performance
    parallel_tables: bool = False   # Migrate tables in parallel


@dataclass
class ValidationResult:
    """Result of data validation."""
    table: str
    source_count: int
    target_count: int
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    sample_mismatches: List[dict] = field(default_factory=list)

    @property
    def count_match(self) -> bool:
        """Check if row counts match."""
        return self.source_count == self.target_count

    @property
    def missing_rows(self) -> int:
        """Number of rows missing in target."""
        return max(0, self.source_count - self.target_count)


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    migration_type: MigrationType
    status: MigrationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Counts
    rows_migrated: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0

    # Validation
    validation: Optional[ValidationResult] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[timedelta]:
        """Duration of migration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def rows_per_second(self) -> float:
        """Migration speed in rows per second."""
        if self.duration and self.duration.total_seconds() > 0:
            return self.rows_migrated / self.duration.total_seconds()
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'migration_type': self.migration_type.value,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'rows_migrated': self.rows_migrated,
            'rows_skipped': self.rows_skipped,
            'rows_failed': self.rows_failed,
            'errors': self.errors,
        }


@dataclass
class MigrationCheckpoint:
    """Checkpoint for resuming interrupted migrations."""
    table: str
    last_timestamp: Optional[datetime] = None
    last_id: Optional[int] = None
    rows_processed: int = 0
    status: MigrationStatus = MigrationStatus.PENDING

    def to_dict(self) -> dict:
        return {
            'table': self.table,
            'last_timestamp': self.last_timestamp.isoformat() if self.last_timestamp else None,
            'last_id': self.last_id,
            'rows_processed': self.rows_processed,
            'status': self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MigrationCheckpoint':
        return cls(
            table=data['table'],
            last_timestamp=datetime.fromisoformat(data['last_timestamp']) if data.get('last_timestamp') else None,
            last_id=data.get('last_id'),
            rows_processed=data.get('rows_processed', 0),
            status=MigrationStatus(data.get('status', 'pending')),
        )


class SQLiteReader:
    """
    Reader for SQLite database.

    Reads data in batches for efficient migration.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False

    def disconnect(self):
        """Close SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_tables(self) -> List[str]:
        """Get list of tables in database."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_row_count(self, table: str) -> int:
        """Get row count for a table."""
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def get_date_range(self, table: str, time_column: str = 'timestamp') -> tuple:
        """Get min and max timestamps for a table."""
        try:
            cursor = self._conn.execute(
                f"SELECT MIN({time_column}), MAX({time_column}) FROM {table}"
            )
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (None, None)
        except Exception:
            return (None, None)

    def read_batches(
        self,
        table: str,
        batch_size: int = 10000,
        time_column: str = 'timestamp',
        start_after: Optional[datetime] = None
    ) -> Iterator[List[dict]]:
        """
        Read table in batches.

        Yields batches of rows as dictionaries.
        """
        offset = 0
        where_clause = ""

        if start_after:
            where_clause = f"WHERE {time_column} > '{start_after.isoformat()}'"

        while True:
            query = f"""
                SELECT * FROM {table}
                {where_clause}
                ORDER BY {time_column}
                LIMIT {batch_size} OFFSET {offset}
            """

            cursor = self._conn.execute(query)
            rows = cursor.fetchall()

            if not rows:
                break

            # Convert to list of dicts
            batch = [dict(row) for row in rows]
            yield batch

            offset += batch_size

            if len(rows) < batch_size:
                break

    def calculate_checksum(self, table: str, time_column: str = 'timestamp') -> str:
        """Calculate a checksum for data validation."""
        query = f"""
            SELECT COUNT(*),
                   MIN({time_column}),
                   MAX({time_column})
            FROM {table}
        """
        cursor = self._conn.execute(query)
        row = cursor.fetchone()

        # Create checksum from aggregate values
        data = f"{row[0]}:{row[1]}:{row[2]}"
        return hashlib.md5(data.encode()).hexdigest()


class MigrationManager:
    """
    Manages the migration from SQLite to TimescaleDB.

    Handles:
    - Batch migration with progress tracking
    - Data validation
    - Rollback support
    - Resume from checkpoint
    """

    # Mapping of SQLite tables to TimescaleDB tables
    TABLE_MAPPING = {
        'ticks': 'market_ticks',
        'market_ticks': 'market_ticks',
        'bars': 'market_bars',
        'market_bars': 'market_bars',
        'ohlcv': 'market_bars',
        'historical': 'market_bars',
        'orders': 'orders',
        'trades': 'trades',
    }

    def __init__(self, config: Optional[MigrationConfig] = None):
        if not TIMESCALE_IMPORT_OK:
            raise ImportError(
                "TimescaleDB module not available. "
                "Ensure psycopg2 is installed: pip install psycopg2-binary"
            )

        self.config = config or MigrationConfig()
        self.sqlite: Optional[SQLiteReader] = None
        self.timescale: Optional[TimescaleConnection] = None
        self.checkpoints: Dict[str, MigrationCheckpoint] = {}
        self.results: List[MigrationResult] = []

        # Progress callback
        self._progress_callback: Optional[Callable[[str, int, int], None]] = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]):
        """Set callback for progress updates: callback(table, current, total)."""
        self._progress_callback = callback

    def connect(self) -> bool:
        """Connect to both SQLite and TimescaleDB."""
        # Connect to SQLite
        self.sqlite = SQLiteReader(self.config.sqlite_path)
        if not self.sqlite.connect():
            return False

        # Connect to TimescaleDB
        ts_config = TimescaleConfig(
            host=self.config.timescale_host,
            port=self.config.timescale_port,
            database=self.config.timescale_database,
            user=self.config.timescale_user,
            password=self.config.timescale_password,
        )
        self.timescale = TimescaleConnection(ts_config)
        if not self.timescale.connect():
            self.sqlite.disconnect()
            return False

        # Load checkpoints
        self._load_checkpoints()

        return True

    def disconnect(self):
        """Disconnect from both databases."""
        if self.sqlite:
            self.sqlite.disconnect()
        if self.timescale:
            self.timescale.disconnect()

    def create_backup(self) -> Optional[str]:
        """
        Create backup of SQLite database.

        Returns backup file path or None if failed.
        """
        if not self.config.create_backup:
            return None

        try:
            source = Path(self.config.sqlite_path)
            if not source.exists():
                logger.warning(f"SQLite file not found: {source}")
                return None

            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = source.parent / f"{source.stem}_backup_{timestamp}{source.suffix}"

            shutil.copy2(source, backup_path)
            logger.info(f"Created backup: {backup_path}")

            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def migrate_all(self) -> List[MigrationResult]:
        """
        Migrate all tables from SQLite to TimescaleDB.

        Returns list of migration results.
        """
        if not self.sqlite or not self.timescale:
            raise RuntimeError("Not connected. Call connect() first.")

        # Create backup
        if self.config.create_backup:
            backup_path = self.create_backup()
            if backup_path:
                logger.info(f"Backup created: {backup_path}")

        # Initialize TimescaleDB schema
        schema = SchemaManager(self.timescale)
        schema.initialize_schema()

        # Get SQLite tables
        sqlite_tables = self.sqlite.get_tables()
        logger.info(f"Found {len(sqlite_tables)} tables in SQLite")

        results = []

        for table in sqlite_tables:
            # Check if we have a mapping for this table
            target_table = self.TABLE_MAPPING.get(table.lower())
            if not target_table:
                logger.warning(f"No mapping for table: {table}, skipping")
                continue

            # Determine migration type
            if 'tick' in table.lower():
                migration_type = MigrationType.TICKS
            elif 'bar' in table.lower() or 'ohlcv' in table.lower() or 'historical' in table.lower():
                migration_type = MigrationType.BARS
            elif 'order' in table.lower():
                migration_type = MigrationType.ORDERS
            elif 'trade' in table.lower():
                migration_type = MigrationType.TRADES
            else:
                migration_type = MigrationType.BARS  # Default

            # Run migration
            result = self._migrate_table(table, target_table, migration_type)
            results.append(result)

        self.results = results
        self._save_checkpoints()

        return results

    def migrate_ticks(self, source_table: str = 'ticks') -> MigrationResult:
        """Migrate tick data."""
        return self._migrate_table(source_table, 'market_ticks', MigrationType.TICKS)

    def migrate_bars(self, source_table: str = 'bars') -> MigrationResult:
        """Migrate OHLCV bar data."""
        return self._migrate_table(source_table, 'market_bars', MigrationType.BARS)

    def _migrate_table(
        self,
        source_table: str,
        target_table: str,
        migration_type: MigrationType
    ) -> MigrationResult:
        """Migrate a single table."""
        result = MigrationResult(
            migration_type=migration_type,
            status=MigrationStatus.IN_PROGRESS,
            started_at=datetime.now()
        )

        try:
            # Get checkpoint for resume
            checkpoint = self.checkpoints.get(source_table)
            start_after = checkpoint.last_timestamp if checkpoint else None

            # Get total row count
            total_rows = self.sqlite.get_row_count(source_table)
            logger.info(f"Migrating {source_table} -> {target_table}: {total_rows} rows")

            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would migrate {total_rows} rows")
                result.rows_migrated = total_rows
                result.status = MigrationStatus.COMPLETED
                result.completed_at = datetime.now()
                return result

            # Determine time column
            time_column = self._detect_time_column(source_table)

            # Migrate in batches
            rows_processed = checkpoint.rows_processed if checkpoint else 0

            for batch in self.sqlite.read_batches(
                source_table,
                batch_size=self.config.batch_size,
                time_column=time_column,
                start_after=start_after
            ):
                try:
                    # Convert and insert batch
                    if migration_type == MigrationType.TICKS:
                        inserted = self._insert_ticks_batch(batch)
                    elif migration_type == MigrationType.BARS:
                        inserted = self._insert_bars_batch(batch)
                    else:
                        inserted = self._insert_generic_batch(batch, target_table)

                    rows_processed += inserted
                    result.rows_migrated += inserted

                    # Update checkpoint
                    if batch:
                        last_row = batch[-1]
                        self.checkpoints[source_table] = MigrationCheckpoint(
                            table=source_table,
                            last_timestamp=self._parse_timestamp(last_row.get(time_column)),
                            rows_processed=rows_processed,
                            status=MigrationStatus.IN_PROGRESS
                        )

                    # Progress callback
                    if self._progress_callback:
                        self._progress_callback(source_table, rows_processed, total_rows)

                except Exception as e:
                    logger.error(f"Batch error: {e}")
                    result.rows_failed += len(batch)
                    result.errors.append(str(e))

            # Validate if enabled
            if self.config.validate_data:
                result.validation = self._validate_table(source_table, target_table, time_column)
                if not result.validation.is_valid:
                    result.errors.extend(result.validation.errors)

            result.status = MigrationStatus.COMPLETED
            result.completed_at = datetime.now()

            # Update checkpoint
            self.checkpoints[source_table] = MigrationCheckpoint(
                table=source_table,
                rows_processed=rows_processed,
                status=MigrationStatus.COMPLETED
            )

            logger.info(
                f"Completed {source_table}: {result.rows_migrated} rows in "
                f"{result.duration.total_seconds():.1f}s "
                f"({result.rows_per_second:.0f} rows/sec)"
            )

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now()
            logger.error(f"Migration failed for {source_table}: {e}")

        return result

    def _detect_time_column(self, table: str) -> str:
        """Detect the time/timestamp column name."""
        # Common column names for timestamps
        candidates = ['timestamp', 'time', 'datetime', 'date', 'ts', 'created_at']

        cursor = self.sqlite._conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1].lower() for row in cursor.fetchall()]

        for candidate in candidates:
            if candidate in columns:
                return candidate

        # Default to first column if no match
        return columns[0] if columns else 'timestamp'

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(value)
        return None

    def _insert_ticks_batch(self, batch: List[dict]) -> int:
        """Insert batch of ticks into TimescaleDB."""
        repo = TickRepository(self.timescale)

        ticks = []
        for row in batch:
            tick = Tick(
                timestamp=self._parse_timestamp(
                    row.get('timestamp') or row.get('time') or row.get('datetime')
                ) or datetime.now(),
                symbol=row.get('symbol', 'UNKNOWN'),
                ltp=float(row.get('ltp') or row.get('close') or row.get('price') or 0),
                volume=int(row.get('volume', 0)),
                bid=float(row['bid']) if row.get('bid') else None,
                ask=float(row['ask']) if row.get('ask') else None,
                bid_qty=int(row.get('bid_qty', 0)),
                ask_qty=int(row.get('ask_qty', 0)),
                oi=int(row.get('oi') or row.get('open_interest') or 0),
            )
            ticks.append(tick)

        return repo.insert_ticks(ticks)

    def _insert_bars_batch(self, batch: List[dict]) -> int:
        """Insert batch of OHLCV bars into TimescaleDB."""
        repo = BarRepository(self.timescale)

        bars = []
        for row in batch:
            bar = Bar(
                timestamp=self._parse_timestamp(
                    row.get('timestamp') or row.get('time') or row.get('datetime') or row.get('date')
                ) or datetime.now(),
                symbol=row.get('symbol', 'UNKNOWN'),
                timeframe=row.get('timeframe') or row.get('interval') or '1d',
                open=float(row.get('open', 0)),
                high=float(row.get('high', 0)),
                low=float(row.get('low', 0)),
                close=float(row.get('close', 0)),
                volume=int(row.get('volume', 0)),
                trades=int(row.get('trades', 0)),
                vwap=float(row['vwap']) if row.get('vwap') else None,
            )
            bars.append(bar)

        return repo.insert_bars(bars)

    def _insert_generic_batch(self, batch: List[dict], table: str) -> int:
        """Insert batch using generic INSERT."""
        if not batch:
            return 0

        # Get column names from first row
        columns = list(batch[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join(columns)

        query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

        count = 0
        with self.timescale.get_cursor() as cursor:
            for row in batch:
                try:
                    values = tuple(row.get(col) for col in columns)
                    cursor.execute(query, values)
                    count += 1
                except Exception as e:
                    logger.debug(f"Row insert failed: {e}")

        return count

    def _validate_table(
        self,
        source_table: str,
        target_table: str,
        time_column: str
    ) -> ValidationResult:
        """Validate migrated data."""
        result = ValidationResult(
            table=source_table,
            source_count=self.sqlite.get_row_count(source_table),
            target_count=0,
            is_valid=False
        )

        try:
            # Get target count
            row = self.timescale.fetch_one(
                f"SELECT COUNT(*) FROM {target_table}"
            )
            result.target_count = row[0] if row else 0

            # Check row counts
            if result.source_count != result.target_count:
                result.errors.append(
                    f"Row count mismatch: source={result.source_count}, "
                    f"target={result.target_count}"
                )

            # Sample validation - compare random rows
            # (Could add more sophisticated validation here)

            result.is_valid = result.count_match and len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Validation error: {e}")

        return result

    def rollback(self, backup_path: str) -> bool:
        """
        Rollback migration by restoring SQLite backup.

        Note: This only restores SQLite. To rollback TimescaleDB,
        you would need to TRUNCATE the target tables.
        """
        try:
            if not Path(backup_path).exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Disconnect from SQLite
            if self.sqlite:
                self.sqlite.disconnect()

            # Restore backup
            shutil.copy2(backup_path, self.config.sqlite_path)
            logger.info(f"Restored SQLite from backup: {backup_path}")

            # Reconnect
            if self.sqlite:
                self.sqlite.connect()

            # Clear checkpoints
            self.checkpoints.clear()
            self._save_checkpoints()

            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def truncate_target(self, table: str = None) -> bool:
        """
        Truncate TimescaleDB tables (for rollback).

        If table is None, truncates all migration target tables.
        """
        try:
            tables = [table] if table else ['market_ticks', 'market_bars']

            for t in tables:
                self.timescale.execute(f"TRUNCATE TABLE {t}")
                logger.info(f"Truncated table: {t}")

            return True

        except Exception as e:
            logger.error(f"Truncate failed: {e}")
            return False

    def _load_checkpoints(self):
        """Load checkpoints from file."""
        try:
            path = Path(self.config.checkpoint_file)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    self.checkpoints = {
                        k: MigrationCheckpoint.from_dict(v)
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints")
        except Exception as e:
            logger.debug(f"No checkpoints loaded: {e}")

    def _save_checkpoints(self):
        """Save checkpoints to file."""
        try:
            path = Path(self.config.checkpoint_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                data = {k: v.to_dict() for k, v in self.checkpoints.items()}
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}")

    def get_migration_report(self) -> dict:
        """Generate a summary report of all migrations."""
        return {
            'total_tables': len(self.results),
            'completed': sum(1 for r in self.results if r.status == MigrationStatus.COMPLETED),
            'failed': sum(1 for r in self.results if r.status == MigrationStatus.FAILED),
            'total_rows': sum(r.rows_migrated for r in self.results),
            'total_errors': sum(len(r.errors) for r in self.results),
            'results': [r.to_dict() for r in self.results],
        }


# Convenience functions
def migrate_sqlite_to_timescale(
    sqlite_path: str,
    timescale_host: str = "localhost",
    timescale_database: str = "trading",
    timescale_user: str = "postgres",
    timescale_password: str = "",
    dry_run: bool = False
) -> dict:
    """
    Quick migration from SQLite to TimescaleDB.

    Returns migration report.
    """
    config = MigrationConfig(
        sqlite_path=sqlite_path,
        timescale_host=timescale_host,
        timescale_database=timescale_database,
        timescale_user=timescale_user,
        timescale_password=timescale_password,
        dry_run=dry_run,
    )

    manager = MigrationManager(config)

    if not manager.connect():
        return {'error': 'Failed to connect to databases'}

    try:
        manager.migrate_all()
        return manager.get_migration_report()
    finally:
        manager.disconnect()


def validate_migration(
    sqlite_path: str,
    timescale_connection: 'TimescaleConnection'
) -> List[ValidationResult]:
    """
    Validate that SQLite data was migrated correctly.

    Returns list of validation results for each table.
    """
    reader = SQLiteReader(sqlite_path)
    if not reader.connect():
        return []

    results = []
    table_mapping = {
        'ticks': 'market_ticks',
        'bars': 'market_bars',
    }

    try:
        for source, target in table_mapping.items():
            if source not in reader.get_tables():
                continue

            source_count = reader.get_row_count(source)
            target_count_row = timescale_connection.fetch_one(
                f"SELECT COUNT(*) FROM {target}"
            )
            target_count = target_count_row[0] if target_count_row else 0

            result = ValidationResult(
                table=source,
                source_count=source_count,
                target_count=target_count,
                is_valid=source_count == target_count,
            )

            if not result.is_valid:
                result.errors.append(
                    f"Count mismatch: SQLite={source_count}, TimescaleDB={target_count}"
                )

            results.append(result)

    finally:
        reader.disconnect()

    return results
