# -*- coding: utf-8 -*-
"""
Strategy State Manager - Remember Everything!
==============================================
Persist strategy state across restarts, crashes, and sessions.

The Problem:
- Strategy crashes mid-trade -> lose track of positions
- Restart the engine -> indicators need to recalculate from scratch
- Multiple sessions -> can't share state

The Solution:
- Save state to SQLite database
- Automatic serialization of complex objects
- Load state on startup -> resume where you left off

Example:
    >>> state_mgr = StrategyStateManager("trading.db")
    >>> state_mgr.save_state("turtle", {"positions": [...], "indicators": {...}})
    >>> # Later, after restart...
    >>> state = state_mgr.load_state("turtle")
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
import pickle
import base64

logger = logging.getLogger(__name__)


@dataclass
class StrategyState:
    """
    Represents the complete state of a strategy.

    Attributes:
        strategy_name: Name of the strategy
        version: State version for migration support
        positions: Current open positions
        indicators: Calculated indicator values
        parameters: Strategy parameters
        bars: Recent bar history (for indicator warmup)
        signals: Recent signals generated
        metadata: Additional metadata
        timestamp: When the state was saved
    """
    strategy_name: str
    version: int = 1
    positions: Dict[str, Any] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    bars: Dict[str, List[Dict]] = field(default_factory=dict)  # symbol -> list of bars
    signals: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyState':
        """Create from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class StateSerializer:
    """
    Handles serialization of complex Python objects.

    Supports:
    - JSON for simple types (dict, list, str, int, float, bool, None)
    - Pickle (base64 encoded) for complex types (DataFrames, numpy arrays)
    """

    @staticmethod
    def serialize(obj: Any) -> str:
        """
        Serialize an object to a string.

        Args:
            obj: Any Python object

        Returns:
            JSON string (for simple types) or base64-encoded pickle
        """
        try:
            # Try JSON first (faster, human-readable)
            return json.dumps({'type': 'json', 'data': obj})
        except (TypeError, ValueError):
            # Fall back to pickle for complex types
            pickled = pickle.dumps(obj)
            encoded = base64.b64encode(pickled).decode('utf-8')
            return json.dumps({'type': 'pickle', 'data': encoded})

    @staticmethod
    def deserialize(data: str) -> Any:
        """
        Deserialize a string back to a Python object.

        Args:
            data: Serialized string

        Returns:
            Original Python object
        """
        parsed = json.loads(data)
        if parsed['type'] == 'json':
            return parsed['data']
        elif parsed['type'] == 'pickle':
            decoded = base64.b64decode(parsed['data'].encode('utf-8'))
            return pickle.loads(decoded)
        else:
            raise ValueError(f"Unknown serialization type: {parsed['type']}")


class StrategyStateManager:
    """
    Manages persistence of strategy state to SQLite database.

    Features:
    - Save/load complete strategy state
    - Automatic versioning
    - Thread-safe operations
    - State history for debugging
    - Cleanup of old states

    Example:
        >>> mgr = StrategyStateManager("data/trading.db")
        >>>
        >>> # Save state
        >>> state = StrategyState(
        ...     strategy_name="turtle",
        ...     positions={"RELIANCE": {"qty": 10, "price": 2500}},
        ...     indicators={"sma_20": 2480, "sma_50": 2420}
        ... )
        >>> mgr.save_state(state)
        >>>
        >>> # Load state
        >>> loaded = mgr.load_state("turtle")
        >>> print(loaded.positions)
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/state.db",
        max_history: int = 100
    ):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database
            max_history: Maximum number of historical states to keep per strategy
        """
        self.db_path = Path(db_path)
        self.max_history = max_history
        self._lock = threading.RLock()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"StrategyStateManager initialized with db: {self.db_path}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main state table - stores latest state per strategy
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL UNIQUE,
                    version INTEGER DEFAULT 1,
                    state_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # State history table - for debugging and recovery
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    state_data TEXT NOT NULL,
                    checkpoint_type TEXT DEFAULT 'manual',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (strategy_name) REFERENCES strategy_state(strategy_name)
                )
            ''')

            # Positions table - denormalized for quick queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    target REAL,
                    entry_time TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, symbol)
                )
            ''')

            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_state_history_strategy
                ON state_history(strategy_name, created_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_positions_strategy
                ON positions(strategy_name)
            ''')

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # State Operations
    # =========================================================================

    def save_state(
        self,
        state: Union[StrategyState, Dict[str, Any]],
        strategy_name: Optional[str] = None,
        checkpoint_type: str = "manual"
    ) -> bool:
        """
        Save strategy state to database.

        Args:
            state: StrategyState object or dict
            strategy_name: Override strategy name (if state is dict)
            checkpoint_type: Type of checkpoint (manual, periodic, crash)

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                # Normalize to StrategyState
                if isinstance(state, dict):
                    if strategy_name:
                        state['strategy_name'] = strategy_name
                    state = StrategyState(**state)

                name = state.strategy_name
                state_data = StateSerializer.serialize(state.to_dict())

                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Update or insert main state
                    cursor.execute('''
                        INSERT INTO strategy_state (strategy_name, version, state_data, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(strategy_name) DO UPDATE SET
                            version = version + 1,
                            state_data = excluded.state_data,
                            updated_at = CURRENT_TIMESTAMP
                    ''', (name, state.version, state_data))

                    # Save to history
                    cursor.execute('''
                        INSERT INTO state_history (strategy_name, version, state_data, checkpoint_type)
                        VALUES (?, ?, ?, ?)
                    ''', (name, state.version, state_data, checkpoint_type))

                    # Update positions table
                    self._sync_positions(cursor, state)

                    # Cleanup old history
                    self._cleanup_history(cursor, name)

                    conn.commit()

                logger.debug(f"Saved state for {name} ({checkpoint_type})")
                return True

            except Exception as e:
                logger.error(f"Failed to save state: {e}", exc_info=True)
                return False

    def load_state(self, strategy_name: str) -> Optional[StrategyState]:
        """
        Load the latest state for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            StrategyState or None if not found
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        SELECT state_data, version, updated_at
                        FROM strategy_state
                        WHERE strategy_name = ?
                    ''', (strategy_name,))

                    row = cursor.fetchone()
                    if not row:
                        logger.debug(f"No state found for {strategy_name}")
                        return None

                    state_dict = StateSerializer.deserialize(row['state_data'])
                    state = StrategyState.from_dict(state_dict)

                    logger.debug(f"Loaded state for {strategy_name} (v{row['version']})")
                    return state

            except Exception as e:
                logger.error(f"Failed to load state for {strategy_name}: {e}", exc_info=True)
                return None

    def delete_state(self, strategy_name: str) -> bool:
        """
        Delete all state for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if deleted successfully
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('DELETE FROM positions WHERE strategy_name = ?', (strategy_name,))
                    cursor.execute('DELETE FROM state_history WHERE strategy_name = ?', (strategy_name,))
                    cursor.execute('DELETE FROM strategy_state WHERE strategy_name = ?', (strategy_name,))

                    conn.commit()

                logger.info(f"Deleted state for {strategy_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete state: {e}", exc_info=True)
                return False

    # =========================================================================
    # Position Operations (Quick Access)
    # =========================================================================

    def _sync_positions(self, cursor: sqlite3.Cursor, state: StrategyState):
        """Sync positions table with state."""
        # Clear old positions for this strategy
        cursor.execute('DELETE FROM positions WHERE strategy_name = ?', (state.strategy_name,))

        # Insert current positions
        for symbol, pos in state.positions.items():
            if isinstance(pos, dict):
                cursor.execute('''
                    INSERT INTO positions (strategy_name, symbol, side, quantity, entry_price, stop_loss, target, entry_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    state.strategy_name,
                    symbol,
                    pos.get('side', 'BUY'),
                    pos.get('quantity', 0),
                    pos.get('entry_price', 0),
                    pos.get('stop_loss'),
                    pos.get('target'),
                    pos.get('entry_time')
                ))

    def get_positions(self, strategy_name: str) -> Dict[str, Dict]:
        """
        Get current positions for a strategy (quick access).

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dict of symbol -> position data
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        SELECT symbol, side, quantity, entry_price, stop_loss, target, entry_time
                        FROM positions
                        WHERE strategy_name = ?
                    ''', (strategy_name,))

                    positions = {}
                    for row in cursor.fetchall():
                        positions[row['symbol']] = {
                            'side': row['side'],
                            'quantity': row['quantity'],
                            'entry_price': row['entry_price'],
                            'stop_loss': row['stop_loss'],
                            'target': row['target'],
                            'entry_time': row['entry_time']
                        }

                    return positions

            except Exception as e:
                logger.error(f"Failed to get positions: {e}")
                return {}

    def get_all_positions(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get all positions across all strategies.

        Returns:
            Dict of strategy_name -> symbol -> position data
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        SELECT strategy_name, symbol, side, quantity, entry_price, stop_loss, target
                        FROM positions
                    ''')

                    result = {}
                    for row in cursor.fetchall():
                        strategy = row['strategy_name']
                        if strategy not in result:
                            result[strategy] = {}
                        result[strategy][row['symbol']] = {
                            'side': row['side'],
                            'quantity': row['quantity'],
                            'entry_price': row['entry_price'],
                            'stop_loss': row['stop_loss'],
                            'target': row['target']
                        }

                    return result

            except Exception as e:
                logger.error(f"Failed to get all positions: {e}")
                return {}

    # =========================================================================
    # History Operations
    # =========================================================================

    def get_state_history(
        self,
        strategy_name: str,
        limit: int = 10
    ) -> List[StrategyState]:
        """
        Get historical states for a strategy.

        Args:
            strategy_name: Name of the strategy
            limit: Maximum number of states to return

        Returns:
            List of StrategyState objects (newest first)
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        SELECT state_data, checkpoint_type, created_at
                        FROM state_history
                        WHERE strategy_name = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', (strategy_name, limit))

                    states = []
                    for row in cursor.fetchall():
                        state_dict = StateSerializer.deserialize(row['state_data'])
                        state = StrategyState.from_dict(state_dict)
                        states.append(state)

                    return states

            except Exception as e:
                logger.error(f"Failed to get state history: {e}")
                return []

    def _cleanup_history(self, cursor: sqlite3.Cursor, strategy_name: str):
        """Remove old history entries beyond max_history."""
        cursor.execute('''
            DELETE FROM state_history
            WHERE strategy_name = ? AND id NOT IN (
                SELECT id FROM state_history
                WHERE strategy_name = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
        ''', (strategy_name, strategy_name, self.max_history))

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_strategies(self) -> List[str]:
        """Get list of strategies with saved state."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT strategy_name FROM strategy_state')
                    return [row['strategy_name'] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to list strategies: {e}")
                return []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all saved states."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()

                    # Count strategies
                    cursor.execute('SELECT COUNT(*) as count FROM strategy_state')
                    strategy_count = cursor.fetchone()['count']

                    # Count positions
                    cursor.execute('SELECT COUNT(*) as count FROM positions')
                    position_count = cursor.fetchone()['count']

                    # Count history entries
                    cursor.execute('SELECT COUNT(*) as count FROM state_history')
                    history_count = cursor.fetchone()['count']

                    # Get strategies with timestamps
                    cursor.execute('''
                        SELECT strategy_name, updated_at FROM strategy_state
                        ORDER BY updated_at DESC
                    ''')
                    strategies = [
                        {'name': row['strategy_name'], 'updated': row['updated_at']}
                        for row in cursor.fetchall()
                    ]

                    return {
                        'strategies': strategy_count,
                        'positions': position_count,
                        'history_entries': history_count,
                        'strategy_list': strategies,
                        'db_path': str(self.db_path)
                    }

            except Exception as e:
                logger.error(f"Failed to get summary: {e}")
                return {}

    def export_state(self, strategy_name: str, file_path: Union[str, Path]) -> bool:
        """
        Export strategy state to a JSON file.

        Args:
            strategy_name: Name of the strategy
            file_path: Path to export file

        Returns:
            True if exported successfully
        """
        state = self.load_state(strategy_name)
        if not state:
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            logger.info(f"Exported state to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export state: {e}")
            return False

    def import_state(self, file_path: Union[str, Path]) -> Optional[StrategyState]:
        """
        Import strategy state from a JSON file.

        Args:
            file_path: Path to import file

        Returns:
            Imported StrategyState or None
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            state = StrategyState.from_dict(data)
            self.save_state(state, checkpoint_type="import")
            logger.info(f"Imported state from {file_path}")
            return state
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[StrategyStateManager] = None


def get_state_manager(db_path: str = "data/state.db") -> StrategyStateManager:
    """Get or create the default state manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StrategyStateManager(db_path)
    return _default_manager


def save_strategy_state(
    strategy_name: str,
    positions: Dict = None,
    indicators: Dict = None,
    parameters: Dict = None,
    **kwargs
) -> bool:
    """
    Quick function to save strategy state.

    Example:
        >>> save_strategy_state(
        ...     "turtle",
        ...     positions={"RELIANCE": {"qty": 10, "price": 2500}},
        ...     indicators={"sma_20": 2480}
        ... )
    """
    mgr = get_state_manager()
    state = StrategyState(
        strategy_name=strategy_name,
        positions=positions or {},
        indicators=indicators or {},
        parameters=parameters or {},
        **kwargs
    )
    return mgr.save_state(state)


def load_strategy_state(strategy_name: str) -> Optional[StrategyState]:
    """Quick function to load strategy state."""
    return get_state_manager().load_state(strategy_name)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("STRATEGY STATE MANAGER - Test")
    print("=" * 50)

    # Create manager with test database
    mgr = StrategyStateManager("data/test_state.db")

    # Create test state
    state = StrategyState(
        strategy_name="test_turtle",
        positions={
            "RELIANCE": {
                "side": "BUY",
                "quantity": 10,
                "entry_price": 2500.0,
                "stop_loss": 2450.0,
                "target": 2600.0
            }
        },
        indicators={
            "sma_20": 2480.5,
            "sma_50": 2420.0,
            "rsi": 55.0
        },
        parameters={
            "fast_period": 20,
            "slow_period": 50
        }
    )

    # Save state
    print("\n1. Saving state...")
    mgr.save_state(state)
    print("   State saved!")

    # Load state
    print("\n2. Loading state...")
    loaded = mgr.load_state("test_turtle")
    print(f"   Positions: {loaded.positions}")
    print(f"   Indicators: {loaded.indicators}")

    # Get positions
    print("\n3. Getting positions...")
    positions = mgr.get_positions("test_turtle")
    print(f"   Positions: {positions}")

    # Get summary
    print("\n4. Getting summary...")
    summary = mgr.get_summary()
    print(f"   Strategies: {summary['strategies']}")
    print(f"   Positions: {summary['positions']}")

    # Cleanup
    print("\n5. Cleaning up...")
    mgr.delete_state("test_turtle")
    print("   Test state deleted!")

    print("\n" + "=" * 50)
    print("Strategy State Manager ready!")
    print("=" * 50)
