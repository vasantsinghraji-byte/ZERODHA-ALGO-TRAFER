# -*- coding: utf-8 -*-
"""
Crash Recovery Module - Never Miss a Beat!
===========================================
Detect crashes, recover state, and resume trading seamlessly.

Features:
- Detect incomplete/crashed sessions
- Load last valid checkpoint
- Replay events since checkpoint
- Graceful session resumption

Example:
    >>> from core.state import RecoveryManager
    >>>
    >>> recovery = RecoveryManager(state_manager)
    >>> if recovery.needs_recovery():
    ...     result = recovery.recover()
    ...     print(f"Recovered from {result.checkpoint_time}")
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_manager import StrategyStateManager, StrategyState
    from ..events import Event

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session termination status."""
    UNKNOWN = "unknown"
    CLEAN = "clean"           # Graceful shutdown
    CRASHED = "crashed"       # Unexpected termination
    INTERRUPTED = "interrupted"  # User interrupt (Ctrl+C)
    TIMEOUT = "timeout"       # Watchdog timeout


@dataclass
class SessionInfo:
    """Information about a trading session."""
    session_id: str
    strategy_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SessionStatus = SessionStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    checkpoint_count: int = 0
    event_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    strategy_id: str
    session_id: str
    checkpoint_time: Optional[datetime] = None
    state: Optional['StrategyState'] = None
    events_replayed: int = 0
    recovery_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    # Session detection
    heartbeat_timeout_seconds: int = 60  # Consider crashed if no heartbeat
    max_session_age_hours: int = 24      # Ignore sessions older than this

    # Recovery behavior
    auto_recover: bool = True            # Automatically recover on startup
    replay_events: bool = True           # Replay events after checkpoint
    verify_positions: bool = True        # Verify positions with broker

    # Safety
    max_recovery_attempts: int = 3       # Max retries before giving up
    require_manual_confirm: bool = False # Require user confirmation


class SessionTracker:
    """
    Tracks trading sessions for crash detection.

    Uses heartbeat mechanism to detect unclean shutdowns.
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[SessionInfo] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize session tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    last_heartbeat TEXT,
                    checkpoint_count INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_strategy
                ON sessions(strategy_id, start_time DESC)
            """)
            conn.commit()

    def start_session(
        self,
        strategy_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Start tracking a new session."""
        import uuid
        import json

        session_id = session_id or str(uuid.uuid4())[:8]
        now = datetime.now()

        session = SessionInfo(
            session_id=session_id,
            strategy_id=strategy_id,
            start_time=now,
            status=SessionStatus.UNKNOWN,
            last_heartbeat=now,
            metadata=metadata or {}
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions
                (session_id, strategy_id, start_time, status, last_heartbeat, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.strategy_id,
                session.start_time.isoformat(),
                session.status.value,
                session.last_heartbeat.isoformat(),
                json.dumps(session.metadata)
            ))
            conn.commit()

        self._current_session = session
        logger.info(f"Started session {session_id} for {strategy_id}")
        return session

    def heartbeat(self) -> None:
        """Update session heartbeat."""
        if not self._current_session:
            return

        now = datetime.now()
        self._current_session.last_heartbeat = now

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET last_heartbeat = ?
                WHERE session_id = ?
            """, (now.isoformat(), self._current_session.session_id))
            conn.commit()

    def increment_events(self, count: int = 1) -> None:
        """Increment event counter for current session."""
        if not self._current_session:
            return

        self._current_session.event_count += count

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET event_count = event_count + ?
                WHERE session_id = ?
            """, (count, self._current_session.session_id))
            conn.commit()

    def increment_checkpoints(self) -> None:
        """Increment checkpoint counter for current session."""
        if not self._current_session:
            return

        self._current_session.checkpoint_count += 1

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET checkpoint_count = checkpoint_count + 1
                WHERE session_id = ?
            """, (self._current_session.session_id,))
            conn.commit()

    def end_session(self, status: SessionStatus = SessionStatus.CLEAN) -> None:
        """Mark current session as ended."""
        if not self._current_session:
            return

        now = datetime.now()
        self._current_session.end_time = now
        self._current_session.status = status

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET end_time = ?, status = ?
                WHERE session_id = ?
            """, (now.isoformat(), status.value, self._current_session.session_id))
            conn.commit()

        logger.info(f"Ended session {self._current_session.session_id} with status {status.value}")
        self._current_session = None

    def get_last_session(self, strategy_id: str) -> Optional[SessionInfo]:
        """Get the most recent session for a strategy."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions
                WHERE strategy_id = ?
                ORDER BY start_time DESC
                LIMIT 1
            """, (strategy_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return SessionInfo(
            session_id=row['session_id'],
            strategy_id=row['strategy_id'],
            start_time=datetime.fromisoformat(row['start_time']),
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            status=SessionStatus(row['status']),
            last_heartbeat=datetime.fromisoformat(row['last_heartbeat']) if row['last_heartbeat'] else None,
            checkpoint_count=row['checkpoint_count'],
            event_count=row['event_count'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def get_incomplete_sessions(
        self,
        strategy_id: Optional[str] = None,
        max_age_hours: int = 24
    ) -> List[SessionInfo]:
        """Find sessions that didn't end cleanly."""
        import json

        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        query = """
            SELECT * FROM sessions
            WHERE (status = 'unknown' OR status = 'crashed' OR status = 'interrupted')
            AND start_time > ?
        """
        params: List[Any] = [cutoff.isoformat()]

        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)

        query += " ORDER BY start_time DESC"

        sessions = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor:
                sessions.append(SessionInfo(
                    session_id=row['session_id'],
                    strategy_id=row['strategy_id'],
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    status=SessionStatus(row['status']),
                    last_heartbeat=datetime.fromisoformat(row['last_heartbeat']) if row['last_heartbeat'] else None,
                    checkpoint_count=row['checkpoint_count'],
                    event_count=row['event_count'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))

        return sessions

    def detect_crashed_sessions(
        self,
        heartbeat_timeout_seconds: int = 60
    ) -> List[SessionInfo]:
        """
        Detect sessions that crashed (no heartbeat, not ended).

        A session is considered crashed if:
        - Status is 'unknown' (not explicitly ended)
        - Last heartbeat is older than timeout
        """
        incomplete = self.get_incomplete_sessions()
        cutoff = datetime.now() - timedelta(seconds=heartbeat_timeout_seconds)

        crashed = []
        for session in incomplete:
            if session.status == SessionStatus.UNKNOWN:
                if session.last_heartbeat and session.last_heartbeat < cutoff:
                    # Mark as crashed
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE sessions SET status = 'crashed'
                            WHERE session_id = ?
                        """, (session.session_id,))
                        conn.commit()
                    session.status = SessionStatus.CRASHED
                    crashed.append(session)

        return crashed


class RecoveryManager:
    """
    Manages crash recovery for trading strategies.

    Detects incomplete sessions, loads checkpoints, and replays events.
    """

    def __init__(
        self,
        state_manager: 'StrategyStateManager',
        config: Optional[RecoveryConfig] = None,
        session_db_path: str = "data/sessions.db"
    ):
        self.state_manager = state_manager
        self.config = config or RecoveryConfig()
        self.session_tracker = SessionTracker(session_db_path)
        self._event_store: Optional['EventStore'] = None
        self._recovery_callbacks: List[Callable[[RecoveryResult], None]] = []

    def set_event_store(self, event_store: 'EventStore') -> None:
        """Set event store for event replay."""
        self._event_store = event_store

    def on_recovery(self, callback: Callable[[RecoveryResult], None]) -> None:
        """Register callback for recovery completion."""
        self._recovery_callbacks.append(callback)

    def needs_recovery(self, strategy_id: str) -> bool:
        """Check if a strategy needs recovery."""
        # Check for crashed sessions
        crashed = self.session_tracker.detect_crashed_sessions(
            self.config.heartbeat_timeout_seconds
        )

        for session in crashed:
            if session.strategy_id == strategy_id:
                logger.warning(f"Strategy {strategy_id} has crashed session {session.session_id}")
                return True

        # Check for incomplete sessions
        incomplete = self.session_tracker.get_incomplete_sessions(
            strategy_id=strategy_id,
            max_age_hours=self.config.max_session_age_hours
        )

        return len(incomplete) > 0

    def get_recovery_info(self, strategy_id: str) -> Optional[SessionInfo]:
        """Get information about session that needs recovery."""
        incomplete = self.session_tracker.get_incomplete_sessions(
            strategy_id=strategy_id,
            max_age_hours=self.config.max_session_age_hours
        )
        return incomplete[0] if incomplete else None

    def recover(
        self,
        strategy_id: str,
        replay_events: Optional[bool] = None
    ) -> RecoveryResult:
        """
        Recover a strategy from crash.

        Steps:
        1. Find last checkpoint
        2. Load state from checkpoint
        3. Optionally replay events since checkpoint
        4. Return recovered state
        """
        import time
        start_time = time.time()

        result = RecoveryResult(
            success=False,
            strategy_id=strategy_id,
            session_id=""
        )

        # Find crashed session
        session_info = self.get_recovery_info(strategy_id)
        if not session_info:
            result.errors.append("No session found that needs recovery")
            return result

        result.session_id = session_info.session_id
        logger.info(f"Starting recovery for {strategy_id} from session {session_info.session_id}")

        # Load last checkpoint
        try:
            state = self.state_manager.load_state(strategy_id)
            if state is None:
                result.errors.append("No checkpoint found for strategy")
                return result

            result.state = state
            result.checkpoint_time = state.timestamp
            logger.info(f"Loaded checkpoint from {state.timestamp}")

        except Exception as e:
            result.errors.append(f"Failed to load checkpoint: {e}")
            logger.error(f"Checkpoint load failed: {e}")
            return result

        # Replay events if enabled
        should_replay = replay_events if replay_events is not None else self.config.replay_events

        if should_replay and self._event_store and result.checkpoint_time:
            try:
                events_replayed = self._replay_events(
                    strategy_id,
                    result.checkpoint_time
                )
                result.events_replayed = events_replayed
                logger.info(f"Replayed {events_replayed} events")
            except Exception as e:
                result.warnings.append(f"Event replay failed: {e}")
                logger.warning(f"Event replay failed: {e}")

        # Mark old session as recovered
        with sqlite3.connect(self.session_tracker.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET status = 'crashed',
                end_time = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_info.session_id))
            conn.commit()

        result.success = True
        result.recovery_time_ms = (time.time() - start_time) * 1000

        # Notify callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

        logger.info(f"Recovery complete in {result.recovery_time_ms:.1f}ms")
        return result

    def _replay_events(
        self,
        strategy_id: str,
        since: datetime
    ) -> int:
        """Replay events since checkpoint time."""
        if not self._event_store:
            return 0

        events = self._event_store.get_events_since(strategy_id, since)
        replayed = 0

        for event in events:
            try:
                self._event_store.replay_event(event)
                replayed += 1
            except Exception as e:
                logger.warning(f"Failed to replay event: {e}")

        return replayed

    def start_session(
        self,
        strategy_id: str,
        session_id: Optional[str] = None,
        auto_recover: Optional[bool] = None
    ) -> RecoveryResult:
        """
        Start a new session with optional auto-recovery.

        If auto_recover is enabled and a crashed session exists,
        attempts recovery before starting new session.
        """
        should_recover = auto_recover if auto_recover is not None else self.config.auto_recover

        result = RecoveryResult(
            success=True,
            strategy_id=strategy_id,
            session_id=session_id or ""
        )

        # Check for crashed session
        if should_recover and self.needs_recovery(strategy_id):
            logger.info(f"Auto-recovering strategy {strategy_id}")
            result = self.recover(strategy_id)

            if not result.success:
                logger.error(f"Auto-recovery failed for {strategy_id}")
                # Continue anyway with new session

        # Start new session
        session = self.session_tracker.start_session(
            strategy_id=strategy_id,
            session_id=session_id
        )
        result.session_id = session.session_id

        return result

    def end_session(self, clean: bool = True) -> None:
        """End current session."""
        status = SessionStatus.CLEAN if clean else SessionStatus.INTERRUPTED
        self.session_tracker.end_session(status)

    def heartbeat(self) -> None:
        """Update session heartbeat."""
        self.session_tracker.heartbeat()


class EventStore:
    """
    Stores events for replay during recovery.

    This is a simple interface - can be extended for different backends.
    """

    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_handlers: Dict[str, Callable[['Event'], None]] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize event storage database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_strategy_time
                ON events(strategy_id, timestamp)
            """)
            conn.commit()

    def store_event(
        self,
        strategy_id: str,
        event: 'Event'
    ) -> None:
        """Store an event for potential replay."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events (strategy_id, event_type, timestamp, data)
                VALUES (?, ?, ?, ?)
            """, (
                strategy_id,
                event.__class__.__name__,
                event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat(),
                json.dumps(event.__dict__, default=str)
            ))
            conn.commit()

    def get_events_since(
        self,
        strategy_id: str,
        since: datetime
    ) -> List[Dict[str, Any]]:
        """Get events since a timestamp."""
        import json

        events = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM events
                WHERE strategy_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (strategy_id, since.isoformat()))

            for row in cursor:
                events.append({
                    'id': row['id'],
                    'event_type': row['event_type'],
                    'timestamp': datetime.fromisoformat(row['timestamp']),
                    'data': json.loads(row['data'])
                })

        return events

    def register_handler(
        self,
        event_type: str,
        handler: Callable[['Event'], None]
    ) -> None:
        """Register handler for replaying events."""
        self._event_handlers[event_type] = handler

    def replay_event(self, event_data: Dict[str, Any]) -> None:
        """Replay a stored event."""
        event_type = event_data.get('event_type')
        if event_type and event_type in self._event_handlers:
            self._event_handlers[event_type](event_data)

    def cleanup_old_events(self, max_age_days: int = 7) -> int:
        """Remove events older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=max_age_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM events WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            conn.commit()
            return cursor.rowcount


# Singleton accessor
_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager(
    state_manager: Optional['StrategyStateManager'] = None,
    config: Optional[RecoveryConfig] = None
) -> RecoveryManager:
    """Get or create singleton recovery manager."""
    global _recovery_manager

    if _recovery_manager is None:
        if state_manager is None:
            from .state_manager import get_state_manager
            state_manager = get_state_manager()
        _recovery_manager = RecoveryManager(state_manager, config)

    return _recovery_manager
