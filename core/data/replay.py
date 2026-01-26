"""
Replay Data Source for Debugging and Analysis.

Replays recorded live sessions for:
- Debugging production issues
- Analyzing past trades
- Compliance/audit trails
- Strategy development without live data

The key insight: Recorded events are replayed exactly as they occurred,
allowing you to reproduce any scenario from live trading.
"""

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from core.events import (
    BarEvent,
    TickEvent,
    EventBus,
    Event,
    EventType,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PriceUpdateEvent,
)
from .source import (
    DataSource,
    DataSourceConfig,
    DataSourceMode,
    DataSourceState,
)


logger = logging.getLogger(__name__)


@dataclass
class RecordedEvent:
    """
    A recorded event with metadata.

    Stores the original event plus recording metadata.
    """
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    sequence: int = 0
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'sequence': self.sequence,
            'session_id': self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RecordedEvent':
        """Create from dictionary."""
        return cls(
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            sequence=data.get('sequence', 0),
            session_id=data.get('session_id', ''),
        )

    @classmethod
    def from_event(cls, event: Event, sequence: int = 0, session_id: str = "") -> 'RecordedEvent':
        """Create from an Event object."""
        # Serialize event to dict
        data = {}
        for field_name in event.__dataclass_fields__:
            value = getattr(event, field_name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif hasattr(value, 'value'):  # Enum
                value = value.value
            elif hasattr(value, 'name'):  # Enum
                value = value.name
            data[field_name] = value

        return cls(
            event_type=event.event_type.name if event.event_type else 'UNKNOWN',
            timestamp=event.timestamp,
            data=data,
            sequence=sequence,
            session_id=session_id,
        )


class EventRecorder:
    """
    Records events to a file for later replay.

    Usage:
        recorder = EventRecorder('session_001')
        recorder.start()

        # Record events as they occur
        recorder.record(tick_event)
        recorder.record(bar_event)

        recorder.stop()
    """

    def __init__(
        self,
        session_id: str,
        output_dir: Union[str, Path] = "data/recordings",
        format: str = "jsonl",  # jsonl, pickle
    ):
        """
        Initialize recorder.

        Args:
            session_id: Unique session identifier
            output_dir: Directory for recordings
            format: Output format (jsonl or pickle)
        """
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.format = format

        self._sequence = 0
        self._file = None
        self._recording = False
        self._events: List[RecordedEvent] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start recording."""
        if self._recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_{timestamp}.{self.format}"
        filepath = self.output_dir / filename

        if self.format == "jsonl":
            self._file = open(filepath, 'w')
        else:
            self._events = []

        self._recording = True
        self._sequence = 0
        logger.info(f"Started recording to {filepath}")

    def stop(self):
        """Stop recording and save."""
        if not self._recording:
            return

        if self.format == "jsonl" and self._file:
            self._file.close()
            self._file = None
        elif self.format == "pickle":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.session_id}_{timestamp}.pkl"
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(self._events, f)
            logger.info(f"Saved {len(self._events)} events to {filepath}")

        self._recording = False
        logger.info("Recording stopped")

    def record(self, event: Event):
        """Record an event."""
        if not self._recording:
            return

        self._sequence += 1
        recorded = RecordedEvent.from_event(event, self._sequence, self.session_id)

        if self.format == "jsonl" and self._file:
            self._file.write(json.dumps(recorded.to_dict()) + "\n")
            self._file.flush()
        else:
            self._events.append(recorded)

    @property
    def event_count(self) -> int:
        """Get number of recorded events."""
        return self._sequence


class ReplayDataSource(DataSource):
    """
    Replay data source for debugging and analysis.

    Replays recorded events from a file, preserving exact timing
    or at accelerated speed.

    Usage:
        config = DataSourceConfig(speed_multiplier=10.0)  # 10x speed
        source = ReplayDataSource(
            config,
            event_bus,
            recording_file='data/recordings/session_001.jsonl'
        )
        source.start()
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        recording_file: Optional[Union[str, Path]] = None,
        events: Optional[List[RecordedEvent]] = None,
    ):
        """
        Initialize replay source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            recording_file: Path to recording file
            events: List of recorded events (alternative to file)
        """
        super().__init__(config, event_bus, DataSourceMode.REPLAY)

        self._recording_file = Path(recording_file) if recording_file else None
        self._events = events or []
        self._current_index = 0

    def connect(self) -> bool:
        """Load recording file."""
        if self._events:
            logger.info(f"Using {len(self._events)} pre-loaded events")
            return True

        if not self._recording_file or not self._recording_file.exists():
            logger.error(f"Recording file not found: {self._recording_file}")
            return False

        try:
            if self._recording_file.suffix == '.jsonl':
                self._events = self._load_jsonl()
            elif self._recording_file.suffix in ['.pkl', '.pickle']:
                self._events = self._load_pickle()
            else:
                logger.error(f"Unsupported format: {self._recording_file.suffix}")
                return False

            # Filter by symbols if specified
            if self.config.symbols:
                self._events = [
                    e for e in self._events
                    if e.data.get('symbol', '') in self.config.symbols or
                    e.event_type not in ['TICK', 'BAR', 'PRICE_UPDATE']
                ]

            # Filter by date range
            if self.config.start_date:
                self._events = [e for e in self._events if e.timestamp >= self.config.start_date]
            if self.config.end_date:
                self._events = [e for e in self._events if e.timestamp <= self.config.end_date]

            # Sort by timestamp
            self._events.sort(key=lambda e: (e.timestamp, e.sequence))

            logger.info(f"Loaded {len(self._events)} events from {self._recording_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load recording: {e}", exc_info=True)
            return False

    def disconnect(self):
        """Clear loaded events."""
        self._current_index = 0

    def _emit_events(self) -> Generator[Union[BarEvent, TickEvent], None, None]:
        """
        Replay recorded events.

        Maintains relative timing between events based on speed_multiplier.
        """
        if not self._events:
            logger.warning("No events to replay")
            return

        logger.info(f"Starting replay of {len(self._events)} events")

        prev_timestamp = None

        for i, recorded in enumerate(self._events):
            if not self._running:
                break

            self._current_index = i

            # Calculate delay
            if prev_timestamp and self.config.speed_multiplier > 0:
                delta = (recorded.timestamp - prev_timestamp).total_seconds()
                sleep_time = delta / self.config.speed_multiplier
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 1.0))  # Cap at 1 second

            prev_timestamp = recorded.timestamp

            # Convert to event object
            event = self._reconstruct_event(recorded)
            if event:
                yield event

    def _load_jsonl(self) -> List[RecordedEvent]:
        """Load events from JSONL file."""
        events = []
        with open(self._recording_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        events.append(RecordedEvent.from_dict(data))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
        return events

    def _load_pickle(self) -> List[RecordedEvent]:
        """Load events from pickle file."""
        with open(self._recording_file, 'rb') as f:
            return pickle.load(f)

    def _reconstruct_event(self, recorded: RecordedEvent) -> Optional[Union[BarEvent, TickEvent, Event]]:
        """Reconstruct an Event from a RecordedEvent."""
        try:
            data = recorded.data.copy()

            # Parse timestamp
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])

            # Reconstruct based on event type
            if recorded.event_type == 'TICK':
                return TickEvent(**{k: v for k, v in data.items() if k in TickEvent.__dataclass_fields__})

            elif recorded.event_type == 'BAR':
                return BarEvent(**{k: v for k, v in data.items() if k in BarEvent.__dataclass_fields__})

            elif recorded.event_type == 'PRICE_UPDATE':
                return PriceUpdateEvent(**{k: v for k, v in data.items() if k in PriceUpdateEvent.__dataclass_fields__})

            elif recorded.event_type == 'SIGNAL_GENERATED':
                return SignalEvent(**{k: v for k, v in data.items() if k in SignalEvent.__dataclass_fields__})

            else:
                # Generic event - just emit as TickEvent with price
                if 'last_price' in data or 'close' in data:
                    return TickEvent(
                        symbol=data.get('symbol', ''),
                        last_price=data.get('last_price', data.get('close', 0)),
                        timestamp=data.get('timestamp', datetime.now()),
                        source="replay"
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to reconstruct event: {e}")
            return None

    # =========================================================================
    # Additional Methods
    # =========================================================================

    @property
    def progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        if not self._events:
            return 0.0
        return self._current_index / len(self._events)

    @property
    def total_events(self) -> int:
        """Get total number of events."""
        return len(self._events)

    @property
    def current_event(self) -> Optional[RecordedEvent]:
        """Get current event being replayed."""
        if 0 <= self._current_index < len(self._events):
            return self._events[self._current_index]
        return None

    def seek(self, index: int):
        """Seek to a specific event index."""
        self._current_index = max(0, min(index, len(self._events) - 1))

    def seek_time(self, timestamp: datetime):
        """Seek to a specific timestamp."""
        for i, event in enumerate(self._events):
            if event.timestamp >= timestamp:
                self._current_index = i
                return
        self._current_index = len(self._events) - 1

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[RecordedEvent]:
        """Get events within a time range."""
        return [
            e for e in self._events
            if start <= e.timestamp <= end
        ]

    def get_symbols(self) -> List[str]:
        """Get unique symbols in the recording."""
        symbols = set()
        for event in self._events:
            symbol = event.data.get('symbol')
            if symbol:
                symbols.add(symbol)
        return sorted(symbols)

    def get_time_range(self) -> tuple:
        """Get start and end time of recording."""
        if not self._events:
            return (None, None)
        return (self._events[0].timestamp, self._events[-1].timestamp)
