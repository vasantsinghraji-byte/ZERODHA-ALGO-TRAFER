# -*- coding: utf-8 -*-
"""
Flight Recorder - Market Replay System
=======================================

⚠️  PHASE 12 FEATURE - PREREQUISITES NOT MET ⚠️

DO NOT USE until these Phase 1 items are working:
1. ✗ TradingBot can place orders via Zerodha API
2. ✗ EventBus events are flowing through the system
3. ✗ Strategies generate signals that execute trades
4. ✗ Basic order lifecycle (place → fill → close) works

WHY THIS IS PREMATURE:
- Recording market data for replay is useless if your bot can't trade
- You'll record hours of data but have nothing to debug
- Focus on getting ONE successful trade first

WHEN TO USE THIS:
- After you've executed live trades successfully
- When debugging why a specific trade went wrong
- For backtesting improvements after production runs

------------------------------------------------------------------------

Production-grade "black box" recorder for trading systems.

Records all market data and internal events for perfect replay,
enabling debugging of failed strategies with complete hindsight.

Features:
- Binary tick recording with microsecond timestamps
- Internal event capture from EventBus
- Compressed storage (LZ4/Zstd)
- Exact replay of any trading day
- Streaming playback with speed control

Example (ONLY AFTER PREREQUISITES MET):
    >>> from core.infrastructure.flight_recorder import FlightRecorder
    >>>
    >>> # Record a trading session
    >>> recorder = FlightRecorder("./recordings")
    >>> recorder.start_session("2024-01-15")
    >>> recorder.record_tick(tick_data)
    >>> recorder.record_event("order_placed", order_data)
    >>> recorder.end_session()
    >>>
    >>> # Replay for debugging
    >>> replayer = MarketReplayer("./recordings/2024-01-15")
    >>> for tick in replayer.play(speed=10.0):
    ...     strategy.on_tick(tick)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Iterator, BinaryIO, Union, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict
import struct
import json
import time
import threading
import queue
import os
import io
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Compression libraries
try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"


class RecordType(Enum):
    """Types of records in the flight recorder."""
    TICK = 0x01
    EVENT = 0x02
    ORDER = 0x03
    FILL = 0x04
    SIGNAL = 0x05
    STATE = 0x06
    ERROR = 0x07
    MARKER = 0x08
    HEARTBEAT = 0x09


class PlaybackState(Enum):
    """Playback state machine."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"


# Binary format constants
MAGIC_NUMBER = b'FLREC'  # File magic number
VERSION = 1
HEADER_SIZE = 64
RECORD_HEADER_SIZE = 16  # type(1) + flags(1) + timestamp(8) + length(4) + checksum(2)


@dataclass
class TickRecord:
    """Recorded market tick."""
    timestamp: int  # Microseconds since epoch
    symbol: str
    ltp: float
    ltq: int
    volume: int
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    oi: int = 0
    exchange: str = ""

    def to_bytes(self) -> bytes:
        """Serialize to binary format."""
        symbol_bytes = self.symbol.encode('utf-8')[:32].ljust(32, b'\x00')
        exchange_bytes = self.exchange.encode('utf-8')[:8].ljust(8, b'\x00')

        return struct.pack(
            '<Q32s8sddQddQQQ',
            self.timestamp,
            symbol_bytes,
            exchange_bytes,
            self.ltp,
            self.bid,
            self.ltq,
            self.ask,
            self.bid_qty,
            self.ask_qty,
            self.volume,
            self.oi
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TickRecord':
        """Deserialize from binary format."""
        unpacked = struct.unpack('<Q32s8sddQddQQQ', data[:112])

        return cls(
            timestamp=unpacked[0],
            symbol=unpacked[1].rstrip(b'\x00').decode('utf-8'),
            exchange=unpacked[2].rstrip(b'\x00').decode('utf-8'),
            ltp=unpacked[3],
            bid=unpacked[4],
            ltq=unpacked[5],
            ask=unpacked[6],
            bid_qty=unpacked[7],
            ask_qty=unpacked[8],
            volume=unpacked[9],
            oi=unpacked[10]
        )

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000)


@dataclass
class EventRecord:
    """Recorded internal event."""
    timestamp: int  # Microseconds since epoch
    event_type: str
    source: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None

    def to_bytes(self) -> bytes:
        """Serialize to binary format."""
        payload_json = json.dumps(self.payload).encode('utf-8')
        event_bytes = self.event_type.encode('utf-8')[:64].ljust(64, b'\x00')
        source_bytes = self.source.encode('utf-8')[:32].ljust(32, b'\x00')
        corr_bytes = (self.correlation_id or "").encode('utf-8')[:36].ljust(36, b'\x00')

        return struct.pack(
            '<Q64s32s36sI',
            self.timestamp,
            event_bytes,
            source_bytes,
            corr_bytes,
            len(payload_json)
        ) + payload_json

    @classmethod
    def from_bytes(cls, data: bytes) -> 'EventRecord':
        """Deserialize from binary format."""
        header = struct.unpack('<Q64s32s36sI', data[:144])
        payload_len = header[4]
        payload_json = data[144:144 + payload_len]

        corr_id = header[3].rstrip(b'\x00').decode('utf-8')

        return cls(
            timestamp=header[0],
            event_type=header[1].rstrip(b'\x00').decode('utf-8'),
            source=header[2].rstrip(b'\x00').decode('utf-8'),
            payload=json.loads(payload_json),
            correlation_id=corr_id if corr_id else None
        )

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000)


@dataclass
class RecordingSession:
    """Metadata for a recording session."""
    session_id: str
    trading_date: date
    start_time: datetime
    end_time: Optional[datetime]
    tick_count: int
    event_count: int
    symbols: List[str]
    compression: CompressionType
    file_size: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'trading_date': self.trading_date.isoformat(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'tick_count': self.tick_count,
            'event_count': self.event_count,
            'symbols': self.symbols,
            'compression': self.compression.value,
            'file_size': self.file_size,
            'checksum': self.checksum
        }


class Compressor:
    """Handles compression/decompression of recorded data."""

    def __init__(self, compression: CompressionType = CompressionType.LZ4):
        """
        Initialize compressor.

        Args:
            compression: Compression algorithm to use
        """
        self.compression = compression

        if compression == CompressionType.LZ4 and not LZ4_AVAILABLE:
            logger.warning("LZ4 not available, falling back to GZIP")
            self.compression = CompressionType.GZIP

        if compression == CompressionType.ZSTD and not ZSTD_AVAILABLE:
            logger.warning("ZSTD not available, falling back to GZIP")
            self.compression = CompressionType.GZIP

        # Initialize ZSTD compressor context for efficiency
        if self.compression == CompressionType.ZSTD and ZSTD_AVAILABLE:
            self._zstd_compressor = zstd.ZstdCompressor(level=3)
            self._zstd_decompressor = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.compression == CompressionType.NONE:
            return data

        if self.compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.compress(data, compression_level=4)

        if self.compression == CompressionType.ZSTD and ZSTD_AVAILABLE:
            return self._zstd_compressor.compress(data)

        if self.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data, compresslevel=6)

        return data

    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.compression == CompressionType.NONE:
            return data

        if self.compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.decompress(data)

        if self.compression == CompressionType.ZSTD and ZSTD_AVAILABLE:
            return self._zstd_decompressor.decompress(data)

        if self.compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)

        return data


class TickRecorder:
    """
    High-performance tick recorder.

    Records market ticks to binary files with minimal latency impact.
    Uses background thread for I/O to avoid blocking the main thread.
    """

    def __init__(
        self,
        output_path: str,
        compression: CompressionType = CompressionType.LZ4,
        buffer_size: int = 10000,
        flush_interval: float = 1.0
    ):
        """
        Initialize tick recorder.

        Args:
            output_path: Path for output file
            compression: Compression algorithm
            buffer_size: Number of ticks to buffer before flush
            flush_interval: Max seconds between flushes
        """
        self.output_path = output_path
        self.compression = compression
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._compressor = Compressor(compression)
        self._buffer: List[TickRecord] = []
        self._file: Optional[BinaryIO] = None
        self._lock = threading.Lock()
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        self._tick_count = 0
        self._symbols: set = set()
        self._last_flush = time.time()

    def start(self) -> None:
        """Start the recorder."""
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        self._file = open(self.output_path, 'wb')
        self._write_header()

        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        logger.info(f"Tick recorder started: {self.output_path}")

    def stop(self) -> None:
        """Stop the recorder."""
        self._running = False
        self._flush()

        if self._writer_thread:
            self._write_queue.put(None)  # Sentinel to stop writer
            self._writer_thread.join(timeout=5.0)

        if self._file:
            self._file.close()
            self._file = None

        logger.info(f"Tick recorder stopped. Total ticks: {self._tick_count}")

    def record(self, tick: Union[TickRecord, Dict[str, Any]]) -> None:
        """
        Record a tick.

        Args:
            tick: TickRecord or dict with tick data
        """
        if isinstance(tick, dict):
            tick = TickRecord(
                timestamp=int(time.time() * 1_000_000),
                symbol=tick.get('symbol', ''),
                ltp=tick.get('ltp', 0.0),
                ltq=tick.get('ltq', 0),
                volume=tick.get('volume', 0),
                bid=tick.get('bid', 0.0),
                ask=tick.get('ask', 0.0),
                bid_qty=tick.get('bid_qty', 0),
                ask_qty=tick.get('ask_qty', 0),
                oi=tick.get('oi', 0),
                exchange=tick.get('exchange', '')
            )

        with self._lock:
            self._buffer.append(tick)
            self._symbols.add(tick.symbol)
            self._tick_count += 1

            # Check if we should flush
            if (len(self._buffer) >= self.buffer_size or
                time.time() - self._last_flush >= self.flush_interval):
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to disk."""
        with self._lock:
            if not self._buffer:
                return

            buffer = self._buffer
            self._buffer = []
            self._last_flush = time.time()

        # Queue for background writing
        self._write_queue.put(buffer)

    def _writer_loop(self) -> None:
        """Background writer thread."""
        while self._running or not self._write_queue.empty():
            try:
                buffer = self._write_queue.get(timeout=0.1)
                if buffer is None:  # Sentinel
                    break

                self._write_buffer(buffer)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer error: {e}")

    def _write_buffer(self, buffer: List[TickRecord]) -> None:
        """Write buffer to file."""
        if not self._file:
            return

        # Serialize all ticks
        data = b''.join(tick.to_bytes() for tick in buffer)

        # Compress
        compressed = self._compressor.compress(data)

        # Write record header + data
        record_header = struct.pack(
            '<BBQIH',
            RecordType.TICK.value,
            0,  # flags
            buffer[0].timestamp,
            len(compressed),
            self._checksum(compressed)
        )

        self._file.write(record_header)
        self._file.write(compressed)
        self._file.flush()

    def _write_header(self) -> None:
        """Write file header."""
        header = struct.pack(
            '<5sBBBQ48s',
            MAGIC_NUMBER,
            VERSION,
            self.compression.value.encode('utf-8')[0] if self.compression != CompressionType.NONE else 0,
            RecordType.TICK.value,
            int(time.time() * 1_000_000),
            b'\x00' * 48  # Reserved
        )
        self._file.write(header)

    def _checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF

    @property
    def tick_count(self) -> int:
        """Get total tick count."""
        return self._tick_count

    @property
    def symbols(self) -> List[str]:
        """Get recorded symbols."""
        return list(self._symbols)


class EventRecorder:
    """
    Records internal system events from EventBus.

    Captures all events for complete system state replay.
    """

    def __init__(
        self,
        output_path: str,
        compression: CompressionType = CompressionType.LZ4
    ):
        """
        Initialize event recorder.

        Args:
            output_path: Path for output file
            compression: Compression algorithm
        """
        self.output_path = output_path
        self.compression = compression

        self._compressor = Compressor(compression)
        self._file: Optional[BinaryIO] = None
        self._lock = threading.Lock()
        self._event_count = 0
        self._event_types: set = set()

    def start(self) -> None:
        """Start the recorder."""
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        self._file = open(self.output_path, 'wb')
        self._write_header()

        logger.info(f"Event recorder started: {self.output_path}")

    def stop(self) -> None:
        """Stop the recorder."""
        if self._file:
            self._file.close()
            self._file = None

        logger.info(f"Event recorder stopped. Total events: {self._event_count}")

    def record(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "system",
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Record an event.

        Args:
            event_type: Type of event
            payload: Event data
            source: Event source
            correlation_id: Optional correlation ID
        """
        event = EventRecord(
            timestamp=int(time.time() * 1_000_000),
            event_type=event_type,
            source=source,
            payload=payload,
            correlation_id=correlation_id
        )

        with self._lock:
            if not self._file:
                return

            data = event.to_bytes()
            compressed = self._compressor.compress(data)

            record_header = struct.pack(
                '<BBQIH',
                RecordType.EVENT.value,
                0,
                event.timestamp,
                len(compressed),
                self._checksum(compressed)
            )

            self._file.write(record_header)
            self._file.write(compressed)
            self._file.flush()

            self._event_count += 1
            self._event_types.add(event_type)

    def _write_header(self) -> None:
        """Write file header."""
        header = struct.pack(
            '<5sBBBQ48s',
            MAGIC_NUMBER,
            VERSION,
            self.compression.value.encode('utf-8')[0] if self.compression != CompressionType.NONE else 0,
            RecordType.EVENT.value,
            int(time.time() * 1_000_000),
            b'\x00' * 48
        )
        self._file.write(header)

    def _checksum(self, data: bytes) -> int:
        """Calculate simple checksum."""
        return sum(data) & 0xFFFF

    @property
    def event_count(self) -> int:
        """Get total event count."""
        return self._event_count


class FlightRecorder:
    """
    Unified flight recorder combining tick and event recording.

    Provides a single interface for recording complete trading sessions.
    """

    def __init__(
        self,
        base_path: str = "./recordings",
        compression: CompressionType = CompressionType.LZ4,
        tick_buffer_size: int = 10000
    ):
        """
        Initialize flight recorder.

        Args:
            base_path: Base directory for recordings
            compression: Compression algorithm
            tick_buffer_size: Tick buffer size
        """
        self.base_path = base_path
        self.compression = compression
        self.tick_buffer_size = tick_buffer_size

        self._tick_recorder: Optional[TickRecorder] = None
        self._event_recorder: Optional[EventRecorder] = None
        self._session: Optional[RecordingSession] = None
        self._session_path: Optional[str] = None
        self._start_time: Optional[datetime] = None

        os.makedirs(base_path, exist_ok=True)

    def start_session(
        self,
        trading_date: Optional[Union[date, str]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new recording session.

        Args:
            trading_date: Trading date (defaults to today)
            session_id: Optional session ID

        Returns:
            Session path
        """
        if trading_date is None:
            trading_date = date.today()
        elif isinstance(trading_date, str):
            trading_date = date.fromisoformat(trading_date)

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create session directory
        self._session_path = os.path.join(
            self.base_path,
            trading_date.isoformat(),
            session_id
        )
        os.makedirs(self._session_path, exist_ok=True)

        # Start recorders
        self._tick_recorder = TickRecorder(
            os.path.join(self._session_path, "ticks.bin"),
            compression=self.compression,
            buffer_size=self.tick_buffer_size
        )
        self._tick_recorder.start()

        self._event_recorder = EventRecorder(
            os.path.join(self._session_path, "events.bin"),
            compression=self.compression
        )
        self._event_recorder.start()

        self._start_time = datetime.now()

        # Record session start event
        self._event_recorder.record(
            "session_start",
            {
                'trading_date': trading_date.isoformat(),
                'session_id': session_id,
                'compression': self.compression.value
            },
            source="flight_recorder"
        )

        logger.info(f"Recording session started: {self._session_path}")

        return self._session_path

    def end_session(self) -> RecordingSession:
        """
        End the current recording session.

        Returns:
            Session metadata
        """
        if not self._tick_recorder or not self._event_recorder:
            raise RuntimeError("No active session")

        # Record session end event
        self._event_recorder.record(
            "session_end",
            {
                'tick_count': self._tick_recorder.tick_count,
                'event_count': self._event_recorder.event_count
            },
            source="flight_recorder"
        )

        # Stop recorders
        self._tick_recorder.stop()
        self._event_recorder.stop()

        # Calculate file sizes
        tick_size = os.path.getsize(
            os.path.join(self._session_path, "ticks.bin")
        )
        event_size = os.path.getsize(
            os.path.join(self._session_path, "events.bin")
        )

        # Create session metadata
        session = RecordingSession(
            session_id=os.path.basename(self._session_path),
            trading_date=date.fromisoformat(
                os.path.basename(os.path.dirname(self._session_path))
            ),
            start_time=self._start_time,
            end_time=datetime.now(),
            tick_count=self._tick_recorder.tick_count,
            event_count=self._event_recorder.event_count,
            symbols=self._tick_recorder.symbols,
            compression=self.compression,
            file_size=tick_size + event_size,
            checksum=""  # TODO: Calculate overall checksum
        )

        # Save session metadata
        metadata_path = os.path.join(self._session_path, "session.json")
        with open(metadata_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

        logger.info(
            f"Recording session ended. Ticks: {session.tick_count}, "
            f"Events: {session.event_count}, Size: {session.file_size / 1024:.1f} KB"
        )

        self._tick_recorder = None
        self._event_recorder = None
        self._session_path = None

        return session

    def record_tick(self, tick: Union[TickRecord, Dict[str, Any]]) -> None:
        """Record a market tick."""
        if self._tick_recorder:
            self._tick_recorder.record(tick)

    def record_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "system",
        correlation_id: Optional[str] = None
    ) -> None:
        """Record an internal event."""
        if self._event_recorder:
            self._event_recorder.record(event_type, payload, source, correlation_id)

    def record_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str,
        **kwargs
    ) -> None:
        """Record an order event."""
        self.record_event(
            "order",
            {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_type': order_type,
                **kwargs
            },
            source="order_manager"
        )

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        **kwargs
    ) -> None:
        """Record a fill event."""
        self.record_event(
            "fill",
            {
                'order_id': order_id,
                'fill_price': fill_price,
                'fill_quantity': fill_quantity,
                **kwargs
            },
            source="execution"
        )

    def record_signal(
        self,
        signal_type: str,
        symbol: str,
        value: float,
        **kwargs
    ) -> None:
        """Record a trading signal."""
        self.record_event(
            "signal",
            {
                'signal_type': signal_type,
                'symbol': symbol,
                'value': value,
                **kwargs
            },
            source="strategy"
        )

    def record_state(
        self,
        component: str,
        state: Dict[str, Any]
    ) -> None:
        """Record component state snapshot."""
        self.record_event(
            "state_snapshot",
            {'component': component, 'state': state},
            source=component
        )

    def record_error(
        self,
        error_type: str,
        message: str,
        stack_trace: Optional[str] = None,
        **kwargs
    ) -> None:
        """Record an error."""
        self.record_event(
            "error",
            {
                'error_type': error_type,
                'message': message,
                'stack_trace': stack_trace,
                **kwargs
            },
            source="error_handler"
        )

    def add_marker(self, label: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a marker for debugging."""
        self.record_event(
            "marker",
            {'label': label, 'data': data or {}},
            source="debug"
        )

    def list_sessions(
        self,
        trading_date: Optional[Union[date, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available recording sessions.

        Args:
            trading_date: Filter by trading date

        Returns:
            List of session metadata
        """
        sessions = []

        if trading_date:
            if isinstance(trading_date, str):
                trading_date = date.fromisoformat(trading_date)
            date_paths = [os.path.join(self.base_path, trading_date.isoformat())]
        else:
            date_paths = [
                os.path.join(self.base_path, d)
                for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d))
            ]

        for date_path in date_paths:
            if not os.path.exists(date_path):
                continue

            for session_dir in os.listdir(date_path):
                metadata_path = os.path.join(date_path, session_dir, "session.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        sessions.append(json.load(f))

        return sorted(sessions, key=lambda s: s['start_time'], reverse=True)


class MarketReplayer:
    """
    Replays recorded market data for strategy debugging.

    Provides exact replay of trading sessions with speed control.
    """

    def __init__(
        self,
        session_path: str,
        compression: Optional[CompressionType] = None
    ):
        """
        Initialize replayer.

        Args:
            session_path: Path to recording session
            compression: Compression type (auto-detected if None)
        """
        self.session_path = session_path

        # Load session metadata
        metadata_path = os.path.join(session_path, "session.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
            compression = CompressionType(self._metadata.get('compression', 'lz4'))
        else:
            self._metadata = {}

        self._compression = compression or CompressionType.LZ4
        self._compressor = Compressor(self._compression)

        self._state = PlaybackState.STOPPED
        self._speed = 1.0
        self._position = 0
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def play(
        self,
        speed: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Iterator[Union[TickRecord, EventRecord]]:
        """
        Play back recorded data.

        Args:
            speed: Playback speed multiplier (1.0 = real-time)
            start_time: Start from this time
            end_time: End at this time
            symbols: Filter to these symbols

        Yields:
            Recorded ticks and events in chronological order
        """
        self._state = PlaybackState.PLAYING
        self._speed = speed

        # Load tick file
        tick_path = os.path.join(self.session_path, "ticks.bin")
        event_path = os.path.join(self.session_path, "events.bin")

        ticks = list(self._read_ticks(tick_path))
        events = list(self._read_events(event_path))

        # Merge and sort by timestamp
        all_records = []

        for tick in ticks:
            if symbols and tick.symbol not in symbols:
                continue
            if start_time and tick.datetime < start_time:
                continue
            if end_time and tick.datetime > end_time:
                continue
            all_records.append(('tick', tick.timestamp, tick))

        for event in events:
            if start_time and event.datetime < start_time:
                continue
            if end_time and event.datetime > end_time:
                continue
            all_records.append(('event', event.timestamp, event))

        all_records.sort(key=lambda r: r[1])

        # Playback with timing
        last_ts = None

        for record_type, timestamp, record in all_records:
            if self._state == PlaybackState.STOPPED:
                break

            while self._state == PlaybackState.PAUSED:
                time.sleep(0.01)

            # Simulate timing
            if speed > 0 and last_ts is not None:
                delay = (timestamp - last_ts) / 1_000_000 / speed
                if delay > 0:
                    time.sleep(min(delay, 1.0))  # Cap at 1 second

            last_ts = timestamp

            # Notify callbacks
            if record_type == 'tick':
                for callback in self._callbacks.get('tick', []):
                    callback(record)
            else:
                for callback in self._callbacks.get('event', []):
                    callback(record)
                for callback in self._callbacks.get(record.event_type, []):
                    callback(record)

            yield record

        self._state = PlaybackState.STOPPED

    def _read_ticks(self, path: str) -> Iterator[TickRecord]:
        """Read ticks from binary file."""
        if not os.path.exists(path):
            return

        with open(path, 'rb') as f:
            # Skip header
            f.seek(HEADER_SIZE)

            while True:
                # Read record header
                header_data = f.read(RECORD_HEADER_SIZE)
                if len(header_data) < RECORD_HEADER_SIZE:
                    break

                record_type, flags, timestamp, length, checksum = struct.unpack(
                    '<BBQIH', header_data
                )

                if record_type != RecordType.TICK.value:
                    f.seek(length, 1)  # Skip non-tick records
                    continue

                # Read compressed data
                compressed = f.read(length)
                if len(compressed) < length:
                    break

                # Decompress
                data = self._compressor.decompress(compressed)

                # Parse ticks (each tick is 112 bytes)
                tick_size = 112
                for i in range(0, len(data), tick_size):
                    if i + tick_size <= len(data):
                        yield TickRecord.from_bytes(data[i:i + tick_size])

    def _read_events(self, path: str) -> Iterator[EventRecord]:
        """Read events from binary file."""
        if not os.path.exists(path):
            return

        with open(path, 'rb') as f:
            # Skip header
            f.seek(HEADER_SIZE)

            while True:
                # Read record header
                header_data = f.read(RECORD_HEADER_SIZE)
                if len(header_data) < RECORD_HEADER_SIZE:
                    break

                record_type, flags, timestamp, length, checksum = struct.unpack(
                    '<BBQIH', header_data
                )

                if record_type != RecordType.EVENT.value:
                    f.seek(length, 1)
                    continue

                # Read compressed data
                compressed = f.read(length)
                if len(compressed) < length:
                    break

                # Decompress and parse
                data = self._compressor.decompress(compressed)
                yield EventRecord.from_bytes(data)

    def pause(self) -> None:
        """Pause playback."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED

    def resume(self) -> None:
        """Resume playback."""
        if self._state == PlaybackState.PAUSED:
            self._state = PlaybackState.PLAYING

    def stop(self) -> None:
        """Stop playback."""
        self._state = PlaybackState.STOPPED

    def set_speed(self, speed: float) -> None:
        """Set playback speed."""
        self._speed = speed

    def on_tick(self, callback: Callable[[TickRecord], None]) -> None:
        """Register tick callback."""
        self._callbacks['tick'].append(callback)

    def on_event(
        self,
        callback: Callable[[EventRecord], None],
        event_type: Optional[str] = None
    ) -> None:
        """Register event callback."""
        if event_type:
            self._callbacks[event_type].append(callback)
        else:
            self._callbacks['event'].append(callback)

    def get_ticks(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TickRecord]:
        """
        Get ticks without playback timing.

        Args:
            symbol: Filter to symbol
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of ticks
        """
        tick_path = os.path.join(self.session_path, "ticks.bin")
        ticks = []

        for tick in self._read_ticks(tick_path):
            if symbol and tick.symbol != symbol:
                continue
            if start_time and tick.datetime < start_time:
                continue
            if end_time and tick.datetime > end_time:
                continue
            ticks.append(tick)

        return ticks

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EventRecord]:
        """
        Get events without playback timing.

        Args:
            event_type: Filter to event type
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of events
        """
        event_path = os.path.join(self.session_path, "events.bin")
        events = []

        for event in self._read_events(event_path):
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.datetime < start_time:
                continue
            if end_time and event.datetime > end_time:
                continue
            events.append(event)

        return events

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get session metadata."""
        return self._metadata

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state


class DebugAnalyzer:
    """
    Analyzes recorded sessions for debugging.

    Provides tools for investigating failed strategies.
    """

    def __init__(self, session_path: str):
        """
        Initialize analyzer.

        Args:
            session_path: Path to recording session
        """
        self.replayer = MarketReplayer(session_path)

    def find_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None
    ) -> List[EventRecord]:
        """Find order events."""
        events = self.replayer.get_events(event_type="order")

        if symbol:
            events = [e for e in events if e.payload.get('symbol') == symbol]
        if side:
            events = [e for e in events if e.payload.get('side') == side]

        return events

    def find_fills(self, order_id: Optional[str] = None) -> List[EventRecord]:
        """Find fill events."""
        events = self.replayer.get_events(event_type="fill")

        if order_id:
            events = [e for e in events if e.payload.get('order_id') == order_id]

        return events

    def find_errors(self) -> List[EventRecord]:
        """Find all error events."""
        return self.replayer.get_events(event_type="error")

    def find_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[EventRecord]:
        """Find signal events."""
        events = self.replayer.get_events(event_type="signal")

        if symbol:
            events = [e for e in events if e.payload.get('symbol') == symbol]
        if signal_type:
            events = [e for e in events if e.payload.get('signal_type') == signal_type]

        return events

    def get_price_at_time(
        self,
        symbol: str,
        target_time: datetime
    ) -> Optional[TickRecord]:
        """Get the price at a specific time."""
        ticks = self.replayer.get_ticks(symbol=symbol, end_time=target_time)
        return ticks[-1] if ticks else None

    def get_price_around_order(
        self,
        order_event: EventRecord,
        window_seconds: int = 60
    ) -> List[TickRecord]:
        """Get prices around an order."""
        order_time = order_event.datetime
        symbol = order_event.payload.get('symbol')

        return self.replayer.get_ticks(
            symbol=symbol,
            start_time=order_time - timedelta(seconds=window_seconds),
            end_time=order_time + timedelta(seconds=window_seconds)
        )

    def calculate_slippage(self, order_event: EventRecord) -> Optional[float]:
        """Calculate slippage for an order."""
        order_id = order_event.payload.get('order_id')
        order_price = order_event.payload.get('price')
        side = order_event.payload.get('side')

        fills = self.find_fills(order_id)
        if not fills:
            return None

        total_qty = 0
        weighted_price = 0

        for fill in fills:
            qty = fill.payload.get('fill_quantity', 0)
            price = fill.payload.get('fill_price', 0)
            total_qty += qty
            weighted_price += qty * price

        if total_qty == 0:
            return None

        avg_fill_price = weighted_price / total_qty

        if side == 'buy':
            return avg_fill_price - order_price
        else:
            return order_price - avg_fill_price

    def generate_report(self) -> Dict[str, Any]:
        """Generate debugging report."""
        metadata = self.replayer.metadata

        orders = self.find_orders()
        fills = self.find_fills()
        errors = self.find_errors()
        signals = self.find_signals()

        return {
            'session': metadata,
            'summary': {
                'total_orders': len(orders),
                'total_fills': len(fills),
                'total_errors': len(errors),
                'total_signals': len(signals)
            },
            'errors': [
                {
                    'time': e.datetime.isoformat(),
                    'type': e.payload.get('error_type'),
                    'message': e.payload.get('message')
                }
                for e in errors
            ],
            'order_summary': {
                'buy_orders': len([o for o in orders if o.payload.get('side') == 'buy']),
                'sell_orders': len([o for o in orders if o.payload.get('side') == 'sell'])
            }
        }


# Convenience functions
_default_recorder: Optional[FlightRecorder] = None


def get_flight_recorder(base_path: str = "./recordings") -> FlightRecorder:
    """Get or create default flight recorder."""
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = FlightRecorder(base_path)
    return _default_recorder


def set_flight_recorder(recorder: FlightRecorder) -> None:
    """Set default flight recorder."""
    global _default_recorder
    _default_recorder = recorder


def start_recording(trading_date: Optional[str] = None) -> str:
    """Start recording session."""
    return get_flight_recorder().start_session(trading_date)


def stop_recording() -> RecordingSession:
    """Stop recording session."""
    return get_flight_recorder().end_session()


def record_tick(tick: Union[TickRecord, Dict[str, Any]]) -> None:
    """Record a tick."""
    recorder = get_flight_recorder()
    if recorder._tick_recorder:
        recorder.record_tick(tick)


def record_event(event_type: str, payload: Dict[str, Any], **kwargs) -> None:
    """Record an event."""
    recorder = get_flight_recorder()
    if recorder._event_recorder:
        recorder.record_event(event_type, payload, **kwargs)
