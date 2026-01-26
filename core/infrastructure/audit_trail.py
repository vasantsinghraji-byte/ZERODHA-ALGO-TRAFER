# -*- coding: utf-8 -*-
"""
Trade Audit Trail - Immutable Compliance Logging
=================================================
Production-grade audit trail for regulatory compliance.

Features:
- Immutable append-only trade log
- Cryptographic hash chain for tamper detection
- Support for regulatory reporting formats (NSE, BSE, SEBI)
- Verification and integrity checking
- Export to multiple formats (CSV, JSON, XML)

Example:
    >>> from core.infrastructure.audit_trail import AuditTrail
    >>>
    >>> # Create audit trail
    >>> audit = AuditTrail("./audit_logs")
    >>>
    >>> # Log a trade
    >>> audit.log_trade(
    ...     order_id="ORD123",
    ...     symbol="RELIANCE",
    ...     side="buy",
    ...     quantity=100,
    ...     price=2500.0
    ... )
    >>>
    >>> # Verify integrity
    >>> is_valid = audit.verify_integrity()
    >>>
    >>> # Export for regulatory reporting
    >>> audit.export_sebi_format("2024-01-15", "sebi_report.csv")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator, BinaryIO, Tuple
from datetime import datetime, date, timedelta
from collections import OrderedDict
import hashlib
import hmac
import json
import csv
import os
import threading
import logging
import uuid
import struct
import io

try:
    from xml.etree import ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    ORDER_PLACED = "order_placed"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    MARGIN_CALL = "margin_call"
    SYSTEM_EVENT = "system_event"
    CONFIG_CHANGE = "config_change"
    USER_ACTION = "user_action"


class ReportFormat(Enum):
    """Regulatory report formats."""
    NSE_TRADE = "nse_trade"
    BSE_TRADE = "bse_trade"
    SEBI_AUDIT = "sebi_audit"
    CSV = "csv"
    JSON = "json"
    XML = "xml"


class IntegrityStatus(Enum):
    """Integrity verification status."""
    VALID = "valid"
    TAMPERED = "tampered"
    MISSING_RECORDS = "missing_records"
    HASH_MISMATCH = "hash_mismatch"
    SEQUENCE_ERROR = "sequence_error"


@dataclass
class AuditRecord:
    """
    Single audit record with cryptographic integrity.

    Each record contains a hash of the previous record,
    creating an immutable chain similar to blockchain.
    """
    sequence_number: int
    timestamp: datetime
    event_type: AuditEventType
    event_id: str
    previous_hash: str
    record_hash: str

    # Trade details
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    filled_quantity: Optional[int] = None
    filled_price: Optional[float] = None

    # Additional context
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sequence_number': self.sequence_number,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'previous_hash': self.previous_hash,
            'record_hash': self.record_hash,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'user_id': self.user_id,
            'strategy_id': self.strategy_id,
            'client_order_id': self.client_order_id,
            'broker_order_id': self.broker_order_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditRecord':
        """Create from dictionary."""
        return cls(
            sequence_number=data['sequence_number'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=AuditEventType(data['event_type']),
            event_id=data['event_id'],
            previous_hash=data['previous_hash'],
            record_hash=data['record_hash'],
            order_id=data.get('order_id'),
            symbol=data.get('symbol'),
            exchange=data.get('exchange'),
            side=data.get('side'),
            quantity=data.get('quantity'),
            price=data.get('price'),
            filled_quantity=data.get('filled_quantity'),
            filled_price=data.get('filled_price'),
            user_id=data.get('user_id'),
            strategy_id=data.get('strategy_id'),
            client_order_id=data.get('client_order_id'),
            broker_order_id=data.get('broker_order_id'),
            metadata=data.get('metadata', {})
        )

    def to_csv_row(self) -> List[str]:
        """Convert to CSV row."""
        return [
            str(self.sequence_number),
            self.timestamp.isoformat(),
            self.event_type.value,
            self.event_id,
            self.order_id or '',
            self.symbol or '',
            self.exchange or '',
            self.side or '',
            str(self.quantity) if self.quantity else '',
            f"{self.price:.4f}" if self.price else '',
            str(self.filled_quantity) if self.filled_quantity else '',
            f"{self.filled_price:.4f}" if self.filled_price else '',
            self.user_id or '',
            self.strategy_id or '',
            self.record_hash
        ]

    @staticmethod
    def csv_headers() -> List[str]:
        """Get CSV column headers."""
        return [
            'sequence_number', 'timestamp', 'event_type', 'event_id',
            'order_id', 'symbol', 'exchange', 'side', 'quantity', 'price',
            'filled_quantity', 'filled_price', 'user_id', 'strategy_id',
            'record_hash'
        ]


@dataclass
class IntegrityReport:
    """Result of integrity verification."""
    status: IntegrityStatus
    records_checked: int
    first_invalid_sequence: Optional[int]
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    error_message: str
    verification_time: datetime
    chain_start: int
    chain_end: int

    def is_valid(self) -> bool:
        """Check if audit trail is valid."""
        return self.status == IntegrityStatus.VALID


class HashChain:
    """
    Cryptographic hash chain for tamper detection.

    Uses SHA-256 to create linked hashes between records.
    """

    GENESIS_HASH = "0" * 64  # Initial hash for first record

    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize hash chain.

        Args:
            secret_key: Optional HMAC key for additional security
        """
        self.secret_key = secret_key

    def compute_hash(
        self,
        record_data: str,
        previous_hash: str
    ) -> str:
        """
        Compute hash for a record.

        Args:
            record_data: Serialized record data
            previous_hash: Hash of previous record

        Returns:
            SHA-256 hash hex string
        """
        # Combine record data with previous hash
        combined = f"{previous_hash}:{record_data}"

        if self.secret_key:
            # Use HMAC for additional security
            return hmac.new(
                self.secret_key,
                combined.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        else:
            return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def verify_chain(
        self,
        records: List[AuditRecord]
    ) -> IntegrityReport:
        """
        Verify integrity of record chain.

        Args:
            records: List of records to verify

        Returns:
            Integrity verification report
        """
        if not records:
            return IntegrityReport(
                status=IntegrityStatus.VALID,
                records_checked=0,
                first_invalid_sequence=None,
                expected_hash=None,
                actual_hash=None,
                error_message="Empty chain",
                verification_time=datetime.now(),
                chain_start=0,
                chain_end=0
            )

        # Sort by sequence number
        sorted_records = sorted(records, key=lambda r: r.sequence_number)

        # Check sequence continuity
        for i, record in enumerate(sorted_records):
            expected_seq = sorted_records[0].sequence_number + i
            if record.sequence_number != expected_seq:
                return IntegrityReport(
                    status=IntegrityStatus.SEQUENCE_ERROR,
                    records_checked=i,
                    first_invalid_sequence=record.sequence_number,
                    expected_hash=None,
                    actual_hash=None,
                    error_message=f"Expected sequence {expected_seq}, got {record.sequence_number}",
                    verification_time=datetime.now(),
                    chain_start=sorted_records[0].sequence_number,
                    chain_end=sorted_records[-1].sequence_number
                )

        # Verify hash chain
        previous_hash = self.GENESIS_HASH

        for i, record in enumerate(sorted_records):
            # Verify previous hash reference
            if i > 0 and record.previous_hash != sorted_records[i - 1].record_hash:
                return IntegrityReport(
                    status=IntegrityStatus.HASH_MISMATCH,
                    records_checked=i,
                    first_invalid_sequence=record.sequence_number,
                    expected_hash=sorted_records[i - 1].record_hash,
                    actual_hash=record.previous_hash,
                    error_message="Previous hash mismatch - possible tampering",
                    verification_time=datetime.now(),
                    chain_start=sorted_records[0].sequence_number,
                    chain_end=sorted_records[-1].sequence_number
                )

            # Recompute and verify record hash
            record_data = self._serialize_for_hash(record)
            expected_hash = self.compute_hash(record_data, record.previous_hash)

            if record.record_hash != expected_hash:
                return IntegrityReport(
                    status=IntegrityStatus.TAMPERED,
                    records_checked=i,
                    first_invalid_sequence=record.sequence_number,
                    expected_hash=expected_hash,
                    actual_hash=record.record_hash,
                    error_message="Record hash mismatch - data tampered",
                    verification_time=datetime.now(),
                    chain_start=sorted_records[0].sequence_number,
                    chain_end=sorted_records[-1].sequence_number
                )

            previous_hash = record.record_hash

        return IntegrityReport(
            status=IntegrityStatus.VALID,
            records_checked=len(sorted_records),
            first_invalid_sequence=None,
            expected_hash=None,
            actual_hash=None,
            error_message="Chain integrity verified",
            verification_time=datetime.now(),
            chain_start=sorted_records[0].sequence_number,
            chain_end=sorted_records[-1].sequence_number
        )

    def _serialize_for_hash(self, record: AuditRecord) -> str:
        """Serialize record for hashing (excluding the hash field)."""
        data = {
            'sequence_number': record.sequence_number,
            'timestamp': record.timestamp.isoformat(),
            'event_type': record.event_type.value,
            'event_id': record.event_id,
            'order_id': record.order_id,
            'symbol': record.symbol,
            'exchange': record.exchange,
            'side': record.side,
            'quantity': record.quantity,
            'price': record.price,
            'filled_quantity': record.filled_quantity,
            'filled_price': record.filled_price,
            'user_id': record.user_id,
            'strategy_id': record.strategy_id,
            'metadata': record.metadata
        }
        return json.dumps(data, sort_keys=True, default=str)


class AuditStorage(ABC):
    """Abstract base for audit log storage."""

    @abstractmethod
    def append(self, record: AuditRecord) -> bool:
        """Append record to storage."""
        pass

    @abstractmethod
    def read_all(self) -> List[AuditRecord]:
        """Read all records."""
        pass

    @abstractmethod
    def read_range(
        self,
        start_seq: int,
        end_seq: int
    ) -> List[AuditRecord]:
        """Read records in sequence range."""
        pass

    @abstractmethod
    def get_last_record(self) -> Optional[AuditRecord]:
        """Get the last record."""
        pass

    @abstractmethod
    def get_record_count(self) -> int:
        """Get total record count."""
        pass


class FileAuditStorage(AuditStorage):
    """
    File-based append-only audit storage.

    Uses append-only file access with file locking
    to ensure immutability.
    """

    def __init__(
        self,
        base_path: str,
        rotate_daily: bool = True
    ):
        """
        Initialize file storage.

        Args:
            base_path: Base directory for audit logs
            rotate_daily: Create new file each day
        """
        self.base_path = base_path
        self.rotate_daily = rotate_daily

        os.makedirs(base_path, exist_ok=True)

        self._lock = threading.Lock()
        self._current_file: Optional[str] = None
        self._record_count = 0

    def _get_current_file(self) -> str:
        """Get current log file path."""
        if self.rotate_daily:
            filename = f"audit_{date.today().isoformat()}.jsonl"
        else:
            filename = "audit.jsonl"

        return os.path.join(self.base_path, filename)

    def append(self, record: AuditRecord) -> bool:
        """Append record to log file."""
        file_path = self._get_current_file()

        with self._lock:
            try:
                # Open in append mode only
                with open(file_path, 'a') as f:
                    # Write as JSON line
                    json.dump(record.to_dict(), f, default=str)
                    f.write('\n')
                    f.flush()
                    os.fsync(f.fileno())  # Ensure written to disk

                self._record_count += 1
                return True

            except Exception as e:
                logger.error(f"Failed to append audit record: {e}")
                return False

    def read_all(self) -> List[AuditRecord]:
        """Read all records from all log files."""
        records = []

        for filename in sorted(os.listdir(self.base_path)):
            if filename.startswith('audit_') and filename.endswith('.jsonl'):
                file_path = os.path.join(self.base_path, filename)
                records.extend(self._read_file(file_path))

        return sorted(records, key=lambda r: r.sequence_number)

    def read_range(
        self,
        start_seq: int,
        end_seq: int
    ) -> List[AuditRecord]:
        """Read records in sequence range."""
        all_records = self.read_all()
        return [
            r for r in all_records
            if start_seq <= r.sequence_number <= end_seq
        ]

    def get_last_record(self) -> Optional[AuditRecord]:
        """Get the last record."""
        # Read current file in reverse
        file_path = self._get_current_file()

        if not os.path.exists(file_path):
            # Check previous files
            files = sorted([
                f for f in os.listdir(self.base_path)
                if f.startswith('audit_') and f.endswith('.jsonl')
            ], reverse=True)

            if not files:
                return None

            file_path = os.path.join(self.base_path, files[0])

        # Read last line
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        data = json.loads(last_line)
                        return AuditRecord.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to read last record: {e}")

        return None

    def get_record_count(self) -> int:
        """Get total record count."""
        count = 0
        for filename in os.listdir(self.base_path):
            if filename.startswith('audit_') and filename.endswith('.jsonl'):
                file_path = os.path.join(self.base_path, filename)
                with open(file_path, 'r') as f:
                    count += sum(1 for _ in f)
        return count

    def _read_file(self, file_path: str) -> List[AuditRecord]:
        """Read records from a single file."""
        records = []

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        records.append(AuditRecord.from_dict(data))
        except Exception as e:
            logger.error(f"Failed to read audit file {file_path}: {e}")

        return records


class RegulatoryReporter:
    """
    Generates regulatory reports in required formats.

    Supports NSE, BSE, and SEBI reporting requirements.
    """

    # NSE Trade Report columns
    NSE_COLUMNS = [
        'TRADE_NO', 'TRADE_DATE', 'TRADE_TIME', 'SYMBOL', 'SERIES',
        'BUY_SELL', 'QTY', 'PRICE', 'ORDER_NO', 'CLIENT_CODE',
        'MEMBER_CODE', 'EXCHANGE_ORDER_NO'
    ]

    # BSE Trade Report columns
    BSE_COLUMNS = [
        'TRADE_ID', 'TRADE_DATE', 'TRADE_TIME', 'SCRIP_CODE', 'SCRIP_NAME',
        'BUY_SELL_FLAG', 'QUANTITY', 'RATE', 'ORDER_NUMBER', 'CLIENT_ID'
    ]

    # SEBI Audit columns
    SEBI_COLUMNS = [
        'AUDIT_SEQ', 'TIMESTAMP', 'EVENT_TYPE', 'ORDER_ID', 'SYMBOL',
        'EXCHANGE', 'SIDE', 'QUANTITY', 'PRICE', 'FILLED_QTY',
        'FILLED_PRICE', 'USER_ID', 'STRATEGY_ID', 'HASH'
    ]

    def __init__(self, member_code: str = "", client_code: str = ""):
        """
        Initialize reporter.

        Args:
            member_code: Exchange member code
            client_code: Default client code
        """
        self.member_code = member_code
        self.client_code = client_code

    def export_nse_format(
        self,
        records: List[AuditRecord],
        output_path: str
    ) -> int:
        """
        Export to NSE trade report format.

        Args:
            records: Audit records to export
            output_path: Output file path

        Returns:
            Number of records exported
        """
        trade_records = [
            r for r in records
            if r.event_type == AuditEventType.ORDER_FILLED and r.exchange == 'NSE'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(self.NSE_COLUMNS)

            for i, record in enumerate(trade_records, 1):
                writer.writerow([
                    i,  # TRADE_NO
                    record.timestamp.strftime('%Y%m%d'),
                    record.timestamp.strftime('%H%M%S'),
                    record.symbol or '',
                    'EQ',  # SERIES
                    'B' if record.side == 'buy' else 'S',
                    record.filled_quantity or record.quantity or 0,
                    record.filled_price or record.price or 0,
                    record.order_id or '',
                    self.client_code,
                    self.member_code,
                    record.broker_order_id or ''
                ])

        return len(trade_records)

    def export_bse_format(
        self,
        records: List[AuditRecord],
        output_path: str
    ) -> int:
        """Export to BSE trade report format."""
        trade_records = [
            r for r in records
            if r.event_type == AuditEventType.ORDER_FILLED and r.exchange == 'BSE'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(self.BSE_COLUMNS)

            for i, record in enumerate(trade_records, 1):
                writer.writerow([
                    i,
                    record.timestamp.strftime('%d-%m-%Y'),
                    record.timestamp.strftime('%H:%M:%S'),
                    record.symbol or '',
                    record.symbol or '',
                    'B' if record.side == 'buy' else 'S',
                    record.filled_quantity or record.quantity or 0,
                    record.filled_price or record.price or 0,
                    record.order_id or '',
                    self.client_code
                ])

        return len(trade_records)

    def export_sebi_audit(
        self,
        records: List[AuditRecord],
        output_path: str
    ) -> int:
        """Export to SEBI audit format."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.SEBI_COLUMNS)

            for record in records:
                writer.writerow([
                    record.sequence_number,
                    record.timestamp.isoformat(),
                    record.event_type.value,
                    record.order_id or '',
                    record.symbol or '',
                    record.exchange or '',
                    record.side or '',
                    record.quantity or '',
                    record.price or '',
                    record.filled_quantity or '',
                    record.filled_price or '',
                    record.user_id or '',
                    record.strategy_id or '',
                    record.record_hash
                ])

        return len(records)

    def export_json(
        self,
        records: List[AuditRecord],
        output_path: str,
        pretty: bool = True
    ) -> int:
        """Export to JSON format."""
        data = [record.to_dict() for record in records]

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

        return len(records)

    def export_csv(
        self,
        records: List[AuditRecord],
        output_path: str
    ) -> int:
        """Export to generic CSV format."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(AuditRecord.csv_headers())

            for record in records:
                writer.writerow(record.to_csv_row())

        return len(records)

    def export_xml(
        self,
        records: List[AuditRecord],
        output_path: str
    ) -> int:
        """Export to XML format."""
        if not XML_AVAILABLE:
            raise RuntimeError("XML support not available")

        root = ET.Element('AuditTrail')
        root.set('generated', datetime.now().isoformat())
        root.set('record_count', str(len(records)))

        for record in records:
            record_elem = ET.SubElement(root, 'Record')
            record_elem.set('sequence', str(record.sequence_number))

            for key, value in record.to_dict().items():
                if value is not None and key != 'metadata':
                    elem = ET.SubElement(record_elem, key)
                    elem.text = str(value)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

        return len(records)


class AuditTrail:
    """
    Main audit trail system.

    Provides complete audit logging with integrity verification
    and regulatory reporting capabilities.
    """

    def __init__(
        self,
        storage_path: str,
        secret_key: Optional[bytes] = None,
        member_code: str = "",
        client_code: str = ""
    ):
        """
        Initialize audit trail.

        Args:
            storage_path: Path for audit log storage
            secret_key: Optional HMAC key for hash chain
            member_code: Exchange member code
            client_code: Default client code
        """
        self.storage = FileAuditStorage(storage_path)
        self.hash_chain = HashChain(secret_key)
        self.reporter = RegulatoryReporter(member_code, client_code)

        self._lock = threading.Lock()
        self._sequence_number = 0
        self._last_hash = HashChain.GENESIS_HASH

        # Initialize from existing records
        self._init_from_storage()

    def _init_from_storage(self) -> None:
        """Initialize state from existing storage."""
        last_record = self.storage.get_last_record()
        if last_record:
            self._sequence_number = last_record.sequence_number
            self._last_hash = last_record.record_hash

    def log_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        event_type: AuditEventType = AuditEventType.ORDER_PLACED,
        exchange: str = "NSE",
        filled_quantity: Optional[int] = None,
        filled_price: Optional[float] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """
        Log a trade event.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            price: Order price
            event_type: Type of event
            exchange: Exchange name
            filled_quantity: Filled quantity (for fills)
            filled_price: Fill price
            user_id: User identifier
            strategy_id: Strategy identifier
            broker_order_id: Broker's order ID
            metadata: Additional metadata

        Returns:
            Created audit record
        """
        return self._create_record(
            event_type=event_type,
            order_id=order_id,
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            price=price,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            user_id=user_id,
            strategy_id=strategy_id,
            broker_order_id=broker_order_id,
            metadata=metadata or {}
        )

    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        **kwargs
    ) -> AuditRecord:
        """Log order placed event."""
        return self.log_trade(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            event_type=AuditEventType.ORDER_PLACED,
            **kwargs
        )

    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        filled_quantity: int,
        filled_price: float,
        **kwargs
    ) -> AuditRecord:
        """Log order filled event."""
        return self.log_trade(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            event_type=AuditEventType.ORDER_FILLED,
            **kwargs
        )

    def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        **kwargs
    ) -> AuditRecord:
        """Log order cancelled event."""
        return self._create_record(
            event_type=AuditEventType.ORDER_CANCELLED,
            order_id=order_id,
            symbol=symbol,
            **kwargs
        )

    def log_system_event(
        self,
        event_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Log system event."""
        return self._create_record(
            event_type=AuditEventType.SYSTEM_EVENT,
            metadata={'description': event_description, **(metadata or {})}
        )

    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user_id: Optional[str] = None
    ) -> AuditRecord:
        """Log configuration change."""
        return self._create_record(
            event_type=AuditEventType.CONFIG_CHANGE,
            user_id=user_id,
            metadata={
                'config_key': config_key,
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )

    def _create_record(
        self,
        event_type: AuditEventType,
        **kwargs
    ) -> AuditRecord:
        """Create and store an audit record."""
        with self._lock:
            self._sequence_number += 1

            # Create record without hash first
            record = AuditRecord(
                sequence_number=self._sequence_number,
                timestamp=datetime.now(),
                event_type=event_type,
                event_id=str(uuid.uuid4()),
                previous_hash=self._last_hash,
                record_hash="",  # Will be computed
                **kwargs
            )

            # Compute hash
            record_data = self.hash_chain._serialize_for_hash(record)
            record.record_hash = self.hash_chain.compute_hash(
                record_data, self._last_hash
            )

            # Store
            if not self.storage.append(record):
                raise RuntimeError("Failed to store audit record")

            self._last_hash = record.record_hash

        logger.debug(
            f"Audit record created: seq={record.sequence_number}, "
            f"type={event_type.value}"
        )

        return record

    def verify_integrity(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None
    ) -> IntegrityReport:
        """
        Verify integrity of audit trail.

        Args:
            start_seq: Start sequence (default: beginning)
            end_seq: End sequence (default: end)

        Returns:
            Integrity verification report
        """
        if start_seq is not None and end_seq is not None:
            records = self.storage.read_range(start_seq, end_seq)
        else:
            records = self.storage.read_all()

        return self.hash_chain.verify_chain(records)

    def get_records(
        self,
        event_type: Optional[AuditEventType] = None,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        order_id: Optional[str] = None
    ) -> List[AuditRecord]:
        """
        Get records with filters.

        Args:
            event_type: Filter by event type
            symbol: Filter by symbol
            start_date: Start date filter
            end_date: End date filter
            order_id: Filter by order ID

        Returns:
            Filtered records
        """
        records = self.storage.read_all()

        if event_type:
            records = [r for r in records if r.event_type == event_type]

        if symbol:
            records = [r for r in records if r.symbol == symbol]

        if start_date:
            records = [r for r in records if r.timestamp.date() >= start_date]

        if end_date:
            records = [r for r in records if r.timestamp.date() <= end_date]

        if order_id:
            records = [r for r in records if r.order_id == order_id]

        return records

    def export(
        self,
        output_path: str,
        format: ReportFormat = ReportFormat.CSV,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> int:
        """
        Export audit trail to file.

        Args:
            output_path: Output file path
            format: Export format
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Number of records exported
        """
        records = self.get_records(start_date=start_date, end_date=end_date)

        if format == ReportFormat.NSE_TRADE:
            return self.reporter.export_nse_format(records, output_path)
        elif format == ReportFormat.BSE_TRADE:
            return self.reporter.export_bse_format(records, output_path)
        elif format == ReportFormat.SEBI_AUDIT:
            return self.reporter.export_sebi_audit(records, output_path)
        elif format == ReportFormat.JSON:
            return self.reporter.export_json(records, output_path)
        elif format == ReportFormat.XML:
            return self.reporter.export_xml(records, output_path)
        else:  # CSV
            return self.reporter.export_csv(records, output_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        records = self.storage.read_all()

        event_counts = {}
        for record in records:
            event_type = record.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        symbols = set(r.symbol for r in records if r.symbol)

        return {
            'total_records': len(records),
            'event_counts': event_counts,
            'unique_symbols': list(symbols),
            'first_record': records[0].timestamp.isoformat() if records else None,
            'last_record': records[-1].timestamp.isoformat() if records else None,
            'last_sequence': self._sequence_number,
            'integrity_status': self.verify_integrity().status.value
        }


# Convenience functions
_default_audit: Optional[AuditTrail] = None


def get_audit_trail(storage_path: str = "./audit_logs") -> AuditTrail:
    """Get or create default audit trail."""
    global _default_audit
    if _default_audit is None:
        _default_audit = AuditTrail(storage_path)
    return _default_audit


def set_audit_trail(audit: AuditTrail) -> None:
    """Set default audit trail."""
    global _default_audit
    _default_audit = audit


def log_trade(
    order_id: str,
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    **kwargs
) -> AuditRecord:
    """Log trade using default audit trail."""
    return get_audit_trail().log_trade(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs
    )


def verify_audit_integrity() -> IntegrityReport:
    """Verify integrity using default audit trail."""
    return get_audit_trail().verify_integrity()
