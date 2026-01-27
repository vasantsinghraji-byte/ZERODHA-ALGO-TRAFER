"""
IPC Message Definitions for Cross-Process Communication.

This module defines the message protocol for communication between
the UI process and the Trading Engine process. Using multiprocessing
bypasses Python's GIL, enabling true parallelism.

Architecture:
    UI Process  <---->  Engine Process
       |                    |
       |   multiprocessing  |
       |      .Queue        |
       |                    |
    Tkinter              EventBus
    Dashboard            Strategies
    User Input           Order Execution
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class Commands(str, Enum):
    """Commands sent from UI to Engine process."""

    # Engine lifecycle
    START_ENGINE = 'START_ENGINE'
    STOP_ENGINE = 'STOP_ENGINE'
    PAUSE = 'PAUSE'
    RESUME = 'RESUME'

    # Trading operations
    PLACE_ORDER = 'PLACE_ORDER'
    CANCEL_ORDER = 'CANCEL_ORDER'
    CLOSE_POSITION = 'CLOSE_POSITION'
    CLOSE_ALL_POSITIONS = 'CLOSE_ALL_POSITIONS'

    # Configuration
    UPDATE_CONFIG = 'UPDATE_CONFIG'
    SET_STRATEGY = 'SET_STRATEGY'

    # Status requests (synchronous - expect response)
    GET_STATUS = 'GET_STATUS'
    GET_POSITIONS = 'GET_POSITIONS'
    GET_ORDERS = 'GET_ORDERS'
    GET_PNL = 'GET_PNL'

    # Data subscriptions
    SUBSCRIBE_SYMBOL = 'SUBSCRIBE_SYMBOL'
    UNSUBSCRIBE_SYMBOL = 'UNSUBSCRIBE_SYMBOL'


class Events(str, Enum):
    """Events sent from Engine to UI process."""

    # Market data
    TICK_UPDATE = 'TICK_UPDATE'
    BAR_UPDATE = 'BAR_UPDATE'
    DEPTH_UPDATE = 'DEPTH_UPDATE'

    # Trading signals
    SIGNAL_GENERATED = 'SIGNAL_GENERATED'

    # Order lifecycle
    ORDER_SUBMITTED = 'ORDER_SUBMITTED'
    ORDER_FILLED = 'ORDER_FILLED'
    ORDER_PARTIALLY_FILLED = 'ORDER_PARTIALLY_FILLED'
    ORDER_REJECTED = 'ORDER_REJECTED'
    ORDER_CANCELLED = 'ORDER_CANCELLED'

    # Position updates
    POSITION_OPENED = 'POSITION_OPENED'
    POSITION_CLOSED = 'POSITION_CLOSED'
    POSITION_UPDATED = 'POSITION_UPDATED'

    # Engine status
    ENGINE_STATUS = 'ENGINE_STATUS'
    ENGINE_ERROR = 'ENGINE_ERROR'
    ENGINE_WARNING = 'ENGINE_WARNING'

    # Responses to sync requests
    STATUS_RESPONSE = 'STATUS_RESPONSE'
    POSITIONS_RESPONSE = 'POSITIONS_RESPONSE'
    ORDERS_RESPONSE = 'ORDERS_RESPONSE'
    PNL_RESPONSE = 'PNL_RESPONSE'

    # System events
    HEARTBEAT = 'HEARTBEAT'
    LOG_MESSAGE = 'LOG_MESSAGE'


@dataclass
class IPCMessage:
    """
    Message container for cross-process communication.

    All data passed between UI and Engine processes must be
    serializable (pickleable). This dataclass ensures consistent
    message format.

    Attributes:
        msg_type: Command or Event type string
        payload: Dictionary containing message data
        timestamp: When the message was created
        request_id: Optional ID for request/response correlation

    Example:
        >>> msg = IPCMessage(
        ...     msg_type=Commands.START_ENGINE,
        ...     payload={'mode': 'paper', 'capital': 100000}
        ... )
        >>> # Send via multiprocessing.Queue
        >>> queue.put(msg)
    """

    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'msg_type': self.msg_type,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCMessage':
        """Create from dictionary."""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            msg_type=data['msg_type'],
            payload=data.get('payload', {}),
            timestamp=timestamp,
            request_id=data.get('request_id')
        )

    def __repr__(self) -> str:
        return f"IPCMessage({self.msg_type}, keys={list(self.payload.keys())})"


# Convenience factory functions for common messages

def cmd_start_engine(mode: str = 'paper', capital: float = 100000.0) -> IPCMessage:
    """Create START_ENGINE command."""
    return IPCMessage(
        msg_type=Commands.START_ENGINE,
        payload={'mode': mode, 'capital': capital}
    )


def cmd_stop_engine() -> IPCMessage:
    """Create STOP_ENGINE command."""
    return IPCMessage(msg_type=Commands.STOP_ENGINE)


def cmd_pause() -> IPCMessage:
    """Create PAUSE command."""
    return IPCMessage(msg_type=Commands.PAUSE)


def cmd_resume() -> IPCMessage:
    """Create RESUME command."""
    return IPCMessage(msg_type=Commands.RESUME)


def cmd_place_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = 'MARKET',
    price: Optional[float] = None,
    request_id: Optional[str] = None
) -> IPCMessage:
    """Create PLACE_ORDER command."""
    return IPCMessage(
        msg_type=Commands.PLACE_ORDER,
        payload={
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price
        },
        request_id=request_id
    )


def cmd_get_positions(request_id: str) -> IPCMessage:
    """Create GET_POSITIONS command with request ID for response matching."""
    return IPCMessage(
        msg_type=Commands.GET_POSITIONS,
        request_id=request_id
    )


def cmd_get_status(request_id: str) -> IPCMessage:
    """Create GET_STATUS command with request ID for response matching."""
    return IPCMessage(
        msg_type=Commands.GET_STATUS,
        request_id=request_id
    )


def evt_tick_update(
    symbol: str,
    price: float,
    volume: int = 0,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    imbalance: Optional[float] = None
) -> IPCMessage:
    """Create TICK_UPDATE event."""
    return IPCMessage(
        msg_type=Events.TICK_UPDATE,
        payload={
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'bid': bid,
            'ask': ask,
            'imbalance': imbalance
        }
    )


def evt_signal_generated(
    symbol: str,
    signal_type: str,
    price: float,
    confidence: float = 0.0,
    reason: str = ''
) -> IPCMessage:
    """Create SIGNAL_GENERATED event."""
    return IPCMessage(
        msg_type=Events.SIGNAL_GENERATED,
        payload={
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'confidence': confidence,
            'reason': reason
        }
    )


def evt_order_filled(
    order_id: str,
    symbol: str,
    side: str,
    quantity: int,
    price: float
) -> IPCMessage:
    """Create ORDER_FILLED event."""
    return IPCMessage(
        msg_type=Events.ORDER_FILLED,
        payload={
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        }
    )


def evt_position_update(
    symbol: str,
    quantity: int,
    entry_price: float,
    current_price: float,
    pnl: float,
    pnl_percent: float
) -> IPCMessage:
    """Create POSITION_UPDATED event."""
    return IPCMessage(
        msg_type=Events.POSITION_UPDATED,
        payload={
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent
        }
    )


def evt_engine_status(
    status: str,
    mode: str = '',
    positions_count: int = 0,
    daily_pnl: float = 0.0
) -> IPCMessage:
    """Create ENGINE_STATUS event."""
    return IPCMessage(
        msg_type=Events.ENGINE_STATUS,
        payload={
            'status': status,
            'mode': mode,
            'positions_count': positions_count,
            'daily_pnl': daily_pnl
        }
    )


def evt_engine_error(error: str, details: str = '') -> IPCMessage:
    """Create ENGINE_ERROR event."""
    return IPCMessage(
        msg_type=Events.ENGINE_ERROR,
        payload={
            'error': error,
            'details': details
        }
    )


def evt_heartbeat() -> IPCMessage:
    """Create HEARTBEAT event for health monitoring."""
    return IPCMessage(
        msg_type=Events.HEARTBEAT,
        payload={'alive': True}
    )


def evt_log_message(level: str, message: str, source: str = 'engine') -> IPCMessage:
    """Create LOG_MESSAGE event for UI activity feed."""
    return IPCMessage(
        msg_type=Events.LOG_MESSAGE,
        payload={
            'level': level,
            'message': message,
            'source': source
        }
    )
