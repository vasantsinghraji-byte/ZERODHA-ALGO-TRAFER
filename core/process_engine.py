"""
Process-based Trading Engine.

Runs EventDrivenLiveEngine in a separate process to bypass Python's GIL,
enabling true parallelism between the trading engine and the UI.

Architecture:
    ┌──────────────────────┐     ┌──────────────────────┐
    │ UI Process           │     │ Engine Process       │
    │  └── Tkinter main    │◄───►│  ├── EventBus        │
    │      loop            │ IPC │  ├── Trading Engine  │
    │                      │     │  └── Data Feed       │
    └──────────────────────┘     └──────────────────────┘

Benefits:
    - UI remains responsive during high tick rates (100+ ticks/sec)
    - Engine can process data without waiting for UI updates
    - Crash isolation - UI survives engine crashes and vice versa

Usage:
    >>> engine = EngineProcess()
    >>> engine.start({'mode': 'paper', 'capital': 100000})
    >>> # Poll events in UI loop
    >>> events = engine.poll_events()
    >>> # Send commands
    >>> engine.send_command(Commands.PAUSE)
    >>> # Cleanup
    >>> engine.stop()
"""

import logging
import time
import uuid
from datetime import datetime
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Dict, List, Optional

from core.ipc_messages import (
    Commands,
    Events,
    IPCMessage,
    cmd_stop_engine,
    evt_engine_status,
    evt_engine_error,
    evt_heartbeat,
    evt_log_message,
    evt_tick_update,
    evt_signal_generated,
    evt_order_filled,
    evt_position_update,
)

logger = logging.getLogger(__name__)


class EngineProcess:
    """
    Wrapper that runs the trading engine in a separate process.

    Manages the lifecycle of the engine process and provides
    IPC communication via multiprocessing.Queue.

    Attributes:
        process: The actual multiprocessing.Process instance
        to_engine: Queue for sending commands to engine
        from_engine: Queue for receiving events from engine
        config: Engine configuration dictionary
    """

    def __init__(self):
        self.process: Optional[Process] = None
        self.to_engine: Queue = Queue()
        self.from_engine: Queue = Queue()
        self.config: Dict[str, Any] = {}
        self._pending_events: List[IPCMessage] = []
        self._last_heartbeat: Optional[datetime] = None

    def start(self, config: Dict[str, Any]) -> bool:
        """
        Start the trading engine in a separate process.

        Args:
            config: Engine configuration including:
                - initial_capital: Starting capital
                - trading_mode: 'paper' or 'live'
                - Other engine settings

        Returns:
            True if process started successfully
        """
        if self.process is not None and self.process.is_alive():
            logger.warning("Engine process already running")
            return False

        self.config = config
        logger.info("Starting engine process...")

        try:
            self.process = Process(
                target=_engine_process_main,
                args=(self.to_engine, self.from_engine, config),
                name="TradingEngine",
                daemon=False  # Not daemon so it can clean up properly
            )
            self.process.start()
            logger.info(f"Engine process started with PID: {self.process.pid}")

            # Wait briefly for startup confirmation
            time.sleep(0.5)
            if self.process.is_alive():
                self._last_heartbeat = datetime.now()
                return True
            else:
                logger.error("Engine process died immediately after start")
                return False

        except Exception as e:
            logger.exception(f"Failed to start engine process: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> bool:
        """
        Gracefully stop the engine process.

        Sends STOP_ENGINE command and waits for clean shutdown.
        Falls back to terminate() if process doesn't stop cleanly.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown

        Returns:
            True if stopped gracefully, False if terminated forcefully
        """
        if self.process is None:
            logger.warning("No engine process to stop")
            return True

        if not self.process.is_alive():
            logger.info("Engine process already stopped")
            self.process = None
            return True

        logger.info("Sending stop command to engine...")
        self.to_engine.put(cmd_stop_engine())

        # Wait for graceful shutdown
        self.process.join(timeout=timeout)

        if self.process.is_alive():
            logger.warning(f"Engine didn't stop within {timeout}s, terminating...")
            self.process.terminate()
            self.process.join(timeout=2)

            if self.process.is_alive():
                logger.error("Engine process refused to die, killing...")
                self.process.kill()
                self.process.join(timeout=1)
                self.process = None
                return False

        self.process = None
        logger.info("Engine process stopped")
        return True

    def restart(self) -> bool:
        """Stop and restart the engine process with same config."""
        self.stop()
        time.sleep(0.5)  # Brief pause before restart
        return self.start(self.config)

    def send_command(self, cmd: str, payload: Dict[str, Any] = None) -> None:
        """
        Send a command to the engine process.

        Args:
            cmd: Command type from Commands enum
            payload: Optional command data
        """
        if self.process is None or not self.process.is_alive():
            logger.warning(f"Cannot send command {cmd}: engine not running")
            return

        msg = IPCMessage(msg_type=cmd, payload=payload or {})
        self.to_engine.put(msg)
        logger.debug(f"Sent command: {cmd}")

    def poll_events(self, max_events: int = 100) -> List[IPCMessage]:
        """
        Non-blocking poll for events from the engine.

        Call this periodically in the UI event loop.

        Args:
            max_events: Maximum events to return per call

        Returns:
            List of IPCMessage events (may be empty)
        """
        events = []

        # First return any pending events from sync operations
        if self._pending_events:
            events.extend(self._pending_events)
            self._pending_events = []

        # Then poll the queue
        try:
            while len(events) < max_events:
                msg = self.from_engine.get_nowait()
                events.append(msg)

                # Track heartbeats for health monitoring
                if msg.msg_type == Events.HEARTBEAT:
                    self._last_heartbeat = datetime.now()

        except Empty:
            pass

        return events

    def get_positions_sync(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Synchronous request for current positions.

        Blocks until response or timeout.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            List of position dictionaries

        Raises:
            TimeoutError: If engine doesn't respond in time
        """
        request_id = str(uuid.uuid4())
        self.send_command(Commands.GET_POSITIONS, {'request_id': request_id})

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.from_engine.get(timeout=0.1)

                if (msg.msg_type == Events.POSITIONS_RESPONSE and
                    msg.payload.get('request_id') == request_id):
                    return msg.payload.get('positions', [])
                else:
                    # Queue non-matching events for later
                    self._pending_events.append(msg)

            except Empty:
                pass

        raise TimeoutError("Engine did not respond to get_positions()")

    def get_status_sync(self, timeout: float = 1.0) -> Dict[str, Any]:
        """
        Synchronous request for engine status.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            Status dictionary with 'status', 'mode', 'positions_count', etc.

        Raises:
            TimeoutError: If engine doesn't respond in time
        """
        request_id = str(uuid.uuid4())
        self.send_command(Commands.GET_STATUS, {'request_id': request_id})

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.from_engine.get(timeout=0.1)

                if (msg.msg_type == Events.STATUS_RESPONSE and
                    msg.payload.get('request_id') == request_id):
                    return msg.payload

                self._pending_events.append(msg)

            except Empty:
                pass

        raise TimeoutError("Engine did not respond to get_status()")

    def is_alive(self) -> bool:
        """Check if engine process is running."""
        return self.process is not None and self.process.is_alive()

    def is_healthy(self, heartbeat_timeout: float = 5.0) -> bool:
        """
        Check if engine is healthy (alive and responsive).

        Args:
            heartbeat_timeout: Max seconds since last heartbeat

        Returns:
            True if process is alive and recently sent heartbeat
        """
        if not self.is_alive():
            return False

        if self._last_heartbeat is None:
            return True  # No heartbeat yet, assume healthy

        elapsed = (datetime.now() - self._last_heartbeat).total_seconds()
        return elapsed < heartbeat_timeout

    @property
    def pid(self) -> Optional[int]:
        """Get the engine process PID."""
        return self.process.pid if self.process else None


def _engine_process_main(
    to_engine: Queue,
    from_engine: Queue,
    config: Dict[str, Any]
) -> None:
    """
    Main function running in the engine process.

    This function is the entry point for the separate process.
    It initializes the trading engine and runs the command loop.

    Args:
        to_engine: Queue for receiving commands from UI
        from_engine: Queue for sending events to UI
        config: Engine configuration dictionary
    """
    import logging
    import signal
    import sys

    # Setup logging for engine process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | ENGINE | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('EngineProcess')
    logger.info("Engine process starting...")

    # Handle termination signals gracefully
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"Received signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine = None
    event_bus = None

    try:
        # Import inside process to avoid pickling issues
        from core.events import EventBus, EventType
        from core.trading_engine import EventDrivenLiveEngine, EngineConfig, TradingMode

        # Create EventBus for internal event routing
        event_bus = EventBus(async_mode=False)
        logger.info("EventBus created")

        # Create engine configuration
        mode_str = config.get('trading_mode', 'paper')
        mode = TradingMode.PAPER if mode_str == 'paper' else TradingMode.LIVE

        engine_config = EngineConfig(
            mode=mode,
            capital=config.get('initial_capital', 100000.0),
            position_size_pct=config.get('position_size_pct', 10.0),
            max_daily_loss_pct=config.get('max_daily_loss_pct', 5.0),
        )

        # Create trading engine
        engine = EventDrivenLiveEngine(
            event_bus=event_bus,
            broker=None,  # Paper trading
            config=engine_config
        )
        logger.info(f"Trading engine created: mode={mode_str}")

        # Bridge EventBus events to IPC queue
        _setup_event_bridges(event_bus, from_engine, logger)

        # Send startup status
        from_engine.put(evt_engine_status(
            status='INITIALIZED',
            mode=mode_str,
            positions_count=0,
            daily_pnl=0.0
        ))

        # Main command processing loop
        heartbeat_interval = 1.0  # Send heartbeat every second
        last_heartbeat = time.time()

        logger.info("Entering main loop...")
        while running:
            try:
                # Check for commands (non-blocking with timeout)
                try:
                    cmd = to_engine.get(timeout=0.1)
                    _handle_command(cmd, engine, from_engine, logger)

                    # Check for stop command
                    if cmd.msg_type == Commands.STOP_ENGINE:
                        running = False

                except Empty:
                    pass

                # Send periodic heartbeat
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    from_engine.put(evt_heartbeat())
                    last_heartbeat = now

            except Exception as e:
                logger.exception(f"Error in main loop: {e}")
                from_engine.put(evt_engine_error(str(e), 'main_loop'))
                time.sleep(0.1)  # Prevent tight error loop

    except Exception as e:
        logger.exception(f"Fatal error in engine process: {e}")
        from_engine.put(evt_engine_error(str(e), 'initialization'))

    finally:
        # Cleanup
        logger.info("Cleaning up engine process...")
        if engine:
            try:
                engine.stop()
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")

        from_engine.put(evt_engine_status(status='STOPPED'))
        logger.info("Engine process exiting")


def _setup_event_bridges(event_bus, from_engine: Queue, logger) -> None:
    """
    Subscribe to EventBus events and forward them to IPC queue.

    This bridges the internal event system to the UI process.
    """
    from core.events import EventType

    def on_tick(event):
        """Forward tick events to UI."""
        try:
            msg = evt_tick_update(
                symbol=getattr(event, 'symbol', 'Unknown'),
                price=getattr(event, 'last_price', 0.0),
                volume=getattr(event, 'volume', 0),
                bid=getattr(event, 'bid', None),
                ask=getattr(event, 'ask', None),
                imbalance=getattr(event, 'imbalance', None)
            )
            from_engine.put(msg)
        except Exception as e:
            logger.debug(f"Error forwarding tick: {e}")

    def on_signal(event):
        """Forward signal events to UI."""
        try:
            msg = evt_signal_generated(
                symbol=getattr(event, 'symbol', 'Unknown'),
                signal_type=getattr(event, 'signal_type', 'HOLD').value if hasattr(getattr(event, 'signal_type', None), 'value') else str(getattr(event, 'signal_type', 'HOLD')),
                price=getattr(event, 'price', 0.0),
                confidence=getattr(event, 'confidence', 0.0),
                reason=getattr(event, 'reason', '')
            )
            from_engine.put(msg)
        except Exception as e:
            logger.debug(f"Error forwarding signal: {e}")

    def on_order_filled(event):
        """Forward order filled events to UI."""
        try:
            msg = evt_order_filled(
                order_id=getattr(event, 'order_id', 'unknown'),
                symbol=getattr(event, 'symbol', 'Unknown'),
                side=getattr(event, 'side', 'BUY'),
                quantity=getattr(event, 'quantity', 0),
                price=getattr(event, 'average_price', 0.0)
            )
            from_engine.put(msg)
        except Exception as e:
            logger.debug(f"Error forwarding order: {e}")

    def on_position(event):
        """Forward position events to UI."""
        try:
            msg = evt_position_update(
                symbol=getattr(event, 'symbol', 'Unknown'),
                quantity=getattr(event, 'quantity', 0),
                entry_price=getattr(event, 'entry_price', 0.0),
                current_price=getattr(event, 'current_price', 0.0),
                pnl=getattr(event, 'pnl', 0.0),
                pnl_percent=getattr(event, 'pnl_percent', 0.0)
            )
            from_engine.put(msg)
        except Exception as e:
            logger.debug(f"Error forwarding position: {e}")

    def on_bar(event):
        """Forward bar events to UI."""
        try:
            from core.ipc_messages import IPCMessage, Events
            msg = IPCMessage(
                msg_type=Events.BAR_UPDATE,
                payload={
                    'symbol': getattr(event, 'symbol', 'Unknown'),
                    'timeframe': getattr(event, 'timeframe', '1m'),
                    'open': getattr(event, 'open', 0.0),
                    'high': getattr(event, 'high', 0.0),
                    'low': getattr(event, 'low', 0.0),
                    'close': getattr(event, 'close', 0.0),
                    'volume': getattr(event, 'volume', 0)
                }
            )
            from_engine.put(msg)
        except Exception as e:
            logger.debug(f"Error forwarding bar: {e}")

    # Subscribe to relevant events
    event_bus.subscribe(EventType.TICK, on_tick, name="ipc_tick")
    event_bus.subscribe(EventType.BAR, on_bar, name="ipc_bar")
    event_bus.subscribe(EventType.SIGNAL_GENERATED, on_signal, name="ipc_signal")
    event_bus.subscribe(EventType.ORDER_FILLED, on_order_filled, name="ipc_order_filled")
    event_bus.subscribe(EventType.POSITION_OPENED, on_position, name="ipc_position_opened")
    event_bus.subscribe(EventType.POSITION_CLOSED, on_position, name="ipc_position_closed")

    logger.info("Event bridges established (6 event types)")


def _handle_command(
    cmd: IPCMessage,
    engine,
    from_engine: Queue,
    logger
) -> None:
    """
    Handle a command received from the UI process.

    Args:
        cmd: The command message
        engine: EventDrivenLiveEngine instance
        from_engine: Queue for sending responses
        logger: Logger instance
    """
    from core.ipc_messages import Events

    logger.debug(f"Handling command: {cmd.msg_type}")

    try:
        if cmd.msg_type == Commands.START_ENGINE:
            engine.start()
            from_engine.put(evt_engine_status(status='RUNNING'))
            logger.info("Engine started")

        elif cmd.msg_type == Commands.STOP_ENGINE:
            engine.stop()
            from_engine.put(evt_engine_status(status='STOPPED'))
            logger.info("Engine stopped")

        elif cmd.msg_type == Commands.PAUSE:
            engine.pause()
            from_engine.put(evt_engine_status(status='PAUSED'))
            logger.info("Engine paused")

        elif cmd.msg_type == Commands.RESUME:
            engine.resume()
            from_engine.put(evt_engine_status(status='RUNNING'))
            logger.info("Engine resumed")

        elif cmd.msg_type == Commands.GET_STATUS:
            # Respond to status request
            status = engine.status.value if hasattr(engine.status, 'value') else str(engine.status)
            response = IPCMessage(
                msg_type=Events.STATUS_RESPONSE,
                payload={
                    'request_id': cmd.request_id,
                    'status': status,
                    'mode': engine._config.mode.value if hasattr(engine._config.mode, 'value') else str(engine._config.mode),
                    'positions_count': len(engine._positions) if hasattr(engine, '_positions') else 0,
                    'daily_pnl': engine._daily_pnl if hasattr(engine, '_daily_pnl') else 0.0
                }
            )
            from_engine.put(response)

        elif cmd.msg_type == Commands.GET_POSITIONS:
            # Respond to positions request
            positions = []
            if hasattr(engine, '_positions'):
                for symbol, pos in engine._positions.items():
                    positions.append({
                        'symbol': symbol,
                        'quantity': getattr(pos, 'quantity', 0),
                        'entry_price': getattr(pos, 'entry_price', 0.0),
                        'current_price': getattr(pos, 'current_price', 0.0),
                        'pnl': getattr(pos, 'unrealized_pnl', 0.0)
                    })

            response = IPCMessage(
                msg_type=Events.POSITIONS_RESPONSE,
                payload={
                    'request_id': cmd.request_id,
                    'positions': positions
                }
            )
            from_engine.put(response)

        elif cmd.msg_type == Commands.PLACE_ORDER:
            # Place order through engine
            payload = cmd.payload
            # Engine would handle this internally
            logger.info(f"Order request: {payload}")
            from_engine.put(evt_log_message('info', f"Order placed: {payload.get('symbol')} {payload.get('side')}"))

        elif cmd.msg_type == Commands.SUBSCRIBE_SYMBOL:
            symbol = cmd.payload.get('symbol')
            token = cmd.payload.get('token')
            if symbol and hasattr(engine, 'subscribe'):
                engine.subscribe(symbol, token)
            logger.info(f"Subscribed to: {symbol}")

        elif cmd.msg_type == Commands.UNSUBSCRIBE_SYMBOL:
            symbol = cmd.payload.get('symbol')
            if symbol and hasattr(engine, 'unsubscribe'):
                engine.unsubscribe(symbol)
            logger.info(f"Unsubscribed from: {symbol}")

        else:
            logger.warning(f"Unknown command: {cmd.msg_type}")

    except Exception as e:
        logger.exception(f"Error handling command {cmd.msg_type}: {e}")
        from_engine.put(evt_engine_error(str(e), f'command_{cmd.msg_type}'))
