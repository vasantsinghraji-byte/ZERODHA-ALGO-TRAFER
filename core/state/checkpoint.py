# -*- coding: utf-8 -*-
"""
Checkpoint Manager - Auto-Save Your Progress!
==============================================
Automatically saves strategy state at regular intervals.

Why Checkpoints?
- Crash recovery: Resume from last checkpoint after unexpected shutdown
- Debugging: Replay state history to understand what happened
- Audit trail: Track all state changes over time

Features:
- Configurable checkpoint intervals
- Event-driven checkpoints (on trade, on signal)
- Manual checkpoint triggers
- Async checkpointing (non-blocking)

Example:
    >>> from core.state import CheckpointManager, StrategyStateManager
    >>>
    >>> state_mgr = StrategyStateManager()
    >>> checkpoint_mgr = CheckpointManager(state_mgr, interval=60)  # Every 60 seconds
    >>>
    >>> # Register strategies
    >>> checkpoint_mgr.register_strategy(my_strategy)
    >>>
    >>> # Start auto-checkpointing
    >>> checkpoint_mgr.start()
    >>>
    >>> # ... trading happens ...
    >>>
    >>> # Stop checkpointing
    >>> checkpoint_mgr.stop()
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .state_manager import StrategyStateManager, StrategyState

logger = logging.getLogger(__name__)


class CheckpointTrigger(Enum):
    """What triggers a checkpoint."""
    PERIODIC = "periodic"       # Time-based interval
    ON_TRADE = "on_trade"       # After each trade
    ON_SIGNAL = "on_signal"     # After each signal
    ON_POSITION = "on_position" # Position change
    MANUAL = "manual"           # Explicit call
    SHUTDOWN = "shutdown"       # Engine shutdown
    ERROR = "error"             # After an error


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint behavior.

    Attributes:
        interval_seconds: Time between periodic checkpoints (0 = disabled)
        checkpoint_on_trade: Create checkpoint after each trade
        checkpoint_on_signal: Create checkpoint after each signal
        checkpoint_on_position: Create checkpoint on position change
        max_checkpoints_per_hour: Rate limit to prevent spam
        async_checkpoint: Run checkpoints in background thread
    """
    interval_seconds: int = 60  # 1 minute default
    checkpoint_on_trade: bool = True
    checkpoint_on_signal: bool = False  # Can be noisy
    checkpoint_on_position: bool = True
    max_checkpoints_per_hour: int = 120  # Max 2 per minute
    async_checkpoint: bool = True


class StrategyStateExtractor:
    """
    Extracts state from a strategy for checkpointing.

    Different strategies store state differently, so this provides
    a unified way to extract the relevant state.
    """

    @staticmethod
    def extract(strategy) -> Dict[str, Any]:
        """
        Extract state from a strategy.

        Args:
            strategy: Strategy instance

        Returns:
            Dict containing strategy state
        """
        state = {
            'name': getattr(strategy, 'name', 'unknown'),
            'positions': {},
            'indicators': {},
            'parameters': {},
            'bars': {},
            'signals': [],
            'metadata': {}
        }

        # Extract positions
        if hasattr(strategy, '_positions'):
            state['positions'] = dict(strategy._positions)
        elif hasattr(strategy, 'positions'):
            state['positions'] = dict(strategy.positions)

        # Extract last prices
        if hasattr(strategy, '_last_prices'):
            state['metadata']['last_prices'] = dict(strategy._last_prices)

        # Extract bar history (last N bars per symbol)
        if hasattr(strategy, '_bars'):
            for symbol, df in strategy._bars.items():
                if hasattr(df, 'to_dict'):
                    # Keep last 100 bars for warmup
                    state['bars'][symbol] = df.tail(100).to_dict('records')

        # Extract parameters
        if hasattr(strategy, 'get_parameters'):
            state['parameters'] = strategy.get_parameters()

        # Extract signals
        if hasattr(strategy, '_signals'):
            state['signals'] = [
                {
                    'signal_type': s.signal_type.value if hasattr(s.signal_type, 'value') else str(s.signal_type),
                    'symbol': s.symbol,
                    'price': s.price,
                    'timestamp': s.timestamp.isoformat() if s.timestamp else None
                }
                for s in list(strategy._signals)[-20:]  # Last 20 signals
            ]

        return state


class CheckpointManager:
    """
    Manages automatic checkpointing of strategy state.

    Example:
        >>> # Create managers
        >>> state_mgr = StrategyStateManager("data/state.db")
        >>> checkpoint_mgr = CheckpointManager(state_mgr)
        >>>
        >>> # Register strategy
        >>> checkpoint_mgr.register_strategy(my_strategy)
        >>>
        >>> # Start periodic checkpoints
        >>> checkpoint_mgr.start()
        >>>
        >>> # Trigger manual checkpoint
        >>> checkpoint_mgr.checkpoint_now()
        >>>
        >>> # Stop
        >>> checkpoint_mgr.stop()
    """

    def __init__(
        self,
        state_manager: StrategyStateManager,
        config: Optional[CheckpointConfig] = None,
        event_bus=None
    ):
        """
        Initialize checkpoint manager.

        Args:
            state_manager: StrategyStateManager for persistence
            config: Checkpoint configuration
            event_bus: Optional EventBus for event-driven checkpoints
        """
        self.state_manager = state_manager
        self.config = config or CheckpointConfig()
        self.event_bus = event_bus

        # Registered strategies
        self._strategies: Dict[str, Any] = {}  # name -> strategy

        # State extractors (can be customized per strategy)
        self._extractors: Dict[str, Callable] = {}  # name -> extractor function

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Rate limiting
        self._checkpoint_times: List[datetime] = []

        # Stats
        self._checkpoint_count = 0
        self._last_checkpoint: Optional[datetime] = None
        self._errors: List[str] = []

        # Event subscriptions
        self._handler_names: List[str] = []

        logger.info(f"CheckpointManager initialized (interval={self.config.interval_seconds}s)")

    # =========================================================================
    # Strategy Registration
    # =========================================================================

    def register_strategy(
        self,
        strategy,
        extractor: Optional[Callable] = None
    ):
        """
        Register a strategy for checkpointing.

        Args:
            strategy: Strategy instance
            extractor: Optional custom state extractor function
        """
        name = getattr(strategy, 'name', str(id(strategy)))
        with self._lock:
            self._strategies[name] = strategy
            if extractor:
                self._extractors[name] = extractor
        logger.debug(f"Registered strategy for checkpointing: {name}")

    def unregister_strategy(self, name: str):
        """Unregister a strategy from checkpointing."""
        with self._lock:
            self._strategies.pop(name, None)
            self._extractors.pop(name, None)
        logger.debug(f"Unregistered strategy: {name}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start periodic checkpointing."""
        if self._running:
            logger.warning("CheckpointManager already running")
            return

        self._running = True

        # Subscribe to events
        if self.event_bus:
            self._subscribe_events()

        # Start periodic thread
        if self.config.interval_seconds > 0:
            self._thread = threading.Thread(
                target=self._checkpoint_loop,
                daemon=True,
                name="CheckpointManager"
            )
            self._thread.start()
            logger.info(f"Started periodic checkpointing (every {self.config.interval_seconds}s)")
        else:
            logger.info("Periodic checkpointing disabled (interval=0)")

    def stop(self):
        """Stop checkpointing and save final state."""
        self._running = False

        # Final checkpoint
        self.checkpoint_now(trigger=CheckpointTrigger.SHUTDOWN)

        # Unsubscribe from events
        if self.event_bus:
            self._unsubscribe_events()

        # Wait for thread
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("CheckpointManager stopped")

    def _checkpoint_loop(self):
        """Background thread for periodic checkpoints."""
        while self._running:
            try:
                time.sleep(self.config.interval_seconds)
                if self._running:
                    self.checkpoint_now(trigger=CheckpointTrigger.PERIODIC)
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                self._errors.append(str(e))

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _subscribe_events(self):
        """Subscribe to relevant events for checkpointing."""
        try:
            from core.events import EventType

            self._handler_names = []

            if self.config.checkpoint_on_trade:
                name = self.event_bus.subscribe(
                    EventType.ORDER_FILLED,
                    self._on_trade,
                    priority=200,  # Low priority - run after main handlers
                    name="checkpoint_on_trade"
                )
                self._handler_names.append(name)

            if self.config.checkpoint_on_signal:
                name = self.event_bus.subscribe(
                    EventType.SIGNAL_GENERATED,
                    self._on_signal,
                    priority=200,
                    name="checkpoint_on_signal"
                )
                self._handler_names.append(name)

            if self.config.checkpoint_on_position:
                for event_type in [EventType.POSITION_OPENED, EventType.POSITION_CLOSED]:
                    name = self.event_bus.subscribe(
                        event_type,
                        self._on_position,
                        priority=200,
                        name=f"checkpoint_on_{event_type.value}"
                    )
                    self._handler_names.append(name)

            logger.debug(f"Subscribed to {len(self._handler_names)} event types for checkpointing")

        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _unsubscribe_events(self):
        """Unsubscribe from events."""
        for name in self._handler_names:
            try:
                self.event_bus.unsubscribe(name)
            except Exception:
                pass
        self._handler_names.clear()

    def _on_trade(self, event):
        """Handle trade event - trigger checkpoint."""
        if self.config.async_checkpoint:
            threading.Thread(
                target=self.checkpoint_now,
                kwargs={'trigger': CheckpointTrigger.ON_TRADE},
                daemon=True
            ).start()
        else:
            self.checkpoint_now(trigger=CheckpointTrigger.ON_TRADE)

    def _on_signal(self, event):
        """Handle signal event - trigger checkpoint."""
        if self.config.async_checkpoint:
            threading.Thread(
                target=self.checkpoint_now,
                kwargs={'trigger': CheckpointTrigger.ON_SIGNAL},
                daemon=True
            ).start()
        else:
            self.checkpoint_now(trigger=CheckpointTrigger.ON_SIGNAL)

    def _on_position(self, event):
        """Handle position event - trigger checkpoint."""
        if self.config.async_checkpoint:
            threading.Thread(
                target=self.checkpoint_now,
                kwargs={'trigger': CheckpointTrigger.ON_POSITION},
                daemon=True
            ).start()
        else:
            self.checkpoint_now(trigger=CheckpointTrigger.ON_POSITION)

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def checkpoint_now(
        self,
        strategy_name: Optional[str] = None,
        trigger: CheckpointTrigger = CheckpointTrigger.MANUAL
    ) -> bool:
        """
        Create a checkpoint immediately.

        Args:
            strategy_name: Specific strategy to checkpoint (None = all)
            trigger: What triggered this checkpoint

        Returns:
            True if checkpoint was created
        """
        # Rate limiting
        if not self._check_rate_limit():
            logger.debug("Checkpoint skipped due to rate limiting")
            return False

        with self._lock:
            try:
                strategies = self._strategies
                if strategy_name:
                    strategies = {
                        strategy_name: self._strategies.get(strategy_name)
                    }

                for name, strategy in strategies.items():
                    if strategy is None:
                        continue

                    # Extract state
                    extractor = self._extractors.get(name, StrategyStateExtractor.extract)
                    state_dict = extractor(strategy)

                    # Create StrategyState
                    state = StrategyState(
                        strategy_name=name,
                        positions=state_dict.get('positions', {}),
                        indicators=state_dict.get('indicators', {}),
                        parameters=state_dict.get('parameters', {}),
                        bars=state_dict.get('bars', {}),
                        signals=state_dict.get('signals', []),
                        metadata=state_dict.get('metadata', {})
                    )

                    # Save
                    self.state_manager.save_state(
                        state,
                        checkpoint_type=trigger.value
                    )

                self._checkpoint_count += 1
                self._last_checkpoint = datetime.now()
                self._checkpoint_times.append(self._last_checkpoint)

                logger.debug(f"Checkpoint created ({trigger.value}): {list(strategies.keys())}")
                return True

            except Exception as e:
                logger.error(f"Checkpoint failed: {e}", exc_info=True)
                self._errors.append(str(e))
                return False

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Remove old entries
        self._checkpoint_times = [
            t for t in self._checkpoint_times if t > hour_ago
        ]

        # Check limit
        return len(self._checkpoint_times) < self.config.max_checkpoints_per_hour

    # =========================================================================
    # Recovery
    # =========================================================================

    def restore_strategy(self, strategy, strategy_name: Optional[str] = None) -> bool:
        """
        Restore a strategy's state from the last checkpoint.

        Args:
            strategy: Strategy instance to restore
            strategy_name: Override strategy name

        Returns:
            True if state was restored
        """
        name = strategy_name or getattr(strategy, 'name', None)
        if not name:
            logger.error("Cannot restore: strategy has no name")
            return False

        state = self.state_manager.load_state(name)
        if not state:
            logger.info(f"No saved state found for {name}")
            return False

        try:
            # Restore positions
            if hasattr(strategy, '_positions'):
                strategy._positions = state.positions
            elif hasattr(strategy, 'positions'):
                strategy.positions = state.positions

            # Restore last prices
            if hasattr(strategy, '_last_prices') and state.metadata.get('last_prices'):
                strategy._last_prices = state.metadata['last_prices']

            # Restore parameters
            if hasattr(strategy, 'set_parameters') and state.parameters:
                strategy.set_parameters(**state.parameters)

            # Restore bar history
            if hasattr(strategy, '_bars') and state.bars:
                import pandas as pd
                for symbol, bars in state.bars.items():
                    if bars:
                        strategy._bars[symbol] = pd.DataFrame(bars)

            logger.info(f"Restored state for {name} from checkpoint")
            return True

        except Exception as e:
            logger.error(f"Failed to restore state for {name}: {e}", exc_info=True)
            return False

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get checkpoint manager status."""
        with self._lock:
            return {
                'running': self._running,
                'strategies': list(self._strategies.keys()),
                'checkpoint_count': self._checkpoint_count,
                'last_checkpoint': self._last_checkpoint.isoformat() if self._last_checkpoint else None,
                'checkpoints_last_hour': len(self._checkpoint_times),
                'config': {
                    'interval_seconds': self.config.interval_seconds,
                    'checkpoint_on_trade': self.config.checkpoint_on_trade,
                    'checkpoint_on_position': self.config.checkpoint_on_position,
                    'max_per_hour': self.config.max_checkpoints_per_hour
                },
                'errors': self._errors[-5:]  # Last 5 errors
            }

    def print_status(self):
        """Print checkpoint status."""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("CHECKPOINT MANAGER STATUS")
        print("=" * 50)
        print(f"Running: {status['running']}")
        print(f"Strategies: {', '.join(status['strategies']) or 'None'}")
        print(f"Total Checkpoints: {status['checkpoint_count']}")
        print(f"Last Checkpoint: {status['last_checkpoint'] or 'Never'}")
        print(f"Checkpoints (last hour): {status['checkpoints_last_hour']}")
        print(f"Interval: {status['config']['interval_seconds']}s")
        if status['errors']:
            print(f"Recent Errors: {len(status['errors'])}")
        print("=" * 50)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(
    state_manager: Optional[StrategyStateManager] = None,
    config: Optional[CheckpointConfig] = None
) -> CheckpointManager:
    """Get or create the default checkpoint manager."""
    global _default_checkpoint_manager
    if _default_checkpoint_manager is None:
        from .state_manager import get_state_manager
        sm = state_manager or get_state_manager()
        _default_checkpoint_manager = CheckpointManager(sm, config)
    return _default_checkpoint_manager


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("CHECKPOINT MANAGER - Test")
    print("=" * 50)

    # Create a mock strategy
    class MockStrategy:
        name = "test_strategy"
        _positions = {"RELIANCE": {"side": "BUY", "quantity": 10, "entry_price": 2500}}
        _last_prices = {"RELIANCE": 2520}
        _signals = []
        _bars = {}

        def get_parameters(self):
            return {"period": 20}

    # Create managers
    state_mgr = StrategyStateManager("data/test_checkpoint.db")
    config = CheckpointConfig(interval_seconds=5)  # 5 second interval for testing
    checkpoint_mgr = CheckpointManager(state_mgr, config)

    # Register strategy
    strategy = MockStrategy()
    checkpoint_mgr.register_strategy(strategy)

    # Create checkpoint
    print("\n1. Creating checkpoint...")
    checkpoint_mgr.checkpoint_now()
    print("   Checkpoint created!")

    # Check status
    print("\n2. Checking status...")
    checkpoint_mgr.print_status()

    # Load state
    print("\n3. Loading state...")
    loaded = state_mgr.load_state("test_strategy")
    print(f"   Positions: {loaded.positions}")

    # Cleanup
    print("\n4. Cleaning up...")
    state_mgr.delete_state("test_strategy")

    print("\n" + "=" * 50)
    print("Checkpoint Manager ready!")
    print("=" * 50)
