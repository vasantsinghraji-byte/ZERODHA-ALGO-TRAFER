# -*- coding: utf-8 -*-
"""
State Persistence Module - Never Lose Your Progress!
=====================================================
Save and restore strategy state across restarts and crashes.

Components:
- StrategyStateManager: Save/load strategy state to SQLite
- CheckpointManager: Automatic periodic state snapshots
- RecoveryManager: Crash detection and recovery
- ReconciliationManager: Position sync with broker
- StrategyState: Data class for strategy state

Example:
    >>> from core.state import StrategyStateManager, CheckpointManager
    >>>
    >>> # Save state manually
    >>> state_mgr = StrategyStateManager()
    >>> state_mgr.save_state(strategy_state)
    >>>
    >>> # Auto-checkpoint every 60 seconds
    >>> checkpoint_mgr = CheckpointManager(state_mgr, interval=60)
    >>> checkpoint_mgr.register_strategy(my_strategy)
    >>> checkpoint_mgr.start()
    >>>
    >>> # Crash recovery
    >>> from core.state import RecoveryManager
    >>> recovery = RecoveryManager(state_mgr)
    >>> if recovery.needs_recovery("my_strategy"):
    ...     result = recovery.recover("my_strategy")
    >>>
    >>> # Position reconciliation
    >>> from core.state import ReconciliationManager
    >>> reconciler = ReconciliationManager(broker, state_mgr)
    >>> report = reconciler.reconcile("my_strategy")
"""

from .state_manager import (
    StrategyState,
    StrategyStateManager,
    StateSerializer,
    get_state_manager,
    save_strategy_state,
    load_strategy_state,
)

from .checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointTrigger,
    StrategyStateExtractor,
    get_checkpoint_manager,
)

from .recovery import (
    RecoveryManager,
    RecoveryConfig,
    RecoveryResult,
    SessionTracker,
    SessionInfo,
    SessionStatus,
    EventStore,
    get_recovery_manager,
)

from .reconciliation import (
    ReconciliationManager,
    ReconciliationConfig,
    ReconciliationReport,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ReconciliationAction,
    PositionSnapshot,
    reconcile_positions,
    check_position_sync,
)

__all__ = [
    # State Manager
    'StrategyState',
    'StrategyStateManager',
    'StateSerializer',
    'get_state_manager',
    'save_strategy_state',
    'load_strategy_state',
    # Checkpoint Manager
    'CheckpointManager',
    'CheckpointConfig',
    'CheckpointTrigger',
    'StrategyStateExtractor',
    'get_checkpoint_manager',
    # Recovery
    'RecoveryManager',
    'RecoveryConfig',
    'RecoveryResult',
    'SessionTracker',
    'SessionInfo',
    'SessionStatus',
    'EventStore',
    'get_recovery_manager',
    # Reconciliation
    'ReconciliationManager',
    'ReconciliationConfig',
    'ReconciliationReport',
    'Discrepancy',
    'DiscrepancyType',
    'DiscrepancySeverity',
    'ReconciliationAction',
    'PositionSnapshot',
    'reconcile_positions',
    'check_position_sync',
]
