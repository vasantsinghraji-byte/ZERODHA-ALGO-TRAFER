# -*- coding: utf-8 -*-
"""
Position Reconciliation Module - Trust but Verify!
===================================================
Compare local state with broker positions and sync discrepancies.

Features:
- Compare local positions with broker
- Generate detailed discrepancy reports
- Auto-sync or alert for manual review
- Audit trail for all reconciliations

Example:
    >>> from core.state import ReconciliationManager
    >>> from core import ZerodhaBroker
    >>>
    >>> broker = ZerodhaBroker(...)
    >>> reconciler = ReconciliationManager(broker, state_manager)
    >>>
    >>> report = reconciler.reconcile("my_strategy")
    >>> if report.has_discrepancies:
    ...     print(report.summary())
    ...     if report.can_auto_sync:
    ...         reconciler.auto_sync(report)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_manager import StrategyStateManager
    from ..broker import ZerodhaBroker

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of position discrepancies."""
    MISSING_LOCAL = "missing_local"       # Position exists in broker but not locally
    MISSING_BROKER = "missing_broker"     # Position exists locally but not in broker
    QUANTITY_MISMATCH = "quantity"        # Different quantities
    PRICE_MISMATCH = "price"              # Different average prices
    SIDE_MISMATCH = "side"                # Long vs Short mismatch


class DiscrepancySeverity(Enum):
    """Severity level of discrepancy."""
    INFO = "info"           # Minor difference, no action needed
    WARNING = "warning"     # Notable difference, review recommended
    CRITICAL = "critical"   # Major difference, action required


class ReconciliationAction(Enum):
    """Actions that can be taken to resolve discrepancies."""
    NONE = "none"                    # No action needed
    UPDATE_LOCAL = "update_local"    # Update local state to match broker
    ALERT_USER = "alert_user"        # Alert user for manual review
    CLOSE_POSITION = "close_position"  # Close orphan position
    OPEN_POSITION = "open_position"    # Open missing position


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""
    symbol: str
    quantity: int
    average_price: float
    side: str  # "long" or "short"
    pnl: float = 0.0
    market_value: float = 0.0
    last_price: float = 0.0
    source: str = ""  # "local" or "broker"


@dataclass
class Discrepancy:
    """A single discrepancy between local and broker state."""
    symbol: str
    type: DiscrepancyType
    severity: DiscrepancySeverity
    local_value: Any
    broker_value: Any
    recommended_action: ReconciliationAction
    description: str
    auto_syncable: bool = True


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    strategy_id: str
    timestamp: datetime
    local_positions: List[PositionSnapshot]
    broker_positions: List[PositionSnapshot]
    discrepancies: List[Discrepancy] = field(default_factory=list)
    reconciliation_time_ms: float = 0.0
    auto_synced: bool = False
    sync_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies exist."""
        return len(self.discrepancies) > 0

    @property
    def critical_count(self) -> int:
        """Count critical discrepancies."""
        return sum(1 for d in self.discrepancies if d.severity == DiscrepancySeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count warning discrepancies."""
        return sum(1 for d in self.discrepancies if d.severity == DiscrepancySeverity.WARNING)

    @property
    def can_auto_sync(self) -> bool:
        """Check if all discrepancies can be auto-synced."""
        return all(d.auto_syncable for d in self.discrepancies)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Reconciliation Report for {self.strategy_id}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {self.reconciliation_time_ms:.1f}ms",
            "",
            f"Local positions: {len(self.local_positions)}",
            f"Broker positions: {len(self.broker_positions)}",
            f"Discrepancies: {len(self.discrepancies)}",
            f"  - Critical: {self.critical_count}",
            f"  - Warning: {self.warning_count}",
            "",
        ]

        if self.discrepancies:
            lines.append("Discrepancy Details:")
            for d in self.discrepancies:
                lines.append(f"  [{d.severity.value.upper()}] {d.symbol}: {d.description}")
                lines.append(f"    Local: {d.local_value}, Broker: {d.broker_value}")
                lines.append(f"    Action: {d.recommended_action.value}")
            lines.append("")

        if self.auto_synced:
            lines.append("Auto-sync performed successfully")

        return "\n".join(lines)


@dataclass
class ReconciliationConfig:
    """Configuration for reconciliation behavior."""
    # Tolerance thresholds
    price_tolerance_pct: float = 0.01      # 1% price difference tolerance
    quantity_tolerance: int = 0            # No tolerance for quantity by default

    # Auto-sync settings
    auto_sync_enabled: bool = False        # Disabled by default for safety
    auto_sync_missing_local: bool = True   # Sync positions missing locally
    auto_sync_quantity: bool = True        # Sync quantity mismatches
    auto_sync_close_orphans: bool = False  # Don't auto-close orphan positions

    # Alerting
    alert_on_critical: bool = True
    alert_callback: Optional[Callable[[ReconciliationReport], None]] = None

    # Scheduling
    reconcile_on_startup: bool = True
    reconcile_interval_minutes: int = 15


class ReconciliationManager:
    """
    Manages position reconciliation between local state and broker.

    Compares positions, generates reports, and optionally auto-syncs.
    """

    def __init__(
        self,
        broker: 'ZerodhaBroker',
        state_manager: 'StrategyStateManager',
        config: Optional[ReconciliationConfig] = None
    ):
        self.broker = broker
        self.state_manager = state_manager
        self.config = config or ReconciliationConfig()
        self._reconciliation_history: List[ReconciliationReport] = []
        self._alert_callbacks: List[Callable[[ReconciliationReport], None]] = []

    def on_alert(self, callback: Callable[[ReconciliationReport], None]) -> None:
        """Register callback for reconciliation alerts."""
        self._alert_callbacks.append(callback)

    def get_local_positions(self, strategy_id: str) -> List[PositionSnapshot]:
        """Get positions from local state."""
        state = self.state_manager.load_state(strategy_id)
        if not state:
            return []

        positions = []
        for symbol, pos_data in state.positions.items():
            # Handle both dict and dataclass position data
            if isinstance(pos_data, dict):
                qty = pos_data.get('quantity', 0)
                avg_price = pos_data.get('average_price', 0.0)
                side = pos_data.get('side', 'long')
                pnl = pos_data.get('unrealized_pnl', 0.0)
            else:
                qty = getattr(pos_data, 'quantity', 0)
                avg_price = getattr(pos_data, 'average_price', 0.0)
                side = getattr(pos_data, 'side', 'long')
                pnl = getattr(pos_data, 'unrealized_pnl', 0.0)

            positions.append(PositionSnapshot(
                symbol=symbol,
                quantity=qty,
                average_price=avg_price,
                side=side,
                pnl=pnl,
                source="local"
            ))

        return positions

    def get_broker_positions(self) -> List[PositionSnapshot]:
        """Get positions from broker."""
        try:
            broker_positions = self.broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to get broker positions: {e}")
            return []

        positions = []
        for pos in broker_positions:
            # Handle Zerodha position format
            if isinstance(pos, dict):
                symbol = pos.get('tradingsymbol', pos.get('symbol', ''))
                qty = pos.get('quantity', 0)
                avg_price = pos.get('average_price', 0.0)
                pnl = pos.get('pnl', 0.0)
                last_price = pos.get('last_price', 0.0)
            else:
                symbol = getattr(pos, 'symbol', getattr(pos, 'tradingsymbol', ''))
                qty = getattr(pos, 'quantity', 0)
                avg_price = getattr(pos, 'average_price', 0.0)
                pnl = getattr(pos, 'pnl', 0.0)
                last_price = getattr(pos, 'last_price', 0.0)

            # Skip zero-quantity positions
            if qty == 0:
                continue

            side = "long" if qty > 0 else "short"
            qty = abs(qty)

            positions.append(PositionSnapshot(
                symbol=symbol,
                quantity=qty,
                average_price=avg_price,
                side=side,
                pnl=pnl,
                last_price=last_price,
                market_value=qty * last_price,
                source="broker"
            ))

        return positions

    def reconcile(self, strategy_id: str) -> ReconciliationReport:
        """
        Perform full reconciliation between local and broker positions.

        Returns a detailed report of any discrepancies found.
        """
        import time
        start_time = time.time()

        local_positions = self.get_local_positions(strategy_id)
        broker_positions = self.get_broker_positions()

        report = ReconciliationReport(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            local_positions=local_positions,
            broker_positions=broker_positions
        )

        # Build lookup dicts
        local_by_symbol = {p.symbol: p for p in local_positions}
        broker_by_symbol = {p.symbol: p for p in broker_positions}

        all_symbols = set(local_by_symbol.keys()) | set(broker_by_symbol.keys())

        for symbol in all_symbols:
            local_pos = local_by_symbol.get(symbol)
            broker_pos = broker_by_symbol.get(symbol)

            discrepancies = self._compare_positions(symbol, local_pos, broker_pos)
            report.discrepancies.extend(discrepancies)

        report.reconciliation_time_ms = (time.time() - start_time) * 1000

        # Store in history
        self._reconciliation_history.append(report)

        # Alert if needed
        if report.has_discrepancies and self.config.alert_on_critical:
            if report.critical_count > 0:
                self._send_alerts(report)

        logger.info(
            f"Reconciliation complete for {strategy_id}: "
            f"{len(report.discrepancies)} discrepancies found"
        )

        return report

    def _compare_positions(
        self,
        symbol: str,
        local_pos: Optional[PositionSnapshot],
        broker_pos: Optional[PositionSnapshot]
    ) -> List[Discrepancy]:
        """Compare local and broker position for a symbol."""
        discrepancies = []

        # Case 1: Position missing locally
        if broker_pos and not local_pos:
            discrepancies.append(Discrepancy(
                symbol=symbol,
                type=DiscrepancyType.MISSING_LOCAL,
                severity=DiscrepancySeverity.CRITICAL,
                local_value=None,
                broker_value=f"{broker_pos.quantity} @ {broker_pos.average_price}",
                recommended_action=ReconciliationAction.UPDATE_LOCAL,
                description=f"Position exists in broker but not in local state",
                auto_syncable=self.config.auto_sync_missing_local
            ))
            return discrepancies

        # Case 2: Position missing in broker
        if local_pos and not broker_pos:
            discrepancies.append(Discrepancy(
                symbol=symbol,
                type=DiscrepancyType.MISSING_BROKER,
                severity=DiscrepancySeverity.WARNING,
                local_value=f"{local_pos.quantity} @ {local_pos.average_price}",
                broker_value=None,
                recommended_action=ReconciliationAction.ALERT_USER,
                description=f"Position exists locally but not in broker (may have been closed)",
                auto_syncable=False  # Needs manual review
            ))
            return discrepancies

        # Case 3: Both exist - compare details
        if local_pos and broker_pos:
            # Check side mismatch
            if local_pos.side != broker_pos.side:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.SIDE_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    local_value=local_pos.side,
                    broker_value=broker_pos.side,
                    recommended_action=ReconciliationAction.ALERT_USER,
                    description=f"Position side mismatch: local={local_pos.side}, broker={broker_pos.side}",
                    auto_syncable=False
                ))

            # Check quantity mismatch
            qty_diff = abs(local_pos.quantity - broker_pos.quantity)
            if qty_diff > self.config.quantity_tolerance:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.QUANTITY_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    local_value=local_pos.quantity,
                    broker_value=broker_pos.quantity,
                    recommended_action=ReconciliationAction.UPDATE_LOCAL,
                    description=f"Quantity mismatch: local={local_pos.quantity}, broker={broker_pos.quantity}",
                    auto_syncable=self.config.auto_sync_quantity
                ))

            # Check price mismatch (less critical)
            if local_pos.average_price > 0 and broker_pos.average_price > 0:
                price_diff_pct = abs(local_pos.average_price - broker_pos.average_price) / broker_pos.average_price
                if price_diff_pct > self.config.price_tolerance_pct:
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=DiscrepancyType.PRICE_MISMATCH,
                        severity=DiscrepancySeverity.INFO,
                        local_value=local_pos.average_price,
                        broker_value=broker_pos.average_price,
                        recommended_action=ReconciliationAction.UPDATE_LOCAL,
                        description=f"Average price differs by {price_diff_pct:.2%}",
                        auto_syncable=True
                    ))

        return discrepancies

    def _send_alerts(self, report: ReconciliationReport) -> None:
        """Send alerts for critical discrepancies."""
        for callback in self._alert_callbacks:
            try:
                callback(report)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        if self.config.alert_callback:
            try:
                self.config.alert_callback(report)
            except Exception as e:
                logger.error(f"Config alert callback failed: {e}")

    def auto_sync(
        self,
        report: ReconciliationReport,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically sync local state to match broker.

        Only syncs discrepancies marked as auto_syncable unless force=True.
        """
        if not self.config.auto_sync_enabled and not force:
            logger.warning("Auto-sync disabled. Use force=True to override.")
            return {"synced": False, "reason": "disabled"}

        results = {
            "synced": True,
            "actions": [],
            "errors": []
        }

        state = self.state_manager.load_state(report.strategy_id)
        if not state:
            from .state_manager import StrategyState
            state = StrategyState(strategy_id=report.strategy_id)

        broker_by_symbol = {p.symbol: p for p in report.broker_positions}

        for discrepancy in report.discrepancies:
            if not discrepancy.auto_syncable and not force:
                continue

            try:
                action_result = self._apply_sync_action(
                    discrepancy,
                    state,
                    broker_by_symbol
                )
                results["actions"].append(action_result)
            except Exception as e:
                results["errors"].append({
                    "symbol": discrepancy.symbol,
                    "error": str(e)
                })

        # Save updated state
        if results["actions"]:
            self.state_manager.save_state(state)
            report.auto_synced = True
            report.sync_results = results
            logger.info(f"Auto-synced {len(results['actions'])} discrepancies")

        return results

    def _apply_sync_action(
        self,
        discrepancy: Discrepancy,
        state: 'StrategyState',
        broker_positions: Dict[str, PositionSnapshot]
    ) -> Dict[str, Any]:
        """Apply a sync action for a single discrepancy."""
        symbol = discrepancy.symbol
        broker_pos = broker_positions.get(symbol)

        if discrepancy.type == DiscrepancyType.MISSING_LOCAL:
            # Add position to local state
            if broker_pos:
                state.positions[symbol] = {
                    'quantity': broker_pos.quantity,
                    'average_price': broker_pos.average_price,
                    'side': broker_pos.side,
                    'synced_from_broker': True,
                    'sync_time': datetime.now().isoformat()
                }
            return {"action": "added", "symbol": symbol}

        elif discrepancy.type == DiscrepancyType.MISSING_BROKER:
            # Remove from local state (position was closed)
            if symbol in state.positions:
                del state.positions[symbol]
            return {"action": "removed", "symbol": symbol}

        elif discrepancy.type == DiscrepancyType.QUANTITY_MISMATCH:
            # Update quantity
            if symbol in state.positions and broker_pos:
                if isinstance(state.positions[symbol], dict):
                    state.positions[symbol]['quantity'] = broker_pos.quantity
                else:
                    state.positions[symbol].quantity = broker_pos.quantity
            return {"action": "updated_quantity", "symbol": symbol}

        elif discrepancy.type == DiscrepancyType.PRICE_MISMATCH:
            # Update price
            if symbol in state.positions and broker_pos:
                if isinstance(state.positions[symbol], dict):
                    state.positions[symbol]['average_price'] = broker_pos.average_price
                else:
                    state.positions[symbol].average_price = broker_pos.average_price
            return {"action": "updated_price", "symbol": symbol}

        return {"action": "none", "symbol": symbol}

    def get_history(self, limit: int = 10) -> List[ReconciliationReport]:
        """Get recent reconciliation history."""
        return self._reconciliation_history[-limit:]

    def schedule_reconciliation(
        self,
        strategy_id: str,
        interval_minutes: Optional[int] = None
    ) -> None:
        """Schedule periodic reconciliation."""
        import threading

        interval = interval_minutes or self.config.reconcile_interval_minutes

        def run_reconciliation():
            while True:
                try:
                    report = self.reconcile(strategy_id)
                    if report.has_discrepancies and self.config.auto_sync_enabled:
                        self.auto_sync(report)
                except Exception as e:
                    logger.error(f"Scheduled reconciliation failed: {e}")

                import time
                time.sleep(interval * 60)

        thread = threading.Thread(target=run_reconciliation, daemon=True)
        thread.start()
        logger.info(f"Scheduled reconciliation every {interval} minutes")


# Convenience functions
def reconcile_positions(
    strategy_id: str,
    broker: 'ZerodhaBroker',
    state_manager: 'StrategyStateManager'
) -> ReconciliationReport:
    """Quick reconciliation without creating manager."""
    manager = ReconciliationManager(broker, state_manager)
    return manager.reconcile(strategy_id)


def check_position_sync(
    strategy_id: str,
    broker: 'ZerodhaBroker',
    state_manager: 'StrategyStateManager'
) -> bool:
    """Quick check if positions are in sync."""
    report = reconcile_positions(strategy_id, broker, state_manager)
    return not report.has_discrepancies
