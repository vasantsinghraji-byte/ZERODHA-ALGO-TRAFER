"""
Corporate Action Handler for Data Integrity.

Handles stock splits, bonuses, dividends, and rights issues to ensure
accurate historical data and proper position adjustments.

Without proper corporate action handling, backtests will show false
signals and positions will have incorrect quantities/prices.
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class CorporateActionType(Enum):
    """Types of corporate actions"""
    STOCK_SPLIT = "stock_split"         # 1:2 split doubles shares, halves price
    REVERSE_SPLIT = "reverse_split"     # 2:1 reverse split halves shares, doubles price
    BONUS = "bonus"                     # Free shares (e.g., 1:1 bonus = 1 free for each held)
    DIVIDEND = "dividend"               # Cash dividend (affects adjusted price)
    SPECIAL_DIVIDEND = "special_div"    # One-time special dividend
    RIGHTS_ISSUE = "rights_issue"       # Right to buy at discount
    MERGER = "merger"                   # Stock conversion due to merger
    DEMERGER = "demerger"               # Spin-off into separate company
    SYMBOL_CHANGE = "symbol_change"     # Trading symbol renamed


class AdjustmentType(Enum):
    """Types of adjustments applied"""
    PRICE_ADJUSTED = "price_adjusted"
    QUANTITY_ADJUSTED = "quantity_adjusted"
    BOTH_ADJUSTED = "both_adjusted"
    DIVIDEND_RECORDED = "dividend_recorded"


@dataclass
class CorporateAction:
    """
    A corporate action event.

    Attributes:
        symbol: Trading symbol (pre-action symbol if changed)
        action_type: Type of corporate action
        ex_date: Ex-date (first trading day without entitlement)
        record_date: Record date (ownership cutoff)
        ratio_from: Original units (e.g., 1 for 1:2 split)
        ratio_to: New units (e.g., 2 for 1:2 split)
        value: Cash value for dividends (per share)
        new_symbol: New symbol if symbol changed
        description: Human-readable description
        source: Data source (exchange, manual, etc.)
    """
    symbol: str
    action_type: CorporateActionType
    ex_date: date
    record_date: Optional[date] = None
    ratio_from: float = 1.0
    ratio_to: float = 1.0
    value: float = 0.0              # For dividends (per share amount)
    new_symbol: Optional[str] = None
    description: str = ""
    source: str = "manual"
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))

    def __post_init__(self):
        # Convert string dates
        if isinstance(self.ex_date, str):
            self.ex_date = datetime.strptime(self.ex_date, "%Y-%m-%d").date()
        if isinstance(self.record_date, str):
            self.record_date = datetime.strptime(self.record_date, "%Y-%m-%d").date()
        if isinstance(self.action_type, str):
            self.action_type = CorporateActionType(self.action_type)

    @property
    def adjustment_factor(self) -> float:
        """
        Calculate the price adjustment factor.

        For splits/bonus: old_price * factor = new_price
        For dividends: old_price - dividend = adjusted_price (simplified)
        """
        if self.action_type in (CorporateActionType.STOCK_SPLIT, CorporateActionType.BONUS):
            # Split 1:2 means price halves, factor = 0.5
            return self.ratio_from / self.ratio_to
        elif self.action_type == CorporateActionType.REVERSE_SPLIT:
            # Reverse split 2:1 means price doubles, factor = 2.0
            return self.ratio_to / self.ratio_from
        elif self.action_type in (CorporateActionType.DIVIDEND, CorporateActionType.SPECIAL_DIVIDEND):
            # For dividends, return the per-share value (used differently)
            return 1.0  # Price adjustment handled separately
        else:
            return 1.0

    @property
    def quantity_factor(self) -> float:
        """
        Calculate the quantity adjustment factor.

        For splits/bonus: old_qty * factor = new_qty
        """
        if self.action_type in (CorporateActionType.STOCK_SPLIT, CorporateActionType.BONUS):
            return self.ratio_to / self.ratio_from
        elif self.action_type == CorporateActionType.REVERSE_SPLIT:
            return self.ratio_from / self.ratio_to
        else:
            return 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'action_type': self.action_type.value,
            'ex_date': self.ex_date.isoformat(),
            'record_date': self.record_date.isoformat() if self.record_date else None,
            'ratio_from': self.ratio_from,
            'ratio_to': self.ratio_to,
            'value': self.value,
            'new_symbol': self.new_symbol,
            'description': self.description,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorporateAction':
        """Create from dictionary"""
        return cls(
            id=data.get('id', ''),
            symbol=data['symbol'],
            action_type=CorporateActionType(data['action_type']),
            ex_date=data['ex_date'],
            record_date=data.get('record_date'),
            ratio_from=data.get('ratio_from', 1.0),
            ratio_to=data.get('ratio_to', 1.0),
            value=data.get('value', 0.0),
            new_symbol=data.get('new_symbol'),
            description=data.get('description', ''),
            source=data.get('source', 'manual')
        )


@dataclass
class AdjustmentRecord:
    """Record of an adjustment applied"""
    timestamp: datetime
    action: CorporateAction
    adjustment_type: AdjustmentType
    affected_rows: int = 0
    original_values: Dict[str, Any] = field(default_factory=dict)
    adjusted_values: Dict[str, Any] = field(default_factory=dict)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'action_id': self.action.id,
            'symbol': self.action.symbol,
            'action_type': self.action.action_type.value,
            'adjustment_type': self.adjustment_type.value,
            'affected_rows': self.affected_rows,
            'original_values': self.original_values,
            'adjusted_values': self.adjusted_values,
            'details': self.details
        }


@dataclass
class CorporateActionConfig:
    """Configuration for corporate action handler"""
    # Data fetching
    auto_fetch_enabled: bool = False
    fetch_sources: List[str] = field(default_factory=lambda: ['nse', 'bse'])
    fetch_days_ahead: int = 30       # Fetch actions this many days ahead

    # Adjustment settings
    auto_adjust_historical: bool = True
    auto_adjust_positions: bool = True
    adjustment_precision: int = 2     # Decimal places for price adjustments

    # Audit
    enable_audit_log: bool = True
    audit_log_path: Optional[str] = None
    max_audit_records: int = 10000

    # Notifications
    notify_on_action: bool = True


class CorporateActionHandler:
    """
    Handles corporate actions for data integrity.

    Ensures historical data is properly adjusted for splits, bonuses,
    dividends, etc. Also adjusts open positions when actions occur.

    Example:
        handler = CorporateActionHandler()

        # Register a stock split
        handler.add_action(CorporateAction(
            symbol="RELIANCE",
            action_type=CorporateActionType.STOCK_SPLIT,
            ex_date=date(2024, 10, 28),
            ratio_from=1,
            ratio_to=2,
            description="1:2 stock split"
        ))

        # Adjust historical data
        adjusted_df = handler.adjust_historical_data(df, "RELIANCE")

        # Adjust positions
        handler.adjust_positions(position_manager)
    """

    def __init__(
        self,
        config: Optional[CorporateActionConfig] = None,
        storage_path: Optional[str] = None
    ):
        """
        Args:
            config: Handler configuration
            storage_path: Path to store corporate action data
        """
        self._config = config or CorporateActionConfig()
        self._storage_path = Path(storage_path) if storage_path else None
        self._lock = threading.RLock()

        # Corporate actions by symbol
        self._actions: Dict[str, List[CorporateAction]] = {}

        # Audit log
        self._audit_log: List[AdjustmentRecord] = []
        self._audit_lock = threading.Lock()

        # Callbacks
        self._callbacks: List[Callable[[CorporateAction], None]] = []

        # Load stored actions if path provided
        if self._storage_path and self._storage_path.exists():
            self._load_actions()

        logger.info("CorporateActionHandler initialized")

    # =========================================================================
    # Action Management
    # =========================================================================

    def add_action(self, action: CorporateAction) -> None:
        """
        Add a corporate action.

        Args:
            action: Corporate action to add
        """
        with self._lock:
            if action.symbol not in self._actions:
                self._actions[action.symbol] = []

            # Check for duplicates
            for existing in self._actions[action.symbol]:
                if (existing.action_type == action.action_type and
                    existing.ex_date == action.ex_date):
                    logger.warning(f"Duplicate action ignored: {action.symbol} {action.action_type.value} on {action.ex_date}")
                    return

            self._actions[action.symbol].append(action)
            # Sort by ex_date
            self._actions[action.symbol].sort(key=lambda a: a.ex_date)

        logger.info(f"Corporate action added: {action.symbol} {action.action_type.value} on {action.ex_date}")

        # Save to storage
        self._save_actions()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def add_stock_split(
        self,
        symbol: str,
        ex_date: Union[date, str],
        ratio_from: int,
        ratio_to: int,
        description: str = ""
    ) -> CorporateAction:
        """
        Convenience method to add a stock split.

        Args:
            symbol: Trading symbol
            ex_date: Ex-date
            ratio_from: Original units (e.g., 1)
            ratio_to: New units (e.g., 2 for 1:2 split)
            description: Optional description

        Returns:
            The created CorporateAction
        """
        action = CorporateAction(
            symbol=symbol,
            action_type=CorporateActionType.STOCK_SPLIT,
            ex_date=ex_date,
            ratio_from=ratio_from,
            ratio_to=ratio_to,
            description=description or f"{ratio_from}:{ratio_to} stock split"
        )
        self.add_action(action)
        return action

    def add_bonus(
        self,
        symbol: str,
        ex_date: Union[date, str],
        ratio_from: int,
        ratio_to: int,
        description: str = ""
    ) -> CorporateAction:
        """
        Convenience method to add a bonus issue.

        Args:
            symbol: Trading symbol
            ex_date: Ex-date
            ratio_from: Existing shares (e.g., 1)
            ratio_to: Total after bonus (e.g., 2 for 1:1 bonus)
            description: Optional description

        Returns:
            The created CorporateAction
        """
        action = CorporateAction(
            symbol=symbol,
            action_type=CorporateActionType.BONUS,
            ex_date=ex_date,
            ratio_from=ratio_from,
            ratio_to=ratio_to,
            description=description or f"{ratio_from}:{ratio_to - ratio_from} bonus"
        )
        self.add_action(action)
        return action

    def add_dividend(
        self,
        symbol: str,
        ex_date: Union[date, str],
        dividend_per_share: float,
        is_special: bool = False,
        description: str = ""
    ) -> CorporateAction:
        """
        Convenience method to add a dividend.

        Args:
            symbol: Trading symbol
            ex_date: Ex-dividend date
            dividend_per_share: Dividend amount per share
            is_special: True for special/one-time dividend
            description: Optional description

        Returns:
            The created CorporateAction
        """
        action_type = CorporateActionType.SPECIAL_DIVIDEND if is_special else CorporateActionType.DIVIDEND
        action = CorporateAction(
            symbol=symbol,
            action_type=action_type,
            ex_date=ex_date,
            value=dividend_per_share,
            description=description or f"Dividend Rs.{dividend_per_share}/share"
        )
        self.add_action(action)
        return action

    def add_rights_issue(
        self,
        symbol: str,
        ex_date: Union[date, str],
        ratio_from: int,
        ratio_to: int,
        price: float,
        description: str = ""
    ) -> CorporateAction:
        """
        Convenience method to add a rights issue.

        Args:
            symbol: Trading symbol
            ex_date: Ex-date
            ratio_from: Existing shares entitled
            ratio_to: Rights shares offered
            price: Rights issue price
            description: Optional description

        Returns:
            The created CorporateAction
        """
        action = CorporateAction(
            symbol=symbol,
            action_type=CorporateActionType.RIGHTS_ISSUE,
            ex_date=ex_date,
            ratio_from=ratio_from,
            ratio_to=ratio_to,
            value=price,
            description=description or f"{ratio_from}:{ratio_to} rights @ Rs.{price}"
        )
        self.add_action(action)
        return action

    def get_actions(
        self,
        symbol: Optional[str] = None,
        action_type: Optional[CorporateActionType] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """
        Get corporate actions with optional filters.

        Args:
            symbol: Filter by symbol
            action_type: Filter by action type
            from_date: Filter from this date
            to_date: Filter to this date

        Returns:
            List of matching corporate actions
        """
        with self._lock:
            if symbol:
                actions = self._actions.get(symbol, []).copy()
            else:
                actions = []
                for symbol_actions in self._actions.values():
                    actions.extend(symbol_actions)

        # Apply filters
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        if from_date:
            actions = [a for a in actions if a.ex_date >= from_date]
        if to_date:
            actions = [a for a in actions if a.ex_date <= to_date]

        return sorted(actions, key=lambda a: a.ex_date)

    def get_pending_actions(self, as_of: Optional[date] = None) -> List[CorporateAction]:
        """Get actions with ex_date >= as_of (default: today)"""
        as_of = as_of or date.today()
        return self.get_actions(from_date=as_of)

    def remove_action(self, action_id: str) -> bool:
        """Remove an action by ID"""
        with self._lock:
            for symbol, actions in self._actions.items():
                for i, action in enumerate(actions):
                    if action.id == action_id:
                        del actions[i]
                        self._save_actions()
                        logger.info(f"Action removed: {action_id}")
                        return True
        return False

    # =========================================================================
    # Historical Data Adjustment
    # =========================================================================

    def adjust_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_columns: List[str] = None,
        volume_column: str = 'volume',
        date_column: str = None,
        in_place: bool = False
    ) -> pd.DataFrame:
        """
        Adjust historical OHLCV data for corporate actions.

        Back-adjusts prices so that historical data reflects splits/bonuses.
        This ensures consistent analysis across time.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            price_columns: Columns to adjust (default: open, high, low, close)
            volume_column: Volume column to inverse-adjust
            date_column: Date column name (default: use index)
            in_place: Modify DataFrame in place

        Returns:
            Adjusted DataFrame
        """
        if not in_place:
            df = df.copy()

        price_columns = price_columns or ['open', 'high', 'low', 'close']

        # Get actions for this symbol
        actions = self.get_actions(symbol=symbol)
        if not actions:
            return df

        # Determine date column
        if date_column:
            dates = pd.to_datetime(df[date_column]).dt.date
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        else:
            dates = pd.to_datetime(df.index).date

        total_adjusted = 0

        # Process each action (oldest first for cumulative adjustment)
        for action in sorted(actions, key=lambda a: a.ex_date):
            # Find rows before ex_date that need adjustment
            mask = dates < action.ex_date

            if not mask.any():
                continue

            rows_affected = mask.sum()

            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                factor = action.adjustment_factor
                qty_factor = action.quantity_factor

                # Adjust prices
                for col in price_columns:
                    if col in df.columns:
                        original_mean = df.loc[mask, col].mean()
                        df.loc[mask, col] = df.loc[mask, col] * factor
                        adjusted_mean = df.loc[mask, col].mean()

                        # Record adjustment
                        self._record_adjustment(
                            action=action,
                            adjustment_type=AdjustmentType.PRICE_ADJUSTED,
                            affected_rows=rows_affected,
                            original_values={'column': col, 'mean': original_mean},
                            adjusted_values={'mean': adjusted_mean, 'factor': factor},
                            details=f"Adjusted {col} by factor {factor}"
                        )

                # Inverse-adjust volume
                if volume_column and volume_column in df.columns:
                    df.loc[mask, volume_column] = (
                        df.loc[mask, volume_column] / qty_factor
                    ).astype(int)

                total_adjusted += rows_affected
                logger.info(
                    f"Adjusted {rows_affected} rows for {symbol} "
                    f"{action.action_type.value} on {action.ex_date} (factor: {factor})"
                )

            elif action.action_type in (
                CorporateActionType.DIVIDEND,
                CorporateActionType.SPECIAL_DIVIDEND
            ):
                # For dividends, we can optionally adjust prices
                # This is less common but sometimes done for total return analysis
                if action.value > 0:
                    for col in price_columns:
                        if col in df.columns:
                            # Simple adjustment: subtract dividend from pre-ex prices
                            # More sophisticated: use dividend yield adjustment
                            pass  # Typically dividends are handled separately

                    self._record_adjustment(
                        action=action,
                        adjustment_type=AdjustmentType.DIVIDEND_RECORDED,
                        affected_rows=0,
                        original_values={},
                        adjusted_values={'dividend': action.value},
                        details=f"Dividend of Rs.{action.value} recorded (no price adjustment)"
                    )

        if total_adjusted > 0:
            logger.info(f"Total {total_adjusted} rows adjusted for {symbol}")

        return df

    def get_adjustment_factor(
        self,
        symbol: str,
        from_date: date,
        to_date: Optional[date] = None
    ) -> float:
        """
        Calculate cumulative adjustment factor between two dates.

        Useful for adjusting a single price point.

        Args:
            symbol: Trading symbol
            from_date: Starting date
            to_date: Ending date (default: today)

        Returns:
            Cumulative adjustment factor
        """
        to_date = to_date or date.today()

        actions = self.get_actions(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date
        )

        factor = 1.0
        for action in actions:
            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                factor *= action.adjustment_factor

        return factor

    # =========================================================================
    # Position Adjustment
    # =========================================================================

    def adjust_positions(
        self,
        position_manager: Any,
        as_of: Optional[date] = None
    ) -> List[AdjustmentRecord]:
        """
        Adjust open positions for corporate actions.

        Should be called at market open or when processing actions.

        Args:
            position_manager: PositionManager instance
            as_of: Date to check actions for (default: today)

        Returns:
            List of adjustment records
        """
        as_of = as_of or date.today()
        adjustments = []

        # Get today's actions
        actions = self.get_actions(from_date=as_of, to_date=as_of)

        if not actions:
            return adjustments

        positions = position_manager.get_all_positions()

        for action in actions:
            # Find matching position
            matching_pos = None
            for pos in positions:
                if pos.symbol == action.symbol:
                    matching_pos = pos
                    break

            if not matching_pos:
                continue

            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                # Adjust quantity and average price
                old_qty = matching_pos.quantity
                old_avg = matching_pos.average_price

                qty_factor = action.quantity_factor
                price_factor = action.adjustment_factor

                new_qty = int(old_qty * qty_factor)
                new_avg = old_avg * price_factor

                # Update position (implementation depends on PositionManager)
                if hasattr(position_manager, '_positions'):
                    position_manager._positions[action.symbol].quantity = new_qty
                    position_manager._positions[action.symbol].average_price = new_avg
                    position_manager._positions[action.symbol].buy_value = new_qty * new_avg

                record = self._record_adjustment(
                    action=action,
                    adjustment_type=AdjustmentType.BOTH_ADJUSTED,
                    affected_rows=1,
                    original_values={'quantity': old_qty, 'average_price': old_avg},
                    adjusted_values={'quantity': new_qty, 'average_price': new_avg},
                    details=f"Position adjusted: {old_qty} @ {old_avg:.2f} -> {new_qty} @ {new_avg:.2f}"
                )
                adjustments.append(record)

                logger.info(
                    f"Position adjusted for {action.symbol}: "
                    f"{old_qty} @ {old_avg:.2f} -> {new_qty} @ {new_avg:.2f}"
                )

            elif action.action_type == CorporateActionType.SYMBOL_CHANGE:
                # Handle symbol change
                if action.new_symbol and hasattr(position_manager, '_positions'):
                    old_pos = position_manager._positions.pop(action.symbol, None)
                    if old_pos:
                        old_pos.symbol = action.new_symbol
                        position_manager._positions[action.new_symbol] = old_pos

                        record = self._record_adjustment(
                            action=action,
                            adjustment_type=AdjustmentType.QUANTITY_ADJUSTED,
                            affected_rows=1,
                            original_values={'symbol': action.symbol},
                            adjusted_values={'symbol': action.new_symbol},
                            details=f"Symbol changed: {action.symbol} -> {action.new_symbol}"
                        )
                        adjustments.append(record)

        return adjustments

    def calculate_adjusted_quantity(
        self,
        symbol: str,
        original_quantity: int,
        original_date: date,
        target_date: Optional[date] = None
    ) -> int:
        """
        Calculate what quantity would be held after corporate actions.

        Args:
            symbol: Trading symbol
            original_quantity: Quantity held on original_date
            original_date: Date when position was opened
            target_date: Date to calculate for (default: today)

        Returns:
            Adjusted quantity
        """
        target_date = target_date or date.today()

        actions = self.get_actions(
            symbol=symbol,
            from_date=original_date,
            to_date=target_date
        )

        quantity = float(original_quantity)
        for action in actions:
            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                quantity *= action.quantity_factor

        return int(quantity)

    # =========================================================================
    # Data Fetching
    # =========================================================================

    def fetch_actions_from_nse(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """
        Fetch corporate actions from NSE website.

        Note: This is a placeholder. Actual implementation would need
        web scraping or API access to NSE corporate actions page.

        Args:
            symbol: Filter by symbol
            from_date: From date
            to_date: To date

        Returns:
            List of corporate actions
        """
        # Placeholder - actual implementation would scrape NSE website
        # or use a data provider API
        logger.warning("NSE fetch not implemented - use manual entry or data provider")
        return []

    def fetch_actions_from_provider(
        self,
        provider: str,
        symbol: Optional[str] = None,
        **kwargs
    ) -> List[CorporateAction]:
        """
        Fetch corporate actions from a data provider.

        Supports extensibility for different data providers.

        Args:
            provider: Provider name (e.g., 'zerodha', 'yahoo', 'alpha_vantage')
            symbol: Filter by symbol
            **kwargs: Provider-specific arguments

        Returns:
            List of corporate actions
        """
        if provider == 'zerodha' and hasattr(self, '_kite'):
            # Zerodha doesn't have a direct corporate actions API
            # Would need to parse from instruments or external source
            pass

        logger.warning(f"Provider '{provider}' not implemented")
        return []

    # =========================================================================
    # Audit Log
    # =========================================================================

    def _record_adjustment(
        self,
        action: CorporateAction,
        adjustment_type: AdjustmentType,
        affected_rows: int,
        original_values: Dict,
        adjusted_values: Dict,
        details: str
    ) -> AdjustmentRecord:
        """Record an adjustment in the audit log"""
        record = AdjustmentRecord(
            timestamp=datetime.now(),
            action=action,
            adjustment_type=adjustment_type,
            affected_rows=affected_rows,
            original_values=original_values,
            adjusted_values=adjusted_values,
            details=details
        )

        if self._config.enable_audit_log:
            with self._audit_lock:
                self._audit_log.append(record)
                # Trim if too large
                if len(self._audit_log) > self._config.max_audit_records:
                    self._audit_log = self._audit_log[-self._config.max_audit_records:]

            # Write to file if configured
            if self._config.audit_log_path:
                self._write_audit_record(record)

        return record

    def _write_audit_record(self, record: AdjustmentRecord) -> None:
        """Write audit record to file"""
        try:
            path = Path(self._config.audit_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit record: {e}")

    def get_audit_log(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[AdjustmentRecord]:
        """Get audit log records"""
        with self._audit_lock:
            records = self._audit_log.copy()

        if symbol:
            records = [r for r in records if r.action.symbol == symbol]

        return records[-limit:]

    def export_audit_log(self, path: str) -> None:
        """Export audit log to JSON file"""
        with self._audit_lock:
            records = [r.to_dict() for r in self._audit_log]

        with open(path, 'w') as f:
            json.dump(records, f, indent=2)

        logger.info(f"Audit log exported to {path}")

    # =========================================================================
    # Storage
    # =========================================================================

    def _save_actions(self) -> None:
        """Save actions to storage"""
        if not self._storage_path:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            with self._lock:
                for symbol, actions in self._actions.items():
                    data[symbol] = [a.to_dict() for a in actions]

            with open(self._storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save actions: {e}")

    def _load_actions(self) -> None:
        """Load actions from storage"""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)

            with self._lock:
                for symbol, actions in data.items():
                    self._actions[symbol] = [
                        CorporateAction.from_dict(a) for a in actions
                    ]

            total = sum(len(a) for a in self._actions.values())
            logger.info(f"Loaded {total} corporate actions from storage")

        except Exception as e:
            logger.error(f"Failed to load actions: {e}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_callback(self, callback: Callable[[CorporateAction], None]) -> None:
        """Register callback for new corporate actions"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[CorporateAction], None]) -> bool:
        """Remove callback"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def summary(self) -> str:
        """Get summary of corporate actions"""
        lines = ["Corporate Actions Summary:"]
        lines.append("-" * 50)

        with self._lock:
            total = sum(len(a) for a in self._actions.values())
            lines.append(f"Total actions: {total}")
            lines.append(f"Symbols tracked: {len(self._actions)}")

            # By type
            by_type: Dict[str, int] = {}
            for actions in self._actions.values():
                for action in actions:
                    key = action.action_type.value
                    by_type[key] = by_type.get(key, 0) + 1

            if by_type:
                lines.append("\nBy type:")
                for action_type, count in sorted(by_type.items()):
                    lines.append(f"  {action_type}: {count}")

            # Pending
            pending = self.get_pending_actions()
            if pending:
                lines.append(f"\nPending actions: {len(pending)}")
                for action in pending[:5]:
                    lines.append(f"  {action.symbol} {action.action_type.value} on {action.ex_date}")

        lines.append(f"\nAudit records: {len(self._audit_log)}")

        return "\n".join(lines)


# =============================================================================
# Global Instance
# =============================================================================

_global_handler: Optional[CorporateActionHandler] = None
_global_handler_lock = threading.Lock()


def get_corporate_action_handler() -> CorporateActionHandler:
    """Get the global corporate action handler instance"""
    global _global_handler
    if _global_handler is None:
        with _global_handler_lock:
            if _global_handler is None:
                _global_handler = CorporateActionHandler()
    return _global_handler


def set_corporate_action_handler(handler: CorporateActionHandler) -> None:
    """Set the global corporate action handler instance"""
    global _global_handler
    with _global_handler_lock:
        _global_handler = handler
