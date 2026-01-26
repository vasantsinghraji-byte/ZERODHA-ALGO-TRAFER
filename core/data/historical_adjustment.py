"""
Historical Data Adjustment for Backtesting.

Integrates with CorporateActionHandler to provide on-the-fly adjustment
during backtest runs. Supports both back-adjustment and forward-fill modes,
with optional dividend reinvestment simulation.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Iterator, Union
import bisect

import pandas as pd
import numpy as np

from .corporate_actions import (
    CorporateActionHandler,
    CorporateAction,
    CorporateActionType,
    get_corporate_action_handler,
)

logger = logging.getLogger(__name__)


class AdjustmentMode(Enum):
    """Adjustment mode for historical data."""
    BACK_ADJUST = "back_adjust"
    FORWARD_FILL = "forward_fill"
    NONE = "none"


class DividendHandling(Enum):
    """How to handle dividends in historical data."""
    IGNORE = "ignore"
    PRICE_ADJUST = "price_adjust"
    REINVEST = "reinvest"


@dataclass
class DividendReinvestment:
    """Record of a dividend reinvestment."""
    date: date
    symbol: str
    dividend_per_share: float
    shares_held: float
    total_dividend: float
    shares_purchased: float
    purchase_price: float
    new_total_shares: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'symbol': self.symbol,
            'dividend_per_share': self.dividend_per_share,
            'shares_held': self.shares_held,
            'total_dividend': self.total_dividend,
            'shares_purchased': self.shares_purchased,
            'purchase_price': self.purchase_price,
            'new_total_shares': self.new_total_shares
        }


@dataclass
class HistoricalAdjustmentConfig:
    """Configuration for historical data adjustment."""
    adjustment_mode: AdjustmentMode = AdjustmentMode.BACK_ADJUST
    dividend_handling: DividendHandling = DividendHandling.IGNORE
    reinvest_commission: float = 0.0
    reinvest_fractional: bool = True
    price_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    volume_column: str = 'volume'
    cache_adjusted_data: bool = True
    max_cache_size: int = 100


class HistoricalDataAdjuster:
    """
    Adjusts historical data for corporate actions during backtest.

    Example:
        adjuster = HistoricalDataAdjuster()
        adjusted_df = adjuster.adjust_dataframe(df, "RELIANCE")

        # With dividend reinvestment
        config = HistoricalAdjustmentConfig(dividend_handling=DividendHandling.REINVEST)
        adjuster = HistoricalDataAdjuster(config=config)
        result = adjuster.run_with_reinvestment(df, "RELIANCE", initial_shares=100)
    """

    def __init__(
        self,
        config: Optional[HistoricalAdjustmentConfig] = None,
        action_handler: Optional[CorporateActionHandler] = None
    ):
        self._config = config or HistoricalAdjustmentConfig()
        self._action_handler = action_handler or get_corporate_action_handler()
        self._lock = threading.RLock()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_keys: List[str] = []
        self._reinvestments: List[DividendReinvestment] = []
        logger.info("HistoricalDataAdjuster initialized")

    def adjust_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of_date: Optional[date] = None,
        mode: Optional[AdjustmentMode] = None
    ) -> pd.DataFrame:
        """Adjust entire DataFrame for corporate actions."""
        mode = mode or self._config.adjustment_mode

        if mode == AdjustmentMode.NONE:
            return df.copy()

        cache_key = self._get_cache_key(symbol, as_of_date, mode)
        if self._config.cache_adjusted_data and cache_key in self._cache:
            return self._cache[cache_key].copy()

        actions = self._action_handler.get_actions(symbol=symbol)
        if not actions:
            return df.copy()

        if mode == AdjustmentMode.BACK_ADJUST:
            adjusted = self._back_adjust(df, symbol, actions, as_of_date)
        else:
            adjusted = self._forward_fill(df, symbol, actions, as_of_date)

        if self._config.cache_adjusted_data:
            self._update_cache(cache_key, adjusted)

        return adjusted

    def _back_adjust(
        self,
        df: pd.DataFrame,
        symbol: str,
        actions: List[CorporateAction],
        as_of_date: Optional[date]
    ) -> pd.DataFrame:
        """Back-adjust historical prices."""
        adjusted = df.copy()

        if isinstance(adjusted.index, pd.DatetimeIndex):
            dates = adjusted.index.date
        else:
            dates = pd.to_datetime(adjusted.index).date

        if as_of_date:
            actions = [a for a in actions if a.ex_date <= as_of_date]

        for action in sorted(actions, key=lambda a: a.ex_date, reverse=True):
            mask = dates < action.ex_date

            if not mask.any():
                continue

            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                factor = action.adjustment_factor
                qty_factor = action.quantity_factor

                for col in self._config.price_columns:
                    if col in adjusted.columns:
                        adjusted.loc[mask, col] = adjusted.loc[mask, col] * factor

                if self._config.volume_column in adjusted.columns:
                    adjusted.loc[mask, self._config.volume_column] = (
                        adjusted.loc[mask, self._config.volume_column] / qty_factor
                    ).astype(int)

            elif action.action_type in (
                CorporateActionType.DIVIDEND,
                CorporateActionType.SPECIAL_DIVIDEND
            ):
                if self._config.dividend_handling == DividendHandling.PRICE_ADJUST:
                    for col in self._config.price_columns:
                        if col in adjusted.columns:
                            adjusted.loc[mask, col] = (
                                adjusted.loc[mask, col] - action.value
                            ).clip(lower=0.01)

        return adjusted

    def _forward_fill(
        self,
        df: pd.DataFrame,
        symbol: str,
        actions: List[CorporateAction],
        as_of_date: Optional[date]
    ) -> pd.DataFrame:
        """Forward-fill adjustments."""
        adjusted = df.copy()

        if isinstance(adjusted.index, pd.DatetimeIndex):
            dates = adjusted.index.date
        else:
            dates = pd.to_datetime(adjusted.index).date

        for action in sorted(actions, key=lambda a: a.ex_date):
            if as_of_date and action.ex_date > as_of_date:
                continue

            mask = dates >= action.ex_date

            if not mask.any():
                continue

            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                factor = 1.0 / action.adjustment_factor
                qty_factor = 1.0 / action.quantity_factor

                for col in self._config.price_columns:
                    if col in adjusted.columns:
                        adjusted.loc[mask, col] = adjusted.loc[mask, col] * factor

                if self._config.volume_column in adjusted.columns:
                    adjusted.loc[mask, self._config.volume_column] = (
                        adjusted.loc[mask, self._config.volume_column] * qty_factor
                    ).astype(int)

        return adjusted

    def stream_adjusted_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        as_of_date: Optional[date] = None
    ) -> Iterator[Dict[str, Any]]:
        """Stream adjusted bars one at a time for memory efficiency."""
        actions = self._action_handler.get_actions(symbol=symbol)
        if as_of_date:
            actions = [a for a in actions if a.ex_date <= as_of_date]
        actions = sorted(actions, key=lambda a: a.ex_date)

        action_dates = [a.ex_date for a in actions]
        cumulative_factors = self._compute_cumulative_factors(actions)

        for idx, row in df.iterrows():
            bar_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
            factor, qty_factor = self._get_factor_for_date(
                bar_date, action_dates, cumulative_factors
            )

            adjusted_bar = {'index': idx, 'date': bar_date, 'symbol': symbol}

            for col in self._config.price_columns:
                if col in row.index:
                    adjusted_bar[col] = row[col] * factor

            if self._config.volume_column in row.index:
                adjusted_bar[self._config.volume_column] = int(
                    row[self._config.volume_column] / qty_factor
                )

            for col in row.index:
                if col not in adjusted_bar and col not in self._config.price_columns:
                    if col != self._config.volume_column:
                        adjusted_bar[col] = row[col]

            yield adjusted_bar

    def _compute_cumulative_factors(
        self,
        actions: List[CorporateAction]
    ) -> List[Tuple[float, float]]:
        """Compute cumulative adjustment factors."""
        if not actions:
            return []

        factors = []
        cumulative_price = 1.0
        cumulative_qty = 1.0

        for action in reversed(actions):
            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                cumulative_price *= action.adjustment_factor
                cumulative_qty *= action.quantity_factor

            factors.append((cumulative_price, cumulative_qty))

        return list(reversed(factors))

    def _get_factor_for_date(
        self,
        bar_date: date,
        action_dates: List[date],
        cumulative_factors: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Get adjustment factor for a specific date using binary search."""
        if not action_dates:
            return (1.0, 1.0)

        idx = bisect.bisect_right(action_dates, bar_date)

        if idx == 0:
            return cumulative_factors[0]
        elif idx >= len(action_dates):
            return (1.0, 1.0)
        else:
            return cumulative_factors[idx]

    def run_with_reinvestment(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_shares: float,
        initial_cash: float = 0.0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Run dividend reinvestment simulation (DRIP)."""
        actions = self._action_handler.get_actions(
            symbol=symbol, action_type=CorporateActionType.DIVIDEND
        )
        special_divs = self._action_handler.get_actions(
            symbol=symbol, action_type=CorporateActionType.SPECIAL_DIVIDEND
        )
        dividend_actions = sorted(actions + special_divs, key=lambda a: a.ex_date)

        split_actions = [
            a for a in self._action_handler.get_actions(symbol=symbol)
            if a.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            )
        ]
        split_actions = sorted(split_actions, key=lambda a: a.ex_date)

        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        else:
            dates = pd.to_datetime(df.index).date

        if start_date:
            mask = dates >= start_date
            df = df[mask]
            dates = dates[mask]
        if end_date:
            mask = dates <= end_date
            df = df[mask]
            dates = dates[mask]

        shares = float(initial_shares)
        cash = float(initial_cash)
        total_dividends = 0.0
        reinvestments = []
        equity_curve = []

        close_col = 'close' if 'close' in df.columns else 'Close'
        processed_divs = set()
        processed_splits = set()

        for i, (idx, row) in enumerate(df.iterrows()):
            current_date = dates[i] if i < len(dates) else idx.date()
            current_price = row[close_col]

            for action in split_actions:
                if action.id not in processed_splits and action.ex_date == current_date:
                    shares *= action.quantity_factor
                    processed_splits.add(action.id)

            for action in dividend_actions:
                if action.id not in processed_divs and action.ex_date == current_date:
                    dividend_amount = shares * action.value
                    total_dividends += dividend_amount

                    if self._config.dividend_handling == DividendHandling.REINVEST:
                        net_dividend = dividend_amount - self._config.reinvest_commission
                        if net_dividend > 0:
                            if self._config.reinvest_fractional:
                                new_shares = net_dividend / current_price
                            else:
                                new_shares = int(net_dividend / current_price)
                                cash += net_dividend - (new_shares * current_price)

                            old_shares = shares
                            shares += new_shares

                            reinvestments.append(DividendReinvestment(
                                date=current_date,
                                symbol=symbol,
                                dividend_per_share=action.value,
                                shares_held=old_shares,
                                total_dividend=dividend_amount,
                                shares_purchased=new_shares,
                                purchase_price=current_price,
                                new_total_shares=shares
                            ))
                    else:
                        cash += dividend_amount

                    processed_divs.add(action.id)

            equity = (shares * current_price) + cash
            equity_curve.append({
                'date': current_date,
                'shares': shares,
                'price': current_price,
                'cash': cash,
                'equity': equity
            })

        self._reinvestments = reinvestments

        initial_equity = initial_shares * df[close_col].iloc[0] + initial_cash
        final_equity = equity_curve[-1]['equity'] if equity_curve else initial_equity

        return {
            'symbol': symbol,
            'initial_shares': initial_shares,
            'final_shares': shares,
            'initial_cash': initial_cash,
            'final_cash': cash,
            'total_dividends': total_dividends,
            'reinvestments': reinvestments,
            'reinvestment_count': len(reinvestments),
            'equity_curve': equity_curve,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': (final_equity - initial_equity) / initial_equity * 100,
            'shares_from_reinvestment': shares - initial_shares
        }

    def get_reinvestments(self, symbol: Optional[str] = None) -> List[DividendReinvestment]:
        """Get dividend reinvestment records."""
        if symbol:
            return [r for r in self._reinvestments if r.symbol == symbol]
        return self._reinvestments.copy()

    def create_adjusted_data_source(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Create adjusted data dict for EventDrivenBacktester."""
        if isinstance(data, pd.DataFrame):
            if not symbols or len(symbols) != 1:
                raise ValueError("symbols must be provided for single DataFrame")
            data_dict = {symbols[0]: data}
        else:
            data_dict = data

        adjusted_dict = {}
        for symbol, df in data_dict.items():
            adjusted_dict[symbol] = self.adjust_dataframe(df, symbol)

        return adjusted_dict

    def wrap_backtest_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convenience wrapper for backtest data adjustment."""
        return self.adjust_dataframe(df, symbol)

    def _get_cache_key(
        self, symbol: str, as_of_date: Optional[date], mode: AdjustmentMode
    ) -> str:
        date_str = as_of_date.isoformat() if as_of_date else "latest"
        return f"{symbol}_{date_str}_{mode.value}"

    def _update_cache(self, key: str, df: pd.DataFrame) -> None:
        with self._lock:
            if key in self._cache:
                self._cache_keys.remove(key)
            elif len(self._cache_keys) >= self._config.max_cache_size:
                oldest = self._cache_keys.pop(0)
                del self._cache[oldest]

            self._cache[key] = df.copy()
            self._cache_keys.append(key)

    def clear_cache(self) -> None:
        """Clear adjustment cache."""
        with self._lock:
            self._cache.clear()
            self._cache_keys.clear()

    def get_adjustment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of adjustments for a symbol."""
        actions = self._action_handler.get_actions(symbol=symbol)

        splits = [a for a in actions if a.action_type == CorporateActionType.STOCK_SPLIT]
        bonuses = [a for a in actions if a.action_type == CorporateActionType.BONUS]
        dividends = [a for a in actions if a.action_type in (
            CorporateActionType.DIVIDEND, CorporateActionType.SPECIAL_DIVIDEND
        )]

        cumulative_price = 1.0
        cumulative_qty = 1.0
        total_dividends = 0.0

        for action in actions:
            if action.action_type in (
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.BONUS,
                CorporateActionType.REVERSE_SPLIT
            ):
                cumulative_price *= action.adjustment_factor
                cumulative_qty *= action.quantity_factor
            elif action.action_type in (
                CorporateActionType.DIVIDEND,
                CorporateActionType.SPECIAL_DIVIDEND
            ):
                total_dividends += action.value

        return {
            'symbol': symbol,
            'total_actions': len(actions),
            'splits': len(splits),
            'bonuses': len(bonuses),
            'dividends': len(dividends),
            'cumulative_price_factor': cumulative_price,
            'cumulative_quantity_factor': cumulative_qty,
            'total_dividend_per_share': total_dividends,
            'adjustment_mode': self._config.adjustment_mode.value,
            'dividend_handling': self._config.dividend_handling.value
        }


def adjust_for_backtest(
    df: pd.DataFrame,
    symbol: str,
    mode: AdjustmentMode = AdjustmentMode.BACK_ADJUST
) -> pd.DataFrame:
    """Quick function to adjust data for backtest."""
    adjuster = HistoricalDataAdjuster()
    return adjuster.adjust_dataframe(df, symbol, mode=mode)


def simulate_drip(
    df: pd.DataFrame,
    symbol: str,
    initial_shares: float,
    initial_cash: float = 0.0
) -> Dict[str, Any]:
    """Simulate dividend reinvestment plan (DRIP)."""
    config = HistoricalAdjustmentConfig(dividend_handling=DividendHandling.REINVEST)
    adjuster = HistoricalDataAdjuster(config=config)
    return adjuster.run_with_reinvestment(df, symbol, initial_shares, initial_cash)


_global_adjuster: Optional[HistoricalDataAdjuster] = None
_global_adjuster_lock = threading.Lock()


def get_historical_adjuster() -> HistoricalDataAdjuster:
    """Get the global historical data adjuster instance."""
    global _global_adjuster
    if _global_adjuster is None:
        with _global_adjuster_lock:
            if _global_adjuster is None:
                _global_adjuster = HistoricalDataAdjuster()
    return _global_adjuster


def set_historical_adjuster(adjuster: HistoricalDataAdjuster) -> None:
    """Set the global historical data adjuster instance."""
    global _global_adjuster
    with _global_adjuster_lock:
        _global_adjuster = adjuster
