# -*- coding: utf-8 -*-
"""
Backtesting Engine - Time Machine for Trading!
==============================================
Test your strategies on old data to see how they would have performed.

Includes both:
- Backtester: Legacy bar-by-bar backtester (backward compatible)
- EventDrivenBacktester: New event-driven backtester (unified architecture)

It's like a practice game before the real match!

PRECISION NOTE:
All financial calculations use Decimal for precision.
This prevents floating-point drift over thousands of trades.
Example: float(0.1 + 0.2) = 0.30000000000000004, but Decimal is exact.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

from strategies.base import Strategy, Signal, SignalType

logger = logging.getLogger(__name__)


# ============== PRECISION HELPERS ==============

# Standard precision for currency (2 decimal places for paise)
CURRENCY_PRECISION = Decimal("0.01")


def _to_decimal(value: Union[float, int, Decimal, None]) -> Decimal:
    """
    Convert a value to Decimal with proper precision.

    Uses string conversion to avoid float precision issues.
    Example: Decimal(0.1) != Decimal("0.1") due to float representation.
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value.quantize(CURRENCY_PRECISION, rounding=ROUND_HALF_UP)
    # Convert via string to avoid float precision issues
    return Decimal(str(value)).quantize(CURRENCY_PRECISION, rounding=ROUND_HALF_UP)


def _to_float(value: Union[Decimal, float, int]) -> float:
    """Convert Decimal back to float for external interfaces."""
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


# ============== DATA CLASSES ==============

@dataclass
class Trade:
    """Record of a single trade"""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float = 0.0
    quantity: int = 1
    stop_loss: float = 0.0
    target: float = 0.0
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    exit_reason: str = ""

    @property
    def is_winner(self) -> bool:
        return self.profit_loss > 0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None


@dataclass
class BacktestResult:
    """Results of a backtest run"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float
    final_capital: float

    # Trades
    trades: List[Trade] = field(default_factory=list)

    # Metrics (calculated)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    # GHOST TRADE FIX: Track rejected trades due to insufficient capital
    rejected_trades: int = 0

    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    return_pct: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)


# ============== BACKTESTER ==============

class Backtester:
    """
    The Time Machine for Testing Strategies!

    How to use:
        1. Create a backtester with your capital
        2. Give it data and a strategy
        3. Run the backtest
        4. See the results!

    Example:
        >>> bt = Backtester(capital=100000)
        >>> result = bt.run(data, strategy)
        >>> print(result.net_profit)

    Reality Gap Fix (2024):
        Pass a BacktestConfig for realistic execution modeling:
        - BID_ASK fill model: Buy at ask, sell at bid
        - OHLC fill model: Worst-case fills using high/low
        - Slippage modeling: Market impact cost
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 10.0,
        commission: float = 0.0,
        slippage_pct: float = 0.1,
        config: 'BacktestConfig' = None
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting money (default Rs.1,00,000)
            position_size_pct: % of capital per trade (default 10%)
            commission: Commission per trade (default 0)
            slippage_pct: Slippage/impact cost (default 0.1%)
            config: BacktestConfig for realistic execution (overrides other params)
        """
        # Store config for realistic execution pricing
        self._config = config

        # Use config values if provided, otherwise use legacy params
        if config:
            self.initial_capital = config.initial_capital
            self.position_size_pct = config.position_size_pct
            self.commission = 0  # Included in config's total_cost_pct
            self.slippage_pct = config.slippage_pct  # Only used as fallback
        else:
            self.initial_capital = initial_capital
            self.position_size_pct = position_size_pct
            self.commission = commission
            self.slippage_pct = slippage_pct

        # State - use Decimal for precise financial calculations
        # This prevents floating-point drift over thousands of trades
        self._capital: Decimal = _to_decimal(self.initial_capital)
        self._position: Optional[Trade] = None
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []  # Float OK for charting
        self._dates: List[datetime] = []
        # GHOST TRADE FIX: Track rejected trades due to insufficient capital
        self._rejected_trades: int = 0

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "STOCK"
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            strategy: Trading strategy to test
            symbol: Stock symbol

        Returns:
            BacktestResult with all metrics
        """
        # Reset state - use Decimal for precise financial calculations
        self._capital = _to_decimal(self.initial_capital)
        self._position = None
        self._trades = []
        self._equity_curve = [self.initial_capital]  # Float OK for charting
        self._dates = []
        self._pending_signal = None  # Signal from previous bar, to execute at current open
        self._rejected_trades = 0  # GHOST TRADE FIX: Track rejected trades

        # Column names (handle both lowercase and capitalized)
        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        open_col = 'open' if 'open' in data.columns else 'Open'

        print(f"\n{'='*50}")
        print(f"BACKTESTING: {strategy.name}")
        print(f"Symbol: {symbol}")
        print(f"Data: {len(data)} candles")
        print(f"Capital: Rs.{self.initial_capital:,.0f}")
        print(f"{'='*50}\n")

        # Iterate through data
        # LOOK-AHEAD BIAS FIX: Signals calculated on bar i execute at OPEN of bar i+1
        # This prevents the "Crystal Ball" bug where we trade at close using close-based signals
        for i in range(50, len(data)):  # Start after warmup period
            current_data = data.iloc[:i+1]
            current_date = data.index[i]
            current_price = data[close_col].iloc[i]
            current_high = data[high_col].iloc[i]
            current_low = data[low_col].iloc[i]
            current_open = data[open_col].iloc[i]

            self._dates.append(current_date)

            # =================================================================
            # STEP 1: Execute pending signal from PREVIOUS bar at current OPEN
            # This is the look-ahead bias fix - we trade on next bar's open
            # =================================================================
            if self._pending_signal is not None:
                pending = self._pending_signal
                self._pending_signal = None  # Clear pending

                if pending['type'] == 'ENTRY':
                    # Execute entry at current bar's OPEN (not previous bar's close)
                    if not self._position:  # Still no position
                        self._enter_trade(
                            date=current_date,
                            price=current_open,  # Execute at OPEN, not close
                            side=pending['side'],
                            symbol=symbol,
                            stop_loss=pending['stop_loss'],
                            target=pending['target'],
                            high=current_high,
                            low=current_low
                        )
                elif pending['type'] == 'EXIT':
                    # Execute exit at current bar's OPEN
                    if self._position:  # Still have position
                        self._exit_trade(current_date, current_open, pending['reason'], current_high, current_low)

            # =================================================================
            # STEP 2: Check stop loss / target for open position (intrabar)
            # These execute immediately as they're price-triggered, not signal-based
            # =================================================================
            exit_occurred_this_bar = False
            if self._position:
                exit_occurred_this_bar = self._check_exit(current_high, current_low, current_price, current_date, current_open)

            # =================================================================
            # STEP 3: Calculate signal on current bar's data (including close)
            # Signal will be EXECUTED on NEXT bar's open (stored as pending)
            #
            # WHIPSAW FIX: If we just exited this bar (stop loss or target),
            # do NOT generate a new entry signal. This prevents the bot from
            # immediately re-entering after a stop loss on a volatile candle.
            # =================================================================
            signal = strategy.analyze(current_data, symbol)

            # Store signal for execution on NEXT bar's open
            # WHIPSAW FIX: Skip entry signals if we just exited this bar
            if not self._position and not exit_occurred_this_bar:  # No position AND no exit this bar
                if signal.signal_type == SignalType.BUY:
                    self._pending_signal = {
                        'type': 'ENTRY',
                        'side': 'BUY',
                        'stop_loss': signal.stop_loss,
                        'target': signal.target
                    }
                elif signal.signal_type == SignalType.SELL:
                    self._pending_signal = {
                        'type': 'ENTRY',
                        'side': 'SELL',
                        'stop_loss': signal.stop_loss,
                        'target': signal.target
                    }
            else:  # Have position - check for exit signals
                if signal.signal_type == SignalType.EXIT:
                    self._pending_signal = {
                        'type': 'EXIT',
                        'reason': 'Signal Exit'
                    }
                elif self._position.side == "BUY" and signal.signal_type == SignalType.SELL:
                    self._pending_signal = {
                        'type': 'EXIT',
                        'reason': 'Reverse Signal'
                    }
                elif self._position.side == "SELL" and signal.signal_type == SignalType.BUY:
                    self._pending_signal = {
                        'type': 'EXIT',
                        'reason': 'Reverse Signal'
                    }

            # Update equity curve
            equity = self._calculate_equity(current_price)
            self._equity_curve.append(equity)

        # Close any open position at end
        if self._position:
            final_price = data[close_col].iloc[-1]
            final_date = data.index[-1]
            self._exit_trade(final_date, final_price, "End of Backtest")

        # Calculate results
        return self._calculate_results(strategy.name, symbol, data)

    def _enter_trade(
        self,
        date: datetime,
        price: float,
        side: str,
        symbol: str,
        stop_loss: float,
        target: float,
        high: float = None,
        low: float = None
    ):
        """Enter a new trade with realistic execution pricing."""
        # Use config for realistic execution if available
        if self._config:
            if side == "BUY":
                # Buy at ASK (above mid) + slippage
                entry_price_float = self._config.get_buy_price(price, high, low)
            else:
                # Sell at BID (below mid) - slippage
                entry_price_float = self._config.get_sell_price(price, high, low)
        else:
            # Legacy: simple slippage on close price
            if side == "BUY":
                entry_price_float = price * (1 + self.slippage_pct / 100)
            else:
                entry_price_float = price * (1 - self.slippage_pct / 100)

        # PRECISION FIX: Use Decimal for position size calculation
        entry_price = _to_decimal(entry_price_float)
        position_size_pct = _to_decimal(self.position_size_pct)
        position_value = self._capital * position_size_pct / Decimal("100")

        # Calculate quantity (round down to avoid over-allocation)
        quantity = int((position_value / entry_price).to_integral_value(rounding=ROUND_DOWN))

        if quantity < 1:
            # GHOST TRADE FIX: Log and count rejected trades instead of silent return
            self._rejected_trades += 1
            logger.warning(
                f"Trade REJECTED for {symbol}: Insufficient capital for 1 share. "
                f"Price: Rs.{_to_float(entry_price):,.2f}, "
                f"Available: Rs.{_to_float(position_value):,.2f}, "
                f"Required: Rs.{_to_float(entry_price):,.2f}"
            )
            return

        # Store as float in Trade for backward compatibility
        self._position = Trade(
            entry_date=date,
            exit_date=None,
            symbol=symbol,
            side=side,
            entry_price=_to_float(entry_price),
            quantity=quantity,
            stop_loss=stop_loss if stop_loss > 0 else (_to_float(entry_price) * 0.98 if side == "BUY" else _to_float(entry_price) * 1.02),
            target=target if target > 0 else (_to_float(entry_price) * 1.04 if side == "BUY" else _to_float(entry_price) * 0.96)
        )

        # Deduct commission using Decimal
        self._capital -= _to_decimal(self.commission)

    def _exit_trade(self, date: datetime, price: float, reason: str, high: float = None, low: float = None):
        """Exit current trade with realistic execution pricing."""
        if not self._position:
            return

        # Use config for realistic execution if available
        if self._config:
            if self._position.side == "BUY":
                # Closing a BUY = SELL at BID (below mid) - slippage
                exit_price_float = self._config.get_sell_price(price, high, low)
            else:
                # Closing a SELL = BUY at ASK (above mid) + slippage
                exit_price_float = self._config.get_buy_price(price, high, low)
        else:
            # Legacy: simple slippage on close price
            if self._position.side == "BUY":
                exit_price_float = price * (1 - self.slippage_pct / 100)
            else:
                exit_price_float = price * (1 + self.slippage_pct / 100)

        # PRECISION FIX: Use Decimal for P&L calculations
        exit_price = _to_decimal(exit_price_float)
        entry_price = _to_decimal(self._position.entry_price)
        quantity = Decimal(str(self._position.quantity))

        # Calculate P&L with Decimal precision
        if self._position.side == "BUY":
            profit_loss = (exit_price - entry_price) * quantity
        else:
            profit_loss = (entry_price - exit_price) * quantity

        # Calculate percentage
        trade_value = entry_price * quantity
        profit_loss_pct = (profit_loss / trade_value * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Update trade (convert back to float for backward compatibility)
        self._position.exit_date = date
        self._position.exit_price = _to_float(exit_price)
        self._position.profit_loss = _to_float(profit_loss)
        self._position.profit_loss_pct = _to_float(profit_loss_pct)
        self._position.exit_reason = reason

        # Update capital with Decimal precision
        commission = _to_decimal(self.commission)
        self._capital += profit_loss - commission

        # Save trade
        self._trades.append(self._position)

        # Print trade
        print(f"  {self._position.side} -> Rs.{_to_float(profit_loss):+.0f} ({_to_float(profit_loss_pct):+.1f}%) - {reason}")

        self._position = None

    def _check_exit(self, high: float, low: float, close: float, date: datetime, open_price: float = None) -> bool:
        """
        Check if stop loss or target is hit.

        Handles gap scenarios realistically:
        - If price gaps through stop loss, fill at the gap price (open), not stop loss
        - This prevents the "Magic Fill" bug where backtests unrealistically save losses

        Args:
            high: Candle high price
            low: Candle low price
            close: Candle close price
            date: Current date/time
            open_price: Candle open price (for gap detection)

        Returns:
            True if an exit occurred, False otherwise (used to prevent same-bar re-entry)
        """
        if not self._position:
            return False

        # Use close as fallback if open not provided (backward compatibility)
        if open_price is None:
            open_price = close

        if self._position.side == "BUY":
            # Check stop loss with gap handling
            if low <= self._position.stop_loss:
                # Gap down: If open is already below stop loss, we get filled at open (worse)
                if open_price < self._position.stop_loss:
                    exit_price = open_price  # Suffer the gap - realistic fill
                else:
                    exit_price = self._position.stop_loss  # Normal stop loss fill
                self._exit_trade(date, exit_price, "Stop Loss")
                return True
            # Check target with gap handling
            elif high >= self._position.target:
                # Gap up: If open is already above target, we get filled at open (better)
                if open_price > self._position.target:
                    exit_price = open_price  # Benefit from gap
                else:
                    exit_price = self._position.target  # Normal target fill
                self._exit_trade(date, exit_price, "Target Hit")
                return True
        else:  # SELL position
            # Check stop loss with gap handling (for shorts, stop is above entry)
            if high >= self._position.stop_loss:
                # Gap up: If open is already above stop loss, we get filled at open (worse)
                if open_price > self._position.stop_loss:
                    exit_price = open_price  # Suffer the gap - realistic fill
                else:
                    exit_price = self._position.stop_loss  # Normal stop loss fill
                self._exit_trade(date, exit_price, "Stop Loss")
                return True
            # Check target with gap handling (for shorts, target is below entry)
            elif low <= self._position.target:
                # Gap down: If open is already below target, we get filled at open (better)
                if open_price < self._position.target:
                    exit_price = open_price  # Benefit from gap
                else:
                    exit_price = self._position.target  # Normal target fill
                self._exit_trade(date, exit_price, "Target Hit")
                return True

        return False

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position"""
        # PRECISION FIX: Use Decimal for equity calculation
        equity = self._capital  # Already Decimal

        if self._position:
            price = _to_decimal(current_price)
            entry = _to_decimal(self._position.entry_price)
            qty = Decimal(str(self._position.quantity))

            if self._position.side == "BUY":
                unrealized = (price - entry) * qty
            else:
                unrealized = (entry - price) * qty
            equity += unrealized

        return _to_float(equity)  # Return float for charting compatibility

    def _calculate_results(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calculate final backtest results"""
        # PRECISION FIX: Convert Decimal capital to float for result
        final_capital = _to_float(self._capital)

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            trades=self._trades,
            equity_curve=self._equity_curve,
            dates=self._dates
        )

        # Trade stats
        result.total_trades = len(self._trades)
        # GHOST TRADE FIX: Include rejected trades in result
        result.rejected_trades = self._rejected_trades

        if result.total_trades > 0:
            winners = [t for t in self._trades if t.is_winner]
            losers = [t for t in self._trades if not t.is_winner]

            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = result.winning_trades / result.total_trades * 100

            # PRECISION FIX: Use Decimal for profit aggregation
            total_profit = sum(_to_decimal(t.profit_loss) for t in winners)
            total_loss = abs(sum(_to_decimal(t.profit_loss) for t in losers))

            result.total_profit = _to_float(total_profit)
            result.total_loss = _to_float(total_loss)

            result.avg_win = _to_float(total_profit / len(winners)) if winners else 0
            result.avg_loss = _to_float(total_loss / len(losers)) if losers else 0

            result.profit_factor = _to_float(total_profit / total_loss) if total_loss > 0 else 0

        # Overall stats - use Decimal for precision
        net_profit = self._capital - _to_decimal(self.initial_capital)
        result.net_profit = _to_float(net_profit)
        result.return_pct = _to_float(net_profit / _to_decimal(self.initial_capital) * Decimal("100"))

        # Drawdown
        equity = np.array(self._equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        result.max_drawdown = drawdown.max()
        result.max_drawdown_pct = (drawdown / peak).max() * 100

        # Sharpe ratio (simplified)
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / self._equity_curve[:-1]
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        return result


def print_backtest_report(result: BacktestResult):
    """Print a nice backtest report"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nStrategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")

    print(f"\n{'--- CAPITAL ---':^60}")
    print(f"Starting Capital:  Rs.{result.initial_capital:>12,.0f}")
    print(f"Final Capital:     Rs.{result.final_capital:>12,.0f}")
    print(f"Net Profit/Loss:   Rs.{result.net_profit:>+12,.0f} ({result.return_pct:+.1f}%)")

    print(f"\n{'--- TRADES ---':^60}")
    print(f"Total Trades:      {result.total_trades:>12}")
    print(f"Winning Trades:    {result.winning_trades:>12} ({result.win_rate:.1f}%)")
    print(f"Losing Trades:     {result.losing_trades:>12}")
    # GHOST TRADE FIX: Show rejected trades with warning
    if result.rejected_trades > 0:
        print(f"REJECTED Trades:   {result.rejected_trades:>12} ⚠️  (insufficient capital)")
        print(f"  → Consider increasing capital or reducing position size")

    print(f"\n{'--- PROFIT/LOSS ---':^60}")
    print(f"Total Profit:      Rs.{result.total_profit:>12,.0f}")
    print(f"Total Loss:        Rs.{result.total_loss:>12,.0f}")
    print(f"Average Win:       Rs.{result.avg_win:>12,.0f}")
    print(f"Average Loss:      Rs.{result.avg_loss:>12,.0f}")
    print(f"Profit Factor:     {result.profit_factor:>12.2f}")

    print(f"\n{'--- RISK ---':^60}")
    print(f"Max Drawdown:      Rs.{result.max_drawdown:>12,.0f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:>12.2f}")

    # Grade the result
    print(f"\n{'--- VERDICT ---':^60}")
    if result.return_pct > 20 and result.win_rate > 50:
        print("EXCELLENT! This strategy looks promising.")
    elif result.return_pct > 0 and result.win_rate > 40:
        print("GOOD. Strategy is profitable but has room for improvement.")
    elif result.return_pct > 0:
        print("OK. Strategy makes money but win rate is low.")
    else:
        print("NEEDS WORK. Strategy lost money. Try different parameters.")

    print("\n" + "=" * 60)


# ============== QUICK BACKTEST ==============

def quick_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    capital: float = 100000,
    symbol: str = "STOCK"
) -> BacktestResult:
    """
    Quick way to run a backtest.

    Example:
        >>> from strategies import get_strategy
        >>> result = quick_backtest(data, get_strategy('turtle'))
    """
    bt = Backtester(initial_capital=capital)
    result = bt.run(data, strategy, symbol)
    print_backtest_report(result)
    return result


# =============================================================================
# EVENT-DRIVEN BACKTESTER
# =============================================================================

class EventDrivenBacktester:
    """
    Event-Driven Backtester - The Unified Engine!

    Unlike the legacy Backtester, this one uses the SAME event-driven
    architecture as live trading. The engine processes events from
    HistoricalDataSource and calls strategy.on_event() - exactly the
    same as live trading would do.

    Benefits:
    - Same code path for backtest and live
    - Easy to test exact same strategy logic
    - Event replay for debugging
    - Multi-symbol support

    Reality Gap Fix (2024):
        Pass a BacktestConfig for realistic execution modeling:
        - BID_ASK fill model: Buy at ask, sell at bid
        - OHLC fill model: Worst-case fills using high/low
        - Slippage modeling: Market impact cost

    Example:
        >>> from core.data import HistoricalDataSource, DataSourceConfig
        >>> from core.events import EventBus
        >>>
        >>> bus = EventBus()
        >>> config = DataSourceConfig(symbols=['RELIANCE'], timeframe='1d')
        >>> backtester = EventDrivenBacktester(bus, capital=100000)
        >>> result = backtester.run(data={'RELIANCE': df}, strategy=strategy)
    """

    def __init__(
        self,
        event_bus=None,
        initial_capital: float = 100000,
        position_size_pct: float = 10.0,
        commission: float = 0.0,
        slippage_pct: float = 0.1,
        warmup_bars: int = 50,
        config: 'BacktestConfig' = None
    ):
        """
        Initialize the event-driven backtester.

        Args:
            event_bus: EventBus instance (created if not provided)
            initial_capital: Starting capital
            position_size_pct: % of capital per trade
            commission: Commission per trade
            slippage_pct: Slippage percentage
            warmup_bars: Bars to skip before trading
            config: BacktestConfig for realistic execution (overrides other params)
        """
        # Lazy import to avoid circular dependencies
        from core.events import EventBus, EventType

        self.event_bus = event_bus or EventBus()

        # Store config for realistic execution pricing
        self._config = config

        # Use config values if provided, otherwise use legacy params
        if config:
            self.initial_capital = config.initial_capital
            self.position_size_pct = config.position_size_pct
            self.commission = 0  # Included in config's total_cost_pct
            self.slippage_pct = config.slippage_pct
            self.warmup_bars = config.warmup_bars
        else:
            self.initial_capital = initial_capital
            self.position_size_pct = position_size_pct
            self.commission = commission
            self.slippage_pct = slippage_pct
            self.warmup_bars = warmup_bars

        # State - use Decimal for precise financial calculations
        # This prevents floating-point drift over thousands of trades
        self._capital: Decimal = _to_decimal(initial_capital)
        self._positions: Dict[str, Trade] = {}  # symbol -> Trade
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []  # Float OK for charting
        self._dates: List[datetime] = []
        self._current_prices: Dict[str, float] = {}

        # WHIPSAW FIX: Track symbols that exited this bar to prevent immediate re-entry
        self._exits_this_bar: set = set()

        # GHOST TRADE FIX: Track rejected trades due to insufficient capital
        self._rejected_trades: int = 0

        # Strategy
        self._strategy: Optional[Strategy] = None

        # Event subscriptions - store handler names for unsubscribe
        self._handler_names: List[str] = []

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy: Strategy,
        symbol: str = "STOCK",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest using event-driven architecture.

        Args:
            data: DataFrame or dict of symbol->DataFrame
            strategy: Trading strategy
            symbol: Stock symbol (if data is single DataFrame)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with all metrics
        """
        from core.data import HistoricalDataSource, DataSourceConfig
        from core.events import EventType, BarEvent, SignalEvent

        # Reset state
        self._reset()
        self._strategy = strategy
        strategy.set_event_bus(self.event_bus)

        # Normalize data to dict format
        if isinstance(data, pd.DataFrame):
            data_dict = {symbol: data}
            symbols = [symbol]
        else:
            data_dict = data
            symbols = list(data.keys())

        # Derive date range from data if not specified
        if start_date is None or end_date is None:
            all_dates = []
            for df in data_dict.values():
                if not df.empty:
                    all_dates.extend(df.index.tolist())
            if all_dates:
                data_start = min(all_dates)
                data_end = max(all_dates)
                if start_date is None:
                    start_date = data_start
                if end_date is None:
                    end_date = data_end

        # Configure data source
        config = DataSourceConfig(
            symbols=symbols,
            timeframe='1d',  # Will be overridden by data
            start_date=start_date,
            end_date=end_date,
            warmup_bars=self.warmup_bars
        )

        # Create historical data source
        source = HistoricalDataSource(
            config=config,
            event_bus=None,  # We'll iterate manually
            data=data_dict
        )

        # Connect and load data
        if not source.connect():
            logger.error("Failed to connect to data source")
            return self._create_empty_result(strategy.name, symbols[0])

        print(f"\n{'='*50}")
        print(f"EVENT-DRIVEN BACKTEST: {strategy.name}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Capital: Rs.{self.initial_capital:,.0f}")
        print(f"{'='*50}\n")

        # Subscribe to events
        self._subscribe_events()

        # Start the data source (sets _running flag)
        source._running = True

        # Process events from data source
        # LOOK-AHEAD BIAS FIX: Signals from bar i execute at OPEN of bar i+1
        bar_count = 0
        for event in source._emit_events():
            if isinstance(event, BarEvent):
                bar_count += 1
                symbol = event.symbol

                # Update current price
                self._current_prices[symbol] = event.close
                self._dates.append(event.timestamp)

                # =============================================================
                # STEP 1: Execute pending signal from PREVIOUS bar at current OPEN
                # This is the look-ahead bias fix - we trade on next bar's open
                # =============================================================
                if symbol in self._pending_signals:
                    pending = self._pending_signals.pop(symbol)
                    open_price = getattr(event, 'open', event.close)

                    if pending['type'] == 'ENTRY':
                        if symbol not in self._positions:  # Still no position
                            self._enter_position(
                                symbol=symbol,
                                side=pending['side'],
                                price=open_price,  # Execute at OPEN, not close
                                date=event.timestamp,
                                stop_loss=pending['stop_loss'],
                                target=pending['target'],
                                high=event.high,
                                low=event.low
                            )
                    elif pending['type'] == 'EXIT':
                        if symbol in self._positions:  # Still have position
                            self._close_position(symbol, open_price, event.timestamp,
                                                pending['reason'], event.high, event.low)

                # =============================================================
                # STEP 2: Check stop-loss/target for open positions (intrabar)
                # These execute immediately as they're price-triggered
                # =============================================================
                exit_occurred = self._check_exits(event)

                # WHIPSAW FIX: Track if exit occurred this bar to prevent re-entry
                if exit_occurred:
                    self._exits_this_bar.add(symbol)
                else:
                    self._exits_this_bar.discard(symbol)

                # =============================================================
                # STEP 3: Publish bar to bus (triggers strategy.on_event)
                # New signals will be STORED as pending, not executed immediately
                # WHIPSAW FIX: _store_pending_signal checks _exits_this_bar
                # =============================================================
                self.event_bus.publish(event)

                # Update equity curve
                equity = self._calculate_equity()
                self._equity_curve.append(equity)

        # Close any open positions at end
        for symbol, position in list(self._positions.items()):
            if position.is_open:
                self._close_position(
                    symbol,
                    self._current_prices.get(symbol, position.entry_price),
                    self._dates[-1] if self._dates else datetime.now(),
                    "End of Backtest"
                )

        # Unsubscribe
        self._unsubscribe_events()

        print(f"\nProcessed {bar_count} bars")

        # Calculate results
        return self._calculate_results(strategy.name, symbols[0], data_dict)

    def _reset(self):
        """Reset backtester state."""
        # PRECISION FIX: Use Decimal for precise financial calculations
        self._capital = _to_decimal(self.initial_capital)
        self._positions.clear()
        self._trades.clear()
        self._equity_curve = [self.initial_capital]  # Float OK for charting
        self._dates.clear()
        self._current_prices.clear()
        # Pending signals from previous bar, keyed by symbol
        # LOOK-AHEAD BIAS FIX: Signals execute on NEXT bar's open
        self._pending_signals: Dict[str, dict] = {}
        # WHIPSAW FIX: Track symbols that exited this bar to prevent immediate re-entry
        self._exits_this_bar: set = set()
        # GHOST TRADE FIX: Track rejected trades due to insufficient capital
        self._rejected_trades = 0

    def _subscribe_events(self):
        """Subscribe to relevant events."""
        from core.events import EventType

        # Store handler names for later unsubscription
        self._handler_names = []
        name1 = self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED, self._on_signal,
            priority=100, name="bt_signal_handler"
        )
        name2 = self.event_bus.subscribe(
            EventType.BAR, self._on_bar_for_strategy,
            priority=50, name="bt_bar_handler"
        )
        self._handler_names.extend([name1, name2])

    def _unsubscribe_events(self):
        """Unsubscribe from events."""
        for name in self._handler_names:
            self.event_bus.unsubscribe(name)
        self._handler_names.clear()

    def _on_bar_for_strategy(self, event):
        """
        Forward bar to strategy for processing.

        LOOK-AHEAD BIAS FIX: Signals are stored as pending and executed
        on the NEXT bar's open, not the current bar's close.
        """
        if self._strategy:
            signal = self._strategy.on_event(event)
            if signal and signal.signal_type != SignalType.HOLD:
                # Store signal for execution on NEXT bar's open
                self._store_pending_signal(signal, event)

    def _store_pending_signal(self, signal: Signal, bar_event):
        """
        Store a signal for execution on the NEXT bar's open.

        LOOK-AHEAD BIAS FIX: Instead of executing immediately at this bar's close,
        we store the signal and execute it at the next bar's open price.

        WHIPSAW FIX: If an exit occurred this bar for this symbol, we do NOT
        store entry signals. This prevents immediate re-entry after stop loss.
        """
        symbol = signal.symbol

        # Check current position state to determine signal type
        if symbol not in self._positions:
            # WHIPSAW FIX: If we just exited this bar, do NOT generate entry signal
            # This prevents the bot from re-entering immediately after a stop loss
            if symbol in self._exits_this_bar:
                return  # Skip entry signal - exit just occurred

            # No position - this is an entry signal
            if signal.signal_type == SignalType.BUY:
                self._pending_signals[symbol] = {
                    'type': 'ENTRY',
                    'side': 'BUY',
                    'stop_loss': signal.stop_loss,
                    'target': signal.target
                }
            elif signal.signal_type == SignalType.SELL:
                self._pending_signals[symbol] = {
                    'type': 'ENTRY',
                    'side': 'SELL',
                    'stop_loss': signal.stop_loss,
                    'target': signal.target
                }
        else:
            # Have position - check for exit signals
            position = self._positions[symbol]
            if signal.signal_type == SignalType.EXIT:
                self._pending_signals[symbol] = {
                    'type': 'EXIT',
                    'reason': 'Signal Exit'
                }
            elif position.side == "BUY" and signal.signal_type == SignalType.SELL:
                self._pending_signals[symbol] = {
                    'type': 'EXIT',
                    'reason': 'Reverse Signal'
                }
            elif position.side == "SELL" and signal.signal_type == SignalType.BUY:
                self._pending_signals[symbol] = {
                    'type': 'EXIT',
                    'reason': 'Reverse Signal'
                }

    def _on_signal(self, event):
        """
        Handle signal events from strategy.

        LOOK-AHEAD BIAS FIX: Signals are stored as pending and executed
        on the NEXT bar's open, not the current bar's close.
        """
        from core.events.events import SignalType as EventSignalType

        # Convert SignalEvent to Signal
        signal = Signal(
            signal_type=self._convert_signal_type(event.signal_type),
            symbol=event.symbol,
            price=event.price,
            stop_loss=event.stop_loss,
            target=event.target,
            confidence=event.confidence,
            reason=event.reason,
            timestamp=event.timestamp
        )

        # Store signal for execution on NEXT bar's open (same as _store_pending_signal)
        symbol = signal.symbol
        if symbol not in self._positions:
            if signal.signal_type == SignalType.BUY:
                self._pending_signals[symbol] = {
                    'type': 'ENTRY',
                    'side': 'BUY',
                    'stop_loss': signal.stop_loss,
                    'target': signal.target
                }
            elif signal.signal_type == SignalType.SELL:
                self._pending_signals[symbol] = {
                    'type': 'ENTRY',
                    'side': 'SELL',
                    'stop_loss': signal.stop_loss,
                    'target': signal.target
                }
        else:
            position = self._positions[symbol]
            if signal.signal_type == SignalType.EXIT:
                self._pending_signals[symbol] = {'type': 'EXIT', 'reason': 'Signal Exit'}
            elif position.side == "BUY" and signal.signal_type == SignalType.SELL:
                self._pending_signals[symbol] = {'type': 'EXIT', 'reason': 'Reverse Signal'}
            elif position.side == "SELL" and signal.signal_type == SignalType.BUY:
                self._pending_signals[symbol] = {'type': 'EXIT', 'reason': 'Reverse Signal'}

    def _convert_signal_type(self, event_signal_type) -> SignalType:
        """Convert event SignalType to strategy SignalType."""
        from core.events.events import SignalType as EventSignalType

        mapping = {
            EventSignalType.BUY: SignalType.BUY,
            EventSignalType.SELL: SignalType.SELL,
            EventSignalType.HOLD: SignalType.HOLD,
            EventSignalType.EXIT: SignalType.EXIT,
        }
        return mapping.get(event_signal_type, SignalType.HOLD)

    def _process_signal(self, signal: Signal, bar_event):
        """Process a signal from strategy.on_event()."""
        symbol = signal.symbol
        current_price = bar_event.close
        current_high = bar_event.high
        current_low = bar_event.low
        current_date = bar_event.timestamp

        # No position - can enter
        if symbol not in self._positions:
            if signal.signal_type == SignalType.BUY:
                self._enter_position(
                    symbol=symbol,
                    side="BUY",
                    price=current_price,
                    date=current_date,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    high=current_high,
                    low=current_low
                )
            elif signal.signal_type == SignalType.SELL:
                self._enter_position(
                    symbol=symbol,
                    side="SELL",
                    price=current_price,
                    date=current_date,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    high=current_high,
                    low=current_low
                )
        else:
            # Have position - check for exit or reverse
            position = self._positions[symbol]
            if signal.signal_type == SignalType.EXIT:
                self._close_position(symbol, current_price, current_date, "Signal Exit", current_high, current_low)
            elif position.side == "BUY" and signal.signal_type == SignalType.SELL:
                self._close_position(symbol, current_price, current_date, "Reverse Signal", current_high, current_low)
            elif position.side == "SELL" and signal.signal_type == SignalType.BUY:
                self._close_position(symbol, current_price, current_date, "Reverse Signal", current_high, current_low)

    def _process_signal_direct(self, signal: Signal, current_price: float):
        """Process signal directly (from SignalEvent)."""
        symbol = signal.symbol
        current_date = signal.timestamp or datetime.now()

        if symbol not in self._positions:
            if signal.signal_type == SignalType.BUY:
                self._enter_position(symbol, "BUY", current_price, current_date,
                                    signal.stop_loss, signal.target)
            elif signal.signal_type == SignalType.SELL:
                self._enter_position(symbol, "SELL", current_price, current_date,
                                    signal.stop_loss, signal.target)
        else:
            position = self._positions[symbol]
            if signal.signal_type == SignalType.EXIT:
                self._close_position(symbol, current_price, current_date, "Signal Exit")
            elif position.side == "BUY" and signal.signal_type == SignalType.SELL:
                self._close_position(symbol, current_price, current_date, "Reverse Signal")
            elif position.side == "SELL" and signal.signal_type == SignalType.BUY:
                self._close_position(symbol, current_price, current_date, "Reverse Signal")

    def _enter_position(
        self,
        symbol: str,
        side: str,
        price: float,
        date: datetime,
        stop_loss: float = 0,
        target: float = 0,
        high: float = None,
        low: float = None
    ):
        """Enter a new position and emit events with realistic execution pricing."""
        from core.events import OrderEvent, FillEvent, PositionEvent
        from core.events.events import Side, OrderStatus, EventType

        # Use config for realistic execution if available
        if self._config:
            if side == "BUY":
                # Buy at ASK (above mid) + slippage
                entry_price_float = self._config.get_buy_price(price, high, low)
            else:
                # Sell at BID (below mid) - slippage
                entry_price_float = self._config.get_sell_price(price, high, low)
        else:
            # Legacy: simple slippage on close price
            if side == "BUY":
                entry_price_float = price * (1 + self.slippage_pct / 100)
            else:
                entry_price_float = price * (1 - self.slippage_pct / 100)

        # PRECISION FIX: Use Decimal for position size calculation
        entry_price = _to_decimal(entry_price_float)
        position_size_pct = _to_decimal(self.position_size_pct)
        position_value = self._capital * position_size_pct / Decimal("100")

        # Calculate quantity (round down to avoid over-allocation)
        quantity = int((position_value / entry_price).to_integral_value(rounding=ROUND_DOWN))

        if quantity < 1:
            # GHOST TRADE FIX: Log and count rejected trades instead of silent return
            self._rejected_trades += 1
            logger.warning(
                f"Trade REJECTED for {symbol}: Insufficient capital for 1 share. "
                f"Price: Rs.{_to_float(entry_price):,.2f}, "
                f"Available: Rs.{_to_float(position_value):,.2f}, "
                f"Required: Rs.{_to_float(entry_price):,.2f}"
            )
            return

        # Default stop-loss and target (use float for Trade compatibility)
        entry_price_f = _to_float(entry_price)
        if stop_loss <= 0:
            stop_loss = entry_price_f * (0.98 if side == "BUY" else 1.02)
        if target <= 0:
            target = entry_price_f * (1.04 if side == "BUY" else 0.96)

        # Create trade record (store as float for backward compatibility)
        trade = Trade(
            entry_date=date,
            exit_date=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price_f,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target
        )

        self._positions[symbol] = trade

        # Deduct commission using Decimal
        self._capital -= _to_decimal(self.commission)

        # Emit order submitted event
        order_event = OrderEvent(
            order_id=f"BT-{len(self._trades)}-{symbol}",
            symbol=symbol,
            side=Side.BUY if side == "BUY" else Side.SELL,
            quantity=quantity,
            order_type="MARKET",
            price=entry_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            average_price=entry_price,
            strategy_name=self._strategy.name if self._strategy else "unknown"
        )
        self.event_bus.publish(order_event)

        # Emit fill event
        fill_event = FillEvent(
            order_id=order_event.order_id,
            symbol=symbol,
            side=Side.BUY if side == "BUY" else Side.SELL,
            quantity=quantity,
            price=entry_price,
            commission=self.commission,
            strategy_name=self._strategy.name if self._strategy else "unknown",
            timestamp=date
        )
        self.event_bus.publish(fill_event)

        # Emit position opened event
        position_event = PositionEvent(
            symbol=symbol,
            side=Side.BUY if side == "BUY" else Side.SELL,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            strategy_name=self._strategy.name if self._strategy else "unknown",
            timestamp=date
        )
        position_event.event_type = EventType.POSITION_OPENED
        self.event_bus.publish(position_event)

        logger.debug(f"Entered {side} position: {symbol} @ {entry_price:.2f}")

    def _close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: str,
        high: float = None,
        low: float = None
    ):
        """Close a position and emit events with realistic execution pricing."""
        from core.events import OrderEvent, FillEvent, PositionEvent
        from core.events.events import Side, OrderStatus, EventType

        if symbol not in self._positions:
            return

        position = self._positions[symbol]

        # Use config for realistic execution if available
        if self._config:
            if position.side == "BUY":
                # Closing a BUY = SELL at BID (below mid) - slippage
                exit_price_float = self._config.get_sell_price(price, high, low)
            else:
                # Closing a SELL = BUY at ASK (above mid) + slippage
                exit_price_float = self._config.get_buy_price(price, high, low)
        else:
            # Legacy: simple slippage on close price
            if position.side == "BUY":
                exit_price_float = price * (1 - self.slippage_pct / 100)
            else:
                exit_price_float = price * (1 + self.slippage_pct / 100)

        # PRECISION FIX: Use Decimal for P&L calculations
        exit_price = _to_decimal(exit_price_float)
        entry_price = _to_decimal(position.entry_price)
        quantity = Decimal(str(position.quantity))

        # Calculate P&L with Decimal precision
        if position.side == "BUY":
            profit_loss = (exit_price - entry_price) * quantity
        else:
            profit_loss = (entry_price - exit_price) * quantity

        # Calculate percentage
        trade_value = entry_price * quantity
        profit_loss_pct = (profit_loss / trade_value * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Update trade record (convert back to float for backward compatibility)
        position.exit_date = date
        position.exit_price = _to_float(exit_price)
        position.profit_loss = _to_float(profit_loss)
        position.profit_loss_pct = _to_float(profit_loss_pct)
        position.exit_reason = reason

        # Update capital with Decimal precision
        commission = _to_decimal(self.commission)
        self._capital += profit_loss - commission

        # Save completed trade
        self._trades.append(position)

        # Remove from positions
        del self._positions[symbol]

        # Print trade result
        print(f"  {position.side} {symbol} -> Rs.{_to_float(profit_loss):+.0f} ({_to_float(profit_loss_pct):+.1f}%) - {reason}")

        # Emit close order
        exit_side = Side.SELL if position.side == "BUY" else Side.BUY
        order_event = OrderEvent(
            order_id=f"BT-EXIT-{len(self._trades)}-{symbol}",
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            order_type="MARKET",
            price=exit_price,
            status=OrderStatus.FILLED,
            filled_quantity=position.quantity,
            average_price=exit_price,
            strategy_name=self._strategy.name if self._strategy else "unknown"
        )
        self.event_bus.publish(order_event)

        # Emit fill event
        fill_event = FillEvent(
            order_id=order_event.order_id,
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            price=exit_price,
            commission=self.commission,
            strategy_name=self._strategy.name if self._strategy else "unknown",
            timestamp=date
        )
        self.event_bus.publish(fill_event)

        # Emit position closed event
        position_event = PositionEvent(
            symbol=symbol,
            side=Side.BUY if position.side == "BUY" else Side.SELL,
            quantity=0,
            entry_price=position.entry_price,
            current_price=exit_price,
            unrealized_pnl=0,
            realized_pnl=profit_loss,
            strategy_name=self._strategy.name if self._strategy else "unknown",
            timestamp=date
        )
        position_event.event_type = EventType.POSITION_CLOSED
        self.event_bus.publish(position_event)

        logger.debug(f"Closed {position.side} position: {symbol} @ {exit_price:.2f}, P&L: {profit_loss:.2f}")

    def _check_exits(self, bar_event) -> bool:
        """
        Check stop-loss and target for open positions.

        Handles gap scenarios realistically:
        - If price gaps through stop loss, fill at the gap price (open), not stop loss
        - This prevents the "Magic Fill" bug where backtests unrealistically save losses

        Returns:
            True if an exit occurred, False otherwise (used to prevent same-bar re-entry)
        """
        symbol = bar_event.symbol
        if symbol not in self._positions:
            return False

        position = self._positions[symbol]
        high = bar_event.high
        low = bar_event.low
        close = bar_event.close
        open_price = getattr(bar_event, 'open', close)  # Fallback to close if no open
        date = bar_event.timestamp

        if position.side == "BUY":
            # Check stop loss with gap handling
            if low <= position.stop_loss:
                # Gap down: If open is already below stop loss, we get filled at open (worse)
                if open_price < position.stop_loss:
                    exit_price = open_price  # Suffer the gap - realistic fill
                else:
                    exit_price = position.stop_loss  # Normal stop loss fill
                self._close_position(symbol, exit_price, date, "Stop Loss", high, low)
                return True
            # Check target with gap handling
            elif high >= position.target:
                # Gap up: If open is already above target, we get filled at open (better)
                if open_price > position.target:
                    exit_price = open_price  # Benefit from gap
                else:
                    exit_price = position.target  # Normal target fill
                self._close_position(symbol, exit_price, date, "Target Hit", high, low)
                return True
        else:  # SELL position
            # Check stop loss with gap handling (for shorts, stop is above entry)
            if high >= position.stop_loss:
                # Gap up: If open is already above stop loss, we get filled at open (worse)
                if open_price > position.stop_loss:
                    exit_price = open_price  # Suffer the gap - realistic fill
                else:
                    exit_price = position.stop_loss  # Normal stop loss fill
                self._close_position(symbol, exit_price, date, "Stop Loss", high, low)
                return True
            # Check target with gap handling (for shorts, target is below entry)
            elif low <= position.target:
                # Gap down: If open is already below target, we get filled at open (better)
                if open_price < position.target:
                    exit_price = open_price  # Benefit from gap
                else:
                    exit_price = position.target  # Normal target fill
                self._close_position(symbol, exit_price, date, "Target Hit", high, low)
                return True

        return False
                self._close_position(symbol, exit_price, date, "Target Hit", high, low)

    def _calculate_equity(self) -> float:
        """Calculate current equity including open positions."""
        # PRECISION FIX: Use Decimal for equity calculation
        equity = self._capital  # Already Decimal

        for symbol, position in self._positions.items():
            current_price = _to_decimal(self._current_prices.get(symbol, position.entry_price))
            entry = _to_decimal(position.entry_price)
            qty = Decimal(str(position.quantity))

            if position.side == "BUY":
                unrealized = (current_price - entry) * qty
            else:
                unrealized = (entry - current_price) * qty
            equity += unrealized

        return _to_float(equity)  # Return float for charting compatibility

    def _calculate_results(
        self,
        strategy_name: str,
        symbol: str,
        data: Dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """Calculate final backtest results."""
        # Get date range from data
        all_dates = []
        for df in data.values():
            if not df.empty:
                all_dates.extend(df.index.tolist())

        if all_dates:
            start_date = min(all_dates)
            end_date = max(all_dates)
        else:
            start_date = end_date = datetime.now()

        # PRECISION FIX: Convert Decimal capital to float for result
        final_capital = _to_float(self._capital)

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            trades=self._trades,
            equity_curve=self._equity_curve,
            dates=self._dates
        )

        # Trade stats
        result.total_trades = len(self._trades)
        # GHOST TRADE FIX: Include rejected trades in result
        result.rejected_trades = self._rejected_trades

        if result.total_trades > 0:
            winners = [t for t in self._trades if t.is_winner]
            losers = [t for t in self._trades if not t.is_winner]

            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = result.winning_trades / result.total_trades * 100

            # PRECISION FIX: Use Decimal for profit aggregation
            total_profit = sum(_to_decimal(t.profit_loss) for t in winners)
            total_loss = abs(sum(_to_decimal(t.profit_loss) for t in losers))

            result.total_profit = _to_float(total_profit)
            result.total_loss = _to_float(total_loss)

            result.avg_win = _to_float(total_profit / len(winners)) if winners else 0
            result.avg_loss = _to_float(total_loss / len(losers)) if losers else 0

            result.profit_factor = _to_float(total_profit / total_loss) if total_loss > 0 else 0

        # Overall stats - use Decimal for precision
        net_profit = self._capital - _to_decimal(self.initial_capital)
        result.net_profit = _to_float(net_profit)
        result.return_pct = _to_float(net_profit / _to_decimal(self.initial_capital) * Decimal("100"))

        # Drawdown
        if self._equity_curve:
            equity = np.array(self._equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = peak - equity
            result.max_drawdown = drawdown.max()
            result.max_drawdown_pct = (drawdown / peak).max() * 100 if peak.max() > 0 else 0

            # Sharpe ratio
            if len(self._equity_curve) > 1:
                returns = np.diff(self._equity_curve) / np.array(self._equity_curve[:-1])
                if returns.std() > 0:
                    result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        return result

    def _create_empty_result(self, strategy_name: str, symbol: str) -> BacktestResult:
        """Create empty result for failed backtest."""
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital
        )


def event_driven_backtest(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    strategy: Strategy,
    capital: float = 100000,
    symbol: str = "STOCK",
    show_report: bool = True
) -> BacktestResult:
    """
    Quick way to run an event-driven backtest.

    Example:
        >>> from strategies import get_strategy
        >>> result = event_driven_backtest(data, get_strategy('turtle'))
    """
    from core.events import EventBus

    bus = EventBus()
    backtester = EventDrivenBacktester(bus, initial_capital=capital)
    result = backtester.run(data, strategy, symbol)

    if show_report:
        print_backtest_report(result)

    return result


# Test
if __name__ == "__main__":
    print("Backtesting Engine ready!")
    print("Usage: quick_backtest(data, strategy)")
    print("       event_driven_backtest(data, strategy)")
