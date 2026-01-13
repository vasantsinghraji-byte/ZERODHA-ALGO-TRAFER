# -*- coding: utf-8 -*-
"""
Backtesting Engine - Time Machine for Trading!
==============================================
Test your strategies on old data to see how they would have performed.

It's like a practice game before the real match!
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from strategies.base import Strategy, Signal, SignalType


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
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 10.0,
        commission: float = 0.0,
        slippage_pct: float = 0.1
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting money (default Rs.1,00,000)
            position_size_pct: % of capital per trade (default 10%)
            commission: Commission per trade (default 0)
            slippage_pct: Slippage/impact cost (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage_pct = slippage_pct

        # State
        self._capital = initial_capital
        self._position: Optional[Trade] = None
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []
        self._dates: List[datetime] = []

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
        # Reset state
        self._capital = self.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = [self.initial_capital]
        self._dates = []

        # Column names
        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'

        print(f"\n{'='*50}")
        print(f"BACKTESTING: {strategy.name}")
        print(f"Symbol: {symbol}")
        print(f"Data: {len(data)} candles")
        print(f"Capital: Rs.{self.initial_capital:,.0f}")
        print(f"{'='*50}\n")

        # Iterate through data
        for i in range(50, len(data)):  # Start after warmup period
            current_data = data.iloc[:i+1]
            current_date = data.index[i]
            current_price = data[close_col].iloc[i]
            current_high = data[high_col].iloc[i]
            current_low = data[low_col].iloc[i]

            self._dates.append(current_date)

            # Check stop loss / target for open position
            if self._position:
                self._check_exit(current_high, current_low, current_price, current_date)

            # Get signal from strategy
            signal = strategy.analyze(current_data, symbol)

            # Process signal
            if not self._position:  # No position - can enter
                if signal.signal_type == SignalType.BUY:
                    self._enter_trade(
                        date=current_date,
                        price=current_price,
                        side="BUY",
                        symbol=symbol,
                        stop_loss=signal.stop_loss,
                        target=signal.target
                    )
                elif signal.signal_type == SignalType.SELL:
                    self._enter_trade(
                        date=current_date,
                        price=current_price,
                        side="SELL",
                        symbol=symbol,
                        stop_loss=signal.stop_loss,
                        target=signal.target
                    )
            else:  # Have position - check for exit
                if signal.signal_type == SignalType.EXIT:
                    self._exit_trade(current_date, current_price, "Signal Exit")
                elif self._position.side == "BUY" and signal.signal_type == SignalType.SELL:
                    self._exit_trade(current_date, current_price, "Reverse Signal")
                elif self._position.side == "SELL" and signal.signal_type == SignalType.BUY:
                    self._exit_trade(current_date, current_price, "Reverse Signal")

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
        target: float
    ):
        """Enter a new trade"""
        # Apply slippage
        if side == "BUY":
            entry_price = price * (1 + self.slippage_pct / 100)
        else:
            entry_price = price * (1 - self.slippage_pct / 100)

        # Calculate position size
        position_value = self._capital * self.position_size_pct / 100
        quantity = int(position_value / entry_price)

        if quantity < 1:
            return  # Not enough capital

        self._position = Trade(
            entry_date=date,
            exit_date=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss if stop_loss > 0 else (entry_price * 0.98 if side == "BUY" else entry_price * 1.02),
            target=target if target > 0 else (entry_price * 1.04 if side == "BUY" else entry_price * 0.96)
        )

        # Deduct commission
        self._capital -= self.commission

    def _exit_trade(self, date: datetime, price: float, reason: str):
        """Exit current trade"""
        if not self._position:
            return

        # Apply slippage
        if self._position.side == "BUY":
            exit_price = price * (1 - self.slippage_pct / 100)
        else:
            exit_price = price * (1 + self.slippage_pct / 100)

        # Calculate P&L
        if self._position.side == "BUY":
            profit_loss = (exit_price - self._position.entry_price) * self._position.quantity
        else:
            profit_loss = (self._position.entry_price - exit_price) * self._position.quantity

        profit_loss_pct = profit_loss / (self._position.entry_price * self._position.quantity) * 100

        # Update trade
        self._position.exit_date = date
        self._position.exit_price = exit_price
        self._position.profit_loss = profit_loss
        self._position.profit_loss_pct = profit_loss_pct
        self._position.exit_reason = reason

        # Update capital
        self._capital += profit_loss - self.commission

        # Save trade
        self._trades.append(self._position)

        # Print trade
        emoji = "+" if profit_loss > 0 else ""
        print(f"  {self._position.side} -> Rs.{profit_loss:+.0f} ({profit_loss_pct:+.1f}%) - {reason}")

        self._position = None

    def _check_exit(self, high: float, low: float, close: float, date: datetime):
        """Check if stop loss or target is hit"""
        if not self._position:
            return

        if self._position.side == "BUY":
            if low <= self._position.stop_loss:
                self._exit_trade(date, self._position.stop_loss, "Stop Loss")
            elif high >= self._position.target:
                self._exit_trade(date, self._position.target, "Target Hit")
        else:  # SELL position
            if high >= self._position.stop_loss:
                self._exit_trade(date, self._position.stop_loss, "Stop Loss")
            elif low <= self._position.target:
                self._exit_trade(date, self._position.target, "Target Hit")

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position"""
        equity = self._capital

        if self._position:
            if self._position.side == "BUY":
                unrealized = (current_price - self._position.entry_price) * self._position.quantity
            else:
                unrealized = (self._position.entry_price - current_price) * self._position.quantity
            equity += unrealized

        return equity

    def _calculate_results(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calculate final backtest results"""

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self._capital,
            trades=self._trades,
            equity_curve=self._equity_curve,
            dates=self._dates
        )

        # Trade stats
        result.total_trades = len(self._trades)

        if result.total_trades > 0:
            winners = [t for t in self._trades if t.is_winner]
            losers = [t for t in self._trades if not t.is_winner]

            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = result.winning_trades / result.total_trades * 100

            result.total_profit = sum(t.profit_loss for t in winners)
            result.total_loss = abs(sum(t.profit_loss for t in losers))

            result.avg_win = result.total_profit / len(winners) if winners else 0
            result.avg_loss = result.total_loss / len(losers) if losers else 0

            result.profit_factor = result.total_profit / result.total_loss if result.total_loss > 0 else 0

        # Overall stats
        result.net_profit = self._capital - self.initial_capital
        result.return_pct = result.net_profit / self.initial_capital * 100

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


# Test
if __name__ == "__main__":
    print("Backtesting Engine ready!")
    print("Usage: quick_backtest(data, strategy)")
