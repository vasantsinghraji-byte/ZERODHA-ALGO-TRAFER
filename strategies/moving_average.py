"""
Moving Average Crossover Strategy
The TURTLE strategy - slow and steady! 🐢

How it works:
- Uses two moving averages (fast and slow)
- When fast crosses ABOVE slow = BUY signal
- When fast crosses BELOW slow = SELL signal

Think of it like two runners:
- Fast runner (short-term average)
- Slow runner (long-term average)
When the fast runner overtakes the slow runner, prices are going up!
"""

import pandas as pd
from .base import Strategy, Signal, SignalType, RiskLevel


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy

    Perfect for beginners! Very safe and easy to understand.

    Parameters:
        fast_period: Days for fast average (default: 9)
        slow_period: Days for slow average (default: 21)
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        return "Moving Average Crossover"

    @property
    def description(self) -> str:
        return "Buy when short-term average crosses above long-term average. Safe for beginners!"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def emoji(self) -> str:
        return "🐢"

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze price data and generate signal.

        Args:
            data: DataFrame with at least 'close' column
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        if len(data) < self.slow_period + 1:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=data['close'].iloc[-1] if len(data) > 0 else 0,
                reason="Not enough data"
            )

        # Calculate moving averages
        data = data.copy()
        data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()

        # Get current and previous values
        current_fast = data['fast_ma'].iloc[-1]
        current_slow = data['slow_ma'].iloc[-1]
        prev_fast = data['fast_ma'].iloc[-2]
        prev_slow = data['slow_ma'].iloc[-2]
        current_price = data['close'].iloc[-1]

        # Check for crossover
        # BUY: Fast crosses ABOVE slow
        if prev_fast <= prev_slow and current_fast > current_slow:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=True),
                target=self.calculate_target(current_price, is_buy=True),
                confidence=0.7,
                reason=f"Fast MA ({self.fast_period}) crossed ABOVE slow MA ({self.slow_period})"
            )

        # SELL: Fast crosses BELOW slow
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=False),
                target=self.calculate_target(current_price, is_buy=False),
                confidence=0.7,
                reason=f"Fast MA ({self.fast_period}) crossed BELOW slow MA ({self.slow_period})"
            )

        # No crossover - HOLD
        else:
            trend = "uptrend" if current_fast > current_slow else "downtrend"
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                reason=f"No crossover. Currently in {trend}."
            )

    def get_parameters(self) -> dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }

    def set_parameters(self, **kwargs) -> None:
        if 'fast_period' in kwargs:
            self.fast_period = kwargs['fast_period']
        if 'slow_period' in kwargs:
            self.slow_period = kwargs['slow_period']
