"""
RSI (Relative Strength Index) Strategy
The MOMENTUM strategy ⚡

How it works:
- RSI measures how "overbought" or "oversold" a stock is
- RSI below 30 = Stock is oversold (cheap) = BUY
- RSI above 70 = Stock is overbought (expensive) = SELL

Think of it like a thermometer:
- Below 30 = Cold (time to buy)
- Above 70 = Hot (time to sell)
- 30-70 = Normal (wait and watch)
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType, RiskLevel


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy

    Buys when stock is oversold, sells when overbought.
    Good for catching reversals!

    Parameters:
        period: RSI calculation period (default: 14)
        oversold: Buy below this level (default: 30)
        overbought: Sell above this level (default: 70)
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return "RSI Strategy"

    @property
    def description(self) -> str:
        return "Buy when oversold (RSI < 30), sell when overbought (RSI > 70). Catches reversals!"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        return "⚡"

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze price data using RSI.

        Args:
            data: DataFrame with 'close' column
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        if len(data) < self.period + 1:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=data['close'].iloc[-1] if len(data) > 0 else 0,
                reason="Not enough data for RSI"
            )

        # Calculate RSI
        data = data.copy()
        data['rsi'] = self._calculate_rsi(data['close'])

        current_rsi = data['rsi'].iloc[-1]
        prev_rsi = data['rsi'].iloc[-2]
        current_price = data['close'].iloc[-1]

        # BUY: RSI crosses above oversold level
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=True),
                target=self.calculate_target(current_price, is_buy=True),
                confidence=0.75,
                reason=f"RSI crossed above {self.oversold} (was oversold, now recovering)"
            )

        # Strong BUY: RSI is very low
        elif current_rsi < self.oversold:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=True),
                target=self.calculate_target(current_price, is_buy=True),
                confidence=0.6,
                reason=f"RSI is {current_rsi:.1f} - Stock is OVERSOLD (cheap)"
            )

        # SELL: RSI crosses below overbought level
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=False),
                target=self.calculate_target(current_price, is_buy=False),
                confidence=0.75,
                reason=f"RSI crossed below {self.overbought} (was overbought, now falling)"
            )

        # Strong SELL: RSI is very high
        elif current_rsi > self.overbought:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, is_buy=False),
                target=self.calculate_target(current_price, is_buy=False),
                confidence=0.6,
                reason=f"RSI is {current_rsi:.1f} - Stock is OVERBOUGHT (expensive)"
            )

        # HOLD: RSI in normal range
        else:
            status = "neutral"
            if current_rsi < 50:
                status = "slightly bearish"
            elif current_rsi > 50:
                status = "slightly bullish"

            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                reason=f"RSI is {current_rsi:.1f} - {status}. Wait for better signal."
            )

    def get_parameters(self) -> dict:
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }

    def set_parameters(self, **kwargs) -> None:
        if 'period' in kwargs:
            self.period = kwargs['period']
        if 'oversold' in kwargs:
            self.oversold = kwargs['oversold']
        if 'overbought' in kwargs:
            self.overbought = kwargs['overbought']
