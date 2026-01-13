"""
Supertrend Strategy
The ROCKET strategy! 🚀

How it works:
- Supertrend is a line that follows the price
- When price is ABOVE the line = UPTREND (BUY)
- When price is BELOW the line = DOWNTREND (SELL)

Think of it like a GPS:
- Line below price = "Go UP" (buy and hold)
- Line above price = "Go DOWN" (sell or short)

This strategy catches big trends but can be volatile!
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType, RiskLevel


class SupertrendStrategy(Strategy):
    """
    Supertrend Strategy

    Follows the trend using Supertrend indicator.
    Great for catching big moves!

    Parameters:
        period: ATR period (default: 10)
        multiplier: ATR multiplier (default: 3)
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__()
        self.period = period
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return "Supertrend"

    @property
    def description(self) -> str:
        return "Follow the trend with Supertrend indicator. Catches big moves but needs patience!"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    @property
    def emoji(self) -> str:
        return "🚀"

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        return atr

    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend indicator"""
        data = data.copy()

        # Calculate ATR
        data['atr'] = self._calculate_atr(data)

        # Calculate basic bands
        hl2 = (data['high'] + data['low']) / 2
        data['basic_ub'] = hl2 + (self.multiplier * data['atr'])
        data['basic_lb'] = hl2 - (self.multiplier * data['atr'])

        # Initialize Supertrend columns
        data['final_ub'] = 0.0
        data['final_lb'] = 0.0
        data['supertrend'] = 0.0
        data['trend'] = 1  # 1 for up, -1 for down

        # Calculate Supertrend
        for i in range(self.period, len(data)):
            # Final Upper Band
            if data['basic_ub'].iloc[i] < data['final_ub'].iloc[i-1] or data['close'].iloc[i-1] > data['final_ub'].iloc[i-1]:
                data.loc[data.index[i], 'final_ub'] = data['basic_ub'].iloc[i]
            else:
                data.loc[data.index[i], 'final_ub'] = data['final_ub'].iloc[i-1]

            # Final Lower Band
            if data['basic_lb'].iloc[i] > data['final_lb'].iloc[i-1] or data['close'].iloc[i-1] < data['final_lb'].iloc[i-1]:
                data.loc[data.index[i], 'final_lb'] = data['basic_lb'].iloc[i]
            else:
                data.loc[data.index[i], 'final_lb'] = data['final_lb'].iloc[i-1]

            # Supertrend
            if data['trend'].iloc[i-1] == 1:
                if data['close'].iloc[i] <= data['final_ub'].iloc[i]:
                    data.loc[data.index[i], 'trend'] = 1
                    data.loc[data.index[i], 'supertrend'] = data['final_lb'].iloc[i]
                else:
                    data.loc[data.index[i], 'trend'] = -1
                    data.loc[data.index[i], 'supertrend'] = data['final_ub'].iloc[i]
            else:
                if data['close'].iloc[i] >= data['final_lb'].iloc[i]:
                    data.loc[data.index[i], 'trend'] = -1
                    data.loc[data.index[i], 'supertrend'] = data['final_ub'].iloc[i]
                else:
                    data.loc[data.index[i], 'trend'] = 1
                    data.loc[data.index[i], 'supertrend'] = data['final_lb'].iloc[i]

        return data

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze using Supertrend indicator.

        Args:
            data: DataFrame with OHLC columns
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Missing OHLC data"
            )

        if len(data) < self.period + 2:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=data['close'].iloc[-1] if len(data) > 0 else 0,
                reason="Not enough data for Supertrend"
            )

        # Calculate Supertrend
        data = self._calculate_supertrend(data)

        current_trend = data['trend'].iloc[-1]
        prev_trend = data['trend'].iloc[-2]
        current_price = data['close'].iloc[-1]
        supertrend_value = data['supertrend'].iloc[-1]

        # BUY: Trend changed from down to up
        if prev_trend == -1 and current_trend == 1:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=supertrend_value,
                target=current_price + (current_price - supertrend_value) * 2,
                confidence=0.8,
                reason="Supertrend turned BULLISH! 🚀 Trend reversal to upside."
            )

        # SELL: Trend changed from up to down
        elif prev_trend == 1 and current_trend == -1:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=supertrend_value,
                target=current_price - (supertrend_value - current_price) * 2,
                confidence=0.8,
                reason="Supertrend turned BEARISH! 📉 Trend reversal to downside."
            )

        # Continuing trend
        else:
            if current_trend == 1:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    price=current_price,
                    reason=f"UPTREND continues. Supertrend support at {supertrend_value:.2f}"
                )
            else:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    price=current_price,
                    reason=f"DOWNTREND continues. Supertrend resistance at {supertrend_value:.2f}"
                )

    def get_parameters(self) -> dict:
        return {
            'period': self.period,
            'multiplier': self.multiplier
        }

    def set_parameters(self, **kwargs) -> None:
        if 'period' in kwargs:
            self.period = kwargs['period']
        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']
