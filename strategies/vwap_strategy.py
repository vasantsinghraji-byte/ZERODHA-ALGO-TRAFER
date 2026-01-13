# -*- coding: utf-8 -*-
"""
VWAP Strategy - The Fair Price Strategy!
========================================
Uses VWAP (Volume Weighted Average Price) to find fair value.

Think of it like the "fair price" of a stock:
- If price is BELOW fair price = It's cheap, BUY!
- If price is ABOVE fair price = It's expensive, SELL!

Professional traders love this strategy!
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class VWAPStrategy(Strategy):
    """
    VWAP Mean Reversion Strategy.

    BUY when price is below VWAP (undervalued)
    SELL when price is above VWAP (overvalued)

    Good for: Intraday trading, institutional-grade
    Risk: Low-Medium
    """

    def __init__(
        self,
        deviation_threshold: float = 1.5
    ):
        """
        Initialize VWAP Strategy.

        Args:
            deviation_threshold: % deviation from VWAP to trigger signal (default 1.5%)
        """
        super().__init__()
        self.deviation_threshold = deviation_threshold

    @property
    def name(self) -> str:
        return "VWAP Strategy"

    @property
    def description(self) -> str:
        return "Buys below fair price (VWAP), sells above it. Pro trader's favorite!"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def emoji(self) -> str:
        return "⚖️"

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        high = data['high'] if 'high' in data.columns else data['High']
        low = data['low'] if 'low' in data.columns else data['Low']
        close = data['close'] if 'close' in data.columns else data['Close']
        volume = data['volume'] if 'volume' in data.columns else data['Volume']

        # Typical price
        typical_price = (high + low + close) / 3

        # Cumulative values
        cum_volume = volume.cumsum()
        cum_tp_volume = (typical_price * volume).cumsum()

        # VWAP
        vwap = cum_tp_volume / cum_volume

        return vwap

    def _calculate_vwap_bands(self, data: pd.DataFrame, vwap: pd.Series) -> Dict:
        """Calculate VWAP with standard deviation bands"""
        high = data['high'] if 'high' in data.columns else data['High']
        low = data['low'] if 'low' in data.columns else data['Low']
        close = data['close'] if 'close' in data.columns else data['Close']
        volume = data['volume'] if 'volume' in data.columns else data['Volume']

        typical_price = (high + low + close) / 3

        # Calculate standard deviation
        squared_diff = (typical_price - vwap) ** 2
        variance = (squared_diff * volume).cumsum() / volume.cumsum()
        std_dev = np.sqrt(variance)

        return {
            'upper_1': vwap + std_dev,
            'lower_1': vwap - std_dev,
            'upper_2': vwap + 2 * std_dev,
            'lower_2': vwap - 2 * std_dev
        }

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze data and generate signal.

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        if len(data) < 10:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Not enough data"
            )

        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[-1]

        # Calculate VWAP
        vwap = self._calculate_vwap(data)
        current_vwap = vwap.iloc[-1]

        # Calculate bands
        bands = self._calculate_vwap_bands(data, vwap)

        # Calculate deviation from VWAP
        deviation = (current_price - current_vwap) / current_vwap * 100

        # Calculate confidence based on deviation
        confidence = min(0.9, 0.4 + abs(deviation) / 3)

        if deviation < -self.deviation_threshold:
            # Price below VWAP - undervalued
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=bands['lower_2'].iloc[-1],
                target=current_vwap,  # Target is VWAP
                confidence=confidence,
                reason=f"Below VWAP by {abs(deviation):.2f}% (undervalued!)"
            )
        elif deviation > self.deviation_threshold:
            # Price above VWAP - overvalued
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=bands['upper_2'].iloc[-1],
                target=current_vwap,
                confidence=confidence,
                reason=f"Above VWAP by {deviation:.2f}% (overvalued!)"
            )
        else:
            position = "above" if deviation > 0 else "below"
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=0.3,
                reason=f"Near VWAP (Rs.{current_vwap:.2f}), {abs(deviation):.2f}% {position}"
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "deviation_threshold": self.deviation_threshold
        }

    def set_parameters(self, **kwargs) -> None:
        if "deviation_threshold" in kwargs:
            self.deviation_threshold = kwargs["deviation_threshold"]
