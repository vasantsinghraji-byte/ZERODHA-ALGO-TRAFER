# -*- coding: utf-8 -*-
"""
Bollinger Bands Strategy - The Rubber Band Strategy!
=====================================================
Uses Bollinger Bands to find oversold/overbought conditions.

Think of it like a rubber band:
- When stretched too far down = will snap back UP (BUY!)
- When stretched too far up = will snap back DOWN (SELL!)
"""

import pandas as pd
from typing import Dict, Any

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion Strategy.

    BUY when price touches lower band (oversold)
    SELL when price touches upper band (overbought)

    Good for: Sideways/ranging markets
    Risk: Medium
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ):
        """
        Initialize Bollinger Bands Strategy.

        Args:
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
        """
        super().__init__()
        self.period = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return "Bollinger Bands"

    @property
    def description(self) -> str:
        return "Buys when price is low (touching bottom band), sells when high"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        return "🎸"

    def _calculate_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        close = data['close'] if 'close' in data.columns else data['Close']

        # Middle band (SMA)
        middle = close.rolling(window=self.period).mean()

        # Standard deviation
        std = close.rolling(window=self.period).std()

        # Upper and lower bands
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        # %B indicator (where price is within the bands)
        percent_b = (close - lower) / (upper - lower)

        # Bandwidth (volatility measure)
        bandwidth = (upper - lower) / middle * 100

        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'percent_b': percent_b,
            'bandwidth': bandwidth
        })

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze data and generate signal.

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        if len(data) < self.period:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Not enough data"
            )

        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[-1]

        # Calculate Bollinger Bands
        bands = self._calculate_bands(data)

        upper = bands['upper'].iloc[-1]
        middle = bands['middle'].iloc[-1]
        lower = bands['lower'].iloc[-1]
        percent_b = bands['percent_b'].iloc[-1]

        # Calculate confidence based on how far from middle
        distance_from_middle = abs(current_price - middle) / middle
        confidence = min(0.9, 0.5 + distance_from_middle * 5)

        # Check signals
        if current_price <= lower:
            # Price at/below lower band - oversold, potential BUY
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=lower * 0.98,  # 2% below lower band
                target=middle,  # Target is middle band
                confidence=confidence,
                reason=f"Price at lower band (oversold)! %B: {percent_b:.2f}"
            )
        elif current_price >= upper:
            # Price at/above upper band - overbought, potential SELL
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=upper * 1.02,  # 2% above upper band
                target=middle,
                confidence=confidence,
                reason=f"Price at upper band (overbought)! %B: {percent_b:.2f}"
            )
        elif percent_b < 0.2:
            # Near lower band
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=lower * 0.98,
                target=middle,
                confidence=confidence * 0.7,
                reason=f"Approaching lower band. %B: {percent_b:.2f}"
            )
        elif percent_b > 0.8:
            # Near upper band
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=upper * 1.02,
                target=middle,
                confidence=confidence * 0.7,
                reason=f"Approaching upper band. %B: {percent_b:.2f}"
            )
        else:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=0.3,
                reason=f"Price in middle zone. %B: {percent_b:.2f}"
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "std_dev": self.std_dev
        }

    def set_parameters(self, **kwargs) -> None:
        if "period" in kwargs:
            self.period = kwargs["period"]
        if "std_dev" in kwargs:
            self.std_dev = kwargs["std_dev"]
