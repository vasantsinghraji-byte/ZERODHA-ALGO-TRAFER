# -*- coding: utf-8 -*-
"""
MACD Strategy - The Traffic Light Strategy!
============================================
Uses MACD (Moving Average Convergence Divergence) to find buy/sell signals.

Think of it like traffic lights:
- Green light (MACD crosses up) = GO BUY!
- Red light (MACD crosses down) = STOP, SELL!
"""

import pandas as pd
from typing import Dict, Any

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class MACDStrategy(Strategy):
    """
    MACD Crossover Strategy.

    BUY when MACD line crosses ABOVE signal line (bullish)
    SELL when MACD line crosses BELOW signal line (bearish)

    Good for: Trending markets
    Risk: Medium
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize MACD Strategy.

        Args:
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return "MACD Crossover"

    @property
    def description(self) -> str:
        return "Buys when fast line crosses above slow line (like green light!)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        return "🚦"

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        close = data['close'] if 'close' in data.columns else data['Close']

        # Calculate EMAs
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
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
        if len(data) < self.slow_period + self.signal_period:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Not enough data"
            )

        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[-1]

        # Calculate MACD
        macd_data = self._calculate_macd(data)

        macd_now = macd_data['macd'].iloc[-1]
        signal_now = macd_data['signal'].iloc[-1]
        macd_prev = macd_data['macd'].iloc[-2]
        signal_prev = macd_data['signal'].iloc[-2]
        histogram = macd_data['histogram'].iloc[-1]

        # Check for crossovers
        bullish_cross = macd_prev < signal_prev and macd_now > signal_now
        bearish_cross = macd_prev > signal_prev and macd_now < signal_now

        # Calculate confidence based on histogram strength
        hist_strength = abs(histogram) / current_price * 100
        confidence = min(0.9, 0.5 + hist_strength)

        if bullish_cross:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, True),
                target=self.calculate_target(current_price, True),
                confidence=confidence,
                reason=f"MACD bullish crossover! Histogram: {histogram:.2f}"
            )
        elif bearish_cross:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, False),
                target=self.calculate_target(current_price, False),
                confidence=confidence,
                reason=f"MACD bearish crossover! Histogram: {histogram:.2f}"
            )
        else:
            trend = "Bullish" if macd_now > signal_now else "Bearish"
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=0.3,
                reason=f"Waiting for crossover. Trend: {trend}"
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period
        }

    def set_parameters(self, **kwargs) -> None:
        if "fast_period" in kwargs:
            self.fast_period = kwargs["fast_period"]
        if "slow_period" in kwargs:
            self.slow_period = kwargs["slow_period"]
        if "signal_period" in kwargs:
            self.signal_period = kwargs["signal_period"]
