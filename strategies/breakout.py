# -*- coding: utf-8 -*-
"""
Breakout Strategy - The Escape Artist!
======================================
Buys when price breaks above resistance, sells when breaks below support.

Think of it like a balloon:
- When it pops through the ceiling = price will fly UP! (BUY!)
- When it falls through the floor = price will drop DOWN! (SELL!)
"""

import pandas as pd
from typing import Dict, Any

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class BreakoutStrategy(Strategy):
    """
    Price Breakout Strategy.

    BUY when price breaks above recent high (resistance)
    SELL when price breaks below recent low (support)

    Good for: Volatile stocks, after consolidation
    Risk: Medium-High
    """

    def __init__(
        self,
        lookback_period: int = 20,
        confirmation_candles: int = 2
    ):
        """
        Initialize Breakout Strategy.

        Args:
            lookback_period: Days to look back for high/low (default 20)
            confirmation_candles: Candles above/below for confirmation (default 2)
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.confirmation_candles = confirmation_candles

    @property
    def name(self) -> str:
        return "Price Breakout"

    @property
    def description(self) -> str:
        return "Buys when price breaks above recent highs (escape artist!)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    @property
    def emoji(self) -> str:
        return "💥"

    def _calculate_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        close_col = 'close' if 'close' in data.columns else 'Close'

        # Get recent data (excluding last candle)
        recent = data.iloc[-(self.lookback_period + 1):-1]

        resistance = recent[high_col].max()
        support = recent[low_col].min()
        current_price = data[close_col].iloc[-1]

        # Calculate range
        range_size = resistance - support
        range_percent = range_size / support * 100

        return {
            'resistance': resistance,
            'support': support,
            'current_price': current_price,
            'range_size': range_size,
            'range_percent': range_percent
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
        if len(data) < self.lookback_period + self.confirmation_candles:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Not enough data"
            )

        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'

        levels = self._calculate_levels(data)
        current_price = levels['current_price']
        resistance = levels['resistance']
        support = levels['support']

        # Check for breakout confirmation
        recent_closes = data[close_col].iloc[-self.confirmation_candles:]
        recent_highs = data[high_col].iloc[-self.confirmation_candles:]
        recent_lows = data[low_col].iloc[-self.confirmation_candles:]

        # Upside breakout: all recent closes above resistance
        upside_breakout = all(recent_closes > resistance)

        # Downside breakout: all recent closes below support
        downside_breakout = all(recent_closes < support)

        # Calculate confidence based on breakout strength
        if upside_breakout:
            breakout_strength = (current_price - resistance) / resistance * 100
            confidence = min(0.9, 0.6 + breakout_strength * 2)

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=resistance * 0.98,  # Just below resistance
                target=current_price + levels['range_size'],  # Project range
                confidence=confidence,
                reason=f"BREAKOUT above Rs.{resistance:.2f}! +{breakout_strength:.1f}%"
            )
        elif downside_breakout:
            breakout_strength = (support - current_price) / support * 100
            confidence = min(0.9, 0.6 + breakout_strength * 2)

            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=support * 1.02,  # Just above support
                target=current_price - levels['range_size'],
                confidence=confidence,
                reason=f"BREAKDOWN below Rs.{support:.2f}! -{breakout_strength:.1f}%"
            )
        else:
            # Price near resistance
            if current_price > resistance * 0.98:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    price=current_price,
                    confidence=0.5,
                    reason=f"Approaching resistance Rs.{resistance:.2f}"
                )
            # Price near support
            elif current_price < support * 1.02:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    price=current_price,
                    confidence=0.5,
                    reason=f"Approaching support Rs.{support:.2f}"
                )
            else:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    price=current_price,
                    confidence=0.3,
                    reason=f"Range: Rs.{support:.2f} - Rs.{resistance:.2f}"
                )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "lookback_period": self.lookback_period,
            "confirmation_candles": self.confirmation_candles
        }

    def set_parameters(self, **kwargs) -> None:
        if "lookback_period" in kwargs:
            self.lookback_period = kwargs["lookback_period"]
        if "confirmation_candles" in kwargs:
            self.confirmation_candles = kwargs["confirmation_candles"]
