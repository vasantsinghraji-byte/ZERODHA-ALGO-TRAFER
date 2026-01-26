# -*- coding: utf-8 -*-
"""
ORB Strategy - Opening Range Breakout!
======================================
Trades based on the first 15-30 minutes of market opening.

Think of it like the first pitch in a baseball game:
- If the ball goes UP first, the game often continues UP!
- If the ball goes DOWN first, the game often continues DOWN!

Most popular intraday strategy in India!
"""

import pandas as pd
from datetime import datetime, time
from typing import Dict, Any, Optional

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class ORBStrategy(Strategy):
    """
    Opening Range Breakout Strategy.

    1. Wait for first 15-30 minutes
    2. Mark the high and low of this period
    3. BUY when price breaks above the high
    4. SELL when price breaks below the low

    Good for: Intraday trading, Nifty/Bank Nifty
    Risk: Medium-High
    """

    def __init__(
        self,
        opening_minutes: int = 15,
        buffer_percent: float = 0.1
    ):
        """
        Initialize ORB Strategy.

        Args:
            opening_minutes: Minutes for opening range (default 15)
            buffer_percent: Buffer above/below range for entry (default 0.1%)
        """
        super().__init__()
        self.opening_minutes = opening_minutes
        self.buffer_percent = buffer_percent

        # Track daily range
        self._daily_high: Optional[float] = None
        self._daily_low: Optional[float] = None
        self._range_set_date: Optional[str] = None

    @property
    def name(self) -> str:
        return "Opening Range Breakout"

    @property
    def description(self) -> str:
        return f"Trades breakout of first {self.opening_minutes} minutes range"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    @property
    def emoji(self) -> str:
        return "🌅"

    def _calculate_opening_range(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the opening range from data using TIME-BASED filtering.

        CRITICAL: Uses between_time() for accurate time-based range calculation.
        This works correctly regardless of candle timeframe (1m, 5m, 15m, etc.).

        Market opens at 9:15 AM IST. Opening range = 9:15 to 9:15 + opening_minutes.
        """
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        close_col = 'close' if 'close' in data.columns else 'Close'

        # Market opening time (IST)
        market_open = time(9, 15)

        # Calculate opening range end time based on opening_minutes
        end_hour = 9
        end_minute = 15 + self.opening_minutes
        if end_minute >= 60:
            end_hour += end_minute // 60
            end_minute = end_minute % 60
        opening_range_end = time(end_hour, end_minute)

        # Try TIME-BASED filtering first (correct approach)
        opening_data = None

        if isinstance(data.index, pd.DatetimeIndex):
            # Use between_time for proper time-based filtering
            try:
                opening_data = data.between_time(market_open, opening_range_end)
            except Exception:
                pass  # Fall through to fallback

        # Fallback: If index is not datetime or between_time failed
        if opening_data is None or opening_data.empty:
            # Check if there's a 'datetime' or 'time' column
            time_col = None
            for col in ['datetime', 'time', 'timestamp', 'date']:
                if col in data.columns:
                    time_col = col
                    break

            if time_col:
                try:
                    # Convert to datetime and filter
                    times = pd.to_datetime(data[time_col])
                    mask = (times.dt.time >= market_open) & (times.dt.time <= opening_range_end)
                    opening_data = data[mask]
                except Exception:
                    pass

        # Last resort fallback: Position-based (with warning)
        if opening_data is None or opening_data.empty:
            # Calculate approximate number of candles based on typical 1-minute data
            # This is a FALLBACK and should log a warning in production
            approx_candles = self.opening_minutes
            if len(data) >= approx_candles:
                opening_data = data.iloc[:approx_candles]
            else:
                opening_data = data.iloc[:max(1, len(data) // 4)]

        # Calculate range from opening data
        range_high = opening_data[high_col].max()
        range_low = opening_data[low_col].min()

        # Handle edge case where range is zero or invalid
        if pd.isna(range_high) or pd.isna(range_low) or range_high <= range_low:
            # Use full data range as fallback
            range_high = data[high_col].max()
            range_low = data[low_col].min()

        # Add buffer for entry (avoid false breakouts)
        buffer = (range_high - range_low) * self.buffer_percent / 100
        entry_high = range_high + buffer
        entry_low = range_low - buffer

        current_price = data[close_col].iloc[-1]

        return {
            'range_high': range_high,
            'range_low': range_low,
            'entry_high': entry_high,
            'entry_low': entry_low,
            'range_size': range_high - range_low,
            'current_price': current_price,
            'opening_candles': len(opening_data)  # For debugging
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

        # Calculate opening range
        orb = self._calculate_opening_range(data)
        current_price = orb['current_price']
        range_high = orb['range_high']
        range_low = orb['range_low']

        # Calculate targets (1:2 risk reward)
        range_size = orb['range_size']

        # Check for breakout
        if current_price > orb['entry_high']:
            # Upside breakout
            stop_loss = range_low  # Below range low
            target = current_price + (range_size * 2)  # 2x range size

            confidence = min(0.85, 0.6 + (current_price - range_high) / range_high * 10)

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                reason=f"ORB Breakout UP! Range: Rs.{range_low:.0f}-{range_high:.0f}"
            )

        elif current_price < orb['entry_low']:
            # Downside breakout
            stop_loss = range_high  # Above range high
            target = current_price - (range_size * 2)

            confidence = min(0.85, 0.6 + (range_low - current_price) / range_low * 10)

            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=stop_loss,
                target=target,
                confidence=confidence,
                reason=f"ORB Breakdown DOWN! Range: Rs.{range_low:.0f}-{range_high:.0f}"
            )

        else:
            # Inside range
            distance_to_high = range_high - current_price
            distance_to_low = current_price - range_low

            if distance_to_high < distance_to_low:
                hint = f"Near top of range (Rs.{range_high:.0f})"
            else:
                hint = f"Near bottom of range (Rs.{range_low:.0f})"

            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=0.4,
                reason=f"Inside ORB range. {hint}"
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "opening_minutes": self.opening_minutes,
            "buffer_percent": self.buffer_percent
        }

    def set_parameters(self, **kwargs) -> None:
        if "opening_minutes" in kwargs:
            self.opening_minutes = kwargs["opening_minutes"]
        if "buffer_percent" in kwargs:
            self.buffer_percent = kwargs["buffer_percent"]
