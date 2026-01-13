# -*- coding: utf-8 -*-
"""
Multi-Indicator Strategy - The Smart Combo!
============================================
Uses multiple indicators together for stronger signals.

Think of it like asking multiple friends for advice:
- If ALL friends say BUY = Very strong signal!
- If friends disagree = Better to wait

Combines: RSI + MACD + Moving Averages
"""

import pandas as pd
from typing import Dict, Any, List

from strategies.base import Strategy, Signal, SignalType, RiskLevel


class MultiIndicatorStrategy(Strategy):
    """
    Multi-Indicator Confirmation Strategy.

    Only generates signals when MULTIPLE indicators agree:
    - RSI (momentum)
    - MACD (trend)
    - Moving Average (direction)

    Good for: Reducing false signals, higher accuracy
    Risk: Low (due to confirmation requirement)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        ma_fast: int = 20,
        ma_slow: int = 50,
        min_confirmations: int = 2
    ):
        """
        Initialize Multi-Indicator Strategy.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI level for oversold (BUY signal)
            rsi_overbought: RSI level for overbought (SELL signal)
            ma_fast: Fast moving average period
            ma_slow: Slow moving average period
            min_confirmations: Minimum indicators that must agree
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.min_confirmations = min_confirmations

    @property
    def name(self) -> str:
        return "Multi-Indicator Combo"

    @property
    def description(self) -> str:
        return "Uses RSI + MACD + MA together for strong signals"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def emoji(self) -> str:
        return "🎯"

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        close = data['close'] if 'close' in data.columns else data['Close']

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        close = data['close'] if 'close' in data.columns else data['Close']

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_ma(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Moving Averages"""
        close = data['close'] if 'close' in data.columns else data['Close']

        return {
            'ma_fast': close.rolling(window=self.ma_fast).mean(),
            'ma_slow': close.rolling(window=self.ma_slow).mean()
        }

    def _get_indicator_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get individual indicator signals"""
        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[-1]

        signals = {}

        # RSI Signal
        rsi = self._calculate_rsi(data)
        rsi_value = rsi.iloc[-1]

        if rsi_value < self.rsi_oversold:
            signals['RSI'] = 'BUY'
        elif rsi_value > self.rsi_overbought:
            signals['RSI'] = 'SELL'
        else:
            signals['RSI'] = 'HOLD'

        # MACD Signal
        macd = self._calculate_macd(data)
        macd_now = macd['macd'].iloc[-1]
        signal_now = macd['signal'].iloc[-1]
        macd_prev = macd['macd'].iloc[-2]
        signal_prev = macd['signal'].iloc[-2]

        if macd_prev < signal_prev and macd_now > signal_now:
            signals['MACD'] = 'BUY'
        elif macd_prev > signal_prev and macd_now < signal_now:
            signals['MACD'] = 'SELL'
        elif macd_now > signal_now:
            signals['MACD'] = 'BULLISH'
        else:
            signals['MACD'] = 'BEARISH'

        # Moving Average Signal
        ma = self._calculate_ma(data)
        ma_fast = ma['ma_fast'].iloc[-1]
        ma_slow = ma['ma_slow'].iloc[-1]

        if current_price > ma_fast > ma_slow:
            signals['MA'] = 'BUY'
        elif current_price < ma_fast < ma_slow:
            signals['MA'] = 'SELL'
        elif ma_fast > ma_slow:
            signals['MA'] = 'BULLISH'
        else:
            signals['MA'] = 'BEARISH'

        return signals

    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze data and generate signal based on multiple indicators.

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Trading signal
        """
        if len(data) < max(self.ma_slow, 26) + 10:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=0,
                reason="Not enough data"
            )

        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[-1]

        # Get individual signals
        signals = self._get_indicator_signals(data)

        # Count bullish and bearish signals
        buy_count = sum(1 for s in signals.values() if s in ['BUY', 'BULLISH'])
        sell_count = sum(1 for s in signals.values() if s in ['SELL', 'BEARISH'])

        strong_buy = sum(1 for s in signals.values() if s == 'BUY')
        strong_sell = sum(1 for s in signals.values() if s == 'SELL')

        # Build reason string
        reason_parts = [f"{k}={v}" for k, v in signals.items()]
        reason = " | ".join(reason_parts)

        # Calculate confidence
        total_indicators = len(signals)
        confidence = max(buy_count, sell_count) / total_indicators

        # Generate signal
        if strong_buy >= self.min_confirmations:
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, True),
                target=self.calculate_target(current_price, True),
                confidence=confidence,
                reason=f"STRONG BUY! {strong_buy}/{total_indicators} confirm. {reason}"
            )
        elif strong_sell >= self.min_confirmations:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=current_price,
                stop_loss=self.calculate_stop_loss(current_price, False),
                target=self.calculate_target(current_price, False),
                confidence=confidence,
                reason=f"STRONG SELL! {strong_sell}/{total_indicators} confirm. {reason}"
            )
        elif buy_count > sell_count:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=confidence * 0.5,
                reason=f"Leaning bullish ({buy_count}/{total_indicators}). {reason}"
            )
        elif sell_count > buy_count:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=confidence * 0.5,
                reason=f"Leaning bearish ({sell_count}/{total_indicators}). {reason}"
            )
        else:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=current_price,
                confidence=0.3,
                reason=f"Mixed signals. {reason}"
            )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "ma_fast": self.ma_fast,
            "ma_slow": self.ma_slow,
            "min_confirmations": self.min_confirmations
        }

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
