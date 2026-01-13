"""
Base Strategy Class
All strategies follow this pattern.

Think of this as a recipe template - each strategy fills in the details!
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import pandas as pd


class SignalType(Enum):
    """What the strategy is telling you to do"""
    BUY = "BUY"           # 🟢 Buy now!
    SELL = "SELL"         # 🔴 Sell now!
    HOLD = "HOLD"         # 🟡 Wait and watch
    EXIT = "EXIT"         # 🚪 Close your position


class RiskLevel(Enum):
    """How risky is this strategy?"""
    LOW = "LOW"           # 🐢 Safe, small gains
    MEDIUM = "MEDIUM"     # ⚡ Balanced
    HIGH = "HIGH"         # 🚀 Risky, big potential


@dataclass
class Signal:
    """
    A trading signal from a strategy.

    This is what the strategy tells you to do.
    """
    signal_type: SignalType    # BUY, SELL, HOLD, or EXIT
    symbol: str                # Which stock
    price: float               # Current price
    stop_loss: float = 0       # Where to cut losses
    target: float = 0          # Where to take profits
    confidence: float = 0.5    # How sure (0 to 1)
    reason: str = ""           # Why this signal
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def emoji(self) -> str:
        """Get emoji for the signal"""
        return {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡",
            SignalType.EXIT: "🚪",
        }.get(self.signal_type, "❓")


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Every strategy must implement:
    1. name - What's it called?
    2. description - What does it do?
    3. analyze() - Look at data and decide what to do
    """

    def __init__(self):
        self._signals: List[Signal] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Simple description a 5th grader can understand"""
        pass

    @property
    def risk_level(self) -> RiskLevel:
        """How risky is this strategy?"""
        return RiskLevel.MEDIUM

    @property
    def emoji(self) -> str:
        """Emoji representing this strategy"""
        return "📊"

    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyze the data and generate a signal.

        Args:
            data: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            symbol: Stock symbol

        Returns:
            Signal telling what to do
        """
        pass

    def calculate_stop_loss(self, price: float, is_buy: bool,
                           risk_percent: float = 2.0) -> float:
        """
        Calculate where to place stop-loss.

        Args:
            price: Entry price
            is_buy: True for buy, False for sell
            risk_percent: How much to risk (default 2%)

        Returns:
            Stop-loss price
        """
        if is_buy:
            return price * (1 - risk_percent / 100)
        else:
            return price * (1 + risk_percent / 100)

    def calculate_target(self, price: float, is_buy: bool,
                        reward_ratio: float = 2.0) -> float:
        """
        Calculate profit target.

        Args:
            price: Entry price
            is_buy: True for buy, False for sell
            reward_ratio: Risk:Reward ratio (default 1:2)

        Returns:
            Target price
        """
        risk = abs(price - self.calculate_stop_loss(price, is_buy)) * reward_ratio
        if is_buy:
            return price + risk
        else:
            return price - risk

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters for display"""
        return {}

    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters"""
        pass

    def __str__(self) -> str:
        return f"{self.emoji} {self.name}"
