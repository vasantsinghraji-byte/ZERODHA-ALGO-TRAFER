"""
Trading Strategies
Pre-built strategies that anyone can use!

Each strategy is like a different game plan:
- TURTLE: Slow and steady (safe)
- MOMENTUM: Follow the winners
- BREAKOUT: Catch big moves
- SUPERTREND: Follow the trend
"""

from .base import Strategy, Signal, SignalType
from .moving_average import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .supertrend import SupertrendStrategy

__all__ = [
    'Strategy',
    'Signal',
    'SignalType',
    'MovingAverageCrossover',
    'RSIStrategy',
    'SupertrendStrategy',
]
