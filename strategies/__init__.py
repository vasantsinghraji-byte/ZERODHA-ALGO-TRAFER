"""
Trading Strategies - Pick Your Game Plan!
==========================================
Pre-built strategies that anyone can use!

8 STRATEGIES TO CHOOSE FROM:
- TURTLE (Moving Average): Slow and steady (safe)
- MOMENTUM (RSI): Follow the winners
- SUPERTREND: Follow the trend aggressively
- MACD: Traffic light signals
- BOLLINGER: Buy low, sell high
- BREAKOUT: Catch big moves
- VWAP: Trade fair value
- ORB: Opening range breakout
- MULTI: Smart combo of all
"""

from .base import Strategy, Signal, SignalType, RiskLevel
from .moving_average import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .supertrend import SupertrendStrategy
from .macd_strategy import MACDStrategy
from .bollinger_bands import BollingerBandsStrategy
from .breakout import BreakoutStrategy
from .vwap_strategy import VWAPStrategy
from .orb_strategy import ORBStrategy
from .multi_indicator import MultiIndicatorStrategy

# All available strategies
ALL_STRATEGIES = {
    'turtle': MovingAverageCrossover,
    'momentum': RSIStrategy,
    'supertrend': SupertrendStrategy,
    'macd': MACDStrategy,
    'bollinger': BollingerBandsStrategy,
    'breakout': BreakoutStrategy,
    'vwap': VWAPStrategy,
    'orb': ORBStrategy,
    'multi': MultiIndicatorStrategy,
}

def get_strategy(name: str) -> Strategy:
    """Get a strategy by name"""
    name = name.lower()
    if name in ALL_STRATEGIES:
        return ALL_STRATEGIES[name]()
    raise ValueError(f"Unknown strategy: {name}. Available: {list(ALL_STRATEGIES.keys())}")

def list_strategies():
    """List all available strategies"""
    print("\n" + "=" * 50)
    print("AVAILABLE STRATEGIES")
    print("=" * 50)
    for name, cls in ALL_STRATEGIES.items():
        strategy = cls()
        print(f"{strategy.emoji} {name.upper()}: {strategy.description}")
        print(f"   Risk: {strategy.risk_level.value}")
        print()

__all__ = [
    'Strategy',
    'Signal',
    'SignalType',
    'RiskLevel',
    'MovingAverageCrossover',
    'RSIStrategy',
    'SupertrendStrategy',
    'MACDStrategy',
    'BollingerBandsStrategy',
    'BreakoutStrategy',
    'VWAPStrategy',
    'ORBStrategy',
    'MultiIndicatorStrategy',
    'ALL_STRATEGIES',
    'get_strategy',
    'list_strategies',
]
