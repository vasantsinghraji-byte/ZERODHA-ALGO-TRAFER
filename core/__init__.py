"""
Core Trading Engine Module - The Brain!
=======================================
Contains all the core trading logic.

Components:
- Broker: Zerodha API connection
- DataManager: Historical & live data
- OrderManager: Buy/sell orders
- PositionManager: Track holdings
- TradingEngine: Main autopilot
- RiskManager: Safety controls
"""

from .broker import ZerodhaBroker, Quote, Order, Position
from .data_manager import DataManager
from .order_manager import OrderManager, OrderStatus, Side, OrderType
from .position_manager import PositionManager, Position as ManagedPosition
from .trading_engine import TradingEngine, TradingMode, EngineConfig, create_paper_engine
from .live_feed import LiveFeed, SimulatedFeed, Tick

# Risk management
from .risk_manager import (
    RiskManager,
    RiskConfig,
    RiskMetrics,
    StopLossType,
    calculate_risk_reward,
    calculate_position_size_fixed
)

__all__ = [
    # Broker
    'ZerodhaBroker',
    'Quote',
    'Order',
    'Position',

    # Data
    'DataManager',
    'LiveFeed',
    'SimulatedFeed',
    'Tick',

    # Orders
    'OrderManager',
    'OrderStatus',
    'Side',
    'OrderType',

    # Positions
    'PositionManager',
    'ManagedPosition',

    # Engine
    'TradingEngine',
    'TradingMode',
    'EngineConfig',
    'create_paper_engine',

    # Risk
    'RiskManager',
    'RiskConfig',
    'RiskMetrics',
    'StopLossType',
    'calculate_risk_reward',
    'calculate_position_size_fixed',
]
