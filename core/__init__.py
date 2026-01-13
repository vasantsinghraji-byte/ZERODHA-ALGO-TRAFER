"""
Core Trading Engine Module

This module contains the core business logic for the AlgoTrader Pro application,
separated from the UI layer for better maintainability and testability.
"""

from .trading_engine import TradingEngine
from .data_manager import DataManager
from .order_manager import OrderManager
from .risk_manager import RiskManager

__all__ = [
    'TradingEngine',
    'DataManager',
    'OrderManager',
    'RiskManager'
]
