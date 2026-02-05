# -*- coding: utf-8 -*-
"""
User Interface Components
=========================
Beautiful and simple GUI that anyone can use!

Components:
- AlgoTraderApp: Main application window
- Dashboard: Trading dashboard with metrics
- StrategyPicker: Visual strategy selection
- SettingsPanel: Configuration management
- StockSearch: Searchable stock selector
- Charts: Candlestick and line charts
- OrderPanel: Manual order placement (Market, Limit, Iceberg, TWAP, etc.)
- Themes: Color schemes (dark, light, neon)
"""

from .app import AlgoTraderApp, main
from .themes import THEMES, get_theme
from .dashboard import Dashboard, MetricCard, StatusWidget, PositionsTable, ActivityFeed
from .strategy_picker import StrategyPicker, StrategyCard, STRATEGY_INFO
from .settings_panel import SettingsPanel, SettingsDialog, SettingsSection
from .stock_search import StockSearchWidget, QuickStockSelector
from .order_panel import OrderPanel, OrderTypeSelector, OrderForm, OrderHistoryTable
from .automation_panel import AutomationPanel, PhaseIndicator, EventStream, StrategyStatus
from .charts import (
    SimpleChart,
    CandlestickChart,
    ChartColors,
    ChartWindow,
    add_moving_average,
    add_bollinger_bands,
    add_ema,
    add_rsi,
    add_macd,
    add_vwap,
    add_supertrend,
    quick_plot,
    quick_candle
)

__all__ = [
    # Main App
    'AlgoTraderApp',
    'main',

    # Themes
    'THEMES',
    'get_theme',

    # Dashboard
    'Dashboard',
    'MetricCard',
    'StatusWidget',
    'PositionsTable',
    'ActivityFeed',

    # Strategy Picker
    'StrategyPicker',
    'StrategyCard',
    'STRATEGY_INFO',

    # Settings
    'SettingsPanel',
    'SettingsDialog',
    'SettingsSection',

    # Stock Search
    'StockSearchWidget',
    'QuickStockSelector',

    # Order Panel
    'OrderPanel',
    'OrderTypeSelector',
    'OrderForm',
    'OrderHistoryTable',

    # Automation Panel
    'AutomationPanel',
    'PhaseIndicator',
    'EventStream',
    'StrategyStatus',

    # Charts
    'SimpleChart',
    'CandlestickChart',
    'ChartColors',
    'ChartWindow',
    'add_moving_average',
    'add_bollinger_bands',
    'add_ema',
    'add_rsi',
    'add_macd',
    'add_vwap',
    'add_supertrend',
    'quick_plot',
    'quick_candle',
]
