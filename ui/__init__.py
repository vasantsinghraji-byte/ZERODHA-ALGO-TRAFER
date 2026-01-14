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
- Charts: Candlestick and line charts
- Themes: Color schemes (dark, light, neon)
"""

from .app import AlgoTraderApp, main
from .themes import THEMES, get_theme
from .dashboard import Dashboard, MetricCard, StatusWidget, PositionsTable, ActivityFeed
from .strategy_picker import StrategyPicker, StrategyCard, STRATEGY_INFO
from .settings_panel import SettingsPanel, SettingsDialog, SettingsSection
from .charts import (
    SimpleChart,
    CandlestickChart,
    ChartColors,
    ChartWindow,
    add_moving_average,
    add_bollinger_bands,
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

    # Charts
    'SimpleChart',
    'CandlestickChart',
    'ChartColors',
    'ChartWindow',
    'add_moving_average',
    'add_bollinger_bands',
    'quick_plot',
    'quick_candle',
]
