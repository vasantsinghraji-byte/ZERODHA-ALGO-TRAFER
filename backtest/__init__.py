# -*- coding: utf-8 -*-
"""
Backtesting Module
==================
Test your strategies on historical data before risking real money!
"""

from .engine import Backtester, BacktestResult
from .metrics import calculate_metrics, print_report

__all__ = [
    'Backtester',
    'BacktestResult',
    'calculate_metrics',
    'print_report',
]
