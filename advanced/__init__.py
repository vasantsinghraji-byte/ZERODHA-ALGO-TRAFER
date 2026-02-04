# -*- coding: utf-8 -*-
"""
Advanced Features Module
========================
Predictions, market scanning, alerts, and portfolio optimization.

Components:
- HeuristicPredictor: Weighted indicator scoring for price predictions
- MarketScanner: Find trading opportunities
- AlertManager: Telegram, Email, Desktop notifications
- PortfolioOptimizer: Smart money allocation
"""

from .ml_predictor import (
    MLPredictor,
    HeuristicPredictor,
    SimpleMLPredictor,  # Backward compatibility alias for HeuristicPredictor
    FeatureEngineering,
    SupportResistanceDetector,
    TrendAnalyzer,
    Prediction,
    PredictionType,
    SignalStrength,
    quick_predict,
    quick_analysis
)

from .market_scanner import (
    MarketScanner,
    MomentumScanner,
    BreakoutScanner,
    OversoldScanner,
    VolumeSpikeScanner,
    StockAnalyzer,
    ScanResult,
    ScanFilter,
    ScanType,
    quick_scan,
    scan_for_buys
)

from .alerts import (
    AlertManager,
    Alert,
    AlertType,
    AlertPriority,
    TelegramNotifier,
    EmailNotifier,
    DesktopNotifier,
    alert_manager,
    send_alert,
    notify_trade
)

from .portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioAnalyzer,
    CorrelationAnalyzer,
    ReturnCalculator,
    AllocationResult,
    PortfolioMetrics,
    OptimizationGoal,
    optimize_portfolio,
    analyze_portfolio
)

__all__ = [
    # Predictor (heuristic scoring, not ML)
    'MLPredictor',
    'HeuristicPredictor',
    'SimpleMLPredictor',  # Deprecated alias for HeuristicPredictor
    'FeatureEngineering',
    'SupportResistanceDetector',
    'TrendAnalyzer',
    'Prediction',
    'PredictionType',
    'SignalStrength',
    'quick_predict',
    'quick_analysis',

    # Market Scanner
    'MarketScanner',
    'MomentumScanner',
    'BreakoutScanner',
    'OversoldScanner',
    'VolumeSpikeScanner',
    'StockAnalyzer',
    'ScanResult',
    'ScanFilter',
    'ScanType',
    'quick_scan',
    'scan_for_buys',

    # Alerts
    'AlertManager',
    'Alert',
    'AlertType',
    'AlertPriority',
    'TelegramNotifier',
    'EmailNotifier',
    'DesktopNotifier',
    'alert_manager',
    'send_alert',
    'notify_trade',

    # Portfolio Optimizer
    'PortfolioOptimizer',
    'PortfolioAnalyzer',
    'CorrelationAnalyzer',
    'ReturnCalculator',
    'AllocationResult',
    'PortfolioMetrics',
    'OptimizationGoal',
    'optimize_portfolio',
    'analyze_portfolio',
]
