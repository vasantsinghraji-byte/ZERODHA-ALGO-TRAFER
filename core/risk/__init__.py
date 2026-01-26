# -*- coding: utf-8 -*-
"""
Portfolio Risk Management Module - Don't Put All Eggs in One Basket!
=====================================================================
Advanced risk controls for portfolio-level risk management.

Components:
- CorrelationAnalyzer: Calculate and monitor asset correlations
- PortfolioRiskManager: Portfolio-wide risk limits and checks
- VaRCalculator: Value at Risk calculations
- BetaCalculator: Portfolio beta relative to market
- DrawdownTracker: Monitor and analyze drawdowns

Example:
    >>> from core.risk import CorrelationAnalyzer, PortfolioRiskManager
    >>>
    >>> # Analyze correlations
    >>> analyzer = CorrelationAnalyzer()
    >>> matrix = analyzer.calculate_correlation_matrix(price_data)
    >>> print(matrix.get_high_correlations(threshold=0.7))
    >>>
    >>> # Value at Risk
    >>> from core.risk import VaRCalculator
    >>> var_calc = VaRCalculator(returns, portfolio_value=1000000)
    >>> print(var_calc.historical_var(confidence=0.95))
"""

from .correlation import (
    CorrelationAnalyzer,
    CorrelationMatrix,
    CorrelationConfig,
    RollingCorrelation,
    calculate_correlation,
    calculate_rolling_correlation,
    find_high_correlations,
)

from .portfolio import (
    PortfolioRiskManager,
    PortfolioRiskConfig,
    SectorExposure,
    PositionRisk,
    PortfolioRiskReport,
    check_correlation_risk,
    check_sector_exposure,
)

from .var import (
    # Value at Risk
    VaRCalculator,
    VaRConfig,
    VaRResult,
    calculate_var,
    # Beta
    BetaCalculator,
    BetaResult,
    calculate_portfolio_beta,
    calculate_beta,
    # Drawdown
    DrawdownTracker,
    DrawdownInfo,
    DrawdownMetrics,
    calculate_max_drawdown,
)

__all__ = [
    # Correlation
    'CorrelationAnalyzer',
    'CorrelationMatrix',
    'CorrelationConfig',
    'RollingCorrelation',
    'calculate_correlation',
    'calculate_rolling_correlation',
    'find_high_correlations',
    # Portfolio Risk
    'PortfolioRiskManager',
    'PortfolioRiskConfig',
    'SectorExposure',
    'PositionRisk',
    'PortfolioRiskReport',
    'check_correlation_risk',
    'check_sector_exposure',
    # Value at Risk
    'VaRCalculator',
    'VaRConfig',
    'VaRResult',
    'calculate_var',
    # Beta
    'BetaCalculator',
    'BetaResult',
    'calculate_portfolio_beta',
    'calculate_beta',
    # Drawdown
    'DrawdownTracker',
    'DrawdownInfo',
    'DrawdownMetrics',
    'calculate_max_drawdown',
]
