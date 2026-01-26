# -*- coding: utf-8 -*-
"""
Advanced Risk Metrics - Know Your Risk Numbers!
=================================================
Value at Risk (VaR), portfolio beta, and drawdown tracking.

VaR answers: "What's the maximum I could lose with X% confidence?"

⚠️  REQUIRES HISTORICAL DATA - NO DATA FETCHING BUILT-IN ⚠️

LIMITATIONS:
1. NO DATA SOURCE INTEGRATION: You must provide historical returns.
   This module does NOT fetch data from any source.

2. MINIMUM DATA REQUIREMENTS:
   - VaR: Needs 60+ days of returns for meaningful results (252 recommended)
   - Beta: Needs aligned portfolio and market returns (60+ days recommended)
   - Drawdown: Works with any length, but short periods are meaningless

3. WILL FAIL SILENTLY with insufficient data:
   - 1 data point → std=NaN, VaR=NaN
   - <30 points → statistically unreliable results

WHERE TO GET HISTORICAL DATA:
- NSE Bhavcopy (free, daily): https://www.nseindia.com/
- yfinance package: `pip install yfinance`
- Your broker's historical API (Zerodha Kite has 2000 candle limit)
- Commercial vendors: TrueData, GlobalDataFeeds

Example (YOU must provide the data):
    >>> import yfinance as yf  # Example data source
    >>>
    >>> # Fetch historical data (you need to do this yourself)
    >>> nifty = yf.download("^NSEI", period="1y")
    >>> returns = nifty['Close'].pct_change().dropna()
    >>>
    >>> # NOW you can use VaRCalculator
    >>> var_calc = VaRCalculator(returns, portfolio_value=1000000)
    >>> var_95 = var_calc.historical_var(confidence=0.95)
    >>> print(f"95% VaR: Rs.{var_95.var_amount:,.0f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE AT RISK (VaR)
# =============================================================================

@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    lookback_days: int = 252           # 1 year of trading days
    holding_period: int = 1            # Days to hold position
    decay_factor: float = 0.94         # For EWMA volatility (RiskMetrics)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_pct: float                     # VaR as percentage of portfolio
    var_amount: float                  # VaR in currency
    confidence: float                  # Confidence level (e.g., 0.95)
    method: str                        # Calculation method
    holding_period: int                # Days
    portfolio_value: float
    expected_shortfall: float = 0.0    # CVaR / Expected Shortfall
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"{self.method} VaR ({self.confidence:.0%}, {self.holding_period}d): "
            f"{self.var_pct:.2%} = Rs.{self.var_amount:,.0f}"
        )


class VaRCalculator:
    """
    Value at Risk Calculator.

    Supports multiple methods:
    - Historical VaR: Based on actual historical returns
    - Parametric VaR: Assumes normal distribution
    - Monte Carlo VaR: Simulation-based (optional)
    - Cornish-Fisher VaR: Adjusts for skewness and kurtosis

    Example:
        >>> returns = prices.pct_change().dropna()
        >>> var_calc = VaRCalculator(returns, portfolio_value=1000000)
        >>>
        >>> # Historical VaR
        >>> result = var_calc.historical_var(confidence=0.95)
        >>> print(f"95% 1-day VaR: Rs.{result.var_amount:,.0f}")
        >>>
        >>> # Compare methods
        >>> comparison = var_calc.compare_methods()
    """

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        portfolio_value: float = 100000,
        config: Optional[VaRConfig] = None
    ):
        """
        Initialize VaR calculator.

        Args:
            returns: Return series or DataFrame (columns = assets)
            portfolio_value: Current portfolio value
            config: VaR configuration
        """
        self.config = config or VaRConfig()
        self.portfolio_value = portfolio_value

        # Handle both Series and DataFrame
        if isinstance(returns, pd.DataFrame):
            # If DataFrame, assume equal-weighted or use first column
            if returns.shape[1] == 1:
                self.returns = returns.iloc[:, 0]
            else:
                # Equal-weighted portfolio returns
                self.returns = returns.mean(axis=1)
        else:
            self.returns = returns

        # Use lookback period
        if len(self.returns) > self.config.lookback_days:
            self.returns = self.returns.iloc[-self.config.lookback_days:]

        # VALIDATION: Check minimum data requirements
        min_required = 30  # Absolute minimum for any statistical significance
        if len(self.returns) < min_required:
            import warnings
            warnings.warn(
                f"VaRCalculator has only {len(self.returns)} data points. "
                f"Minimum {min_required} required for meaningful results, "
                f"252 recommended (1 year). Results will be unreliable!",
                UserWarning,
                stacklevel=2
            )

        if len(self.returns) < 2:
            raise ValueError(
                f"VaRCalculator requires at least 2 data points, got {len(self.returns)}. "
                "You need historical returns data - this module does NOT fetch data."
            )

        # Precompute statistics
        self._mean = self.returns.mean()
        self._std = self.returns.std()
        self._skew = self.returns.skew()
        self._kurtosis = self.returns.kurtosis()

    def historical_var(
        self,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate Historical VaR.

        Uses actual historical returns - no distribution assumption.
        Most common and intuitive method.

        Args:
            confidence: Confidence level (0.95 = 95%)
            holding_period: Days to hold position

        Returns:
            VaRResult with VaR metrics
        """
        # Scale for holding period (square root of time)
        scaled_returns = self.returns * np.sqrt(holding_period)

        # VaR is the percentile of the loss distribution
        alpha = 1 - confidence
        var_pct = -np.percentile(scaled_returns, alpha * 100)

        # Expected Shortfall (CVaR) - average of losses beyond VaR
        losses = -scaled_returns
        es_pct = losses[losses >= var_pct].mean()

        var_amount = var_pct * self.portfolio_value

        return VaRResult(
            var_pct=var_pct,
            var_amount=var_amount,
            confidence=confidence,
            method="Historical",
            holding_period=holding_period,
            portfolio_value=self.portfolio_value,
            expected_shortfall=es_pct * self.portfolio_value
        )

    def parametric_var(
        self,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        Assumes returns are normally distributed.
        Fast but may underestimate tail risk.

        Args:
            confidence: Confidence level
            holding_period: Days to hold

        Returns:
            VaRResult
        """
        # Z-score for confidence level
        z = stats.norm.ppf(confidence)

        # Scale for holding period
        scaled_std = self._std * np.sqrt(holding_period)
        scaled_mean = self._mean * holding_period

        # VaR = mean - z * std (we want loss, so flip sign)
        var_pct = z * scaled_std - scaled_mean
        var_amount = var_pct * self.portfolio_value

        # Expected Shortfall for normal distribution
        es_z = stats.norm.pdf(z) / (1 - confidence)
        es_pct = es_z * scaled_std - scaled_mean

        return VaRResult(
            var_pct=var_pct,
            var_amount=var_amount,
            confidence=confidence,
            method="Parametric",
            holding_period=holding_period,
            portfolio_value=self.portfolio_value,
            expected_shortfall=es_pct * self.portfolio_value
        )

    def cornish_fisher_var(
        self,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate Cornish-Fisher VaR.

        Adjusts for non-normality (skewness and kurtosis).
        Better for fat-tailed distributions.

        Args:
            confidence: Confidence level
            holding_period: Days to hold

        Returns:
            VaRResult
        """
        z = stats.norm.ppf(confidence)
        s = self._skew
        k = self._kurtosis

        # Cornish-Fisher expansion
        z_cf = (z +
                (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * k / 24 -
                (2*z**3 - 5*z) * s**2 / 36)

        # Scale for holding period
        scaled_std = self._std * np.sqrt(holding_period)
        scaled_mean = self._mean * holding_period

        var_pct = z_cf * scaled_std - scaled_mean
        var_amount = var_pct * self.portfolio_value

        return VaRResult(
            var_pct=var_pct,
            var_amount=var_amount,
            confidence=confidence,
            method="Cornish-Fisher",
            holding_period=holding_period,
            portfolio_value=self.portfolio_value,
            expected_shortfall=0.0  # Not easily calculated for CF
        )

    def ewma_var(
        self,
        confidence: float = 0.95,
        holding_period: int = 1,
        decay: Optional[float] = None
    ) -> VaRResult:
        """
        Calculate EWMA (Exponentially Weighted) VaR.

        Gives more weight to recent observations.
        RiskMetrics methodology with decay factor.

        Args:
            confidence: Confidence level
            holding_period: Days to hold
            decay: Decay factor (default: 0.94)

        Returns:
            VaRResult
        """
        decay = decay or self.config.decay_factor

        # EWMA variance
        weights = np.array([(1 - decay) * decay**i for i in range(len(self.returns))])
        weights = weights[::-1]  # Most recent gets highest weight
        weights = weights / weights.sum()

        ewma_var = np.sum(weights * (self.returns - self._mean)**2)
        ewma_std = np.sqrt(ewma_var)

        # Scale for holding period
        scaled_std = ewma_std * np.sqrt(holding_period)

        z = stats.norm.ppf(confidence)
        var_pct = z * scaled_std
        var_amount = var_pct * self.portfolio_value

        return VaRResult(
            var_pct=var_pct,
            var_amount=var_amount,
            confidence=confidence,
            method="EWMA",
            holding_period=holding_period,
            portfolio_value=self.portfolio_value,
            expected_shortfall=0.0
        )

    def monte_carlo_var(
        self,
        confidence: float = 0.95,
        holding_period: int = 1,
        simulations: int = 10000
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.

        Simulates many possible outcomes using historical parameters.
        Most flexible but computationally intensive.

        Args:
            confidence: Confidence level
            holding_period: Days to hold
            simulations: Number of Monte Carlo simulations

        Returns:
            VaRResult
        """
        # Simulate returns
        simulated = np.random.normal(
            self._mean * holding_period,
            self._std * np.sqrt(holding_period),
            simulations
        )

        # VaR from simulated distribution
        alpha = 1 - confidence
        var_pct = -np.percentile(simulated, alpha * 100)

        # Expected Shortfall
        losses = -simulated
        es_pct = losses[losses >= var_pct].mean()

        var_amount = var_pct * self.portfolio_value

        return VaRResult(
            var_pct=var_pct,
            var_amount=var_amount,
            confidence=confidence,
            method="Monte Carlo",
            holding_period=holding_period,
            portfolio_value=self.portfolio_value,
            expected_shortfall=es_pct * self.portfolio_value
        )

    def compare_methods(
        self,
        confidence: float = 0.95,
        holding_period: int = 1
    ) -> Dict[str, VaRResult]:
        """Compare all VaR methods."""
        return {
            'historical': self.historical_var(confidence, holding_period),
            'parametric': self.parametric_var(confidence, holding_period),
            'cornish_fisher': self.cornish_fisher_var(confidence, holding_period),
            'ewma': self.ewma_var(confidence, holding_period),
            'monte_carlo': self.monte_carlo_var(confidence, holding_period)
        }

    def summary(self, confidence: float = 0.95) -> str:
        """Generate VaR summary report."""
        results = self.compare_methods(confidence)

        lines = [
            "=" * 60,
            f"VALUE AT RISK REPORT ({confidence:.0%} Confidence)",
            "=" * 60,
            f"Portfolio Value: Rs.{self.portfolio_value:,.0f}",
            f"Returns: {len(self.returns)} observations",
            f"Mean Daily Return: {self._mean:.4%}",
            f"Daily Volatility: {self._std:.4%}",
            f"Annualized Vol: {self._std * np.sqrt(252):.2%}",
            f"Skewness: {self._skew:.2f}",
            f"Kurtosis: {self._kurtosis:.2f}",
            "",
            "1-DAY VaR BY METHOD:",
            "-" * 40,
        ]

        for name, result in results.items():
            lines.append(f"  {result}")

        lines.extend([
            "",
            f"Average VaR: Rs.{np.mean([r.var_amount for r in results.values()]):,.0f}",
            "=" * 60
        ])

        return "\n".join(lines)


# =============================================================================
# PORTFOLIO BETA
# =============================================================================

@dataclass
class BetaResult:
    """Portfolio beta calculation result."""
    beta: float
    alpha: float                       # Jensen's alpha
    r_squared: float                   # Explained variance
    correlation: float                 # Correlation with market
    tracking_error: float              # Volatility of excess returns
    information_ratio: float           # Risk-adjusted excess return
    lookback_days: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"Beta: {self.beta:.2f}, Alpha: {self.alpha:.4%}, R²: {self.r_squared:.2%}"


class BetaCalculator:
    """
    Portfolio Beta Calculator.

    Beta measures systematic risk relative to the market.
    - Beta = 1: Moves with market
    - Beta > 1: More volatile than market
    - Beta < 1: Less volatile than market
    - Beta < 0: Moves opposite to market

    Example:
        >>> calc = BetaCalculator(portfolio_returns, nifty_returns)
        >>> result = calc.calculate()
        >>> print(f"Portfolio Beta: {result.beta:.2f}")
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.06  # Annual risk-free rate
    ):
        """
        Initialize beta calculator.

        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market (benchmark) return series
            risk_free_rate: Annual risk-free rate (default 6%)
        """
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()

        self.portfolio_returns = aligned['portfolio']
        self.market_returns = aligned['market']
        self.rf_daily = risk_free_rate / 252

    def calculate(self) -> BetaResult:
        """Calculate beta and related metrics."""
        # Excess returns
        port_excess = self.portfolio_returns - self.rf_daily
        mkt_excess = self.market_returns - self.rf_daily

        # Beta = Cov(Rp, Rm) / Var(Rm)
        covariance = np.cov(port_excess, mkt_excess)[0, 1]
        market_variance = np.var(mkt_excess)
        beta = covariance / market_variance if market_variance > 0 else 0

        # Alpha = Rp - Rf - Beta * (Rm - Rf)
        alpha = port_excess.mean() - beta * mkt_excess.mean()

        # Correlation
        correlation = np.corrcoef(self.portfolio_returns, self.market_returns)[0, 1]

        # R-squared
        r_squared = correlation ** 2

        # Tracking error (std of excess returns vs market)
        tracking_diff = self.portfolio_returns - self.market_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)  # Annualized

        # Information ratio
        avg_excess = tracking_diff.mean() * 252  # Annualized
        information_ratio = avg_excess / tracking_error if tracking_error > 0 else 0

        return BetaResult(
            beta=beta,
            alpha=alpha * 252,  # Annualized
            r_squared=r_squared,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            lookback_days=len(self.portfolio_returns)
        )

    def rolling_beta(self, window: int = 60) -> pd.Series:
        """Calculate rolling beta."""
        betas = []
        for i in range(window, len(self.portfolio_returns) + 1):
            port = self.portfolio_returns.iloc[i-window:i]
            mkt = self.market_returns.iloc[i-window:i]

            cov = np.cov(port, mkt)[0, 1]
            var = np.var(mkt)
            betas.append(cov / var if var > 0 else 0)

        return pd.Series(
            betas,
            index=self.portfolio_returns.index[window-1:]
        )


def calculate_portfolio_beta(
    position_returns: Dict[str, pd.Series],
    position_weights: Dict[str, float],
    market_returns: pd.Series
) -> BetaResult:
    """
    Calculate weighted portfolio beta.

    Args:
        position_returns: {symbol: return_series}
        position_weights: {symbol: weight}
        market_returns: Market benchmark returns

    Returns:
        BetaResult for the portfolio
    """
    # Calculate weighted portfolio returns
    portfolio_returns = None

    for symbol, weight in position_weights.items():
        if symbol not in position_returns:
            continue

        returns = position_returns[symbol] * weight

        if portfolio_returns is None:
            portfolio_returns = returns
        else:
            portfolio_returns = portfolio_returns.add(returns, fill_value=0)

    if portfolio_returns is None:
        return BetaResult(
            beta=0, alpha=0, r_squared=0, correlation=0,
            tracking_error=0, information_ratio=0, lookback_days=0
        )

    calc = BetaCalculator(portfolio_returns, market_returns)
    return calc.calculate()


# =============================================================================
# DRAWDOWN TRACKING
# =============================================================================

@dataclass
class DrawdownInfo:
    """Information about a specific drawdown."""
    start_date: datetime
    end_date: Optional[datetime]
    trough_date: datetime
    peak_value: float
    trough_value: float
    drawdown_pct: float
    recovery_date: Optional[datetime] = None
    duration_days: int = 0
    recovery_days: int = 0
    is_recovered: bool = False

    @property
    def is_active(self) -> bool:
        """Check if drawdown is still active."""
        return not self.is_recovered


@dataclass
class DrawdownMetrics:
    """Complete drawdown metrics for a portfolio."""
    max_drawdown: float
    max_drawdown_date: datetime
    current_drawdown: float
    avg_drawdown: float
    drawdown_count: int               # Number of >5% drawdowns
    avg_recovery_days: float
    longest_drawdown_days: int
    all_drawdowns: List[DrawdownInfo] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate drawdown summary."""
        lines = [
            "DRAWDOWN METRICS",
            "-" * 40,
            f"Maximum Drawdown: {self.max_drawdown:.2%}",
            f"Current Drawdown: {self.current_drawdown:.2%}",
            f"Average Drawdown: {self.avg_drawdown:.2%}",
            f"Drawdowns (>5%): {self.drawdown_count}",
            f"Avg Recovery: {self.avg_recovery_days:.0f} days",
            f"Longest Drawdown: {self.longest_drawdown_days} days",
        ]
        return "\n".join(lines)


class DrawdownTracker:
    """
    Track and analyze portfolio drawdowns.

    Monitors:
    - Current drawdown from peak
    - Historical drawdowns
    - Recovery periods
    - Underwater periods

    Example:
        >>> tracker = DrawdownTracker(equity_curve)
        >>> metrics = tracker.calculate_metrics()
        >>> print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        >>>
        >>> # Get current status
        >>> if tracker.is_in_drawdown():
        ...     print(f"Currently down {tracker.current_drawdown:.2%}")
    """

    def __init__(
        self,
        equity: Union[pd.Series, List[float], np.ndarray],
        dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
        threshold: float = 0.05  # Minimum drawdown to track
    ):
        """
        Initialize drawdown tracker.

        Args:
            equity: Equity curve (portfolio values)
            dates: Corresponding dates (optional)
            threshold: Minimum drawdown percentage to track
        """
        if isinstance(equity, pd.Series):
            self.equity = equity.values
            self.dates = equity.index if isinstance(equity.index, pd.DatetimeIndex) else None
        else:
            self.equity = np.array(equity)
            self.dates = dates

        if self.dates is None:
            self.dates = pd.date_range(
                start=datetime.now() - timedelta(days=len(self.equity)),
                periods=len(self.equity),
                freq='D'
            )

        self.threshold = threshold

        # Calculate drawdown series
        self._calculate_drawdowns()

    def _calculate_drawdowns(self) -> None:
        """Calculate drawdown time series."""
        # Running maximum
        self.peak = np.maximum.accumulate(self.equity)

        # Drawdown percentage
        self.drawdown = (self.peak - self.equity) / self.peak

        # Max drawdown
        self.max_drawdown = np.max(self.drawdown)
        self.max_drawdown_idx = np.argmax(self.drawdown)

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        return self.drawdown[-1]

    def is_in_drawdown(self, threshold: Optional[float] = None) -> bool:
        """Check if currently in drawdown."""
        threshold = threshold or self.threshold
        return self.current_drawdown >= threshold

    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown as time series."""
        return pd.Series(self.drawdown, index=self.dates)

    def get_underwater_curve(self) -> pd.Series:
        """Get underwater curve (drawdown over time)."""
        return -self.get_drawdown_series()

    def identify_drawdowns(self) -> List[DrawdownInfo]:
        """Identify all significant drawdowns."""
        drawdowns = []
        in_drawdown = False
        current_dd = None

        for i in range(len(self.equity)):
            dd = self.drawdown[i]
            date = self.dates[i]

            if dd >= self.threshold and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                current_dd = DrawdownInfo(
                    start_date=date,
                    end_date=None,
                    trough_date=date,
                    peak_value=self.peak[i],
                    trough_value=self.equity[i],
                    drawdown_pct=dd
                )

            elif in_drawdown:
                if dd > current_dd.drawdown_pct:
                    # New trough
                    current_dd.trough_date = date
                    current_dd.trough_value = self.equity[i]
                    current_dd.drawdown_pct = dd

                if dd < self.threshold / 2:
                    # Recovered (drawdown < half threshold)
                    current_dd.end_date = date
                    current_dd.recovery_date = date
                    current_dd.is_recovered = True
                    current_dd.duration_days = (date - current_dd.start_date).days
                    current_dd.recovery_days = (date - current_dd.trough_date).days
                    drawdowns.append(current_dd)
                    in_drawdown = False
                    current_dd = None

        # Handle ongoing drawdown
        if in_drawdown and current_dd:
            current_dd.duration_days = (self.dates[-1] - current_dd.start_date).days
            drawdowns.append(current_dd)

        return drawdowns

    def calculate_metrics(self) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics."""
        drawdowns = self.identify_drawdowns()

        if not drawdowns:
            return DrawdownMetrics(
                max_drawdown=self.max_drawdown,
                max_drawdown_date=self.dates[self.max_drawdown_idx],
                current_drawdown=self.current_drawdown,
                avg_drawdown=0.0,
                drawdown_count=0,
                avg_recovery_days=0.0,
                longest_drawdown_days=0,
                all_drawdowns=[]
            )

        recovered = [d for d in drawdowns if d.is_recovered]

        return DrawdownMetrics(
            max_drawdown=self.max_drawdown,
            max_drawdown_date=self.dates[self.max_drawdown_idx],
            current_drawdown=self.current_drawdown,
            avg_drawdown=np.mean([d.drawdown_pct for d in drawdowns]),
            drawdown_count=len(drawdowns),
            avg_recovery_days=np.mean([d.recovery_days for d in recovered]) if recovered else 0,
            longest_drawdown_days=max(d.duration_days for d in drawdowns),
            all_drawdowns=drawdowns
        )

    def get_calmar_ratio(self, annual_return: float) -> float:
        """
        Calculate Calmar ratio.

        Calmar = Annual Return / Max Drawdown
        Higher is better.
        """
        if self.max_drawdown == 0:
            return 0.0
        return annual_return / self.max_drawdown

    def get_ulcer_index(self) -> float:
        """
        Calculate Ulcer Index.

        Measures downside risk - RMS of drawdown.
        Lower is better.
        """
        return np.sqrt(np.mean(self.drawdown ** 2))

    def get_pain_index(self) -> float:
        """
        Calculate Pain Index.

        Mean of all drawdown values.
        Lower is better.
        """
        return np.mean(self.drawdown)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_var(
    returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    method: str = "historical"
) -> VaRResult:
    """Quick VaR calculation."""
    calc = VaRCalculator(returns, portfolio_value)

    if method == "historical":
        return calc.historical_var(confidence)
    elif method == "parametric":
        return calc.parametric_var(confidence)
    elif method == "monte_carlo":
        return calc.monte_carlo_var(confidence)
    else:
        return calc.historical_var(confidence)


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Quick max drawdown calculation."""
    tracker = DrawdownTracker(equity)
    return tracker.max_drawdown


def calculate_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """Quick beta calculation."""
    calc = BetaCalculator(portfolio_returns, market_returns)
    return calc.calculate().beta
