# -*- coding: utf-8 -*-
"""
Portfolio Optimization - Smart Money Allocation!
=================================================
Optimizes how to spread money across stocks.

Like a chef balancing ingredients for the perfect dish!

Features:
- Modern Portfolio Theory (MPT)
- Risk-adjusted returns optimization
- Diversification analysis
- Rebalancing recommendations
- Correlation analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# Import Money for precise monetary calculations
try:
    from utils.money import Money
    HAS_MONEY = True
except ImportError:
    HAS_MONEY = False

logger = logging.getLogger(__name__)


def _to_decimal(value: Union[float, int, Decimal], precision: str = "0.01") -> Decimal:
    """Convert value to Decimal with specified precision for monetary calculations."""
    if isinstance(value, Decimal):
        return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal(precision), rounding=ROUND_HALF_UP)


class OptimizationGoal(Enum):
    """Portfolio optimization goals"""
    MAX_SHARPE = "max_sharpe"           # Maximize risk-adjusted returns
    MIN_VOLATILITY = "min_volatility"   # Minimize risk
    MAX_RETURN = "max_return"           # Maximize returns
    TARGET_RETURN = "target_return"     # Achieve specific return
    EQUAL_WEIGHT = "equal_weight"       # Equal allocation
    RISK_PARITY = "risk_parity"         # Equal risk contribution


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    expected_return: float      # Annual expected return
    volatility: float           # Annual volatility (std dev)
    sharpe_ratio: float         # Risk-adjusted return
    sortino_ratio: float        # Downside risk-adjusted
    max_drawdown: float         # Worst peak-to-trough
    var_95: float               # Value at Risk (95%)
    diversification_ratio: float
    correlation_avg: float


@dataclass
class AllocationResult:
    """Result of portfolio optimization"""
    weights: Dict[str, float]           # Symbol -> weight (0-1)
    metrics: PortfolioMetrics
    goal: OptimizationGoal
    timestamp: datetime = field(default_factory=datetime.now)

    def get_allocation(self, capital: float) -> Dict[str, Decimal]:
        """
        Get allocation in rupees using precise Decimal arithmetic.

        Prevents floating-point errors that could cause:
        - Vanishing pennies over many calculations
        - Invalid allocation totals that don't sum to capital

        Args:
            capital: Total capital to allocate

        Returns:
            Dict of symbol -> allocation amount (Decimal)
        """
        capital_decimal = _to_decimal(capital)
        return {
            symbol: _to_decimal(weight * float(capital_decimal))
            for symbol, weight in self.weights.items()
        }

    def get_shares(self, capital: float, prices: Dict[str, float]) -> Dict[str, int]:
        """
        Get number of shares to buy using precise arithmetic.

        Uses Decimal division to avoid floating-point errors that could cause
        incorrect share counts or broker rejection due to invalid quantities.

        Args:
            capital: Total capital to allocate
            prices: Current prices for each symbol

        Returns:
            Dict of symbol -> share count (int)
        """
        allocation = self.get_allocation(capital)
        result = {}
        for symbol, amount in allocation.items():
            price = prices.get(symbol, 1)
            if price <= 0:
                result[symbol] = 0
                continue
            # Use Decimal division for precision
            price_decimal = _to_decimal(price)
            shares = int(amount / price_decimal)
            result[symbol] = shares
        return result

    def print_summary(self):
        """Print allocation summary"""
        print("\n" + "=" * 50)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 50)

        print(f"\nGoal: {self.goal.value}")
        print(f"\n📊 Allocation:")
        for symbol, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 40)
            print(f"  {symbol:12} {weight:6.1%} {bar}")

        print(f"\n📈 Expected Metrics:")
        print(f"  Annual Return:     {self.metrics.expected_return:6.1%}")
        print(f"  Annual Volatility: {self.metrics.volatility:6.1%}")
        print(f"  Sharpe Ratio:      {self.metrics.sharpe_ratio:6.2f}")
        print(f"  Max Drawdown:      {self.metrics.max_drawdown:6.1%}")
        print(f"  VaR (95%):         {self.metrics.var_95:6.1%}")

        print("=" * 50)


class ReturnCalculator:
    """Calculate returns and statistics from price data"""

    @staticmethod
    def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return prices.pct_change().dropna()

    # Minimum days required for reliable annualization
    MIN_DAYS_FOR_COMPOUND = 30  # Use compound extrapolation only with 30+ days
    MIN_DAYS_FOR_CALC = 5       # Require at least 5 days of data

    @staticmethod
    def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
        """
        Calculate annualized return with safeguards against unrealistic extrapolation.

        For short periods (< 30 days): Uses mean daily return × 252 (simple annualization)
        For longer periods (>= 30 days): Uses compound return extrapolation

        This prevents unrealistic results like 1127% annual return from 1 day of 1% gain.
        """
        n_days = len(returns)

        # Require minimum data for any calculation
        if n_days < ReturnCalculator.MIN_DAYS_FOR_CALC:
            logger.warning(
                f"Insufficient data for annualization: {n_days} days < {ReturnCalculator.MIN_DAYS_FOR_CALC} minimum"
            )
            return 0.0

        # For short periods: use simple average daily return annualization
        # This is more conservative and realistic than compound extrapolation
        if n_days < ReturnCalculator.MIN_DAYS_FOR_COMPOUND:
            mean_daily_return = returns.mean()
            return mean_daily_return * trading_days

        # For longer periods: use compound return extrapolation (original method)
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (trading_days / n_days) - 1

    @staticmethod
    def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """
        Calculate annualized volatility.

        Requires minimum sample size for reliable estimate.
        """
        n_days = len(returns)
        if n_days < ReturnCalculator.MIN_DAYS_FOR_CALC:
            logger.warning(
                f"Insufficient data for volatility: {n_days} days < {ReturnCalculator.MIN_DAYS_FOR_CALC} minimum"
            )
            return 0.0

        return returns.std() * np.sqrt(trading_days)

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        ann_return = ReturnCalculator.annualized_return(returns)
        ann_vol = ReturnCalculator.annualized_volatility(returns)
        if ann_vol == 0:
            return 0
        return (ann_return - risk_free_rate) / ann_vol

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        ann_return = ReturnCalculator.annualized_return(returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float('inf')
        downside_vol = downside.std() * np.sqrt(252)
        if downside_vol == 0:
            return float('inf')
        return (ann_return - risk_free_rate) / downside_vol

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def var_95(returns: pd.Series) -> float:
        """Calculate Value at Risk at 95% confidence"""
        return returns.quantile(0.05)


class CorrelationAnalyzer:
    """Analyze correlations between assets"""

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return returns.corr()

    @staticmethod
    def find_diversifiers(returns: pd.DataFrame, threshold: float = 0.3) -> List[Tuple[str, str]]:
        """Find pairs of assets with low correlation"""
        corr = returns.corr()
        pairs = []

        symbols = corr.columns.tolist()
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if abs(corr.loc[sym1, sym2]) < threshold:
                    pairs.append((sym1, sym2, corr.loc[sym1, sym2]))

        return sorted(pairs, key=lambda x: abs(x[2]))

    @staticmethod
    def find_highly_correlated(returns: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str]]:
        """Find pairs of highly correlated assets (redundant)"""
        corr = returns.corr()
        pairs = []

        symbols = corr.columns.tolist()
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if corr.loc[sym1, sym2] > threshold:
                    pairs.append((sym1, sym2, corr.loc[sym1, sym2]))

        return sorted(pairs, key=lambda x: x[2], reverse=True)

    @staticmethod
    def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        # Portfolio volatility
        port_vol = np.sqrt(weights @ cov_matrix @ weights)

        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = weights @ individual_vols

        if port_vol == 0:
            return 1.0

        return weighted_avg_vol / port_vol


class PortfolioOptimizer:
    """
    Main portfolio optimizer using Modern Portfolio Theory.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default 5% for India)
        """
        self.risk_free_rate = risk_free_rate
        self.returns: Optional[pd.DataFrame] = None
        self.symbols: List[str] = []
        self.n_assets: int = 0

    def load_data(self, price_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Load price data for optimization.

        Args:
            price_data: Dict of symbol -> DataFrame with 'close' column

        Returns:
            True if successful
        """
        try:
            # Extract closing prices
            prices = pd.DataFrame()

            for symbol, df in price_data.items():
                col = 'close' if 'close' in df.columns else 'Close'
                if col in df.columns:
                    prices[symbol] = df[col]

            if prices.empty:
                logger.error("No price data found")
                return False

            # Calculate returns
            self.returns = ReturnCalculator.daily_returns(prices)
            self.symbols = list(self.returns.columns)
            self.n_assets = len(self.symbols)

            logger.info(f"Loaded data for {self.n_assets} assets")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> PortfolioMetrics:
        """Calculate metrics for a given weight allocation"""

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Expected return and volatility
        expected_return = ReturnCalculator.annualized_return(portfolio_returns)
        volatility = ReturnCalculator.annualized_volatility(portfolio_returns)

        # Risk metrics
        sharpe = ReturnCalculator.sharpe_ratio(portfolio_returns, self.risk_free_rate)
        sortino = ReturnCalculator.sortino_ratio(portfolio_returns, self.risk_free_rate)
        max_dd = ReturnCalculator.max_drawdown(portfolio_returns)
        var_95 = ReturnCalculator.var_95(portfolio_returns)

        # Diversification
        cov_matrix = returns.cov().values * 252
        div_ratio = CorrelationAnalyzer.diversification_ratio(weights, cov_matrix)
        corr_avg = returns.corr().values[np.triu_indices(len(weights), k=1)].mean()

        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            diversification_ratio=div_ratio,
            correlation_avg=corr_avg
        )

    def _validate_constraints(
        self,
        min_weight: float,
        max_weight: float
    ) -> Tuple[bool, str]:
        """
        Validate that weight constraints are mathematically feasible.

        Returns:
            (is_valid, error_message)
        """
        # Basic sanity checks
        if min_weight < 0 or max_weight > 1:
            return False, "Weights must be in range [0, 1]"

        if min_weight > max_weight:
            return False, f"min_weight ({min_weight}) > max_weight ({max_weight})"

        # CRITICAL: Check mathematical feasibility
        # Sum of min weights must be <= 1.0
        min_sum = min_weight * self.n_assets
        if min_sum > 1.0:
            return False, (
                f"Impossible constraints: {self.n_assets} assets × {min_weight:.1%} min = "
                f"{min_sum:.1%} > 100%. Reduce min_weight to {1.0/self.n_assets:.2%} or less."
            )

        # Sum of max weights must be >= 1.0
        max_sum = max_weight * self.n_assets
        if max_sum < 1.0:
            return False, (
                f"Impossible constraints: {self.n_assets} assets × {max_weight:.1%} max = "
                f"{max_sum:.1%} < 100%. Increase max_weight to {1.0/self.n_assets:.2%} or more."
            )

        return True, ""

    def optimize(
        self,
        goal: OptimizationGoal = OptimizationGoal.MAX_SHARPE,
        target_return: float = None,
        max_weight: float = 0.4,
        min_weight: float = 0.0,
        n_simulations: int = 10000
    ) -> AllocationResult:
        """
        Optimize portfolio allocation.

        Args:
            goal: Optimization goal
            target_return: Target return (for TARGET_RETURN goal)
            max_weight: Maximum weight per asset (0-1)
            min_weight: Minimum weight per asset (0-1)
            n_simulations: Number of random portfolios to simulate

        Returns:
            Optimized allocation

        Raises:
            ValueError: If constraints are mathematically impossible
        """
        if self.returns is None or self.n_assets == 0:
            raise ValueError("No data loaded. Call load_data() first.")

        # CRITICAL: Validate constraints BEFORE optimization
        is_valid, error_msg = self._validate_constraints(min_weight, max_weight)
        if not is_valid:
            raise ValueError(f"Invalid constraints: {error_msg}")

        if goal == OptimizationGoal.EQUAL_WEIGHT:
            weights = np.array([1.0 / self.n_assets] * self.n_assets)
        elif goal == OptimizationGoal.RISK_PARITY:
            weights = self._risk_parity_weights()
        else:
            weights = self._monte_carlo_optimization(
                goal, target_return, max_weight, min_weight, n_simulations
            )

        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(weights, self.returns)

        # Create result
        weight_dict = {symbol: float(w) for symbol, w in zip(self.symbols, weights)}

        return AllocationResult(
            weights=weight_dict,
            metrics=metrics,
            goal=goal
        )

    def _generate_constrained_weights(
        self,
        min_weight: float,
        max_weight: float
    ) -> np.ndarray:
        """
        Generate random weights that satisfy min/max constraints.

        Uses Dirichlet distribution with rejection sampling to ensure
        weights sum to 1.0 AND satisfy all constraints.

        Returns:
            Valid weight array or None if generation fails
        """
        # Method 1: Direct constrained generation
        # Start with minimum allocation, distribute remaining randomly

        # Allocate minimum to each asset
        weights = np.full(self.n_assets, min_weight)
        remaining = 1.0 - (min_weight * self.n_assets)

        if remaining < 0:
            # Should not happen if _validate_constraints passed
            return None

        if remaining > 0:
            # Distribute remaining weight randomly
            # But respect max_weight constraint
            max_additional = max_weight - min_weight

            for _ in range(100):  # Max attempts
                additional = np.random.random(self.n_assets)
                additional = additional / additional.sum() * remaining

                # Check if any exceeds max additional
                if np.all(additional <= max_additional + 1e-9):
                    weights = weights + additional
                    break
            else:
                # Fallback: uniform additional distribution
                weights = weights + (remaining / self.n_assets)

        return weights

    def _weights_satisfy_constraints(
        self,
        weights: np.ndarray,
        min_weight: float,
        max_weight: float,
        tolerance: float = 1e-6
    ) -> bool:
        """Check if weights satisfy all constraints."""
        # Check bounds
        if np.any(weights < min_weight - tolerance):
            return False
        if np.any(weights > max_weight + tolerance):
            return False
        # Check sum to 1
        if abs(weights.sum() - 1.0) > tolerance:
            return False
        return True

    def _monte_carlo_optimization(
        self,
        goal: OptimizationGoal,
        target_return: float,
        max_weight: float,
        min_weight: float,
        n_simulations: int
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation to find optimal weights.

        FIXED: Properly generates constrained weights instead of naive
        clip-then-normalize which violates constraints.

        Uses two strategies:
        1. Constrained generation: Generate weights that already satisfy bounds
        2. Rejection sampling: Filter out invalid portfolios
        """
        best_weights = None
        best_score = float('-inf') if goal != OptimizationGoal.MIN_VOLATILITY else float('inf')
        valid_portfolios_found = 0

        for i in range(n_simulations):
            # Strategy 1: Use constrained weight generation
            if i % 2 == 0:
                weights = self._generate_constrained_weights(min_weight, max_weight)
            else:
                # Strategy 2: Random + rejection sampling
                weights = np.random.random(self.n_assets)
                weights = weights / weights.sum()

            if weights is None:
                continue

            # CRITICAL: Verify constraints BEFORE accepting
            # Never use clip-then-normalize which violates constraints!
            if not self._weights_satisfy_constraints(weights, min_weight, max_weight):
                continue  # Skip invalid portfolio

            valid_portfolios_found += 1

            # Calculate portfolio returns
            portfolio_returns = (self.returns * weights).sum(axis=1)
            exp_return = ReturnCalculator.annualized_return(portfolio_returns)
            volatility = ReturnCalculator.annualized_volatility(portfolio_returns)

            # Score based on goal
            if goal == OptimizationGoal.MAX_SHARPE:
                if volatility > 0:
                    score = (exp_return - self.risk_free_rate) / volatility
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()

            elif goal == OptimizationGoal.MIN_VOLATILITY:
                score = volatility
                if score < best_score:
                    best_score = score
                    best_weights = weights.copy()

            elif goal == OptimizationGoal.MAX_RETURN:
                score = exp_return
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()

            elif goal == OptimizationGoal.TARGET_RETURN:
                if target_return is not None:
                    if abs(exp_return - target_return) < 0.02:  # Within 2%
                        score = -volatility  # Minimize vol at target return
                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()

        # Log how many valid portfolios were found
        if valid_portfolios_found < n_simulations * 0.1:
            logger.warning(
                f"Only {valid_portfolios_found}/{n_simulations} valid portfolios found. "
                f"Consider relaxing constraints."
            )

        if best_weights is None:
            # Fallback: Generate a guaranteed valid portfolio
            logger.warning("No optimal portfolio found, using constrained equal weight")
            best_weights = self._generate_constrained_weights(min_weight, max_weight)

            if best_weights is None:
                # Ultimate fallback
                best_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Final verification
        assert self._weights_satisfy_constraints(best_weights, min_weight, max_weight), \
            f"Bug: Final weights violate constraints! weights={best_weights}"

        return best_weights

    def _risk_parity_weights(self, max_iterations: int = 100, tolerance: float = 1e-8) -> np.ndarray:
        """
        Calculate true Equal Risk Contribution (ERC) weights.

        Unlike naive inverse-volatility weighting, this accounts for CORRELATIONS
        between assets. Each asset contributes equally to total portfolio risk.

        Uses the Spinu (2013) iterative algorithm for efficiency.

        Risk contribution for asset i: RC_i = w_i * (Σw)_i / σ_portfolio
        Goal: RC_i = RC_j for all i, j (equal risk contribution)
        """
        # Covariance matrix (annualized)
        cov_matrix = self.returns.cov().values * 252

        # Handle edge case: single asset
        if self.n_assets == 1:
            return np.array([1.0])

        # Initialize with inverse volatility (good starting point)
        volatilities = np.sqrt(np.diag(cov_matrix))
        volatilities = np.where(volatilities == 0, 1e-10, volatilities)  # Avoid div by zero
        weights = 1.0 / volatilities
        weights = weights / weights.sum()

        # Iterative optimization for true ERC
        # Using the Spinu (2013) closed-form iteration:
        # w_new = w * (1/λ) * (Σw / (w'Σw))^(-1) where we want equal marginal risk
        for iteration in range(max_iterations):
            # Portfolio variance and risk
            portfolio_var = weights @ cov_matrix @ weights
            portfolio_risk = np.sqrt(portfolio_var)

            if portfolio_risk < 1e-10:
                logger.warning("Portfolio risk near zero, using equal weights")
                return np.ones(self.n_assets) / self.n_assets

            # Marginal risk contribution: ∂σ/∂w = Σw / σ
            marginal_risk = cov_matrix @ weights / portfolio_risk

            # Risk contribution: RC_i = w_i * marginal_risk_i
            risk_contributions = weights * marginal_risk

            # Target: equal risk contribution = portfolio_risk / n_assets
            target_rc = portfolio_risk / self.n_assets

            # Check convergence: all RCs should be equal
            rc_deviation = np.max(np.abs(risk_contributions - target_rc))
            if rc_deviation < tolerance:
                logger.debug(f"ERC converged in {iteration + 1} iterations")
                break

            # Update weights using Newton-like step
            # New weight proportional to sqrt(target_rc / marginal_risk)
            # This ensures assets with higher marginal risk get lower weights
            marginal_risk = np.where(marginal_risk < 1e-10, 1e-10, marginal_risk)
            weights_new = np.sqrt(target_rc / marginal_risk)
            weights_new = weights_new / weights_new.sum()

            # Damped update for stability
            weights = 0.5 * weights + 0.5 * weights_new

        else:
            logger.warning(f"ERC did not converge after {max_iterations} iterations, "
                          f"max RC deviation: {rc_deviation:.6f}")

        # Final normalization
        weights = weights / weights.sum()

        # Log risk contributions for verification
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_risk = np.sqrt(portfolio_var)
        if portfolio_risk > 1e-10:
            marginal_risk = cov_matrix @ weights / portfolio_risk
            risk_contributions = weights * marginal_risk
            rc_pct = risk_contributions / risk_contributions.sum() * 100
            logger.debug(f"Risk contributions (%): {rc_pct.round(1)}")

        return weights

    def efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 0.4
    ) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Calculate the efficient frontier.

        Returns:
            List of (return, volatility, weights) tuples
        """
        frontier = []

        # Get range of possible returns
        individual_returns = [
            ReturnCalculator.annualized_return(self.returns[s])
            for s in self.symbols
        ]
        min_return = min(individual_returns)
        max_return = max(individual_returns)

        target_returns = np.linspace(min_return, max_return, n_points)

        for target in target_returns:
            result = self.optimize(
                goal=OptimizationGoal.TARGET_RETURN,
                target_return=target,
                max_weight=max_weight,
                n_simulations=5000
            )

            frontier.append((
                result.metrics.expected_return,
                result.metrics.volatility,
                result.weights
            ))

        return frontier

    def rebalance_recommendations(
        self,
        current_holdings: Dict[str, float],
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get rebalancing recommendations using precise Decimal arithmetic.

        Prevents floating-point errors that could cause:
        - Incorrect share counts leading to broker rejection
        - Accumulated errors in portfolio value calculations
        - Invalid tick sizes for order prices

        Args:
            current_holdings: Symbol -> current value
            target_allocation: Symbol -> target weight (0-1)
            current_prices: Symbol -> current price
            threshold: Rebalance if drift > threshold

        Returns:
            Rebalancing instructions with precise monetary values
        """
        # Convert to Decimal for precise calculations
        total_value = _to_decimal(sum(current_holdings.values()))
        if total_value == 0:
            return {}

        recommendations = {}

        for symbol in set(list(current_holdings.keys()) + list(target_allocation.keys())):
            current_value = _to_decimal(current_holdings.get(symbol, 0))
            current_weight = float(current_value / total_value) if total_value > 0 else 0.0
            target_weight = target_allocation.get(symbol, 0)

            drift = target_weight - current_weight

            if abs(drift) > threshold:
                target_value = _to_decimal(target_weight * float(total_value))
                change_value = target_value - current_value
                price = _to_decimal(current_prices.get(symbol, 1))

                # Precise share calculation
                if price > 0:
                    shares = int(abs(change_value) / price)
                else:
                    shares = 0

                recommendations[symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'drift': drift,
                    'action': 'BUY' if change_value > 0 else 'SELL',
                    'shares': shares,
                    'value': float(abs(change_value))  # Convert back for API compatibility
                }

        return recommendations


class PortfolioAnalyzer:
    """Analyze an existing portfolio"""

    @staticmethod
    def analyze(
        holdings: Dict[str, float],
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze current portfolio using precise Decimal arithmetic for monetary values.

        Args:
            holdings: Symbol -> current value
            price_data: Symbol -> DataFrame with prices

        Returns:
            Analysis results
        """
        # Use Decimal for precise monetary totals
        total_value_decimal = sum(_to_decimal(v) for v in holdings.values())
        if total_value_decimal == 0:
            return {'error': 'Empty portfolio'}

        total_value = float(total_value_decimal)  # Convert for API compatibility

        # Calculate weights using Decimal for precision
        weights = {
            s: float(_to_decimal(v) / total_value_decimal)
            for s, v in holdings.items()
        }

        # Get returns
        returns = pd.DataFrame()
        for symbol in holdings.keys():
            if symbol in price_data:
                df = price_data[symbol]
                col = 'close' if 'close' in df.columns else 'Close'
                returns[symbol] = df[col].pct_change()

        returns = returns.dropna()

        if returns.empty:
            return {'error': 'No return data'}

        # Portfolio returns
        weight_array = np.array([weights.get(s, 0) for s in returns.columns])
        portfolio_returns = (returns * weight_array).sum(axis=1)

        # Metrics
        analysis = {
            'total_value': total_value,
            'weights': weights,
            'n_holdings': len(holdings),
            'expected_return': ReturnCalculator.annualized_return(portfolio_returns),
            'volatility': ReturnCalculator.annualized_volatility(portfolio_returns),
            'sharpe_ratio': ReturnCalculator.sharpe_ratio(portfolio_returns),
            'max_drawdown': ReturnCalculator.max_drawdown(portfolio_returns),
            'diversification': CorrelationAnalyzer.diversification_ratio(
                weight_array, returns.cov().values * 252
            ),
            'top_holding': max(weights.items(), key=lambda x: x[1]),
            'concentration': max(weights.values()),
        }

        # Warnings
        warnings = []
        if analysis['concentration'] > 0.3:
            warnings.append(f"High concentration in {analysis['top_holding'][0]}")
        if analysis['n_holdings'] < 5:
            warnings.append("Low diversification - consider adding more stocks")
        if analysis['volatility'] > 0.25:
            warnings.append("High volatility portfolio")

        analysis['warnings'] = warnings

        return analysis


# ============== QUICK FUNCTIONS ==============

def optimize_portfolio(
    price_data: Dict[str, pd.DataFrame],
    goal: str = "max_sharpe"
) -> AllocationResult:
    """Quick portfolio optimization"""
    goal_map = {
        'max_sharpe': OptimizationGoal.MAX_SHARPE,
        'min_risk': OptimizationGoal.MIN_VOLATILITY,
        'max_return': OptimizationGoal.MAX_RETURN,
        'equal': OptimizationGoal.EQUAL_WEIGHT,
        'risk_parity': OptimizationGoal.RISK_PARITY,
    }

    optimizer = PortfolioOptimizer()
    optimizer.load_data(price_data)
    return optimizer.optimize(goal_map.get(goal, OptimizationGoal.MAX_SHARPE))


def analyze_portfolio(
    holdings: Dict[str, float],
    price_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Quick portfolio analysis"""
    return PortfolioAnalyzer.analyze(holdings, price_data)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("PORTFOLIO OPTIMIZER - Test")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)

    def create_stock_data(base: float, vol: float, trend: float) -> pd.DataFrame:
        days = 252
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.randn(days) * vol + trend
        prices = base * np.exp(np.cumsum(returns))
        return pd.DataFrame({'close': prices}, index=dates)

    # Create diverse portfolio
    price_data = {
        'RELIANCE': create_stock_data(2500, 0.02, 0.0005),    # Low vol, positive trend
        'TCS': create_stock_data(3500, 0.018, 0.0004),        # Tech
        'INFY': create_stock_data(1500, 0.022, 0.0003),       # Tech (correlated)
        'HDFC': create_stock_data(2800, 0.015, 0.0003),       # Finance
        'ICICIBANK': create_stock_data(950, 0.025, 0.0002),   # Finance (correlated)
        'ITC': create_stock_data(450, 0.012, 0.0002),         # FMCG (defensive)
        'SBIN': create_stock_data(600, 0.03, 0.0001),         # High vol bank
        'BHARTIARTL': create_stock_data(1200, 0.02, 0.0004),  # Telecom
    }

    print(f"\nAnalyzing {len(price_data)} stocks...")

    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    optimizer.load_data(price_data)

    # Test different optimization goals
    print("\n" + "=" * 50)
    print("1. MAX SHARPE RATIO OPTIMIZATION")
    result = optimizer.optimize(OptimizationGoal.MAX_SHARPE)
    result.print_summary()

    print("\n" + "=" * 50)
    print("2. MINIMUM VOLATILITY OPTIMIZATION")
    result = optimizer.optimize(OptimizationGoal.MIN_VOLATILITY)
    result.print_summary()

    print("\n" + "=" * 50)
    print("3. RISK PARITY OPTIMIZATION")
    result = optimizer.optimize(OptimizationGoal.RISK_PARITY)
    result.print_summary()

    # Correlation analysis
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)

    returns = pd.DataFrame({s: d['close'].pct_change() for s, d in price_data.items()}).dropna()

    print("\nLow correlation pairs (diversifiers):")
    diversifiers = CorrelationAnalyzer.find_diversifiers(returns, threshold=0.5)
    for s1, s2, corr in diversifiers[:5]:
        print(f"  {s1} <-> {s2}: {corr:.2f}")

    print("\nHigh correlation pairs (redundant):")
    redundant = CorrelationAnalyzer.find_highly_correlated(returns, threshold=0.7)
    for s1, s2, corr in redundant[:5]:
        print(f"  {s1} <-> {s2}: {corr:.2f}")

    # Portfolio analysis
    print("\n" + "=" * 50)
    print("PORTFOLIO ANALYSIS")
    print("=" * 50)

    # Sample current holdings
    holdings = {
        'RELIANCE': 50000,
        'TCS': 40000,
        'HDFC': 30000,
        'ITC': 20000,
    }

    analysis = PortfolioAnalyzer.analyze(holdings, price_data)
    print(f"\nTotal Value: ₹{analysis['total_value']:,.0f}")
    print(f"Expected Return: {analysis['expected_return']:.1%}")
    print(f"Volatility: {analysis['volatility']:.1%}")
    print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {analysis['max_drawdown']:.1%}")

    if analysis['warnings']:
        print("\n⚠️ Warnings:")
        for warning in analysis['warnings']:
            print(f"  - {warning}")

    print("\n" + "=" * 50)
    print("Portfolio Optimizer ready!")
    print("=" * 50)
