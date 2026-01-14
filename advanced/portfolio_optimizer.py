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
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def get_allocation(self, capital: float) -> Dict[str, float]:
        """Get allocation in rupees"""
        return {symbol: weight * capital for symbol, weight in self.weights.items()}

    def get_shares(self, capital: float, prices: Dict[str, float]) -> Dict[str, int]:
        """Get number of shares to buy"""
        allocation = self.get_allocation(capital)
        return {
            symbol: int(amount / prices.get(symbol, 1))
            for symbol, amount in allocation.items()
        }

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

    @staticmethod
    def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
        """Calculate annualized return"""
        total_return = (1 + returns).prod() - 1
        n_days = len(returns)
        if n_days == 0:
            return 0
        return (1 + total_return) ** (trading_days / n_days) - 1

    @staticmethod
    def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """Calculate annualized volatility"""
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
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            n_simulations: Number of random portfolios to simulate

        Returns:
            Optimized allocation
        """
        if self.returns is None or self.n_assets == 0:
            raise ValueError("No data loaded. Call load_data() first.")

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

    def _monte_carlo_optimization(
        self,
        goal: OptimizationGoal,
        target_return: float,
        max_weight: float,
        min_weight: float,
        n_simulations: int
    ) -> np.ndarray:
        """Run Monte Carlo simulation to find optimal weights"""

        best_weights = None
        best_score = float('-inf') if goal != OptimizationGoal.MIN_VOLATILITY else float('inf')

        for _ in range(n_simulations):
            # Generate random weights
            weights = np.random.random(self.n_assets)
            weights = weights / weights.sum()  # Normalize to sum to 1

            # Apply constraints
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()  # Re-normalize

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
                    if abs(exp_return - target_return) < 0.01:  # Within 1%
                        score = -volatility  # Minimize vol at target return
                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()

        if best_weights is None:
            # Fallback to equal weight
            best_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        return best_weights

    def _risk_parity_weights(self) -> np.ndarray:
        """Calculate risk parity weights (equal risk contribution)"""

        # Calculate volatility for each asset
        volatilities = self.returns.std().values * np.sqrt(252)

        # Inverse volatility weighting
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

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
        Get rebalancing recommendations.

        Args:
            current_holdings: Symbol -> current value
            target_allocation: Symbol -> target weight (0-1)
            current_prices: Symbol -> current price
            threshold: Rebalance if drift > threshold

        Returns:
            Rebalancing instructions
        """
        total_value = sum(current_holdings.values())
        if total_value == 0:
            return {}

        recommendations = {}

        for symbol in set(list(current_holdings.keys()) + list(target_allocation.keys())):
            current_value = current_holdings.get(symbol, 0)
            current_weight = current_value / total_value
            target_weight = target_allocation.get(symbol, 0)

            drift = target_weight - current_weight

            if abs(drift) > threshold:
                target_value = target_weight * total_value
                change_value = target_value - current_value
                price = current_prices.get(symbol, 1)
                shares = int(abs(change_value) / price)

                recommendations[symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'drift': drift,
                    'action': 'BUY' if change_value > 0 else 'SELL',
                    'shares': shares,
                    'value': abs(change_value)
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
        Analyze current portfolio.

        Args:
            holdings: Symbol -> current value
            price_data: Symbol -> DataFrame with prices

        Returns:
            Analysis results
        """
        total_value = sum(holdings.values())
        if total_value == 0:
            return {'error': 'Empty portfolio'}

        # Calculate weights
        weights = {s: v / total_value for s, v in holdings.items()}

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
