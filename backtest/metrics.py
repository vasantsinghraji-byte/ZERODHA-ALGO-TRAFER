# -*- coding: utf-8 -*-
"""
Backtest Metrics - The Report Card!
===================================
Calculates how well your strategy performed.

Like getting grades in school:
- Return %: How much money you made
- Win Rate: How often you were right
- Sharpe Ratio: Risk-adjusted return
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """All the performance numbers"""

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    var_95: float = 0.0  # Value at Risk

    # Risk-Adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Win/Loss
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0

    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_trade_duration: float = 0.0


def calculate_metrics(
    equity_curve: List[float],
    trades: List[Any] = None,
    risk_free_rate: float = 0.05
) -> PerformanceMetrics:
    """
    Calculate performance metrics from equity curve.

    Args:
        equity_curve: List of portfolio values over time
        trades: List of Trade objects (optional)
        risk_free_rate: Annual risk-free rate (default 5%)

    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()

    if len(equity_curve) < 2:
        return metrics

    equity = np.array(equity_curve)
    initial = equity[0]
    final = equity[-1]

    # Returns
    metrics.total_return = final - initial
    metrics.total_return_pct = (final / initial - 1) * 100

    # Calculate daily returns
    returns = np.diff(equity) / equity[:-1]

    # Annualize (assume 252 trading days)
    trading_days = len(returns)
    years = trading_days / 252

    if years > 0:
        metrics.annual_return = ((final / initial) ** (1 / years) - 1) * 100
        metrics.monthly_return = metrics.annual_return / 12

    # Volatility
    if len(returns) > 0:
        metrics.volatility = returns.std() * np.sqrt(252) * 100

    # Maximum Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity)
    drawdown_pct = drawdown / peak

    metrics.max_drawdown = drawdown.max()
    metrics.max_drawdown_pct = drawdown_pct.max() * 100

    # Value at Risk (95%)
    if len(returns) > 0:
        metrics.var_95 = np.percentile(returns, 5) * initial

    # Sharpe Ratio
    if len(returns) > 0 and returns.std() > 0:
        excess_return = returns.mean() * 252 - risk_free_rate
        metrics.sharpe_ratio = excess_return / (returns.std() * np.sqrt(252))

    # Sortino Ratio (only downside volatility)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        excess_return = returns.mean() * 252 - risk_free_rate
        metrics.sortino_ratio = excess_return / (negative_returns.std() * np.sqrt(252))

    # Calmar Ratio
    if metrics.max_drawdown_pct > 0:
        metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown_pct

    # Trade-based metrics
    if trades:
        metrics.total_trades = len(trades)

        winners = [t for t in trades if t.profit_loss > 0]
        losers = [t for t in trades if t.profit_loss <= 0]

        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)

        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100

        if winners:
            profits = [t.profit_loss for t in winners]
            metrics.avg_win = sum(profits) / len(profits)
            metrics.largest_win = max(profits)

        if losers:
            losses = [abs(t.profit_loss) for t in losers]
            metrics.avg_loss = sum(losses) / len(losses)
            metrics.largest_loss = max(losses)

        # Profit Factor
        total_profit = sum(t.profit_loss for t in winners) if winners else 0
        total_loss = abs(sum(t.profit_loss for t in losers)) if losers else 0

        if total_loss > 0:
            metrics.profit_factor = total_profit / total_loss

        # Win/Loss Streaks
        if trades:
            current_streak = 0
            max_win = 0
            max_loss = 0

            for trade in trades:
                if trade.profit_loss > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    max_win = max(max_win, current_streak)
                else:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    max_loss = max(max_loss, abs(current_streak))

            metrics.max_win_streak = max_win
            metrics.max_loss_streak = max_loss

    return metrics


def print_report(metrics: PerformanceMetrics):
    """Print a beautiful performance report"""

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)

    print(f"\n{'--- RETURNS ---':^60}")
    print(f"Total Return:      Rs.{metrics.total_return:>12,.0f} ({metrics.total_return_pct:+.1f}%)")
    print(f"Annual Return:     {metrics.annual_return:>12.1f}%")
    print(f"Monthly Return:    {metrics.monthly_return:>12.1f}%")

    print(f"\n{'--- RISK ---':^60}")
    print(f"Volatility:        {metrics.volatility:>12.1f}%")
    print(f"Max Drawdown:      Rs.{metrics.max_drawdown:>12,.0f} ({metrics.max_drawdown_pct:.1f}%)")
    print(f"Value at Risk:     Rs.{metrics.var_95:>12,.0f}")

    print(f"\n{'--- RISK-ADJUSTED RETURNS ---':^60}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:>12.2f}")
    print(f"Sortino Ratio:     {metrics.sortino_ratio:>12.2f}")
    print(f"Calmar Ratio:      {metrics.calmar_ratio:>12.2f}")

    print(f"\n{'--- TRADE ANALYSIS ---':^60}")
    print(f"Total Trades:      {metrics.total_trades:>12}")
    print(f"Win Rate:          {metrics.win_rate:>12.1f}%")
    print(f"Avg Win:           Rs.{metrics.avg_win:>12,.0f}")
    print(f"Avg Loss:          Rs.{metrics.avg_loss:>12,.0f}")
    print(f"Largest Win:       Rs.{metrics.largest_win:>12,.0f}")
    print(f"Largest Loss:      Rs.{metrics.largest_loss:>12,.0f}")
    print(f"Profit Factor:     {metrics.profit_factor:>12.2f}")

    print(f"\n{'--- STREAKS ---':^60}")
    print(f"Max Win Streak:    {metrics.max_win_streak:>12}")
    print(f"Max Loss Streak:   {metrics.max_loss_streak:>12}")

    # Rating
    print(f"\n{'--- RATING ---':^60}")

    score = 0
    if metrics.total_return_pct > 0:
        score += 1
    if metrics.sharpe_ratio > 1:
        score += 1
    if metrics.win_rate > 50:
        score += 1
    if metrics.profit_factor > 1.5:
        score += 1
    if metrics.max_drawdown_pct < 20:
        score += 1

    stars = "*" * score
    rating = ["POOR", "FAIR", "GOOD", "VERY GOOD", "EXCELLENT", "OUTSTANDING"][score]

    print(f"Score: {score}/5 {stars}")
    print(f"Rating: {rating}")

    print("\n" + "=" * 60)


def compare_strategies(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare multiple strategy results.

    Args:
        results: Dict of strategy_name -> BacktestResult

    Returns:
        DataFrame comparing strategies
    """
    data = []

    for name, result in results.items():
        data.append({
            'Strategy': name,
            'Return %': result.return_pct,
            'Win Rate %': result.win_rate,
            'Trades': result.total_trades,
            'Profit Factor': result.profit_factor,
            'Max DD %': result.max_drawdown_pct,
            'Sharpe': result.sharpe_ratio
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Return %', ascending=False)

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


# Quick scoring function
def score_strategy(result) -> Dict[str, Any]:
    """
    Give a simple score to a strategy result.

    Returns dict with score and explanation.
    """
    score = 0
    reasons = []

    # Profitability
    if result.return_pct > 20:
        score += 2
        reasons.append("Excellent returns")
    elif result.return_pct > 0:
        score += 1
        reasons.append("Profitable")
    else:
        reasons.append("Lost money")

    # Win rate
    if result.win_rate > 60:
        score += 2
        reasons.append("High win rate")
    elif result.win_rate > 45:
        score += 1
        reasons.append("Decent win rate")
    else:
        reasons.append("Low win rate")

    # Risk
    if result.max_drawdown_pct < 10:
        score += 2
        reasons.append("Low risk")
    elif result.max_drawdown_pct < 20:
        score += 1
        reasons.append("Moderate risk")
    else:
        reasons.append("High risk")

    # Profit factor
    if result.profit_factor > 2:
        score += 2
        reasons.append("Great profit factor")
    elif result.profit_factor > 1.5:
        score += 1
        reasons.append("Good profit factor")

    max_score = 8
    rating = score / max_score * 100

    return {
        'score': score,
        'max_score': max_score,
        'rating': rating,
        'grade': 'A' if rating >= 80 else 'B' if rating >= 60 else 'C' if rating >= 40 else 'D',
        'reasons': reasons
    }
