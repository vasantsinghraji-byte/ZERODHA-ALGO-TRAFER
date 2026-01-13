# -*- coding: utf-8 -*-
"""
Backtesting Example - Demonstrates how to backtest trading strategies

This example shows how to:
1. Load historical data from CSV
2. Run a backtest using the Backtester class
3. Analyze the results

The backtester reuses the exact same production code:
- EnhancedTradingStrategy (your strategy logic)
- StrategyExecutor (signal generation)
- OrderManager (order execution)
- RiskManager (risk validation)
- PaperTradingBrokerService (simulated trading)
"""
import asyncio
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from zerodha_trader.backtesting import Backtester


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(output_path: Path, num_candles: int = 1000):
    """
    Generate sample OHLC data for demonstration

    Args:
        output_path: Path to save CSV file
        num_candles: Number of candles to generate
    """
    logger.info(f"Generating {num_candles} sample candles...")

    # Generate realistic price data
    np.random.seed(42)

    # Start with a base price
    base_price = 3500.0
    prices = [base_price]

    # Generate random walk with trend
    for _ in range(num_candles - 1):
        change = np.random.normal(0, 10)  # Mean 0, std 10
        prices.append(max(prices[-1] + change, 100))  # Don't go below 100

    # Create OHLC data
    data = []
    for i, close in enumerate(prices):
        # Generate OHLC around close price
        high = close + np.random.uniform(0, 15)
        low = close - np.random.uniform(0, 15)
        open_price = np.random.uniform(low, high)

        data.append({
            'timestamp': f'2024-01-01 09:{i % 60:02d}:{i // 60:02d}',
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(np.random.uniform(1000, 10000))
        })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Sample data saved to {output_path}")
    logger.info(f"Price range: �{df['close'].min():.2f} - �{df['close'].max():.2f}")


async def run_backtest_example():
    """Run a complete backtest example"""

    # Setup paths
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / 'data' / 'historical'
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_file = data_dir / 'sample_data.csv'

    # Generate sample data if it doesn't exist
    if not csv_file.exists():
        generate_sample_data(csv_file, num_candles=1000)
    else:
        logger.info(f"Using existing data: {csv_file}")

    # Run backtest
    logger.info("\n" + "=" * 70)
    logger.info("Starting Backtest")
    logger.info("=" * 70)

    result = await Backtester.run_from_csv(
        csv_path=csv_file,
        instrument_token=738561,  # RELIANCE
        symbol='RELIANCE',
        account_balance=100000.0
    )

    # Print results
    result.print_summary()

    # Print detailed metrics
    metrics = result.get_performance_metrics()

    print("Detailed Metrics:")
    print(f"  Signals per Trade: {metrics['total_signals'] / max(metrics['total_trades'], 1):.2f}")
    print(f"  Average P&L per Trade: �{metrics['total_pnl'] / max(metrics['total_trades'], 1):.2f}")

    # Save results to file
    results_dir = project_root / 'backtest_results'
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write("BACKTEST RESULTS\n")
        f.write("=" * 70 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {results_file}")


async def run_parameter_sweep():
    """
    Example: Run backtest with different parameter combinations
    to find optimal strategy parameters
    """
    logger.info("\n" + "=" * 70)
    logger.info("Parameter Sweep Example")
    logger.info("=" * 70)

    # Setup
    project_root = Path(__file__).parent.parent.parent.parent
    csv_file = project_root / 'data' / 'historical' / 'sample_data.csv'

    if not csv_file.exists():
        generate_sample_data(csv_file, num_candles=1000)

    # Different account balances to test
    test_balances = [50000, 100000, 200000]

    results_comparison = []

    for balance in test_balances:
        logger.info(f"\nTesting with balance: �{balance:,}")

        result = await Backtester.run_from_csv(
            csv_path=csv_file,
            instrument_token=738561,
            symbol='RELIANCE',
            account_balance=balance
        )

        metrics = result.get_performance_metrics()
        results_comparison.append({
            'balance': balance,
            'returns': metrics['returns_pct'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate']
        })

    # Print comparison
    print("\n" + "=" * 70)
    print("Parameter Sweep Results")
    print("=" * 70)
    print(f"{'Balance':<15} {'Returns %':<15} {'Trades':<15} {'Win Rate %':<15}")
    print("-" * 70)

    for res in results_comparison:
        print(f"�{res['balance']:<14,} {res['returns']:<15.2f} {res['trades']:<15} {res['win_rate']:<15.2f}")

    print("=" * 70)


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("BACKTESTING EXAMPLE")
    print("=" * 70)
    print("This example demonstrates backtesting your trading strategy")
    print("using historical data with the exact same code that runs in production.")
    print("=" * 70 + "\n")

    # Run basic backtest
    asyncio.run(run_backtest_example())

    # Optionally run parameter sweep
    print("\nWould you like to run a parameter sweep? (This tests different configurations)")
    print("Uncomment the line below to enable:\n")
    # asyncio.run(run_parameter_sweep())


if __name__ == "__main__":
    main()
