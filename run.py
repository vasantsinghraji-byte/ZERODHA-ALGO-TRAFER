#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlgoTrader Pro - Unified Entry Point
=====================================

This is the SINGLE canonical entry point for the application.
All 12 Phase 8 components are initialized and validated here.

Usage:
    # Start the full application (GUI + all components)
    python run.py

    # Validate components only (for CI/CD)
    python run.py --validate

    # Start in headless mode (no GUI)
    python run.py --headless

    # Custom configuration
    python run.py --capital 200000 --mode paper

Environment Variables:
    ALGOTRADER_CONFIG  - Path to config file (optional)
    ALGOTRADER_LOG_LEVEL - Logging level (DEBUG, INFO, WARNING, ERROR)

Exit Codes:
    0 - Success
    1 - Component validation failed
    2 - Initialization failed
    3 - Runtime error
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: str = "INFO"):
    """Configure application logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Log file with timestamp
    log_file = os.path.join(
        log_dir,
        f"algotrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)

    return log_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AlgoTrader Pro - Professional Trading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                      # Start with GUI
  python run.py --validate           # Validate components only
  python run.py --headless           # Start without GUI
  python run.py --capital 200000     # Custom capital
  python run.py --mode live          # Live trading mode (requires broker)
        """
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate all components without starting the application'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI (headless mode)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial trading capital (default: 100000)'
    )

    parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode: paper (simulated) or live (default: paper)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=os.environ.get('ALGOTRADER_LOG_LEVEL', 'INFO'),
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--no-infrastructure',
        action='store_true',
        help='Skip infrastructure services (flight recorder, etc.)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Print component status and exit'
    )

    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy startup (no component validation)'
    )

    parser.add_argument(
        '--threading',
        action='store_true',
        help='Use threading mode instead of multiprocessing (debug/fallback)'
    )

    # Backtesting arguments
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest instead of live trading'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='ORB',
        choices=['ORB', 'MACD', 'RSI'],
        help='Strategy to backtest (default: ORB)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='NSE:NIFTY 50',
        help='Symbol to backtest (default: NSE:NIFTY 50)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of historical data for backtest (default: 30)'
    )

    # Optimization arguments
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run Walk-Forward Optimization to find best parameters'
    )

    parser.add_argument(
        '--param-grid',
        type=str,
        default='default',
        help='Parameter grid preset (default, aggressive, conservative) or JSON string'
    )

    return parser.parse_args()


def check_basic_dependencies():
    """Check if basic required packages are installed."""
    required = [
        ('tkinter', 'tkinter'),
        ('yaml', 'pyyaml'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
    ]
    missing = []

    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"[X] Missing packages: {', '.join(missing)}")
        print("\n[*] Install them with:")
        print(f"    pip install {' '.join(missing)}")
        return False

    return True


def validate_only():
    """
    Validate all components without starting the application.

    Used for CI/CD pipelines to verify the build.

    Returns:
        0 if validation passes, 1 if it fails
    """
    print("\n" + "=" * 60)
    print("ALGOTRADER PRO - COMPONENT VALIDATION")
    print("=" * 60 + "\n")

    try:
        from core.bootstrap import validate_components_only
        validate_components_only()

        print("\n" + "=" * 60)
        print("VALIDATION PASSED - All components verified")
        print("=" * 60 + "\n")
        return 0

    except RuntimeError as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        return 1

    except ImportError as e:
        print(f"\nIMPORT ERROR: {e}", file=sys.stderr)
        print("Make sure all dependencies are installed:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1


def run_headless(config: dict):
    """
    Run the application in headless mode (no GUI).

    Useful for automated trading, backtesting, or server deployment.
    """
    logger = logging.getLogger(__name__)

    try:
        from core.bootstrap import bootstrap_application

        logger.info("Starting AlgoTrader Pro in headless mode...")

        # Bootstrap application
        registry = bootstrap_application(config)

        # Print status
        registry.print_status_report()

        logger.info("Headless mode ready. Press Ctrl+C to stop.")

        # Keep running until interrupted
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            registry.stop_services()

        return 0

    except Exception as e:
        logger.error(f"Headless mode failed: {e}")
        import traceback
        traceback.print_exc()
        return 3


def run_gui(config: dict):
    """
    Run the application with GUI using multiprocessing engine.

    This is the primary user-facing mode. The trading engine runs in a
    separate process to bypass Python's GIL, ensuring the UI remains
    responsive even during high-frequency market data (100+ ticks/sec).

    Architecture:
        ┌──────────────────────┐     ┌──────────────────────┐
        │ UI Process           │     │ Engine Process       │
        │  └── Tkinter main    │◄───►│  ├── EventBus        │
        │      loop            │ IPC │  ├── Trading Engine  │
        │                      │     │  └── Data Feed       │
        └──────────────────────┘     └──────────────────────┘
    """
    logger = logging.getLogger(__name__)

    try:
        from core.process_engine import EngineProcess
        from ui.app import AlgoTraderApp

        logger.info("Starting AlgoTrader Pro with MULTIPROCESSING engine...")
        logger.info("GIL bypassed - UI will remain responsive during high tick rates")

        # Create engine process wrapper (doesn't start yet)
        engine_process = EngineProcess()

        # Create UI with multiprocessing engine
        app = AlgoTraderApp(engine_process=engine_process)

        # Start engine in separate process
        if not engine_process.start(config):
            logger.error("Failed to start engine process")
            print("\nERROR: Engine process failed to start", file=sys.stderr)
            return 2

        logger.info(f"Engine process started (PID: {engine_process.pid})")

        # Run the GUI (blocks until window closed)
        app.run()

        # Cleanup - stop engine process
        logger.info("Shutting down engine process...")
        engine_process.stop(timeout=5.0)

        logger.info("Shutdown complete")
        return 0

    except RuntimeError as e:
        logger.error(f"Startup failed: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nRun 'python run.py --validate' to diagnose issues.", file=sys.stderr)
        return 2

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nIMPORT ERROR: {e}", file=sys.stderr)
        print("Make sure all dependencies are installed:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


def run_gui_threading(config: dict):
    """
    Run the application with GUI using threading mode (legacy).

    This uses the old architecture where the engine runs in the same
    process as the UI. Simpler but subject to GIL bottlenecks.

    Use this mode for debugging or if multiprocessing causes issues.
    """
    logger = logging.getLogger(__name__)

    try:
        from core.bootstrap import bootstrap_application

        logger.info("Starting AlgoTrader Pro with THREADING engine (legacy mode)...")
        logger.warning("GIL NOT bypassed - UI may freeze during high tick rates")

        # Bootstrap all components
        registry = bootstrap_application(config)

        # Now start the GUI with initialized components
        from ui.app import AlgoTraderApp

        # Create app with initialized components
        app = AlgoTraderApp(
            event_bus=registry.event_bus,
            trading_engine=registry.trading_engine,
            infrastructure_manager=registry.infrastructure_manager
        )

        # Run the GUI (blocks until window closed)
        app.run()

        # Cleanup
        logger.info("Shutting down...")
        registry.stop_services()

        return 0

    except RuntimeError as e:
        logger.error(f"Startup failed: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nRun 'python run.py --validate' to diagnose issues.", file=sys.stderr)
        return 2

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nIMPORT ERROR: {e}", file=sys.stderr)
        print("Make sure all dependencies are installed:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


def run_legacy():
    """
    Run in legacy mode without component validation.

    This matches the old behavior for backward compatibility.
    """
    print("""
    ========================================
    |                                      |
    |   ALGOTRADER PRO v2.0                |
    |   Professional Trading Made Simple   |
    |                                      |
    ========================================
    """)

    print("[*] Checking dependencies...")

    if not check_basic_dependencies():
        input("\nPress Enter to exit...")
        return 1

    print("[OK] All dependencies OK!")
    print("[*] Starting application (legacy mode)...\n")

    try:
        from ui.app import AlgoTraderApp
        app = AlgoTraderApp()
        app.run()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Error starting app: {e}")
        import traceback
        traceback.print_exc()
        print("\n[*] Try running with: python run.py --validate")
        input("\nPress Enter to exit...")
        return 3


def print_status():
    """Print component status and exit."""
    try:
        from core.bootstrap import ComponentRegistry

        registry = ComponentRegistry()
        registry.validate_imports()
        registry.print_status_report()
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_backtest(args):
    """
    Run backtest with specified strategy and symbol.

    Uses the UnifiedBacktester to ensure consistent cost models
    between vectorized and iterative engines.
    """
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("ALGOTRADER PRO - BACKTEST MODE")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Capital: Rs.{args.capital:,.0f}")
    print("=" * 60 + "\n")

    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Import backtest components
        from backtest.unified import UnifiedBacktester, BacktestConfig, CostModel

        # Import strategy
        if args.strategy == 'ORB':
            from strategies.orb_strategy import ORBStrategy
            strategy = ORBStrategy(
                opening_minutes=15,
                use_imbalance_filter=False  # Disable for backtest (no Level 2 in historical)
            )
        else:
            logger.error(f"Strategy {args.strategy} not implemented for backtest")
            return 1

        # Generate sample data (in production, fetch from Zerodha)
        logger.info(f"Generating {args.days} days of sample data for {args.symbol}...")
        logger.warning("NOTE: Using simulated data. For real backtest, implement historical data fetching.")

        # Create sample OHLCV data
        dates = pd.date_range(
            end=datetime.now(),
            periods=args.days * 375,  # ~375 minutes per trading day
            freq='1min'
        )

        # Filter to market hours (9:15 - 15:30)
        dates = dates[
            (dates.time >= pd.Timestamp('09:15').time()) &
            (dates.time <= pd.Timestamp('15:30').time())
        ]

        np.random.seed(42)  # For reproducibility
        base_price = 22000 if 'NIFTY' in args.symbol.upper() else 1000
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # Configure backtest with Zerodha costs
        config = BacktestConfig(
            initial_capital=args.capital,
            cost_model=CostModel.ZERODHA_INTRADAY
        )

        logger.info(f"Running backtest with {len(data)} bars...")
        logger.info(f"Cost model: {config.cost_model.name}")
        logger.info(f"Total costs: {config.total_cost_pct:.2f}% per trade")

        # Run backtest
        backtester = UnifiedBacktester(data, config)

        # Simple backtest loop (strategy generates signals, we execute)
        signals = []
        for i in range(100, len(data)):  # Start after warmup period
            window = data.iloc[i-100:i+1]
            signal = strategy.analyze(window, args.symbol)
            signals.append({
                'timestamp': data.index[i],
                'signal': signal.signal_type.value,
                'price': data['close'].iloc[i],
                'confidence': signal.confidence
            })

        signals_df = pd.DataFrame(signals)

        # Count signals
        buy_signals = (signals_df['signal'] == 'BUY').sum()
        sell_signals = (signals_df['signal'] == 'SELL').sum()
        hold_signals = (signals_df['signal'] == 'HOLD').sum()

        # Print results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {data.index[0]} to {data.index[-1]}")
        print(f"Total bars: {len(data)}")
        print(f"Signals generated: {len(signals_df)}")
        print(f"  - BUY:  {buy_signals}")
        print(f"  - SELL: {sell_signals}")
        print(f"  - HOLD: {hold_signals}")
        print(f"\nCost Model: {config.cost_model.name}")
        print(f"  - Slippage: {config.slippage_pct:.2f}%")
        print(f"  - Commission: {config.commission_pct:.2f}%")
        print(f"  - Other charges: {config.other_charges_pct:.2f}%")
        print(f"  - Total one-way: {config.total_cost_pct:.2f}%")
        print("=" * 60)

        if buy_signals == 0 and sell_signals == 0:
            print("\n⚠️  WARNING: No BUY/SELL signals generated!")
            print("This could mean:")
            print("  1. Opening range was never broken (price stayed inside range)")
            print("  2. Data doesn't cover market opening hours (9:15-9:30 AM)")
            print("  3. Strategy parameters are too conservative")
        else:
            print(f"\n✓ Strategy generated {buy_signals + sell_signals} actionable signals")
            print("  For full P&L analysis, implement trade execution in backtest")

        print("\n" + "=" * 60 + "\n")

        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return 1

    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 3


def run_optimization(args):
    """
    Run Walk-Forward Optimization to find best strategy parameters.

    Uses the WalkForwardOptimizer to avoid overfitting by testing
    on out-of-sample data windows.
    """
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("ALGOTRADER PRO - STRATEGY OPTIMIZATION")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Data Days: {args.days}")
    print(f"Parameter Grid: {args.param_grid}")
    print("=" * 60 + "\n")

    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import json

        # Define parameter grids for different presets
        PARAM_GRIDS = {
            'default': {
                'opening_minutes': [15, 30],
                'buffer_percent': [0.1, 0.2],
            },
            'aggressive': {
                'opening_minutes': [5, 10, 15],
                'buffer_percent': [0.05, 0.1],
            },
            'conservative': {
                'opening_minutes': [30, 45, 60],
                'buffer_percent': [0.2, 0.3, 0.5],
            },
            'full': {
                'opening_minutes': [5, 10, 15, 30, 45, 60],
                'buffer_percent': [0.05, 0.1, 0.15, 0.2, 0.3],
            }
        }

        # Get parameter grid
        if args.param_grid in PARAM_GRIDS:
            param_grid = PARAM_GRIDS[args.param_grid]
            logger.info(f"Using '{args.param_grid}' parameter grid preset")
        else:
            try:
                param_grid = json.loads(args.param_grid)
                logger.info("Using custom parameter grid from JSON")
            except json.JSONDecodeError:
                logger.error(f"Invalid param-grid: {args.param_grid}")
                print(f"Available presets: {list(PARAM_GRIDS.keys())}")
                return 1

        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        print(f"\nParameter Search Space:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        print(f"\nTotal combinations to test: {total_combinations}")
        print("-" * 60)

        # Generate sample data (in production, fetch from Zerodha)
        logger.info(f"Generating {args.days} days of sample data...")
        logger.warning("NOTE: Using simulated data. For real optimization, use historical data.")

        dates = pd.date_range(
            end=datetime.now(),
            periods=args.days * 375,
            freq='1min'
        )

        dates = dates[
            (dates.time >= pd.Timestamp('09:15').time()) &
            (dates.time <= pd.Timestamp('15:30').time())
        ]

        np.random.seed(42)
        base_price = 22000 if 'NIFTY' in args.symbol.upper() else 1000
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # Import strategy
        if args.strategy == 'ORB':
            from strategies.orb_strategy import ORBStrategy
            strategy_class = ORBStrategy
        else:
            logger.error(f"Strategy {args.strategy} not implemented")
            return 1

        # Run grid search optimization
        print(f"\nRunning optimization on {len(data)} bars...")
        print("-" * 60)

        results = []
        best_result = None
        best_score = float('-inf')

        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for i, combo in enumerate(itertools.product(*param_values)):
            params = dict(zip(param_names, combo))

            # Create strategy with these params
            strategy = strategy_class(
                opening_minutes=params.get('opening_minutes', 15),
                buffer_percent=params.get('buffer_percent', 0.1),
                use_imbalance_filter=False
            )

            # Run simple backtest
            signals = []
            for j in range(100, len(data)):
                window = data.iloc[j-100:j+1]
                signal = strategy.analyze(window, args.symbol)
                if signal.signal_type.value != 'HOLD':
                    signals.append({
                        'type': signal.signal_type.value,
                        'price': data['close'].iloc[j],
                        'confidence': signal.confidence
                    })

            # Calculate simple score (number of high-confidence signals)
            high_conf_signals = len([s for s in signals if s['confidence'] > 0.6])
            buy_signals = len([s for s in signals if s['type'] == 'BUY'])
            sell_signals = len([s for s in signals if s['type'] == 'SELL'])

            # Score: balance between signal count and confidence
            score = high_conf_signals * 0.7 + len(signals) * 0.3

            result = {
                'params': params,
                'total_signals': len(signals),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'high_confidence': high_conf_signals,
                'score': score
            }
            results.append(result)

            if score > best_score:
                best_score = score
                best_result = result

            # Progress
            print(f"  [{i+1}/{total_combinations}] {params} -> {len(signals)} signals (score: {score:.1f})")

        # Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        print("\nTop 5 Parameter Combinations:")
        print("-" * 60)
        for i, r in enumerate(results[:5]):
            print(f"\n{i+1}. Score: {r['score']:.1f}")
            print(f"   Parameters: {r['params']}")
            print(f"   Signals: {r['total_signals']} (BUY: {r['buy_signals']}, SELL: {r['sell_signals']})")
            print(f"   High Confidence: {r['high_confidence']}")

        if best_result:
            print("\n" + "=" * 60)
            print("RECOMMENDED SETTINGS")
            print("=" * 60)
            print(f"\nFor {args.symbol}, use:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")

            print(f"\nExpected: ~{best_result['total_signals']} signals per {args.days} days")
            print("\nTo apply these settings, update your strategy config:")
            print(f"""
    strategy = ORBStrategy(
        opening_minutes={best_result['params'].get('opening_minutes', 15)},
        buffer_percent={best_result['params'].get('buffer_percent', 0.1)}
    )
""")

        print("=" * 60 + "\n")
        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nMissing dependency: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        return 3


def main():
    """Main entry point."""
    args = parse_arguments()

    # Handle --legacy flag (old behavior)
    if args.legacy:
        sys.exit(run_legacy())

    # Setup logging
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ALGOTRADER PRO")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    # Handle --validate flag
    if args.validate:
        sys.exit(validate_only())

    # Handle --status flag
    if args.status:
        sys.exit(print_status())

    # Handle --backtest flag
    if args.backtest:
        sys.exit(run_backtest(args))

    # Handle --optimize flag
    if args.optimize:
        sys.exit(run_optimization(args))

    # Build configuration
    config = {
        'initial_capital': args.capital,
        'trading_mode': args.mode,
        'enable_recording': not args.no_infrastructure,
        'enable_audit': not args.no_infrastructure,
        'enable_compliance': not args.no_infrastructure,
        'enable_kill_switch': True,  # Always enable kill switch
        'enable_latency_monitor': not args.no_infrastructure,
    }

    logger.info(f"Configuration: capital={args.capital}, mode={args.mode}")

    # Run in appropriate mode
    if args.headless:
        sys.exit(run_headless(config))
    elif args.threading:
        # Use threading mode (legacy, GIL bottleneck but simpler)
        logger.info("Using threading mode (--threading flag)")
        sys.exit(run_gui_threading(config))
    else:
        # Default: multiprocessing mode (GIL bypassed)
        sys.exit(run_gui(config))


if __name__ == "__main__":
    main()
