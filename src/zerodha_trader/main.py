# -*- coding: utf-8 -*-
"""
Main Application Entry Point (Async)
Orchestrates configuration loading, dependency injection, and application startup
"""
import sys
import logging
from typing import Optional
import asyncio

from zerodha_trader.core.config import Settings
from zerodha_trader.core.config_loader import load_config_from_yaml
from zerodha_trader.core.container import AppContainer
from zerodha_trader.core.instrument_loader import load_instruments_from_yaml


# Configure logging
def setup_logging(settings: Settings) -> None:
    """
    Configure application logging based on settings

    Args:
        settings: Application settings
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.logs_dir / 'algotrader.log')
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("ZERODHA ALGO TRADER")
    logger.info("=" * 70)
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Live Trading: {settings.live_trading}")
    logger.info("=" * 70)


def load_configuration(use_yaml: bool = True) -> Settings:
    """
    Load application configuration

    Args:
        use_yaml: If True, load from YAML files. Otherwise use .env

    Returns:
        Loaded and validated Settings
    """
    logger = logging.getLogger(__name__)

    if use_yaml:
        logger.info("Loading configuration from YAML files...")
        settings = load_config_from_yaml()
    else:
        logger.info("Loading configuration from .env file...")
        from zerodha_trader.core.config import get_settings
        settings = get_settings()

    # Log configuration summary
    settings.log_config()

    return settings


def create_application(settings: Settings) -> AppContainer:
    """
    Create application container with all dependencies

    Args:
        settings: Application settings

    Returns:
        Initialized AppContainer
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating application container...")

    # Create container
    container = AppContainer(settings)

    # Perform health check
    logger.info("Performing health check...")
    health = container.health_check()

    for service, status in health.items():
        status_str = " OK" if status else " FAILED"
        logger.info(f"  {service}: {status_str}")

    if not all(health.values()):
        logger.warning("Some services failed health check!")
    else:
        logger.info("All services healthy")

    return container


async def run_application(container: AppContainer) -> None:
    """
    Main application logic (async)

    Args:
        container: Application container with all dependencies
    """
    from zerodha_trader.core.trading_bot import TradingBot

    logger = logging.getLogger(__name__)
    logger.info("Starting application...")

    # Send startup notification
    container.notifier.send_notification(
        f"Algo Trader started in {container.settings.environment.value} mode",
        level="success"
    )

    # Display application info
    print("\n" + "=" * 70)
    print("ZERODHA ALGO TRADER")
    print("=" * 70)
    print(f"Environment: {container.settings.environment.value}")
    print(f"Live Trading: {container.settings.live_trading}")
    print(f"Account Balance: Rs.{container.settings.account_balance:,.2f}")
    print(f"Max Risk/Trade: {container.settings.max_risk_per_trade * 100:.1f}%")
    print(f"Max Open Positions: {container.settings.max_open_positions}")
    print("=" * 70)

    # Initialize TradingBot with container
    logger.info("Initializing TradingBot...")
    bot = TradingBot(container=container)

    # Load instruments from configuration
    logger.info("Loading instruments from configuration...")
    instruments = load_instruments_from_yaml()

    print(f"\nMonitoring {len(instruments)} instruments:")
    for token, symbol in instruments.items():
        print(f"  - {symbol} (Token: {token})")
    print("=" * 70)

    try:
        # Start the trading bot (async)
        logger.info("Starting TradingBot...")
        await bot.start(
            instrument_tokens=list(instruments.keys()),
            symbols=instruments
        )

        logger.info("TradingBot is running. Press Ctrl+C to stop...")
        print("\nBot is running!")
        print("Monitoring market data and generating signals...")
        print("Press Ctrl+C to stop.\n")

        # Keep application running (async sleep instead of blocking)
        while bot.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        container.notifier.send_notification(
            f"Critical error: {str(e)}",
            level="error"
        )
        raise
    finally:
        # Stop the bot gracefully (async)
        if bot.is_running:
            logger.info("Stopping TradingBot...")
            await bot.stop()


def shutdown_application(container: AppContainer) -> None:
    """
    Gracefully shutdown application

    Args:
        container: Application container
    """
    logger = logging.getLogger(__name__)
    logger.info("Shutting down application...")

    # Send shutdown notification
    try:
        container.notifier.send_notification(
            "=� Algo Trader shutting down...",
            level="info"
        )
    except Exception as e:
        logger.warning(f"Failed to send shutdown notification: {e}")

    # Shutdown container
    container.shutdown()

    logger.info("Application shutdown complete")


async def main_async(use_yaml_config: bool = True) -> int:
    """
    Main application entry point (async)

    Args:
        use_yaml_config: If True, load config from YAML. Otherwise from .env

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    container: Optional[AppContainer] = None

    try:
        # 1. Load configuration
        settings = load_configuration(use_yaml=use_yaml_config)

        # 2. Setup logging
        setup_logging(settings)

        # 3. Create application container (DI)
        container = create_application(settings)

        # 4. Run application (async)
        await run_application(container)

        return 0

    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 0

    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
        if container:
            container.notifier.send_notification(
                f"❌ Error: {str(e)}",
                level="error"
            )
        return 1

    finally:
        if container:
            shutdown_application(container)


def main(use_yaml_config: bool = True) -> int:
    """
    Main application entry point (sync wrapper for async main)

    Args:
        use_yaml_config: If True, load config from YAML. Otherwise from .env

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    return asyncio.run(main_async(use_yaml_config))


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Zerodha Algo Trader")
    parser.add_argument(
        '--config',
        choices=['yaml', 'env'],
        default='yaml',
        help='Configuration source (yaml or env)'
    )
    parser.add_argument(
        '--environment',
        choices=['development', 'paper', 'live'],
        help='Override environment'
    )

    args = parser.parse_args()

    # Override environment if specified
    if args.environment:
        import os
        os.environ['ENVIRONMENT'] = args.environment

    # Run application
    exit_code = main(use_yaml_config=(args.config == 'yaml'))
    sys.exit(exit_code)
