"""
Logging System
Easy to understand logs for everyone!

Logs are like a diary - they record everything that happens.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Save logs to file
        log_to_console: Show logs in console

    Returns:
        Configured logger
    """
    # Create logs directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("algotrader")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_file = LOG_DIR / f"algotrader_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "algotrader") -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


# Trade-specific logger for important events
def log_trade(action: str, symbol: str, quantity: int, price: float,
              reason: str = "", logger: Optional[logging.Logger] = None):
    """
    Log a trade event.

    Args:
        action: BUY or SELL
        symbol: Stock symbol
        quantity: Number of shares
        price: Trade price
        reason: Why this trade
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger("algotrader.trades")

    emoji = "🟢" if action.upper() == "BUY" else "🔴"
    message = f"{emoji} {action.upper()} {quantity} x {symbol} @ Rs.{price:.2f}"
    if reason:
        message += f" | Reason: {reason}"

    logger.info(message)

    # Also log to trade file
    trade_file = LOG_DIR / "trades.log"
    with open(trade_file, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | {message}\n")


def log_signal(signal_type: str, symbol: str, price: float,
               confidence: float, reason: str = "",
               logger: Optional[logging.Logger] = None):
    """
    Log a trading signal.

    Args:
        signal_type: BUY, SELL, HOLD
        symbol: Stock symbol
        price: Current price
        confidence: Signal confidence (0-1)
        reason: Signal reason
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger("algotrader.signals")

    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "EXIT": "🚪"}
    emoji = emoji_map.get(signal_type.upper(), "❓")

    message = f"{emoji} Signal: {signal_type} {symbol} @ Rs.{price:.2f} (conf: {confidence:.0%})"
    if reason:
        message += f" | {reason}"

    logger.info(message)


# Initialize default logger
_default_logger: Optional[logging.Logger] = None

def init_logging(level: str = "INFO") -> logging.Logger:
    """Initialize the default logger"""
    global _default_logger
    _default_logger = setup_logging(level=level)
    return _default_logger
