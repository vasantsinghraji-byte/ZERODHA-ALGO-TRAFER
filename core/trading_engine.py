# -*- coding: utf-8 -*-
"""
Trading Engine - The Heart of the System!
==========================================
This is where all the magic happens!

Combines:
- Strategy signals
- Order execution
- Position tracking
- Risk management

Like an autopilot for trading!
"""

import logging
import threading
import time
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from core.broker import ZerodhaBroker
from core.order_manager import OrderManager, Side, OrderStatus
from core.position_manager import PositionManager, Position
from strategies.base import Strategy, Signal, SignalType

logger = logging.getLogger(__name__)


# ============== ENUMS ==============

class EngineStatus(Enum):
    """Engine status"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class TradingMode(Enum):
    """Trading mode"""
    PAPER = "PAPER"       # Fake money
    LIVE = "LIVE"         # Real money


# ============== CONFIG ==============

@dataclass
class EngineConfig:
    """Configuration for trading engine"""

    # Trading settings
    mode: TradingMode = TradingMode.PAPER
    capital: float = 100000.0
    position_size_pct: float = 10.0       # % of capital per trade
    max_positions: int = 5

    # Risk settings
    max_daily_loss_pct: float = 5.0       # Stop trading if lose this much
    stop_loss_pct: float = 2.0            # Default stop loss
    target_pct: float = 4.0               # Default target

    # Time settings
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    square_off_time: dt_time = dt_time(15, 15)  # Close positions before market

    # Execution settings
    check_interval: int = 60              # Seconds between strategy checks
    use_live_data: bool = False


# ============== TRADING ENGINE ==============

class TradingEngine:
    """
    The main trading engine!

    This is like the brain that:
    1. Gets signals from strategies
    2. Decides whether to trade
    3. Places orders
    4. Tracks positions
    5. Manages risk

    Example:
        engine = TradingEngine(config, broker)
        engine.add_strategy(my_strategy)
        engine.start()
    """

    def __init__(
        self,
        config: EngineConfig = None,
        broker: ZerodhaBroker = None
    ):
        """
        Initialize Trading Engine.

        Args:
            config: Engine configuration
            broker: Zerodha broker instance
        """
        self.config = config or EngineConfig()
        self.broker = broker

        # Core components
        self.order_manager = OrderManager(
            broker=broker,
            paper_trading=(self.config.mode == TradingMode.PAPER)
        )
        self.position_manager = PositionManager(broker=broker)

        # Strategies
        self._strategies: Dict[str, Strategy] = {}
        self._symbols: Dict[str, str] = {}  # strategy -> symbol mapping

        # State
        self._status = EngineStatus.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._start_capital = self.config.capital

        # Callbacks
        self._on_signal: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Set paper balance
        if self.config.mode == TradingMode.PAPER:
            self.order_manager.set_paper_balance(self.config.capital)

        logger.info(f"TradingEngine initialized. Mode: {self.config.mode.value}")

    # ============== STRATEGY MANAGEMENT ==============

    def add_strategy(self, strategy: Strategy, symbol: str):
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance
            symbol: Symbol to trade with this strategy
        """
        name = strategy.name
        self._strategies[name] = strategy
        self._symbols[name] = symbol
        logger.info(f"Strategy added: {name} for {symbol}")

    def remove_strategy(self, name: str):
        """Remove a strategy"""
        if name in self._strategies:
            del self._strategies[name]
            del self._symbols[name]
            logger.info(f"Strategy removed: {name}")

    def get_strategies(self) -> List[str]:
        """Get list of active strategies"""
        return list(self._strategies.keys())

    # ============== ENGINE CONTROL ==============

    def start(self):
        """Start the trading engine"""
        if self._status == EngineStatus.RUNNING:
            logger.warning("Engine already running")
            return

        self._status = EngineStatus.STARTING
        self._running = True

        # Reset daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0

        # Start main loop in thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._status = EngineStatus.RUNNING
        logger.info("Trading Engine STARTED")

    def stop(self):
        """Stop the trading engine"""
        self._running = False
        self._status = EngineStatus.STOPPED

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Trading Engine STOPPED")

    def pause(self):
        """Pause trading (keeps monitoring)"""
        self._status = EngineStatus.PAUSED
        logger.info("Trading Engine PAUSED")

    def resume(self):
        """Resume trading"""
        self._status = EngineStatus.RUNNING
        logger.info("Trading Engine RESUMED")

    @property
    def status(self) -> EngineStatus:
        """Get engine status"""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if engine is running"""
        return self._status == EngineStatus.RUNNING

    # ============== MAIN LOOP ==============

    def _run_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")

        while self._running:
            try:
                # Check if market is open
                if not self._is_market_hours():
                    time.sleep(60)
                    continue

                # Check daily loss limit
                if self._check_daily_loss_limit():
                    logger.warning("Daily loss limit reached! Stopping trading.")
                    self._status = EngineStatus.PAUSED
                    time.sleep(60)
                    continue

                # Check for square off time
                if self._is_square_off_time():
                    self._square_off_all()
                    time.sleep(60)
                    continue

                # Run strategies (if not paused)
                if self._status == EngineStatus.RUNNING:
                    self._run_strategies()

                # Update positions
                self._update_positions()

                # Check stop losses and targets
                self._check_risk()

                # Wait for next cycle
                time.sleep(self.config.check_interval)

            except Exception as e:
                logger.error(f"Engine error: {e}")
                if self._on_error:
                    self._on_error(str(e))
                time.sleep(10)

        logger.info("Trading loop ended")

    def _run_strategies(self):
        """Run all strategies and process signals"""
        for strategy_name, strategy in self._strategies.items():
            symbol = self._symbols[strategy_name]

            try:
                # Get data for strategy
                data = self._get_data(symbol)
                if data is None or data.empty:
                    continue

                # Get signal from strategy
                signal = strategy.analyze(data, symbol)

                # Notify callback
                if self._on_signal:
                    self._on_signal(signal)

                # Process signal
                self._process_signal(signal, strategy_name)

            except Exception as e:
                logger.error(f"Strategy error ({strategy_name}): {e}")

    def _process_signal(self, signal: Signal, strategy_name: str):
        """Process a trading signal"""
        if signal.signal_type == SignalType.HOLD:
            return

        symbol = signal.symbol

        # Check if we already have position
        has_position = self.position_manager.has_position(symbol)

        # Check max positions
        current_positions = len(self.position_manager.get_all_positions())

        if signal.signal_type == SignalType.BUY:
            if has_position:
                logger.debug(f"Already have position in {symbol}, skipping BUY")
                return

            if current_positions >= self.config.max_positions:
                logger.warning(f"Max positions reached ({self.config.max_positions})")
                return

            # Calculate quantity
            position_value = self.config.capital * self.config.position_size_pct / 100
            quantity = int(position_value / signal.price)

            if quantity < 1:
                logger.warning(f"Calculated quantity is 0 for {symbol}")
                return

            # Place buy order
            self._execute_buy(
                symbol=symbol,
                quantity=quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                target=signal.target,
                strategy=strategy_name,
                reason=signal.reason
            )

        elif signal.signal_type == SignalType.SELL:
            if has_position:
                # Close existing position
                self._execute_sell(
                    symbol=symbol,
                    price=signal.price,
                    strategy=strategy_name,
                    reason=signal.reason
                )

        elif signal.signal_type == SignalType.EXIT:
            if has_position:
                self._execute_sell(
                    symbol=symbol,
                    price=signal.price,
                    strategy=strategy_name,
                    reason="Exit signal"
                )

    # ============== ORDER EXECUTION ==============

    def _execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: float = 0,
        target: float = 0,
        strategy: str = "",
        reason: str = ""
    ):
        """Execute a buy order"""
        logger.info(f"BUY Signal: {quantity} x {symbol} @ Rs.{price:.2f} | {reason}")

        # Place order
        order = self.order_manager.buy(
            symbol=symbol,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy
        )

        if order.status == OrderStatus.COMPLETE:
            # Add to positions
            self.position_manager.add_position(
                symbol=symbol,
                quantity=quantity,
                price=order.average_price,
                stop_loss=stop_loss or (price * (1 - self.config.stop_loss_pct / 100)),
                target=target or (price * (1 + self.config.target_pct / 100)),
                strategy=strategy
            )

            self._daily_trades += 1

            if self._on_trade:
                self._on_trade(order)

            logger.info(f"BUY executed: {order}")

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        strategy: str = "",
        reason: str = ""
    ):
        """Execute a sell order"""
        position = self.position_manager.get_position(symbol)
        if not position:
            return

        logger.info(f"SELL Signal: {position.quantity} x {symbol} @ Rs.{price:.2f} | {reason}")

        # Place order
        order = self.order_manager.sell(
            symbol=symbol,
            quantity=position.quantity,
            price=price,
            strategy=strategy
        )

        if order.status == OrderStatus.COMPLETE:
            # Close position
            pnl = self.position_manager.close_position(symbol, order.average_price)

            self._daily_pnl += pnl
            self._daily_trades += 1

            if self._on_trade:
                self._on_trade(order)

            logger.info(f"SELL executed: {order} | P&L: Rs.{pnl:+.0f}")

    # ============== RISK MANAGEMENT ==============

    def _check_risk(self):
        """Check stop losses and targets"""
        prices = self._get_current_prices()

        # Check stop losses
        sl_triggered = self.position_manager.check_stop_losses(prices)
        for symbol in sl_triggered:
            price = prices.get(symbol, 0)
            self._execute_sell(symbol, price, reason="Stop Loss Hit")

        # Check targets
        target_triggered = self.position_manager.check_targets(prices)
        for symbol in target_triggered:
            price = prices.get(symbol, 0)
            self._execute_sell(symbol, price, reason="Target Hit")

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit reached"""
        if self._daily_pnl < 0:
            loss_pct = abs(self._daily_pnl) / self._start_capital * 100
            return loss_pct >= self.config.max_daily_loss_pct
        return False

    def _square_off_all(self):
        """Close all positions (end of day)"""
        logger.info("Squaring off all positions...")

        positions = self.position_manager.get_all_positions()
        prices = self._get_current_prices()

        for pos in positions:
            price = prices.get(pos.symbol, pos.last_price)
            self._execute_sell(pos.symbol, price, reason="End of Day Square Off")

    # ============== HELPER METHODS ==============

    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now().time()
        return self.config.market_open <= now <= self.config.market_close

    def _is_square_off_time(self) -> bool:
        """Check if it's time to square off"""
        now = datetime.now().time()
        return now >= self.config.square_off_time

    def _get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data for a symbol"""
        # For now, return empty - would connect to data manager
        # In production, this would fetch from DataManager or LiveFeed
        return pd.DataFrame()

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all positions"""
        prices = {}

        if self.broker and self.broker.is_connected:
            for pos in self.position_manager.get_all_positions():
                quote = self.broker.get_quote(pos.symbol)
                if quote:
                    prices[pos.symbol] = quote.last_price
        else:
            # Use last known prices
            for pos in self.position_manager.get_all_positions():
                prices[pos.symbol] = pos.last_price

        return prices

    def _update_positions(self):
        """Update position prices"""
        prices = self._get_current_prices()
        self.position_manager.update_all_prices(prices)

    # ============== CALLBACKS ==============

    def on_signal(self, callback: Callable):
        """Set callback for signals"""
        self._on_signal = callback

    def on_trade(self, callback: Callable):
        """Set callback for trades"""
        self._on_trade = callback

    def on_error(self, callback: Callable):
        """Set callback for errors"""
        self._on_error = callback

    # ============== STATUS & STATS ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        portfolio = self.position_manager.get_summary()

        return {
            'status': self._status.value,
            'mode': self.config.mode.value,
            'strategies': len(self._strategies),
            'positions': portfolio['total_positions'],
            'daily_trades': self._daily_trades,
            'daily_pnl': self._daily_pnl,
            'unrealized_pnl': portfolio['unrealized_pnl'],
            'total_invested': portfolio['total_invested'],
            'current_value': portfolio['current_value']
        }

    def print_status(self):
        """Print current status"""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("TRADING ENGINE STATUS")
        print("=" * 50)
        print(f"Status: {stats['status']}")
        print(f"Mode: {stats['mode']}")
        print(f"Strategies: {stats['strategies']}")
        print(f"Open Positions: {stats['positions']}")
        print(f"Today's Trades: {stats['daily_trades']}")
        print(f"Today's P&L: Rs.{stats['daily_pnl']:+.0f}")
        print(f"Unrealized P&L: Rs.{stats['unrealized_pnl']:+.0f}")
        print("=" * 50)


# ============== QUICK START ==============

def create_paper_engine(capital: float = 100000) -> TradingEngine:
    """Create a paper trading engine"""
    config = EngineConfig(
        mode=TradingMode.PAPER,
        capital=capital
    )
    return TradingEngine(config)


def create_live_engine(broker: ZerodhaBroker, capital: float = 100000) -> TradingEngine:
    """Create a live trading engine"""
    config = EngineConfig(
        mode=TradingMode.LIVE,
        capital=capital
    )
    return TradingEngine(config, broker)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("TRADING ENGINE - Test")
    print("=" * 50)

    # Create paper engine
    engine = create_paper_engine(capital=100000)

    # Add a strategy
    from strategies import get_strategy
    strategy = get_strategy('turtle')
    engine.add_strategy(strategy, "RELIANCE")

    # Print status
    engine.print_status()

    print("\n" + "=" * 50)
    print("Trading Engine ready!")
    print("=" * 50)
