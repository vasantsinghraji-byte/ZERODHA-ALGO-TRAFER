# -*- coding: utf-8 -*-
"""
Trading Engine - The Heart of the System!
==========================================
This is where all the magic happens!

Includes both:
- TradingEngine: Legacy poll-based engine (backward compatible)
- EventDrivenLiveEngine: New event-driven engine (unified architecture)

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
from typing import Dict, List, Optional, Any, Callable, Union
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

        # Thread synchronization lock (RLock allows same thread to acquire multiple times)
        # CRITICAL: Protects shared state accessed by main loop thread and callbacks
        self._state_lock = threading.RLock()

        # Strategies
        self._strategies: Dict[str, Strategy] = {}
        self._symbols: Dict[str, str] = {}  # strategy -> symbol mapping

        # State (protected by _state_lock)
        self._status = EngineStatus.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Daily tracking (protected by _state_lock)
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
        """
        Start the trading engine.

        Thread-safe: Uses _state_lock to protect status changes.
        """
        with self._state_lock:
            if self._status == EngineStatus.RUNNING:
                logger.warning("Engine already running")
                return

            self._status = EngineStatus.STARTING
            self._running = True

            # Reset daily tracking
            self._daily_pnl = 0.0
            self._daily_trades = 0

        # Start main loop in thread (outside lock to avoid holding during thread start)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        with self._state_lock:
            self._status = EngineStatus.RUNNING
        logger.info("Trading Engine STARTED")

    def stop(self):
        """
        Stop the trading engine.

        Thread-safe: Uses _state_lock to protect status changes.
        """
        with self._state_lock:
            self._running = False
            self._status = EngineStatus.STOPPED

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Trading Engine STOPPED")

    def pause(self):
        """Pause trading (keeps monitoring). Thread-safe."""
        with self._state_lock:
            self._status = EngineStatus.PAUSED
        logger.info("Trading Engine PAUSED")

    def resume(self):
        """Resume trading. Thread-safe."""
        with self._state_lock:
            self._status = EngineStatus.RUNNING
        logger.info("Trading Engine RESUMED")

    @property
    def status(self) -> EngineStatus:
        """Get engine status. Thread-safe."""
        with self._state_lock:
            return self._status

    @property
    def is_running(self) -> bool:
        """Check if engine is running. Thread-safe."""
        with self._state_lock:
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
        """
        Process a trading signal.

        Thread-safe: Uses _state_lock to prevent race conditions when
        checking positions and placing orders from concurrent callbacks.
        """
        if signal.signal_type == SignalType.HOLD:
            return

        symbol = signal.symbol

        # Acquire lock for entire signal processing to ensure atomic check-then-act
        with self._state_lock:
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

                # Place buy order (lock held to prevent duplicate orders)
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
        """
        Execute a buy order.

        Thread-safe: Uses _state_lock (reentrant) to protect position
        and daily stats modifications.
        """
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
            # Lock protects position and stats updates
            with self._state_lock:
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
        """
        Execute a sell order.

        Thread-safe: Uses _state_lock (reentrant) to protect position
        and daily stats modifications.
        """
        # Lock for reading position (may be called directly or from locked context)
        with self._state_lock:
            position = self.position_manager.get_position(symbol)
            if not position:
                return
            quantity = position.quantity

        logger.info(f"SELL Signal: {quantity} x {symbol} @ Rs.{price:.2f} | {reason}")

        # Place order
        order = self.order_manager.sell(
            symbol=symbol,
            quantity=quantity,
            price=price,
            strategy=strategy
        )

        if order.status == OrderStatus.COMPLETE:
            # Lock protects position close and stats updates
            with self._state_lock:
                # Close position
                pnl = self.position_manager.close_position(symbol, order.average_price)

                self._daily_pnl += pnl
                self._daily_trades += 1

            if self._on_trade:
                self._on_trade(order)

            logger.info(f"SELL executed: {order} | P&L: Rs.{pnl:+.0f}")

    # ============== RISK MANAGEMENT ==============

    def _check_risk(self):
        """
        Check stop losses and targets.

        Thread-safe: Uses _state_lock to ensure consistent position state
        during risk checks.
        """
        with self._state_lock:
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
        """
        Check if daily loss limit reached.

        Thread-safe: Uses _state_lock to read _daily_pnl consistently.
        """
        with self._state_lock:
            if self._daily_pnl < 0:
                loss_pct = abs(self._daily_pnl) / self._start_capital * 100
                return loss_pct >= self.config.max_daily_loss_pct
            return False

    def _square_off_all(self):
        """
        Close all positions (end of day).

        Thread-safe: Uses _state_lock to get consistent position list.
        """
        logger.info("Squaring off all positions...")

        with self._state_lock:
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
        """
        Get current prices for all positions.

        Attempts to fetch live prices from broker first, then falls back
        to last known prices if broker unavailable.

        Returns:
            Dict mapping symbol to current price
        """
        prices = {}
        positions = self.position_manager.get_all_positions()

        if not positions:
            return prices

        failed_symbols = []

        if self.broker and self.broker.is_connected:
            for pos in positions:
                try:
                    quote = self.broker.get_quote(pos.symbol)
                    if quote and quote.last_price > 0:
                        prices[pos.symbol] = quote.last_price
                    else:
                        failed_symbols.append(pos.symbol)
                except Exception as e:
                    logger.warning(f"Failed to get quote for {pos.symbol}: {e}")
                    failed_symbols.append(pos.symbol)

            # Log warning if some symbols failed
            if failed_symbols:
                logger.warning(f"Could not fetch prices for: {', '.join(failed_symbols)}")

        # Fallback: use last known prices for symbols without fresh data
        for pos in positions:
            if pos.symbol not in prices:
                if pos.last_price > 0:
                    prices[pos.symbol] = pos.last_price
                    logger.debug(f"Using stale price for {pos.symbol}: {pos.last_price}")
                else:
                    logger.warning(f"No valid price available for {pos.symbol}")

        return prices

    def _update_positions(self):
        """
        Update position prices from market data.

        Logs warnings when price data is unavailable to help diagnose
        issues with stop loss/target triggers.
        """
        positions = self.position_manager.get_all_positions()
        if not positions:
            return  # Nothing to update

        prices = self._get_current_prices()

        if not prices:
            logger.error(
                f"Failed to get any price data for {len(positions)} positions. "
                "Stop losses and targets will not trigger correctly!"
            )
            return

        # Check coverage
        missing = [pos.symbol for pos in positions if pos.symbol not in prices]
        if missing:
            logger.warning(
                f"Missing price data for {len(missing)} positions: {', '.join(missing)}. "
                "These positions will not be updated."
            )

        # Update positions that have price data
        if prices:
            self.position_manager.update_all_prices(prices)
            logger.debug(f"Updated prices for {len(prices)} positions")

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
        """
        Get engine statistics.

        Thread-safe: Uses _state_lock to read consistent state.
        """
        with self._state_lock:
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


# =============================================================================
# EVENT-DRIVEN LIVE ENGINE
# =============================================================================

class EventDrivenLiveEngine:
    """
    Event-Driven Live Trading Engine - The Unified Approach!

    Unlike the legacy TradingEngine, this one uses the SAME event-driven
    architecture as backtesting. Events flow through the EventBus:

    LiveFeed -> TickEvent -> BarAggregator -> BarEvent -> Strategy -> Signal

    Benefits:
    - Same code path as backtesting
    - No polling - reacts immediately to events
    - Easy to test with recorded events
    - Cleaner separation of concerns

    Example:
        >>> from core.events import EventBus
        >>> from core.data import LiveDataSource, DataSourceConfig
        >>>
        >>> bus = EventBus()
        >>> engine = EventDrivenLiveEngine(bus, broker, capital=100000)
        >>> engine.add_strategy(strategy, "RELIANCE")
        >>> engine.start()
    """

    def __init__(
        self,
        event_bus=None,
        broker: Optional[ZerodhaBroker] = None,
        config: Optional[EngineConfig] = None,
        initial_capital: float = 100000,
        position_size_pct: float = 10.0,
    ):
        """
        Initialize event-driven trading engine.

        Args:
            event_bus: EventBus instance (created if not provided)
            broker: Zerodha broker instance (for live trading)
            config: Engine configuration
            initial_capital: Starting capital
            position_size_pct: % of capital per trade
        """
        # Lazy import to avoid circular dependencies
        from core.events import EventBus, EventType

        self.event_bus = event_bus or EventBus()
        self.broker = broker
        self.config = config or EngineConfig(
            capital=initial_capital,
            position_size_pct=position_size_pct,
            mode=TradingMode.PAPER if broker is None else TradingMode.LIVE
        )

        # Core components
        self.order_manager = OrderManager(
            broker=broker,
            paper_trading=(self.config.mode == TradingMode.PAPER)
        )
        self.position_manager = PositionManager(broker=broker)

        # Thread synchronization
        self._state_lock = threading.RLock()

        # Strategies
        self._strategies: Dict[str, Strategy] = {}
        self._symbols: Dict[str, str] = {}  # strategy -> symbol
        self._strategies_by_symbol: Dict[str, List[str]] = {}  # symbol -> [strategies] for O(1) lookup

        # Data source
        self._data_source = None
        self._data_thread: Optional[threading.Thread] = None

        # State
        self._status = EngineStatus.STOPPED
        self._running = False

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._start_capital = self.config.capital

        # Event handler registration
        self._handler_names: List[str] = []

        # Current prices from events
        self._current_prices: Dict[str, float] = {}

        # Order book imbalance tracking (from Level 2 depth)
        self._current_imbalance: Dict[str, float] = {}
        self._order_flow_analyzer = None  # Lazy init

        # Set paper balance
        if self.config.mode == TradingMode.PAPER:
            self.order_manager.set_paper_balance(self.config.capital)

        logger.info(f"EventDrivenLiveEngine initialized. Mode: {self.config.mode.value}")

    # =========================================================================
    # Strategy Management
    # =========================================================================

    def add_strategy(self, strategy: Strategy, symbol: str):
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance
            symbol: Symbol to trade with this strategy (or comma-separated list)
        """
        name = strategy.name
        self._strategies[name] = strategy
        self._symbols[name] = symbol

        # Maintain reverse index for O(1) lookup during event routing
        # This fixes the O(n) strategy filtering bottleneck
        if symbol not in self._strategies_by_symbol:
            self._strategies_by_symbol[symbol] = []
        if name not in self._strategies_by_symbol[symbol]:
            self._strategies_by_symbol[symbol].append(name)

        # Connect strategy to event bus
        strategy.set_event_bus(self.event_bus)

        logger.info(f"Strategy added: {name} for {symbol}")

    def remove_strategy(self, name: str):
        """Remove a strategy."""
        if name in self._strategies:
            # Get symbol before removing for reverse index cleanup
            symbol = self._symbols.get(name)

            del self._strategies[name]
            del self._symbols[name]

            # Maintain reverse index for O(1) lookup
            if symbol and symbol in self._strategies_by_symbol:
                if name in self._strategies_by_symbol[symbol]:
                    self._strategies_by_symbol[symbol].remove(name)
                # Clean up empty lists
                if not self._strategies_by_symbol[symbol]:
                    del self._strategies_by_symbol[symbol]

            logger.info(f"Strategy removed: {name}")

    def get_strategies(self) -> List[str]:
        """Get list of active strategies."""
        return list(self._strategies.keys())

    # =========================================================================
    # Data Source Management
    # =========================================================================

    def set_data_source(self, data_source):
        """
        Set the data source for live/simulated data.

        Args:
            data_source: LiveDataSource or SimulatedLiveSource instance
        """
        self._data_source = data_source

    def create_simulated_source(
        self,
        symbols: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        tick_interval: float = 1.0
    ):
        """
        Create a simulated data source for testing.

        Args:
            symbols: Symbols to simulate (defaults to strategy symbols)
            base_prices: Starting prices
            tick_interval: Seconds between ticks
        """
        from core.data import SimulatedLiveSource, DataSourceConfig

        if symbols is None:
            symbols = list(set(self._symbols.values()))

        config = DataSourceConfig(
            symbols=symbols,
            timeframe='1m',
            emit_bars=True,
            emit_ticks=False
        )

        self._data_source = SimulatedLiveSource(
            config=config,
            event_bus=self.event_bus,
            base_prices=base_prices,
            tick_interval=tick_interval
        )

        logger.info(f"Created simulated source for {len(symbols)} symbols")

    def create_live_source(
        self,
        symbols: Optional[List[str]] = None,
        api_key: str = "",
        access_token: str = "",
        symbol_to_token: Optional[Dict[str, int]] = None,
        timeframe: str = "1m"
    ):
        """
        Create a live data source.

        Args:
            symbols: Symbols to subscribe to
            api_key: Zerodha API key
            access_token: Zerodha access token
            symbol_to_token: Symbol to instrument token mapping
            timeframe: Bar timeframe
        """
        from core.data import LiveDataSource, DataSourceConfig

        if symbols is None:
            symbols = list(set(self._symbols.values()))

        config = DataSourceConfig(
            symbols=symbols,
            timeframe=timeframe,
            emit_bars=True,
            emit_ticks=False
        )

        self._data_source = LiveDataSource(
            config=config,
            event_bus=self.event_bus,
            api_key=api_key,
            access_token=access_token,
            symbol_to_token=symbol_to_token
        )

        logger.info(f"Created live source for {len(symbols)} symbols")

    # =========================================================================
    # Engine Control
    # =========================================================================

    def start(self):
        """
        Start the event-driven trading engine.

        This subscribes to events and starts the data source.
        """
        with self._state_lock:
            if self._status == EngineStatus.RUNNING:
                logger.warning("Engine already running")
                return

            self._status = EngineStatus.STARTING
            self._running = True

            # Reset daily tracking
            self._daily_pnl = 0.0
            self._daily_trades = 0

        # Subscribe to events
        self._subscribe_events()

        # Start data source in background
        if self._data_source:
            self._start_data_source()

        with self._state_lock:
            self._status = EngineStatus.RUNNING

        logger.info("EventDrivenLiveEngine STARTED")

    def stop(self):
        """Stop the trading engine."""
        with self._state_lock:
            self._running = False
            self._status = EngineStatus.STOPPED

        # Stop data source
        if self._data_source:
            self._data_source.stop()

        # Wait for data thread
        if self._data_thread:
            self._data_thread.join(timeout=5)

        # Unsubscribe from events
        self._unsubscribe_events()

        # Square off if needed
        if self._is_square_off_time():
            self._square_off_all()

        logger.info("EventDrivenLiveEngine STOPPED")

    def pause(self):
        """Pause trading (keeps receiving events but doesn't trade)."""
        with self._state_lock:
            self._status = EngineStatus.PAUSED
        logger.info("EventDrivenLiveEngine PAUSED")

    def resume(self):
        """Resume trading."""
        with self._state_lock:
            self._status = EngineStatus.RUNNING
        logger.info("EventDrivenLiveEngine RESUMED")

    @property
    def status(self) -> EngineStatus:
        """Get engine status."""
        with self._state_lock:
            return self._status

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        with self._state_lock:
            return self._status == EngineStatus.RUNNING

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _subscribe_events(self):
        """Subscribe to trading events."""
        from core.events import EventType

        self._handler_names = []

        # Subscribe to bar events for strategy processing
        name1 = self.event_bus.subscribe(
            EventType.BAR, self._on_bar_event,
            priority=50, name="live_engine_bar"
        )
        self._handler_names.append(name1)

        # Subscribe to tick events for price updates
        name2 = self.event_bus.subscribe(
            EventType.TICK, self._on_tick_event,
            priority=50, name="live_engine_tick"
        )
        self._handler_names.append(name2)

        # Subscribe to signal events (from strategy.emit_signal)
        name3 = self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED, self._on_signal_event,
            priority=100, name="live_engine_signal"
        )
        self._handler_names.append(name3)

        # Subscribe to fill events
        name4 = self.event_bus.subscribe(
            EventType.ORDER_FILLED, self._on_fill_event,
            priority=100, name="live_engine_fill"
        )
        self._handler_names.append(name4)

        logger.debug(f"Subscribed to {len(self._handler_names)} event types")

    def _unsubscribe_events(self):
        """Unsubscribe from events."""
        for name in self._handler_names:
            self.event_bus.unsubscribe(name)
        self._handler_names.clear()

    def _on_bar_event(self, event):
        """Handle bar events - process through strategies."""
        from core.events.events import BarEvent

        if not isinstance(event, BarEvent):
            return

        symbol = event.symbol

        # Update current price
        self._current_prices[symbol] = event.close

        # Check market hours
        if not self._is_market_hours():
            return

        # Check if paused
        if self._status != EngineStatus.RUNNING:
            return

        # Check daily loss limit
        if self._check_daily_loss_limit():
            logger.warning("Daily loss limit reached!")
            self.pause()
            return

        # Check stop-loss and targets for open positions
        self._check_risk(event)

        # Process through strategies - O(1) lookup using reverse index
        # Fixes the "One-Asset Trap": Now processes only strategies for THIS symbol
        # instead of filtering through ALL strategies (O(n) -> O(1))
        for strategy_name in self._strategies_by_symbol.get(symbol, []):
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                continue

            try:
                # Call strategy.on_event() - unified interface
                signal = strategy.on_event(event)

                if signal and signal.signal_type != SignalType.HOLD:
                    self._process_signal(signal, strategy_name, event.close)

            except Exception as e:
                logger.error(f"Strategy error ({strategy_name}): {e}")

    def _on_tick_event(self, event):
        """Handle tick events - update prices and process Level 2 depth."""
        from core.events.events import TickEvent

        if not isinstance(event, TickEvent):
            return

        # Update current price
        self._current_prices[event.symbol] = event.last_price

        # Process Level 2 depth data if available
        if hasattr(event, 'depth') and event.depth:
            self._process_depth(event)

    def _process_depth(self, event):
        """
        Process Level 2 market depth from tick event.

        Calculates order book imbalance and makes it available to strategies.
        Imbalance > 0.3 indicates strong buying pressure (bullish).
        Imbalance < -0.3 indicates strong selling pressure (bearish).
        """
        try:
            # Lazy initialize OrderFlowAnalyzer
            if self._order_flow_analyzer is None:
                from indicators.microstructure import OrderFlowAnalyzer
                self._order_flow_analyzer = OrderFlowAnalyzer()

            # Convert depth to dict format expected by analyzer
            depth_dict = event.depth
            if hasattr(event.depth, 'to_dict'):
                depth_dict = event.depth.to_dict()

            # Calculate imbalance
            imbalance = self._order_flow_analyzer.calculate_imbalance(depth_dict)
            self._current_imbalance[event.symbol] = imbalance

            # Attach imbalance to event for strategy access
            event.imbalance = imbalance

            # Log significant imbalances
            if abs(imbalance) > 0.5:
                direction = "BUY PRESSURE" if imbalance > 0 else "SELL PRESSURE"
                logger.debug(f"{event.symbol}: {direction} (imbalance={imbalance:+.2f})")

        except Exception as e:
            logger.error(f"Error processing depth for {event.symbol}: {e}")

    def get_imbalance(self, symbol: str) -> float:
        """
        Get current order book imbalance for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Imbalance from -1.0 (sell pressure) to +1.0 (buy pressure)
        """
        return self._current_imbalance.get(symbol, 0.0)

    def _on_signal_event(self, event):
        """Handle signal events from strategy.emit_signal()."""
        from core.events.events import SignalEvent, SignalType as EventSignalType

        # Convert EventSignalType to SignalType
        signal_type_map = {
            EventSignalType.BUY: SignalType.BUY,
            EventSignalType.SELL: SignalType.SELL,
            EventSignalType.HOLD: SignalType.HOLD,
            EventSignalType.EXIT: SignalType.EXIT,
        }

        signal = Signal(
            signal_type=signal_type_map.get(event.signal_type, SignalType.HOLD),
            symbol=event.symbol,
            price=event.price,
            stop_loss=event.stop_loss,
            target=event.target,
            confidence=event.confidence,
            reason=event.reason,
            timestamp=event.timestamp
        )

        if signal.signal_type != SignalType.HOLD:
            self._process_signal(signal, event.strategy_name, event.price)

    def _on_fill_event(self, event):
        """Handle fill events."""
        from core.events.events import FillEvent

        if not isinstance(event, FillEvent):
            return

        logger.info(f"Order filled: {event.symbol} {event.side.value} "
                   f"{event.quantity}x @ Rs.{event.price:.2f}")

    # =========================================================================
    # Signal Processing
    # =========================================================================

    def _process_signal(self, signal: Signal, strategy_name: str, current_price: float):
        """Process a trading signal."""
        symbol = signal.symbol

        with self._state_lock:
            has_position = self.position_manager.has_position(symbol)
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
                quantity = int(position_value / current_price)

                if quantity < 1:
                    logger.warning(f"Calculated quantity is 0 for {symbol}")
                    return

                self._execute_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=current_price,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    strategy=strategy_name,
                    reason=signal.reason
                )

            elif signal.signal_type == SignalType.SELL:
                if has_position:
                    self._execute_sell(
                        symbol=symbol,
                        price=current_price,
                        strategy=strategy_name,
                        reason=signal.reason
                    )

            elif signal.signal_type == SignalType.EXIT:
                if has_position:
                    self._execute_sell(
                        symbol=symbol,
                        price=current_price,
                        strategy=strategy_name,
                        reason="Exit signal"
                    )

    # =========================================================================
    # Order Execution
    # =========================================================================

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
        """Execute a buy order and emit events."""
        from core.events import OrderEvent, FillEvent, PositionEvent
        from core.events.events import Side as EventSide, OrderStatus as EventOrderStatus, EventType

        logger.info(f"BUY Signal: {quantity} x {symbol} @ Rs.{price:.2f} | {reason}")

        # Place order via order manager
        order = self.order_manager.buy(
            symbol=symbol,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy
        )

        if order.status == OrderStatus.COMPLETE:
            with self._state_lock:
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

            # Emit order event
            order_event = OrderEvent(
                order_id=order.order_id,
                symbol=symbol,
                side=EventSide.BUY,
                quantity=quantity,
                price=order.average_price,
                status=EventOrderStatus.FILLED,
                filled_quantity=quantity,
                average_price=order.average_price,
                strategy_name=strategy
            )
            self.event_bus.publish(order_event)

            # Emit fill event
            fill_event = FillEvent(
                order_id=order.order_id,
                symbol=symbol,
                side=EventSide.BUY,
                quantity=quantity,
                price=order.average_price,
                strategy_name=strategy
            )
            self.event_bus.publish(fill_event)

            # Emit position event
            position_event = PositionEvent(
                symbol=symbol,
                side=EventSide.BUY,
                quantity=quantity,
                entry_price=order.average_price,
                current_price=order.average_price,
                stop_loss=stop_loss or (price * (1 - self.config.stop_loss_pct / 100)),
                target=target or (price * (1 + self.config.target_pct / 100)),
                strategy_name=strategy
            )
            position_event.event_type = EventType.POSITION_OPENED
            self.event_bus.publish(position_event)

            logger.info(f"BUY executed: {order}")

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        strategy: str = "",
        reason: str = ""
    ):
        """Execute a sell order and emit events."""
        from core.events import OrderEvent, FillEvent, PositionEvent
        from core.events.events import Side as EventSide, OrderStatus as EventOrderStatus, EventType

        with self._state_lock:
            position = self.position_manager.get_position(symbol)
            if not position:
                return
            quantity = position.quantity

        logger.info(f"SELL Signal: {quantity} x {symbol} @ Rs.{price:.2f} | {reason}")

        # Place order
        order = self.order_manager.sell(
            symbol=symbol,
            quantity=quantity,
            price=price,
            strategy=strategy
        )

        if order.status == OrderStatus.COMPLETE:
            with self._state_lock:
                # Close position
                pnl = self.position_manager.close_position(symbol, order.average_price)
                self._daily_pnl += pnl
                self._daily_trades += 1

            # Emit order event
            order_event = OrderEvent(
                order_id=order.order_id,
                symbol=symbol,
                side=EventSide.SELL,
                quantity=quantity,
                price=order.average_price,
                status=EventOrderStatus.FILLED,
                filled_quantity=quantity,
                average_price=order.average_price,
                strategy_name=strategy
            )
            self.event_bus.publish(order_event)

            # Emit fill event
            fill_event = FillEvent(
                order_id=order.order_id,
                symbol=symbol,
                side=EventSide.SELL,
                quantity=quantity,
                price=order.average_price,
                strategy_name=strategy
            )
            self.event_bus.publish(fill_event)

            # Emit position closed event
            position_event = PositionEvent(
                symbol=symbol,
                side=EventSide.SELL,
                quantity=0,
                entry_price=position.entry_price,
                current_price=order.average_price,
                realized_pnl=pnl,
                strategy_name=strategy
            )
            position_event.event_type = EventType.POSITION_CLOSED
            self.event_bus.publish(position_event)

            logger.info(f"SELL executed: {order} | P&L: Rs.{pnl:+.0f}")

    # =========================================================================
    # Risk Management
    # =========================================================================

    def _check_risk(self, bar_event):
        """Check stop-loss and targets for open positions."""
        symbol = bar_event.symbol
        high = bar_event.high
        low = bar_event.low
        close = bar_event.close

        with self._state_lock:
            position = self.position_manager.get_position(symbol)
            if not position:
                return

            # Check stop loss
            if position.stop_loss > 0:
                if position.quantity > 0:  # Long position
                    if low <= position.stop_loss:
                        self._execute_sell(symbol, position.stop_loss, reason="Stop Loss")
                        return
                else:  # Short position
                    if high >= position.stop_loss:
                        self._execute_sell(symbol, position.stop_loss, reason="Stop Loss")
                        return

            # Check target
            if position.target > 0:
                if position.quantity > 0:  # Long position
                    if high >= position.target:
                        self._execute_sell(symbol, position.target, reason="Target Hit")
                        return
                else:  # Short position
                    if low <= position.target:
                        self._execute_sell(symbol, position.target, reason="Target Hit")
                        return

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit reached."""
        with self._state_lock:
            if self._daily_pnl < 0:
                loss_pct = abs(self._daily_pnl) / self._start_capital * 100
                return loss_pct >= self.config.max_daily_loss_pct
            return False

    def _square_off_all(self):
        """Close all positions (end of day)."""
        logger.info("Squaring off all positions...")

        with self._state_lock:
            positions = self.position_manager.get_all_positions()

            for pos in positions:
                price = self._current_prices.get(pos.symbol, pos.last_price)
                self._execute_sell(pos.symbol, price, reason="End of Day Square Off")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _start_data_source(self):
        """Start the data source in a background thread."""
        if not self._data_source:
            return

        def run_data_source():
            try:
                if self._data_source.connect():
                    self._data_source.start()
            except Exception as e:
                logger.error(f"Data source error: {e}")

        self._data_thread = threading.Thread(target=run_data_source, daemon=True)
        self._data_thread.start()

    def _is_market_hours(self) -> bool:
        """Check if market is open."""
        now = datetime.now().time()
        return self.config.market_open <= now <= self.config.market_close

    def _is_square_off_time(self) -> bool:
        """Check if it's time to square off."""
        now = datetime.now().time()
        return now >= self.config.square_off_time

    # =========================================================================
    # Status & Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._state_lock:
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
        """Print current status."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("EVENT-DRIVEN LIVE ENGINE STATUS")
        print("=" * 50)
        print(f"Status: {stats['status']}")
        print(f"Mode: {stats['mode']}")
        print(f"Strategies: {stats['strategies']}")
        print(f"Open Positions: {stats['positions']}")
        print(f"Today's Trades: {stats['daily_trades']}")
        print(f"Today's P&L: Rs.{stats['daily_pnl']:+.0f}")
        print(f"Unrealized P&L: Rs.{stats['unrealized_pnl']:+.0f}")
        print("=" * 50)


# ============== QUICK START HELPERS ==============

def create_event_driven_paper_engine(capital: float = 100000) -> EventDrivenLiveEngine:
    """Create an event-driven paper trading engine."""
    from core.events import EventBus

    bus = EventBus()
    config = EngineConfig(
        mode=TradingMode.PAPER,
        capital=capital
    )
    return EventDrivenLiveEngine(bus, broker=None, config=config)


def create_event_driven_live_engine(
    broker: ZerodhaBroker,
    capital: float = 100000
) -> EventDrivenLiveEngine:
    """Create an event-driven live trading engine."""
    from core.events import EventBus

    bus = EventBus()
    config = EngineConfig(
        mode=TradingMode.LIVE,
        capital=capital
    )
    return EventDrivenLiveEngine(bus, broker=broker, config=config)


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
