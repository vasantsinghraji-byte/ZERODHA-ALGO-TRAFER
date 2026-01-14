# -*- coding: utf-8 -*-
"""
TradingBot - Main coordinator for automated trading
Orchestrates all components: DataManager, StrategyExecutor, OrderManager, RiskManager

Now using Dependency Injection pattern with AppContainer
"""
from typing import List, Dict, Optional, TYPE_CHECKING
import logging
from datetime import datetime
import time
import asyncio

from .data_manager import DataManager
from .strategy_executor import StrategyExecutor
from .order_manager import OrderManager, Order
from .risk_manager import RiskManager, RiskLimits
from .enhanced_strategy import EnhancedSignal, EnhancedTradingStrategy, SignalType

if TYPE_CHECKING:
    from .container import AppContainer

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot coordinator

    Responsibilities:
    - Initialize and coordinate all components
    - Connect components via callbacks
    - Manage bot lifecycle (start, stop, pause)
    - Provide unified API for trading operations
    - Handle component failures gracefully

    Now using Dependency Injection:
    - Accepts AppContainer with all services
    - Components are injected, not created
    - Easier to test and maintain
    """

    def __init__(self, container: 'AppContainer'):
        """
        Initialize TradingBot with dependency injection

        Args:
            container: AppContainer with all injected services
        """
        self.container = container
        self.settings = container.settings

        # Access injected services
        self.kite = container.kite
        self.broker = container.broker
        self.storage = container.storage
        self.notifier = container.notifier

        # Initialize components (now with container dependencies)
        self.data_manager = DataManager(
            self.settings.zerodha_api_key,
            self.settings.zerodha_access_token
        )
        self.strategy_executor = StrategyExecutor(
            EnhancedTradingStrategy(self.settings.account_balance)
        )
        self.order_manager = OrderManager(self.kite)
        self.risk_manager = RiskManager(
            RiskLimits(self.settings.account_balance)
        )

        # Bot state
        self.is_running = False
        self.is_paused = False
        self.start_time: Optional[datetime] = None

        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'errors': 0
        }

        # Connect components
        self._connect_components()

        logger.info("TradingBot initialized successfully")

    def _connect_components(self):
        """Wire components together via callbacks"""

        # DataManager -> StrategyExecutor (ticks -> signals)
        self.data_manager.register_tick_callback(self._handle_ticks)

        # StrategyExecutor -> OrderManager (signals -> orders)
        self.strategy_executor.register_signal_callback(self._handle_signal)

        # OrderManager -> RiskManager (orders -> P&L tracking)
        self.order_manager.register_order_callback(self._handle_order_update)

        logger.info("Components connected via callbacks")

    def start(self, instrument_tokens: List[int], symbols: Dict[int, str]):
        """
        Start the trading bot

        Args:
            instrument_tokens: List of instrument tokens to trade
            symbols: Mapping of instrument_token -> symbol
        """
        try:
            if self.is_running:
                logger.warning("Bot already running")
                return

            logger.info("Starting TradingBot...")
            self.notifier.send_notification(
                f"🤖 Starting bot - monitoring {len(instrument_tokens)} instruments",
                level="info"
            )

            # Store symbols for order placement
            self.symbols = symbols

            # Connect to WebSocket
            self.data_manager.connect()

            # Wait for connection
            time.sleep(2)

            # Subscribe to instruments
            self.data_manager.subscribe(instrument_tokens, mode="full")

            # Add instruments to strategy executor
            for token in instrument_tokens:
                self.strategy_executor.add_instrument(token)

            # Mark as running
            self.is_running = True
            self.start_time = datetime.now()

            logger.info(f"✅ TradingBot started - monitoring {len(instrument_tokens)} instruments")
            self.notifier.send_notification(
                f"✅ Bot started successfully in {self.settings.environment.value} mode",
                level="success"
            )

        except Exception as e:
            logger.error(f"Failed to start TradingBot: {e}")
            self.notifier.send_notification(
                f"❌ Failed to start bot: {str(e)}",
                level="error"
            )
            raise

    def stop(self):
        """Stop the trading bot gracefully"""
        try:
            if not self.is_running:
                logger.warning("Bot not running")
                return

            logger.info("Stopping TradingBot...")

            # Disconnect WebSocket
            self.data_manager.disconnect()

            # Cancel all pending orders
            for order in self.order_manager.get_active_orders():
                self.order_manager.cancel_order(order.order_id)

            # Mark as stopped
            self.is_running = False

            # Log session summary
            self._log_session_summary()

            logger.info("✅ TradingBot stopped")

        except Exception as e:
            logger.error(f"Error stopping TradingBot: {e}")

    def pause(self):
        """Pause trading (stops new orders but keeps monitoring)"""
        self.is_paused = True
        logger.info("TradingBot paused - no new orders will be placed")

    def resume(self):
        """Resume trading after pause"""
        self.is_paused = False
        logger.info("TradingBot resumed")

    # Component callbacks

    def _handle_ticks(self, ticks: List[Dict]):
        """
        Handle ticks from DataManager

        Args:
            ticks: List of tick dictionaries
        """
        try:
            self.stats['ticks_processed'] += len(ticks)

            # Forward to strategy executor
            self.strategy_executor.process_ticks(ticks)

        except Exception as e:
            logger.error(f"Error handling ticks: {e}")
            self.stats['errors'] += 1

    def _handle_signal(self, signal: EnhancedSignal, instrument_token: int):
        """
        Handle trading signal from StrategyExecutor

        Args:
            signal: Trading signal
            instrument_token: Instrument token
        """
        try:
            self.stats['signals_generated'] += 1

            # Get symbol
            symbol = self.symbols.get(instrument_token)
            if not symbol:
                logger.warning(f"Symbol not found for instrument {instrument_token}")
                return

            # Save signal to database
            self.storage.save_signal({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target': signal.take_profit_1,
                'metadata': {
                    'adx': signal.adx,
                    'rsi': signal.rsi,
                    'position_size': signal.position_size
                }
            })

            # Skip if paused
            if self.is_paused:
                logger.info(f"Signal ignored (paused): {signal.signal_type.value} for {symbol}")
                return

            # Skip neutral signals
            if signal.signal_type == SignalType.NEUTRAL:
                return

            # Validate with risk manager
            current_positions = len(self.order_manager.positions)

            is_valid, reason = self.risk_manager.validate_trade(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target=signal.take_profit_1,
                quantity=signal.position_size,
                current_positions=current_positions
            )

            if not is_valid:
                logger.warning(f"Trade rejected by risk manager: {reason}")
                self.notifier.send_notification(
                    f"⚠️ Trade rejected: {symbol} - {reason}",
                    level="warning"
                )
                return

            # Place order
            logger.info(f"Placing order from signal: {signal.signal_type.value} "
                       f"{symbol} @ {signal.entry_price}")

            # Get or create event loop to run async order placement
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a task
                    future = asyncio.run_coroutine_threadsafe(
                        self.order_manager.place_order_from_signal(
                            signal=signal,
                            instrument_token=instrument_token,
                            symbol=symbol,
                            quantity=signal.position_size
                        ),
                        loop
                    )
                    order_id = future.result(timeout=30)
                else:
                    order_id = loop.run_until_complete(
                        self.order_manager.place_order_from_signal(
                            signal=signal,
                            instrument_token=instrument_token,
                            symbol=symbol,
                            quantity=signal.position_size
                        )
                    )
            except RuntimeError:
                # No event loop exists, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    order_id = loop.run_until_complete(
                        self.order_manager.place_order_from_signal(
                            signal=signal,
                            instrument_token=instrument_token,
                            symbol=symbol,
                            quantity=signal.position_size
                        )
                    )
                finally:
                    loop.close()

            if order_id:
                self.stats['orders_placed'] += 1
                logger.info(f"✅ Order placed: {order_id}")
                self.notifier.send_notification(
                    f"📈 {signal.signal_type.value} order placed: {symbol} @ ₹{signal.entry_price:.2f}",
                    level="success"
                )
            else:
                logger.error("Failed to place order")
                self.notifier.send_notification(
                    f"❌ Failed to place order: {symbol}",
                    level="error"
                )

        except Exception as e:
            logger.error(f"Error handling signal: {e}")
            self.stats['errors'] += 1
            self.notifier.send_notification(
                f"❌ Error handling signal: {str(e)}",
                level="error"
            )

    def _handle_order_update(self, order: Order):
        """
        Handle order update from OrderManager

        Args:
            order: Updated order
        """
        try:
            logger.info(f"Order update: {order.order_id} - {order.status.value}")

            # Update positions
            self.order_manager.update_positions()

            # Calculate P&L if order completed
            if order.status.value == "COMPLETE":
                # Get current position
                position = self.order_manager.get_position(order.instrument_token)

                if position:
                    pnl = position.get('pnl', 0)
                    self.risk_manager.update_pnl(realized_pnl=pnl)
                    self.risk_manager.record_trade(pnl)

        except Exception as e:
            logger.error(f"Error handling order update: {e}")
            self.stats['errors'] += 1

    # Public API methods

    def add_instrument(self, instrument_token: int, symbol: str, historical_data=None):
        """
        Add instrument to monitor

        Args:
            instrument_token: Instrument token
            symbol: Trading symbol
            historical_data: Historical OHLC data (optional)
        """
        try:
            # Add to symbols mapping
            self.symbols[instrument_token] = symbol

            # Subscribe via DataManager
            if self.is_running:
                self.data_manager.subscribe([instrument_token], mode="full")

            # Add to StrategyExecutor
            self.strategy_executor.add_instrument(instrument_token, historical_data)

            logger.info(f"Added instrument: {symbol} ({instrument_token})")

        except Exception as e:
            logger.error(f"Failed to add instrument: {e}")

    def remove_instrument(self, instrument_token: int):
        """
        Remove instrument from monitoring

        Args:
            instrument_token: Instrument token
        """
        try:
            # Unsubscribe from DataManager
            if self.is_running:
                self.data_manager.unsubscribe([instrument_token])

            # Remove from StrategyExecutor
            self.strategy_executor.remove_instrument(instrument_token)

            # Remove from symbols
            symbol = self.symbols.pop(instrument_token, None)

            logger.info(f"Removed instrument: {symbol} ({instrument_token})")

        except Exception as e:
            logger.error(f"Failed to remove instrument: {e}")

    def get_status(self) -> Dict:
        """
        Get bot status and statistics

        Returns:
            Dictionary with bot status
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'uptime_seconds': uptime,
            'stats': self.stats,
            'data_manager': self.data_manager.get_stats(),
            'strategy_executor': self.strategy_executor.get_stats(),
            'order_manager': self.order_manager.get_stats(),
            'risk_manager': self.risk_manager.get_stats()
        }

    def get_risk_report(self) -> Dict:
        """Get detailed risk report"""
        return self.risk_manager.get_risk_report()

    def _log_session_summary(self):
        """Log session summary on stop"""
        try:
            if not self.start_time:
                return

            duration = (datetime.now() - self.start_time).total_seconds()
            risk_report = self.risk_manager.get_risk_report()

            logger.info("="*70)
            logger.info("SESSION SUMMARY")
            logger.info("="*70)
            logger.info(f"Duration: {duration/60:.1f} minutes")
            logger.info(f"Ticks Processed: {self.stats['ticks_processed']:,}")
            logger.info(f"Signals Generated: {self.stats['signals_generated']}")
            logger.info(f"Orders Placed: {self.stats['orders_placed']}")
            logger.info(f"Trades Completed: {risk_report['trades_today']}")
            logger.info(f"Win Rate: {risk_report['win_rate']:.1f}%")
            logger.info(f"Daily P&L: ₹{risk_report['daily_pnl']:.2f}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"Error logging session summary: {e}")
