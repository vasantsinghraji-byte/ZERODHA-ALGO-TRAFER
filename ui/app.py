# -*- coding: utf-8 -*-
"""
AlgoTrader Pro - Main Application
=================================
A beautiful trading app that anyone can use!

Features:
- Big colorful buttons
- Simple language
- One-click trading
- Real-time updates
- Multiple views (Dashboard, Charts, Strategies, Scanner, Portfolio, Settings)
- AI/ML Predictions
- Telegram/Email Alerts
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue  # Thread-safe communication for UI updates
import webbrowser
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
import logging
import os
import json
import numpy as np
import pandas as pd

from .themes import get_theme, THEMES
from .dashboard import Dashboard
from .charts import (
    CandlestickChart, SimpleChart, add_moving_average, add_bollinger_bands,
    add_ema, add_rsi, add_macd, add_vwap, add_supertrend
)
from .strategy_picker import StrategyPicker, STRATEGY_INFO
from .settings_panel import SettingsPanel, SettingsDialog
from .stock_search import StockSearchWidget, QuickStockSelector
from .infrastructure_panel import (
    KillSwitchButton,
    SystemHealthCard,
    InfrastructureMonitor,
    InfrastructureManagerWidget,
)
from .order_panel import OrderPanel
from .automation_panel import AutomationPanel

logger = logging.getLogger(__name__)


def _load_trading_symbols() -> list:
    """Load trading symbols from watchlist configuration."""
    try:
        from config.loader import get_watchlist
        watchlist = get_watchlist()
        all_symbols = []

        # Collect from all watchlists
        for key in ['nifty50', 'banknifty', 'custom']:
            symbols = watchlist.get(key, [])
            if symbols:
                for sym in symbols:
                    if sym not in all_symbols:
                        all_symbols.append(sym)

        # Extract just symbol names (remove exchange prefix for sample data keys)
        symbol_names = []
        for sym in all_symbols:
            if ':' in sym:
                _, name = sym.split(':', 1)
                if name not in symbol_names:
                    symbol_names.append(name)
            else:
                if sym not in symbol_names:
                    symbol_names.append(sym)

        return symbol_names[:20] if symbol_names else ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

    except Exception as e:
        logger.warning(f"Could not load symbols from watchlist: {e}")
        return ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']


# Default symbols loaded dynamically from watchlist
DEFAULT_TRADING_SYMBOLS = _load_trading_symbols()


class NavigationButton:
    """Navigation button for sidebar"""

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        text: str,
        emoji: str,
        command=None
    ):
        self.theme = theme
        self.selected = False

        self.frame = tk.Frame(parent, bg=theme['bg_secondary'], cursor='hand2')

        self.label = tk.Label(
            self.frame,
            text=f"{emoji}  {text}",
            bg=theme['bg_secondary'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 11),
            anchor=tk.W,
            padx=15,
            pady=10
        )
        self.label.pack(fill=tk.X)

        self.frame.bind('<Button-1>', lambda e: command() if command else None)
        self.label.bind('<Button-1>', lambda e: command() if command else None)

        self.frame.bind('<Enter>', self._on_hover)
        self.frame.bind('<Leave>', self._on_leave)
        self.label.bind('<Enter>', self._on_hover)
        self.label.bind('<Leave>', self._on_leave)

    def _on_hover(self, event=None):
        if not self.selected:
            self.frame.config(bg=self.theme['bg_card'])
            self.label.config(bg=self.theme['bg_card'])

    def _on_leave(self, event=None):
        if not self.selected:
            self.frame.config(bg=self.theme['bg_secondary'])
            self.label.config(bg=self.theme['bg_secondary'])

    def set_selected(self, selected: bool):
        self.selected = selected
        if selected:
            self.frame.config(bg=self.theme['accent'])
            self.label.config(bg=self.theme['accent'], fg='white')
        else:
            self.frame.config(bg=self.theme['bg_secondary'])
            self.label.config(bg=self.theme['bg_secondary'], fg=self.theme['text_secondary'])

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class AlgoTraderApp:
    """
    Main Trading Application

    Simple enough for a 5th grader, powerful enough for professionals!

    Can be instantiated with pre-initialized components from bootstrap:
        app = AlgoTraderApp(
            event_bus=registry.event_bus,
            trading_engine=registry.trading_engine,
            infrastructure_manager=registry.infrastructure_manager
        )

    Or without (legacy mode):
        app = AlgoTraderApp()
    """

    def __init__(
        self,
        event_bus=None,
        trading_engine=None,
        infrastructure_manager=None,
        engine_process=None  # NEW: Multiprocessing engine wrapper
    ):
        """
        Initialize AlgoTrader application.

        Args:
            event_bus: Optional pre-initialized EventBus instance
            trading_engine: Optional pre-initialized EventDrivenLiveEngine instance
            infrastructure_manager: Optional pre-initialized InfrastructureManager instance
            engine_process: Optional EngineProcess for multiprocessing mode (bypasses GIL)

        Modes:
            1. Multiprocessing mode (engine_process): Engine runs in separate process
               - Best for high-frequency data (100+ ticks/sec)
               - UI stays responsive during heavy load
               - Recommended for production

            2. Threading mode (event_bus + trading_engine): Engine runs in same process
               - Simpler architecture, easier debugging
               - OK for low-medium frequency trading

            3. Legacy mode (nothing provided): Lazy initialization
               - For backward compatibility
        """
        self.root = tk.Tk()
        self.root.title("AlgoTrader Pro")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.85))
        window_height = min(900, int(screen_height * 0.85))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.current_theme = 'dark'
        self.theme = get_theme(self.current_theme)
        self.root.configure(bg=self.theme['bg_primary'])

        # Thread-safe UI update queue
        # Tkinter is NOT thread-safe. All UI updates from background threads
        # must go through this queue to be processed on the main thread.
        self.ui_queue: queue.Queue = queue.Queue()

        # NEW: Multiprocessing engine (bypasses GIL)
        self._engine_process = engine_process
        self._multiprocessing_mode = engine_process is not None

        # Core components (from bootstrap or lazy-initialized)
        self._event_bus = event_bus
        self._trading_engine = trading_engine
        self._infrastructure_manager = infrastructure_manager
        self._components_initialized = all([event_bus, trading_engine, infrastructure_manager])

        if self._multiprocessing_mode:
            logger.info("AlgoTraderApp initialized with MULTIPROCESSING engine (GIL bypassed)")
        elif self._components_initialized:
            logger.info("AlgoTraderApp initialized with pre-loaded components (threading mode)")
        else:
            logger.info("AlgoTraderApp initialized in legacy mode (lazy component loading)")

        # State
        self.connected = False
        self.bot_running = False
        self.paper_trading = True
        self.balance = 100000.0
        self.todays_pnl = 0.0
        self.selected_strategy = 'turtle'
        self.current_view = 'dashboard'

        # Sample data for demo
        self.sample_data = self._generate_sample_data()

        # Settings
        self.settings = self._load_settings()

        # Alert manager
        self.alert_manager = None
        self._init_alerts()

        # Broker
        self.broker = None

        # Views
        self.views: Dict[str, tk.Frame] = {}
        self.nav_buttons: Dict[str, NavigationButton] = {}

        # Build UI
        self._create_styles()
        self._create_ui()

        # Start update loop
        self._start_updates()

        # Setup EventBus -> Tkinter bridge (if components initialized)
        if self._components_initialized:
            self._setup_event_listeners()

    def _setup_event_listeners(self):
        """
        Bridge EventBus events to Tkinter's main thread.

        CRITICAL: Tkinter is NOT thread-safe. EventBus events come from
        background threads (data feeds, trading engine). This method:
        1. Subscribes to relevant events on the EventBus
        2. Queues UI updates to be processed on the main thread
        3. Enables real-time UI updates from live trading

        This is the "Nervous System" that connects the engine to the GUI.
        """
        if self._event_bus is None:
            logger.warning("Cannot setup event listeners: EventBus not available")
            return

        try:
            from core.events import EventType

            # Subscribe to tick events (price updates)
            self._event_bus.subscribe(
                EventType.TICK,
                self._on_tick_event,
                name="ui_tick_handler"
            )

            # Subscribe to bar events (candle updates)
            self._event_bus.subscribe(
                EventType.BAR,
                self._on_bar_event,
                name="ui_bar_handler"
            )

            # Subscribe to order events
            self._event_bus.subscribe(
                EventType.ORDER_SUBMITTED,
                self._on_order_event,
                name="ui_order_submitted"
            )
            self._event_bus.subscribe(
                EventType.ORDER_FILLED,
                self._on_order_event,
                name="ui_order_filled"
            )
            self._event_bus.subscribe(
                EventType.ORDER_REJECTED,
                self._on_order_event,
                name="ui_order_rejected"
            )

            # Subscribe to signal events (strategy signals)
            self._event_bus.subscribe(
                EventType.SIGNAL_GENERATED,
                self._on_signal_event,
                name="ui_signal_handler"
            )

            # Subscribe to position events
            self._event_bus.subscribe(
                EventType.POSITION_OPENED,
                self._on_position_event,
                name="ui_position_opened"
            )
            self._event_bus.subscribe(
                EventType.POSITION_CLOSED,
                self._on_position_event,
                name="ui_position_closed"
            )

            logger.info("EventBus -> Tkinter bridge established (6 event types)")

            # Update automation panel with event bus if it exists
            if hasattr(self, 'automation_panel') and self.automation_panel:
                self.automation_panel.set_event_bus(self._event_bus)
                self.automation_panel.set_trading_engine(self.trading_engine)
                logger.info("Automation panel connected to EventBus")

        except Exception as e:
            logger.error(f"Failed to setup event listeners: {e}")

    def _on_tick_event(self, event):
        """Handle tick events from EventBus - update prices in UI."""
        def update():
            try:
                symbol = getattr(event, 'symbol', 'Unknown')
                price = getattr(event, 'last_price', 0)
                # Update dashboard price display
                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.update_price(symbol, price)
            except Exception as e:
                logger.debug(f"Tick UI update error: {e}")
        self.ui_queue.put(update)

    def _on_bar_event(self, event):
        """Handle bar events from EventBus - update charts."""
        def update():
            try:
                symbol = getattr(event, 'symbol', 'Unknown')
                # Log for now, chart update would go here
                logger.debug(f"Bar event: {symbol}")
            except Exception as e:
                logger.debug(f"Bar UI update error: {e}")
        self.ui_queue.put(update)

    def _on_order_event(self, event):
        """Handle order events from EventBus - show notifications."""
        def update():
            try:
                event_type = type(event).__name__
                symbol = getattr(event, 'symbol', 'Unknown')
                order_id = getattr(event, 'order_id', 'N/A')

                if 'Filled' in event_type:
                    msg = f"Order FILLED: {symbol} (ID: {order_id})"
                    level = 'success'
                elif 'Rejected' in event_type:
                    msg = f"Order REJECTED: {symbol} (ID: {order_id})"
                    level = 'danger'
                else:
                    msg = f"Order submitted: {symbol} (ID: {order_id})"
                    level = 'info'

                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.log_activity(msg, level)

                # Show alert for fills/rejects
                if self.alert_manager and 'Filled' in event_type:
                    self.alert_manager.trade_alert(symbol, "Order Filled", str(order_id))

            except Exception as e:
                logger.debug(f"Order UI update error: {e}")
        self.ui_queue.put(update)

    def _on_signal_event(self, event):
        """Handle signal events from EventBus - log strategy signals."""
        def update():
            try:
                symbol = getattr(event, 'symbol', 'Unknown')
                signal_type = getattr(event, 'signal_type', 'HOLD')
                price = getattr(event, 'price', 0)

                msg = f"Signal: {signal_type.name if hasattr(signal_type, 'name') else signal_type} on {symbol} @ Rs.{price:.2f}"

                if hasattr(self, 'dashboard') and self.dashboard:
                    level = 'success' if 'BUY' in str(signal_type) else 'warning' if 'SELL' in str(signal_type) else 'info'
                    self.dashboard.log_activity(msg, level)

            except Exception as e:
                logger.debug(f"Signal UI update error: {e}")
        self.ui_queue.put(update)

    def _on_position_event(self, event):
        """Handle position events from EventBus - update portfolio."""
        def update():
            try:
                event_type = type(event).__name__
                symbol = getattr(event, 'symbol', 'Unknown')
                quantity = getattr(event, 'quantity', 0)
                pnl = getattr(event, 'pnl', 0)

                if 'Closed' in event_type:
                    pnl_str = f"+Rs.{pnl:.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):.2f}"
                    msg = f"Position CLOSED: {symbol} ({quantity} qty) P&L: {pnl_str}"
                    level = 'success' if pnl >= 0 else 'danger'
                else:
                    msg = f"Position OPENED: {symbol} ({quantity} qty)"
                    level = 'info'

                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.log_activity(msg, level)

                # Update P&L display
                self.todays_pnl += pnl
                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.update_pnl(self.todays_pnl)

            except Exception as e:
                logger.debug(f"Position UI update error: {e}")
        self.ui_queue.put(update)

    @property
    def event_bus(self):
        """Get the EventBus instance (lazy-load if not provided)."""
        if self._event_bus is None:
            try:
                from core.events import EventBus
                self._event_bus = EventBus()
                logger.debug("EventBus lazy-initialized")
            except ImportError:
                logger.warning("EventBus not available")
        return self._event_bus

    @property
    def trading_engine(self):
        """Get the TradingEngine instance (lazy-load if not provided)."""
        if self._trading_engine is None:
            try:
                from core.trading_engine import EventDrivenLiveEngine
                self._trading_engine = EventDrivenLiveEngine(
                    event_bus=self.event_bus,
                    broker=None,
                    initial_capital=self.balance
                )
                logger.debug("EventDrivenLiveEngine lazy-initialized")
            except ImportError:
                logger.warning("EventDrivenLiveEngine not available")
        return self._trading_engine

    @property
    def infrastructure_manager(self):
        """Get the InfrastructureManager instance (lazy-load if not provided)."""
        if self._infrastructure_manager is None:
            try:
                from core.infrastructure import get_infrastructure_manager
                self._infrastructure_manager = get_infrastructure_manager()
                if self._infrastructure_manager is None:
                    logger.debug("InfrastructureManager not yet initialized globally")
            except ImportError:
                logger.warning("InfrastructureManager not available")
        return self._infrastructure_manager

    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data for stocks from watchlist"""
        np.random.seed(42)

        # Base parameters for common stocks (used if available in watchlist)
        base_params = {
            'RELIANCE': (2500, 0.02, 0.0005),
            'TCS': (3500, 0.018, 0.0004),
            'INFY': (1500, 0.022, 0.0003),
            'HDFCBANK': (1600, 0.015, 0.0003),
            'ICICIBANK': (950, 0.025, 0.0002),
            'SBIN': (600, 0.03, 0.0004),
            'BHARTIARTL': (1200, 0.02, 0.0003),
            'ITC': (450, 0.012, 0.0002),
            'KOTAKBANK': (1800, 0.018, 0.0003),
            'LT': (2800, 0.02, 0.0004),
            'AXISBANK': (1100, 0.025, 0.0002),
            'HDFC': (2800, 0.015, 0.0003),
            'MARUTI': (10000, 0.018, 0.0003),
            'ASIANPAINT': (3200, 0.015, 0.0002),
            'HCLTECH': (1300, 0.022, 0.0003),
            'WIPRO': (450, 0.02, 0.0002),
            'SUNPHARMA': (1100, 0.025, 0.0003),
            'TITAN': (3000, 0.02, 0.0004),
            'BAJFINANCE': (7000, 0.025, 0.0003),
            'TATAMOTORS': (700, 0.03, 0.0004),
        }

        # Use dynamic symbols from watchlist
        symbols_to_generate = DEFAULT_TRADING_SYMBOLS[:15]

        # Build stock parameters - use base params if available, generate random otherwise
        stocks = {}
        for symbol in symbols_to_generate:
            if symbol in base_params:
                stocks[symbol] = base_params[symbol]
            else:
                # Generate random but reasonable parameters for unknown stocks
                base_price = np.random.randint(100, 5000)
                volatility = 0.015 + np.random.random() * 0.015
                trend = (np.random.random() - 0.5) * 0.001
                stocks[symbol] = (base_price, volatility, trend)

        data = {}
        for symbol, (base, vol, trend) in stocks.items():
            days = 100
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            returns = np.random.randn(days) * vol + trend
            prices = base * np.exp(np.cumsum(returns))

            # Use cache key format: symbol_timeframe
            cache_key = f"{symbol}_1 Day"
            data[cache_key] = pd.DataFrame({
                'open': prices * (1 + np.random.randn(days) * 0.005),
                'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
                'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
                'close': prices,
                'volume': np.random.randint(100000, 2000000, days)
            }, index=dates)

        return data

    def _init_alerts(self):
        """Initialize alert manager"""
        try:
            from advanced.alerts import AlertManager
            self.alert_manager = AlertManager()
            self.alert_manager.start()
            logger.info("Alert manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize alerts: {e}")

    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from file.

        SECURITY: API credentials are NOT loaded from file - they should
        come from environment variables via CredentialsManager.
        """
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ui_settings.json')
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    loaded = json.load(f)
                    # SECURITY: Remove any credentials that might be in file
                    if 'api' in loaded:
                        loaded['api'] = {'api_key': '', 'api_secret': '', 'user_id': ''}
                    return loaded
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")

        return {
            'api': {'api_key': '', 'api_secret': '', 'user_id': ''},
            'trading': {'paper_trading': True, 'initial_capital': 100000, 'max_positions': 5},
            'risk': {'risk_per_trade': 2.0, 'max_daily_loss': 5.0},
            'appearance': {'theme': 'dark', 'show_emojis': True}
        }

    def _save_settings(self, settings: Dict[str, Any]):
        """
        Save settings to file.

        SECURITY: API credentials are NEVER saved to file - they should
        be stored in .env via CredentialsManager.
        """
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ui_settings.json')
        try:
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)

            # SECURITY: Create a copy without credentials
            safe_settings = settings.copy()
            if 'api' in safe_settings:
                # Strip credentials - only keep non-sensitive fields
                safe_settings['api'] = {'user_id': settings.get('api', {}).get('user_id', '')}

            with open(settings_path, 'w') as f:
                json.dump(safe_settings, f, indent=2)
            self.settings = settings  # Keep full settings in memory
            logger.info("Settings saved (credentials excluded)")
        except Exception as e:
            logger.error(f"Could not save settings: {e}")

    def _create_styles(self):
        """Create ttk styles"""
        style = ttk.Style()
        style.configure('Dark.TFrame', background=self.theme['bg_primary'])
        style.configure('Card.TFrame', background=self.theme['bg_card'])
        style.configure('Title.TLabel',
                       background=self.theme['bg_primary'],
                       foreground=self.theme['text_primary'],
                       font=('Segoe UI', 24, 'bold'))

    def _create_ui(self):
        """Build the user interface"""
        main_container = tk.Frame(self.root, bg=self.theme['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True)

        self._create_sidebar(main_container)

        self.content_frame = tk.Frame(main_container, bg=self.theme['bg_primary'])
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_header(self.content_frame)

        self.view_container = tk.Frame(self.content_frame, bg=self.theme['bg_primary'])
        self.view_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self._create_views()
        self._show_view('dashboard')

    def _create_sidebar(self, parent):
        """Create navigation sidebar"""
        sidebar = tk.Frame(parent, bg=self.theme['bg_secondary'], width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Logo
        logo_frame = tk.Frame(sidebar, bg=self.theme['bg_secondary'])
        logo_frame.pack(fill=tk.X, pady=15)

        tk.Label(
            logo_frame, text="AlgoTrader",
            bg=self.theme['bg_secondary'],
            fg=self.theme['accent'],
            font=('Segoe UI', 18, 'bold')
        ).pack()

        tk.Label(
            logo_frame, text="PRO",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        ).pack()

        tk.Frame(sidebar, bg=self.theme['border'], height=1).pack(fill=tk.X, padx=15, pady=10)

        # Navigation - now with 10 items including automation, orders and infrastructure
        nav_items = [
            ('dashboard', 'Dashboard', '📊'),
            ('automation', 'Automation', '🤖'),
            ('orders', 'Orders', '📝'),
            ('charts', 'Charts', '📈'),
            ('scanner', 'Scanner', '🔍'),
            ('predictions', 'AI Predict', '🧠'),
            ('portfolio', 'Portfolio', '💼'),
            ('strategies', 'Strategies', '🎯'),
            ('infrastructure', 'Infrastructure', '🏗️'),
            ('settings', 'Settings', '⚙️'),
        ]

        for key, text, emoji in nav_items:
            btn = NavigationButton(
                sidebar, self.theme, text, emoji,
                command=lambda k=key: self._show_view(k)
            )
            btn.pack(fill=tk.X, padx=10, pady=2)
            self.nav_buttons[key] = btn

        tk.Frame(sidebar, bg=self.theme['bg_secondary']).pack(fill=tk.BOTH, expand=True)

        # Bottom section
        bottom = tk.Frame(sidebar, bg=self.theme['bg_secondary'])
        bottom.pack(fill=tk.X, padx=15, pady=15)

        self.conn_indicator = tk.Label(
            bottom, text="● Disconnected",
            bg=self.theme['bg_secondary'],
            fg=self.theme['danger'],
            font=('Segoe UI', 10)
        )
        self.conn_indicator.pack(anchor=tk.W)

        self.mode_indicator = tk.Label(
            bottom, text="Paper Trading",
            bg=self.theme['bg_secondary'],
            fg=self.theme['info'],
            font=('Segoe UI', 9)
        )
        self.mode_indicator.pack(anchor=tk.W)

    def _create_header(self, parent):
        """Create header with controls"""
        header = tk.Frame(parent, bg=self.theme['bg_primary'])
        header.pack(fill=tk.X, padx=20, pady=15)

        self.view_title = tk.Label(
            header, text="Dashboard",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 24, 'bold')
        )
        self.view_title.pack(side=tk.LEFT)

        controls = tk.Frame(header, bg=self.theme['bg_primary'])
        controls.pack(side=tk.RIGHT)

        # Kill Switch Button (always visible for safety)
        self.kill_switch_btn = KillSwitchButton(
            controls, self.theme,
            on_trigger=self._on_kill_switch,
            compact=True
        )
        self.kill_switch_btn.pack(side=tk.LEFT, padx=(0, 15))

        self.bot_btn = tk.Button(
            controls, text="▶ START BOT",
            bg=self.theme['btn_success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._toggle_bot
        )
        self.bot_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=15, ipady=5)

        self.login_btn = tk.Button(
            controls, text="🔐 Login",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._show_login
        )
        self.login_btn.pack(side=tk.LEFT, ipadx=10, ipady=5)

    def _create_views(self):
        """Create all application views"""
        # Dashboard View
        dashboard_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self.dashboard = Dashboard(dashboard_frame, self.current_theme)
        self.dashboard.pack(fill=tk.BOTH, expand=True)
        self.views['dashboard'] = dashboard_frame

        # Automation View (NEW - See the trading loop in action!)
        automation_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_automation_view(automation_frame)
        self.views['automation'] = automation_frame

        # Orders View (Manual Order Placement)
        orders_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_orders_view(orders_frame)
        self.views['orders'] = orders_frame

        # Charts View
        charts_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_charts_view(charts_frame)
        self.views['charts'] = charts_frame

        # Scanner View (NEW)
        scanner_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_scanner_view(scanner_frame)
        self.views['scanner'] = scanner_frame

        # AI Predictions View (NEW)
        predictions_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_predictions_view(predictions_frame)
        self.views['predictions'] = predictions_frame

        # Portfolio View (NEW)
        portfolio_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_portfolio_view(portfolio_frame)
        self.views['portfolio'] = portfolio_frame

        # Strategies View
        strategies_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self.strategy_picker = StrategyPicker(
            strategies_frame,
            self.current_theme,
            on_strategy_selected=self._on_strategy_selected
        )
        self.strategy_picker.pack(fill=tk.BOTH, expand=True)
        self.views['strategies'] = strategies_frame

        # Infrastructure View (NEW)
        infrastructure_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self._create_infrastructure_view(infrastructure_frame)
        self.views['infrastructure'] = infrastructure_frame

        # Settings View
        settings_frame = tk.Frame(self.view_container, bg=self.theme['bg_primary'])
        self.settings_panel = SettingsPanel(
            settings_frame,
            self.current_theme,
            on_save=self._save_settings,
            initial_settings=self.settings
        )
        self.settings_panel.pack(fill=tk.BOTH, expand=True)
        self.views['settings'] = settings_frame

    def _create_charts_view(self, parent):
        """Create charts view with searchable stock selector"""
        # Controls row
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        # Stock search widget
        search_frame = tk.Frame(controls, bg=self.theme['bg_primary'])
        search_frame.pack(side=tk.LEFT)

        tk.Label(
            search_frame,
            text="🔍 Search Stock:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.chart_stock_search = StockSearchWidget(
            search_frame,
            self.theme,
            on_select=self._on_chart_symbol_selected,
            placeholder="Type symbol...",
            show_exchange=False,
            width=18
        )
        self.chart_stock_search.pack(side=tk.LEFT)

        # Set default selection (don't trigger callback during init)
        if self.sample_data:
            first_symbol = list(self.sample_data.keys())[0]
            self.chart_stock_search.set_selected(f"NSE:{first_symbol}", trigger_callback=False)
            self._selected_chart_symbol = first_symbol
        else:
            self._selected_chart_symbol = 'RELIANCE'

        # Timeframe selector
        tk.Label(
            controls,
            text="     Timeframe:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT)

        self.chart_timeframe = ttk.Combobox(
            controls, values=['1 Min', '5 Min', '15 Min', '1 Hour', '1 Day'],
            state='readonly', width=10
        )
        self.chart_timeframe.set('1 Day')
        self.chart_timeframe.pack(side=tk.LEFT, padx=(5, 20))

        # Load chart button
        tk.Button(
            controls,
            text="📊 Load Chart",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._load_chart
        ).pack(side=tk.LEFT, ipadx=12, ipady=4)

        # Second row for indicators
        indicator_row = tk.Frame(parent, bg=self.theme['bg_primary'])
        indicator_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            indicator_row,
            text="Indicators:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Price overlay indicators
        self.show_ma = tk.BooleanVar(value=True)
        tk.Checkbutton(
            indicator_row, text="MA(20)", variable=self.show_ma,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        self.show_ema = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="EMA(20)", variable=self.show_ema,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        self.show_bb = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="Bollinger", variable=self.show_bb,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        self.show_vwap = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="VWAP", variable=self.show_vwap,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        self.show_supertrend = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="SuperTrend", variable=self.show_supertrend,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        # Separator
        tk.Label(
            indicator_row, text=" | ",
            bg=self.theme['bg_primary'], fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        # Separate panel indicators
        self.show_rsi = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="RSI", variable=self.show_rsi,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        self.show_macd = tk.BooleanVar(value=False)
        tk.Checkbutton(
            indicator_row, text="MACD", variable=self.show_macd,
            bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=3)

        # Zoom hint
        tk.Label(
            indicator_row,
            text="      Use toolbar below chart to Zoom/Pan",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9, 'italic')
        ).pack(side=tk.RIGHT, padx=10)

        # Chart container
        self.chart_container = tk.Frame(parent, bg=self.theme['bg_card'])
        self.chart_container.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        self.chart_container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.chart_container,
            text="🔍 Search for a stock above, then click 'Load Chart'\n\nFeatures:\n• Zoom: Use scroll wheel or toolbar buttons\n• Pan: Click and drag\n• Reset: Home button in toolbar",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 12),
            justify=tk.CENTER
        ).pack(expand=True)

    def _on_chart_symbol_selected(self, symbol: str):
        """Handle stock selection from search widget"""
        # Extract symbol name without exchange
        if ':' in symbol:
            _, name = symbol.split(':', 1)
        else:
            name = symbol
        self._selected_chart_symbol = name
        logger.debug(f"Chart symbol selected: {name}")

    def _create_automation_view(self, parent):
        """Create automation view showing the live trading loop.

        Displays:
        - Trading loop phases: Connect -> Listen -> Process -> Decide -> Act
        - Live event stream: Ticks -> Bars -> Signals -> Orders
        - Session metrics: Events/sec, latency, fills
        - Active strategies and their symbol mappings

        This is the "heart" of the algo bot - you can see it beating!
        """
        self.automation_panel = AutomationPanel(
            parent,
            theme_name=self.current_theme,
            event_bus=self.event_bus,
            trading_engine=self.trading_engine,
            on_start=self._on_automation_start,
            on_stop=self._on_automation_stop
        )
        self.automation_panel.pack(fill=tk.BOTH, expand=True)

    def _on_automation_start(self):
        """Handle automation start from automation panel."""
        if not self.connected and not self.paper_trading:
            messagebox.showwarning("Not Connected", "Please login to Zerodha first!")
            return

        try:
            self._start_trading_engine()

            # Update UI to reflect running state
            self.bot_running = True
            self.bot_btn.config(text="⏹ STOP BOT", bg=self.theme['btn_danger'])
            self.dashboard.update_bot_status(True)
            self.dashboard.log_activity(
                f"Automation started with {self.selected_strategy.upper()} strategy",
                'success'
            )

            if self.alert_manager:
                self.alert_manager.bot_status("Started", f"Using {self.selected_strategy} strategy")

        except Exception as e:
            logger.error(f"Failed to start automation: {e}")
            self.dashboard.log_activity(f"Failed to start: {e}", 'danger')
            messagebox.showerror("Error", f"Failed to start automation: {e}")

    def _on_automation_stop(self):
        """Handle automation stop from automation panel."""
        try:
            self._stop_trading_engine()

            # Update UI to reflect stopped state
            self.bot_running = False
            self.bot_btn.config(text="▶ START BOT", bg=self.theme['btn_success'])
            self.dashboard.update_bot_status(False)
            self.dashboard.log_activity("Automation stopped", 'warning')

            if self.alert_manager:
                self.alert_manager.bot_status("Stopped", "Trading halted")

        except Exception as e:
            logger.error(f"Failed to stop automation: {e}")

    def _create_orders_view(self, parent):
        """Create orders view for manual order placement

        Supports:
        - Basic Orders: Market, Limit, Stop Loss, SL-Market
        - Advanced Orders: Bracket, Iceberg, TWAP, VWAP

        Order manager is connected when user logs in.
        """
        self.order_panel = OrderPanel(
            parent,
            theme_name=self.current_theme,
            order_manager=getattr(self, '_order_manager', None),
            broker=self.broker,
            on_order_placed=self._on_order_placed
        )
        self.order_panel.pack(fill=tk.BOTH, expand=True)

    def _on_order_placed(self, order_data: dict):
        """Handle order placed from order panel"""
        # Log to dashboard activity feed
        if hasattr(self, 'dashboard') and self.dashboard:
            side = order_data.get('side', 'BUY')
            symbol = order_data.get('symbol', 'Unknown')
            qty = order_data.get('quantity', 0)
            order_type = order_data.get('order_type', 'MARKET')
            status = order_data.get('status', 'PENDING')

            if status == 'COMPLETE':
                msg_type = 'success'
                msg = f"{side} {qty} {symbol} ({order_type}) - FILLED"
            elif status == 'REJECTED':
                msg_type = 'danger'
                msg = f"{side} {qty} {symbol} ({order_type}) - REJECTED"
            else:
                msg_type = 'info'
                msg = f"{side} {qty} {symbol} ({order_type}) - {status}"

            self.dashboard.log_activity(msg, msg_type)

        # Send alert if alert manager is configured
        if self.alert_manager and order_data.get('status') == 'COMPLETE':
            self.alert_manager.trade_alert(
                order_data.get('symbol', 'Unknown'),
                order_data.get('side', 'BUY'),
                str(order_data.get('quantity', 0))
            )

        logger.info(f"Order placed: {order_data}")

    def _create_scanner_view(self, parent):
        """Create market scanner view"""
        # Controls
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        tk.Label(controls, text="Scan Type:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.scan_type = ttk.Combobox(
            controls,
            values=['Momentum', 'Breakout', 'Oversold', 'Volume Spike', 'All'],
            state='readonly', width=15
        )
        self.scan_type.set('Momentum')
        self.scan_type.pack(side=tk.LEFT, padx=(5, 20))

        tk.Label(controls, text="Min Score:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.min_score = ttk.Combobox(
            controls, values=['40', '50', '60', '70', '80'],
            state='readonly', width=8
        )
        self.min_score.set('50')
        self.min_score.pack(side=tk.LEFT, padx=(5, 20))

        self.run_scan_btn = tk.Button(
            controls, text="🔍 Run Scan",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', command=self._run_scan
        )
        self.run_scan_btn.pack(side=tk.LEFT, ipadx=15, ipady=5)

        # Results container
        results_frame = tk.Frame(parent, bg=self.theme['bg_card'])
        results_frame.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        header = tk.Frame(results_frame, bg=self.theme['bg_card'])
        header.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(header, text="🔍 Scan Results", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT)

        self.scan_count_label = tk.Label(header, text="0 stocks found",
                                        bg=self.theme['bg_card'], fg=self.theme['text_dim'],
                                        font=('Segoe UI', 11))
        self.scan_count_label.pack(side=tk.RIGHT)

        # Results list
        columns = ('symbol', 'type', 'signal', 'score', 'price', 'change', 'reason')

        style = ttk.Style()

        # Use clam theme for better customization support
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure("Scanner.Treeview",
                       background=self.theme['bg_secondary'],
                       foreground=self.theme['text_primary'],
                       fieldbackground=self.theme['bg_secondary'],
                       rowheight=35,
                       font=('Segoe UI', 10))
        style.configure("Scanner.Treeview.Heading",
                       background=self.theme['bg_card'],
                       foreground=self.theme['text_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat')

        # Map colors for selection states (important for dark theme visibility)
        style.map("Scanner.Treeview",
                 background=[('selected', self.theme['accent']), ('!selected', self.theme['bg_secondary'])],
                 foreground=[('selected', 'white'), ('!selected', self.theme['text_primary'])])

        self.scanner_tree = ttk.Treeview(results_frame, columns=columns, show='headings',
                                        height=12, style="Scanner.Treeview")

        self.scanner_tree.heading('symbol', text='Symbol')
        self.scanner_tree.heading('type', text='Type')
        self.scanner_tree.heading('signal', text='Signal')
        self.scanner_tree.heading('score', text='Score')
        self.scanner_tree.heading('price', text='Price')
        self.scanner_tree.heading('change', text='Change')
        self.scanner_tree.heading('reason', text='Reason')

        self.scanner_tree.column('symbol', width=100)
        self.scanner_tree.column('type', width=100)
        self.scanner_tree.column('signal', width=80)
        self.scanner_tree.column('score', width=80)
        self.scanner_tree.column('price', width=100)
        self.scanner_tree.column('change', width=80)
        self.scanner_tree.column('reason', width=300)

        self.scanner_tree.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

    def _create_predictions_view(self, parent):
        """Create AI predictions view with searchable stock selector"""
        # Controls
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        # Stock search widget
        search_frame = tk.Frame(controls, bg=self.theme['bg_primary'])
        search_frame.pack(side=tk.LEFT)

        tk.Label(
            search_frame,
            text="🔍 Search Stock:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.predict_stock_search = StockSearchWidget(
            search_frame,
            self.theme,
            on_select=self._on_predict_symbol_selected,
            placeholder="Type symbol...",
            show_exchange=False,
            width=18
        )
        self.predict_stock_search.pack(side=tk.LEFT, padx=(0, 20))

        # Set default selection (don't trigger callback during init)
        if self.sample_data:
            first_symbol = list(self.sample_data.keys())[0]
            self.predict_stock_search.set_selected(f"NSE:{first_symbol}", trigger_callback=False)
            self._selected_predict_symbol = first_symbol
        else:
            self._selected_predict_symbol = 'RELIANCE'

        tk.Button(
            controls,
            text="🤖 Get AI Prediction",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._get_prediction
        ).pack(side=tk.LEFT, ipadx=15, ipady=5)

        tk.Button(
            controls,
            text="📊 Full Analysis",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._get_full_analysis
        ).pack(side=tk.LEFT, padx=(10, 0), ipadx=15, ipady=5)

        # Create the main content area
        self._create_predictions_content(parent)

    def _on_predict_symbol_selected(self, symbol: str):
        """Handle stock selection for AI predictions"""
        if ':' in symbol:
            _, name = symbol.split(':', 1)
        else:
            name = symbol
        self._selected_predict_symbol = name
        logger.debug(f"Prediction symbol selected: {name}")

    def _create_predictions_content(self, parent):
        """Create the main content area for predictions view"""
        # Main content - two columns
        content = tk.Frame(parent, bg=self.theme['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left - Prediction card
        left = tk.Frame(content, bg=self.theme['bg_card'])
        left.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        left_inner = tk.Frame(left, bg=self.theme['bg_card'])
        left_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(left_inner, text="🤖 AI Prediction", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)

        self.prediction_symbol_label = tk.Label(left_inner, text="Search for a stock above",
                                               bg=self.theme['bg_card'], fg=self.theme['text_dim'],
                                               font=('Segoe UI', 12))
        self.prediction_symbol_label.pack(anchor=tk.W, pady=(10, 0))

        # Direction indicator
        self.direction_frame = tk.Frame(left_inner, bg=self.theme['bg_card'])
        self.direction_frame.pack(fill=tk.X, pady=20)

        self.direction_emoji = tk.Label(self.direction_frame, text="➡️",
                                       bg=self.theme['bg_card'], font=('Segoe UI', 48))
        self.direction_emoji.pack(side=tk.LEFT)

        direction_text = tk.Frame(self.direction_frame, bg=self.theme['bg_card'])
        direction_text.pack(side=tk.LEFT, padx=20)

        self.direction_label = tk.Label(direction_text, text="NEUTRAL",
                                       bg=self.theme['bg_card'], fg=self.theme['text_primary'],
                                       font=('Segoe UI', 24, 'bold'))
        self.direction_label.pack(anchor=tk.W)

        self.confidence_label = tk.Label(direction_text, text="Confidence: --",
                                        bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
                                        font=('Segoe UI', 14))
        self.confidence_label.pack(anchor=tk.W)

        # Details
        details = tk.Frame(left_inner, bg=self.theme['bg_card'])
        details.pack(fill=tk.X, pady=10)

        self.target_label = tk.Label(details, text="🎯 Target: --",
                                    bg=self.theme['bg_card'], fg=self.theme['success'],
                                    font=('Segoe UI', 12))
        self.target_label.pack(anchor=tk.W, pady=2)

        self.stoploss_label = tk.Label(details, text="🛑 Stop Loss: --",
                                      bg=self.theme['bg_card'], fg=self.theme['danger'],
                                      font=('Segoe UI', 12))
        self.stoploss_label.pack(anchor=tk.W, pady=2)

        self.strength_label = tk.Label(details, text="💪 Strength: --",
                                      bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
                                      font=('Segoe UI', 12))
        self.strength_label.pack(anchor=tk.W, pady=2)

        # Right - Trend & S/R
        right = tk.Frame(content, bg=self.theme['bg_card'])
        right.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        right_inner = tk.Frame(right, bg=self.theme['bg_card'])
        right_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(right_inner, text="📊 Technical Analysis", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)

        # Trend
        trend_frame = tk.Frame(right_inner, bg=self.theme['bg_card'])
        trend_frame.pack(fill=tk.X, pady=15)

        tk.Label(trend_frame, text="Trend:", bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.trend_label = tk.Label(trend_frame, text="--",
                                   bg=self.theme['bg_card'], fg=self.theme['text_primary'],
                                   font=('Segoe UI', 14, 'bold'))
        self.trend_label.pack(side=tk.LEFT, padx=(10, 0))

        # Support/Resistance
        tk.Label(right_inner, text="Support & Resistance", bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(anchor=tk.W, pady=(10, 5))

        self.sr_frame = tk.Frame(right_inner, bg=self.theme['bg_card'])
        self.sr_frame.pack(fill=tk.X)

        self.resistance_labels = []
        self.support_labels = []

        for i in range(3):
            r_label = tk.Label(self.sr_frame, text="R: --", bg=self.theme['bg_card'],
                              fg=self.theme['danger'], font=('Segoe UI', 11))
            r_label.pack(anchor=tk.W, pady=2)
            self.resistance_labels.append(r_label)

        tk.Label(self.sr_frame, text="─" * 30, bg=self.theme['bg_card'],
                fg=self.theme['text_dim'], font=('Segoe UI', 8)).pack(anchor=tk.W, pady=5)

        for i in range(3):
            s_label = tk.Label(self.sr_frame, text="S: --", bg=self.theme['bg_card'],
                              fg=self.theme['success'], font=('Segoe UI', 11))
            s_label.pack(anchor=tk.W, pady=2)
            self.support_labels.append(s_label)

    def _create_portfolio_view(self, parent):
        """Create portfolio optimization view"""
        # Controls
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        tk.Label(controls, text="Optimization Goal:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.opt_goal = ttk.Combobox(
            controls,
            values=['Max Sharpe Ratio', 'Min Volatility', 'Risk Parity', 'Equal Weight'],
            state='readonly', width=18
        )
        self.opt_goal.set('Max Sharpe Ratio')
        self.opt_goal.pack(side=tk.LEFT, padx=(5, 20))

        tk.Label(controls, text="Capital:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.opt_capital = tk.Entry(controls, width=12, font=('Segoe UI', 11))
        self.opt_capital.insert(0, "100000")
        self.opt_capital.pack(side=tk.LEFT, padx=(5, 20))

        self.optimize_btn = tk.Button(
            controls, text="💼 Optimize Portfolio",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', command=self._optimize_portfolio
        )
        self.optimize_btn.pack(side=tk.LEFT, ipadx=15, ipady=5)

        # Main content - two columns
        content = tk.Frame(parent, bg=self.theme['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left - Allocation
        left = tk.Frame(content, bg=self.theme['bg_card'])
        left.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        left_inner = tk.Frame(left, bg=self.theme['bg_card'])
        left_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(left_inner, text="💼 Optimal Allocation", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)

        self.allocation_frame = tk.Frame(left_inner, bg=self.theme['bg_card'])
        self.allocation_frame.pack(fill=tk.BOTH, expand=True, pady=15)

        self.allocation_labels: List[tk.Label] = []

        # Right - Metrics
        right = tk.Frame(content, bg=self.theme['bg_card'])
        right.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        right_inner = tk.Frame(right, bg=self.theme['bg_card'])
        right_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(right_inner, text="📈 Expected Metrics", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)

        metrics_frame = tk.Frame(right_inner, bg=self.theme['bg_card'])
        metrics_frame.pack(fill=tk.X, pady=15)

        self.metric_labels = {}
        metrics = [
            ('return', 'Annual Return', '--'),
            ('volatility', 'Volatility', '--'),
            ('sharpe', 'Sharpe Ratio', '--'),
            ('max_dd', 'Max Drawdown', '--'),
            ('var', 'VaR (95%)', '--'),
        ]

        for key, name, default in metrics:
            row = tk.Frame(metrics_frame, bg=self.theme['bg_card'])
            row.pack(fill=tk.X, pady=5)

            tk.Label(row, text=name, bg=self.theme['bg_card'],
                    fg=self.theme['text_secondary'], font=('Segoe UI', 12)).pack(side=tk.LEFT)

            label = tk.Label(row, text=default, bg=self.theme['bg_card'],
                           fg=self.theme['text_primary'], font=('Segoe UI', 14, 'bold'))
            label.pack(side=tk.RIGHT)
            self.metric_labels[key] = label

    def _create_infrastructure_view(self, parent):
        """Create infrastructure monitoring view"""
        # Title and description
        header = tk.Frame(parent, bg=self.theme['bg_primary'])
        header.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            header,
            text="System Infrastructure & Monitoring",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 12)
        ).pack(anchor=tk.W)

        # Main content area with two columns
        content = tk.Frame(parent, bg=self.theme['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left column - Infrastructure Manager Widget
        left_col = tk.Frame(content, bg=self.theme['bg_primary'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Infrastructure Manager Widget
        self.infra_widget = InfrastructureManagerWidget(left_col, self.theme)
        self.infra_widget.pack(fill=tk.BOTH, expand=True)

        # Right column - Quick actions and info
        right_col = tk.Frame(content, bg=self.theme['bg_primary'], width=300)
        right_col.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        right_col.pack_propagate(False)

        # Quick Actions Card
        actions_card = tk.Frame(right_col, bg=self.theme['bg_card'])
        actions_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        actions_card.pack(fill=tk.X, pady=(0, 10))

        actions_inner = tk.Frame(actions_card, bg=self.theme['bg_card'])
        actions_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            actions_inner,
            text="Quick Actions",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Action buttons
        actions = [
            ("Start Recording", self._start_recording, self.theme['success']),
            ("Stop Recording", self._stop_recording, self.theme['danger']),
            ("Export Audit Log", self._export_audit, self.theme['info']),
            ("Run Compliance Check", self._run_compliance_check, self.theme['warning']),
        ]

        for text, cmd, color in actions:
            btn = tk.Button(
                actions_inner,
                text=text,
                bg=color,
                fg='white',
                font=('Segoe UI', 10),
                relief=tk.FLAT,
                cursor='hand2',
                command=cmd
            )
            btn.pack(fill=tk.X, pady=3, ipady=5)

        # Info Card
        info_card = tk.Frame(right_col, bg=self.theme['bg_card'])
        info_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        info_card.pack(fill=tk.X, pady=(0, 10))

        info_inner = tk.Frame(info_card, bg=self.theme['bg_card'])
        info_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            info_inner,
            text="Infrastructure Components",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        components = [
            ("Flight Recorder", "Market replay & debugging"),
            ("Shadow Engine", "Paper trading validation"),
            ("A/B Testing", "Strategy comparison"),
            ("Audit Trail", "Compliance logging"),
            ("Risk Compliance", "SEBI regulations"),
            ("Kill Switch", "Emergency stop"),
        ]

        for name, desc in components:
            row = tk.Frame(info_inner, bg=self.theme['bg_card'])
            row.pack(fill=tk.X, pady=3)

            tk.Label(
                row, text=f"• {name}",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 10, 'bold')
            ).pack(anchor=tk.W)

            tk.Label(
                row, text=f"  {desc}",
                bg=self.theme['bg_card'],
                fg=self.theme['text_dim'],
                font=('Segoe UI', 9)
            ).pack(anchor=tk.W)

    def _start_recording(self):
        """Start flight recorder"""
        try:
            from core.infrastructure import get_infrastructure_manager
            manager = get_infrastructure_manager()
            if manager and manager.flight_recorder:
                manager.flight_recorder.start_recording()
                self.dashboard.log_activity("Flight recorder started", 'info')
            else:
                self.dashboard.log_activity("Infrastructure not initialized", 'warning')
        except Exception as e:
            self.dashboard.log_activity(f"Error: {e}", 'error')

    def _stop_recording(self):
        """Stop flight recorder"""
        try:
            from core.infrastructure import get_infrastructure_manager
            manager = get_infrastructure_manager()
            if manager and manager.flight_recorder:
                manager.flight_recorder.stop_recording()
                self.dashboard.log_activity("Flight recorder stopped", 'info')
            else:
                self.dashboard.log_activity("Infrastructure not initialized", 'warning')
        except Exception as e:
            self.dashboard.log_activity(f"Error: {e}", 'error')

    def _export_audit(self):
        """Export audit log"""
        try:
            from core.infrastructure import get_infrastructure_manager
            from tkinter import filedialog
            manager = get_infrastructure_manager()
            if manager and manager.audit_trail:
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                )
                if filepath:
                    report = manager.get_audit_report()
                    import json
                    with open(filepath, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                    self.dashboard.log_activity(f"Audit exported to {filepath}", 'info')
            else:
                self.dashboard.log_activity("Audit trail not initialized", 'warning')
        except Exception as e:
            self.dashboard.log_activity(f"Error: {e}", 'error')

    def _run_compliance_check(self):
        """Run compliance check"""
        try:
            from core.infrastructure import get_infrastructure_manager
            manager = get_infrastructure_manager()
            if manager and manager.compliance_engine:
                report = manager.get_compliance_report()
                self.dashboard.log_activity("Compliance check complete", 'info')
            else:
                self.dashboard.log_activity("Compliance engine not initialized", 'warning')
        except Exception as e:
            self.dashboard.log_activity(f"Error: {e}", 'error')

    def _show_view(self, view_name: str):
        """Switch to a different view"""
        for view in self.views.values():
            view.pack_forget()

        if view_name in self.views:
            self.views[view_name].pack(fill=tk.BOTH, expand=True)

        for key, btn in self.nav_buttons.items():
            btn.set_selected(key == view_name)

        titles = {
            'dashboard': '📊 Dashboard',
            'charts': '📈 Charts',
            'scanner': '🔍 Market Scanner',
            'predictions': '🤖 AI Predictions',
            'portfolio': '💼 Portfolio',
            'strategies': '🎯 Strategies',
            'infrastructure': '🏗️ Infrastructure',
            'settings': '⚙️ Settings'
        }
        self.view_title.config(text=titles.get(view_name, view_name.title()))

        self.current_view = view_name

    def _show_login(self):
        """Show login dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Login to Zerodha")
        dialog.geometry("450x350")
        dialog.configure(bg=self.theme['bg_card'])
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (self.root.winfo_width() - 450) // 2 + self.root.winfo_x()
        y = (self.root.winfo_height() - 350) // 2 + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")

        inner = tk.Frame(dialog, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        tk.Label(inner, text="🔐 Login to Zerodha", bg=self.theme['bg_card'],
                fg=self.theme['text_primary'], font=('Segoe UI', 18, 'bold')).pack(pady=(0, 20))

        tk.Label(inner, text="API Key", bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(anchor=tk.W)

        api_key_entry = tk.Entry(inner, width=45, bg=self.theme['bg_secondary'],
                                fg=self.theme['text_primary'], font=('Segoe UI', 11), relief=tk.FLAT)
        api_key_entry.pack(fill=tk.X, pady=(5, 15), ipady=5)

        tk.Label(inner, text="API Secret", bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(anchor=tk.W)

        api_secret_entry = tk.Entry(inner, width=45, show="*", bg=self.theme['bg_secondary'],
                                   fg=self.theme['text_primary'], font=('Segoe UI', 11), relief=tk.FLAT)
        api_secret_entry.pack(fill=tk.X, pady=(5, 20), ipady=5)

        paper_var = tk.BooleanVar(value=True)
        tk.Checkbutton(inner, text="Start in Paper Trading mode (recommended)",
                      variable=paper_var, bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
                      selectcolor=self.theme['bg_secondary'], font=('Segoe UI', 10)).pack(anchor=tk.W, pady=(0, 20))

        def do_login():
            api_key = api_key_entry.get().strip()
            api_secret = api_secret_entry.get().strip()

            if not api_key or not api_secret:
                messagebox.showerror("Error", "Please enter both API Key and Secret!")
                return

            self.connected = True
            self.paper_trading = paper_var.get()

            # Initialize OrderManager for manual order placement
            try:
                from core.order_manager import OrderManager
                self._order_manager = OrderManager(
                    broker=self.broker,
                    paper_trading=self.paper_trading,
                    auto_server_sl=True
                )
                # Update order panel with order manager
                if hasattr(self, 'order_panel') and self.order_panel:
                    self.order_panel.set_order_manager(self._order_manager)
                    self.order_panel.set_broker(self.broker)
                logger.info("OrderManager initialized for order panel")
            except Exception as e:
                logger.warning(f"Could not initialize OrderManager: {e}")

            self.conn_indicator.config(text="● Connected", fg=self.theme['success'])
            self.mode_indicator.config(
                text="Paper Trading" if self.paper_trading else "LIVE Trading",
                fg=self.theme['info'] if self.paper_trading else self.theme['warning']
            )
            self.login_btn.config(text="✓ Connected", state=tk.DISABLED, bg=self.theme['success'])

            self.dashboard.update_connection(True)
            self.dashboard.log_activity("Connected to Zerodha!", 'success')

            if self.alert_manager:
                self.alert_manager.bot_status("Connected", "Ready to trade")

            dialog.destroy()

        tk.Button(inner, text="🚀 Connect", bg=self.theme['btn_primary'], fg='white',
                 font=('Segoe UI', 12, 'bold'), relief=tk.FLAT, cursor='hand2',
                 command=do_login).pack(fill=tk.X, ipady=10)

    def _toggle_bot(self):
        """
        Toggle bot on/off - NOW CONNECTED TO EVENT-DRIVEN ARCHITECTURE!

        This method properly integrates with the EventDrivenLiveEngine:
        1. Creates the selected strategy instance
        2. Adds it to the trading engine with event bus
        3. Sets up a data source (simulated for paper trading)
        4. Starts/stops the engine
        """
        if not self.connected and not self.paper_trading:
            messagebox.showwarning("Not Connected", "Please login to Zerodha first!")
            return

        if not self.bot_running:
            # ==================== START BOT ====================
            try:
                self._start_trading_engine()
                self.bot_running = True
                self.bot_btn.config(text="⏹ STOP BOT", bg=self.theme['btn_danger'])
                self.dashboard.update_bot_status(True)
                self.dashboard.log_activity(
                    f"Bot started with {self.selected_strategy.upper()} strategy "
                    f"(Event-Driven Mode)", 'success'
                )

                if self.alert_manager:
                    self.alert_manager.bot_status("Started", f"Using {self.selected_strategy} strategy")

            except Exception as e:
                logger.error(f"Failed to start bot: {e}")
                self.dashboard.log_activity(f"Failed to start bot: {e}", 'danger')
                messagebox.showerror("Start Failed", f"Could not start trading engine:\n{e}")
        else:
            # ==================== STOP BOT ====================
            try:
                self._stop_trading_engine()
                self.bot_running = False
                self.bot_btn.config(text="▶ START BOT", bg=self.theme['btn_success'])
                self.dashboard.update_bot_status(False)
                self.dashboard.log_activity("Bot stopped", 'warning')

                if self.alert_manager:
                    self.alert_manager.bot_status("Stopped")

            except Exception as e:
                logger.error(f"Failed to stop bot: {e}")
                self.dashboard.log_activity(f"Error stopping bot: {e}", 'warning')

    def _start_trading_engine(self):
        """
        Start the Event-Driven Trading Engine.

        This is the UNIFIED architecture - same code path for live and backtest:
        DataSource -> EventBus -> Strategy -> Signal -> Execution
        """
        engine = self.trading_engine
        if engine is None:
            raise RuntimeError("Trading engine not available")

        # Get the strategy using the strategies module
        from strategies import get_strategy, ALL_STRATEGIES

        if self.selected_strategy not in ALL_STRATEGIES:
            raise RuntimeError(
                f"Unknown strategy: {self.selected_strategy}. "
                f"Available: {list(ALL_STRATEGIES.keys())}"
            )

        # Create strategy instance
        strategy = get_strategy(self.selected_strategy)

        # Determine symbols to trade
        symbols = self.settings.get('trading', {}).get('symbols', DEFAULT_TRADING_SYMBOLS)
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]

        # Add strategy for each symbol
        for symbol in symbols[:5]:  # Limit to 5 symbols
            engine.add_strategy(strategy, symbol)
            logger.info(f"Added strategy {strategy.name} for {symbol}")

        # Create data source (simulated for paper trading)
        if self.paper_trading:
            # Use simulated data source for paper trading
            base_prices = {s: self.sample_data.get(s, {}).get('close', pd.Series([1000])).iloc[-1]
                          for s in symbols if s in self.sample_data}
            engine.create_simulated_source(
                symbols=symbols,
                base_prices=base_prices,
                tick_interval=1.0
            )
            logger.info("Created simulated data source for paper trading")
        else:
            # Use live data source with broker
            try:
                from core.data import LiveDataSource, DataSourceConfig
                config = DataSourceConfig(
                    symbols=symbols,
                    timeframe='1m',
                    emit_bars=True,
                    emit_ticks=True
                )
                data_source = LiveDataSource(
                    event_bus=self.event_bus,
                    config=config,
                    broker=self.broker
                )
                engine.set_data_source(data_source)
                logger.info("Created live data source")
            except Exception as e:
                logger.warning(f"Could not create live data source: {e}, falling back to simulated")
                engine.create_simulated_source(symbols=symbols)

        # Start the engine!
        engine.start()
        logger.info("Event-Driven Trading Engine STARTED")

        # Log to dashboard
        self.dashboard.log_activity(
            f"Engine started: {len(symbols)} symbols, "
            f"{'Paper' if self.paper_trading else 'Live'} mode",
            'info'
        )

    def _stop_trading_engine(self):
        """Stop the Event-Driven Trading Engine."""
        engine = self.trading_engine
        if engine is None:
            return

        engine.stop()
        logger.info("Event-Driven Trading Engine STOPPED")

    def _on_kill_switch(self, triggered: bool, reason: str):
        """
        Handle kill switch activation/deactivation.

        NOW PROPERLY INTEGRATED: Stops the EventDrivenLiveEngine and
        triggers infrastructure-level emergency stop.
        """
        if triggered:
            # ==================== EMERGENCY STOP ====================
            logger.warning(f"KILL SWITCH TRIGGERED: {reason}")

            # 1. Stop the trading engine FIRST
            if self.bot_running:
                try:
                    self._stop_trading_engine()
                except Exception as e:
                    logger.error(f"Error stopping engine during kill switch: {e}")

                self.bot_running = False
                self.bot_btn.config(text="▶ START BOT", bg=self.theme['btn_success'])
                self.dashboard.update_bot_status(False)

            # 2. Trigger infrastructure-level kill switch (cancels orders, etc.)
            try:
                from core.infrastructure import trigger_emergency_stop
                trigger_emergency_stop(reason)
            except Exception as e:
                logger.error(f"Could not trigger infrastructure kill switch: {e}")

            # 3. Log and alert
            self.dashboard.log_activity(f"KILL SWITCH TRIGGERED: {reason}", 'danger')

            if self.alert_manager:
                self.alert_manager.bot_status("EMERGENCY STOP", reason)

            # 4. Show warning to user
            messagebox.showwarning(
                "Kill Switch Activated",
                f"Emergency stop triggered!\n\nReason: {reason}\n\n"
                "All trading has been stopped. Check your positions."
            )
        else:
            self.dashboard.log_activity("Kill switch reset - system normal", 'success')

    def _on_strategy_selected(self, strategy_key: str):
        """Handle strategy selection"""
        self.selected_strategy = strategy_key
        info = STRATEGY_INFO.get(strategy_key, {})
        self.dashboard.log_activity(
            f"Strategy changed to {info.get('emoji', '')} {info.get('name', strategy_key)}",
            'info'
        )

    def _load_chart(self):
        """Load chart for selected symbol and timeframe"""
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        # Get symbol from search widget
        symbol = getattr(self, '_selected_chart_symbol', None)
        if not symbol:
            # Try to get from search widget
            selected = self.chart_stock_search.get_selected()
            if selected:
                if ':' in selected:
                    _, symbol = selected.split(':', 1)
                else:
                    symbol = selected
            else:
                symbol = 'RELIANCE'

        # Get selected timeframe
        timeframe = self.chart_timeframe.get()

        # Cache key includes both symbol and timeframe
        cache_key = f"{symbol}_{timeframe}"
        data = self.sample_data.get(cache_key)

        # If no data for this symbol+timeframe combo, generate it
        if data is None:
            data = self._generate_symbol_data(symbol, timeframe)
            if data is not None:
                self.sample_data[cache_key] = data

        if data is None:
            # Show error message in chart container
            tk.Label(
                self.chart_container,
                text=f"Could not load data for {symbol}\n\nTry a different symbol from your watchlist",
                bg=self.theme['bg_card'],
                fg=self.theme['text_dim'],
                font=('Segoe UI', 12),
                justify=tk.CENTER
            ).pack(expand=True)
            return

        # Determine chart height based on selected indicators
        show_rsi = self.show_rsi.get()
        show_macd = self.show_macd.get()

        # Adjust height for additional panels
        chart_height = 500
        if show_rsi:
            chart_height += 80
        if show_macd:
            chart_height += 80

        chart = CandlestickChart(self.chart_container, 800, chart_height, show_toolbar=True)
        chart.plot(data, f"{symbol} - {timeframe}")

        # Price overlay indicators
        if self.show_ma.get():
            add_moving_average(chart.ax_price, data, 20)

        if self.show_ema.get():
            add_ema(chart.ax_price, data, 20)

        if self.show_bb.get():
            add_bollinger_bands(chart.ax_price, data)

        if self.show_vwap.get():
            add_vwap(chart.ax_price, data)

        if self.show_supertrend.get():
            add_supertrend(chart.ax_price, data)

        # Add legend if any price indicators selected
        if self.show_ma.get() or self.show_ema.get() or self.show_bb.get() or self.show_vwap.get():
            chart.ax_price.legend(loc='upper left', fontsize=8, framealpha=0.7,
                                  facecolor='#1e1e1e', edgecolor='#333333', labelcolor='white')

        # RSI panel (separate subplot)
        if show_rsi:
            # Create new axis for RSI
            ax_rsi = chart.fig.add_axes([0.1, 0.05, 0.85, 0.12])
            add_rsi(ax_rsi, data)
            # Adjust volume axis position
            chart.ax_volume.set_position([0.1, 0.20, 0.85, 0.12])

        # MACD panel (separate subplot)
        if show_macd:
            # Create new axis for MACD
            y_pos = 0.05 if not show_rsi else 0.05
            if show_rsi:
                # Move RSI up to make room
                ax_rsi.set_position([0.1, 0.18, 0.85, 0.10])
                chart.ax_volume.set_position([0.1, 0.30, 0.85, 0.10])
            ax_macd = chart.fig.add_axes([0.1, 0.05, 0.85, 0.10])
            add_macd(ax_macd, data)

        chart.canvas.draw()
        self.dashboard.log_activity(f"Loaded chart for {symbol}", 'info')

    def _generate_symbol_data(self, symbol: str, timeframe: str = '1 Day'):
        """Generate sample data for a symbol based on timeframe"""
        try:
            # Configure based on timeframe
            timeframe_config = {
                '1 Min': {'periods': 390, 'freq': 'min', 'volatility': 0.001, 'trend': 0.00001},      # ~1 trading day
                '5 Min': {'periods': 390, 'freq': '5min', 'volatility': 0.002, 'trend': 0.00005},    # ~5 trading days
                '15 Min': {'periods': 260, 'freq': '15min', 'volatility': 0.004, 'trend': 0.0001},   # ~10 trading days
                '1 Hour': {'periods': 195, 'freq': 'h', 'volatility': 0.008, 'trend': 0.0002},       # ~30 trading days
                '1 Day': {'periods': 100, 'freq': 'D', 'volatility': 0.02, 'trend': 0.0003},         # ~100 trading days
            }

            config = timeframe_config.get(timeframe, timeframe_config['1 Day'])
            periods = config['periods']
            freq = config['freq']
            volatility = config['volatility']
            trend = config['trend']

            # Generate dates based on frequency
            dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)

            # Use symbol hash for consistent base price (same symbol = same price range)
            np.random.seed(hash(symbol) % (2**32))
            base_price = np.random.randint(500, 3000)

            # Reset seed with symbol+timeframe for unique but reproducible patterns
            np.random.seed(hash(f"{symbol}_{timeframe}") % (2**32))

            returns = np.random.randn(periods) * volatility + trend
            prices = base_price * np.exp(np.cumsum(returns))

            # Intraday timeframes have higher price variations within bars
            high_mult = 0.003 if timeframe in ['1 Min', '5 Min'] else 0.01
            low_mult = 0.003 if timeframe in ['1 Min', '5 Min'] else 0.01

            data = pd.DataFrame({
                'open': prices * (1 + np.random.randn(periods) * 0.005),
                'high': prices * (1 + np.abs(np.random.randn(periods) * high_mult)),
                'low': prices * (1 - np.abs(np.random.randn(periods) * low_mult)),
                'close': prices,
                'volume': np.random.randint(10000, 500000, periods)
            }, index=dates)

            # Reset random seed to default state
            np.random.seed(None)

            return data
        except Exception as e:
            logger.error(f"Failed to generate data for {symbol} ({timeframe}): {e}")
            return None

    def _run_scan(self):
        """Run market scanner (offloaded to background thread)"""
        self.run_scan_btn.config(state='disabled')
        self.scan_count_label.config(text="Scanning... Please wait.")

        # Read UI inputs on main thread before spawning worker
        scan_type_val = self.scan_type.get()
        min_score_val = self.min_score.get()

        def task():
            try:
                from advanced.market_scanner import MarketScanner, ScanType, ScanFilter

                scan_map = {
                    'Momentum': [ScanType.MOMENTUM],
                    'Breakout': [ScanType.BREAKOUT],
                    'Oversold': [ScanType.OVERSOLD],
                    'Volume Spike': [ScanType.VOLUME_SPIKE],
                    'All': None
                }
                scan_types = scan_map.get(scan_type_val)

                scanner = MarketScanner()
                min_score = int(min_score_val)
                filter_criteria = ScanFilter(min_score=min_score)

                results = scanner.scan_watchlist(self.sample_data, scan_types, filter_criteria)

                # Update UI on the main thread
                self.master.after(0, lambda: self._display_scan_results(results))

            except Exception as e:
                logger.error(f"Scan error: {e}")
                self.master.after(0, lambda: messagebox.showerror("Scan Error", f"Could not run scan: {e}"))
            finally:
                self.master.after(0, lambda: self.run_scan_btn.config(state='normal'))

        threading.Thread(target=task, daemon=True).start()

    def _display_scan_results(self, results):
        """Display scan results on the main thread"""
        # Clear existing results
        for item in self.scanner_tree.get_children():
            self.scanner_tree.delete(item)

        for result in results[:20]:
            signal_emoji = "🟢" if result.signal == "BUY" else "🔴" if result.signal == "SELL" else "🟡"
            change_str = f"{result.change_percent:+.1f}%"

            self.scanner_tree.insert('', tk.END, values=(
                result.symbol,
                result.scan_type.value.title(),
                f"{signal_emoji} {result.signal}",
                f"{result.score:.0f}",
                f"₹{result.current_price:,.2f}",
                change_str,
                result.reason[:40] + "..." if len(result.reason) > 40 else result.reason
            ))

        self.scan_count_label.config(text=f"{len(results)} stocks found")
        self.dashboard.log_activity(f"Scanner found {len(results)} opportunities", 'info')

    def _get_prediction(self):
        """Get AI prediction for selected symbol"""
        try:
            from advanced.ml_predictor import MLPredictor

            # Get symbol from search widget
            symbol = getattr(self, '_selected_predict_symbol', None)
            if not symbol:
                selected = self.predict_stock_search.get_selected()
                if selected:
                    if ':' in selected:
                        _, symbol = selected.split(':', 1)
                    else:
                        symbol = selected
                else:
                    symbol = 'RELIANCE'

            data = self.sample_data.get(symbol)

            if data is None:
                return

            predictor = MLPredictor()
            prediction = predictor.predict(data, symbol)

            # Update UI
            self.prediction_symbol_label.config(text=f"Prediction for {symbol}")

            # Direction
            direction_emojis = {"UP": "📈", "DOWN": "📉", "NEUTRAL": "➡️"}
            direction_colors = {"UP": self.theme['success'], "DOWN": self.theme['danger'],
                              "NEUTRAL": self.theme['text_secondary']}

            self.direction_emoji.config(text=direction_emojis.get(prediction.direction, "➡️"))
            self.direction_label.config(text=prediction.direction,
                                       fg=direction_colors.get(prediction.direction))
            self.confidence_label.config(text=f"Confidence: {prediction.confidence_pct}")

            # Details
            if prediction.target_price:
                self.target_label.config(text=f"🎯 Target: ₹{prediction.target_price:,.2f}")
            if prediction.stop_loss:
                self.stoploss_label.config(text=f"🛑 Stop Loss: ₹{prediction.stop_loss:,.2f}")
            self.strength_label.config(text=f"💪 Strength: {prediction.strength.name}")

            self.dashboard.log_activity(
                f"AI Prediction: {symbol} → {prediction.direction} ({prediction.confidence_pct})",
                'success' if prediction.direction == "UP" else 'danger' if prediction.direction == "DOWN" else 'info'
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            messagebox.showerror("Prediction Error", f"Could not get prediction: {e}")

    def _get_full_analysis(self):
        """Get full technical analysis"""
        try:
            from advanced.ml_predictor import MLPredictor

            # Get symbol from search widget
            symbol = getattr(self, '_selected_predict_symbol', None)
            if not symbol:
                selected = self.predict_stock_search.get_selected()
                if selected:
                    if ':' in selected:
                        _, symbol = selected.split(':', 1)
                    else:
                        symbol = selected
                else:
                    symbol = 'RELIANCE'

            data = self.sample_data.get(symbol)

            if data is None:
                return

            predictor = MLPredictor()

            # Trend
            trend = predictor.get_trend(data)
            self.trend_label.config(text=f"{trend['emoji']} {trend['trend']}")

            # Support/Resistance
            sr = predictor.get_support_resistance(data)

            # Update resistance labels
            for i, label in enumerate(self.resistance_labels):
                if i < len(sr.get('resistance', [])):
                    label.config(text=f"R{i+1}: ₹{sr['resistance'][i]:,.2f}")
                else:
                    label.config(text=f"R{i+1}: --")

            # Update support labels
            for i, label in enumerate(self.support_labels):
                if i < len(sr.get('support', [])):
                    label.config(text=f"S{i+1}: ₹{sr['support'][-(i+1)]:,.2f}")
                else:
                    label.config(text=f"S{i+1}: --")

            self.dashboard.log_activity(f"Full analysis for {symbol}: {trend['trend']}", 'info')

        except Exception as e:
            logger.error(f"Analysis error: {e}")

    def _optimize_portfolio(self):
        """Optimize portfolio allocation (offloaded to background thread)"""
        self.optimize_btn.config(state='disabled')

        # Read UI inputs on main thread before spawning worker
        goal_val = self.opt_goal.get()
        try:
            capital = float(self.opt_capital.get())
        except (ValueError, TypeError):
            capital = 100000

        def task():
            try:
                from advanced.portfolio_optimizer import PortfolioOptimizer, OptimizationGoal

                goal_map = {
                    'Max Sharpe Ratio': OptimizationGoal.MAX_SHARPE,
                    'Min Volatility': OptimizationGoal.MIN_VOLATILITY,
                    'Risk Parity': OptimizationGoal.RISK_PARITY,
                    'Equal Weight': OptimizationGoal.EQUAL_WEIGHT,
                }
                goal = goal_map.get(goal_val, OptimizationGoal.MAX_SHARPE)

                optimizer = PortfolioOptimizer()
                optimizer.load_data(self.sample_data)
                result = optimizer.optimize(goal)

                # Update UI on the main thread
                self.master.after(0, lambda: self._display_optimization_results(result, capital, goal))

            except Exception as e:
                logger.error(f"Optimization error: {e}")
                self.master.after(0, lambda: messagebox.showerror("Optimization Error", f"Could not optimize: {e}"))
            finally:
                self.master.after(0, lambda: self.optimize_btn.config(state='normal'))

        threading.Thread(target=task, daemon=True).start()

    def _display_optimization_results(self, result, capital, goal):
        """Display optimization results on the main thread"""
        # Clear existing allocation labels
        for label in self.allocation_labels:
            label.destroy()
        self.allocation_labels.clear()

        # Display allocation
        sorted_weights = sorted(result.weights.items(), key=lambda x: x[1], reverse=True)

        for symbol, weight in sorted_weights:
            if weight > 0.01:  # Only show > 1%
                row = tk.Frame(self.allocation_frame, bg=self.theme['bg_card'])
                row.pack(fill=tk.X, pady=3)

                tk.Label(row, text=symbol, bg=self.theme['bg_card'],
                        fg=self.theme['text_primary'], font=('Segoe UI', 11),
                        width=12, anchor=tk.W).pack(side=tk.LEFT)

                # Progress bar
                bar_width = int(weight * 200)
                bar = tk.Frame(row, bg=self.theme['accent'], width=bar_width, height=20)
                bar.pack(side=tk.LEFT, padx=10)
                bar.pack_propagate(False)

                tk.Label(row, text=f"{weight:.1%}", bg=self.theme['bg_card'],
                        fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

                amount = capital * weight
                tk.Label(row, text=f"₹{amount:,.0f}", bg=self.theme['bg_card'],
                        fg=self.theme['text_dim'], font=('Segoe UI', 10)).pack(side=tk.RIGHT)

                self.allocation_labels.append(row)

        # Update metrics
        m = result.metrics
        self.metric_labels['return'].config(
            text=f"{m.expected_return:.1%}",
            fg=self.theme['success'] if m.expected_return > 0 else self.theme['danger']
        )
        self.metric_labels['volatility'].config(text=f"{m.volatility:.1%}")
        self.metric_labels['sharpe'].config(
            text=f"{m.sharpe_ratio:.2f}",
            fg=self.theme['success'] if m.sharpe_ratio > 1 else self.theme['text_primary']
        )
        self.metric_labels['max_dd'].config(
            text=f"{m.max_drawdown:.1%}",
            fg=self.theme['danger']
        )
        self.metric_labels['var'].config(text=f"{m.var_95:.1%}")

        self.dashboard.log_activity(
            f"Portfolio optimized: {goal.value} | Sharpe: {m.sharpe_ratio:.2f}",
            'success'
        )

    def _start_updates(self):
        """Start periodic updates"""
        self._update_dashboard()
        self._process_ui_queue()  # Start thread-safe queue processor

        # Start multiprocessing event processor if in that mode
        if self._multiprocessing_mode:
            self._process_engine_events()
            self._check_engine_health()

    def _process_ui_queue(self):
        """
        Process UI updates from background threads.

        Tkinter is NOT thread-safe. This method runs on the main thread
        and processes callbacks queued by background threads, preventing
        "Tcl/Tk error" crashes.

        Called every 50ms to ensure responsive UI updates.
        """
        try:
            # Process all pending UI updates (non-blocking)
            while True:
                try:
                    callback = self.ui_queue.get_nowait()
                    if callable(callback):
                        callback()
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing UI queue: {e}")
        finally:
            # Schedule next check (50ms for responsive updates)
            self.root.after(50, self._process_ui_queue)

    def _process_engine_events(self):
        """
        Process events from multiprocessing engine (IPC queue).

        This is the GIL-bypassing alternative to EventBus subscriptions.
        Events come from a separate process via multiprocessing.Queue.

        Called every 50ms to ensure responsive UI updates.
        """
        if not self._multiprocessing_mode or not self._engine_process:
            return

        try:
            # Import IPC event types
            from core.ipc_messages import Events

            # Poll events from engine process
            events = self._engine_process.poll_events(max_events=50)

            for msg in events:
                try:
                    self._handle_ipc_event(msg)
                except Exception as e:
                    logger.debug(f"Error handling IPC event {msg.msg_type}: {e}")

        except Exception as e:
            logger.error(f"Error processing engine events: {e}")
        finally:
            # Schedule next check (50ms for responsive updates)
            self.root.after(50, self._process_engine_events)

    def _handle_ipc_event(self, msg):
        """
        Handle a single IPC event from the engine process.

        Routes events to appropriate handlers based on message type.
        """
        from core.ipc_messages import Events

        if msg.msg_type == Events.TICK_UPDATE:
            self._handle_ipc_tick(msg.payload)

        elif msg.msg_type == Events.BAR_UPDATE:
            self._handle_ipc_bar(msg.payload)

        elif msg.msg_type == Events.SIGNAL_GENERATED:
            self._handle_ipc_signal(msg.payload)

        elif msg.msg_type in (Events.ORDER_SUBMITTED, Events.ORDER_FILLED,
                              Events.ORDER_REJECTED, Events.ORDER_CANCELLED):
            self._handle_ipc_order(msg.msg_type, msg.payload)

        elif msg.msg_type in (Events.POSITION_OPENED, Events.POSITION_CLOSED,
                              Events.POSITION_UPDATED):
            self._handle_ipc_position(msg.msg_type, msg.payload)

        elif msg.msg_type == Events.ENGINE_STATUS:
            self._handle_ipc_status(msg.payload)

        elif msg.msg_type == Events.ENGINE_ERROR:
            self._handle_ipc_error(msg.payload)

        elif msg.msg_type == Events.LOG_MESSAGE:
            self._handle_ipc_log(msg.payload)

        elif msg.msg_type == Events.HEARTBEAT:
            pass  # Heartbeat handled by poll_events()

    def _handle_ipc_tick(self, payload: dict):
        """Handle tick update from engine process."""
        try:
            symbol = payload.get('symbol', 'Unknown')
            price = payload.get('price', 0)
            if hasattr(self, 'dashboard') and self.dashboard:
                self.dashboard.update_price(symbol, price)
        except Exception as e:
            logger.debug(f"IPC tick update error: {e}")

    def _handle_ipc_bar(self, payload: dict):
        """Handle bar update from engine process."""
        try:
            symbol = payload.get('symbol', 'Unknown')
            logger.debug(f"IPC bar event: {symbol}")
        except Exception as e:
            logger.debug(f"IPC bar update error: {e}")

    def _handle_ipc_signal(self, payload: dict):
        """Handle signal from engine process."""
        try:
            symbol = payload.get('symbol', 'Unknown')
            signal_type = payload.get('signal_type', 'HOLD')
            price = payload.get('price', 0)
            confidence = payload.get('confidence', 0)

            msg = f"Signal: {signal_type} on {symbol} @ Rs.{price:.2f} (conf: {confidence:.0%})"

            if hasattr(self, 'dashboard') and self.dashboard:
                level = 'success' if 'BUY' in str(signal_type) else 'warning' if 'SELL' in str(signal_type) else 'info'
                self.dashboard.log_activity(msg, level)
        except Exception as e:
            logger.debug(f"IPC signal update error: {e}")

    def _handle_ipc_order(self, event_type: str, payload: dict):
        """Handle order events from engine process."""
        try:
            from core.ipc_messages import Events

            symbol = payload.get('symbol', 'Unknown')
            order_id = payload.get('order_id', 'N/A')

            if event_type == Events.ORDER_FILLED:
                msg = f"Order FILLED: {symbol} (ID: {order_id})"
                level = 'success'
            elif event_type == Events.ORDER_REJECTED:
                msg = f"Order REJECTED: {symbol} (ID: {order_id})"
                level = 'danger'
            else:
                msg = f"Order {event_type}: {symbol} (ID: {order_id})"
                level = 'info'

            if hasattr(self, 'dashboard') and self.dashboard:
                self.dashboard.log_activity(msg, level)

            if self.alert_manager and event_type == Events.ORDER_FILLED:
                self.alert_manager.trade_alert(symbol, "Order Filled", str(order_id))

        except Exception as e:
            logger.debug(f"IPC order update error: {e}")

    def _handle_ipc_position(self, event_type: str, payload: dict):
        """Handle position events from engine process."""
        try:
            from core.ipc_messages import Events

            symbol = payload.get('symbol', 'Unknown')
            pnl = payload.get('pnl', 0)
            quantity = payload.get('quantity', 0)

            if event_type == Events.POSITION_CLOSED:
                direction = "Profit" if pnl >= 0 else "Loss"
                msg = f"Position CLOSED: {symbol} | {direction}: Rs.{abs(pnl):.2f}"
                level = 'success' if pnl >= 0 else 'danger'
                self.todays_pnl += pnl
                self.dashboard.update_pnl(self.todays_pnl)
            elif event_type == Events.POSITION_OPENED:
                msg = f"Position OPENED: {symbol} | Qty: {quantity}"
                level = 'info'
            else:
                msg = f"Position updated: {symbol} | PnL: Rs.{pnl:.2f}"
                level = 'info'

            if hasattr(self, 'dashboard') and self.dashboard:
                self.dashboard.log_activity(msg, level)

        except Exception as e:
            logger.debug(f"IPC position update error: {e}")

    def _handle_ipc_status(self, payload: dict):
        """Handle engine status updates."""
        try:
            status = payload.get('status', 'UNKNOWN')
            mode = payload.get('mode', '')

            if status == 'RUNNING':
                self.bot_running = True
                self.dashboard.update_bot_status(True)
                self.dashboard.log_activity(f"Engine RUNNING ({mode} mode)", 'success')
            elif status == 'STOPPED':
                self.bot_running = False
                self.dashboard.update_bot_status(False)
                self.dashboard.log_activity("Engine STOPPED", 'warning')
            elif status == 'PAUSED':
                self.dashboard.log_activity("Engine PAUSED", 'warning')
            elif status == 'INITIALIZED':
                self.dashboard.log_activity(f"Engine initialized ({mode} mode)", 'info')

        except Exception as e:
            logger.debug(f"IPC status update error: {e}")

    def _handle_ipc_error(self, payload: dict):
        """Handle engine error events."""
        try:
            error = payload.get('error', 'Unknown error')
            details = payload.get('details', '')

            if hasattr(self, 'dashboard') and self.dashboard:
                self.dashboard.log_activity(f"ENGINE ERROR: {error}", 'danger')
                if details:
                    self.dashboard.log_activity(f"Details: {details}", 'warning')

        except Exception as e:
            logger.debug(f"IPC error handling failed: {e}")

    def _handle_ipc_log(self, payload: dict):
        """Handle log message from engine process."""
        try:
            level = payload.get('level', 'info')
            message = payload.get('message', '')

            if hasattr(self, 'dashboard') and self.dashboard:
                self.dashboard.log_activity(message, level)

        except Exception as e:
            logger.debug(f"IPC log handling failed: {e}")

    def _check_engine_health(self):
        """
        Periodic check if engine process is alive and healthy.

        Auto-restarts engine if it crashes (with warning to user).
        """
        if not self._multiprocessing_mode or not self._engine_process:
            return

        try:
            if not self._engine_process.is_alive():
                logger.warning("Engine process died unexpectedly")
                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.log_activity("ENGINE CRASHED - check logs", 'danger')
                self.bot_running = False
                self.dashboard.update_bot_status(False)

            elif not self._engine_process.is_healthy(heartbeat_timeout=10.0):
                logger.warning("Engine process not responding (no heartbeat)")
                if hasattr(self, 'dashboard') and self.dashboard:
                    self.dashboard.log_activity("ENGINE NOT RESPONDING", 'warning')

        except Exception as e:
            logger.error(f"Error checking engine health: {e}")
        finally:
            # Check every 2 seconds
            self.root.after(2000, self._check_engine_health)

    # =========================================================================
    # Engine Control Methods (for multiprocessing mode)
    # =========================================================================

    def engine_start(self):
        """Start the trading engine (multiprocessing mode)."""
        if self._multiprocessing_mode and self._engine_process:
            from core.ipc_messages import Commands
            self._engine_process.send_command(Commands.START_ENGINE)
            logger.info("Sent START_ENGINE command")

    def engine_stop(self):
        """Stop the trading engine (multiprocessing mode)."""
        if self._multiprocessing_mode and self._engine_process:
            from core.ipc_messages import Commands
            self._engine_process.send_command(Commands.STOP_ENGINE)
            logger.info("Sent STOP_ENGINE command")

    def engine_pause(self):
        """Pause the trading engine (multiprocessing mode)."""
        if self._multiprocessing_mode and self._engine_process:
            from core.ipc_messages import Commands
            self._engine_process.send_command(Commands.PAUSE)
            logger.info("Sent PAUSE command")

    def engine_resume(self):
        """Resume the trading engine (multiprocessing mode)."""
        if self._multiprocessing_mode and self._engine_process:
            from core.ipc_messages import Commands
            self._engine_process.send_command(Commands.RESUME)
            logger.info("Sent RESUME command")

    def thread_safe_call(self, callback: Callable):
        """
        Schedule a callback to run on the main thread.

        Use this when updating UI elements from a background thread.
        The callback will be executed on the next queue processing cycle.

        Args:
            callback: Function to call on main thread (no arguments)

        Example:
            # From a background thread:
            self.thread_safe_call(lambda: self.label.config(text="Updated!"))
        """
        self.ui_queue.put(callback)

    def thread_safe_log(self, message: str, msg_type: str = 'info'):
        """
        Thread-safe logging to the activity feed.

        Args:
            message: Log message
            msg_type: 'info', 'success', 'warning', or 'error'
        """
        self.ui_queue.put(lambda: self.dashboard.log_activity(message, msg_type))

    def thread_safe_update_balance(self, balance: float):
        """Thread-safe balance update"""
        self.ui_queue.put(lambda: self.dashboard.update_balance(balance))

    def thread_safe_update_pnl(self, pnl: float):
        """Thread-safe P&L update"""
        self.ui_queue.put(lambda: self.dashboard.update_pnl(pnl))

    def thread_safe_update_status(self, connected: bool = None, bot_running: bool = None):
        """Thread-safe status updates"""
        if connected is not None:
            self.ui_queue.put(lambda: self.dashboard.update_connection(connected))
        if bot_running is not None:
            self.ui_queue.put(lambda: self.dashboard.update_bot_status(bot_running))

    def _update_dashboard(self):
        """Update dashboard with current data"""
        self.dashboard.update_balance(self.balance)
        self.dashboard.update_pnl(self.todays_pnl)
        self.root.after(5000, self._update_dashboard)

    def run(self):
        """Start the application"""
        logger.info("Starting AlgoTrader Pro...")
        self.dashboard.log_activity("AlgoTrader Pro started", 'info')
        self.dashboard.log_activity("Click 'Login' to connect to Zerodha", 'warning')
        self.root.mainloop()


def main():
    """Entry point"""
    app = AlgoTraderApp()
    app.run()


if __name__ == "__main__":
    main()
