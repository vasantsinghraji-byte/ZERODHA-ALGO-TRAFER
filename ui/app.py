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
import webbrowser
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import os
import json
import numpy as np
import pandas as pd

from .themes import get_theme, THEMES
from .dashboard import Dashboard
from .charts import CandlestickChart, SimpleChart, add_moving_average, add_bollinger_bands
from .strategy_picker import StrategyPicker, STRATEGY_INFO
from .settings_panel import SettingsPanel, SettingsDialog

logger = logging.getLogger(__name__)


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
    """

    def __init__(self):
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

    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data for multiple stocks"""
        np.random.seed(42)
        stocks = {
            'RELIANCE': (2500, 0.02, 0.0005),
            'TCS': (3500, 0.018, 0.0004),
            'INFY': (1500, 0.022, 0.0003),
            'HDFC': (2800, 0.015, 0.0003),
            'ICICIBANK': (950, 0.025, 0.0002),
            'SBIN': (600, 0.03, 0.0004),
            'BHARTIARTL': (1200, 0.02, 0.0003),
            'ITC': (450, 0.012, 0.0002),
        }

        data = {}
        for symbol, (base, vol, trend) in stocks.items():
            days = 100
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            returns = np.random.randn(days) * vol + trend
            prices = base * np.exp(np.cumsum(returns))

            data[symbol] = pd.DataFrame({
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
        """Load settings from file"""
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ui_settings.json')
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")

        return {
            'api': {'api_key': '', 'api_secret': '', 'user_id': ''},
            'trading': {'paper_trading': True, 'initial_capital': 100000, 'max_positions': 5},
            'risk': {'risk_per_trade': 2.0, 'max_daily_loss': 5.0},
            'appearance': {'theme': 'dark', 'show_emojis': True}
        }

    def _save_settings(self, settings: Dict[str, Any]):
        """Save settings to file"""
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ui_settings.json')
        try:
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            self.settings = settings
            logger.info("Settings saved")
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

        # Navigation - now with 7 items including new advanced features
        nav_items = [
            ('dashboard', 'Dashboard', '📊'),
            ('charts', 'Charts', '📈'),
            ('scanner', 'Scanner', '🔍'),
            ('predictions', 'AI Predict', '🤖'),
            ('portfolio', 'Portfolio', '💼'),
            ('strategies', 'Strategies', '🎯'),
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
        """Create charts view"""
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        tk.Label(controls, text="Symbol:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.chart_symbol = ttk.Combobox(
            controls, values=list(self.sample_data.keys()),
            state='readonly', width=15
        )
        self.chart_symbol.set('RELIANCE')
        self.chart_symbol.pack(side=tk.LEFT, padx=(5, 20))

        tk.Label(controls, text="Timeframe:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.chart_timeframe = ttk.Combobox(
            controls, values=['1 Min', '5 Min', '15 Min', '1 Hour', '1 Day'],
            state='readonly', width=10
        )
        self.chart_timeframe.set('1 Day')
        self.chart_timeframe.pack(side=tk.LEFT, padx=(5, 20))

        tk.Button(
            controls, text="📊 Load Chart",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 10), relief=tk.FLAT,
            cursor='hand2', command=self._load_chart
        ).pack(side=tk.LEFT, ipadx=10, ipady=3)

        tk.Label(controls, text="     Indicators:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.show_ma = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="MA", variable=self.show_ma,
                      bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
                      selectcolor=self.theme['bg_secondary']).pack(side=tk.LEFT, padx=5)

        self.show_bb = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Bollinger", variable=self.show_bb,
                      bg=self.theme['bg_primary'], fg=self.theme['text_primary'],
                      selectcolor=self.theme['bg_secondary']).pack(side=tk.LEFT, padx=5)

        self.chart_container = tk.Frame(parent, bg=self.theme['bg_card'])
        self.chart_container.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        self.chart_container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.chart_container,
            text="📊 Click 'Load Chart' to display price data",
            bg=self.theme['bg_card'], fg=self.theme['text_dim'],
            font=('Segoe UI', 14), justify=tk.CENTER
        ).pack(expand=True)

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

        tk.Button(
            controls, text="🔍 Run Scan",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', command=self._run_scan
        ).pack(side=tk.LEFT, ipadx=15, ipady=5)

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
        style.configure("Scanner.Treeview", background=self.theme['bg_secondary'],
                       foreground=self.theme['text_primary'], fieldbackground=self.theme['bg_secondary'],
                       rowheight=35)
        style.configure("Scanner.Treeview.Heading", background=self.theme['bg_card'],
                       foreground=self.theme['text_primary'], font=('Segoe UI', 10, 'bold'))

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
        """Create AI predictions view"""
        # Controls
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        tk.Label(controls, text="Symbol:", bg=self.theme['bg_primary'],
                fg=self.theme['text_secondary'], font=('Segoe UI', 11)).pack(side=tk.LEFT)

        self.predict_symbol = ttk.Combobox(
            controls, values=list(self.sample_data.keys()),
            state='readonly', width=15
        )
        self.predict_symbol.set('RELIANCE')
        self.predict_symbol.pack(side=tk.LEFT, padx=(5, 20))

        tk.Button(
            controls, text="🤖 Get AI Prediction",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', command=self._get_prediction
        ).pack(side=tk.LEFT, ipadx=15, ipady=5)

        tk.Button(
            controls, text="📊 Full Analysis",
            bg=self.theme['bg_secondary'], fg=self.theme['text_primary'],
            font=('Segoe UI', 11), relief=tk.FLAT,
            cursor='hand2', command=self._get_full_analysis
        ).pack(side=tk.LEFT, padx=(10, 0), ipadx=15, ipady=5)

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

        self.prediction_symbol_label = tk.Label(left_inner, text="Select a symbol",
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

        tk.Button(
            controls, text="💼 Optimize Portfolio",
            bg=self.theme['btn_primary'], fg='white',
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', command=self._optimize_portfolio
        ).pack(side=tk.LEFT, ipadx=15, ipady=5)

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
        """Toggle bot on/off"""
        if not self.connected:
            messagebox.showwarning("Not Connected", "Please login to Zerodha first!")
            return

        if not self.bot_running:
            self.bot_running = True
            self.bot_btn.config(text="⏹ STOP BOT", bg=self.theme['btn_danger'])
            self.dashboard.update_bot_status(True)
            self.dashboard.log_activity(f"Bot started with {self.selected_strategy.upper()} strategy", 'success')

            if self.alert_manager:
                self.alert_manager.bot_status("Started", f"Using {self.selected_strategy} strategy")
        else:
            self.bot_running = False
            self.bot_btn.config(text="▶ START BOT", bg=self.theme['btn_success'])
            self.dashboard.update_bot_status(False)
            self.dashboard.log_activity("Bot stopped", 'warning')

            if self.alert_manager:
                self.alert_manager.bot_status("Stopped")

    def _on_strategy_selected(self, strategy_key: str):
        """Handle strategy selection"""
        self.selected_strategy = strategy_key
        info = STRATEGY_INFO.get(strategy_key, {})
        self.dashboard.log_activity(
            f"Strategy changed to {info.get('emoji', '')} {info.get('name', strategy_key)}",
            'info'
        )

    def _load_chart(self):
        """Load chart"""
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        symbol = self.chart_symbol.get()
        data = self.sample_data.get(symbol)

        if data is None:
            return

        chart = CandlestickChart(self.chart_container, 800, 500)
        chart.plot(data, f"{symbol} - {self.chart_timeframe.get()}")

        if self.show_ma.get():
            add_moving_average(chart.ax_price, data, 20)

        if self.show_bb.get():
            add_bollinger_bands(chart.ax_price, data)

        chart.canvas.draw()
        self.dashboard.log_activity(f"Loaded chart for {symbol}", 'info')

    def _run_scan(self):
        """Run market scanner"""
        try:
            from advanced.market_scanner import MarketScanner, ScanType, ScanFilter

            # Clear existing results
            for item in self.scanner_tree.get_children():
                self.scanner_tree.delete(item)

            # Map scan type
            scan_map = {
                'Momentum': [ScanType.MOMENTUM],
                'Breakout': [ScanType.BREAKOUT],
                'Oversold': [ScanType.OVERSOLD],
                'Volume Spike': [ScanType.VOLUME_SPIKE],
                'All': None
            }
            scan_types = scan_map.get(self.scan_type.get())

            # Run scan
            scanner = MarketScanner()
            min_score = int(self.min_score.get())
            filter_criteria = ScanFilter(min_score=min_score)

            results = scanner.scan_watchlist(self.sample_data, scan_types, filter_criteria)

            # Display results
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

        except Exception as e:
            logger.error(f"Scan error: {e}")
            messagebox.showerror("Scan Error", f"Could not run scan: {e}")

    def _get_prediction(self):
        """Get AI prediction for selected symbol"""
        try:
            from advanced.ml_predictor import MLPredictor

            symbol = self.predict_symbol.get()
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

            symbol = self.predict_symbol.get()
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
        """Optimize portfolio allocation"""
        try:
            from advanced.portfolio_optimizer import PortfolioOptimizer, OptimizationGoal

            # Map goal
            goal_map = {
                'Max Sharpe Ratio': OptimizationGoal.MAX_SHARPE,
                'Min Volatility': OptimizationGoal.MIN_VOLATILITY,
                'Risk Parity': OptimizationGoal.RISK_PARITY,
                'Equal Weight': OptimizationGoal.EQUAL_WEIGHT,
            }
            goal = goal_map.get(self.opt_goal.get(), OptimizationGoal.MAX_SHARPE)

            # Get capital
            try:
                capital = float(self.opt_capital.get())
            except:
                capital = 100000

            # Run optimization
            optimizer = PortfolioOptimizer()
            optimizer.load_data(self.sample_data)
            result = optimizer.optimize(goal)

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

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            messagebox.showerror("Optimization Error", f"Could not optimize: {e}")

    def _start_updates(self):
        """Start periodic updates"""
        self._update_dashboard()

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
