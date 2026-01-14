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
- Multiple views (Dashboard, Charts, Strategies, Settings)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import os
import json

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
            font=('Segoe UI', 12),
            anchor=tk.W,
            padx=15,
            pady=12
        )
        self.label.pack(fill=tk.X)

        # Bind click
        self.frame.bind('<Button-1>', lambda e: command() if command else None)
        self.label.bind('<Button-1>', lambda e: command() if command else None)

        # Hover effects
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

        # Get screen size and set window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.85))
        window_height = min(900, int(screen_height * 0.85))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Load theme
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

        # Settings
        self.settings = self._load_settings()

        # Broker (will be initialized on login)
        self.broker = None

        # Views
        self.views: Dict[str, tk.Frame] = {}
        self.nav_buttons: Dict[str, NavigationButton] = {}

        # Build UI
        self._create_styles()
        self._create_ui()

        # Start update loop
        self._start_updates()

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
        # Main container with sidebar
        main_container = tk.Frame(self.root, bg=self.theme['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self._create_sidebar(main_container)

        # Content area
        self.content_frame = tk.Frame(main_container, bg=self.theme['bg_primary'])
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header
        self._create_header(self.content_frame)

        # View container
        self.view_container = tk.Frame(self.content_frame, bg=self.theme['bg_primary'])
        self.view_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Create all views
        self._create_views()

        # Show default view
        self._show_view('dashboard')

    def _create_sidebar(self, parent):
        """Create navigation sidebar"""
        sidebar = tk.Frame(parent, bg=self.theme['bg_secondary'], width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Logo
        logo_frame = tk.Frame(sidebar, bg=self.theme['bg_secondary'])
        logo_frame.pack(fill=tk.X, pady=20)

        tk.Label(
            logo_frame,
            text="AlgoTrader",
            bg=self.theme['bg_secondary'],
            fg=self.theme['accent'],
            font=('Segoe UI', 18, 'bold')
        ).pack()

        tk.Label(
            logo_frame,
            text="PRO",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        ).pack()

        # Separator
        tk.Frame(sidebar, bg=self.theme['border'], height=1).pack(fill=tk.X, padx=15, pady=10)

        # Navigation buttons
        nav_items = [
            ('dashboard', 'Dashboard', '📊'),
            ('charts', 'Charts', '📈'),
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

        # Spacer
        tk.Frame(sidebar, bg=self.theme['bg_secondary']).pack(fill=tk.BOTH, expand=True)

        # Bottom section - Connection status
        bottom = tk.Frame(sidebar, bg=self.theme['bg_secondary'])
        bottom.pack(fill=tk.X, padx=15, pady=15)

        self.conn_indicator = tk.Label(
            bottom,
            text="● Disconnected",
            bg=self.theme['bg_secondary'],
            fg=self.theme['danger'],
            font=('Segoe UI', 10)
        )
        self.conn_indicator.pack(anchor=tk.W)

        self.mode_indicator = tk.Label(
            bottom,
            text="Paper Trading",
            bg=self.theme['bg_secondary'],
            fg=self.theme['info'],
            font=('Segoe UI', 9)
        )
        self.mode_indicator.pack(anchor=tk.W)

    def _create_header(self, parent):
        """Create header with controls"""
        header = tk.Frame(parent, bg=self.theme['bg_primary'])
        header.pack(fill=tk.X, padx=20, pady=15)

        # Left - Title (dynamic based on view)
        self.view_title = tk.Label(
            header,
            text="Dashboard",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 24, 'bold')
        )
        self.view_title.pack(side=tk.LEFT)

        # Right - Quick controls
        controls = tk.Frame(header, bg=self.theme['bg_primary'])
        controls.pack(side=tk.RIGHT)

        # Bot toggle
        self.bot_btn = tk.Button(
            controls,
            text="▶ START BOT",
            bg=self.theme['btn_success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._toggle_bot
        )
        self.bot_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=15, ipady=5)

        # Login button
        self.login_btn = tk.Button(
            controls,
            text="🔐 Login",
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
        """Create charts view with controls"""
        # Controls bar
        controls = tk.Frame(parent, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, pady=(0, 15))

        # Symbol selector
        tk.Label(
            controls,
            text="Symbol:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT)

        self.chart_symbol = ttk.Combobox(
            controls,
            values=['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBIN'],
            state='readonly',
            width=15
        )
        self.chart_symbol.set('RELIANCE')
        self.chart_symbol.pack(side=tk.LEFT, padx=(5, 20))

        # Timeframe selector
        tk.Label(
            controls,
            text="Timeframe:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT)

        self.chart_timeframe = ttk.Combobox(
            controls,
            values=['1 Min', '5 Min', '15 Min', '1 Hour', '1 Day'],
            state='readonly',
            width=10
        )
        self.chart_timeframe.set('1 Day')
        self.chart_timeframe.pack(side=tk.LEFT, padx=(5, 20))

        # Load button
        tk.Button(
            controls,
            text="📊 Load Chart",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._load_chart
        ).pack(side=tk.LEFT, ipadx=10, ipady=3)

        # Indicator toggles
        tk.Label(
            controls,
            text="     Indicators:",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT)

        self.show_ma = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="MA",
            variable=self.show_ma,
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'],
            activebackground=self.theme['bg_primary']
        ).pack(side=tk.LEFT, padx=5)

        self.show_bb = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls,
            text="Bollinger",
            variable=self.show_bb,
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'],
            activebackground=self.theme['bg_primary']
        ).pack(side=tk.LEFT, padx=5)

        # Chart container
        self.chart_container = tk.Frame(parent, bg=self.theme['bg_card'])
        self.chart_container.configure(
            highlightbackground=self.theme['border'],
            highlightthickness=1
        )
        self.chart_container.pack(fill=tk.BOTH, expand=True)

        # Placeholder message
        tk.Label(
            self.chart_container,
            text="📊 Click 'Load Chart' to display price data\n\nConnect to Zerodha to load live data",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 14),
            justify=tk.CENTER
        ).pack(expand=True)

    def _show_view(self, view_name: str):
        """Switch to a different view"""
        # Hide all views
        for view in self.views.values():
            view.pack_forget()

        # Show selected view
        if view_name in self.views:
            self.views[view_name].pack(fill=tk.BOTH, expand=True)

        # Update navigation
        for key, btn in self.nav_buttons.items():
            btn.set_selected(key == view_name)

        # Update title
        titles = {
            'dashboard': '📊 Dashboard',
            'charts': '📈 Charts',
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

        # Center dialog
        dialog.update_idletasks()
        x = (self.root.winfo_width() - 450) // 2 + self.root.winfo_x()
        y = (self.root.winfo_height() - 350) // 2 + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")

        # Content
        inner = tk.Frame(dialog, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        tk.Label(
            inner,
            text="🔐 Login to Zerodha",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 18, 'bold')
        ).pack(pady=(0, 20))

        # API Key
        tk.Label(
            inner, text="API Key",
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(anchor=tk.W)

        api_key_entry = tk.Entry(
            inner, width=45,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT
        )
        api_key_entry.pack(fill=tk.X, pady=(5, 15), ipady=5)

        # API Secret
        tk.Label(
            inner, text="API Secret",
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(anchor=tk.W)

        api_secret_entry = tk.Entry(
            inner, width=45, show="*",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT
        )
        api_secret_entry.pack(fill=tk.X, pady=(5, 20), ipady=5)

        # Paper trading toggle
        paper_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            inner,
            text="Start in Paper Trading mode (recommended)",
            variable=paper_var,
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            selectcolor=self.theme['bg_secondary'],
            activebackground=self.theme['bg_card'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=(0, 20))

        def do_login():
            api_key = api_key_entry.get().strip()
            api_secret = api_secret_entry.get().strip()

            if not api_key or not api_secret:
                messagebox.showerror("Error", "Please enter both API Key and Secret!")
                return

            # Update state
            self.connected = True
            self.paper_trading = paper_var.get()

            # Update UI
            self.conn_indicator.config(text="● Connected", fg=self.theme['success'])
            self.mode_indicator.config(
                text="Paper Trading" if self.paper_trading else "LIVE Trading",
                fg=self.theme['info'] if self.paper_trading else self.theme['warning']
            )
            self.login_btn.config(text="✓ Connected", state=tk.DISABLED, bg=self.theme['success'])

            # Update dashboard
            self.dashboard.update_connection(True)
            self.dashboard.log_activity("Connected to Zerodha!", 'success')

            dialog.destroy()

        tk.Button(
            inner,
            text="🚀 Connect",
            bg=self.theme['btn_primary'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            command=do_login
        ).pack(fill=tk.X, ipady=10)

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
        else:
            self.bot_running = False
            self.bot_btn.config(text="▶ START BOT", bg=self.theme['btn_success'])
            self.dashboard.update_bot_status(False)
            self.dashboard.log_activity("Bot stopped", 'warning')

    def _on_strategy_selected(self, strategy_key: str):
        """Handle strategy selection"""
        self.selected_strategy = strategy_key
        info = STRATEGY_INFO.get(strategy_key, {})
        self.dashboard.log_activity(
            f"Strategy changed to {info.get('emoji', '')} {info.get('name', strategy_key)}",
            'info'
        )

    def _load_chart(self):
        """Load chart with sample data"""
        import pandas as pd
        import numpy as np

        # Clear chart container
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        # Generate sample data
        np.random.seed(42)
        days = 60
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = 2500.0
        returns = np.random.randn(days) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, days)
        }, index=dates)

        # Create chart
        symbol = self.chart_symbol.get()
        chart = CandlestickChart(self.chart_container, 800, 500)
        chart.plot(data, f"{symbol} - {self.chart_timeframe.get()}")

        # Add indicators
        if self.show_ma.get():
            add_moving_average(chart.ax_price, data, 20)

        if self.show_bb.get():
            add_bollinger_bands(chart.ax_price, data)

        chart.canvas.draw()

        self.dashboard.log_activity(f"Loaded chart for {symbol}", 'info')

    def _start_updates(self):
        """Start periodic updates"""
        self._update_dashboard()

    def _update_dashboard(self):
        """Update dashboard with current data"""
        # Update balance and P&L
        self.dashboard.update_balance(self.balance)
        self.dashboard.update_pnl(self.todays_pnl)

        # Schedule next update
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
