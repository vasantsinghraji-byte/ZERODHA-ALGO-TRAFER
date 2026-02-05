"""
ALGOTRADER PRO - Professional Trading Platform
==============================================
A complete desktop application for algorithmic trading with Zerodha.

Features:
- Modern tabbed interface
- Live charts with technical indicators
- Automated trading bot with ORB strategy
- Manual trading capabilities
- Real-time P&L tracking
- Settings management
- Performance analytics
- One-click login and setup

© 2025 AlgoTrader Pro
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import webbrowser
import threading
import time
from datetime import datetime, time as dt_time
import json
from pathlib import Path
import queue  # For thread-safe communication
from typing import Dict, List, Optional

# Try importing required packages with helpful error messages
try:
    from kiteconnect import KiteConnect
except ImportError:
    messagebox.showerror("Missing Dependency",
                        "kiteconnect not installed!\n\n"
                        "Install with: pip install kiteconnect")
    raise

try:
    import yaml
except ImportError:
    messagebox.showerror("Missing Dependency",
                        "PyYAML not installed!\n\n"
                        "Install with: pip install pyyaml")
    raise

try:
    import pandas as pd
    import numpy as np
except ImportError:
    messagebox.showerror("Missing Dependencies",
                        "pandas/numpy not installed!\n\n"
                        "Install with: pip install pandas numpy")
    raise

try:
    import requests
except ImportError:
    messagebox.showerror("Missing Dependency",
                        "requests not installed!\n\n"
                        "Install with: pip install requests")
    raise

# Optional: matplotlib for charts
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("matplotlib not available - Charts will be disabled")

# Portfolio Analysis - Import portfolio package
try:
    from portfolio import (
        CorrelationAnalyzer, MarkowitzOptimizer, BlackLittermanOptimizer,
        SectorAnalyzer, PortfolioBacktester,
        BetaCalculator, DynamicHedger
    )
    PORTFOLIO_AVAILABLE = True
    print("[OK] Portfolio analysis package loaded successfully")
    print("     - CorrelationAnalyzer, MarkowitzOptimizer, BlackLittermanOptimizer")
    print("     - SectorAnalyzer, PortfolioBacktester, BetaCalculator, DynamicHedger")
except ImportError as e:
    PORTFOLIO_AVAILABLE = False
    print(f"[WARNING] Portfolio analysis not available (ImportError)")
    print(f"         Error: {e}")
    print("         Portfolio tab will be disabled")
    print("         Run: python diagnose_portfolio.py for details")
except Exception as e:
    PORTFOLIO_AVAILABLE = False
    print(f"[ERROR] Error loading portfolio package")
    print(f"        Error: {e}")
    print(f"        Type: {type(e).__name__}")
    import traceback
    print("        Traceback:")
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            print(f"        {line}")

# Machine Learning - Import ML package
try:
    from ml import (
        MLEngine, RandomForestModel, XGBoostModel, LSTMModel,
        AutoFeatureEngineer,
        SentimentAnalyzer, AggregatedSentiment,
        AnomalyDetector, StatisticalAnomalyDetector, AnomalyType,
        PatternClassifier, PatternType,
        QLearningAgent, DQNAgent, TradingEnvironment, RLTrainer
    )
    ML_AVAILABLE = True
    print("[OK] Machine Learning package loaded successfully")
    print("     - ML Engine: RandomForest, XGBoost, LSTM")
    print("     - Feature Engineering: 100+ automated features")
    print("     - Sentiment Analysis: News & social media")
    print("     - Anomaly Detection: Statistical, Isolation Forest")
    print("     - Pattern Classification: Candlestick & chart patterns")
    print("     - Reinforcement Learning: Q-Learning, DQN")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[WARNING] Machine Learning not available (ImportError)")
    print(f"         Error: {e}")
    print("         ML tab will be disabled")
except Exception as e:
    ML_AVAILABLE = False
    print(f"[ERROR] Error loading ML package")
    print(f"        Error: {e}")
    print(f"        Type: {type(e).__name__}")
    import traceback
    print("        Traceback:")
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            print(f"        {line}")

# Constants
CONFIG_DIR = Path(__file__).parent / "config"
BOT_API_URL = "http://localhost:8000"

# Load secrets from config file (NEVER hardcode credentials!)
class SecretsLoadError(Exception):
    """Raised when secrets cannot be loaded - application cannot function without credentials."""
    pass


def load_secrets():
    """
    Load API secrets from config/secrets.yaml.

    FAIL FAST: Raises SecretsLoadError if secrets cannot be loaded.
    The application cannot function without valid API credentials.

    Raises:
        SecretsLoadError: If the secrets file is missing, corrupted, or unreadable.
    """
    secrets_path = CONFIG_DIR / 'secrets.yaml'

    # Check if file exists
    if not secrets_path.exists():
        raise SecretsLoadError(
            f"Secrets file not found: {secrets_path}\n\n"
            f"To fix this:\n"
            f"1. Copy secrets.yaml.example to secrets.yaml\n"
            f"2. Add your Zerodha API credentials\n"
            f"3. Restart the application\n\n"
            f"Or use environment variables:\n"
            f"  ZERODHA_API_KEY, ZERODHA_API_SECRET"
        )

    # Try to load and parse the file
    try:
        with open(secrets_path, 'r') as f:
            secrets = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SecretsLoadError(
            f"Invalid YAML in secrets file: {secrets_path}\n"
            f"Error: {e}\n\n"
            f"Please fix the YAML syntax or restore from secrets.yaml.example"
        ) from e
    except PermissionError as e:
        raise SecretsLoadError(
            f"Permission denied reading: {secrets_path}\n"
            f"Check file permissions."
        ) from e

    # Validate that we got a dict
    if not isinstance(secrets, dict):
        raise SecretsLoadError(
            f"Invalid secrets file format: {secrets_path}\n"
            f"Expected a YAML dictionary, got: {type(secrets).__name__}"
        )

    # Extract credentials (empty strings are still returned, but at least the file was valid)
    return {
        'api_key': secrets.get('zerodha', {}).get('api_key', ''),
        'api_secret': secrets.get('zerodha', {}).get('api_secret', ''),
        'bot_api_key': secrets.get('api_secret_key', '')
    }

SECRETS = load_secrets()
API_KEY = SECRETS['api_key']
API_SECRET = SECRETS['api_secret']
BOT_API_KEY = SECRETS['bot_api_key']

# CYBER-MINIMALIST COLOR SCHEME
COLORS = {
    # Dark Backgrounds (Professional algo trading theme)
    'bg_primary': '#121212',      # Deep charcoal - Main background
    'bg_secondary': '#1E1E1E',    # Slightly lighter - Panels
    'bg_tertiary': '#2A2A2A',     # Widget backgrounds
    'bg_hover': '#333333',        # Hover state

    # Neon Accents (High-contrast for quick recognition)
    'neon_green': '#00FF94',      # Profit/Active (Electric green)
    'neon_red': '#FF4D4D',        # Loss/Error (Bright red)
    'neon_amber': '#FFC107',      # Warnings (Gold)
    'neon_blue': '#00D4FF',       # Info (Cyan)
    'neon_purple': '#B721FF',     # Special actions

    # Text Colors
    'text_primary': '#FFFFFF',    # White - Main text
    'text_secondary': '#B0B0B0',  # Grey - Labels
    'text_dim': '#707070',        # Dimmed - Hints

    # Borders & Grid
    'border': '#404040',
    'grid': '#2A2A2A',

    # Legacy colors (for backward compatibility)
    'primary': '#00D4FF',         # Neon blue
    'success': '#00FF94',         # Neon green
    'danger': '#FF4D4D',          # Neon red
    'warning': '#FFC107',         # Neon amber
}

class AlgoTraderPro:
    def __init__(self, root):
        self.root = root
        self.root.title("◢ ALGOTRADER PRO ◣ Mission Control")

        # Get screen dimensions and set to 90% of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = int(screen_width * 0.9)
        height = int(screen_height * 0.9)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Dark theme background
        self.root.configure(bg=COLORS['bg_primary'])

        # Set app icon (if exists)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

        # Variables
        self.kite = None
        self.access_token = None
        self.positions = {}
        self.orders = []
        self.pnl = 0.0
        self.bot_enabled = False
        self.live_prices = {}
        self.historical_data = {}
        self.update_threads = []

        # Thread-safe queue for UI updates
        self.ui_update_queue = queue.Queue()

        # Initialize instrument and watchlist managers
        from utils import InstrumentManager, WatchlistManager
        self.instrument_manager = InstrumentManager()
        self.watchlist_manager = WatchlistManager()

        # Initialize master orchestrator (full automation)
        self.master_orchestrator = None
        self.automation_enabled = False

        # Load settings
        self.load_settings()

        # Create menu bar
        self.create_menu_bar()

        # Create UI
        self.create_ui()

        # Try to auto-login
        self.try_auto_login()

        # Start update loops
        self.start_update_loops()

        # Start UI update processor (runs in main thread)
        self.process_ui_updates()

    def process_ui_updates(self):
        """Process UI updates from queue (thread-safe)"""
        try:
            # Process all pending updates
            while True:
                update_type, data = self.ui_update_queue.get_nowait()

                if update_type == 'watchlist':
                    self._update_watchlist_ui(data)
                elif update_type == 'bot_stats':
                    self._update_bot_stats_ui(data)
                elif update_type == 'stat_card':
                    self._update_stat_card_ui(data)
                elif update_type == 'log':
                    self.log_activity(data)
                elif update_type == 'corr_result':
                    self._update_correlation_ui(data)
                elif update_type == 'corr_error':
                    self._show_correlation_error(data)
                elif update_type == 'opt_result':
                    self._update_optimization_ui(data)
                elif update_type == 'opt_error':
                    self._show_optimization_error(data)

        except queue.Empty:
            pass
        finally:
            # Schedule next check (every 100ms)
            self.root.after(100, self.process_ui_updates)

    def _update_watchlist_ui(self, prices):
        """Update watchlist in UI (called from main thread) - NO FLICKERING"""
        if not hasattr(self, 'watchlist_tree'):
            return

        # Build a map of existing items by symbol
        existing_items = {}
        for item_id in self.watchlist_tree.get_children():
            symbol = self.watchlist_tree.item(item_id)['values'][0]
            existing_items[symbol] = item_id

        # Update existing items or add new ones
        for symbol, data in prices.items():
            ltp = data.get('last_price', 0)
            volume = data.get('volume', 0)

            # Calculate change from previous close
            ohlc = data.get('ohlc', {})
            prev_close = 0
            if isinstance(ohlc, dict):
                prev_close = ohlc.get('close', 0)

            if prev_close > 0 and ltp > 0:
                change = ltp - prev_close
                change_pct = (change / prev_close) * 100
            else:
                change = 0
                change_pct = 0

            change_str = f"+₹{change:.2f}" if change >= 0 else f"₹{change:.2f}"
            change_pct_str = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
            volume_str = f"{volume:,}"

            # Update existing item or add new
            if symbol in existing_items:
                # Update in-place (no flickering)
                self.watchlist_tree.item(existing_items[symbol], values=(
                    symbol, f"₹{ltp:.2f}", change_str, change_pct_str, volume_str
                ))
                # Color code
                if change_pct > 0:
                    self.watchlist_tree.item(existing_items[symbol], tags=('positive',))
                elif change_pct < 0:
                    self.watchlist_tree.item(existing_items[symbol], tags=('negative',))
            else:
                # Add new item
                self.watchlist_tree.insert('', tk.END,
                    values=(symbol, f"₹{ltp:.2f}", change_str, change_pct_str, volume_str))

        # Configure tag colors
        self.watchlist_tree.tag_configure('positive', foreground=COLORS['neon_green'])
        self.watchlist_tree.tag_configure('negative', foreground=COLORS['neon_red'])

    def _update_bot_stats_ui(self, status):
        """Update bot stats in UI (called from main thread)"""
        if not hasattr(self, 'bot_stats_text'):
            return

        # Check if this is an error status
        if status.get('environment') == 'ERROR':
            stats_text = f"⚠️ Cannot connect to bot API!\n\n{status.get('error_message', 'Unknown error')}"
        else:
            stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ALGO BOT STATUS                           ║
╚══════════════════════════════════════════════════════════════╝

✓ Bot Running: {status.get('is_running', False)}
✓ Environment: {status.get('environment', 'N/A').upper()}
✓ Trading Mode: {'PAPER' if not status.get('live_trading', False) else 'LIVE'}
✓ Uptime: {int(status.get('uptime_seconds', 0) / 3600)}h {int((status.get('uptime_seconds', 0) % 3600) / 60)}m

📊 STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Signals Generated: {status.get('signals_generated', 0)}
• Trades Executed: {status.get('trades_executed', 0)}
• Open Positions: {status.get('open_positions_count', 0)}
• Active Orders: {status.get('active_orders_count', 0)}

📈 MONITORED INSTRUMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• RELIANCE (NSE: 738561)
• TCS (NSE: 2953217)
            """

        self.bot_stats_text.delete(1.0, tk.END)
        self.bot_stats_text.insert(1.0, stats_text)

    def _update_stat_card_ui(self, data):
        """Update stat card in UI (called from main thread)"""
        if 'stat_cards' not in dir(self):
            return

        card_name = data.get('name')
        value = data.get('value')

        if card_name in self.stat_cards:
            self.stat_cards[card_name].config(text=str(value))

    def _update_correlation_ui(self, data):
        """Update correlation analysis results in UI"""
        if not hasattr(self, 'corr_matrix_text'):
            return

        matrix_str, insights_str = data

        self.corr_matrix_text.delete('1.0', tk.END)
        self.corr_matrix_text.insert('1.0', matrix_str)

        self.corr_insights_text.delete('1.0', tk.END)
        self.corr_insights_text.insert('1.0', insights_str)

    def _show_correlation_error(self, error_msg):
        """Show correlation error in UI"""
        if not hasattr(self, 'corr_matrix_text'):
            return

        self.corr_matrix_text.delete('1.0', tk.END)
        self.corr_matrix_text.insert('1.0', f"Error: {error_msg}")

    def _update_optimization_ui(self, data):
        """Update portfolio optimization results in UI"""
        if not hasattr(self, 'opt_weights_text'):
            return

        weights_str, stats = data

        self.opt_weights_text.delete('1.0', tk.END)
        self.opt_weights_text.insert('1.0', weights_str)

        # Update stats labels
        self.opt_stats_labels['Expected Return'].config(text=f"{stats['return']:.2%}")
        self.opt_stats_labels['Volatility'].config(text=f"{stats['volatility']:.2%}")
        self.opt_stats_labels['Sharpe Ratio'].config(text=f"{stats['sharpe']:.3f}")

    def _show_optimization_error(self, error_msg):
        """Show optimization error in UI"""
        if not hasattr(self, 'opt_weights_text'):
            return

        self.opt_weights_text.delete('1.0', tk.END)
        self.opt_weights_text.insert('1.0', f"Error: {error_msg}")

    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Login", command=self.login)
        file_menu.add_command(label="Logout", command=self.logout)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Bot menu
        bot_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bot", menu=bot_menu)
        bot_menu.add_command(label="Start Bot", command=self.start_bot)
        bot_menu.add_command(label="Stop Bot", command=self.stop_bot)
        bot_menu.add_separator()
        bot_menu.add_command(label="Bot Settings", command=self.show_bot_settings)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Backtest Strategy", command=self.run_backtest)
        tools_menu.add_command(label="Performance Report", command=self.show_performance)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self.show_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Start Guide", command=self.show_quick_guide)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

    def create_ui(self):
        """Create main UI with tabbed interface"""

        # Top bar with status
        self.create_top_bar()

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tab style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Arial', 10, 'bold'))

        # Create tabs
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_bot_tab()
        self.create_charts_tab()
        self.create_positions_tab()
        self.create_analytics_tab()
        self.create_backtest_tab()
        self.create_scanner_tab()
        self.create_portfolio_tab()
        self.create_ml_tab()

        # Status bar
        self.create_status_bar()

    def create_top_bar(self):