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
def load_secrets():
    """Load API secrets from config/secrets.yaml"""
    try:
        with open(CONFIG_DIR / 'secrets.yaml', 'r') as f:
            secrets = yaml.safe_load(f)
        return {
            'api_key': secrets.get('zerodha', {}).get('api_key', ''),
            'api_secret': secrets.get('zerodha', {}).get('api_secret', ''),
            'bot_api_key': secrets.get('api_secret_key', '')
        }
    except Exception as e:
        print(f"⚠️ Error loading secrets: {e}")
        return {'api_key': '', 'api_secret': '', 'bot_api_key': ''}

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