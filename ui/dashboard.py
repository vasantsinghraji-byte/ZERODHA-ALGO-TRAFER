# -*- coding: utf-8 -*-
"""
Dashboard Component - Your Trading Command Center!
===================================================
Shows everything at a glance like a car dashboard.

Displays:
- Account balance and P&L
- Active positions
- Recent trades
- Bot status
- Quick stats
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

from .themes import get_theme

logger = logging.getLogger(__name__)


class StatusWidget:
    """
    Status indicator widget.

    Shows status with colored dot:
    - Green = Good/Running
    - Yellow = Warning/Paused
    - Red = Error/Stopped
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme
        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        self.status_colors = {
            'running': theme['success'],
            'paused': theme['warning'],
            'stopped': theme['danger'],
            'connected': theme['success'],
            'disconnected': theme['danger'],
        }

        self.dot_label = tk.Label(
            self.frame, text="●",
            bg=theme['bg_card'],
            fg=theme['text_dim'],
            font=('Segoe UI', 14)
        )
        self.dot_label.pack(side=tk.LEFT, padx=(0, 5))

        self.text_label = tk.Label(
            self.frame, text="Unknown",
            bg=theme['bg_card'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 11)
        )
        self.text_label.pack(side=tk.LEFT)

    def set_status(self, status: str, text: str = None):
        """Update status display"""
        color = self.status_colors.get(status.lower(), self.theme['text_dim'])
        self.dot_label.config(fg=color)
        self.text_label.config(text=text or status.capitalize())

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class MetricCard:
    """
    A card showing a single metric with icon and value.

    Example: 💰 Balance: ₹1,00,000
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        title: str,
        emoji: str = "📊",
        value: str = "0",
        subtitle: str = ""
    ):
        self.theme = theme

        # Card frame
        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        inner = tk.Frame(self.frame, bg=theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header with emoji and title
        header = tk.Frame(inner, bg=theme['bg_card'])
        header.pack(fill=tk.X)

        tk.Label(
            header, text=emoji,
            bg=theme['bg_card'],
            font=('Segoe UI', 20)
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text=title,
            bg=theme['bg_card'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Value
        self.value_label = tk.Label(
            inner, text=value,
            bg=theme['bg_card'],
            fg=theme['text_primary'],
            font=('Segoe UI', 24, 'bold')
        )
        self.value_label.pack(anchor=tk.W, pady=(10, 0))

        # Subtitle
        if subtitle:
            self.subtitle_label = tk.Label(
                inner, text=subtitle,
                bg=theme['bg_card'],
                fg=theme['text_dim'],
                font=('Segoe UI', 10)
            )
            self.subtitle_label.pack(anchor=tk.W)
        else:
            self.subtitle_label = None

    def update_value(self, value: str, color: str = None):
        """Update the displayed value"""
        self.value_label.config(text=value)
        if color:
            self.value_label.config(fg=color)

    def update_subtitle(self, text: str):
        """Update subtitle text"""
        if self.subtitle_label:
            self.subtitle_label.config(text=text)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class PositionsTable:
    """
    Table showing open positions.

    Columns: Symbol | Qty | Entry | LTP | P&L | %
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        # Frame
        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        inner = tk.Frame(self.frame, bg=theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        tk.Label(
            inner, text="📊 Open Positions",
            bg=theme['bg_card'],
            fg=theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Create treeview for table
        columns = ('symbol', 'qty', 'entry', 'ltp', 'pnl', 'pct')

        style = ttk.Style()
        style.configure(
            "Positions.Treeview",
            background=theme['bg_secondary'],
            foreground=theme['text_primary'],
            fieldbackground=theme['bg_secondary'],
            rowheight=30
        )
        style.configure(
            "Positions.Treeview.Heading",
            background=theme['bg_card'],
            foreground=theme['text_primary'],
            font=('Segoe UI', 10, 'bold')
        )

        self.tree = ttk.Treeview(
            inner,
            columns=columns,
            show='headings',
            height=5,
            style="Positions.Treeview"
        )

        # Define headings
        self.tree.heading('symbol', text='Symbol')
        self.tree.heading('qty', text='Qty')
        self.tree.heading('entry', text='Entry')
        self.tree.heading('ltp', text='LTP')
        self.tree.heading('pnl', text='P&L')
        self.tree.heading('pct', text='%')

        # Column widths
        self.tree.column('symbol', width=100)
        self.tree.column('qty', width=60, anchor=tk.E)
        self.tree.column('entry', width=80, anchor=tk.E)
        self.tree.column('ltp', width=80, anchor=tk.E)
        self.tree.column('pnl', width=80, anchor=tk.E)
        self.tree.column('pct', width=60, anchor=tk.E)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Empty message
        self.empty_label = tk.Label(
            inner, text="No open positions",
            bg=theme['bg_card'],
            fg=theme['text_dim'],
            font=('Segoe UI', 11)
        )

    def update_positions(self, positions: List[Dict]):
        """Update positions table"""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not positions:
            self.empty_label.pack(pady=20)
            return

        self.empty_label.pack_forget()

        for pos in positions:
            pnl = pos.get('pnl', 0)
            pct = pos.get('pnl_percent', 0)

            # Format values
            pnl_str = f"+₹{pnl:,.0f}" if pnl >= 0 else f"-₹{abs(pnl):,.0f}"
            pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"

            self.tree.insert('', tk.END, values=(
                pos.get('symbol', 'N/A'),
                pos.get('quantity', 0),
                f"₹{pos.get('entry_price', 0):,.2f}",
                f"₹{pos.get('ltp', 0):,.2f}",
                pnl_str,
                pct_str
            ))

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class ActivityFeed:
    """
    Live activity feed showing recent actions.

    Displays timestamped messages like:
    [10:30:15] 🟢 BUY RELIANCE @ ₹2,500
    """

    def __init__(self, parent: tk.Widget, theme: dict, max_items: int = 50):
        self.theme = theme
        self.max_items = max_items

        # Frame
        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        inner = tk.Frame(self.frame, bg=theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        tk.Label(
            inner, text="📝 Activity",
            bg=theme['bg_card'],
            fg=theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Text widget
        self.text = tk.Text(
            inner,
            height=8,
            bg=theme['bg_secondary'],
            fg=theme['text_primary'],
            font=('Consolas', 10),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for colors
        self.text.tag_configure('time', foreground=theme['text_dim'])
        self.text.tag_configure('success', foreground=theme['success'])
        self.text.tag_configure('danger', foreground=theme['danger'])
        self.text.tag_configure('warning', foreground=theme['warning'])
        self.text.tag_configure('info', foreground=theme['info'])

    def add_message(self, message: str, msg_type: str = 'info'):
        """Add a message to the feed"""
        self.text.config(state=tk.NORMAL)

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Insert timestamp
        self.text.insert(tk.END, f"[{timestamp}] ", 'time')

        # Insert message with appropriate color
        self.text.insert(tk.END, f"{message}\n", msg_type)

        # Trim old messages
        lines = int(self.text.index('end-1c').split('.')[0])
        if lines > self.max_items:
            self.text.delete('1.0', '2.0')

        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def clear(self):
        """Clear all messages"""
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)
        self.text.config(state=tk.DISABLED)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class Dashboard:
    """
    Complete trading dashboard.

    Combines all widgets into a unified view.
    """

    def __init__(self, parent: tk.Widget, theme_name: str = 'dark'):
        self.theme = get_theme(theme_name)
        self.parent = parent

        # Main container
        self.frame = tk.Frame(parent, bg=self.theme['bg_primary'])

        # Data
        self.balance = 0.0
        self.todays_pnl = 0.0
        self.total_pnl = 0.0
        self.open_positions = 0
        self.total_trades = 0

        self._create_widgets()

    def _create_widgets(self):
        """Build dashboard layout"""
        # Top row - Key metrics
        metrics_frame = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        metrics_frame.pack(fill=tk.X, pady=(0, 15))

        # Balance card
        self.balance_card = MetricCard(
            metrics_frame, self.theme,
            title="Balance", emoji="💰",
            value="₹0", subtitle="Available capital"
        )
        self.balance_card.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Today's P&L card
        self.pnl_card = MetricCard(
            metrics_frame, self.theme,
            title="Today's P&L", emoji="📈",
            value="₹0", subtitle="Realized + Unrealized"
        )
        self.pnl_card.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Win Rate card
        self.winrate_card = MetricCard(
            metrics_frame, self.theme,
            title="Win Rate", emoji="🎯",
            value="0%", subtitle="Winning trades"
        )
        self.winrate_card.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Middle row - Status and positions
        middle_frame = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Left side - Status cards
        status_frame = tk.Frame(middle_frame, bg=self.theme['bg_primary'])
        status_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # Bot Status
        bot_card = tk.Frame(status_frame, bg=self.theme['bg_card'])
        bot_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        bot_card.pack(fill=tk.X, pady=(0, 10))

        bot_inner = tk.Frame(bot_card, bg=self.theme['bg_card'])
        bot_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            bot_inner, text="🤖 Bot Status",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W)

        self.bot_status = StatusWidget(bot_inner, self.theme)
        self.bot_status.set_status('stopped', 'Bot Stopped')
        self.bot_status.pack(anchor=tk.W, pady=(10, 0))

        # Connection Status
        conn_card = tk.Frame(status_frame, bg=self.theme['bg_card'])
        conn_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        conn_card.pack(fill=tk.X, pady=(0, 10))

        conn_inner = tk.Frame(conn_card, bg=self.theme['bg_card'])
        conn_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            conn_inner, text="🔗 Connection",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W)

        self.conn_status = StatusWidget(conn_inner, self.theme)
        self.conn_status.set_status('disconnected', 'Not Connected')
        self.conn_status.pack(anchor=tk.W, pady=(10, 0))

        # Quick Stats
        stats_card = tk.Frame(status_frame, bg=self.theme['bg_card'])
        stats_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        stats_card.pack(fill=tk.X)

        stats_inner = tk.Frame(stats_card, bg=self.theme['bg_card'])
        stats_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            stats_inner, text="📊 Today's Stats",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Stats rows
        stats_data = [
            ("Trades", "0"),
            ("Open", "0"),
            ("Closed", "0"),
        ]

        self.stats_labels = {}
        for label, value in stats_data:
            row = tk.Frame(stats_inner, bg=self.theme['bg_card'])
            row.pack(fill=tk.X, pady=2)

            tk.Label(
                row, text=label,
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 10)
            ).pack(side=tk.LEFT)

            val_label = tk.Label(
                row, text=value,
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 10, 'bold')
            )
            val_label.pack(side=tk.RIGHT)
            self.stats_labels[label.lower()] = val_label

        # Right side - Positions table
        self.positions_table = PositionsTable(middle_frame, self.theme)
        self.positions_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom row - Activity feed
        self.activity_feed = ActivityFeed(self.frame, self.theme)
        self.activity_feed.pack(fill=tk.BOTH, expand=True)

        # Add welcome message
        self.activity_feed.add_message("Welcome to AlgoTrader Pro!", 'info')
        self.activity_feed.add_message("Login to start trading", 'warning')

    def update_balance(self, balance: float):
        """Update balance display"""
        self.balance = balance
        self.balance_card.update_value(f"₹{balance:,.0f}")

    def update_pnl(self, pnl: float):
        """Update P&L display"""
        self.todays_pnl = pnl
        color = self.theme['success'] if pnl >= 0 else self.theme['danger']
        sign = "+" if pnl >= 0 else ""
        self.pnl_card.update_value(f"{sign}₹{pnl:,.0f}", color)

    def update_winrate(self, rate: float):
        """Update win rate display"""
        color = self.theme['success'] if rate >= 50 else self.theme['danger']
        self.winrate_card.update_value(f"{rate:.1f}%", color)

    def update_bot_status(self, running: bool):
        """Update bot status"""
        if running:
            self.bot_status.set_status('running', 'Bot Running')
        else:
            self.bot_status.set_status('stopped', 'Bot Stopped')

    def update_connection(self, connected: bool):
        """Update connection status"""
        if connected:
            self.conn_status.set_status('connected', 'Connected')
        else:
            self.conn_status.set_status('disconnected', 'Not Connected')

    def update_stats(self, trades: int = 0, open_pos: int = 0, closed: int = 0):
        """Update quick stats"""
        self.stats_labels['trades'].config(text=str(trades))
        self.stats_labels['open'].config(text=str(open_pos))
        self.stats_labels['closed'].config(text=str(closed))

    def update_positions(self, positions: List[Dict]):
        """Update positions table"""
        self.positions_table.update_positions(positions)

    def log_activity(self, message: str, msg_type: str = 'info'):
        """Add message to activity feed"""
        self.activity_feed.add_message(message, msg_type)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("DASHBOARD - Test")
    print("=" * 50)

    # Create test window
    root = tk.Tk()
    root.title("Dashboard Test")
    root.geometry("1000x700")
    root.configure(bg='#1a1a2e')

    # Create dashboard
    dashboard = Dashboard(root, 'dark')
    dashboard.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Update with test data
    dashboard.update_balance(100000)
    dashboard.update_pnl(2500)
    dashboard.update_winrate(65.5)
    dashboard.update_connection(True)
    dashboard.update_bot_status(True)
    dashboard.update_stats(trades=15, open_pos=3, closed=12)

    # Add test positions
    test_positions = [
        {'symbol': 'RELIANCE', 'quantity': 10, 'entry_price': 2500, 'ltp': 2550, 'pnl': 500, 'pnl_percent': 2.0},
        {'symbol': 'TCS', 'quantity': 5, 'entry_price': 3500, 'ltp': 3450, 'pnl': -250, 'pnl_percent': -1.4},
        {'symbol': 'INFY', 'quantity': 20, 'entry_price': 1500, 'ltp': 1525, 'pnl': 500, 'pnl_percent': 1.7},
    ]
    dashboard.update_positions(test_positions)

    # Add activity messages
    dashboard.log_activity("Connected to Zerodha", 'success')
    dashboard.log_activity("Bot started with TURTLE strategy", 'info')
    dashboard.log_activity("BUY RELIANCE @ 2500", 'success')
    dashboard.log_activity("Stop loss triggered for TCS", 'danger')

    root.mainloop()

    print("\n" + "=" * 50)
    print("Dashboard ready!")
    print("=" * 50)
