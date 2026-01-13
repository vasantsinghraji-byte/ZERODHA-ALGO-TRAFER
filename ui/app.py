"""
AlgoTrader Pro - Main Application
A beautiful trading app that anyone can use!

Features:
- Big colorful buttons
- Simple language
- One-click trading
- Real-time updates
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import webbrowser
from datetime import datetime
from typing import Optional
import logging

from .themes import get_theme

logger = logging.getLogger(__name__)


class AlgoTraderApp:
    """
    Main Trading Application

    Simple enough for a 5th grader, powerful enough for professionals!
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 AlgoTrader Pro")

        # Get screen size and set window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1200, int(screen_width * 0.8))
        window_height = min(800, int(screen_height * 0.8))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Theme
        self.theme = get_theme('dark')
        self.root.configure(bg=self.theme['bg_primary'])

        # State
        self.connected = False
        self.bot_running = False
        self.paper_trading = True
        self.balance = 100000.0
        self.todays_pnl = 0.0

        # Broker (will be initialized on login)
        self.broker = None

        # Build UI
        self._create_styles()
        self._create_ui()

    def _create_styles(self):
        """Create ttk styles"""
        style = ttk.Style()

        # Configure colors
        style.configure('Dark.TFrame', background=self.theme['bg_primary'])
        style.configure('Card.TFrame', background=self.theme['bg_card'])

        style.configure('Title.TLabel',
                       background=self.theme['bg_primary'],
                       foreground=self.theme['text_primary'],
                       font=('Segoe UI', 24, 'bold'))

        style.configure('Subtitle.TLabel',
                       background=self.theme['bg_primary'],
                       foreground=self.theme['text_secondary'],
                       font=('Segoe UI', 12))

        style.configure('Card.TLabel',
                       background=self.theme['bg_card'],
                       foreground=self.theme['text_primary'],
                       font=('Segoe UI', 11))

        style.configure('CardTitle.TLabel',
                       background=self.theme['bg_card'],
                       foreground=self.theme['text_primary'],
                       font=('Segoe UI', 14, 'bold'))

        style.configure('Success.TLabel',
                       background=self.theme['bg_card'],
                       foreground=self.theme['success'],
                       font=('Segoe UI', 16, 'bold'))

        style.configure('Danger.TLabel',
                       background=self.theme['bg_card'],
                       foreground=self.theme['danger'],
                       font=('Segoe UI', 16, 'bold'))

    def _create_ui(self):
        """Build the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.theme['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        self._create_header(main_frame)

        # Content area
        content_frame = tk.Frame(main_frame, bg=self.theme['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # Left panel - Status & Controls
        left_panel = tk.Frame(content_frame, bg=self.theme['bg_primary'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._create_status_card(left_panel)
        self._create_controls_card(left_panel)

        # Right panel - Strategies & Activity
        right_panel = tk.Frame(content_frame, bg=self.theme['bg_primary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_strategies_card(right_panel)
        self._create_activity_card(right_panel)

        # Footer
        self._create_footer(main_frame)

    def _create_header(self, parent):
        """Create header with title and connection status"""
        header = tk.Frame(parent, bg=self.theme['bg_primary'])
        header.pack(fill=tk.X, pady=(0, 10))

        # Title
        title = tk.Label(header, text="🚀 AlgoTrader Pro",
                        bg=self.theme['bg_primary'],
                        fg=self.theme['text_primary'],
                        font=('Segoe UI', 28, 'bold'))
        title.pack(side=tk.LEFT)

        # Connection status
        self.status_label = tk.Label(header, text="⚫ Not Connected",
                                    bg=self.theme['bg_primary'],
                                    fg=self.theme['text_secondary'],
                                    font=('Segoe UI', 12))
        self.status_label.pack(side=tk.RIGHT)

    def _create_status_card(self, parent):
        """Create status card with balance and P&L"""
        card = tk.Frame(parent, bg=self.theme['bg_card'], relief=tk.FLAT)
        card.pack(fill=tk.X, pady=(0, 10))
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=20, pady=20)

        # Title
        tk.Label(inner, text="📊 Your Account",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)

        # Balance
        balance_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        balance_frame.pack(fill=tk.X, pady=(15, 5))

        tk.Label(balance_frame, text="💰 Balance:",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 12)).pack(side=tk.LEFT)

        self.balance_label = tk.Label(balance_frame, text=f"₹{self.balance:,.2f}",
                                     bg=self.theme['bg_card'],
                                     fg=self.theme['text_primary'],
                                     font=('Segoe UI', 18, 'bold'))
        self.balance_label.pack(side=tk.RIGHT)

        # Today's P&L
        pnl_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        pnl_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(pnl_frame, text="📈 Today's P&L:",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 12)).pack(side=tk.LEFT)

        pnl_color = self.theme['success'] if self.todays_pnl >= 0 else self.theme['danger']
        pnl_sign = "+" if self.todays_pnl >= 0 else ""
        self.pnl_label = tk.Label(pnl_frame, text=f"{pnl_sign}₹{self.todays_pnl:,.2f}",
                                 bg=self.theme['bg_card'],
                                 fg=pnl_color,
                                 font=('Segoe UI', 18, 'bold'))
        self.pnl_label.pack(side=tk.RIGHT)

        # Mode indicator
        mode_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        mode_frame.pack(fill=tk.X, pady=(15, 0))

        mode_text = "🎮 Paper Trading (Practice Mode)" if self.paper_trading else "💵 Live Trading (Real Money)"
        mode_color = self.theme['info'] if self.paper_trading else self.theme['warning']
        tk.Label(mode_frame, text=mode_text,
                bg=self.theme['bg_card'],
                fg=mode_color,
                font=('Segoe UI', 11, 'bold')).pack(anchor=tk.CENTER)

    def _create_controls_card(self, parent):
        """Create control buttons"""
        card = tk.Frame(parent, bg=self.theme['bg_card'], relief=tk.FLAT)
        card.pack(fill=tk.X, pady=(0, 10))
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=20, pady=20)

        # Title
        tk.Label(inner, text="🎮 Controls",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W, pady=(0, 15))

        # Login Button
        self.login_btn = tk.Button(inner, text="🔐 LOGIN TO ZERODHA",
                                  bg=self.theme['btn_primary'],
                                  fg='white',
                                  font=('Segoe UI', 14, 'bold'),
                                  relief=tk.FLAT,
                                  cursor='hand2',
                                  command=self._on_login)
        self.login_btn.pack(fill=tk.X, pady=(0, 10), ipady=10)

        # Start/Stop Bot Button
        self.bot_btn = tk.Button(inner, text="▶️ START BOT",
                                bg=self.theme['btn_success'],
                                fg='white',
                                font=('Segoe UI', 14, 'bold'),
                                relief=tk.FLAT,
                                cursor='hand2',
                                state=tk.DISABLED,
                                command=self._on_toggle_bot)
        self.bot_btn.pack(fill=tk.X, pady=(0, 10), ipady=10)

        # Settings Button
        settings_btn = tk.Button(inner, text="⚙️ SETTINGS",
                                bg=self.theme['bg_secondary'],
                                fg=self.theme['text_primary'],
                                font=('Segoe UI', 12),
                                relief=tk.FLAT,
                                cursor='hand2',
                                command=self._on_settings)
        settings_btn.pack(fill=tk.X, ipady=8)

    def _create_strategies_card(self, parent):
        """Create strategy selection card"""
        card = tk.Frame(parent, bg=self.theme['bg_card'], relief=tk.FLAT)
        card.pack(fill=tk.X, pady=(0, 10))
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=20, pady=20)

        # Title
        tk.Label(inner, text="🎯 Pick Your Strategy",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W, pady=(0, 15))

        # Strategy options
        strategies = [
            ("🐢 TURTLE (Safe)", "Moving Average - Low Risk", self.theme['success']),
            ("⚡ MOMENTUM (Medium)", "RSI Strategy - Medium Risk", self.theme['warning']),
            ("🚀 ROCKET (Aggressive)", "Supertrend - High Risk", self.theme['danger']),
        ]

        self.selected_strategy = tk.StringVar(value="turtle")

        for emoji_name, desc, color in strategies:
            strategy_frame = tk.Frame(inner, bg=self.theme['bg_card'])
            strategy_frame.pack(fill=tk.X, pady=5)

            rb = tk.Radiobutton(strategy_frame, text=emoji_name,
                               variable=self.selected_strategy,
                               value=emoji_name.split()[0].lower(),
                               bg=self.theme['bg_card'],
                               fg=color,
                               selectcolor=self.theme['bg_secondary'],
                               activebackground=self.theme['bg_card'],
                               font=('Segoe UI', 12, 'bold'),
                               cursor='hand2')
            rb.pack(side=tk.LEFT)

            tk.Label(strategy_frame, text=desc,
                    bg=self.theme['bg_card'],
                    fg=self.theme['text_secondary'],
                    font=('Segoe UI', 10)).pack(side=tk.RIGHT)

    def _create_activity_card(self, parent):
        """Create activity log card"""
        card = tk.Frame(parent, bg=self.theme['bg_card'], relief=tk.FLAT)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        tk.Label(inner, text="📝 Activity Log",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        # Activity text
        self.activity_text = tk.Text(inner, height=10, width=40,
                                    bg=self.theme['bg_secondary'],
                                    fg=self.theme['text_primary'],
                                    font=('Consolas', 10),
                                    relief=tk.FLAT,
                                    state=tk.DISABLED)
        self.activity_text.pack(fill=tk.BOTH, expand=True)

        # Add welcome message
        self._log_activity("👋 Welcome to AlgoTrader Pro!")
        self._log_activity("🔐 Please login to start trading.")

    def _create_footer(self, parent):
        """Create footer with status"""
        footer = tk.Frame(parent, bg=self.theme['bg_primary'])
        footer.pack(fill=tk.X, pady=(10, 0))

        # Time
        self.time_label = tk.Label(footer, text="",
                                  bg=self.theme['bg_primary'],
                                  fg=self.theme['text_dim'],
                                  font=('Segoe UI', 10))
        self.time_label.pack(side=tk.LEFT)

        # Version
        tk.Label(footer, text="AlgoTrader Pro v2.0 | Made with ❤️",
                bg=self.theme['bg_primary'],
                fg=self.theme['text_dim'],
                font=('Segoe UI', 10)).pack(side=tk.RIGHT)

        # Update time
        self._update_time()

    def _update_time(self):
        """Update time display"""
        now = datetime.now().strftime("%H:%M:%S | %d %b %Y")
        self.time_label.config(text=now)
        self.root.after(1000, self._update_time)

    def _log_activity(self, message: str):
        """Add message to activity log"""
        self.activity_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.activity_text.see(tk.END)
        self.activity_text.config(state=tk.DISABLED)

    def _on_login(self):
        """Handle login button click"""
        self._log_activity("🔐 Opening Zerodha login...")

        # Show API key dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("🔐 Login to Zerodha")
        dialog.geometry("400x300")
        dialog.configure(bg=self.theme['bg_card'])
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (self.root.winfo_width() - 400) // 2 + self.root.winfo_x()
        y = (self.root.winfo_height() - 300) // 2 + self.root.winfo_y()
        dialog.geometry(f"+{x}+{y}")

        # Content
        tk.Label(dialog, text="🔐 Enter Your API Credentials",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 14, 'bold')).pack(pady=20)

        # API Key
        tk.Label(dialog, text="API Key:",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary']).pack(anchor=tk.W, padx=30)
        api_key_entry = tk.Entry(dialog, width=40, font=('Segoe UI', 11))
        api_key_entry.pack(padx=30, pady=(5, 15))

        # API Secret
        tk.Label(dialog, text="API Secret:",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary']).pack(anchor=tk.W, padx=30)
        api_secret_entry = tk.Entry(dialog, width=40, font=('Segoe UI', 11), show="*")
        api_secret_entry.pack(padx=30, pady=(5, 20))

        def do_login():
            api_key = api_key_entry.get().strip()
            api_secret = api_secret_entry.get().strip()

            if not api_key or not api_secret:
                messagebox.showerror("Error", "Please enter both API Key and Secret!")
                return

            # Simulate login for demo
            self._log_activity("✅ Connected to Zerodha!")
            self.connected = True
            self.status_label.config(text="🟢 Connected", fg=self.theme['success'])
            self.login_btn.config(text="✅ CONNECTED", state=tk.DISABLED,
                                 bg=self.theme['success'])
            self.bot_btn.config(state=tk.NORMAL)
            dialog.destroy()

        tk.Button(dialog, text="🚀 LOGIN",
                 bg=self.theme['btn_primary'],
                 fg='white',
                 font=('Segoe UI', 12, 'bold'),
                 relief=tk.FLAT,
                 cursor='hand2',
                 command=do_login).pack(pady=10, ipadx=20, ipady=5)

    def _on_toggle_bot(self):
        """Toggle bot on/off"""
        if not self.bot_running:
            self.bot_running = True
            self.bot_btn.config(text="⏹️ STOP BOT", bg=self.theme['btn_danger'])
            self._log_activity("🤖 Bot started!")
            self._log_activity(f"📊 Using strategy: {self.selected_strategy.get()}")
        else:
            self.bot_running = False
            self.bot_btn.config(text="▶️ START BOT", bg=self.theme['btn_success'])
            self._log_activity("🛑 Bot stopped.")

    def _on_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "Settings panel coming soon!\n\nFor now, edit config/settings.yaml")

    def run(self):
        """Start the application"""
        logger.info("Starting AlgoTrader Pro...")
        self.root.mainloop()


def main():
    """Entry point"""
    app = AlgoTraderApp()
    app.run()


if __name__ == "__main__":
    main()
