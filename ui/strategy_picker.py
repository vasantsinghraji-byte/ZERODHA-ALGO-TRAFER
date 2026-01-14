# -*- coding: utf-8 -*-
"""
Strategy Picker - Choose Your Game Plan!
=========================================
Pick trading strategies like choosing a character in a game.

Features:
- Visual strategy cards
- Risk level indicators
- One-click selection
- Strategy details view
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Callable
import logging

from .themes import get_theme

logger = logging.getLogger(__name__)


# ============== STRATEGY DATA ==============

STRATEGY_INFO = {
    'turtle': {
        'name': 'TURTLE',
        'emoji': '🐢',
        'description': 'Moving Average Crossover',
        'risk_level': 'LOW',
        'risk_color': '#00ff88',
        'expected_return': '5-10% / month',
        'best_for': 'Beginners, long-term',
        'how_it_works': 'Buys when 20-day average crosses above 50-day average. '
                       'Like following a slow but steady turtle - wins the race!',
        'pros': ['Very safe', 'Easy to understand', 'Low stress'],
        'cons': ['Slower profits', 'May miss fast moves'],
    },
    'momentum': {
        'name': 'MOMENTUM',
        'emoji': '⚡',
        'description': 'RSI Strategy',
        'risk_level': 'MEDIUM',
        'risk_color': '#ffaa00',
        'expected_return': '10-20% / month',
        'best_for': 'Intermediate traders',
        'how_it_works': 'Buys when RSI shows oversold (below 30) and sells when '
                       'overbought (above 70). Catches stocks that moved too much!',
        'pros': ['Good returns', 'Clear signals', 'Well-tested'],
        'cons': ['Needs patience', 'Can be wrong sometimes'],
    },
    'supertrend': {
        'name': 'SUPERTREND',
        'emoji': '🚀',
        'description': 'Trend Following',
        'risk_level': 'HIGH',
        'risk_color': '#ff4444',
        'expected_return': '20-40% / month',
        'best_for': 'Experienced traders',
        'how_it_works': 'Follows the supertrend indicator. When price is above '
                       'the line, it\'s bullish. Below = bearish. Rides big trends!',
        'pros': ['Big profit potential', 'Catches major moves'],
        'cons': ['Higher risk', 'More volatile', 'Needs monitoring'],
    },
    'macd': {
        'name': 'MACD',
        'emoji': '🚦',
        'description': 'Traffic Light Signals',
        'risk_level': 'MEDIUM',
        'risk_color': '#ffaa00',
        'expected_return': '10-15% / month',
        'best_for': 'All traders',
        'how_it_works': 'Uses MACD crossover signals. When MACD line crosses above '
                       'signal line = BUY. Crosses below = SELL. Simple traffic lights!',
        'pros': ['Very popular', 'Clear signals', 'Works in trends'],
        'cons': ['Lagging indicator', 'Whipsaws in ranges'],
    },
    'bollinger': {
        'name': 'BOLLINGER',
        'emoji': '📊',
        'description': 'Mean Reversion',
        'risk_level': 'MEDIUM',
        'risk_color': '#ffaa00',
        'expected_return': '8-15% / month',
        'best_for': 'Range traders',
        'how_it_works': 'Buys when price touches lower band (cheap) and sells when '
                       'it touches upper band (expensive). Buy low, sell high!',
        'pros': ['Works in ranges', 'Clear entry/exit', 'Low drawdowns'],
        'cons': ['Fails in strong trends', 'Needs range markets'],
    },
    'breakout': {
        'name': 'BREAKOUT',
        'emoji': '💥',
        'description': 'Price Breakout',
        'risk_level': 'HIGH',
        'risk_color': '#ff4444',
        'expected_return': '15-30% / month',
        'best_for': 'Active traders',
        'how_it_works': 'Buys when price breaks above resistance. Catches stocks '
                       'that are about to make big moves!',
        'pros': ['Catches big moves', 'High reward potential'],
        'cons': ['False breakouts', 'Needs quick action'],
    },
    'vwap': {
        'name': 'VWAP',
        'emoji': '⚖️',
        'description': 'Fair Value Trading',
        'risk_level': 'LOW',
        'risk_color': '#00ff88',
        'expected_return': '5-12% / month',
        'best_for': 'Intraday traders',
        'how_it_works': 'VWAP is the average price weighted by volume. Buy below '
                       'VWAP (cheap), sell above VWAP (expensive). Find fair value!',
        'pros': ['Institutional favorite', 'Good for intraday'],
        'cons': ['Intraday only', 'Resets daily'],
    },
    'orb': {
        'name': 'ORB',
        'emoji': '🌅',
        'description': 'Opening Range Breakout',
        'risk_level': 'MEDIUM',
        'risk_color': '#ffaa00',
        'expected_return': '10-20% / month',
        'best_for': 'Morning traders',
        'how_it_works': 'Waits for first 15-30 minutes range, then trades the '
                       'breakout. Morning momentum is powerful!',
        'pros': ['Quick trades', 'Clear setup', 'Morning only'],
        'cons': ['Limited window', 'Needs early start'],
    },
    'multi': {
        'name': 'MULTI',
        'emoji': '🎯',
        'description': 'Smart Combination',
        'risk_level': 'LOW',
        'risk_color': '#00ff88',
        'expected_return': '8-15% / month',
        'best_for': 'Cautious traders',
        'how_it_works': 'Combines RSI + MACD + Moving Average. Only trades when '
                       'all indicators agree. Triple confirmation = high confidence!',
        'pros': ['Most reliable', 'Fewer false signals', 'Best accuracy'],
        'cons': ['Fewer trades', 'May miss some moves'],
    },
}


class StrategyCard:
    """
    Visual card for a single strategy.

    Shows strategy info in a clickable card format.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        strategy_key: str,
        on_select: Callable = None
    ):
        self.theme = theme
        self.strategy_key = strategy_key
        self.info = STRATEGY_INFO.get(strategy_key, {})
        self.on_select = on_select
        self.selected = False

        # Card frame
        self.frame = tk.Frame(
            parent,
            bg=theme['bg_card'],
            cursor='hand2'
        )
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=2
        )

        # Bind click
        self.frame.bind('<Button-1>', self._on_click)
        self.frame.bind('<Enter>', self._on_hover)
        self.frame.bind('<Leave>', self._on_leave)

        self._create_content()

    def _create_content(self):
        """Build card content"""
        inner = tk.Frame(self.frame, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Bind to inner frame too
        inner.bind('<Button-1>', self._on_click)

        # Header row with emoji and name
        header = tk.Frame(inner, bg=self.theme['bg_card'])
        header.pack(fill=tk.X)
        header.bind('<Button-1>', self._on_click)

        emoji_label = tk.Label(
            header,
            text=self.info.get('emoji', '📊'),
            bg=self.theme['bg_card'],
            font=('Segoe UI', 28)
        )
        emoji_label.pack(side=tk.LEFT)
        emoji_label.bind('<Button-1>', self._on_click)

        name_frame = tk.Frame(header, bg=self.theme['bg_card'])
        name_frame.pack(side=tk.LEFT, padx=(10, 0))
        name_frame.bind('<Button-1>', self._on_click)

        self.name_label = tk.Label(
            name_frame,
            text=self.info.get('name', 'UNKNOWN'),
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 16, 'bold')
        )
        self.name_label.pack(anchor=tk.W)
        self.name_label.bind('<Button-1>', self._on_click)

        desc_label = tk.Label(
            name_frame,
            text=self.info.get('description', ''),
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        )
        desc_label.pack(anchor=tk.W)
        desc_label.bind('<Button-1>', self._on_click)

        # Risk level badge
        risk_level = self.info.get('risk_level', 'MEDIUM')
        risk_color = self.info.get('risk_color', self.theme['warning'])

        risk_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        risk_frame.pack(fill=tk.X, pady=(10, 5))
        risk_frame.bind('<Button-1>', self._on_click)

        tk.Label(
            risk_frame,
            text="Risk:",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)

        risk_badge = tk.Label(
            risk_frame,
            text=f" {risk_level} ",
            bg=risk_color,
            fg='white' if risk_level != 'LOW' else 'black',
            font=('Segoe UI', 9, 'bold')
        )
        risk_badge.pack(side=tk.LEFT, padx=(5, 0))
        risk_badge.bind('<Button-1>', self._on_click)

        # Expected return
        return_label = tk.Label(
            inner,
            text=f"Expected: {self.info.get('expected_return', 'N/A')}",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 10, 'bold')
        )
        return_label.pack(anchor=tk.W)
        return_label.bind('<Button-1>', self._on_click)

        # Best for
        best_for_label = tk.Label(
            inner,
            text=f"Best for: {self.info.get('best_for', 'Everyone')}",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        )
        best_for_label.pack(anchor=tk.W, pady=(5, 0))
        best_for_label.bind('<Button-1>', self._on_click)

    def _on_click(self, event=None):
        """Handle card click"""
        if self.on_select:
            self.on_select(self.strategy_key)

    def _on_hover(self, event=None):
        """Handle mouse enter"""
        if not self.selected:
            self.frame.configure(highlightbackground=self.theme['accent'])

    def _on_leave(self, event=None):
        """Handle mouse leave"""
        if not self.selected:
            self.frame.configure(highlightbackground=self.theme['border'])

    def set_selected(self, selected: bool):
        """Set selection state"""
        self.selected = selected
        if selected:
            self.frame.configure(
                highlightbackground=self.theme['accent'],
                highlightthickness=3
            )
        else:
            self.frame.configure(
                highlightbackground=self.theme['border'],
                highlightthickness=2
            )

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class StrategyDetails:
    """
    Detailed view of a strategy.

    Shows full description, pros/cons, and settings.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme
        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        self.inner = tk.Frame(self.frame, bg=theme['bg_card'])
        self.inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self._show_empty()

    def _show_empty(self):
        """Show empty state"""
        self._clear()
        tk.Label(
            self.inner,
            text="👈 Select a strategy to see details",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 12)
        ).pack(expand=True)

    def _clear(self):
        """Clear current content"""
        for widget in self.inner.winfo_children():
            widget.destroy()

    def show_strategy(self, strategy_key: str):
        """Display strategy details"""
        self._clear()

        info = STRATEGY_INFO.get(strategy_key)
        if not info:
            self._show_empty()
            return

        # Header
        header = tk.Frame(self.inner, bg=self.theme['bg_card'])
        header.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            header,
            text=f"{info['emoji']} {info['name']}",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 20, 'bold')
        ).pack(side=tk.LEFT)

        # Risk badge
        risk_color = info.get('risk_color', self.theme['warning'])
        tk.Label(
            header,
            text=f" {info['risk_level']} RISK ",
            bg=risk_color,
            fg='white',
            font=('Segoe UI', 10, 'bold')
        ).pack(side=tk.RIGHT)

        # Description
        tk.Label(
            self.inner,
            text=info['description'],
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 12)
        ).pack(anchor=tk.W)

        # How it works
        tk.Label(
            self.inner,
            text="How It Works:",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(15, 5))

        how_text = tk.Text(
            self.inner,
            height=3,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        how_text.insert(tk.END, info['how_it_works'])
        how_text.config(state=tk.DISABLED)
        how_text.pack(fill=tk.X, pady=(0, 10))

        # Pros and Cons
        pros_cons_frame = tk.Frame(self.inner, bg=self.theme['bg_card'])
        pros_cons_frame.pack(fill=tk.X, pady=10)

        # Pros
        pros_frame = tk.Frame(pros_cons_frame, bg=self.theme['bg_card'])
        pros_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            pros_frame,
            text="✅ Pros",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W)

        for pro in info.get('pros', []):
            tk.Label(
                pros_frame,
                text=f"  • {pro}",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 10)
            ).pack(anchor=tk.W)

        # Cons
        cons_frame = tk.Frame(pros_cons_frame, bg=self.theme['bg_card'])
        cons_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            cons_frame,
            text="❌ Cons",
            bg=self.theme['bg_card'],
            fg=self.theme['danger'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W)

        for con in info.get('cons', []):
            tk.Label(
                cons_frame,
                text=f"  • {con}",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 10)
            ).pack(anchor=tk.W)

        # Expected return highlight
        return_frame = tk.Frame(self.inner, bg=self.theme['bg_secondary'])
        return_frame.pack(fill=tk.X, pady=(15, 0))

        tk.Label(
            return_frame,
            text=f"💰 Expected Return: {info['expected_return']}",
            bg=self.theme['bg_secondary'],
            fg=self.theme['success'],
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=10)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class StrategyPicker:
    """
    Complete strategy selection interface.

    Shows all strategies as cards with details panel.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_name: str = 'dark',
        on_strategy_selected: Callable = None
    ):
        self.theme = get_theme(theme_name)
        self.parent = parent
        self.on_strategy_selected = on_strategy_selected

        self.selected_strategy = None
        self.strategy_cards: Dict[str, StrategyCard] = {}

        self.frame = tk.Frame(parent, bg=self.theme['bg_primary'])
        self._create_widgets()

    def _create_widgets(self):
        """Build picker layout"""
        # Title
        title_frame = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            title_frame,
            text="🎯 Pick Your Strategy",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 18, 'bold')
        ).pack(side=tk.LEFT)

        tk.Label(
            title_frame,
            text="Choose wisely - this is your game plan!",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=(15, 0))

        # Main content
        content = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left side - Strategy cards grid
        cards_frame = tk.Frame(content, bg=self.theme['bg_primary'])
        cards_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create scrollable frame for cards
        canvas = tk.Canvas(cards_frame, bg=self.theme['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(cards_frame, orient=tk.VERTICAL, command=canvas.yview)

        scrollable = tk.Frame(canvas, bg=self.theme['bg_primary'])
        scrollable.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        canvas.create_window((0, 0), window=scrollable, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create strategy cards
        row = 0
        col = 0
        for key in STRATEGY_INFO.keys():
            card = StrategyCard(
                scrollable,
                self.theme,
                key,
                on_select=self._on_strategy_select
            )
            card.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')

            self.strategy_cards[key] = card

            col += 1
            if col >= 3:  # 3 cards per row
                col = 0
                row += 1

        # Configure grid weights
        for i in range(3):
            scrollable.columnconfigure(i, weight=1)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right side - Strategy details
        details_frame = tk.Frame(content, bg=self.theme['bg_primary'], width=350)
        details_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        details_frame.pack_propagate(False)

        self.details = StrategyDetails(details_frame, self.theme)
        self.details.pack(fill=tk.BOTH, expand=True)

        # Bottom - Confirm button
        bottom = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        bottom.pack(fill=tk.X, pady=(15, 0))

        self.confirm_btn = tk.Button(
            bottom,
            text="✅ USE THIS STRATEGY",
            bg=self.theme['btn_success'],
            fg='white',
            font=('Segoe UI', 14, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            command=self._on_confirm
        )
        self.confirm_btn.pack(side=tk.RIGHT, ipadx=20, ipady=10)

        self.selected_label = tk.Label(
            bottom,
            text="No strategy selected",
            bg=self.theme['bg_primary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 11)
        )
        self.selected_label.pack(side=tk.LEFT)

    def _on_strategy_select(self, strategy_key: str):
        """Handle strategy selection"""
        # Deselect previous
        if self.selected_strategy and self.selected_strategy in self.strategy_cards:
            self.strategy_cards[self.selected_strategy].set_selected(False)

        # Select new
        self.selected_strategy = strategy_key
        self.strategy_cards[strategy_key].set_selected(True)

        # Update details
        self.details.show_strategy(strategy_key)

        # Update bottom bar
        info = STRATEGY_INFO.get(strategy_key, {})
        self.selected_label.config(
            text=f"Selected: {info.get('emoji', '')} {info.get('name', '')}",
            fg=self.theme['text_primary']
        )
        self.confirm_btn.config(state=tk.NORMAL)

    def _on_confirm(self):
        """Handle confirm button click"""
        if self.selected_strategy and self.on_strategy_selected:
            self.on_strategy_selected(self.selected_strategy)
            messagebox.showinfo(
                "Strategy Selected",
                f"You selected: {STRATEGY_INFO[self.selected_strategy]['name']}\n\n"
                f"The bot will use this strategy for trading!"
            )

    def get_selected(self) -> Optional[str]:
        """Get currently selected strategy"""
        return self.selected_strategy

    def set_selected(self, strategy_key: str):
        """Programmatically select a strategy"""
        if strategy_key in STRATEGY_INFO:
            self._on_strategy_select(strategy_key)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("STRATEGY PICKER - Test")
    print("=" * 50)

    def on_select(strategy):
        print(f"Strategy selected: {strategy}")

    # Create test window
    root = tk.Tk()
    root.title("Strategy Picker Test")
    root.geometry("1100x700")
    root.configure(bg='#1a1a2e')

    picker = StrategyPicker(root, 'dark', on_strategy_selected=on_select)
    picker.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    root.mainloop()

    print("\n" + "=" * 50)
    print("Strategy Picker ready!")
    print("=" * 50)
