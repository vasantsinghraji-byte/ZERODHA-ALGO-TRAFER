# -*- coding: utf-8 -*-
"""
Stock Search Widget - Find Any Stock Instantly!
================================================
Type to search, click to select. Simple as that.

Features:
- Real-time search as you type
- Shows matching stocks from your watchlist
- Supports adding custom symbols
- Recent searches for quick access
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StockSearchWidget:
    """
    Autocomplete search widget for finding stocks.

    Usage:
        search = StockSearchWidget(parent, theme, on_select=my_callback)
        search.pack(fill=tk.X)

        # Get selected symbol
        symbol = search.get_selected()
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        on_select: Callable[[str], None] = None,
        placeholder: str = "Search stocks...",
        show_exchange: bool = True,
        width: int = 25
    ):
        self.parent = parent
        self.theme = theme
        self.on_select = on_select
        self.show_exchange = show_exchange
        self.placeholder = placeholder

        # Stock data
        self._all_symbols: List[str] = []
        self._symbol_info: Dict[str, Dict[str, Any]] = {}  # Extra info like name, sector
        self._recent_searches: List[str] = []
        self._selected_symbol: str = ""

        # UI state
        self._dropdown_visible = False

        # Build widget
        self.frame = tk.Frame(parent, bg=theme['bg_primary'])
        self._create_widgets(width)
        self._load_symbols()

    def _create_widgets(self, width: int):
        """Build the search UI"""
        # Search entry with icon
        entry_frame = tk.Frame(self.frame, bg=self.theme['bg_secondary'])
        entry_frame.pack(fill=tk.X)

        # Search icon
        tk.Label(
            entry_frame,
            text="🔍",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Entry field
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            entry_frame,
            textvariable=self.entry_var,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            width=width,
            insertbackground=self.theme['text_primary']
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=8)

        # Clear button (hidden initially)
        self.clear_btn = tk.Label(
            entry_frame,
            text="✕",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10),
            cursor='hand2'
        )
        self.clear_btn.bind('<Button-1>', self._clear_search)

        # Dropdown for results
        self.dropdown_frame = tk.Frame(
            self.frame,
            bg=self.theme['bg_card'],
            highlightbackground=self.theme['border'],
            highlightthickness=1
        )

        # Listbox for results
        self.listbox = tk.Listbox(
            self.dropdown_frame,
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            relief=tk.FLAT,
            selectbackground=self.theme['accent'],
            selectforeground='white',
            height=8,
            activestyle='none',
            cursor='hand2'
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.dropdown_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Bindings
        self.entry.bind('<KeyRelease>', self._on_key_release)
        self.entry.bind('<FocusIn>', self._on_focus_in)
        self.entry.bind('<FocusOut>', self._on_focus_out)
        self.entry.bind('<Return>', self._on_enter)
        self.entry.bind('<Down>', self._on_arrow_down)
        self.entry.bind('<Up>', self._on_arrow_up)
        self.entry.bind('<Escape>', self._hide_dropdown)

        self.listbox.bind('<ButtonRelease-1>', self._on_listbox_click)
        self.listbox.bind('<Return>', self._on_listbox_enter)

        # Set placeholder
        self._show_placeholder()

    def _load_symbols(self):
        """Load symbols from watchlist configuration"""
        try:
            from config.loader import get_watchlist

            watchlist = get_watchlist()
            all_symbols = set()

            # Collect from all watchlists
            for key in ['nifty50', 'banknifty', 'custom', 'fno']:
                symbols = watchlist.get(key, [])
                if symbols:
                    all_symbols.update(symbols)

            self._all_symbols = sorted(list(all_symbols))
            logger.info(f"Loaded {len(self._all_symbols)} symbols for search")

        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")
            # Fallback to common symbols
            self._all_symbols = [
                'NSE:RELIANCE', 'NSE:TCS', 'NSE:INFY', 'NSE:HDFCBANK',
                'NSE:ICICIBANK', 'NSE:SBIN', 'NSE:BHARTIARTL', 'NSE:ITC',
                'NSE:KOTAKBANK', 'NSE:LT', 'NSE:AXISBANK', 'NSE:HDFC',
                'NSE:MARUTI', 'NSE:ASIANPAINT', 'NSE:HCLTECH', 'NSE:WIPRO'
            ]

    def _show_placeholder(self):
        """Show placeholder text"""
        if not self.entry_var.get():
            self.entry.config(fg=self.theme['text_dim'])
            self.entry_var.set(self.placeholder)

    def _hide_placeholder(self):
        """Hide placeholder text"""
        if self.entry_var.get() == self.placeholder:
            self.entry_var.set('')
            self.entry.config(fg=self.theme['text_primary'])

    def _on_focus_in(self, event=None):
        """Handle focus in"""
        self._hide_placeholder()
        self._show_dropdown()

    def _on_focus_out(self, event=None):
        """Handle focus out - delay to allow click on dropdown"""
        self.frame.after(200, self._check_focus_out)

    def _check_focus_out(self):
        """Check if we should hide dropdown"""
        focused = self.frame.focus_get()
        if focused not in (self.entry, self.listbox):
            self._hide_dropdown()
            self._show_placeholder()

    def _on_key_release(self, event=None):
        """Handle typing in search box"""
        if event and event.keysym in ('Up', 'Down', 'Return', 'Escape'):
            return

        query = self.entry_var.get().strip()
        if query == self.placeholder:
            query = ''

        # Show/hide clear button
        if query:
            self.clear_btn.pack(side=tk.RIGHT, padx=(0, 8))
        else:
            self.clear_btn.pack_forget()

        self._filter_symbols(query)
        self._show_dropdown()

    def _filter_symbols(self, query: str):
        """Filter symbols based on search query"""
        self.listbox.delete(0, tk.END)

        if not query:
            # Show recent searches or popular stocks
            if self._recent_searches:
                self.listbox.insert(tk.END, "── Recent ──")
                for sym in self._recent_searches[:5]:
                    self.listbox.insert(tk.END, f"  {sym}")

            self.listbox.insert(tk.END, "── All Stocks ──")
            for sym in self._all_symbols[:15]:
                display = self._format_symbol(sym)
                self.listbox.insert(tk.END, f"  {display}")

            if len(self._all_symbols) > 15:
                self.listbox.insert(tk.END, f"  ... and {len(self._all_symbols) - 15} more")
            return

        query_upper = query.upper()
        matches = []

        for sym in self._all_symbols:
            # Extract just the symbol name (after exchange:)
            if ':' in sym:
                _, name = sym.split(':', 1)
            else:
                name = sym

            # Match against symbol name or full string
            if query_upper in name.upper() or query_upper in sym.upper():
                matches.append(sym)

        # Also allow custom entry
        if not matches:
            self.listbox.insert(tk.END, f"  Add custom: {query.upper()}")
        else:
            for sym in matches[:20]:
                display = self._format_symbol(sym)
                self.listbox.insert(tk.END, f"  {display}")

            if len(matches) > 20:
                self.listbox.insert(tk.END, f"  ... {len(matches) - 20} more matches")

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for display"""
        if self.show_exchange:
            return symbol
        else:
            if ':' in symbol:
                _, name = symbol.split(':', 1)
                return name
            return symbol

    def _show_dropdown(self, event=None):
        """Show the dropdown list"""
        if not self._dropdown_visible:
            self.dropdown_frame.pack(fill=tk.X, pady=(2, 0))
            self._dropdown_visible = True

    def _hide_dropdown(self, event=None):
        """Hide the dropdown list"""
        if self._dropdown_visible:
            self.dropdown_frame.pack_forget()
            self._dropdown_visible = False

    def _on_arrow_down(self, event=None):
        """Move selection down in listbox"""
        self._show_dropdown()
        current = self.listbox.curselection()
        if current:
            next_idx = min(current[0] + 1, self.listbox.size() - 1)
        else:
            next_idx = 0

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(next_idx)
        self.listbox.see(next_idx)
        return 'break'

    def _on_arrow_up(self, event=None):
        """Move selection up in listbox"""
        current = self.listbox.curselection()
        if current:
            prev_idx = max(current[0] - 1, 0)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(prev_idx)
            self.listbox.see(prev_idx)
        return 'break'

    def _on_enter(self, event=None):
        """Handle Enter key"""
        current = self.listbox.curselection()
        if current:
            self._select_from_listbox(current[0])
        else:
            # Use what's typed as custom symbol
            query = self.entry_var.get().strip().upper()
            if query and query != self.placeholder.upper():
                self._select_symbol(f"NSE:{query}" if ':' not in query else query)

    def _on_listbox_click(self, event=None):
        """Handle click on listbox item"""
        current = self.listbox.curselection()
        if current:
            self._select_from_listbox(current[0])

    def _on_listbox_enter(self, event=None):
        """Handle Enter in listbox"""
        current = self.listbox.curselection()
        if current:
            self._select_from_listbox(current[0])

    def _select_from_listbox(self, index: int):
        """Select item from listbox"""
        item = self.listbox.get(index).strip()

        # Skip headers
        if item.startswith('──') or item.startswith('...'):
            return

        # Handle custom entry
        if item.startswith('Add custom:'):
            custom = item.replace('Add custom:', '').strip()
            symbol = f"NSE:{custom}" if ':' not in custom else custom
        else:
            # Extract symbol from display
            symbol = item.strip()
            if not self.show_exchange and ':' not in symbol:
                # Find full symbol
                for sym in self._all_symbols:
                    if sym.endswith(f":{symbol}") or sym == symbol:
                        symbol = sym
                        break

        self._select_symbol(symbol)

    def _select_symbol(self, symbol: str, trigger_callback: bool = True):
        """Select a symbol.

        Args:
            symbol: The symbol to select
            trigger_callback: If True, calls the on_select callback
        """
        self._selected_symbol = symbol

        # Update entry
        display = self._format_symbol(symbol)
        self.entry_var.set(display)
        self.entry.config(fg=self.theme['text_primary'])

        # Add to recent
        if symbol in self._recent_searches:
            self._recent_searches.remove(symbol)
        self._recent_searches.insert(0, symbol)
        self._recent_searches = self._recent_searches[:10]

        # Hide dropdown
        self._hide_dropdown()

        # Callback (only if requested)
        if trigger_callback and self.on_select:
            self.on_select(symbol)

        logger.debug(f"Selected symbol: {symbol}")

    def _clear_search(self, event=None):
        """Clear search box"""
        self.entry_var.set('')
        self._selected_symbol = ''
        self.clear_btn.pack_forget()
        self._filter_symbols('')
        self.entry.focus_set()

    # Public API

    def get_selected(self) -> str:
        """Get the currently selected symbol"""
        return self._selected_symbol

    def set_selected(self, symbol: str, trigger_callback: bool = True):
        """Set the selected symbol.

        Args:
            symbol: The symbol to select
            trigger_callback: If False, won't call on_select callback (useful during init)
        """
        self._select_symbol(symbol, trigger_callback=trigger_callback)

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        return self._all_symbols.copy()

    def add_symbol(self, symbol: str):
        """Add a new symbol to the list"""
        if symbol not in self._all_symbols:
            self._all_symbols.append(symbol)
            self._all_symbols.sort()

    def set_symbols(self, symbols: List[str]):
        """Set the list of available symbols"""
        self._all_symbols = sorted(symbols)

    def refresh_symbols(self):
        """Reload symbols from config"""
        self._load_symbols()

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class QuickStockSelector:
    """
    Quick stock selector with preset buttons + search.

    Shows popular stocks as quick-select buttons plus a search box.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        on_select: Callable[[str], None] = None,
        quick_picks: List[str] = None
    ):
        self.parent = parent
        self.theme = theme
        self.on_select = on_select

        self.frame = tk.Frame(parent, bg=theme['bg_primary'])

        # Quick pick buttons
        quick_frame = tk.Frame(self.frame, bg=theme['bg_primary'])
        quick_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            quick_frame,
            text="Quick Select:",
            bg=theme['bg_primary'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Default quick picks
        if not quick_picks:
            quick_picks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'SBIN']

        for symbol in quick_picks:
            btn = tk.Button(
                quick_frame,
                text=symbol,
                bg=theme['bg_secondary'],
                fg=theme['text_primary'],
                font=('Segoe UI', 9),
                relief=tk.FLAT,
                cursor='hand2',
                command=lambda s=symbol: self._quick_select(s)
            )
            btn.pack(side=tk.LEFT, padx=2, ipadx=8, ipady=2)

        # Search widget
        search_frame = tk.Frame(self.frame, bg=theme['bg_primary'])
        search_frame.pack(fill=tk.X)

        tk.Label(
            search_frame,
            text="Or search:",
            bg=theme['bg_primary'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.search = StockSearchWidget(
            search_frame,
            theme,
            on_select=self._on_search_select,
            width=20
        )
        self.search.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _quick_select(self, symbol: str):
        """Handle quick pick selection"""
        full_symbol = f"NSE:{symbol}"
        self.search.set_selected(full_symbol)
        if self.on_select:
            self.on_select(full_symbol)

    def _on_search_select(self, symbol: str):
        """Handle search selection"""
        if self.on_select:
            self.on_select(symbol)

    def get_selected(self) -> str:
        return self.search.get_selected()

    def set_selected(self, symbol: str):
        self.search.set_selected(symbol)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    from themes import get_theme

    root = tk.Tk()
    root.title("Stock Search Test")
    root.geometry("500x400")

    theme = get_theme('dark')
    root.configure(bg=theme['bg_primary'])

    def on_select(symbol):
        print(f"Selected: {symbol}")
        result_label.config(text=f"Selected: {symbol}")

    # Test StockSearchWidget
    tk.Label(
        root, text="Stock Search Widget",
        bg=theme['bg_primary'], fg=theme['text_primary'],
        font=('Segoe UI', 14, 'bold')
    ).pack(pady=20)

    search = StockSearchWidget(root, theme, on_select=on_select)
    search.pack(fill=tk.X, padx=20)

    # Test QuickStockSelector
    tk.Label(
        root, text="Quick Stock Selector",
        bg=theme['bg_primary'], fg=theme['text_primary'],
        font=('Segoe UI', 14, 'bold')
    ).pack(pady=(30, 10))

    quick = QuickStockSelector(root, theme, on_select=on_select)
    quick.pack(fill=tk.X, padx=20)

    # Result display
    result_label = tk.Label(
        root, text="No selection",
        bg=theme['bg_primary'], fg=theme['text_secondary'],
        font=('Segoe UI', 12)
    )
    result_label.pack(pady=30)

    root.mainloop()
