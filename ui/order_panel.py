# -*- coding: utf-8 -*-
"""
Order Panel - Place Different Types of Orders!
===============================================
A comprehensive order placement panel that supports:
- Basic Orders: Market, Limit, SL, SL-M
- Advanced Orders: Iceberg, TWAP, VWAP, Bracket

Integrates with the OrderManager and execution modules.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Dict, Any
import logging
import threading

from .themes import get_theme
from .stock_search import StockSearchWidget

logger = logging.getLogger(__name__)


class OrderTypeSelector:
    """
    Selector for order type with visual tabs.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        on_select: Optional[Callable[[str], None]] = None
    ):
        self.theme = theme
        self.on_select = on_select
        self.selected = "MARKET"

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Basic order types
        basic_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        basic_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            basic_frame, text="Basic Orders",
            bg=theme['bg_card'], fg=theme['text_dim'],
            font=('Segoe UI', 9)
        ).pack(anchor=tk.W, pady=(0, 5))

        basic_types = tk.Frame(basic_frame, bg=theme['bg_card'])
        basic_types.pack(fill=tk.X)

        self.buttons: Dict[str, tk.Button] = {}

        basic_order_types = [
            ("MARKET", "Market"),
            ("LIMIT", "Limit"),
            ("SL", "Stop Loss"),
            ("SL-M", "SL Market"),
        ]

        for order_type, label in basic_order_types:
            btn = tk.Button(
                basic_types, text=label,
                bg=theme['bg_secondary'],
                fg=theme['text_secondary'],
                font=('Segoe UI', 10),
                relief=tk.FLAT,
                cursor='hand2',
                command=lambda t=order_type: self._select(t)
            )
            btn.pack(side=tk.LEFT, padx=(0, 5), ipadx=10, ipady=5)
            self.buttons[order_type] = btn

        # Advanced order types
        advanced_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        advanced_frame.pack(fill=tk.X)

        tk.Label(
            advanced_frame, text="Advanced Orders",
            bg=theme['bg_card'], fg=theme['text_dim'],
            font=('Segoe UI', 9)
        ).pack(anchor=tk.W, pady=(0, 5))

        advanced_types = tk.Frame(advanced_frame, bg=theme['bg_card'])
        advanced_types.pack(fill=tk.X)

        advanced_order_types = [
            ("BRACKET", "Bracket"),
            ("ICEBERG", "Iceberg"),
            ("TWAP", "TWAP"),
            ("VWAP", "VWAP"),
        ]

        for order_type, label in advanced_order_types:
            btn = tk.Button(
                advanced_types, text=label,
                bg=theme['bg_secondary'],
                fg=theme['text_secondary'],
                font=('Segoe UI', 10),
                relief=tk.FLAT,
                cursor='hand2',
                command=lambda t=order_type: self._select(t)
            )
            btn.pack(side=tk.LEFT, padx=(0, 5), ipadx=10, ipady=5)
            self.buttons[order_type] = btn

        # Highlight default selection
        self._update_visual()

    def _select(self, order_type: str):
        """Handle order type selection"""
        self.selected = order_type
        self._update_visual()
        if self.on_select:
            self.on_select(order_type)

    def _update_visual(self):
        """Update button visuals based on selection"""
        for otype, btn in self.buttons.items():
            if otype == self.selected:
                btn.config(
                    bg=self.theme['accent'],
                    fg='white'
                )
            else:
                btn.config(
                    bg=self.theme['bg_secondary'],
                    fg=self.theme['text_secondary']
                )

    def get_selected(self) -> str:
        return self.selected

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class OrderForm:
    """
    Dynamic order form that changes based on order type.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        on_submit: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.theme = theme
        self.on_submit = on_submit
        self.order_type = "MARKET"
        self.side = "BUY"

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Side selector (BUY/SELL)
        side_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        side_frame.pack(fill=tk.X, pady=(0, 15))

        self.buy_btn = tk.Button(
            side_frame, text="BUY",
            bg=theme['success'], fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT, cursor='hand2',
            command=lambda: self._set_side("BUY")
        )
        self.buy_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)

        self.sell_btn = tk.Button(
            side_frame, text="SELL",
            bg=theme['bg_secondary'], fg=theme['text_secondary'],
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT, cursor='hand2',
            command=lambda: self._set_side("SELL")
        )
        self.sell_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=8)

        # Symbol selection
        symbol_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        symbol_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            symbol_frame, text="Symbol",
            bg=theme['bg_card'], fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.symbol_search = StockSearchWidget(
            symbol_frame, theme,
            on_select=self._on_symbol_selected,
            placeholder="Search stock...",
            width=25
        )
        self.symbol_search.pack(fill=tk.X, pady=(5, 0))
        self.selected_symbol = ""

        # Quantity
        qty_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        qty_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            qty_frame, text="Quantity",
            bg=theme['bg_card'], fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.qty_entry = tk.Entry(
            qty_frame,
            bg=theme['bg_secondary'],
            fg=theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=theme['text_primary'],
            relief=tk.FLAT
        )
        self.qty_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.qty_entry.insert(0, "1")

        # Product type (Intraday/Delivery)
        product_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        product_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            product_frame, text="Product",
            bg=theme['bg_card'], fg=theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.product_var = tk.StringVar(value="MIS")
        product_combo = ttk.Combobox(
            product_frame,
            textvariable=self.product_var,
            values=["MIS (Intraday)", "CNC (Delivery)", "NRML (F&O)"],
            state='readonly',
            font=('Segoe UI', 10)
        )
        product_combo.set("MIS (Intraday)")
        product_combo.pack(fill=tk.X, pady=(5, 0))

        # Dynamic fields container
        self.dynamic_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        self.dynamic_frame.pack(fill=tk.X)

        # Initialize dynamic fields
        self._create_dynamic_fields()

        # Submit button
        self.submit_btn = tk.Button(
            self.frame, text="PLACE ORDER",
            bg=theme['success'], fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT, cursor='hand2',
            command=self._submit_order
        )
        self.submit_btn.pack(fill=tk.X, pady=(20, 0), ipady=10)

    def _set_side(self, side: str):
        """Set order side"""
        self.side = side
        if side == "BUY":
            self.buy_btn.config(bg=self.theme['success'], fg='white')
            self.sell_btn.config(bg=self.theme['bg_secondary'], fg=self.theme['text_secondary'])
            self.submit_btn.config(bg=self.theme['success'])
        else:
            self.sell_btn.config(bg=self.theme['danger'], fg='white')
            self.buy_btn.config(bg=self.theme['bg_secondary'], fg=self.theme['text_secondary'])
            self.submit_btn.config(bg=self.theme['danger'])

    def _on_symbol_selected(self, symbol: str):
        """Handle symbol selection"""
        # Extract symbol name without exchange
        if ':' in symbol:
            _, name = symbol.split(':', 1)
        else:
            name = symbol
        self.selected_symbol = name

    def set_order_type(self, order_type: str):
        """Update form based on order type"""
        self.order_type = order_type
        self._create_dynamic_fields()

    def _create_dynamic_fields(self):
        """Create fields based on order type"""
        # Clear existing dynamic fields
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        # Price field (for Limit, SL, etc.)
        if self.order_type in ["LIMIT", "SL", "SL-M", "BRACKET", "ICEBERG"]:
            price_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
            price_frame.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                price_frame, text="Price",
                bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
                font=('Segoe UI', 10)
            ).pack(anchor=tk.W)

            self.price_entry = tk.Entry(
                price_frame,
                bg=self.theme['bg_secondary'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 11),
                insertbackground=self.theme['text_primary'],
                relief=tk.FLAT
            )
            self.price_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        else:
            self.price_entry = None

        # Trigger price (for SL orders)
        if self.order_type in ["SL", "SL-M"]:
            trigger_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
            trigger_frame.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                trigger_frame, text="Trigger Price",
                bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
                font=('Segoe UI', 10)
            ).pack(anchor=tk.W)

            self.trigger_entry = tk.Entry(
                trigger_frame,
                bg=self.theme['bg_secondary'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 11),
                insertbackground=self.theme['text_primary'],
                relief=tk.FLAT
            )
            self.trigger_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        else:
            self.trigger_entry = None

        # Bracket order fields
        if self.order_type == "BRACKET":
            self._create_bracket_fields()

        # Iceberg fields
        if self.order_type == "ICEBERG":
            self._create_iceberg_fields()

        # TWAP fields
        if self.order_type == "TWAP":
            self._create_twap_fields()

        # VWAP fields
        if self.order_type == "VWAP":
            self._create_vwap_fields()

    def _create_bracket_fields(self):
        """Create bracket order specific fields"""
        # Stop Loss
        sl_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        sl_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            sl_frame, text="Stop Loss",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.sl_entry = tk.Entry(
            sl_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.sl_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)

        # Target
        target_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        target_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            target_frame, text="Target",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.target_entry = tk.Entry(
            target_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.target_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)

        # Trailing SL option
        trail_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        trail_frame.pack(fill=tk.X, pady=(0, 10))

        self.trailing_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            trail_frame, text="Enable Trailing Stop Loss",
            variable=self.trailing_var,
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

    def _create_iceberg_fields(self):
        """Create iceberg order specific fields"""
        # Visible quantity
        vis_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        vis_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            vis_frame, text="Visible Quantity (shown to market)",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.visible_qty_entry = tk.Entry(
            vis_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.visible_qty_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.visible_qty_entry.insert(0, "10")

        # Randomize visible
        rand_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        rand_frame.pack(fill=tk.X, pady=(0, 10))

        self.randomize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            rand_frame, text="Randomize visible quantity (+/-20%)",
            variable=self.randomize_var,
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            selectcolor=self.theme['bg_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

    def _create_twap_fields(self):
        """Create TWAP order specific fields"""
        # Duration
        dur_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        dur_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            dur_frame, text="Duration (minutes)",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.duration_entry = tk.Entry(
            dur_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.duration_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.duration_entry.insert(0, "30")

        # Number of slices
        slice_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        slice_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            slice_frame, text="Number of Slices",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.slices_entry = tk.Entry(
            slice_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.slices_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.slices_entry.insert(0, "10")

    def _create_vwap_fields(self):
        """Create VWAP order specific fields"""
        # Duration
        dur_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        dur_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            dur_frame, text="Duration (minutes)",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.duration_entry = tk.Entry(
            dur_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.duration_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.duration_entry.insert(0, "60")

        # Participation rate
        part_frame = tk.Frame(self.dynamic_frame, bg=self.theme['bg_card'])
        part_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            part_frame, text="Max Participation Rate (%)",
            bg=self.theme['bg_card'], fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)

        self.participation_entry = tk.Entry(
            part_frame,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11),
            insertbackground=self.theme['text_primary'],
            relief=tk.FLAT
        )
        self.participation_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        self.participation_entry.insert(0, "20")

    def _submit_order(self):
        """Submit the order"""
        # Validate inputs
        if not self.selected_symbol:
            messagebox.showerror("Error", "Please select a symbol")
            return

        try:
            quantity = int(self.qty_entry.get())
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid quantity: {e}")
            return

        # Build order data
        order_data = {
            'symbol': self.selected_symbol,
            'side': self.side,
            'quantity': quantity,
            'order_type': self.order_type,
            'product': self.product_var.get().split()[0],  # Extract MIS/CNC/NRML
        }

        # Add price if applicable
        if self.price_entry and self.price_entry.get():
            try:
                order_data['price'] = float(self.price_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid price")
                return

        # Add trigger price if applicable
        if hasattr(self, 'trigger_entry') and self.trigger_entry and self.trigger_entry.get():
            try:
                order_data['trigger_price'] = float(self.trigger_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid trigger price")
                return

        # Add bracket order fields
        if self.order_type == "BRACKET":
            try:
                order_data['stop_loss'] = float(self.sl_entry.get())
                order_data['target'] = float(self.target_entry.get())
                order_data['trailing_sl'] = self.trailing_var.get()
            except (ValueError, AttributeError) as e:
                messagebox.showerror("Error", f"Invalid bracket order values: {e}")
                return

        # Add iceberg fields
        if self.order_type == "ICEBERG":
            try:
                order_data['visible_quantity'] = int(self.visible_qty_entry.get())
                order_data['randomize_visible'] = self.randomize_var.get()
            except (ValueError, AttributeError) as e:
                messagebox.showerror("Error", f"Invalid iceberg values: {e}")
                return

        # Add TWAP fields
        if self.order_type == "TWAP":
            try:
                order_data['duration_minutes'] = float(self.duration_entry.get())
                order_data['num_slices'] = int(self.slices_entry.get())
            except (ValueError, AttributeError) as e:
                messagebox.showerror("Error", f"Invalid TWAP values: {e}")
                return

        # Add VWAP fields
        if self.order_type == "VWAP":
            try:
                order_data['duration_minutes'] = float(self.duration_entry.get())
                order_data['participation_rate'] = float(self.participation_entry.get()) / 100
            except (ValueError, AttributeError) as e:
                messagebox.showerror("Error", f"Invalid VWAP values: {e}")
                return

        # Call callback
        if self.on_submit:
            self.on_submit(order_data)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class OrderHistoryTable:
    """
    Table showing recent orders.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        inner = tk.Frame(self.frame, bg=theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        tk.Label(
            inner, text="Recent Orders",
            bg=theme['bg_card'], fg=theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Create treeview
        columns = ('time', 'symbol', 'side', 'type', 'qty', 'price', 'status')

        style = ttk.Style()
        style.configure(
            "Orders.Treeview",
            background=theme['bg_secondary'],
            foreground=theme['text_primary'],
            fieldbackground=theme['bg_secondary'],
            rowheight=28,
            font=('Segoe UI', 10)
        )
        style.configure(
            "Orders.Treeview.Heading",
            background=theme['bg_card'],
            foreground=theme['text_primary'],
            font=('Segoe UI', 10, 'bold')
        )

        self.tree = ttk.Treeview(
            inner,
            columns=columns,
            show='headings',
            height=8,
            style="Orders.Treeview"
        )

        # Configure tags
        self.tree.tag_configure('buy', foreground='#00ff88')
        self.tree.tag_configure('sell', foreground='#ff4444')
        self.tree.tag_configure('pending', foreground='#ffa500')
        self.tree.tag_configure('complete', foreground='#00ff88')
        self.tree.tag_configure('rejected', foreground='#ff4444')

        # Define headings
        self.tree.heading('time', text='Time')
        self.tree.heading('symbol', text='Symbol')
        self.tree.heading('side', text='Side')
        self.tree.heading('type', text='Type')
        self.tree.heading('qty', text='Qty')
        self.tree.heading('price', text='Price')
        self.tree.heading('status', text='Status')

        # Column widths
        self.tree.column('time', width=80)
        self.tree.column('symbol', width=80)
        self.tree.column('side', width=50)
        self.tree.column('type', width=70)
        self.tree.column('qty', width=50, anchor=tk.E)
        self.tree.column('price', width=70, anchor=tk.E)
        self.tree.column('status', width=80)

        self.tree.pack(fill=tk.BOTH, expand=True)

    def add_order(self, order_data: Dict):
        """Add an order to the table"""
        from datetime import datetime

        time_str = datetime.now().strftime("%H:%M:%S")
        side = order_data.get('side', 'BUY')
        status = order_data.get('status', 'PENDING')

        # Determine tag
        if status == 'COMPLETE':
            tag = 'complete'
        elif status == 'REJECTED':
            tag = 'rejected'
        elif side == 'BUY':
            tag = 'buy'
        else:
            tag = 'sell'

        self.tree.insert('', 0, values=(
            time_str,
            order_data.get('symbol', 'N/A'),
            side,
            order_data.get('order_type', 'MARKET'),
            order_data.get('quantity', 0),
            f"Rs.{order_data.get('price', 0):,.2f}" if order_data.get('price') else 'MKT',
            status
        ), tags=(tag,))

        # Keep only last 50 orders
        children = self.tree.get_children()
        if len(children) > 50:
            self.tree.delete(children[-1])

    def update_order_status(self, order_id: str, status: str):
        """Update status of an existing order"""
        # This would require tracking order IDs - simplified for now
        pass

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class OrderPanel:
    """
    Complete Order Panel for placing different types of orders.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_name: str = 'dark',
        order_manager=None,
        broker=None,
        on_order_placed: Optional[Callable[[Dict], None]] = None
    ):
        self.theme = get_theme(theme_name)
        self.order_manager = order_manager
        self.broker = broker
        self.on_order_placed = on_order_placed

        # Main container
        self.frame = tk.Frame(parent, bg=self.theme['bg_primary'])

        # Two-column layout
        left_col = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_col = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Order Form Card (left)
        form_card = tk.Frame(left_col, bg=self.theme['bg_card'])
        form_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        form_card.pack(fill=tk.BOTH, expand=True)

        form_inner = tk.Frame(form_card, bg=self.theme['bg_card'])
        form_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        tk.Label(
            form_inner, text="Place Order",
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            font=('Segoe UI', 18, 'bold')
        ).pack(anchor=tk.W, pady=(0, 15))

        # Order type selector
        self.type_selector = OrderTypeSelector(
            form_inner, self.theme,
            on_select=self._on_order_type_changed
        )
        self.type_selector.pack(fill=tk.X, pady=(0, 20))

        # Separator
        tk.Frame(form_inner, bg=self.theme['border'], height=1).pack(fill=tk.X, pady=10)

        # Order form
        self.order_form = OrderForm(
            form_inner, self.theme,
            on_submit=self._place_order
        )
        self.order_form.pack(fill=tk.X)

        # Order History (right)
        self.order_history = OrderHistoryTable(right_col, self.theme)
        self.order_history.pack(fill=tk.BOTH, expand=True)

        # Execution status card (right bottom)
        status_card = tk.Frame(right_col, bg=self.theme['bg_card'])
        status_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        status_card.pack(fill=tk.X, pady=(10, 0))

        status_inner = tk.Frame(status_card, bg=self.theme['bg_card'])
        status_inner.pack(fill=tk.X, padx=15, pady=15)

        tk.Label(
            status_inner, text="Execution Status",
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W)

        self.status_label = tk.Label(
            status_inner, text="Ready to place orders",
            bg=self.theme['bg_card'], fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        )
        self.status_label.pack(anchor=tk.W, pady=(5, 0))

        # Progress bar for advanced orders
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_inner,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        self.progress_bar.pack_forget()  # Hide initially

    def _on_order_type_changed(self, order_type: str):
        """Handle order type change"""
        self.order_form.set_order_type(order_type)

    def _place_order(self, order_data: Dict):
        """Place the order using appropriate executor"""
        order_type = order_data.get('order_type', 'MARKET')

        self.status_label.config(
            text=f"Placing {order_type} order for {order_data['symbol']}...",
            fg=self.theme['warning']
        )

        # Add to history as pending
        order_data['status'] = 'PENDING'
        self.order_history.add_order(order_data)

        # Route to appropriate handler
        if order_type in ["MARKET", "LIMIT", "SL", "SL-M"]:
            self._place_basic_order(order_data)
        elif order_type == "BRACKET":
            self._place_bracket_order(order_data)
        elif order_type == "ICEBERG":
            self._place_iceberg_order(order_data)
        elif order_type == "TWAP":
            self._place_twap_order(order_data)
        elif order_type == "VWAP":
            self._place_vwap_order(order_data)

    def _place_basic_order(self, order_data: Dict):
        """Place basic order via OrderManager"""
        try:
            if self.order_manager:
                from core.order_manager import OrderType, Side, ProductType

                # Map order type
                type_map = {
                    'MARKET': OrderType.MARKET,
                    'LIMIT': OrderType.LIMIT,
                    'SL': OrderType.SL,
                    'SL-M': OrderType.SL_M,
                }

                # Map side
                side = Side.BUY if order_data['side'] == 'BUY' else Side.SELL

                # Map product
                product_map = {
                    'MIS': ProductType.INTRADAY,
                    'CNC': ProductType.DELIVERY,
                    'NRML': ProductType.MARGIN,
                }

                order = self.order_manager.place_order(
                    symbol=order_data['symbol'],
                    side=side,
                    quantity=order_data['quantity'],
                    order_type=type_map.get(order_data['order_type'], OrderType.MARKET),
                    price=order_data.get('price', 0.0),
                    product=product_map.get(order_data['product'], ProductType.INTRADAY),
                )

                status = order.status.value
                order_data['status'] = status

                self.status_label.config(
                    text=f"Order {status}: {order.symbol} {order.quantity} @ Rs.{order.average_price:.2f}",
                    fg=self.theme['success'] if status == 'COMPLETE' else self.theme['warning']
                )

            else:
                # Demo mode - simulate order
                order_data['status'] = 'COMPLETE'
                self.status_label.config(
                    text=f"Demo: Order placed for {order_data['symbol']}",
                    fg=self.theme['success']
                )

            # Update history
            self.order_history.add_order(order_data)

            # Callback
            if self.on_order_placed:
                self.on_order_placed(order_data)

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            order_data['status'] = 'REJECTED'
            self.status_label.config(
                text=f"Order failed: {str(e)}",
                fg=self.theme['danger']
            )
            self.order_history.add_order(order_data)

    def _place_bracket_order(self, order_data: Dict):
        """Place bracket order"""
        try:
            # Use bracket order manager if available
            self.status_label.config(
                text=f"Placing bracket order: Entry + SL + Target",
                fg=self.theme['info']
            )

            # For now, simulate bracket order
            order_data['status'] = 'COMPLETE'
            self.status_label.config(
                text=f"Bracket order placed: {order_data['symbol']} with SL @ {order_data.get('stop_loss')} and Target @ {order_data.get('target')}",
                fg=self.theme['success']
            )

            self.order_history.add_order(order_data)

            if self.on_order_placed:
                self.on_order_placed(order_data)

        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            order_data['status'] = 'REJECTED'
            self.status_label.config(text=f"Bracket order failed: {str(e)}", fg=self.theme['danger'])

    def _place_iceberg_order(self, order_data: Dict):
        """Place iceberg order"""
        try:
            self.progress_bar.pack(fill=tk.X, pady=(10, 0))
            self.progress_var.set(0)

            self.status_label.config(
                text=f"Executing iceberg order: {order_data['quantity']} (visible: {order_data.get('visible_quantity', 10)})",
                fg=self.theme['info']
            )

            # Execute in background thread
            def execute_iceberg():
                try:
                    from core.execution.iceberg import IcebergExecutor, IcebergConfig, Side

                    config = IcebergConfig(
                        visible_quantity=order_data.get('visible_quantity', 10),
                        randomize_visible=order_data.get('randomize_visible', True),
                    )

                    executor = IcebergExecutor(self.broker, config)
                    side = Side.BUY if order_data['side'] == 'BUY' else Side.SELL

                    result = executor.execute_sync(
                        symbol=order_data['symbol'],
                        total_quantity=order_data['quantity'],
                        side=side,
                        limit_price=order_data.get('price', 100.0),
                    )

                    # Update UI from main thread
                    self.frame.after(0, lambda: self._on_iceberg_complete(order_data, result))

                except Exception as e:
                    self.frame.after(0, lambda: self._on_advanced_order_error(order_data, str(e)))

            thread = threading.Thread(target=execute_iceberg, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"Iceberg order failed: {e}")
            self._on_advanced_order_error(order_data, str(e))

    def _on_iceberg_complete(self, order_data: Dict, result):
        """Handle iceberg order completion"""
        self.progress_var.set(100)
        order_data['status'] = 'COMPLETE' if result.success else 'PARTIAL'

        self.status_label.config(
            text=f"Iceberg complete: Filled {result.total_filled}/{order_data['quantity']} @ Rs.{result.average_price:.2f}",
            fg=self.theme['success'] if result.success else self.theme['warning']
        )

        self.order_history.add_order(order_data)
        self.frame.after(2000, lambda: self.progress_bar.pack_forget())

        if self.on_order_placed:
            self.on_order_placed(order_data)

    def _place_twap_order(self, order_data: Dict):
        """Place TWAP order"""
        try:
            self.progress_bar.pack(fill=tk.X, pady=(10, 0))
            self.progress_var.set(0)

            duration = order_data.get('duration_minutes', 30)
            slices = order_data.get('num_slices', 10)

            self.status_label.config(
                text=f"Executing TWAP: {order_data['quantity']} in {slices} slices over {duration} mins",
                fg=self.theme['info']
            )

            # Execute in background
            def execute_twap():
                try:
                    from core.execution.twap import TWAPExecutor, TWAPConfig, Side

                    config = TWAPConfig(
                        duration_minutes=duration,
                        num_slices=slices,
                    )

                    executor = TWAPExecutor(self.broker, config)
                    side = Side.BUY if order_data['side'] == 'BUY' else Side.SELL

                    result = executor.execute_sync(
                        symbol=order_data['symbol'],
                        quantity=order_data['quantity'],
                        side=side,
                    )

                    self.frame.after(0, lambda: self._on_twap_complete(order_data, result))

                except Exception as e:
                    self.frame.after(0, lambda: self._on_advanced_order_error(order_data, str(e)))

            thread = threading.Thread(target=execute_twap, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"TWAP order failed: {e}")
            self._on_advanced_order_error(order_data, str(e))

    def _on_twap_complete(self, order_data: Dict, result):
        """Handle TWAP order completion"""
        self.progress_var.set(100)
        order_data['status'] = 'COMPLETE' if result.success else 'PARTIAL'

        self.status_label.config(
            text=f"TWAP complete: Filled {result.total_filled}/{order_data['quantity']} @ Rs.{result.average_price:.2f}",
            fg=self.theme['success'] if result.success else self.theme['warning']
        )

        self.order_history.add_order(order_data)
        self.frame.after(2000, lambda: self.progress_bar.pack_forget())

        if self.on_order_placed:
            self.on_order_placed(order_data)

    def _place_vwap_order(self, order_data: Dict):
        """Place VWAP order"""
        try:
            self.progress_bar.pack(fill=tk.X, pady=(10, 0))
            self.progress_var.set(0)

            duration = order_data.get('duration_minutes', 60)

            self.status_label.config(
                text=f"Executing VWAP: {order_data['quantity']} over {duration} mins",
                fg=self.theme['info']
            )

            # For now, simulate VWAP (similar to TWAP)
            def execute_vwap():
                try:
                    # VWAP would use volume profile - simplified for demo
                    from core.execution.twap import TWAPExecutor, TWAPConfig, Side

                    config = TWAPConfig(
                        duration_minutes=duration,
                        num_slices=12,  # More slices for VWAP
                    )

                    executor = TWAPExecutor(self.broker, config)
                    side = Side.BUY if order_data['side'] == 'BUY' else Side.SELL

                    result = executor.execute_sync(
                        symbol=order_data['symbol'],
                        quantity=order_data['quantity'],
                        side=side,
                    )

                    self.frame.after(0, lambda: self._on_vwap_complete(order_data, result))

                except Exception as e:
                    self.frame.after(0, lambda: self._on_advanced_order_error(order_data, str(e)))

            thread = threading.Thread(target=execute_vwap, daemon=True)
            thread.start()

        except Exception as e:
            logger.error(f"VWAP order failed: {e}")
            self._on_advanced_order_error(order_data, str(e))

    def _on_vwap_complete(self, order_data: Dict, result):
        """Handle VWAP order completion"""
        self.progress_var.set(100)
        order_data['status'] = 'COMPLETE' if result.success else 'PARTIAL'

        self.status_label.config(
            text=f"VWAP complete: Filled {result.total_filled}/{order_data['quantity']} @ Rs.{result.average_price:.2f}",
            fg=self.theme['success'] if result.success else self.theme['warning']
        )

        self.order_history.add_order(order_data)
        self.frame.after(2000, lambda: self.progress_bar.pack_forget())

        if self.on_order_placed:
            self.on_order_placed(order_data)

    def _on_advanced_order_error(self, order_data: Dict, error: str):
        """Handle advanced order error"""
        order_data['status'] = 'REJECTED'
        self.status_label.config(
            text=f"Order failed: {error}",
            fg=self.theme['danger']
        )
        self.order_history.add_order(order_data)
        self.progress_bar.pack_forget()

    def set_order_manager(self, order_manager):
        """Set order manager"""
        self.order_manager = order_manager

    def set_broker(self, broker):
        """Set broker"""
        self.broker = broker

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("ORDER PANEL - Test")
    print("=" * 50)

    root = tk.Tk()
    root.title("Order Panel Test")
    root.geometry("1100x700")
    root.configure(bg='#1a1a2e')

    def on_order(data):
        print(f"Order placed: {data}")

    panel = OrderPanel(root, 'dark', on_order_placed=on_order)
    panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    root.mainloop()

    print("\n" + "=" * 50)
    print("Order Panel ready!")
    print("=" * 50)
