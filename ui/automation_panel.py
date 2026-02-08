# -*- coding: utf-8 -*-
"""
Automation Panel - See Your Bot's Brain in Action!
====================================================
Visualizes the complete algo trading loop:

    Connect -> Listen -> Process -> Decide -> Act -> Repeat

Shows:
- Live event flow (Ticks -> Bars -> Signals -> Orders)
- Strategy status and symbol mapping
- Real-time metrics (events/sec, latency, fills)
- Phase indicators for each step of the loop
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Dict, Any, List, Deque
from collections import deque
from datetime import datetime
import threading
import logging
import time

from .themes import get_theme

logger = logging.getLogger(__name__)


class PhaseIndicator:
    """
    Visual indicator showing current phase of the trading loop.

    Phases:
    1. CONNECT - Handshake with broker
    2. LISTEN - Receiving ticks from WebSocket
    3. PROCESS - Resampling ticks to bars, calculating indicators
    4. DECIDE - Strategy evaluating signals
    5. ACT - Placing orders
    """

    PHASES = [
        ("CONNECT", "Handshake with broker"),
        ("LISTEN", "Receiving market data"),
        ("PROCESS", "Building candles & indicators"),
        ("DECIDE", "Strategy evaluation"),
        ("ACT", "Order execution"),
    ]

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme
        self.current_phase = 0
        self.phase_active = [False] * 5

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Title
        tk.Label(
            self.frame, text="Trading Loop",
            bg=theme['bg_card'], fg=theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Phase indicators
        self.indicators = []
        phases_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        phases_frame.pack(fill=tk.X)

        for i, (name, desc) in enumerate(self.PHASES):
            phase_frame = tk.Frame(phases_frame, bg=theme['bg_card'])
            phase_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

            # Indicator dot
            dot = tk.Label(
                phase_frame, text="●",
                bg=theme['bg_card'], fg=theme['text_dim'],
                font=('Segoe UI', 16)
            )
            dot.pack()

            # Phase name
            tk.Label(
                phase_frame, text=name,
                bg=theme['bg_card'], fg=theme['text_secondary'],
                font=('Segoe UI', 8)
            ).pack()

            self.indicators.append(dot)

        # Arrow connectors
        arrow_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        arrow_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            arrow_frame, text="→ → → → → (repeat)",
            bg=theme['bg_card'], fg=theme['text_dim'],
            font=('Segoe UI', 10)
        ).pack()

    def set_phase(self, phase_index: int, active: bool = True):
        """Activate/deactivate a phase."""
        if 0 <= phase_index < len(self.indicators):
            self.phase_active[phase_index] = active
            color = self.theme['success'] if active else self.theme['text_dim']
            self.indicators[phase_index].config(fg=color)

    def pulse_phase(self, phase_index: int):
        """Briefly highlight a phase (for events)."""
        if 0 <= phase_index < len(self.indicators):
            self.indicators[phase_index].config(fg=self.theme['accent'])
            self.frame.after(200, lambda: self._reset_phase(phase_index))

    def _reset_phase(self, phase_index: int):
        """Reset phase to its steady state."""
        if self.phase_active[phase_index]:
            self.indicators[phase_index].config(fg=self.theme['success'])
        else:
            self.indicators[phase_index].config(fg=self.theme['text_dim'])

    def set_all_active(self, active: bool):
        """Set all phases active/inactive."""
        for i in range(len(self.PHASES)):
            self.set_phase(i, active)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class EventStream:
    """
    Live event stream visualization.
    Shows events flowing through the system in real-time.
    """

    MAX_EVENTS = 100

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme
        self.events: Deque = deque(maxlen=self.MAX_EVENTS)

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Title row
        title_row = tk.Frame(self.frame, bg=theme['bg_card'])
        title_row.pack(fill=tk.X, pady=(0, 5))

        tk.Label(
            title_row, text="Live Event Stream",
            bg=theme['bg_card'], fg=theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(side=tk.LEFT)

        # Events per second counter
        self.eps_label = tk.Label(
            title_row, text="0 events/sec",
            bg=theme['bg_card'], fg=theme['text_dim'],
            font=('Segoe UI', 10)
        )
        self.eps_label.pack(side=tk.RIGHT)

        # Event type filters
        filter_row = tk.Frame(self.frame, bg=theme['bg_card'])
        filter_row.pack(fill=tk.X, pady=(0, 5))

        self.show_ticks = tk.BooleanVar(value=False)  # Too noisy by default
        self.show_bars = tk.BooleanVar(value=True)
        self.show_signals = tk.BooleanVar(value=True)
        self.show_orders = tk.BooleanVar(value=True)

        filters = [
            ("Ticks", self.show_ticks, theme['info']),
            ("Bars", self.show_bars, theme['accent']),
            ("Signals", self.show_signals, theme['warning']),
            ("Orders", self.show_orders, theme['success']),
        ]

        for name, var, color in filters:
            cb = tk.Checkbutton(
                filter_row, text=name, variable=var,
                bg=theme['bg_card'], fg=color,
                selectcolor=theme['bg_secondary'],
                font=('Segoe UI', 9),
                command=self._refresh_display
            )
            cb.pack(side=tk.LEFT, padx=5)

        # Event display
        self.text = tk.Text(
            self.frame,
            height=12,
            bg=theme['bg_secondary'],
            fg=theme['text_primary'],
            font=('Consolas', 9),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.NONE
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Horizontal scrollbar
        h_scroll = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.text.xview)
        h_scroll.pack(fill=tk.X)
        self.text.config(xscrollcommand=h_scroll.set)

        # Configure tags
        self.text.tag_configure('time', foreground=theme['text_dim'])
        self.text.tag_configure('tick', foreground=theme['info'])
        self.text.tag_configure('bar', foreground=theme['accent'])
        self.text.tag_configure('signal', foreground=theme['warning'])
        self.text.tag_configure('order', foreground=theme['success'])
        self.text.tag_configure('fill', foreground='#00ff88')
        self.text.tag_configure('error', foreground=theme['danger'])

        # Event counter for EPS calculation
        self._event_count = 0
        self._last_eps_update = time.time()

    def add_event(self, event_type: str, message: str):
        """Add an event to the stream."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.events.append((timestamp, event_type, message))
        self._event_count += 1

        # Check if we should display this event
        should_show = (
            (event_type == 'tick' and self.show_ticks.get()) or
            (event_type == 'bar' and self.show_bars.get()) or
            (event_type == 'signal' and self.show_signals.get()) or
            (event_type in ['order', 'fill'] and self.show_orders.get()) or
            (event_type == 'error')
        )

        if should_show:
            self._display_event(timestamp, event_type, message)

        # Update EPS every second
        now = time.time()
        if now - self._last_eps_update >= 1.0:
            eps = self._event_count / (now - self._last_eps_update)
            self.eps_label.config(text=f"{eps:.1f} events/sec")
            self._event_count = 0
            self._last_eps_update = now

    def _display_event(self, timestamp: str, event_type: str, message: str):
        """Display a single event."""
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, f"[{timestamp}] ", 'time')
        self.text.insert(tk.END, f"[{event_type.upper():6}] ", event_type)
        self.text.insert(tk.END, f"{message}\n")

        # Auto-scroll and trim
        self.text.see(tk.END)
        lines = int(self.text.index('end-1c').split('.')[0])
        if lines > 500:
            self.text.delete('1.0', '100.0')

        self.text.config(state=tk.DISABLED)

    def _refresh_display(self):
        """Refresh display based on current filters."""
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)

        for timestamp, event_type, message in self.events:
            should_show = (
                (event_type == 'tick' and self.show_ticks.get()) or
                (event_type == 'bar' and self.show_bars.get()) or
                (event_type == 'signal' and self.show_signals.get()) or
                (event_type in ['order', 'fill'] and self.show_orders.get()) or
                (event_type == 'error')
            )
            if should_show:
                self.text.insert(tk.END, f"[{timestamp}] ", 'time')
                self.text.insert(tk.END, f"[{event_type.upper():6}] ", event_type)
                self.text.insert(tk.END, f"{message}\n")

        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def clear(self):
        """Clear all events."""
        self.events.clear()
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)
        self.text.config(state=tk.DISABLED)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class StrategyStatus:
    """
    Shows active strategies and their symbol mappings.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Title
        tk.Label(
            self.frame, text="Active Strategies",
            bg=theme['bg_card'], fg=theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        # Strategies list
        self.list_frame = tk.Frame(self.frame, bg=theme['bg_card'])
        self.list_frame.pack(fill=tk.BOTH, expand=True)

        # Empty state
        self.empty_label = tk.Label(
            self.list_frame, text="No strategies active\nStart the bot to begin trading",
            bg=theme['bg_card'], fg=theme['text_dim'],
            font=('Segoe UI', 10), justify=tk.CENTER
        )
        self.empty_label.pack(expand=True, pady=20)

        self.strategy_widgets: Dict[str, tk.Frame] = {}

    def update_strategies(self, strategies: Dict[str, str]):
        """
        Update displayed strategies.

        Args:
            strategies: Dict of strategy_name -> symbol
        """
        # Clear existing
        for widget in self.strategy_widgets.values():
            widget.destroy()
        self.strategy_widgets.clear()

        if not strategies:
            self.empty_label.pack(expand=True, pady=20)
            return

        self.empty_label.pack_forget()

        for name, symbol in strategies.items():
            row = tk.Frame(self.list_frame, bg=self.theme['bg_secondary'])
            row.pack(fill=tk.X, pady=2)

            # Strategy name
            tk.Label(
                row, text=f"  {name}",
                bg=self.theme['bg_secondary'], fg=self.theme['text_primary'],
                font=('Segoe UI', 10, 'bold'), width=20, anchor=tk.W
            ).pack(side=tk.LEFT, padx=5, pady=5)

            # Arrow
            tk.Label(
                row, text="→",
                bg=self.theme['bg_secondary'], fg=self.theme['text_dim'],
                font=('Segoe UI', 10)
            ).pack(side=tk.LEFT)

            # Symbol
            tk.Label(
                row, text=symbol,
                bg=self.theme['bg_secondary'], fg=self.theme['accent'],
                font=('Segoe UI', 10, 'bold')
            ).pack(side=tk.LEFT, padx=5)

            # Status indicator
            status_dot = tk.Label(
                row, text="●",
                bg=self.theme['bg_secondary'], fg=self.theme['success'],
                font=('Segoe UI', 12)
            )
            status_dot.pack(side=tk.RIGHT, padx=10)

            self.strategy_widgets[name] = row

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class MetricsPanel:
    """
    Real-time metrics: events processed, latency, fills, etc.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        # Metrics in a row
        metrics = [
            ("Ticks", "0"),
            ("Bars", "0"),
            ("Signals", "0"),
            ("Orders", "0"),
            ("Fills", "0"),
            ("Latency", "0ms"),
        ]

        self.labels: Dict[str, tk.Label] = {}

        for name, value in metrics:
            col = tk.Frame(self.frame, bg=theme['bg_card'])
            col.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Value
            val_label = tk.Label(
                col, text=value,
                bg=theme['bg_card'], fg=theme['text_primary'],
                font=('Segoe UI', 16, 'bold')
            )
            val_label.pack()

            # Name
            tk.Label(
                col, text=name,
                bg=theme['bg_card'], fg=theme['text_dim'],
                font=('Segoe UI', 9)
            ).pack()

            self.labels[name.lower()] = val_label

    def update_metric(self, name: str, value: str):
        """Update a specific metric."""
        if name in self.labels:
            self.labels[name].config(text=value)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class AutomationPanel:
    """
    Complete Automation Control Panel.

    Visualizes the full algo trading loop:
    Connect -> Listen -> Process -> Decide -> Act
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme_name: str = 'dark',
        event_bus=None,
        trading_engine=None,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None
    ):
        self.theme = get_theme(theme_name)
        self.event_bus = event_bus
        self.trading_engine = trading_engine
        self.on_start = on_start
        self.on_stop = on_stop

        self.is_running = False

        # Metrics counters
        self._tick_count = 0
        self._bar_count = 0
        self._signal_count = 0
        self._order_count = 0
        self._fill_count = 0

        # Main container
        self.frame = tk.Frame(parent, bg=self.theme['bg_primary'])

        # Two columns
        left_col = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_col = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # === LEFT COLUMN ===

        # Control Card
        control_card = tk.Frame(left_col, bg=self.theme['bg_card'])
        control_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        control_card.pack(fill=tk.X, pady=(0, 10))

        control_inner = tk.Frame(control_card, bg=self.theme['bg_card'])
        control_inner.pack(fill=tk.X, padx=20, pady=20)

        # Title and status
        title_row = tk.Frame(control_inner, bg=self.theme['bg_card'])
        title_row.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            title_row, text="Automation Control",
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            font=('Segoe UI', 18, 'bold')
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            title_row, text="STOPPED",
            bg=self.theme['bg_card'], fg=self.theme['danger'],
            font=('Segoe UI', 12, 'bold')
        )
        self.status_label.pack(side=tk.RIGHT)

        # Start/Stop buttons
        btn_frame = tk.Frame(control_inner, bg=self.theme['bg_card'])
        btn_frame.pack(fill=tk.X)

        self.start_btn = tk.Button(
            btn_frame, text="▶  START AUTOMATION",
            bg=self.theme['success'], fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT, cursor='hand2',
            command=self._start_automation
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)

        self.stop_btn = tk.Button(
            btn_frame, text="⏹  STOP",
            bg=self.theme['bg_secondary'], fg=self.theme['text_dim'],
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT, cursor='hand2',
            command=self._stop_automation,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0), ipadx=20, ipady=10)

        # Phase indicator
        phase_card = tk.Frame(left_col, bg=self.theme['bg_card'])
        phase_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        phase_card.pack(fill=tk.X, pady=(0, 10))

        phase_inner = tk.Frame(phase_card, bg=self.theme['bg_card'])
        phase_inner.pack(fill=tk.X, padx=20, pady=20)

        self.phase_indicator = PhaseIndicator(phase_inner, self.theme)
        self.phase_indicator.pack(fill=tk.X)

        # Metrics
        metrics_card = tk.Frame(left_col, bg=self.theme['bg_card'])
        metrics_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        metrics_card.pack(fill=tk.X, pady=(0, 10))

        metrics_inner = tk.Frame(metrics_card, bg=self.theme['bg_card'])
        metrics_inner.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(
            metrics_inner, text="Session Metrics",
            bg=self.theme['bg_card'], fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))

        self.metrics_panel = MetricsPanel(metrics_inner, self.theme)
        self.metrics_panel.pack(fill=tk.X)

        # Strategy status
        strategy_card = tk.Frame(left_col, bg=self.theme['bg_card'])
        strategy_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        strategy_card.pack(fill=tk.BOTH, expand=True)

        strategy_inner = tk.Frame(strategy_card, bg=self.theme['bg_card'])
        strategy_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.strategy_status = StrategyStatus(strategy_inner, self.theme)
        self.strategy_status.pack(fill=tk.BOTH, expand=True)

        # === RIGHT COLUMN ===

        # Event stream
        stream_card = tk.Frame(right_col, bg=self.theme['bg_card'])
        stream_card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        stream_card.pack(fill=tk.BOTH, expand=True)

        stream_inner = tk.Frame(stream_card, bg=self.theme['bg_card'])
        stream_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.event_stream = EventStream(stream_inner, self.theme)
        self.event_stream.pack(fill=tk.BOTH, expand=True)

        # Subscribe to EventBus if available
        if self.event_bus:
            self._subscribe_to_events()

    def _subscribe_to_events(self):
        """Subscribe to EventBus events for visualization."""
        try:
            from core.events import EventType

            # Subscribe to tick events
            self.event_bus.subscribe(
                EventType.TICK,
                self._on_tick_event,
                name="automation_panel_tick"
            )

            # Subscribe to bar events
            self.event_bus.subscribe(
                EventType.BAR,
                self._on_bar_event,
                name="automation_panel_bar"
            )

            # Subscribe to signal events
            self.event_bus.subscribe(
                EventType.SIGNAL_GENERATED,
                self._on_signal_event,
                name="automation_panel_signal"
            )

            # Subscribe to order events
            self.event_bus.subscribe(
                EventType.ORDER_SUBMITTED,
                self._on_order_event,
                name="automation_panel_order"
            )
            self.event_bus.subscribe(
                EventType.ORDER_FILLED,
                self._on_fill_event,
                name="automation_panel_fill"
            )

            logger.info("AutomationPanel subscribed to EventBus")

        except Exception as e:
            logger.warning(f"Could not subscribe to EventBus: {e}")

    def _on_tick_event(self, event):
        """Handle tick event."""
        self._tick_count += 1
        self.phase_indicator.pulse_phase(1)  # LISTEN phase

        symbol = getattr(event, 'symbol', 'Unknown')
        price = getattr(event, 'last_price', 0)

        self.frame.after(0, lambda: self._update_ui_tick(symbol, price))

    def _update_ui_tick(self, symbol, price):
        """Update UI from main thread."""
        self.event_stream.add_event('tick', f"{symbol} @ Rs.{price:.2f}")
        self.metrics_panel.update_metric('ticks', str(self._tick_count))

    def _on_bar_event(self, event):
        """Handle bar event."""
        self._bar_count += 1
        self.phase_indicator.pulse_phase(2)  # PROCESS phase

        symbol = getattr(event, 'symbol', 'Unknown')
        close = getattr(event, 'close', 0)

        self.frame.after(0, lambda: self._update_ui_bar(symbol, close))

    def _update_ui_bar(self, symbol, close):
        """Update UI from main thread."""
        self.event_stream.add_event('bar', f"{symbol} bar closed @ Rs.{close:.2f}")
        self.metrics_panel.update_metric('bars', str(self._bar_count))

    def _on_signal_event(self, event):
        """Handle signal event."""
        self._signal_count += 1
        self.phase_indicator.pulse_phase(3)  # DECIDE phase

        symbol = getattr(event, 'symbol', 'Unknown')
        signal_type = getattr(event, 'signal_type', 'HOLD')

        self.frame.after(0, lambda: self._update_ui_signal(symbol, signal_type))

    def _update_ui_signal(self, symbol, signal_type):
        """Update UI from main thread."""
        self.event_stream.add_event('signal', f"{signal_type} signal for {symbol}")
        self.metrics_panel.update_metric('signals', str(self._signal_count))

    def _on_order_event(self, event):
        """Handle order event."""
        self._order_count += 1
        self.phase_indicator.pulse_phase(4)  # ACT phase

        symbol = getattr(event, 'symbol', 'Unknown')
        order_id = getattr(event, 'order_id', 'N/A')

        self.frame.after(0, lambda: self._update_ui_order(symbol, order_id))

    def _update_ui_order(self, symbol, order_id):
        """Update UI from main thread."""
        self.event_stream.add_event('order', f"Order placed: {symbol} (ID: {order_id})")
        self.metrics_panel.update_metric('orders', str(self._order_count))

    def _on_fill_event(self, event):
        """Handle fill event."""
        self._fill_count += 1

        symbol = getattr(event, 'symbol', 'Unknown')
        qty = getattr(event, 'quantity', 0)
        price = getattr(event, 'price', 0)

        self.frame.after(0, lambda: self._update_ui_fill(symbol, qty, price))

    def _update_ui_fill(self, symbol, qty, price):
        """Update UI from main thread."""
        self.event_stream.add_event('fill', f"FILLED: {qty} {symbol} @ Rs.{price:.2f}")
        self.metrics_panel.update_metric('fills', str(self._fill_count))

    def _start_automation(self):
        """Start the automation."""
        self.is_running = True

        # Update UI
        self.status_label.config(text="RUNNING", fg=self.theme['success'])
        self.start_btn.config(state=tk.DISABLED, bg=self.theme['bg_secondary'])
        self.stop_btn.config(state=tk.NORMAL, bg=self.theme['danger'])

        # Activate phases
        self.phase_indicator.set_all_active(True)

        # Clear metrics
        self._tick_count = 0
        self._bar_count = 0
        self._signal_count = 0
        self._order_count = 0
        self._fill_count = 0

        # Clear event stream
        self.event_stream.clear()
        self.event_stream.add_event('bar', "Automation STARTED - waiting for market data...")

        # Update strategies display
        if self.trading_engine:
            strategies = {}
            for name in self.trading_engine.get_strategies():
                symbol = self.trading_engine._symbols.get(name, 'Unknown')
                strategies[name] = symbol
            self.strategy_status.update_strategies(strategies)

        # Call external handler
        if self.on_start:
            try:
                self.on_start()
            except Exception as e:
                logger.error(f"Start callback failed: {e}")
                self.event_stream.add_event('error', f"Start failed: {e}")

    def _stop_automation(self):
        """Stop the automation."""
        self.is_running = False

        # Update UI
        self.status_label.config(text="STOPPED", fg=self.theme['danger'])
        self.start_btn.config(state=tk.NORMAL, bg=self.theme['success'])
        self.stop_btn.config(state=tk.DISABLED, bg=self.theme['bg_secondary'])

        # Deactivate phases
        self.phase_indicator.set_all_active(False)

        # Log
        self.event_stream.add_event('bar', "Automation STOPPED")

        # Clear strategies
        self.strategy_status.update_strategies({})

        # Call external handler
        if self.on_stop:
            try:
                self.on_stop()
            except Exception as e:
                logger.error(f"Stop callback failed: {e}")

    def set_event_bus(self, event_bus):
        """Set the event bus (late binding)."""
        self.event_bus = event_bus
        self._subscribe_to_events()

    def set_trading_engine(self, engine):
        """Set the trading engine (late binding)."""
        self.trading_engine = engine

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("AUTOMATION PANEL - Test")
    print("=" * 50)

    root = tk.Tk()
    root.title("Automation Panel Test")
    root.geometry("1200x700")
    root.configure(bg='#1a1a2e')

    def on_start():
        print("Automation started!")

    def on_stop():
        print("Automation stopped!")

    panel = AutomationPanel(
        root, 'dark',
        on_start=on_start,
        on_stop=on_stop
    )
    panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Simulate some events
    def simulate_events():
        import time
        for i in range(100):
            time.sleep(0.5)
            panel.event_stream.add_event('tick', f"RELIANCE @ Rs.{2500 + i}")
            if i % 5 == 0:
                panel.event_stream.add_event('bar', f"RELIANCE bar closed @ Rs.{2500 + i}")
            if i % 20 == 0:
                panel.event_stream.add_event('signal', f"BUY signal for RELIANCE")

    # Start simulation in thread
    # threading.Thread(target=simulate_events, daemon=True).start()

    root.mainloop()

    print("\n" + "=" * 50)
    print("Automation Panel ready!")
    print("=" * 50)
