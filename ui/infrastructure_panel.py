# -*- coding: utf-8 -*-
"""
Infrastructure Panel - System Health & Controls
================================================
Provides UI widgets for infrastructure monitoring and control.

Components:
- KillSwitchButton: Emergency stop button
- LatencyDisplay: Real-time latency stats
- RateLimitStatus: API rate limit status
- SystemHealthCard: Combined health overview
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Dict, Optional, Callable, Any
import logging
import threading

from .themes import get_theme

logger = logging.getLogger(__name__)


class KillSwitchButton:
    """
    Emergency Kill Switch Button.

    Big red button that immediately stops all trading activity.
    Always visible and accessible for safety.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        on_trigger: Callable = None,
        compact: bool = False
    ):
        self.theme = theme
        self.on_trigger = on_trigger
        self.triggered = False
        self.compact = compact

        self.frame = tk.Frame(parent, bg=theme['bg_primary'])

        if compact:
            self._create_compact()
        else:
            self._create_full()

    def _create_compact(self):
        """Create compact kill switch (for header)"""
        self.button = tk.Button(
            self.frame,
            text="STOP",
            bg='#dc3545',
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            activebackground='#c82333',
            activeforeground='white',
            command=self._on_click
        )
        self.button.pack(ipadx=8, ipady=2)

        # Hover effects
        self.button.bind('<Enter>', lambda e: self.button.config(bg='#c82333'))
        self.button.bind('<Leave>', lambda e: self.button.config(
            bg='#28a745' if self.triggered else '#dc3545'
        ))

    def _create_full(self):
        """Create full kill switch card"""
        card = tk.Frame(self.frame, bg=self.theme['bg_card'])
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        card.pack(fill=tk.X)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=15, pady=15)

        # Title
        tk.Label(
            inner,
            text="Emergency Stop",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W)

        tk.Label(
            inner,
            text="Cancel all orders & stop trading",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9)
        ).pack(anchor=tk.W, pady=(0, 10))

        # Big red button
        self.button = tk.Button(
            inner,
            text="KILL SWITCH",
            bg='#dc3545',
            fg='white',
            font=('Segoe UI', 14, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            activebackground='#c82333',
            activeforeground='white',
            command=self._on_click
        )
        self.button.pack(fill=tk.X, ipady=10)

        # Status label
        self.status_label = tk.Label(
            inner,
            text="System Normal",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 10)
        )
        self.status_label.pack(anchor=tk.W, pady=(10, 0))

        # Hover effects
        self.button.bind('<Enter>', lambda e: self.button.config(bg='#c82333'))
        self.button.bind('<Leave>', lambda e: self.button.config(
            bg='#28a745' if self.triggered else '#dc3545'
        ))

    def _on_click(self):
        """Handle kill switch click"""
        if self.triggered:
            # Reset
            if messagebox.askyesno(
                "Reset Kill Switch",
                "Resume trading operations?"
            ):
                self.reset()
        else:
            # Trigger
            if messagebox.askyesno(
                "EMERGENCY STOP",
                "This will:\n"
                "- Cancel ALL open orders\n"
                "- Stop the trading bot\n"
                "- Disable new order placement\n\n"
                "Are you sure?",
                icon='warning'
            ):
                self.trigger()

    def trigger(self, reason: str = "Manual trigger"):
        """Activate kill switch"""
        self.triggered = True
        self.button.config(text="RESET" if not self.compact else "RESET", bg='#28a745')

        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"STOPPED: {reason}",
                fg=self.theme['danger']
            )

        if self.on_trigger:
            self.on_trigger(triggered=True, reason=reason)

        logger.warning(f"Kill switch triggered: {reason}")

    def reset(self):
        """Reset kill switch"""
        self.triggered = False
        self.button.config(
            text="KILL SWITCH" if not self.compact else "STOP",
            bg='#dc3545'
        )

        if hasattr(self, 'status_label'):
            self.status_label.config(text="System Normal", fg=self.theme['success'])

        if self.on_trigger:
            self.on_trigger(triggered=False, reason="Manual reset")

        logger.info("Kill switch reset")

    def is_triggered(self) -> bool:
        return self.triggered

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class LatencyDisplay:
    """
    Real-time Latency Statistics Display.

    Shows p50, p95, p99 latency metrics with color-coded status.
    """

    def __init__(self, parent: tk.Widget, theme: dict, compact: bool = False):
        self.theme = theme
        self.compact = compact

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        if not compact:
            self.frame.configure(
                highlightbackground=theme['border'],
                highlightthickness=1
            )

        self._create_widgets()

    def _create_widgets(self):
        inner = tk.Frame(self.frame, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=10 if not self.compact else 5, pady=10 if not self.compact else 5)

        if not self.compact:
            # Title
            tk.Label(
                inner,
                text="Latency",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 11, 'bold')
            ).pack(anchor=tk.W, pady=(0, 8))

        # Metrics row
        metrics_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        metrics_frame.pack(fill=tk.X)

        self.metric_labels = {}
        metrics = [('p50', 'P50'), ('p95', 'P95'), ('p99', 'P99')]

        for key, label in metrics:
            col = tk.Frame(metrics_frame, bg=self.theme['bg_card'])
            col.pack(side=tk.LEFT, expand=True)

            tk.Label(
                col,
                text=label,
                bg=self.theme['bg_card'],
                fg=self.theme['text_dim'],
                font=('Segoe UI', 8)
            ).pack()

            value_label = tk.Label(
                col,
                text="--ms",
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 10 if not self.compact else 9, 'bold')
            )
            value_label.pack()
            self.metric_labels[key] = value_label

    def update_latency(self, p50: float, p95: float, p99: float):
        """Update latency display"""
        # Color thresholds (in ms)
        def get_color(val):
            if val < 50:
                return self.theme['success']
            elif val < 200:
                return self.theme['warning']
            else:
                return self.theme['danger']

        self.metric_labels['p50'].config(
            text=f"{p50:.0f}ms",
            fg=get_color(p50)
        )
        self.metric_labels['p95'].config(
            text=f"{p95:.0f}ms",
            fg=get_color(p95)
        )
        self.metric_labels['p99'].config(
            text=f"{p99:.0f}ms",
            fg=get_color(p99)
        )

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class RateLimitStatus:
    """
    API Rate Limit Status Display.

    Shows remaining requests and visual progress bar.
    """

    def __init__(self, parent: tk.Widget, theme: dict, compact: bool = False):
        self.theme = theme
        self.compact = compact

        self.frame = tk.Frame(parent, bg=theme['bg_card'])

        if not compact:
            self.frame.configure(
                highlightbackground=theme['border'],
                highlightthickness=1
            )

        self._create_widgets()

    def _create_widgets(self):
        inner = tk.Frame(self.frame, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=10 if not self.compact else 5, pady=10 if not self.compact else 5)

        if not self.compact:
            # Title
            header = tk.Frame(inner, bg=self.theme['bg_card'])
            header.pack(fill=tk.X, pady=(0, 8))

            tk.Label(
                header,
                text="API Limits",
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 11, 'bold')
            ).pack(side=tk.LEFT)

            self.status_label = tk.Label(
                header,
                text="OK",
                bg=self.theme['bg_card'],
                fg=self.theme['success'],
                font=('Segoe UI', 9)
            )
            self.status_label.pack(side=tk.RIGHT)
        else:
            self.status_label = None

        # Progress bar
        self.progress_frame = tk.Frame(inner, bg=self.theme['bg_secondary'], height=8)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        self.progress_frame.pack_propagate(False)

        self.progress_bar = tk.Frame(self.progress_frame, bg=self.theme['success'], width=0)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.Y)

        # Text
        self.text_label = tk.Label(
            inner,
            text="--/-- requests",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 9 if not self.compact else 8)
        )
        self.text_label.pack(anchor=tk.W)

    def update_status(self, used: int, limit: int, endpoint: str = ""):
        """Update rate limit display"""
        remaining = limit - used
        pct = (remaining / limit * 100) if limit > 0 else 0

        # Color based on remaining
        if pct > 50:
            color = self.theme['success']
            status = "OK"
        elif pct > 20:
            color = self.theme['warning']
            status = "Warning"
        else:
            color = self.theme['danger']
            status = "Critical"

        # Update progress bar
        bar_width = int(self.progress_frame.winfo_width() * (pct / 100))
        self.progress_bar.config(bg=color, width=max(bar_width, 1))

        # Update text
        self.text_label.config(text=f"{remaining}/{limit} remaining")

        if self.status_label:
            self.status_label.config(text=status, fg=color)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class TickFilterStats:
    """
    Tick Filter Statistics Display.

    Shows rejected ticks count and reasons.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        self._create_widgets()

    def _create_widgets(self):
        inner = tk.Frame(self.frame, bg=self.theme['bg_card'])
        inner.pack(fill=tk.X, padx=10, pady=10)

        # Title
        header = tk.Frame(inner, bg=self.theme['bg_card'])
        header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            header,
            text="Tick Filter",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 11, 'bold')
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header,
            text="Active",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 9)
        )
        self.status_label.pack(side=tk.RIGHT)

        # Stats
        stats_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        stats_frame.pack(fill=tk.X)

        self.stats_labels = {}
        stats = [
            ('accepted', 'Accepted', '0'),
            ('rejected', 'Rejected', '0'),
            ('spikes', 'Spikes', '0'),
        ]

        for key, label, default in stats:
            row = tk.Frame(stats_frame, bg=self.theme['bg_card'])
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=label,
                bg=self.theme['bg_card'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 9)
            ).pack(side=tk.LEFT)

            val_label = tk.Label(
                row,
                text=default,
                bg=self.theme['bg_card'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 9, 'bold')
            )
            val_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = val_label

    def update_stats(self, accepted: int, rejected: int, spikes: int = 0):
        """Update tick filter stats"""
        self.stats_labels['accepted'].config(text=f"{accepted:,}")
        self.stats_labels['rejected'].config(
            text=f"{rejected:,}",
            fg=self.theme['danger'] if rejected > 0 else self.theme['text_primary']
        )
        self.stats_labels['spikes'].config(
            text=f"{spikes:,}",
            fg=self.theme['warning'] if spikes > 0 else self.theme['text_primary']
        )

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class SystemHealthCard:
    """
    Combined System Health Overview Card.

    Shows overall system status including latency, rate limits, and tick filter.
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme

        self.frame = tk.Frame(parent, bg=theme['bg_card'])
        self.frame.configure(
            highlightbackground=theme['border'],
            highlightthickness=1
        )

        self._create_widgets()

    def _create_widgets(self):
        inner = tk.Frame(self.frame, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        header = tk.Frame(inner, bg=self.theme['bg_card'])
        header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header,
            text="System Health",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 12, 'bold')
        ).pack(side=tk.LEFT)

        self.overall_status = tk.Label(
            header,
            text="● Healthy",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 10)
        )
        self.overall_status.pack(side=tk.RIGHT)

        # Latency section
        latency_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        latency_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            latency_frame,
            text="Latency",
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT)

        self.latency_display = LatencyDisplay(latency_frame, self.theme, compact=True)
        self.latency_display.pack(side=tk.RIGHT)

        # Rate limit section
        rate_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        rate_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            rate_frame,
            text="API Rate",
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT)

        self.rate_bar = tk.Frame(rate_frame, bg=self.theme['bg_secondary'], width=100, height=8)
        self.rate_bar.pack(side=tk.RIGHT)
        self.rate_bar.pack_propagate(False)

        self.rate_fill = tk.Frame(self.rate_bar, bg=self.theme['success'], width=100)
        self.rate_fill.pack(side=tk.LEFT, fill=tk.Y)

        # Tick filter section
        tick_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        tick_frame.pack(fill=tk.X)

        tk.Label(
            tick_frame,
            text="Tick Filter",
            bg=self.theme['bg_card'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT)

        self.tick_status = tk.Label(
            tick_frame,
            text="0 rejected",
            bg=self.theme['bg_card'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        )
        self.tick_status.pack(side=tk.RIGHT)

    def update_latency(self, p50: float, p95: float, p99: float):
        """Update latency display"""
        self.latency_display.update_latency(p50, p95, p99)
        self._update_overall_status()

    def update_rate_limit(self, used: int, limit: int):
        """Update rate limit display"""
        pct = ((limit - used) / limit * 100) if limit > 0 else 100

        if pct > 50:
            color = self.theme['success']
        elif pct > 20:
            color = self.theme['warning']
        else:
            color = self.theme['danger']

        width = int(100 * (pct / 100))
        self.rate_fill.config(bg=color, width=max(width, 1))
        self._update_overall_status()

    def update_tick_filter(self, rejected: int):
        """Update tick filter display"""
        self.tick_status.config(
            text=f"{rejected:,} rejected",
            fg=self.theme['danger'] if rejected > 10 else self.theme['text_dim']
        )
        self._update_overall_status()

    def _update_overall_status(self):
        """Update overall health status"""
        # Could implement more sophisticated health calculation
        self.overall_status.config(text="● Healthy", fg=self.theme['success'])

    def set_status(self, status: str, healthy: bool):
        """Set overall status"""
        self.overall_status.config(
            text=f"● {status}",
            fg=self.theme['success'] if healthy else self.theme['danger']
        )

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class InfrastructureMonitor:
    """
    Background infrastructure monitoring thread.

    Periodically updates UI with infrastructure stats.
    """

    def __init__(
        self,
        update_callback: Callable[[Dict[str, Any]], None],
        interval_ms: int = 1000
    ):
        self.update_callback = update_callback
        self.interval_ms = interval_ms
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, root: tk.Tk):
        """Start monitoring"""
        self._running = True
        self._schedule_update(root)

    def stop(self):
        """Stop monitoring"""
        self._running = False

    def _schedule_update(self, root: tk.Tk):
        """Schedule next update"""
        if self._running:
            self._do_update()
            root.after(self.interval_ms, lambda: self._schedule_update(root))

    def _do_update(self):
        """Perform update"""
        try:
            stats = self._collect_stats()
            self.update_callback(stats)
        except Exception as e:
            logger.error(f"Infrastructure monitoring error: {e}")

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect infrastructure statistics"""
        stats = {
            'latency': {'p50': 0, 'p95': 0, 'p99': 0},
            'rate_limit': {'used': 0, 'limit': 100},
            'tick_filter': {'accepted': 0, 'rejected': 0, 'spikes': 0},
            'kill_switch': {'triggered': False}
        }

        try:
            # Try to get latency stats
            from core.infrastructure import get_latency_monitor, LatencyType
            monitor = get_latency_monitor()
            latency_stats = monitor.get_stats(LatencyType.TICK_TO_ORDER)
            if latency_stats:
                stats['latency'] = {
                    'p50': latency_stats.p50,
                    'p95': latency_stats.p95,
                    'p99': latency_stats.p99
                }
        except Exception:
            pass

        try:
            # Try to get kill switch status
            from core.infrastructure import get_kill_switch
            ks = get_kill_switch()
            stats['kill_switch']['triggered'] = ks.is_triggered()
        except Exception:
            pass

        try:
            # Try to get tick filter stats
            from core.data import get_tick_filter
            tf = get_tick_filter()
            tf_stats = tf.get_stats()
            stats['tick_filter'] = {
                'accepted': tf_stats.accepted_count,
                'rejected': tf_stats.rejected_count,
                'spikes': tf_stats.spike_count
            }
        except Exception:
            pass

        return stats


class InfrastructureStatusPanel:
    """
    Comprehensive Infrastructure Status Panel.

    Displays status of all infrastructure components:
    - Flight Recorder (market replay)
    - Shadow Engine (paper trading validation)
    - A/B Testing (strategy comparison)
    - Audit Trail (compliance logging)
    - Risk Compliance (SEBI regulations)
    - Kill Switch (emergency stop)
    - Latency Monitor (performance)
    """

    def __init__(self, parent: tk.Widget, theme: dict):
        self.theme = theme
        self.frame = tk.Frame(parent, bg=theme['bg_primary'])
        self._status_labels = {}
        self._create_widgets()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main card
        card = tk.Frame(self.frame, bg=self.theme['bg_card'])
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        inner = tk.Frame(card, bg=self.theme['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Title
        header = tk.Frame(inner, bg=self.theme['bg_card'])
        header.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            header,
            text="Infrastructure Status",
            bg=self.theme['bg_card'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 14, 'bold')
        ).pack(side=tk.LEFT)

        self.overall_status = tk.Label(
            header,
            text="● Running",
            bg=self.theme['bg_card'],
            fg=self.theme['success'],
            font=('Segoe UI', 11)
        )
        self.overall_status.pack(side=tk.RIGHT)

        # Components grid (2 columns)
        grid_frame = tk.Frame(inner, bg=self.theme['bg_card'])
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # Left column
        left_col = tk.Frame(grid_frame, bg=self.theme['bg_card'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right column
        right_col = tk.Frame(grid_frame, bg=self.theme['bg_card'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Components
        self._create_component_card(left_col, "flight_recorder", "Flight Recorder", {
            "recording": "Recording",
            "tick_count": "Ticks"
        })

        self._create_component_card(left_col, "shadow_engine", "Shadow Engine", {
            "strategies": "Strategies",
            "total_pnl": "Shadow P&L"
        })

        self._create_component_card(left_col, "ab_testing", "A/B Testing", {
            "active_tests": "Active Tests"
        })

        self._create_component_card(right_col, "audit_trail", "Audit Trail", {
            "record_count": "Records",
            "integrity_valid": "Integrity"
        })

        self._create_component_card(right_col, "compliance", "Compliance", {
            "checks_performed": "Checks",
            "violations": "Violations"
        })

        self._create_component_card(right_col, "kill_switch", "Kill Switch", {
            "triggered": "Status",
            "current_pnl": "P&L"
        })

    def _create_component_card(
        self,
        parent: tk.Widget,
        key: str,
        title: str,
        fields: Dict[str, str]
    ):
        """Create a component status card."""
        card = tk.Frame(parent, bg=self.theme['bg_secondary'])
        card.configure(highlightbackground=self.theme['border'], highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 10))

        inner = tk.Frame(card, bg=self.theme['bg_secondary'])
        inner.pack(fill=tk.X, padx=10, pady=8)

        # Header with title and status
        header = tk.Frame(inner, bg=self.theme['bg_secondary'])
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text=title,
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side=tk.LEFT)

        status_label = tk.Label(
            header,
            text="●",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_dim'],
            font=('Segoe UI', 10)
        )
        status_label.pack(side=tk.RIGHT)
        self._status_labels[f"{key}_enabled"] = status_label

        # Fields
        self._status_labels[key] = {}
        for field_key, field_label in fields.items():
            row = tk.Frame(inner, bg=self.theme['bg_secondary'])
            row.pack(fill=tk.X, pady=2)

            tk.Label(
                row,
                text=field_label,
                bg=self.theme['bg_secondary'],
                fg=self.theme['text_secondary'],
                font=('Segoe UI', 9)
            ).pack(side=tk.LEFT)

            value_label = tk.Label(
                row,
                text="--",
                bg=self.theme['bg_secondary'],
                fg=self.theme['text_primary'],
                font=('Segoe UI', 9)
            )
            value_label.pack(side=tk.RIGHT)
            self._status_labels[key][field_key] = value_label

    def update_status(self, status: Dict[str, Any]):
        """
        Update the panel with infrastructure status.

        Args:
            status: Dict from InfrastructureManager.get_status()
        """
        # Overall status
        infra_status = status.get('status', 'unknown')
        if infra_status == 'running':
            self.overall_status.config(text="● Running", fg=self.theme['success'])
        elif infra_status == 'stopped':
            self.overall_status.config(text="● Stopped", fg=self.theme['text_dim'])
        elif infra_status == 'error':
            self.overall_status.config(text="● Error", fg=self.theme['danger'])
        else:
            self.overall_status.config(text=f"● {infra_status.title()}", fg=self.theme['warning'])

        # Flight Recorder
        fr = status.get('flight_recorder', {})
        self._update_component_status('flight_recorder', fr.get('enabled', False))
        if 'flight_recorder' in self._status_labels:
            labels = self._status_labels['flight_recorder']
            labels['recording'].config(
                text="Yes" if fr.get('recording') else "No",
                fg=self.theme['success'] if fr.get('recording') else self.theme['text_dim']
            )
            labels['tick_count'].config(text=f"{fr.get('tick_count', 0):,}")

        # Shadow Engine
        se = status.get('shadow_engine', {})
        self._update_component_status('shadow_engine', se.get('enabled', False))
        if 'shadow_engine' in self._status_labels:
            labels = self._status_labels['shadow_engine']
            labels['strategies'].config(text=str(se.get('strategies', 0)))
            pnl = se.get('total_pnl', 0)
            labels['total_pnl'].config(
                text=f"Rs.{pnl:+,.0f}",
                fg=self.theme['success'] if pnl >= 0 else self.theme['danger']
            )

        # A/B Testing
        ab = status.get('ab_testing', {})
        self._update_component_status('ab_testing', ab.get('enabled', False))
        if 'ab_testing' in self._status_labels:
            labels = self._status_labels['ab_testing']
            labels['active_tests'].config(text=str(ab.get('active_tests', 0)))

        # Audit Trail
        at = status.get('audit_trail', {})
        self._update_component_status('audit_trail', at.get('enabled', False))
        if 'audit_trail' in self._status_labels:
            labels = self._status_labels['audit_trail']
            labels['record_count'].config(text=f"{at.get('record_count', 0):,}")
            integrity = at.get('integrity_valid', None)
            if integrity is True:
                labels['integrity_valid'].config(text="Valid", fg=self.theme['success'])
            elif integrity is False:
                labels['integrity_valid'].config(text="INVALID", fg=self.theme['danger'])
            else:
                labels['integrity_valid'].config(text="--", fg=self.theme['text_dim'])

        # Compliance
        comp = status.get('compliance', {})
        self._update_component_status('compliance', comp.get('enabled', False))
        if 'compliance' in self._status_labels:
            labels = self._status_labels['compliance']
            labels['checks_performed'].config(text=f"{comp.get('checks_performed', 0):,}")
            violations = comp.get('violations', 0)
            labels['violations'].config(
                text=str(violations),
                fg=self.theme['danger'] if violations > 0 else self.theme['success']
            )

        # Kill Switch
        ks = status.get('kill_switch', {})
        self._update_component_status('kill_switch', ks.get('enabled', False))
        if 'kill_switch' in self._status_labels:
            labels = self._status_labels['kill_switch']
            triggered = ks.get('triggered', False)
            labels['triggered'].config(
                text="TRIGGERED" if triggered else "Armed",
                fg=self.theme['danger'] if triggered else self.theme['success']
            )
            pnl = ks.get('current_pnl', 0)
            labels['current_pnl'].config(
                text=f"Rs.{pnl:+,.0f}",
                fg=self.theme['success'] if pnl >= 0 else self.theme['danger']
            )

    def _update_component_status(self, key: str, enabled: bool):
        """Update component enabled status indicator."""
        label_key = f"{key}_enabled"
        if label_key in self._status_labels:
            self._status_labels[label_key].config(
                text="● Enabled" if enabled else "● Disabled",
                fg=self.theme['success'] if enabled else self.theme['text_dim']
            )

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class InfrastructureManagerWidget:
    """
    Widget for managing InfrastructureManager from UI.

    Provides start/stop controls and real-time status updates.
    """

    def __init__(
        self,
        parent: tk.Widget,
        theme: dict,
        infrastructure_manager=None
    ):
        self.theme = theme
        self._manager = infrastructure_manager
        self.frame = tk.Frame(parent, bg=theme['bg_primary'])

        self._create_widgets()
        self._update_interval = 1000  # ms

    def _create_widgets(self):
        """Create UI widgets."""
        # Status panel
        self.status_panel = InfrastructureStatusPanel(self.frame, self.theme)
        self.status_panel.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        controls = tk.Frame(self.frame, bg=self.theme['bg_primary'])
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.start_btn = tk.Button(
            controls,
            text="Start Infrastructure",
            bg=self.theme['success'],
            fg='white',
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._on_start
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = tk.Button(
            controls,
            text="Stop Infrastructure",
            bg=self.theme['danger'],
            fg='white',
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            command=self._on_stop
        )
        self.stop_btn.pack(side=tk.LEFT)

        self.refresh_btn = tk.Button(
            controls,
            text="Refresh",
            bg=self.theme['bg_secondary'],
            fg=self.theme['text_primary'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            cursor='hand2',
            command=self._refresh_status
        )
        self.refresh_btn.pack(side=tk.RIGHT)

    def set_manager(self, manager):
        """Set the infrastructure manager instance."""
        self._manager = manager
        self._refresh_status()

    def _on_start(self):
        """Handle start button click."""
        if self._manager:
            if self._manager.status.value == 'uninitialized':
                self._manager.initialize()
            self._manager.start()
            self._refresh_status()
            self._update_buttons()

    def _on_stop(self):
        """Handle stop button click."""
        if self._manager:
            self._manager.stop()
            self._refresh_status()
            self._update_buttons()

    def _refresh_status(self):
        """Refresh status display."""
        if self._manager:
            status = self._manager.get_status()
            self.status_panel.update_status(status)
            self._update_buttons()

    def _update_buttons(self):
        """Update button states based on manager status."""
        if self._manager:
            status = self._manager.status.value
            if status == 'running':
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
            else:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)

    def start_auto_refresh(self, root: tk.Tk):
        """Start automatic status refresh."""
        def refresh():
            self._refresh_status()
            root.after(self._update_interval, refresh)
        refresh()

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("INFRASTRUCTURE PANEL - Test")
    print("=" * 50)

    root = tk.Tk()
    root.title("Infrastructure Panel Test")
    root.geometry("400x600")
    root.configure(bg='#1a1a2e')

    theme = get_theme('dark')

    # Kill switch
    def on_kill_switch(triggered, reason):
        print(f"Kill switch: triggered={triggered}, reason={reason}")

    ks = KillSwitchButton(root, theme, on_trigger=on_kill_switch)
    ks.pack(fill=tk.X, padx=20, pady=10)

    # Latency display
    latency = LatencyDisplay(root, theme)
    latency.pack(fill=tk.X, padx=20, pady=10)
    latency.update_latency(25, 85, 150)

    # Rate limit
    rate = RateLimitStatus(root, theme)
    rate.pack(fill=tk.X, padx=20, pady=10)
    root.after(100, lambda: rate.update_status(30, 100))

    # Tick filter
    tick = TickFilterStats(root, theme)
    tick.pack(fill=tk.X, padx=20, pady=10)
    tick.update_stats(10000, 15, 3)

    # System health card
    health = SystemHealthCard(root, theme)
    health.pack(fill=tk.X, padx=20, pady=10)
    health.update_latency(30, 90, 180)
    root.after(100, lambda: health.update_rate_limit(25, 100))
    health.update_tick_filter(5)

    root.mainloop()

    print("\n" + "=" * 50)
    print("Infrastructure Panel ready!")
    print("=" * 50)
