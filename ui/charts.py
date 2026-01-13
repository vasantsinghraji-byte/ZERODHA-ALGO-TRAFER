# -*- coding: utf-8 -*-
"""
Charts Module - See Your Stocks in Pictures!
=============================================
Creates beautiful charts to visualize price data.

Like drawing pictures of how prices move up and down!
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")


# ============== COLORS ==============

class ChartColors:
    """Chart color scheme"""
    BACKGROUND = "#1e1e1e"
    GRID = "#333333"
    TEXT = "#ffffff"
    GREEN = "#00ff88"
    RED = "#ff4444"
    BLUE = "#4488ff"
    YELLOW = "#ffaa00"
    PURPLE = "#aa44ff"


# ============== SIMPLE LINE CHART ==============

class SimpleChart:
    """
    Simple line chart for quick price visualization.

    Usage:
        chart = SimpleChart(parent_frame)
        chart.plot(prices, "RELIANCE")
    """

    def __init__(self, parent: tk.Widget, width: int = 600, height: int = 400):
        """
        Initialize chart.

        Args:
            parent: Parent Tkinter widget
            width: Chart width
            height: Chart height
        """
        self.parent = parent
        self.width = width
        self.height = height

        if not HAS_MATPLOTLIB:
            self._create_fallback()
            return

        # Create figure
        self.fig = Figure(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor(ChartColors.BACKGROUND)

        self.ax = self.fig.add_subplot(111)
        self._style_axis()

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_fallback(self):
        """Create fallback when matplotlib not available"""
        label = ttk.Label(
            self.parent,
            text="Charts require matplotlib\nInstall: pip install matplotlib",
            font=("Arial", 14)
        )
        label.pack(expand=True)

    def _style_axis(self):
        """Apply dark theme to axis"""
        self.ax.set_facecolor(ChartColors.BACKGROUND)
        self.ax.tick_params(colors=ChartColors.TEXT)
        self.ax.spines['bottom'].set_color(ChartColors.GRID)
        self.ax.spines['top'].set_color(ChartColors.GRID)
        self.ax.spines['left'].set_color(ChartColors.GRID)
        self.ax.spines['right'].set_color(ChartColors.GRID)
        self.ax.xaxis.label.set_color(ChartColors.TEXT)
        self.ax.yaxis.label.set_color(ChartColors.TEXT)
        self.ax.title.set_color(ChartColors.TEXT)
        self.ax.grid(True, color=ChartColors.GRID, alpha=0.3)

    def plot(
        self,
        data: pd.DataFrame,
        title: str = "Price Chart",
        column: str = "close"
    ):
        """
        Plot price data.

        Args:
            data: DataFrame with price data
            title: Chart title
            column: Column to plot (default: close)
        """
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()
        self._style_axis()

        # Get column (handle different naming conventions)
        col = column if column in data.columns else column.capitalize()
        if col not in data.columns:
            col = data.columns[0]

        prices = data[col]

        # Determine color based on trend
        color = ChartColors.GREEN if prices.iloc[-1] >= prices.iloc[0] else ChartColors.RED

        # Plot
        self.ax.plot(data.index, prices, color=color, linewidth=2)
        self.ax.fill_between(data.index, prices, alpha=0.1, color=color)

        # Labels
        self.ax.set_title(title, fontsize=14, color=ChartColors.TEXT)
        self.ax.set_xlabel("Date", fontsize=10)
        self.ax.set_ylabel("Price (Rs.)", fontsize=10)

        # Format dates
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        self.fig.autofmt_xdate()

        self.canvas.draw()

    def clear(self):
        """Clear the chart"""
        if HAS_MATPLOTLIB:
            self.ax.clear()
            self._style_axis()
            self.canvas.draw()


# ============== CANDLESTICK CHART ==============

class CandlestickChart:
    """
    Candlestick chart for OHLC data.

    Shows open, high, low, close as candles.
    Green candle = price went up
    Red candle = price went down
    """

    def __init__(self, parent: tk.Widget, width: int = 800, height: int = 500):
        """
        Initialize candlestick chart.

        Args:
            parent: Parent Tkinter widget
            width: Chart width
            height: Chart height
        """
        self.parent = parent
        self.width = width
        self.height = height

        if not HAS_MATPLOTLIB:
            self._create_fallback()
            return

        # Create figure with subplots
        self.fig = Figure(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor(ChartColors.BACKGROUND)

        # Price axis (top 70%)
        self.ax_price = self.fig.add_axes([0.1, 0.35, 0.85, 0.6])
        # Volume axis (bottom 25%)
        self.ax_volume = self.fig.add_axes([0.1, 0.1, 0.85, 0.2])

        self._style_axes()

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_fallback(self):
        """Create fallback when matplotlib not available"""
        label = ttk.Label(
            self.parent,
            text="Candlestick charts require matplotlib\nInstall: pip install matplotlib",
            font=("Arial", 14)
        )
        label.pack(expand=True)

    def _style_axes(self):
        """Apply dark theme to both axes"""
        for ax in [self.ax_price, self.ax_volume]:
            ax.set_facecolor(ChartColors.BACKGROUND)
            ax.tick_params(colors=ChartColors.TEXT)
            ax.spines['bottom'].set_color(ChartColors.GRID)
            ax.spines['top'].set_color(ChartColors.GRID)
            ax.spines['left'].set_color(ChartColors.GRID)
            ax.spines['right'].set_color(ChartColors.GRID)
            ax.grid(True, color=ChartColors.GRID, alpha=0.3)

    def plot(
        self,
        data: pd.DataFrame,
        title: str = "Candlestick Chart"
    ):
        """
        Plot OHLC data as candlesticks.

        Args:
            data: DataFrame with open, high, low, close, volume columns
            title: Chart title
        """
        if not HAS_MATPLOTLIB:
            return

        self.ax_price.clear()
        self.ax_volume.clear()
        self._style_axes()

        # Normalize column names
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        # Get data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df.get('volume', pd.Series([0] * len(df))).values

        x = range(len(df))
        width = 0.6

        # Draw candlesticks
        for i in range(len(df)):
            # Determine color
            if closes[i] >= opens[i]:
                color = ChartColors.GREEN
                body_bottom = opens[i]
                body_height = closes[i] - opens[i]
            else:
                color = ChartColors.RED
                body_bottom = closes[i]
                body_height = opens[i] - closes[i]

            # Draw wick (high-low line)
            self.ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)

            # Draw body (rectangle)
            self.ax_price.bar(i, body_height, width, bottom=body_bottom,
                              color=color, edgecolor=color)

            # Draw volume bar
            vol_color = ChartColors.GREEN if closes[i] >= opens[i] else ChartColors.RED
            self.ax_volume.bar(i, volumes[i], width, color=vol_color, alpha=0.5)

        # Labels
        self.ax_price.set_title(title, fontsize=14, color=ChartColors.TEXT)
        self.ax_price.set_ylabel("Price (Rs.)", fontsize=10, color=ChartColors.TEXT)
        self.ax_volume.set_ylabel("Volume", fontsize=10, color=ChartColors.TEXT)
        self.ax_volume.set_xlabel("Date", fontsize=10, color=ChartColors.TEXT)

        # Set x-axis labels
        if len(df) > 0:
            dates = df.index
            step = max(1, len(dates) // 10)
            labels = [str(d)[:10] if hasattr(d, 'strftime') else str(d)[:10] for d in dates[::step]]
            self.ax_volume.set_xticks(range(0, len(dates), step))
            self.ax_volume.set_xticklabels(labels, rotation=45, fontsize=8, color=ChartColors.TEXT)
            self.ax_price.set_xticks([])

        self.canvas.draw()

    def clear(self):
        """Clear the chart"""
        if HAS_MATPLOTLIB:
            self.ax_price.clear()
            self.ax_volume.clear()
            self._style_axes()
            self.canvas.draw()


# ============== INDICATOR OVERLAYS ==============

def add_moving_average(
    ax,
    data: pd.DataFrame,
    period: int = 20,
    color: str = ChartColors.BLUE,
    label: str = None
):
    """
    Add moving average line to chart.

    Args:
        ax: Matplotlib axis
        data: Price data
        period: MA period
        color: Line color
        label: Legend label
    """
    if not HAS_MATPLOTLIB:
        return

    close_col = 'close' if 'close' in data.columns else 'Close'
    ma = data[close_col].rolling(window=period).mean()

    ax.plot(range(len(data)), ma, color=color, linewidth=1.5,
            label=label or f"MA{period}")


def add_bollinger_bands(
    ax,
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
):
    """
    Add Bollinger Bands to chart.

    Args:
        ax: Matplotlib axis
        data: Price data
        period: BB period
        std_dev: Standard deviation multiplier
    """
    if not HAS_MATPLOTLIB:
        return

    close_col = 'close' if 'close' in data.columns else 'Close'
    close = data[close_col]

    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)

    x = range(len(data))
    ax.plot(x, ma, color=ChartColors.YELLOW, linewidth=1, label="BB Mid")
    ax.plot(x, upper, color=ChartColors.PURPLE, linewidth=1, linestyle='--', label="BB Upper")
    ax.plot(x, lower, color=ChartColors.PURPLE, linewidth=1, linestyle='--', label="BB Lower")
    ax.fill_between(x, lower, upper, alpha=0.1, color=ChartColors.PURPLE)


# ============== CHART WINDOW ==============

class ChartWindow:
    """
    Standalone chart window.

    Opens a new window with price chart.
    """

    def __init__(self, data: pd.DataFrame, title: str = "Chart", chart_type: str = "line"):
        """
        Create chart window.

        Args:
            data: Price data
            title: Window title
            chart_type: "line" or "candle"
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib required for charts!")
            return

        # Create window
        self.window = tk.Toplevel()
        self.window.title(title)
        self.window.geometry("900x600")
        self.window.configure(bg=ChartColors.BACKGROUND)

        # Create chart
        if chart_type == "candle":
            self.chart = CandlestickChart(self.window, 900, 600)
        else:
            self.chart = SimpleChart(self.window, 900, 600)

        self.chart.plot(data, title)

    def show(self):
        """Show the window"""
        self.window.mainloop()


# ============== QUICK PLOT FUNCTIONS ==============

def quick_plot(data: pd.DataFrame, title: str = "Price Chart"):
    """
    Quick way to plot price data.

    Opens a new window with the chart.

    Args:
        data: DataFrame with price data
        title: Chart title
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required! Install: pip install matplotlib")
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("800x500")
    root.configure(bg=ChartColors.BACKGROUND)

    chart = SimpleChart(root, 800, 500)
    chart.plot(data, title)

    root.mainloop()


def quick_candle(data: pd.DataFrame, title: str = "Candlestick Chart"):
    """
    Quick way to plot candlestick chart.

    Args:
        data: DataFrame with OHLCV data
        title: Chart title
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required! Install: pip install matplotlib")
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("900x600")
    root.configure(bg=ChartColors.BACKGROUND)

    chart = CandlestickChart(root, 900, 600)
    chart.plot(data, title)

    root.mainloop()


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("CHARTS MODULE - Test")
    print("=" * 50)

    # Create sample data
    import numpy as np

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

    print("\nSample data created:")
    print(data.tail())

    if HAS_MATPLOTLIB:
        print("\nOpening chart window...")
        quick_candle(data, "RELIANCE - Sample Data")
    else:
        print("\nmatplotlib not installed. Skipping visual test.")
        print("Install with: pip install matplotlib")

    print("\n" + "=" * 50)
    print("Charts module ready!")
    print("=" * 50)
