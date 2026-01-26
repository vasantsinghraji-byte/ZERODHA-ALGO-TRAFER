"""
Historical Data Source for Backtesting.

Converts historical OHLCV data to events for the event-driven backtester.
Supports multiple data sources:
- DataFrame (direct input)
- CSV files
- Pickle files
- DataManager (Zerodha API)

The key insight: Historical bars are replayed as BarEvents,
making the backtest engine use THE SAME event handlers as live trading.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import pandas as pd

from core.events import BarEvent, TickEvent, EventBus
from .source import (
    DataSource,
    DataSourceConfig,
    DataSourceMode,
    DataSourceState,
    dataframe_to_bar_events,
)


logger = logging.getLogger(__name__)


class HistoricalDataSource(DataSource):
    """
    Historical data source for backtesting.

    Replays historical OHLCV data as BarEvents, allowing the same
    strategy code to run in both backtest and live modes.

    Usage:
        # From DataFrame
        config = DataSourceConfig(symbols=['RELIANCE'], timeframe='1d')
        source = HistoricalDataSource(config, event_bus, data={'RELIANCE': df})
        source.start()

        # From CSV files
        source = HistoricalDataSource(config, event_bus, data_dir='data/historical')
        source.start()

        # From DataManager
        source = HistoricalDataSource(config, event_bus, data_manager=dm)
        source.start()
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        data_manager: Optional[object] = None,  # DataManager instance
    ):
        """
        Initialize historical data source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            data: Dictionary mapping symbol to DataFrame
            data_dir: Directory containing CSV/pickle files
            data_manager: DataManager instance for fetching from API
        """
        super().__init__(config, event_bus, DataSourceMode.BACKTEST)

        self._data = data or {}
        self._data_dir = Path(data_dir) if data_dir else None
        self._data_manager = data_manager

        # Merged data for multi-symbol replay
        self._merged_data: Optional[pd.DataFrame] = None

        # Progress tracking
        self._total_bars = 0
        self._current_bar = 0

    def connect(self) -> bool:
        """
        Load historical data.

        Returns:
            True if data loaded successfully
        """
        logger.info(f"Loading historical data for {self.config.symbols}")

        try:
            # Load data for each symbol
            for symbol in self.config.symbols:
                if symbol in self._data:
                    # Already have data
                    continue

                # Try loading from file
                if self._data_dir:
                    df = self._load_from_file(symbol)
                    if df is not None:
                        self._data[symbol] = df
                        continue

                # Try loading from DataManager
                if self._data_manager:
                    df = self._load_from_api(symbol)
                    if df is not None:
                        self._data[symbol] = df
                        continue

                logger.warning(f"No data found for {symbol}")

            if not self._data:
                logger.error("No data loaded for any symbol")
                return False

            # Filter data to date range
            self._filter_date_range()

            # Merge data for multi-symbol replay
            self._merge_data()

            self._total_bars = len(self._merged_data) if self._merged_data is not None else 0
            logger.info(f"Loaded {self._total_bars} bars for {len(self._data)} symbols")

            return True

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}", exc_info=True)
            return False

    def disconnect(self):
        """Clear loaded data."""
        self._merged_data = None
        self._current_bar = 0

    def _emit_events(self) -> Generator[Union[BarEvent, TickEvent], None, None]:
        """
        Generate BarEvents from historical data.

        Yields bars in chronological order, handling multiple symbols.
        """
        if self._merged_data is None or self._merged_data.empty:
            logger.warning("No data to emit")
            return

        # Skip warmup period
        start_idx = self.config.warmup_bars
        if start_idx >= len(self._merged_data):
            start_idx = 0

        logger.info(f"Starting replay from bar {start_idx} (warmup={self.config.warmup_bars})")

        last_time = time.time()

        for idx in range(start_idx, len(self._merged_data)):
            if not self._running:
                break

            row = self._merged_data.iloc[idx]
            timestamp = self._merged_data.index[idx]

            # Emit bar for each symbol in this row
            for symbol in self.config.symbols:
                if symbol not in self._data:
                    continue

                # Get OHLCV for this symbol at this timestamp
                try:
                    symbol_data = self._get_bar_data(symbol, timestamp, idx)
                    if symbol_data is None:
                        continue

                    bar_event = BarEvent(
                        symbol=symbol,
                        timestamp=timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp),
                        timeframe=self.config.timeframe,
                        open=symbol_data['open'],
                        high=symbol_data['high'],
                        low=symbol_data['low'],
                        close=symbol_data['close'],
                        volume=symbol_data['volume'],
                        bar_index=idx,
                        source="historical"
                    )

                    yield bar_event

                except Exception as e:
                    logger.error(f"Error creating bar event for {symbol}: {e}")
                    continue

            self._current_bar = idx

            # Apply speed multiplier delay
            if self.config.speed_multiplier > 0:
                sleep_time = 1.0 / self.config.speed_multiplier
                elapsed = time.time() - last_time
                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)
                last_time = time.time()

    def _load_from_file(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from CSV or pickle file."""
        if not self._data_dir:
            return None

        # Try different file formats
        for ext in ['.csv', '.pkl', '.pickle', '.parquet']:
            file_path = self._data_dir / f"{symbol}{ext}"
            if file_path.exists():
                try:
                    if ext == '.csv':
                        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                    elif ext in ['.pkl', '.pickle']:
                        df = pd.read_pickle(file_path)
                    elif ext == '.parquet':
                        df = pd.read_parquet(file_path)
                    else:
                        continue

                    logger.info(f"Loaded {len(df)} bars from {file_path}")
                    return df

                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        return None

    def _load_from_api(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from DataManager (Zerodha API)."""
        if not self._data_manager:
            return None

        try:
            # Get instrument token for symbol
            # This assumes DataManager has a method to get historical data by symbol
            if hasattr(self._data_manager, 'get_historical_data_by_symbol'):
                df = self._data_manager.get_historical_data_by_symbol(
                    symbol,
                    self.config.start_date,
                    self.config.end_date,
                    self._timeframe_to_interval(self.config.timeframe)
                )
                return df

            logger.warning(f"DataManager does not support get_historical_data_by_symbol")
            return None

        except Exception as e:
            logger.error(f"Failed to load {symbol} from API: {e}")
            return None

    def _filter_date_range(self):
        """Filter data to configured date range."""
        for symbol, df in self._data.items():
            if df.empty:
                continue

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)

            # Filter by date range
            mask = pd.Series(True, index=df.index)
            if self.config.start_date:
                mask &= df.index >= pd.to_datetime(self.config.start_date)
            if self.config.end_date:
                mask &= df.index <= pd.to_datetime(self.config.end_date)

            self._data[symbol] = df[mask]
            logger.debug(f"{symbol}: {len(self._data[symbol])} bars after date filter")

    def _merge_data(self):
        """Merge data from multiple symbols for synchronized replay."""
        if len(self._data) == 0:
            self._merged_data = pd.DataFrame()
            return

        if len(self._data) == 1:
            # Single symbol - use directly
            symbol = list(self._data.keys())[0]
            self._merged_data = self._data[symbol]
            return

        # Multiple symbols - merge on timestamp
        # Use union of all timestamps
        all_indices = set()
        for df in self._data.values():
            if not df.empty:
                all_indices.update(df.index.tolist())

        if not all_indices:
            self._merged_data = pd.DataFrame()
            return

        all_indices = sorted(all_indices)

        # Create merged DataFrame with a dummy column so it's not empty
        self._merged_data = pd.DataFrame({'_idx': range(len(all_indices))}, index=all_indices)

        for symbol, df in self._data.items():
            # Reindex to common timeline
            self._data[symbol] = df.reindex(all_indices, method='ffill')

    def _get_bar_data(self, symbol: str, timestamp: datetime, idx: int) -> Optional[dict]:
        """Get OHLCV data for a symbol at a timestamp."""
        if symbol not in self._data:
            return None

        df = self._data[symbol]

        # Try to get data at this timestamp
        if timestamp in df.index:
            row = df.loc[timestamp]
        elif idx < len(df):
            row = df.iloc[idx]
        else:
            return None

        # Check for NaN values (no data at this time)
        if pd.isna(row.get('close', row.get('Close', None))):
            return None

        # Normalize column names
        return {
            'open': float(row.get('open', row.get('Open', 0))),
            'high': float(row.get('high', row.get('High', 0))),
            'low': float(row.get('low', row.get('Low', 0))),
            'close': float(row.get('close', row.get('Close', 0))),
            'volume': int(row.get('volume', row.get('Volume', 0)))
        }

    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to Zerodha interval."""
        mapping = {
            '1m': 'minute',
            '5m': '5minute',
            '15m': '15minute',
            '30m': '30minute',
            '1h': '60minute',
            '1d': 'day',
        }
        return mapping.get(timeframe, 'day')

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get loaded DataFrame for a symbol."""
        return self._data.get(symbol)

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded DataFrames."""
        return self._data.copy()

    @property
    def progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        if self._total_bars == 0:
            return 0.0
        return self._current_bar / self._total_bars

    def set_data(self, symbol: str, df: pd.DataFrame):
        """Set data for a symbol."""
        self._data[symbol] = df

    def add_data(self, data: Dict[str, pd.DataFrame]):
        """Add data for multiple symbols."""
        self._data.update(data)


class MultiTimeframeHistoricalSource(HistoricalDataSource):
    """
    Historical data source supporting multiple timeframes.

    Useful for strategies that use multiple timeframes
    (e.g., trend from daily, entry from 5-minute).
    """

    def __init__(
        self,
        config: DataSourceConfig,
        event_bus: Optional[EventBus] = None,
        data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,  # symbol -> {timeframe -> df}
        **kwargs
    ):
        """
        Initialize multi-timeframe source.

        Args:
            config: Data source configuration
            event_bus: EventBus to emit events to
            data: Nested dict: symbol -> timeframe -> DataFrame
        """
        super().__init__(config, event_bus, **kwargs)
        self._mtf_data = data or {}

    def get_data_for_timeframe(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for specific symbol and timeframe."""
        if symbol in self._mtf_data:
            return self._mtf_data[symbol].get(timeframe)
        return None
