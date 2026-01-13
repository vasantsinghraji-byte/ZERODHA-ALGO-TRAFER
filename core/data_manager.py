"""
Data Manager Module

Handles all data fetching, caching, and management operations.
Separates data concerns from UI and business logic.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from kiteconnect import KiteConnect


class DataCache:
    """Simple in-memory cache with TTL"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    return entry['data']
                else:
                    del self.cache[key]
        return None

    def set(self, key: str, data: Any):
        """Set cache value"""
        with self._lock:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }

    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()


class DataManager:
    """
    Manages all data fetching and caching operations.

    Responsibilities:
    - Historical data fetching
    - Quote data fetching
    - Data caching
    - Rate limiting
    - Batch operations
    """

    def __init__(self, kite_client: Optional[KiteConnect] = None):
        self.kite = kite_client
        self.cache = DataCache(ttl_seconds=60)  # 1 minute cache for quotes
        self.historical_cache = DataCache(ttl_seconds=3600)  # 1 hour cache for historical
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    def set_kite_client(self, kite_client: KiteConnect):
        """Set or update Kite client"""
        self.kite = kite_client

    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()

    def get_quote(self, instruments: List[str], use_cache: bool = True) -> Dict:
        """
        Get quotes for instruments with caching.

        Args:
            instruments: List of instrument tokens or symbols
            use_cache: Whether to use cached data

        Returns:
            Dictionary of quotes
        """
        if not self.kite:
            raise ValueError("Kite client not initialized")

        cache_key = f"quote_{','.join(map(str, instruments))}"

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        self._rate_limit()
        quotes = self.kite.quote(instruments)
        self.cache.set(cache_key, quotes)

        return quotes

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data with caching.

        Args:
            instrument_token: Instrument token
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, day, etc.)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if not self.kite:
            raise ValueError("Kite client not initialized")

        cache_key = f"hist_{instrument_token}_{from_date.date()}_{to_date.date()}_{interval}"

        if use_cache:
            cached = self.historical_cache.get(cache_key)
            if cached is not None:
                return cached

        self._rate_limit()
        data = self.kite.historical_data(
            instrument_token,
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
            interval
        )

        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            self.historical_cache.set(cache_key, df)
            return df

        return pd.DataFrame()

    def get_historical_data_batch(
        self,
        instrument_tokens: List[int],
        days_back: int = 365,
        interval: str = "day"
    ) -> Dict[int, pd.DataFrame]:
        """
        Fetch historical data for multiple instruments.

        Args:
            instrument_tokens: List of instrument tokens
            days_back: Number of days of history
            interval: Candle interval

        Returns:
            Dictionary mapping token to DataFrame
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        results = {}
        for token in instrument_tokens:
            try:
                df = self.get_historical_data(token, start_date, end_date, interval)
                if not df.empty:
                    results[token] = df
            except Exception as e:
                print(f"[ERROR] Failed to fetch data for token {token}: {e}")
                continue

        return results

    def get_ltp(self, instruments: List[str]) -> Dict[str, float]:
        """
        Get Last Traded Price for instruments.

        Args:
            instruments: List of instrument tokens or symbols

        Returns:
            Dictionary mapping instrument to LTP
        """
        quotes = self.get_quote(instruments)
        return {
            instrument: data.get('last_price', 0)
            for instrument, data in quotes.items()
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.historical_cache.clear()

    def get_ohlc(self, instruments: List[str]) -> Dict:
        """
        Get OHLC data for instruments.

        Args:
            instruments: List of instrument tokens or symbols

        Returns:
            Dictionary with OHLC data
        """
        quotes = self.get_quote(instruments)
        return {
            instrument: {
                'open': data.get('ohlc', {}).get('open', 0),
                'high': data.get('ohlc', {}).get('high', 0),
                'low': data.get('ohlc', {}).get('low', 0),
                'close': data.get('ohlc', {}).get('close', 0),
                'volume': data.get('volume', 0)
            }
            for instrument, data in quotes.items()
        }
