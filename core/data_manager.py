# -*- coding: utf-8 -*-
"""
Data Manager Module - Your Market Data Helper!
==============================================
Downloads, stores, and retrieves stock price data.

Think of it like a library that stores all the price history
so you can look back and see what happened!
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from kiteconnect import KiteConnect

# Configure logger
logger = logging.getLogger(__name__)

# Import our database
from utils.database import get_db, db_session
# BRITTLE PATH FIX: Use robust path resolution
from utils.paths import find_project_root

# Data storage paths - using robust path resolution
BASE_DIR = find_project_root()
DATA_DIR = BASE_DIR / "data" / "historical"
CACHE_DIR = BASE_DIR / "data" / "cache"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Symbol validation regex - allows only safe characters
import re
_VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9][A-Z0-9_\-&]{0,29}$')


class TokenBucketRateLimiter:
    """
    Non-blocking token bucket rate limiter.

    Unlike time.sleep()-based rate limiting, this implementation:
    - Does NOT block the calling thread by default
    - Is thread-safe
    - Supports multiple modes: 'block', 'reject', 'warn'
    - Logs rate limit events for monitoring

    Token Bucket Algorithm:
    - Tokens are added at a fixed rate (refill_rate per second)
    - Each request consumes one token
    - If no tokens available, request is handled according to mode

    Args:
        requests_per_second: Maximum requests allowed per second
        burst_size: Maximum tokens that can accumulate (for burst handling)
        mode: How to handle rate limit hits:
            - 'block': Sleep until token available (legacy behavior)
            - 'reject': Raise RateLimitExceeded exception
            - 'warn': Log warning and proceed anyway (for monitoring)
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 10,
        mode: str = 'warn'
    ):
        self.refill_rate = requests_per_second
        self.burst_size = burst_size
        self.mode = mode

        self._tokens = float(burst_size)
        self._last_refill = time.time()
        self._lock = threading.Lock()

        # Stats for monitoring
        self._total_requests = 0
        self._rate_limited_requests = 0

    def _refill_tokens(self):
        """Add tokens based on elapsed time (called while holding lock)."""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.burst_size, self._tokens + tokens_to_add)
        self._last_refill = now

    def acquire(self, block: bool = None) -> bool:
        """
        Attempt to acquire a token for making a request.

        Args:
            block: Override the default mode. If True, block until token available.
                   If False, return immediately. If None, use instance mode.

        Returns:
            True if token acquired, False if rejected/rate-limited.

        Raises:
            RateLimitExceeded: If mode is 'reject' and no token available.
        """
        # Determine blocking behavior
        should_block = block if block is not None else (self.mode == 'block')

        with self._lock:
            self._total_requests += 1
            self._refill_tokens()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            # No token available - handle according to mode
            self._rate_limited_requests += 1

            if should_block:
                # Calculate wait time and sleep
                wait_time = (1.0 - self._tokens) / self.refill_rate
                # Release lock during sleep to allow other operations
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()
                # After sleeping, refill and consume
                self._refill_tokens()
                self._tokens -= 1.0
                return True

            elif self.mode == 'reject':
                raise RateLimitExceeded(
                    f"Rate limit exceeded. {self._rate_limited_requests} requests "
                    f"rate-limited out of {self._total_requests} total."
                )

            else:  # 'warn' mode
                logger.warning(
                    f"Rate limit hit (non-blocking). "
                    f"Stats: {self._rate_limited_requests}/{self._total_requests} rate-limited. "
                    f"Tokens: {self._tokens:.2f}/{self.burst_size}"
                )
                return True  # Proceed anyway but log the warning

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics for monitoring."""
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'rate_limited_requests': self._rate_limited_requests,
                'current_tokens': self._tokens,
                'burst_size': self.burst_size,
                'requests_per_second': self.refill_rate,
            }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded in 'reject' mode."""
    pass


def validate_symbol(symbol: str) -> str:
    """
    Validate and sanitize stock symbol to prevent injection attacks.

    Stock symbols should only contain:
    - Uppercase letters (A-Z)
    - Numbers (0-9)
    - Hyphen (-), underscore (_), ampersand (&)
    - Max 30 characters

    Args:
        symbol: Stock symbol to validate

    Returns:
        Validated symbol (uppercase)

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    # Normalize to uppercase
    symbol = symbol.strip().upper()

    if not _VALID_SYMBOL_PATTERN.match(symbol):
        raise ValueError(
            f"Invalid symbol '{symbol}'. Symbols must contain only "
            "A-Z, 0-9, hyphen, underscore, or ampersand (max 30 chars)"
        )

    return symbol


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
    - Rate limiting (non-blocking token bucket)
    - Batch operations
    """

    def __init__(
        self,
        kite_client: Optional[KiteConnect] = None,
        rate_limit_mode: str = 'warn'
    ):
        """
        Initialize DataManager.

        Args:
            kite_client: KiteConnect client instance
            rate_limit_mode: How to handle rate limits:
                - 'warn': Log warning but proceed (default, non-blocking)
                - 'block': Sleep until token available (legacy behavior)
                - 'reject': Raise RateLimitExceeded exception
        """
        self.kite = kite_client
        self.cache = DataCache(ttl_seconds=60)  # 1 minute cache for quotes
        self.historical_cache = DataCache(ttl_seconds=3600)  # 1 hour cache for historical

        # NON-BLOCKING RATE LIMITER FIX:
        # Zerodha allows ~10 requests/second. Using token bucket with:
        # - 10 req/s sustained rate
        # - Burst of 10 for handling spikes
        # - 'warn' mode by default (non-blocking, logs issues)
        self._rate_limiter = TokenBucketRateLimiter(
            requests_per_second=10.0,
            burst_size=10,
            mode=rate_limit_mode
        )

    def set_kite_client(self, kite_client: KiteConnect):
        """Set or update Kite client"""
        self.kite = kite_client

    def _rate_limit(self, block: bool = None):
        """
        Enforce rate limiting between API requests.

        FIX: Now uses non-blocking token bucket algorithm by default.
        The old time.sleep() approach blocked the entire thread, which
        was problematic in web server/async contexts.

        Args:
            block: If True, block until token available (legacy behavior).
                   If False/None, use the rate limiter's default mode.
        """
        self._rate_limiter.acquire(block=block)

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics for monitoring."""
        return self._rate_limiter.get_stats()

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

        # CACHE KEY FIX: Use full ISO timestamp instead of just date
        # This prevents cache collisions for intraday requests
        # e.g., 10:00-11:00 vs 11:00-12:00 on the same day
        cache_key = f"hist_{instrument_token}_{from_date.isoformat()}_{to_date.isoformat()}_{interval}"

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

    # ============== FILE STORAGE ==============

    def save_to_file(self, symbol: str, data: pd.DataFrame, interval: str = "day") -> Path:
        """
        Save data to CSV file for offline use.

        Args:
            symbol: Stock symbol
            data: Price data DataFrame
            interval: Time interval

        Returns:
            Path to saved file
        """
        symbol = validate_symbol(symbol)  # SECURITY: Validate before file operations
        filename = f"{symbol}_{interval}.csv"
        filepath = DATA_DIR / filename
        data.to_csv(filepath, index=True)
        print(f"Saved {len(data)} candles to {filepath}")
        return filepath

    def load_from_file(self, symbol: str, interval: str = "day") -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            symbol: Stock symbol
            interval: Time interval

        Returns:
            DataFrame with price data
        """
        symbol = validate_symbol(symbol)  # SECURITY: Validate before file operations
        filename = f"{symbol}_{interval}.csv"
        filepath = DATA_DIR / filename

        if filepath.exists():
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded {len(data)} candles from {filepath}")
            return data
        else:
            print(f"No cached data found for {symbol}")
            return pd.DataFrame()

    # ============== DATABASE STORAGE ==============

    def save_to_database(self, symbol: str, data: pd.DataFrame):
        """
        Save price data to SQLite database.

        PERFORMANCE FIX: Uses executemany() for batch inserts instead of
        individual execute() calls. This is 10-100x faster for large datasets.
        Inserting 100k rows now takes seconds instead of minutes.

        Args:
            symbol: Stock symbol
            data: Price data DataFrame
        """
        if data.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        symbol = validate_symbol(symbol)  # SECURITY: Validate before DB operations

        # Standardize column names to lowercase for consistent access
        df = data.copy()
        df.columns = df.columns.str.lower()

        # Ensure required columns exist with defaults
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0

        # PERFORMANCE FIX: Prepare all data for batch insert
        # This is 10-100x faster than individual INSERT statements
        df_with_index = df.reset_index()
        index_col = df_with_index.columns[0]  # Get the index column name

        # Convert timestamps to ISO format strings
        def format_timestamp(ts):
            if isinstance(ts, str):
                return ts
            elif hasattr(ts, 'isoformat'):
                return ts.isoformat()
            else:
                return str(ts)

        # Build list of tuples for executemany
        records = [
            (
                symbol,
                format_timestamp(row[index_col]),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            )
            for _, row in df_with_index.iterrows()
        ]

        with db_session() as conn:
            cursor = conn.cursor()

            try:
                # BATCH INSERT: executemany is much faster than individual execute()
                cursor.executemany('''
                    INSERT OR REPLACE INTO price_history
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', records)

                saved = cursor.rowcount if cursor.rowcount > 0 else len(records)
                logger.info(f"Saved {saved} candles to database for {symbol}")
                print(f"Saved {saved} candles to database for {symbol}")

            except Exception as e:
                logger.error(f"Failed to save data for {symbol}: {e}")
                # Fall back to individual inserts for partial save
                saved = 0
                for record in records:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO price_history
                            (symbol, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', record)
                        saved += 1
                    except Exception:
                        continue
                print(f"Saved {saved}/{len(records)} candles to database for {symbol} (fallback mode)")

    def load_from_database(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Load price data from database.

        Args:
            symbol: Stock symbol
            days: Number of days to load

        Returns:
            DataFrame with price data
        """
        symbol = validate_symbol(symbol)  # SECURITY: Validate before DB operations
        conn = get_db()
        from_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM price_history
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''

        df = pd.read_sql_query(query, conn, params=(symbol, from_date))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    # ============== DOWNLOAD WITH STORAGE ==============

    def download_and_save(
        self,
        symbol: str,
        instrument_token: int,
        days: int = 365,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Download historical data and save to both file and database.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            instrument_token: Zerodha instrument token
            days: Days of history to download
            interval: Time interval

        Returns:
            DataFrame with price data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"Downloading {symbol} data ({days} days)...")

        try:
            data = self.get_historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
                use_cache=False
            )

            if not data.empty:
                # Save to file
                self.save_to_file(symbol, data, interval)

                # Save to database
                self.save_to_database(symbol, data)

                print(f"Downloaded {len(data)} candles for {symbol}")
                return data
            else:
                print(f"No data received for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            # Try to load from file cache
            return self.load_from_file(symbol, interval)

    def download_multiple(
        self,
        symbols_tokens: Dict[str, int],
        days: int = 365,
        interval: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.

        Args:
            symbols_tokens: Dict of symbol -> instrument_token
            days: Days of history
            interval: Time interval

        Returns:
            Dictionary of symbol -> DataFrame
        """
        result = {}
        total = len(symbols_tokens)

        for i, (symbol, token) in enumerate(symbols_tokens.items(), 1):
            print(f"[{i}/{total}] Downloading {symbol}...")
            result[symbol] = self.download_and_save(symbol, token, days, interval)
            time.sleep(0.5)  # Rate limiting

        print(f"\nDownloaded data for {len(result)} symbols!")
        return result

    # ============== QUICK ACCESS ==============

    def get_data(
        self,
        symbol: str,
        instrument_token: Optional[int] = None,
        days: int = 365,
        interval: str = "day",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get data from best available source.

        Tries in order: memory cache -> file -> database -> download

        Args:
            symbol: Stock symbol
            instrument_token: Zerodha instrument token (for download)
            days: Days of history
            interval: Time interval
            force_download: Force fresh download

        Returns:
            DataFrame with price data
        """
        # 1. Try file cache first (fastest)
        if not force_download:
            file_data = self.load_from_file(symbol, interval)
            if not file_data.empty:
                return file_data

            # 2. Try database
            db_data = self.load_from_database(symbol, days)
            if not db_data.empty:
                return db_data

        # 3. Download if we have token
        if instrument_token and self.kite:
            return self.download_and_save(symbol, instrument_token, days, interval)

        print(f"No data available for {symbol}")
        return pd.DataFrame()

    def get_symbols_with_data(self) -> List[str]:
        """Get list of symbols with cached data"""
        files = list(DATA_DIR.glob("*_day.csv"))
        return [f.stem.replace("_day", "") for f in files]

    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """Get info about cached data for a symbol"""
        data = self.load_from_file(symbol)
        if data.empty:
            return {"symbol": symbol, "has_data": False}

        close_col = 'close' if 'close' in data.columns else 'Close'
        return {
            "symbol": symbol,
            "has_data": True,
            "candles": len(data),
            "from_date": str(data.index[0]),
            "to_date": str(data.index[-1]),
            "latest_close": data[close_col].iloc[-1]
        }


# ============== SAMPLE DATA FOR TESTING ==============

def create_sample_data(symbol: str = "SAMPLE", days: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing (when no broker connected).

    Args:
        symbol: Symbol name
        days: Number of days

    Returns:
        DataFrame with fake OHLCV data
    """
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Starting price
    base_price = 1000.0
    returns = np.random.randn(days) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)

    print(f"Created {days} days of sample data for {symbol}")
    return data


# ============== POPULAR STOCK LISTS ==============

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "BAJFINANCE", "LT", "ASIANPAINT", "AXISBANK", "MARUTI",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND", "WIPRO",
    "HCLTECH", "M&M", "BAJAJFINSV", "NTPC", "POWERGRID",
    "TATASTEEL", "TECHM", "ADANIENT", "JSWSTEEL", "TATAMOTORS",
    "INDUSINDBK", "ONGC", "COALINDIA", "SBILIFE", "HDFCLIFE",
    "DIVISLAB", "GRASIM", "BRITANNIA", "BPCL", "DRREDDY",
    "CIPLA", "APOLLOHOSP", "EICHERMOT", "TATACONSUM", "HEROMOTOCO",
    "UPL", "HINDALCO", "ADANIPORTS", "BAJAJ-AUTO", "LTIM"
]

BANK_NIFTY = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "BANKBARODA", "AUBANK"
]


# Test the module
if __name__ == "__main__":
    print("=" * 50)
    print("DATA MANAGER - Test")
    print("=" * 50)

    # Create sample data for testing
    sample = create_sample_data("TEST", days=30)
    print("\nSample data:")
    print(sample.tail())

    # Test DataManager file operations
    dm = DataManager()
    dm.save_to_file("TEST", sample)

    loaded = dm.load_from_file("TEST")
    print(f"\nLoaded {len(loaded)} candles from file")

    print("\n" + "=" * 50)
    print("Data Manager ready!")
    print("=" * 50)
