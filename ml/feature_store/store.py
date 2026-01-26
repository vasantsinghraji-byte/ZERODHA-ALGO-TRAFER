# -*- coding: utf-8 -*-
"""
Feature Store - Centralized Feature Repository
===============================================
Manages feature storage, retrieval, and real-time updates.

Supports:
- In-memory storage for development
- Redis for production real-time features
- File-based historical feature storage
- Point-in-time feature retrieval for training

Example:
    >>> from ml.feature_store import FeatureStore, FeatureStoreConfig
    >>>
    >>> # Create store
    >>> store = FeatureStore()
    >>>
    >>> # Store features
    >>> store.put("RELIANCE", {"rsi_14": 65.5, "volatility": 0.023})
    >>>
    >>> # Retrieve features
    >>> features = store.get("RELIANCE")
    >>>
    >>> # Historical retrieval
    >>> history = store.get_historical("RELIANCE", start_date, end_date)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import threading
import json
import pickle
import logging
import hashlib

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .features import FeatureSet, FeatureValue, FeatureDefinition

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Storage backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    SQLITE = "sqlite"


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""
    # Backend selection
    backend: StorageBackend = StorageBackend.MEMORY

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "features:"
    redis_ttl_seconds: int = 86400  # 24 hours default TTL

    # File storage settings
    file_path: str = "data/features"
    file_format: str = "parquet"  # parquet, csv, pickle

    # Cache settings
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 60

    # Historical settings
    keep_history: bool = True
    max_history_days: int = 365

    # Versioning
    enable_versioning: bool = True
    default_version: str = "1.0.0"


class StorageAdapter(ABC):
    """Abstract storage adapter interface."""

    @abstractmethod
    def put(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store a value."""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a value."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        pass


class MemoryStorageAdapter(StorageAdapter):
    """In-memory storage adapter."""

    def __init__(self):
        self._data: Dict[str, Tuple[Dict[str, Any], Optional[datetime]]] = {}
        self._lock = threading.RLock()

    def put(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl) if ttl else None
            self._data[key] = (value, expiry)
        return True

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._data:
                return None

            value, expiry = self._data[key]

            # Check expiry
            if expiry and datetime.now() > expiry:
                del self._data[key]
                return None

            return value

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> List[str]:
        with self._lock:
            if pattern == "*":
                return list(self._data.keys())

            # Simple pattern matching
            import fnmatch
            return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]


class RedisStorageAdapter(StorageAdapter):
    """Redis storage adapter for production use."""

    def __init__(self, config: FeatureStoreConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("redis library not installed. Install with: pip install redis")

        self.config = config
        self.prefix = config.redis_prefix

        self._client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False
        )

        # Test connection
        try:
            self._client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def put(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            full_key = self._make_key(key)
            serialized = pickle.dumps(value)

            if ttl:
                self._client.setex(full_key, ttl, serialized)
            else:
                self._client.set(full_key, serialized)

            return True
        except Exception as e:
            logger.error(f"Redis put error: {e}")
            return False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)

            if data is None:
                return None

            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            full_key = self._make_key(key)
            return self._client.delete(full_key) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            full_key = self._make_key(key)
            return self._client.exists(full_key) > 0
        except Exception as e:
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        try:
            full_pattern = self._make_key(pattern)
            keys = self._client.keys(full_pattern)
            prefix_len = len(self.prefix)
            return [k.decode('utf-8')[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []


class FileStorageAdapter(StorageAdapter):
    """File-based storage adapter for historical data."""

    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.base_path = Path(config.file_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.format = config.file_format

    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        # Sanitize key for filesystem
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.base_path / f"{safe_key}.{self.format}"

    def put(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            file_path = self._get_file_path(key)

            if self.format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            elif self.format == "json":
                with open(file_path, 'w') as f:
                    json.dump(value, f, default=str)
            elif self.format == "parquet" and PANDAS_AVAILABLE:
                df = pd.DataFrame([value])
                df.to_parquet(file_path)
            else:
                # Default to pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)

            return True
        except Exception as e:
            logger.error(f"File put error: {e}")
            return False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                return None

            if self.format == "pickle":
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif self.format == "json":
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif self.format == "parquet" and PANDAS_AVAILABLE:
                df = pd.read_parquet(file_path)
                return df.iloc[0].to_dict()
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"File get error: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"File delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        return self._get_file_path(key).exists()

    def keys(self, pattern: str = "*") -> List[str]:
        try:
            import fnmatch
            all_files = list(self.base_path.glob(f"*.{self.format}"))
            keys = [f.stem for f in all_files]

            if pattern == "*":
                return keys
            return [k for k in keys if fnmatch.fnmatch(k, pattern)]
        except Exception as e:
            logger.error(f"File keys error: {e}")
            return []


class FeatureCache:
    """LRU cache for features."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
                del self._cache[key]
                return None

            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (value, datetime.now())

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class FeatureStore:
    """
    Centralized Feature Store.

    Manages storage and retrieval of ML features with support for:
    - Real-time feature serving
    - Historical feature retrieval
    - Point-in-time correctness
    - Feature versioning

    Example:
        >>> store = FeatureStore()
        >>>
        >>> # Store real-time features
        >>> store.put("RELIANCE", {
        ...     "rsi_14": 65.5,
        ...     "volatility_20": 0.023,
        ...     "vwap_deviation": 0.15
        ... })
        >>>
        >>> # Retrieve latest features
        >>> features = store.get("RELIANCE")
        >>>
        >>> # Historical retrieval for training
        >>> history = store.get_historical(
        ...     "RELIANCE",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31)
        ... )
    """

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or FeatureStoreConfig()

        # Initialize storage adapter
        if self.config.backend == StorageBackend.REDIS:
            self._adapter = RedisStorageAdapter(self.config)
        elif self.config.backend == StorageBackend.FILE:
            self._adapter = FileStorageAdapter(self.config)
        else:
            self._adapter = MemoryStorageAdapter()

        # Initialize cache
        self._cache: Optional[FeatureCache] = None
        if self.config.enable_cache:
            self._cache = FeatureCache(
                max_size=self.config.cache_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )

        # History storage (always in-memory or file)
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._history_lock = threading.RLock()

        # Feature definitions registry
        self._definitions: Dict[str, FeatureDefinition] = {}

    def put(
        self,
        symbol: str,
        features: Union[Dict[str, Any], FeatureSet],
        timestamp: Optional[datetime] = None,
        version: Optional[str] = None
    ) -> bool:
        """
        Store features for a symbol.

        Args:
            symbol: Trading symbol
            features: Feature dict or FeatureSet
            timestamp: Feature timestamp (default: now)
            version: Feature version (default: config default)

        Returns:
            True if stored successfully
        """
        timestamp = timestamp or datetime.now()
        version = version or self.config.default_version

        # Convert FeatureSet to dict
        if isinstance(features, FeatureSet):
            feature_dict = features.get_all()
        else:
            feature_dict = features

        # Build storage record
        record = {
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'version': version,
            'features': feature_dict
        }

        # Store in primary storage
        key = f"{symbol}:latest"
        success = self._adapter.put(key, record, ttl=self.config.redis_ttl_seconds)

        # Update cache
        if self._cache:
            self._cache.put(key, record)

        # Store in history
        if self.config.keep_history:
            self._store_history(symbol, record)

        return success

    def get(
        self,
        symbol: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest features for a symbol.

        Args:
            symbol: Trading symbol
            version: Specific version (None = latest)

        Returns:
            Feature dict or None if not found
        """
        key = f"{symbol}:latest"

        # Check cache first
        if self._cache:
            cached = self._cache.get(key)
            if cached:
                record = cached
                if version is None or record.get('version') == version:
                    return record.get('features')

        # Fetch from storage
        record = self._adapter.get(key)
        if record is None:
            return None

        # Update cache
        if self._cache:
            self._cache.put(key, record)

        # Version check
        if version and record.get('version') != version:
            return None

        return record.get('features')

    def get_feature(
        self,
        symbol: str,
        feature_name: str
    ) -> Optional[Any]:
        """Get a single feature value."""
        features = self.get(symbol)
        if features:
            return features.get(feature_name)
        return None

    def get_multiple(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple symbols."""
        return {symbol: self.get(symbol) or {} for symbol in symbols}

    def _store_history(self, symbol: str, record: Dict[str, Any]) -> None:
        """Store record in history."""
        with self._history_lock:
            self._history[symbol].append(record.copy())

            # Trim old history
            max_records = self.config.max_history_days * 24 * 60  # Assuming 1-min granularity
            if len(self._history[symbol]) > max_records:
                self._history[symbol] = self._history[symbol][-max_records:]

    def get_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        features: Optional[List[str]] = None
    ) -> 'pd.DataFrame':
        """
        Get historical features for training.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date (default: now)
            features: Specific features to retrieve (None = all)

        Returns:
            DataFrame with historical features
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for get_historical()")

        end_date = end_date or datetime.now()

        with self._history_lock:
            records = self._history.get(symbol, [])

        # Filter by date
        filtered = []
        for record in records:
            ts = datetime.fromisoformat(record['timestamp'])
            if start_date <= ts <= end_date:
                row = {'timestamp': ts, 'symbol': symbol}
                row.update(record.get('features', {}))
                filtered.append(row)

        if not filtered:
            return pd.DataFrame()

        df = pd.DataFrame(filtered)

        # Filter features if specified
        if features:
            available = ['timestamp', 'symbol'] + [f for f in features if f in df.columns]
            df = df[available]

        return df.sort_values('timestamp').reset_index(drop=True)

    def get_point_in_time(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get features as they were at a specific point in time.

        Important for avoiding lookahead bias in training.
        """
        with self._history_lock:
            records = self._history.get(symbol, [])

        # Find the most recent record before timestamp
        best_record = None
        best_time = None

        for record in records:
            ts = datetime.fromisoformat(record['timestamp'])
            if ts <= timestamp:
                if best_time is None or ts > best_time:
                    best_time = ts
                    best_record = record

        return best_record.get('features') if best_record else None

    def delete(self, symbol: str) -> bool:
        """Delete features for a symbol."""
        key = f"{symbol}:latest"

        if self._cache:
            self._cache.invalidate(key)

        with self._history_lock:
            if symbol in self._history:
                del self._history[symbol]

        return self._adapter.delete(key)

    def list_symbols(self) -> List[str]:
        """List all symbols with stored features."""
        keys = self._adapter.keys("*:latest")
        return [k.replace(":latest", "") for k in keys]

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._history_lock:
            history_counts = {s: len(records) for s, records in self._history.items()}

        return {
            'backend': self.config.backend.value,
            'symbols_count': len(self.list_symbols()),
            'cache_enabled': self.config.enable_cache,
            'history_records': history_counts,
            'total_history_records': sum(history_counts.values())
        }

    def register_definition(self, definition: FeatureDefinition) -> None:
        """Register a feature definition."""
        self._definitions[definition.name] = definition

    def get_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get all registered feature definitions."""
        return self._definitions.copy()

    def clear(self) -> None:
        """Clear all data."""
        for symbol in self.list_symbols():
            self.delete(symbol)

        if self._cache:
            self._cache.clear()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_store_instance: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get global feature store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = FeatureStore()
    return _store_instance


def set_feature_store(store: FeatureStore) -> None:
    """Set global feature store instance."""
    global _store_instance
    _store_instance = store


def put_features(symbol: str, features: Dict[str, Any]) -> bool:
    """Store features using global store."""
    return get_feature_store().put(symbol, features)


def get_features(symbol: str) -> Optional[Dict[str, Any]]:
    """Get features using global store."""
    return get_feature_store().get(symbol)


def get_historical_features(
    symbol: str,
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> 'pd.DataFrame':
    """Get historical features using global store."""
    return get_feature_store().get_historical(symbol, start_date, end_date)
