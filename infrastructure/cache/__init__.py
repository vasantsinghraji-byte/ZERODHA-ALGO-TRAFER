import json
import logging
from functools import wraps
from typing import Any, Optional, Callable
import time

import redis
from redis.connection import ConnectionPool
from config.config import settings

logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Custom cache error"""
    pass

class RedisManager:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        if self._pool is None:
            self._pool = ConnectionPool.from_url(settings.REDIS_URL)
            logger.info("Redis connection pool initialized")

    def get_connection(self) -> redis.Redis:
        if self._pool is None:
            self.initialize()
        return redis.Redis(connection_pool=self._pool)

    def close_all(self) -> None:
        if self._pool:
            self._pool.disconnect()
            self._pool = None

    def set_market_data(self, symbol: str, data: dict, ttl: int = 300) -> None:
        """Cache market data with TTL"""
        conn = self.get_connection()
        key = f"market:data:{symbol}"
        conn.setex(key, ttl, json.dumps(data))

    def get_market_data(self, symbol: str) -> Optional[dict]:
        """Get cached market data"""
        conn = self.get_connection()
        key = f"market:data:{symbol}"
        data = conn.get(key)
        return json.loads(data) if data else None

    def update_ticker(self, symbol: str, price: float, volume: int) -> None:
        """Update real-time ticker data"""
        conn = self.get_connection()
        key = f"market:ticker:{symbol}"
        data = {
            "price": price,
            "volume": volume,
            "timestamp": int(time.time())
        }
        conn.set(key, json.dumps(data))
        
    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Get real-time ticker data"""
        conn = self.get_connection()
        key = f"market:ticker:{symbol}"
        data = conn.get(key)
        return json.loads(data) if data else None

redis_manager = RedisManager()

def cache_result(prefix: str, ttl: int = 3600):
    """Cache decorator for database queries"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_key = f"{prefix}:{json.dumps(args)}:{json.dumps(kwargs)}"
            
            # Try to get from cache
            redis_conn = redis_manager.get_connection()
            cached = redis_conn.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                redis_conn.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str) -> None:
    """Invalidate cache keys matching pattern"""
    redis_conn = redis_manager.get_connection()
    keys = redis_conn.keys(pattern)
    if keys:
        redis_conn.delete(*keys)

def cache_market_data(ttl: int = 300):
    """Specific decorator for market data caching"""
    return cache_result(prefix="market:data", ttl=ttl)
