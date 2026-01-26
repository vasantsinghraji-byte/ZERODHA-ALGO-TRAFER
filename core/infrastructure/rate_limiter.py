"""
Rate Limiter for Zerodha API
Prevents 429 errors with intelligent request throttling.

Uses Token Bucket algorithm with per-endpoint tracking and automatic
request queuing to ensure API limits are never exceeded.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class APIEndpoint(Enum):
    """Zerodha API endpoint categories with different rate limits"""
    QUOTE = "quote"              # Real-time quotes
    ORDER = "order"              # Order placement/modification
    POSITIONS = "positions"      # Position queries
    HOLDINGS = "holdings"        # Holdings queries
    MARGINS = "margins"          # Margin queries
    HISTORICAL = "historical"    # Historical data
    INSTRUMENTS = "instruments"  # Instrument list
    PROFILE = "profile"          # User profile
    DEFAULT = "default"          # Fallback for uncategorized


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and queuing is disabled"""

    def __init__(self, endpoint: str, retry_after: float):
        self.endpoint = endpoint
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {endpoint}. Retry after {retry_after:.2f}s"
        )


@dataclass
class EndpointConfig:
    """Configuration for a single endpoint's rate limit"""
    requests_per_second: float = 3.0   # Token refill rate
    burst_size: int = 10               # Max tokens (bucket capacity)
    queue_enabled: bool = True         # Enable request queuing
    queue_timeout: float = 30.0        # Max wait time in queue (seconds)

    def __post_init__(self):
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size <= 0:
            raise ValueError("burst_size must be positive")
        if self.queue_timeout < 0:
            raise ValueError("queue_timeout cannot be negative")


@dataclass
class RateLimiterConfig:
    """
    Configuration for the rate limiter.

    Zerodha API limits (approximate):
    - Quote API: 1 request/second
    - Order API: 10 requests/second
    - Historical: 3 requests/second
    - Others: 3 requests/second
    """
    endpoints: Dict[str, EndpointConfig] = field(default_factory=dict)
    global_requests_per_second: float = 10.0  # Overall limit across all endpoints
    global_burst_size: int = 20
    enable_queuing: bool = True

    def __post_init__(self):
        # Set defaults for known Zerodha endpoints if not specified
        defaults = {
            APIEndpoint.QUOTE.value: EndpointConfig(
                requests_per_second=1.0,
                burst_size=5,
            ),
            APIEndpoint.ORDER.value: EndpointConfig(
                requests_per_second=10.0,
                burst_size=25,
            ),
            APIEndpoint.HISTORICAL.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=10,
            ),
            APIEndpoint.POSITIONS.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=10,
            ),
            APIEndpoint.HOLDINGS.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=10,
            ),
            APIEndpoint.MARGINS.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=10,
            ),
            APIEndpoint.INSTRUMENTS.value: EndpointConfig(
                requests_per_second=1.0,
                burst_size=3,
            ),
            APIEndpoint.PROFILE.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=5,
            ),
            APIEndpoint.DEFAULT.value: EndpointConfig(
                requests_per_second=3.0,
                burst_size=10,
            ),
        }

        for endpoint, config in defaults.items():
            if endpoint not in self.endpoints:
                self.endpoints[endpoint] = config


class TokenBucket:
    """
    Token Bucket algorithm implementation for rate limiting.

    Tokens are added at a constant rate up to a maximum (burst) capacity.
    Each request consumes one token. If no tokens available, request waits
    or is rejected based on configuration.

    Thread-safe implementation using RLock.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second (requests per second limit)
            capacity: Maximum tokens (burst capacity)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)  # Start full
        self._last_update = time.monotonic()
        self._lock = threading.RLock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called with lock held."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_update = now

    def acquire(self, blocking: bool = True, timeout: float = 0.0) -> bool:
        """
        Attempt to acquire a token.

        Args:
            blocking: If True, wait for token availability
            timeout: Max time to wait (0 = wait indefinitely if blocking)

        Returns:
            True if token acquired, False otherwise
        """
        deadline = time.monotonic() + timeout if timeout > 0 else None

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                if not blocking:
                    return False

                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._rate

                # Check timeout
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

            # Wait outside lock
            time.sleep(min(wait_time, 0.1))  # Cap sleep to remain responsive

    def time_until_available(self) -> float:
        """Returns seconds until a token will be available"""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                return 0.0
            return (1.0 - self._tokens) / self._rate

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens"""
        with self._lock:
            self._refill()
            return self._tokens


@dataclass
class RequestStats:
    """Statistics for rate limiter monitoring"""
    total_requests: int = 0
    queued_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0
    last_request_time: Optional[datetime] = None

    @property
    def average_wait_time(self) -> float:
        """Average wait time per request"""
        if self.total_requests == 0:
            return 0.0
        return self.total_wait_time / self.total_requests


class RateLimiter:
    """
    Rate limiter with per-endpoint tracking and automatic request queuing.

    Prevents 429 errors by proactively throttling requests to stay within
    Zerodha API limits. Supports different limits per endpoint type.

    Thread-safe implementation suitable for multi-threaded trading engines.

    Example:
        config = RateLimiterConfig()
        limiter = RateLimiter(config)

        # Wrap API calls
        result = limiter.execute(
            endpoint=APIEndpoint.QUOTE,
            func=broker.get_quote,
            args=("RELIANCE",)
        )

        # Or use as context manager
        with limiter.acquire(APIEndpoint.ORDER):
            broker.place_order(...)
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Args:
            config: Rate limiter configuration. Uses defaults if not provided.
        """
        self._config = config or RateLimiterConfig()
        self._lock = threading.RLock()

        # Create token buckets for each endpoint
        self._buckets: Dict[str, TokenBucket] = {}
        for endpoint, ep_config in self._config.endpoints.items():
            self._buckets[endpoint] = TokenBucket(
                rate=ep_config.requests_per_second,
                capacity=ep_config.burst_size
            )

        # Global bucket for overall rate limiting
        self._global_bucket = TokenBucket(
            rate=self._config.global_requests_per_second,
            capacity=self._config.global_burst_size
        )

        # Request queues per endpoint
        self._queues: Dict[str, deque] = {
            ep: deque() for ep in self._config.endpoints
        }

        # Statistics per endpoint
        self._stats: Dict[str, RequestStats] = {
            ep: RequestStats() for ep in self._config.endpoints
        }
        self._global_stats = RequestStats()

        logger.debug(
            f"RateLimiter initialized with {len(self._buckets)} endpoint configs"
        )

    def _get_endpoint_key(self, endpoint: APIEndpoint | str) -> str:
        """Convert endpoint to string key"""
        if isinstance(endpoint, APIEndpoint):
            return endpoint.value
        return endpoint if endpoint in self._buckets else APIEndpoint.DEFAULT.value

    def _get_bucket(self, endpoint: str) -> TokenBucket:
        """Get the token bucket for an endpoint"""
        return self._buckets.get(endpoint, self._buckets[APIEndpoint.DEFAULT.value])

    def _get_config(self, endpoint: str) -> EndpointConfig:
        """Get configuration for an endpoint"""
        return self._config.endpoints.get(
            endpoint,
            self._config.endpoints[APIEndpoint.DEFAULT.value]
        )

    def acquire(self, endpoint: APIEndpoint | str) -> 'RateLimitContext':
        """
        Context manager for rate-limited operations.

        Args:
            endpoint: The API endpoint category

        Returns:
            Context manager that acquires rate limit on enter

        Example:
            with limiter.acquire(APIEndpoint.ORDER):
                broker.place_order(...)
        """
        return RateLimitContext(self, endpoint)

    def wait_for_capacity(
        self,
        endpoint: APIEndpoint | str,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait until rate limit allows a request.

        Args:
            endpoint: The API endpoint category
            timeout: Max wait time (uses config default if None)

        Returns:
            True if capacity available, False if timeout

        Raises:
            RateLimitExceeded: If queuing disabled and no capacity
        """
        ep_key = self._get_endpoint_key(endpoint)
        ep_config = self._get_config(ep_key)
        bucket = self._get_bucket(ep_key)

        effective_timeout = timeout if timeout is not None else ep_config.queue_timeout
        queue_enabled = ep_config.queue_enabled and self._config.enable_queuing

        start_time = time.monotonic()

        # First check global limit
        if not self._global_bucket.acquire(
            blocking=queue_enabled,
            timeout=effective_timeout
        ):
            retry_after = self._global_bucket.time_until_available()
            with self._lock:
                self._global_stats.rejected_requests += 1
            if not queue_enabled:
                raise RateLimitExceeded("global", retry_after)
            logger.warning(f"Global rate limit timeout for {ep_key}")
            return False

        # Then check endpoint-specific limit
        if not bucket.acquire(blocking=queue_enabled, timeout=effective_timeout):
            retry_after = bucket.time_until_available()
            with self._lock:
                self._stats[ep_key].rejected_requests += 1
            if not queue_enabled:
                raise RateLimitExceeded(ep_key, retry_after)
            logger.warning(f"Endpoint rate limit timeout for {ep_key}")
            return False

        # Update statistics
        wait_time = time.monotonic() - start_time
        with self._lock:
            stats = self._stats[ep_key]
            stats.total_requests += 1
            stats.total_wait_time += wait_time
            stats.last_request_time = datetime.now()
            if wait_time > 0.01:  # More than 10ms wait
                stats.queued_requests += 1

            self._global_stats.total_requests += 1
            self._global_stats.total_wait_time += wait_time
            self._global_stats.last_request_time = datetime.now()

        if wait_time > 0.1:
            logger.debug(f"Rate limit wait for {ep_key}: {wait_time:.3f}s")

        return True

    def execute(
        self,
        endpoint: APIEndpoint | str,
        func: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> T:
        """
        Execute a function with rate limiting.

        Args:
            endpoint: The API endpoint category
            func: Function to execute
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            timeout: Max wait time for rate limit

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            RateLimitExceeded: If timeout and queuing disabled
        """
        kwargs = kwargs or {}

        if not self.wait_for_capacity(endpoint, timeout):
            ep_key = self._get_endpoint_key(endpoint)
            bucket = self._get_bucket(ep_key)
            raise RateLimitExceeded(ep_key, bucket.time_until_available())

        return func(*args, **kwargs)

    def get_stats(self, endpoint: Optional[APIEndpoint | str] = None) -> RequestStats:
        """
        Get statistics for an endpoint or global stats.

        Args:
            endpoint: Specific endpoint, or None for global stats

        Returns:
            RequestStats for the endpoint
        """
        with self._lock:
            if endpoint is None:
                return RequestStats(
                    total_requests=self._global_stats.total_requests,
                    queued_requests=self._global_stats.queued_requests,
                    rejected_requests=self._global_stats.rejected_requests,
                    total_wait_time=self._global_stats.total_wait_time,
                    last_request_time=self._global_stats.last_request_time
                )

            ep_key = self._get_endpoint_key(endpoint)
            stats = self._stats.get(ep_key, RequestStats())
            return RequestStats(
                total_requests=stats.total_requests,
                queued_requests=stats.queued_requests,
                rejected_requests=stats.rejected_requests,
                total_wait_time=stats.total_wait_time,
                last_request_time=stats.last_request_time
            )

    def get_available_capacity(self, endpoint: APIEndpoint | str) -> Dict[str, float]:
        """
        Get current available capacity for an endpoint.

        Args:
            endpoint: The API endpoint category

        Returns:
            Dict with 'tokens' and 'time_until_next' keys
        """
        ep_key = self._get_endpoint_key(endpoint)
        bucket = self._get_bucket(ep_key)

        return {
            'tokens': bucket.available_tokens,
            'time_until_next': bucket.time_until_available(),
            'global_tokens': self._global_bucket.available_tokens,
        }

    def reset_stats(self, endpoint: Optional[APIEndpoint | str] = None) -> None:
        """Reset statistics for an endpoint or all endpoints"""
        with self._lock:
            if endpoint is None:
                for key in self._stats:
                    self._stats[key] = RequestStats()
                self._global_stats = RequestStats()
            else:
                ep_key = self._get_endpoint_key(endpoint)
                if ep_key in self._stats:
                    self._stats[ep_key] = RequestStats()


class RateLimitContext:
    """Context manager for rate-limited operations"""

    def __init__(self, limiter: RateLimiter, endpoint: APIEndpoint | str):
        self._limiter = limiter
        self._endpoint = endpoint

    def __enter__(self) -> 'RateLimitContext':
        self._limiter.wait_for_capacity(self._endpoint)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # No cleanup needed
