# -*- coding: utf-8 -*-
"""
Retry Utilities with Exponential Backoff
=========================================

NETWORK RELIABILITY FIX (Bug #14):
Financial APIs have transient failures - packet loss, micro-outages, etc.
A 50ms network blip shouldn't cause missed trades.

This module provides retry decorators for resilient network operations.

Usage:
    from utils.retry import retry_on_network_error

    @retry_on_network_error(tries=3, delay=0.1, backoff=2)
    def fetch_data():
        return api.get_data()

    # Tries 3 times with delays: 0.1s, 0.2s, 0.4s
"""

import functools
import logging
import time
from typing import Tuple, Type, Callable, Any, Optional

logger = logging.getLogger(__name__)

# Default exceptions to retry on
NETWORK_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)

# Try to add requests exceptions if available
try:
    from requests.exceptions import (
        ConnectionError as RequestsConnectionError,
        Timeout as RequestsTimeout,
        ChunkedEncodingError,
    )
    NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (
        RequestsConnectionError,
        RequestsTimeout,
        ChunkedEncodingError,
    )
except ImportError:
    pass


def retry_on_network_error(
    tries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """
    Retry decorator with exponential backoff for network operations.

    Args:
        tries: Maximum number of attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 0.1)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exceptions to catch (default: NETWORK_EXCEPTIONS)
        on_retry: Optional callback(exception, attempt, next_delay) called before retry

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @retry_on_network_error(tries=3, delay=0.1, backoff=2)
        def api_call():
            return requests.get(url)

        # Attempt 1: immediate
        # Attempt 2: wait 0.1s
        # Attempt 3: wait 0.2s (0.1 * 2)
        # Total max wait: 0.3s

    Note:
        - Only retries on transient network errors, not on auth/validation errors
        - Logs each retry attempt with the exception details
        - Final exception is re-raised if all retries fail
    """
    if exceptions is None:
        exceptions = NETWORK_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == tries:
                        # Final attempt failed - log and re-raise
                        logger.error(
                            f"{func.__name__} failed after {tries} attempts: {e}"
                        )
                        raise

                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{tries} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )

                    # Call optional retry callback
                    if on_retry:
                        on_retry(e, attempt, current_delay)

                    # Wait before retry
                    time.sleep(current_delay)

                    # Increase delay for next retry (exponential backoff)
                    current_delay *= backoff

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_with_result(
    tries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    default: Any = None,
) -> Callable:
    """
    Retry decorator that returns a default value instead of raising.

    Same as retry_on_network_error but returns `default` on final failure
    instead of re-raising the exception. Useful for non-critical operations.

    Args:
        tries: Maximum number of attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 0.1)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exceptions to catch (default: NETWORK_EXCEPTIONS)
        default: Value to return if all retries fail (default: None)

    Example:
        @retry_with_result(tries=3, default=None)
        def get_quote(symbol):
            return api.quote(symbol)

        quote = get_quote("RELIANCE")  # Returns None if all retries fail
    """
    if exceptions is None:
        exceptions = NETWORK_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        # Final attempt failed - return default
                        logger.error(
                            f"{func.__name__} failed after {tries} attempts: {e}. "
                            f"Returning default: {default}"
                        )
                        return default

                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{tries} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )

                    # Wait before retry
                    time.sleep(current_delay)

                    # Increase delay for next retry
                    current_delay *= backoff

            return default

        return wrapper
    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations.

    Use when you need more control over retry logic than decorators provide.

    Example:
        with RetryableOperation(tries=3, delay=0.1) as retry:
            while retry.should_continue():
                try:
                    result = api.call()
                    break
                except ConnectionError as e:
                    retry.failed(e)
    """

    def __init__(
        self,
        tries: int = 3,
        delay: float = 0.1,
        backoff: float = 2.0,
        exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.tries = tries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions or NETWORK_EXCEPTIONS
        self.attempt = 0
        self.current_delay = delay
        self.last_exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def should_continue(self) -> bool:
        """Check if more attempts are available."""
        return self.attempt < self.tries

    def failed(self, exception: Exception) -> None:
        """
        Record a failed attempt and wait before next retry.

        Raises the exception if no more retries available.
        """
        self.attempt += 1
        self.last_exception = exception

        if self.attempt >= self.tries:
            logger.error(f"Operation failed after {self.tries} attempts: {exception}")
            raise exception

        logger.warning(
            f"Attempt {self.attempt}/{self.tries} failed: {exception}. "
            f"Retrying in {self.current_delay:.2f}s..."
        )

        time.sleep(self.current_delay)
        self.current_delay *= self.backoff
