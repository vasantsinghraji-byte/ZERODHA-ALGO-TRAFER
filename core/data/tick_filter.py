"""
Bad Tick Filter for Data Quality.

Filters out erroneous ticks that could corrupt trading decisions:
- Price spikes/drops beyond normal volatility
- Flash crash detection
- Stale tick detection
- Zero/negative price rejection

Essential for protecting strategies from garbage data.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FilterReason(Enum):
    """Reasons for rejecting a tick"""
    PRICE_SPIKE = "price_spike"           # Price changed too much too fast
    PRICE_DROP = "price_drop"             # Price dropped too much too fast
    FLASH_CRASH = "flash_crash"           # Multiple rapid price drops
    ZERO_PRICE = "zero_price"             # Zero or negative price
    STALE_TICK = "stale_tick"             # Timestamp too old
    INVALID_VOLUME = "invalid_volume"     # Negative or impossibly large volume
    CIRCUIT_BREAKER = "circuit_breaker"   # Price outside circuit limits
    DUPLICATE = "duplicate"               # Exact duplicate tick


@dataclass
class RejectedTick:
    """Record of a rejected tick"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    reason: FilterReason
    last_valid_price: float
    deviation_percent: float
    message: str
    raw_tick: Optional[Dict] = None


@dataclass
class SymbolFilterConfig:
    """Per-symbol filter configuration"""
    max_deviation_percent: float = 5.0    # Max allowed price change %
    deviation_window_seconds: float = 1.0  # Time window for deviation check
    flash_crash_threshold: float = 10.0    # % drop that triggers flash crash mode
    flash_crash_window: int = 5            # Ticks to check for flash crash
    circuit_upper_percent: float = 20.0    # Upper circuit limit
    circuit_lower_percent: float = 20.0    # Lower circuit limit
    min_tick_interval_ms: float = 0.0      # Min time between ticks (0 = no limit)
    max_volume: int = 100_000_000          # Max allowed volume per tick


@dataclass
class TickFilterConfig:
    """Global tick filter configuration"""
    # Default thresholds (can be overridden per symbol)
    default_max_deviation_percent: float = 5.0
    default_deviation_window_seconds: float = 1.0
    default_flash_crash_threshold: float = 10.0

    # Per-symbol overrides
    symbol_configs: Dict[str, SymbolFilterConfig] = field(default_factory=dict)

    # Global settings
    reject_zero_prices: bool = True
    reject_negative_volumes: bool = True
    max_tick_age_seconds: float = 60.0     # Reject ticks older than this
    log_rejected_ticks: bool = True
    store_rejected_history: bool = True
    rejected_history_size: int = 1000

    # Flash crash protection
    enable_flash_crash_protection: bool = True
    flash_crash_cooldown_seconds: float = 30.0

    # Statistics
    track_statistics: bool = True


@dataclass
class TickFilterStats:
    """Statistics for tick filtering"""
    total_ticks: int = 0
    accepted_ticks: int = 0
    rejected_ticks: int = 0
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)
    rejections_by_symbol: Dict[str, int] = field(default_factory=dict)
    last_rejection_time: Optional[datetime] = None
    flash_crash_activations: int = 0

    @property
    def rejection_rate(self) -> float:
        """Percentage of ticks rejected"""
        if self.total_ticks == 0:
            return 0.0
        return (self.rejected_ticks / self.total_ticks) * 100


class TickFilter:
    """
    Filter for detecting and rejecting bad ticks.

    Protects trading strategies from erroneous market data that could
    cause incorrect signals or catastrophic losses.

    Features:
    - Price spike/drop detection with configurable thresholds
    - Flash crash protection with cooldown period
    - Per-symbol configuration
    - Comprehensive rejection logging
    - Statistics tracking

    Example:
        config = TickFilterConfig(
            default_max_deviation_percent=5.0
        )
        tick_filter = TickFilter(config)

        # Filter a tick
        is_valid, reason = tick_filter.filter_tick(
            symbol="RELIANCE",
            price=2500.0,
            volume=100,
            timestamp=datetime.now()
        )

        if not is_valid:
            print(f"Tick rejected: {reason}")

        # Wrap LiveDataSource
        tick_filter.wrap_live_source(live_source)
    """

    def __init__(self, config: Optional[TickFilterConfig] = None):
        """
        Args:
            config: Filter configuration
        """
        self._config = config or TickFilterConfig()
        self._lock = threading.RLock()

        # Last valid price per symbol
        self._last_prices: Dict[str, float] = {}
        self._last_price_times: Dict[str, datetime] = {}

        # Recent tick history for flash crash detection
        self._recent_ticks: Dict[str, deque] = {}

        # Flash crash state
        self._flash_crash_active: Dict[str, bool] = {}
        self._flash_crash_start: Dict[str, datetime] = {}

        # Reference prices (e.g., previous close for circuit limits)
        self._reference_prices: Dict[str, float] = {}

        # Rejected tick history
        self._rejected_ticks: deque = deque(maxlen=self._config.rejected_history_size)
        self._rejected_lock = threading.Lock()

        # Statistics
        self._stats = TickFilterStats()
        self._stats_lock = threading.Lock()

        # Callbacks for rejected ticks
        self._rejection_callbacks: List[Callable[[RejectedTick], None]] = []

        logger.info("TickFilter initialized")

    def _get_symbol_config(self, symbol: str) -> SymbolFilterConfig:
        """Get configuration for a symbol (custom or default)"""
        if symbol in self._config.symbol_configs:
            return self._config.symbol_configs[symbol]

        # Return default config
        return SymbolFilterConfig(
            max_deviation_percent=self._config.default_max_deviation_percent,
            deviation_window_seconds=self._config.default_deviation_window_seconds,
            flash_crash_threshold=self._config.default_flash_crash_threshold
        )

    def set_symbol_config(self, symbol: str, config: SymbolFilterConfig) -> None:
        """Set custom configuration for a symbol"""
        with self._lock:
            self._config.symbol_configs[symbol] = config
        logger.debug(f"Custom config set for {symbol}")

    def set_reference_price(self, symbol: str, price: float) -> None:
        """
        Set reference price for circuit limit calculations.

        Usually the previous day's close.

        Args:
            symbol: Trading symbol
            price: Reference price
        """
        with self._lock:
            self._reference_prices[symbol] = price
        logger.debug(f"Reference price for {symbol}: {price}")

    def filter_tick(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: Optional[datetime] = None,
        raw_tick: Optional[Dict] = None
    ) -> Tuple[bool, Optional[FilterReason]]:
        """
        Filter a tick and determine if it should be accepted.

        Args:
            symbol: Trading symbol
            price: Tick price
            volume: Tick volume
            timestamp: Tick timestamp (default: now)
            raw_tick: Original tick data for logging

        Returns:
            Tuple of (is_valid, rejection_reason)
            If valid: (True, None)
            If rejected: (False, FilterReason)
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            # Update stats
            if self._config.track_statistics:
                with self._stats_lock:
                    self._stats.total_ticks += 1

            # Get symbol config
            sym_config = self._get_symbol_config(symbol)

            # Initialize tracking for new symbols
            if symbol not in self._recent_ticks:
                self._recent_ticks[symbol] = deque(maxlen=sym_config.flash_crash_window * 2)
                self._flash_crash_active[symbol] = False

            # === Check 1: Zero/Negative Price ===
            if self._config.reject_zero_prices and price <= 0:
                return self._reject(
                    symbol, price, volume, timestamp,
                    FilterReason.ZERO_PRICE,
                    0.0, 0.0,
                    f"Zero or negative price: {price}",
                    raw_tick
                )

            # === Check 2: Invalid Volume ===
            if self._config.reject_negative_volumes and volume < 0:
                return self._reject(
                    symbol, price, volume, timestamp,
                    FilterReason.INVALID_VOLUME,
                    self._last_prices.get(symbol, price), 0.0,
                    f"Negative volume: {volume}",
                    raw_tick
                )

            if volume > sym_config.max_volume:
                return self._reject(
                    symbol, price, volume, timestamp,
                    FilterReason.INVALID_VOLUME,
                    self._last_prices.get(symbol, price), 0.0,
                    f"Volume too large: {volume} > {sym_config.max_volume}",
                    raw_tick
                )

            # === Check 3: Stale Tick ===
            now = datetime.now()
            tick_age = (now - timestamp).total_seconds()
            if tick_age > self._config.max_tick_age_seconds:
                return self._reject(
                    symbol, price, volume, timestamp,
                    FilterReason.STALE_TICK,
                    self._last_prices.get(symbol, price), 0.0,
                    f"Tick too old: {tick_age:.1f}s",
                    raw_tick
                )

            # === Check 4: Circuit Breaker Limits ===
            if symbol in self._reference_prices:
                ref_price = self._reference_prices[symbol]
                upper_limit = ref_price * (1 + sym_config.circuit_upper_percent / 100)
                lower_limit = ref_price * (1 - sym_config.circuit_lower_percent / 100)

                if price > upper_limit or price < lower_limit:
                    deviation = ((price - ref_price) / ref_price) * 100
                    return self._reject(
                        symbol, price, volume, timestamp,
                        FilterReason.CIRCUIT_BREAKER,
                        ref_price, deviation,
                        f"Price outside circuit limits: {price} (ref: {ref_price}, limits: {lower_limit:.2f}-{upper_limit:.2f})",
                        raw_tick
                    )

            # === Check 5: Flash Crash Active ===
            if self._flash_crash_active.get(symbol, False):
                start_time = self._flash_crash_start.get(symbol)
                if start_time:
                    elapsed = (now - start_time).total_seconds()
                    if elapsed < self._config.flash_crash_cooldown_seconds:
                        # Still in cooldown - reject all ticks
                        return self._reject(
                            symbol, price, volume, timestamp,
                            FilterReason.FLASH_CRASH,
                            self._last_prices.get(symbol, price), 0.0,
                            f"Flash crash cooldown active ({self._config.flash_crash_cooldown_seconds - elapsed:.0f}s remaining)",
                            raw_tick
                        )
                    else:
                        # Cooldown expired
                        self._flash_crash_active[symbol] = False
                        logger.info(f"Flash crash cooldown ended for {symbol}")

            # === Check 6: Price Deviation ===
            last_price = self._last_prices.get(symbol)
            last_time = self._last_price_times.get(symbol)

            if last_price is not None and last_price > 0:
                deviation_percent = ((price - last_price) / last_price) * 100

                # Check time window
                if last_time:
                    time_delta = (timestamp - last_time).total_seconds()
                else:
                    time_delta = float('inf')

                # Only check deviation if within time window
                if time_delta <= sym_config.deviation_window_seconds:
                    if abs(deviation_percent) > sym_config.max_deviation_percent:
                        reason = FilterReason.PRICE_SPIKE if deviation_percent > 0 else FilterReason.PRICE_DROP
                        return self._reject(
                            symbol, price, volume, timestamp,
                            reason,
                            last_price, deviation_percent,
                            f"Price deviation {deviation_percent:+.2f}% exceeds {sym_config.max_deviation_percent}% in {time_delta:.2f}s",
                            raw_tick
                        )

                # === Check 7: Flash Crash Detection ===
                if self._config.enable_flash_crash_protection and deviation_percent < 0:
                    # Store tick for flash crash detection
                    self._recent_ticks[symbol].append((timestamp, price, deviation_percent))

                    # Check for flash crash pattern
                    if self._detect_flash_crash(symbol, sym_config):
                        self._flash_crash_active[symbol] = True
                        self._flash_crash_start[symbol] = now

                        with self._stats_lock:
                            self._stats.flash_crash_activations += 1

                        logger.warning(f"🚨 Flash crash detected for {symbol}! Cooldown activated.")

                        return self._reject(
                            symbol, price, volume, timestamp,
                            FilterReason.FLASH_CRASH,
                            last_price, deviation_percent,
                            f"Flash crash detected: rapid price drops",
                            raw_tick
                        )

            # === Tick Accepted ===
            self._last_prices[symbol] = price
            self._last_price_times[symbol] = timestamp

            if self._config.track_statistics:
                with self._stats_lock:
                    self._stats.accepted_ticks += 1

            return (True, None)

    def _detect_flash_crash(self, symbol: str, config: SymbolFilterConfig) -> bool:
        """
        Detect flash crash pattern from recent ticks.

        Flash crash = multiple consecutive large drops within a short window.
        """
        recent = list(self._recent_ticks[symbol])
        if len(recent) < config.flash_crash_window:
            return False

        # Look at last N ticks
        window = recent[-config.flash_crash_window:]

        # Count consecutive drops exceeding threshold
        total_drop = sum(
            dev for _, _, dev in window
            if dev < -config.max_deviation_percent
        )

        # If cumulative drop exceeds flash crash threshold, trigger
        return abs(total_drop) >= config.flash_crash_threshold

    def _reject(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
        reason: FilterReason,
        last_valid_price: float,
        deviation_percent: float,
        message: str,
        raw_tick: Optional[Dict]
    ) -> Tuple[bool, FilterReason]:
        """Record rejection and return result"""

        # Create rejection record
        rejection = RejectedTick(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume=volume,
            reason=reason,
            last_valid_price=last_valid_price,
            deviation_percent=deviation_percent,
            message=message,
            raw_tick=raw_tick
        )

        # Log rejection
        if self._config.log_rejected_ticks:
            logger.warning(f"Tick rejected [{symbol}]: {message}")

        # Store in history
        if self._config.store_rejected_history:
            with self._rejected_lock:
                self._rejected_ticks.append(rejection)

        # Update statistics
        if self._config.track_statistics:
            with self._stats_lock:
                self._stats.rejected_ticks += 1
                self._stats.last_rejection_time = datetime.now()

                reason_key = reason.value
                self._stats.rejections_by_reason[reason_key] = \
                    self._stats.rejections_by_reason.get(reason_key, 0) + 1

                self._stats.rejections_by_symbol[symbol] = \
                    self._stats.rejections_by_symbol.get(symbol, 0) + 1

        # Notify callbacks
        for callback in self._rejection_callbacks:
            try:
                callback(rejection)
            except Exception as e:
                logger.error(f"Rejection callback error: {e}")

        return (False, reason)

    def wrap_live_source(self, live_source: Any) -> None:
        """
        Wrap a LiveDataSource to filter ticks automatically.

        Replaces the _handle_tick method with a filtered version.

        Args:
            live_source: LiveDataSource instance to wrap
        """
        if not hasattr(live_source, '_handle_tick'):
            logger.warning("LiveDataSource has no _handle_tick method - cannot wrap")
            return

        original_handler = live_source._handle_tick

        def filtered_handler(tick):
            # Extract tick data
            symbol = getattr(tick, 'symbol', '')
            if not symbol and hasattr(tick, 'instrument_token'):
                token_map = getattr(live_source, '_token_to_symbol', {})
                symbol = token_map.get(tick.instrument_token, '')

            price = getattr(tick, 'last_price', 0.0)
            volume = getattr(tick, 'volume', 0)
            timestamp = getattr(tick, 'timestamp', datetime.now())

            # Filter tick
            is_valid, reason = self.filter_tick(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                raw_tick=vars(tick) if hasattr(tick, '__dict__') else None
            )

            # Only pass through valid ticks
            if is_valid:
                original_handler(tick)

        live_source._handle_tick = filtered_handler
        logger.info("LiveDataSource wrapped with tick filter")

    def create_filter_wrapper(self) -> Callable:
        """
        Create a decorator for filtering tick callbacks.

        Returns:
            Decorator function

        Example:
            @tick_filter.create_filter_wrapper()
            def on_tick(tick):
                process_tick(tick)
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(tick, *args, **kwargs):
                symbol = getattr(tick, 'symbol', '')
                price = getattr(tick, 'last_price', 0.0)
                volume = getattr(tick, 'volume', 0)
                timestamp = getattr(tick, 'timestamp', datetime.now())

                is_valid, _ = self.filter_tick(symbol, price, volume, timestamp)
                if is_valid:
                    return func(tick, *args, **kwargs)
                return None
            return wrapper
        return decorator

    def add_rejection_callback(self, callback: Callable[[RejectedTick], None]) -> None:
        """Register callback for rejected ticks"""
        self._rejection_callbacks.append(callback)

    def remove_rejection_callback(self, callback: Callable[[RejectedTick], None]) -> bool:
        """Remove rejection callback"""
        try:
            self._rejection_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def get_rejected_ticks(
        self,
        symbol: Optional[str] = None,
        reason: Optional[FilterReason] = None,
        limit: int = 100
    ) -> List[RejectedTick]:
        """
        Get rejected tick history.

        Args:
            symbol: Filter by symbol
            reason: Filter by rejection reason
            limit: Max ticks to return

        Returns:
            List of rejected ticks
        """
        with self._rejected_lock:
            ticks = list(self._rejected_ticks)

        if symbol:
            ticks = [t for t in ticks if t.symbol == symbol]
        if reason:
            ticks = [t for t in ticks if t.reason == reason]

        return ticks[-limit:]

    def get_stats(self) -> TickFilterStats:
        """Get filter statistics"""
        with self._stats_lock:
            return TickFilterStats(
                total_ticks=self._stats.total_ticks,
                accepted_ticks=self._stats.accepted_ticks,
                rejected_ticks=self._stats.rejected_ticks,
                rejections_by_reason=dict(self._stats.rejections_by_reason),
                rejections_by_symbol=dict(self._stats.rejections_by_symbol),
                last_rejection_time=self._stats.last_rejection_time,
                flash_crash_activations=self._stats.flash_crash_activations
            )

    def reset_stats(self) -> None:
        """Reset statistics"""
        with self._stats_lock:
            self._stats = TickFilterStats()

    def clear_history(self) -> None:
        """Clear rejected tick history"""
        with self._rejected_lock:
            self._rejected_ticks.clear()

    def reset_symbol(self, symbol: str) -> None:
        """Reset tracking state for a symbol"""
        with self._lock:
            if symbol in self._last_prices:
                del self._last_prices[symbol]
            if symbol in self._last_price_times:
                del self._last_price_times[symbol]
            if symbol in self._recent_ticks:
                self._recent_ticks[symbol].clear()
            if symbol in self._flash_crash_active:
                self._flash_crash_active[symbol] = False

    def summary(self) -> str:
        """Get formatted summary of filter statistics"""
        stats = self.get_stats()

        lines = ["Tick Filter Summary:"]
        lines.append("-" * 50)
        lines.append(f"Total ticks:     {stats.total_ticks:,}")
        lines.append(f"Accepted:        {stats.accepted_ticks:,}")
        lines.append(f"Rejected:        {stats.rejected_ticks:,} ({stats.rejection_rate:.2f}%)")
        lines.append(f"Flash crashes:   {stats.flash_crash_activations}")

        if stats.rejections_by_reason:
            lines.append("\nRejections by reason:")
            for reason, count in sorted(stats.rejections_by_reason.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")

        if stats.rejections_by_symbol:
            lines.append("\nTop rejected symbols:")
            sorted_symbols = sorted(stats.rejections_by_symbol.items(), key=lambda x: -x[1])[:5]
            for symbol, count in sorted_symbols:
                lines.append(f"  {symbol}: {count}")

        return "\n".join(lines)


# =============================================================================
# Global Instance
# =============================================================================

_global_tick_filter: Optional[TickFilter] = None
_global_tick_filter_lock = threading.Lock()


def get_tick_filter() -> TickFilter:
    """Get the global tick filter instance"""
    global _global_tick_filter
    if _global_tick_filter is None:
        with _global_tick_filter_lock:
            if _global_tick_filter is None:
                _global_tick_filter = TickFilter()
    return _global_tick_filter


def set_tick_filter(tick_filter: TickFilter) -> None:
    """Set the global tick filter instance"""
    global _global_tick_filter
    with _global_tick_filter_lock:
        _global_tick_filter = tick_filter
