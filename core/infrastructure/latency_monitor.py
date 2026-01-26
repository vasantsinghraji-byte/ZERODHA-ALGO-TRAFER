"""
Latency Monitor for Trading System
Tracks and analyzes latency across the trading pipeline.

Measures tick-to-order latency, exchange vs local timestamp drift,
computes percentiles, and alerts on lag spikes via EventBus.
"""

import bisect
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LatencyType(Enum):
    """Types of latency measurements"""
    TICK_TO_SIGNAL = "tick_to_signal"       # Time from tick to strategy signal
    SIGNAL_TO_ORDER = "signal_to_order"     # Time from signal to order submission
    TICK_TO_ORDER = "tick_to_order"         # End-to-end latency
    ORDER_TO_FILL = "order_to_fill"         # Order submission to fill
    EXCHANGE_DRIFT = "exchange_drift"       # Exchange timestamp vs local time
    EVENT_DELIVERY = "event_delivery"       # EventBus delivery time
    DATA_FEED = "data_feed"                 # Data feed latency


class AlertSeverity(Enum):
    """Severity levels for latency alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LatencyAlert:
    """Latency alert data structure"""
    latency_type: LatencyType
    severity: AlertSeverity
    latency_ms: float
    threshold_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict = field(default_factory=dict)


@dataclass
class LatencyThresholds:
    """Configurable thresholds for latency alerts"""
    warning_ms: float = 200.0      # Warning threshold
    critical_ms: float = 500.0     # Critical threshold (alert)
    info_ms: float = 100.0         # Info threshold (logging only)

    def __post_init__(self):
        if self.info_ms >= self.warning_ms:
            raise ValueError("info_ms must be less than warning_ms")
        if self.warning_ms >= self.critical_ms:
            raise ValueError("warning_ms must be less than critical_ms")


@dataclass
class LatencyMonitorConfig:
    """Configuration for the latency monitor"""
    # Per-type thresholds
    thresholds: Dict[str, LatencyThresholds] = field(default_factory=dict)

    # History settings
    history_size: int = 10000           # Max samples per type
    percentile_window: int = 1000       # Samples for percentile calculation

    # Alert settings
    alert_cooldown_seconds: float = 30.0  # Min time between same alerts
    enable_alerts: bool = True

    # EventBus integration
    publish_to_eventbus: bool = True

    def __post_init__(self):
        # Set defaults for known latency types
        defaults = {
            LatencyType.TICK_TO_ORDER.value: LatencyThresholds(
                info_ms=50.0, warning_ms=200.0, critical_ms=500.0
            ),
            LatencyType.TICK_TO_SIGNAL.value: LatencyThresholds(
                info_ms=20.0, warning_ms=100.0, critical_ms=300.0
            ),
            LatencyType.SIGNAL_TO_ORDER.value: LatencyThresholds(
                info_ms=10.0, warning_ms=50.0, critical_ms=150.0
            ),
            LatencyType.ORDER_TO_FILL.value: LatencyThresholds(
                info_ms=100.0, warning_ms=500.0, critical_ms=1000.0
            ),
            LatencyType.EXCHANGE_DRIFT.value: LatencyThresholds(
                info_ms=100.0, warning_ms=300.0, critical_ms=1000.0
            ),
            LatencyType.EVENT_DELIVERY.value: LatencyThresholds(
                info_ms=5.0, warning_ms=20.0, critical_ms=50.0
            ),
            LatencyType.DATA_FEED.value: LatencyThresholds(
                info_ms=50.0, warning_ms=200.0, critical_ms=500.0
            ),
        }

        for lt, thresholds in defaults.items():
            if lt not in self.thresholds:
                self.thresholds[lt] = thresholds


class LatencyHistogram:
    """
    Efficient histogram for latency percentile calculations.

    Uses a sorted list for accurate percentiles with O(log n) insertion.
    Maintains a sliding window of recent samples.
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum samples to keep (sliding window)
        """
        self._max_size = max_size
        self._samples: List[float] = []
        self._insertion_order: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # Cached statistics
        self._sum = 0.0
        self._count = 0
        self._min = float('inf')
        self._max = float('-inf')

    def add(self, value_ms: float) -> None:
        """Add a latency sample"""
        with self._lock:
            # If at capacity, remove oldest
            if len(self._samples) >= self._max_size:
                oldest = self._insertion_order.popleft()
                idx = bisect.bisect_left(self._samples, oldest)
                if idx < len(self._samples) and self._samples[idx] == oldest:
                    self._samples.pop(idx)
                    self._sum -= oldest
                    self._count -= 1

            # Insert in sorted position
            bisect.insort(self._samples, value_ms)
            self._insertion_order.append(value_ms)

            # Update stats
            self._sum += value_ms
            self._count += 1
            self._min = min(self._min, value_ms)
            self._max = max(self._max, value_ms)

    def percentile(self, p: float) -> float:
        """
        Get the p-th percentile.

        Args:
            p: Percentile (0-100)

        Returns:
            Value at percentile, or 0 if no samples
        """
        with self._lock:
            if not self._samples:
                return 0.0

            idx = int((p / 100.0) * (len(self._samples) - 1))
            return self._samples[idx]

    def percentiles(self, ps: List[float]) -> Dict[str, float]:
        """Get multiple percentiles at once"""
        with self._lock:
            if not self._samples:
                return {f"p{int(p)}": 0.0 for p in ps}

            result = {}
            n = len(self._samples)
            for p in ps:
                idx = int((p / 100.0) * (n - 1))
                result[f"p{int(p)}"] = self._samples[idx]
            return result

    @property
    def mean(self) -> float:
        """Average latency"""
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    @property
    def min_value(self) -> float:
        """Minimum latency"""
        with self._lock:
            return self._min if self._min != float('inf') else 0.0

    @property
    def max_value(self) -> float:
        """Maximum latency"""
        with self._lock:
            return self._max if self._max != float('-inf') else 0.0

    @property
    def count(self) -> int:
        """Number of samples"""
        with self._lock:
            return self._count

    def clear(self) -> None:
        """Reset histogram"""
        with self._lock:
            self._samples.clear()
            self._insertion_order.clear()
            self._sum = 0.0
            self._count = 0
            self._min = float('inf')
            self._max = float('-inf')


@dataclass
class LatencyStats:
    """Statistics for a latency type"""
    latency_type: str
    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_value_ms: float = 0.0
    last_timestamp: Optional[datetime] = None
    alerts_triggered: int = 0


class LatencyMonitor:
    """
    Comprehensive latency monitoring for the trading system.

    Tracks multiple latency types, computes percentiles, and publishes
    alerts to EventBus when thresholds are exceeded.

    Thread-safe implementation for use in multi-threaded trading engines.

    Example:
        monitor = LatencyMonitor()

        # Record tick-to-order latency
        monitor.record(LatencyType.TICK_TO_ORDER, 150.0)

        # Or use timing context
        with monitor.measure(LatencyType.TICK_TO_SIGNAL):
            strategy.process_tick(tick)

        # Get statistics
        stats = monitor.get_stats(LatencyType.TICK_TO_ORDER)
        print(f"p99: {stats.p99_ms}ms")

        # Track exchange drift
        monitor.record_exchange_drift(exchange_ts, local_ts)
    """

    def __init__(
        self,
        config: Optional[LatencyMonitorConfig] = None,
        event_bus: Optional['EventBus'] = None
    ):
        """
        Args:
            config: Monitor configuration
            event_bus: EventBus instance for publishing alerts (uses global if None)
        """
        self._config = config or LatencyMonitorConfig()
        self._event_bus = event_bus
        self._lock = threading.RLock()

        # Histograms per latency type
        self._histograms: Dict[str, LatencyHistogram] = {}
        for lt in LatencyType:
            self._histograms[lt.value] = LatencyHistogram(
                max_size=self._config.percentile_window
            )

        # Alert tracking
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, int] = {lt.value: 0 for lt in LatencyType}

        # Last values for quick access
        self._last_values: Dict[str, Tuple[float, datetime]] = {}

        # Callbacks for custom alert handling
        self._alert_callbacks: List[Callable[[LatencyAlert], None]] = []

        logger.debug("LatencyMonitor initialized")

    def _get_event_bus(self):
        """Get EventBus instance (lazy import to avoid circular deps)"""
        if self._event_bus is not None:
            return self._event_bus

        try:
            from core.events.event_bus import get_event_bus
            return get_event_bus()
        except ImportError:
            logger.warning("EventBus not available for latency alerts")
            return None

    def record(
        self,
        latency_type: LatencyType,
        latency_ms: float,
        context: Optional[Dict] = None
    ) -> None:
        """
        Record a latency measurement.

        Args:
            latency_type: Type of latency being measured
            latency_ms: Latency value in milliseconds
            context: Optional context for alerts (symbol, order_id, etc.)
        """
        type_key = latency_type.value
        now = datetime.now()

        with self._lock:
            # Add to histogram
            histogram = self._histograms.get(type_key)
            if histogram:
                histogram.add(latency_ms)

            # Update last value
            self._last_values[type_key] = (latency_ms, now)

        # Check thresholds and alert
        if self._config.enable_alerts:
            self._check_and_alert(latency_type, latency_ms, context or {})

        # Debug logging for high latencies
        thresholds = self._config.thresholds.get(type_key)
        if thresholds and latency_ms >= thresholds.info_ms:
            logger.debug(f"Latency [{type_key}]: {latency_ms:.2f}ms")

    def record_exchange_drift(
        self,
        exchange_timestamp: datetime,
        local_timestamp: Optional[datetime] = None
    ) -> float:
        """
        Record drift between exchange and local timestamps.

        Args:
            exchange_timestamp: Timestamp from exchange
            local_timestamp: Local timestamp (default: now)

        Returns:
            Drift in milliseconds (positive = exchange ahead)
        """
        local_ts = local_timestamp or datetime.now()
        drift_ms = (local_ts - exchange_timestamp).total_seconds() * 1000

        self.record(
            LatencyType.EXCHANGE_DRIFT,
            abs(drift_ms),
            context={"direction": "ahead" if drift_ms < 0 else "behind"}
        )

        return drift_ms

    def record_tick_to_order(
        self,
        tick_timestamp: datetime,
        order_timestamp: Optional[datetime] = None,
        symbol: str = ""
    ) -> float:
        """
        Convenience method for tick-to-order latency.

        Args:
            tick_timestamp: When tick was received
            order_timestamp: When order was submitted (default: now)
            symbol: Trading symbol for context

        Returns:
            Latency in milliseconds
        """
        order_ts = order_timestamp or datetime.now()
        latency_ms = (order_ts - tick_timestamp).total_seconds() * 1000

        self.record(
            LatencyType.TICK_TO_ORDER,
            latency_ms,
            context={"symbol": symbol} if symbol else None
        )

        return latency_ms

    def measure(self, latency_type: LatencyType, context: Optional[Dict] = None):
        """
        Context manager for measuring operation latency.

        Args:
            latency_type: Type of latency being measured
            context: Optional context for alerts

        Example:
            with monitor.measure(LatencyType.SIGNAL_TO_ORDER):
                broker.place_order(...)
        """
        return _LatencyMeasureContext(self, latency_type, context)

    def _check_and_alert(
        self,
        latency_type: LatencyType,
        latency_ms: float,
        context: Dict
    ) -> None:
        """Check thresholds and publish alert if exceeded"""
        type_key = latency_type.value
        thresholds = self._config.thresholds.get(type_key)

        if not thresholds:
            return

        # Determine severity
        if latency_ms >= thresholds.critical_ms:
            severity = AlertSeverity.CRITICAL
        elif latency_ms >= thresholds.warning_ms:
            severity = AlertSeverity.WARNING
        else:
            return  # Below alert threshold

        # Check cooldown
        now = datetime.now()
        alert_key = f"{type_key}_{severity.value}"
        last_alert = self._last_alert_time.get(alert_key)

        if last_alert:
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self._config.alert_cooldown_seconds:
                return  # Still in cooldown

        # Create alert
        threshold_ms = (
            thresholds.critical_ms if severity == AlertSeverity.CRITICAL
            else thresholds.warning_ms
        )

        alert = LatencyAlert(
            latency_type=latency_type,
            severity=severity,
            latency_ms=latency_ms,
            threshold_ms=threshold_ms,
            message=f"Latency spike: {type_key} = {latency_ms:.1f}ms (threshold: {threshold_ms:.1f}ms)",
            context=context
        )

        # Update tracking
        with self._lock:
            self._last_alert_time[alert_key] = now
            self._alert_counts[type_key] = self._alert_counts.get(type_key, 0) + 1

        # Log
        if severity == AlertSeverity.CRITICAL:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)

        # Publish to EventBus
        if self._config.publish_to_eventbus:
            self._publish_alert(alert)

        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _publish_alert(self, alert: LatencyAlert) -> None:
        """Publish alert to EventBus as SystemEvent"""
        event_bus = self._get_event_bus()
        if not event_bus:
            return

        try:
            from core.events.events import SystemEvent, ErrorEvent, EventType

            # Use ErrorEvent for critical alerts, SystemEvent for warnings
            if alert.severity == AlertSeverity.CRITICAL:
                event = ErrorEvent(
                    event_type=EventType.ERROR,
                    error_code=f"LATENCY_{alert.latency_type.value.upper()}",
                    message=alert.message,
                    data={
                        "latency_type": alert.latency_type.value,
                        "latency_ms": alert.latency_ms,
                        "threshold_ms": alert.threshold_ms,
                        "severity": alert.severity.value,
                        "context": alert.context
                    },
                    source="latency_monitor"
                )
            else:
                event = SystemEvent(
                    event_type=EventType.HEARTBEAT,  # Using heartbeat for non-critical
                    message=alert.message,
                    data={
                        "latency_type": alert.latency_type.value,
                        "latency_ms": alert.latency_ms,
                        "threshold_ms": alert.threshold_ms,
                        "severity": alert.severity.value,
                        "context": alert.context,
                        "alert_type": "latency_warning"
                    },
                    source="latency_monitor"
                )

            event_bus.publish(event)

        except Exception as e:
            logger.error(f"Failed to publish latency alert: {e}")

    def add_alert_callback(self, callback: Callable[[LatencyAlert], None]) -> None:
        """Register a callback for latency alerts"""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[LatencyAlert], None]) -> bool:
        """Remove an alert callback"""
        try:
            self._alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def get_stats(self, latency_type: LatencyType) -> LatencyStats:
        """
        Get statistics for a latency type.

        Args:
            latency_type: Type to get stats for

        Returns:
            LatencyStats with count, mean, min, max, percentiles
        """
        type_key = latency_type.value

        with self._lock:
            histogram = self._histograms.get(type_key)
            if not histogram or histogram.count == 0:
                return LatencyStats(latency_type=type_key)

            percentiles = histogram.percentiles([50, 95, 99])
            last_value, last_ts = self._last_values.get(type_key, (0.0, None))

            return LatencyStats(
                latency_type=type_key,
                count=histogram.count,
                mean_ms=histogram.mean,
                min_ms=histogram.min_value,
                max_ms=histogram.max_value,
                p50_ms=percentiles.get("p50", 0.0),
                p95_ms=percentiles.get("p95", 0.0),
                p99_ms=percentiles.get("p99", 0.0),
                last_value_ms=last_value,
                last_timestamp=last_ts,
                alerts_triggered=self._alert_counts.get(type_key, 0)
            )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all latency types"""
        return {lt.value: self.get_stats(lt) for lt in LatencyType}

    def get_percentiles(
        self,
        latency_type: LatencyType,
        percentiles: List[float] = None
    ) -> Dict[str, float]:
        """
        Get specific percentiles for a latency type.

        Args:
            latency_type: Type to get percentiles for
            percentiles: List of percentiles (default: [50, 95, 99])

        Returns:
            Dict mapping "pXX" to value
        """
        percentiles = percentiles or [50, 95, 99]
        type_key = latency_type.value

        with self._lock:
            histogram = self._histograms.get(type_key)
            if not histogram:
                return {f"p{int(p)}": 0.0 for p in percentiles}
            return histogram.percentiles(percentiles)

    def reset(self, latency_type: Optional[LatencyType] = None) -> None:
        """
        Reset statistics.

        Args:
            latency_type: Specific type to reset, or None for all
        """
        with self._lock:
            if latency_type:
                type_key = latency_type.value
                if type_key in self._histograms:
                    self._histograms[type_key].clear()
                if type_key in self._last_values:
                    del self._last_values[type_key]
                self._alert_counts[type_key] = 0
            else:
                for histogram in self._histograms.values():
                    histogram.clear()
                self._last_values.clear()
                self._last_alert_time.clear()
                self._alert_counts = {lt.value: 0 for lt in LatencyType}

        logger.debug(f"LatencyMonitor reset: {latency_type or 'all'}")

    def summary(self) -> str:
        """Get a formatted summary of all latency statistics"""
        lines = ["Latency Summary:"]
        lines.append("-" * 60)
        lines.append(f"{'Type':<20} {'Count':>8} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
        lines.append("-" * 60)

        for lt in LatencyType:
            stats = self.get_stats(lt)
            if stats.count > 0:
                lines.append(
                    f"{lt.value:<20} {stats.count:>8} "
                    f"{stats.mean_ms:>7.1f}ms {stats.p50_ms:>7.1f}ms "
                    f"{stats.p95_ms:>7.1f}ms {stats.p99_ms:>7.1f}ms"
                )

        return "\n".join(lines)


class _LatencyMeasureContext:
    """Context manager for latency measurement"""

    def __init__(
        self,
        monitor: LatencyMonitor,
        latency_type: LatencyType,
        context: Optional[Dict]
    ):
        self._monitor = monitor
        self._latency_type = latency_type
        self._context = context
        self._start_time: float = 0

    def __enter__(self) -> '_LatencyMeasureContext':
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._monitor.record(self._latency_type, elapsed_ms, self._context)


# Module-level convenience instance
_global_monitor: Optional[LatencyMonitor] = None
_global_monitor_lock = threading.Lock()


def get_latency_monitor() -> LatencyMonitor:
    """Get the global latency monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        with _global_monitor_lock:
            if _global_monitor is None:
                _global_monitor = LatencyMonitor()
    return _global_monitor


def set_latency_monitor(monitor: LatencyMonitor) -> None:
    """Set the global latency monitor instance"""
    global _global_monitor
    with _global_monitor_lock:
        _global_monitor = monitor


def reset_latency_monitor() -> None:
    """Reset the global latency monitor"""
    global _global_monitor
    with _global_monitor_lock:
        _global_monitor = None
