# -*- coding: utf-8 -*-
"""
Concept Drift Detection - Model Performance Monitoring
=======================================================

⚠️  PHASE 12 FEATURE - PREREQUISITES NOT MET ⚠️

DO NOT USE until these items are working:
1. ✗ ML models are trained and producing predictions
2. ✗ MLEngine is integrated with live trading
3. ✗ Predictions are being used to generate signals
4. ✗ Ground truth (actual outcomes) is being recorded

WHY THIS IS PREMATURE:
- You can't detect drift in a model that isn't running
- ml_engine.py was corrupted and class names don't match bootstrap
- No predictions = no drift detection possible

WHEN TO USE THIS:
- After ML models are generating live predictions
- When you have weeks of prediction vs actual data
- For monitoring production model degradation

PRIORITY: Fix ml_engine.py class mismatches first!
- Bootstrap expects: MLEngine
- ml_engine.py defines: ??? (check current state)

------------------------------------------------------------------------

Production-grade concept drift detection for ML models in trading.

Concept drift occurs when the statistical relationship between
input features and target variable changes over time, causing
model performance degradation.

Features:
- Real-time model accuracy monitoring
- Statistical drift tests (KS-test, PSI, Page-Hinkley)
- Configurable alert thresholds
- Automatic retraining triggers

Example (ONLY AFTER PREREQUISITES MET):
    >>> from ml.drift_detector import ConceptDriftDetector, AccuracyMonitor
    >>>
    >>> # Create detector
    >>> detector = ConceptDriftDetector(
    ...     accuracy_threshold=0.6,
    ...     drift_sensitivity=0.05
    ... )
    >>>
    >>> # Monitor predictions
    >>> detector.record_prediction(predicted=1, actual=1)
    >>> detector.record_prediction(predicted=1, actual=0)
    >>>
    >>> # Check for drift
    >>> if detector.is_drift_detected():
    ...     print("Concept drift detected! Triggering retraining...")
    ...     detector.trigger_retraining()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Deque
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import logging
import threading
import time

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity levels for concept drift."""
    NONE = "none"
    WARNING = "warning"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class DriftTestType(Enum):
    """Types of drift detection tests."""
    PAGE_HINKLEY = "page_hinkley"
    ADWIN = "adwin"
    DDM = "ddm"  # Drift Detection Method
    EDDM = "eddm"  # Early Drift Detection Method
    KS_TEST = "ks_test"
    PSI = "psi"


class AlertType(Enum):
    """Types of alerts."""
    ACCURACY_DROP = "accuracy_drop"
    DRIFT_DETECTED = "drift_detected"
    RETRAINING_TRIGGERED = "retraining_triggered"
    RETRAINING_COMPLETED = "retraining_completed"
    MODEL_SWITCHED = "model_switched"


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    timestamp: datetime
    predicted: Any
    actual: Any
    probability: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    is_correct: bool = field(init=False)

    def __post_init__(self):
        self.is_correct = self.predicted == self.actual


@dataclass
class AccuracyWindow:
    """Accuracy metrics over a time window."""
    window_start: datetime
    window_end: datetime
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    alert_id: str
    alert_type: AlertType
    severity: DriftSeverity
    timestamp: datetime
    message: str
    metrics: Dict[str, float]
    model_id: Optional[str] = None
    acknowledged: bool = False

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True


@dataclass
class DriftTestResult:
    """Result of a drift detection test."""
    test_type: DriftTestType
    is_drift_detected: bool
    statistic: float
    p_value: Optional[float]
    threshold: float
    severity: DriftSeverity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingRequest:
    """Request to retrain a model."""
    request_id: str
    model_id: str
    timestamp: datetime
    reason: str
    priority: int  # 1 = highest
    metrics_snapshot: Dict[str, float]
    status: str = "pending"  # pending, in_progress, completed, failed


class AccuracyMonitor:
    """
    Real-time model accuracy monitoring.

    Tracks prediction accuracy over sliding windows and computes
    various performance metrics.
    """

    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 100,
        metrics_interval: timedelta = timedelta(minutes=5)
    ):
        """
        Initialize accuracy monitor.

        Args:
            window_size: Number of predictions to keep in sliding window
            min_samples: Minimum samples before computing metrics
            metrics_interval: Interval for computing windowed metrics
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.metrics_interval = metrics_interval

        self._predictions: Deque[PredictionRecord] = deque(maxlen=window_size)
        self._accuracy_history: List[AccuracyWindow] = []
        self._lock = threading.Lock()

        # Confusion matrix components for binary classification
        self._tp = 0  # True positives
        self._fp = 0  # False positives
        self._tn = 0  # True negatives
        self._fn = 0  # False negatives

    def record_prediction(
        self,
        predicted: Any,
        actual: Any,
        probability: Optional[float] = None,
        features: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> PredictionRecord:
        """
        Record a prediction and its actual outcome.

        Args:
            predicted: Model's prediction
            actual: Actual observed value
            probability: Prediction probability (if available)
            features: Input features used for prediction
            timestamp: When prediction was made

        Returns:
            The recorded prediction
        """
        if timestamp is None:
            timestamp = datetime.now()

        record = PredictionRecord(
            timestamp=timestamp,
            predicted=predicted,
            actual=actual,
            probability=probability,
            features=features
        )

        with self._lock:
            # Update confusion matrix for binary classification
            if predicted in (0, 1) and actual in (0, 1):
                if predicted == 1 and actual == 1:
                    self._tp += 1
                elif predicted == 1 and actual == 0:
                    self._fp += 1
                elif predicted == 0 and actual == 0:
                    self._tn += 1
                else:  # predicted == 0 and actual == 1
                    self._fn += 1

            # Handle window overflow - remove oldest and adjust counts
            if len(self._predictions) == self.window_size:
                old = self._predictions[0]
                if old.predicted in (0, 1) and old.actual in (0, 1):
                    if old.predicted == 1 and old.actual == 1:
                        self._tp -= 1
                    elif old.predicted == 1 and old.actual == 0:
                        self._fp -= 1
                    elif old.predicted == 0 and old.actual == 0:
                        self._tn -= 1
                    else:
                        self._fn -= 1

            self._predictions.append(record)

        return record

    def get_current_accuracy(self) -> Optional[float]:
        """Get current accuracy over the sliding window."""
        with self._lock:
            if len(self._predictions) < self.min_samples:
                return None

            correct = sum(1 for p in self._predictions if p.is_correct)
            return correct / len(self._predictions)

    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary with accuracy, precision, recall, F1, etc.
        """
        with self._lock:
            total = len(self._predictions)

            if total < self.min_samples:
                return {
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'total_predictions': total,
                    'min_samples_required': self.min_samples
                }

            correct = sum(1 for p in self._predictions if p.is_correct)
            accuracy = correct / total

            # Binary classification metrics
            precision = None
            recall = None
            f1 = None

            if self._tp + self._fp > 0:
                precision = self._tp / (self._tp + self._fp)

            if self._tp + self._fn > 0:
                recall = self._tp / (self._tp + self._fn)

            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_predictions': total,
                'true_positives': self._tp,
                'false_positives': self._fp,
                'true_negatives': self._tn,
                'false_negatives': self._fn
            }

    def get_accuracy_trend(self, num_windows: int = 10) -> List[float]:
        """
        Get accuracy trend over recent windows.

        Args:
            num_windows: Number of historical windows to include

        Returns:
            List of accuracy values (oldest to newest)
        """
        with self._lock:
            if len(self._predictions) < self.min_samples:
                return []

            predictions = list(self._predictions)
            window_size = len(predictions) // num_windows

            if window_size < 10:
                return [self.get_current_accuracy()]

            accuracies = []
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_preds = predictions[start_idx:end_idx]

                if window_preds:
                    correct = sum(1 for p in window_preds if p.is_correct)
                    accuracies.append(correct / len(window_preds))

            return accuracies

    def get_recent_errors(self, n: int = 10) -> List[PredictionRecord]:
        """Get the most recent incorrect predictions."""
        with self._lock:
            errors = [p for p in self._predictions if not p.is_correct]
            return list(errors)[-n:]

    def reset(self) -> None:
        """Reset all monitoring state."""
        with self._lock:
            self._predictions.clear()
            self._accuracy_history.clear()
            self._tp = self._fp = self._tn = self._fn = 0


class DriftDetector(ABC):
    """Abstract base class for drift detection algorithms."""

    @abstractmethod
    def update(self, value: float) -> bool:
        """
        Update detector with new value.

        Args:
            value: New observation (typically 0 for correct, 1 for error)

        Returns:
            True if drift is detected
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        pass


class PageHinkleyDetector(DriftDetector):
    """
    Page-Hinkley drift detection algorithm.

    Detects changes in the mean of a sequence by tracking
    cumulative sum deviations from the mean.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999
    ):
        """
        Initialize Page-Hinkley detector.

        Args:
            delta: Minimum magnitude of change to detect
            threshold: Detection threshold
            alpha: Forgetting factor for mean estimation
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self._sum = 0.0
        self._mean = 0.0
        self._min_sum = float('inf')
        self._count = 0
        self._drift_detected = False

    def update(self, value: float) -> bool:
        """Update detector with new value."""
        self._count += 1

        # Update running mean
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value

        # Update cumulative sum
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        # Check for drift
        self._drift_detected = (self._sum - self._min_sum) > self.threshold

        return self._drift_detected

    def reset(self) -> None:
        """Reset detector state."""
        self._sum = 0.0
        self._mean = 0.0
        self._min_sum = float('inf')
        self._count = 0
        self._drift_detected = False

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            'algorithm': 'Page-Hinkley',
            'count': self._count,
            'mean': self._mean,
            'cumulative_sum': self._sum,
            'min_sum': self._min_sum,
            'difference': self._sum - self._min_sum,
            'threshold': self.threshold,
            'drift_detected': self._drift_detected
        }


class DDMDetector(DriftDetector):
    """
    Drift Detection Method (DDM).

    Monitors error rate and standard deviation, triggering
    warnings and drift alerts based on statistical thresholds.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30
    ):
        """
        Initialize DDM detector.

        Args:
            warning_level: Standard deviations for warning
            drift_level: Standard deviations for drift
            min_samples: Minimum samples before detection
        """
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples

        self._error_count = 0
        self._sample_count = 0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._in_warning = False
        self._drift_detected = False

    def update(self, is_error: float) -> bool:
        """
        Update detector with prediction result.

        Args:
            is_error: 1 if prediction was wrong, 0 if correct
        """
        self._sample_count += 1
        self._error_count += is_error

        if self._sample_count < self.min_samples:
            return False

        # Calculate error rate and standard deviation
        p = self._error_count / self._sample_count
        s = np.sqrt(p * (1 - p) / self._sample_count)

        # Update minimums
        if p + s < self._p_min + self._s_min:
            self._p_min = p
            self._s_min = s

        # Check thresholds
        if p + s >= self._p_min + self.drift_level * self._s_min:
            self._drift_detected = True
            return True
        elif p + s >= self._p_min + self.warning_level * self._s_min:
            self._in_warning = True
        else:
            self._in_warning = False

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self._error_count = 0
        self._sample_count = 0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._in_warning = False
        self._drift_detected = False

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        p = self._error_count / max(1, self._sample_count)
        s = np.sqrt(p * (1 - p) / max(1, self._sample_count))

        return {
            'algorithm': 'DDM',
            'sample_count': self._sample_count,
            'error_rate': p,
            'std_dev': s,
            'p_min': self._p_min,
            's_min': self._s_min,
            'in_warning': self._in_warning,
            'drift_detected': self._drift_detected
        }


class EDDMDetector(DriftDetector):
    """
    Early Drift Detection Method (EDDM).

    Improvement over DDM that detects gradual drift earlier
    by monitoring distance between errors.
    """

    def __init__(
        self,
        warning_level: float = 0.95,
        drift_level: float = 0.90,
        min_errors: int = 30
    ):
        """
        Initialize EDDM detector.

        Args:
            warning_level: Ratio threshold for warning
            drift_level: Ratio threshold for drift
            min_errors: Minimum errors before detection
        """
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_errors = min_errors

        self._error_count = 0
        self._sample_count = 0
        self._last_error_idx = 0
        self._mean_distance = 0.0
        self._std_distance = 0.0
        self._max_metric = 0.0
        self._in_warning = False
        self._drift_detected = False

    def update(self, is_error: float) -> bool:
        """Update detector with prediction result."""
        self._sample_count += 1

        if is_error:
            self._error_count += 1

            if self._error_count > 1:
                distance = self._sample_count - self._last_error_idx

                # Update mean and std of distance
                old_mean = self._mean_distance
                self._mean_distance += (distance - old_mean) / self._error_count
                self._std_distance = np.sqrt(
                    self._std_distance ** 2 +
                    (distance - old_mean) * (distance - self._mean_distance)
                )

            self._last_error_idx = self._sample_count

        if self._error_count < self.min_errors:
            return False

        # Calculate metric
        metric = self._mean_distance + 2 * self._std_distance

        if metric > self._max_metric:
            self._max_metric = metric

        # Check thresholds
        ratio = metric / self._max_metric if self._max_metric > 0 else 1.0

        if ratio < self.drift_level:
            self._drift_detected = True
            return True
        elif ratio < self.warning_level:
            self._in_warning = True
        else:
            self._in_warning = False

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self._error_count = 0
        self._sample_count = 0
        self._last_error_idx = 0
        self._mean_distance = 0.0
        self._std_distance = 0.0
        self._max_metric = 0.0
        self._in_warning = False
        self._drift_detected = False

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        metric = self._mean_distance + 2 * self._std_distance
        ratio = metric / self._max_metric if self._max_metric > 0 else 1.0

        return {
            'algorithm': 'EDDM',
            'sample_count': self._sample_count,
            'error_count': self._error_count,
            'mean_distance': self._mean_distance,
            'std_distance': self._std_distance,
            'current_metric': metric,
            'max_metric': self._max_metric,
            'ratio': ratio,
            'in_warning': self._in_warning,
            'drift_detected': self._drift_detected
        }


class AlertManager:
    """
    Manages drift alerts and notifications.

    Provides configurable alerting with multiple severity levels
    and callback support for integration with external systems.
    """

    def __init__(
        self,
        cooldown_period: timedelta = timedelta(minutes=15),
        max_alerts_per_hour: int = 10
    ):
        """
        Initialize alert manager.

        Args:
            cooldown_period: Minimum time between alerts of same type
            max_alerts_per_hour: Rate limit for alerts
        """
        self.cooldown_period = cooldown_period
        self.max_alerts_per_hour = max_alerts_per_hour

        self._alerts: List[DriftAlert] = []
        self._callbacks: List[Callable[[DriftAlert], None]] = []
        self._last_alert_time: Dict[AlertType, datetime] = {}
        self._alert_count_window: Deque[datetime] = deque()
        self._lock = threading.Lock()
        self._alert_counter = 0

    def register_callback(
        self,
        callback: Callable[[DriftAlert], None]
    ) -> None:
        """Register a callback to be called when alerts are raised."""
        self._callbacks.append(callback)

    def raise_alert(
        self,
        alert_type: AlertType,
        severity: DriftSeverity,
        message: str,
        metrics: Dict[str, float],
        model_id: Optional[str] = None
    ) -> Optional[DriftAlert]:
        """
        Raise a drift alert.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Human-readable message
            metrics: Associated metrics
            model_id: Optional model identifier

        Returns:
            The created alert, or None if rate-limited
        """
        now = datetime.now()

        with self._lock:
            # Check cooldown
            if alert_type in self._last_alert_time:
                elapsed = now - self._last_alert_time[alert_type]
                if elapsed < self.cooldown_period:
                    logger.debug(f"Alert {alert_type} in cooldown, skipping")
                    return None

            # Check rate limit
            hour_ago = now - timedelta(hours=1)
            while self._alert_count_window and self._alert_count_window[0] < hour_ago:
                self._alert_count_window.popleft()

            if len(self._alert_count_window) >= self.max_alerts_per_hour:
                logger.warning("Alert rate limit reached, skipping")
                return None

            # Create alert
            self._alert_counter += 1
            alert = DriftAlert(
                alert_id=f"alert_{self._alert_counter}_{now.strftime('%Y%m%d%H%M%S')}",
                alert_type=alert_type,
                severity=severity,
                timestamp=now,
                message=message,
                metrics=metrics,
                model_id=model_id
            )

            self._alerts.append(alert)
            self._last_alert_time[alert_type] = now
            self._alert_count_window.append(now)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"[{severity.value.upper()}] {alert_type.value}: {message}")

        return alert

    def get_alerts(
        self,
        severity: Optional[DriftSeverity] = None,
        alert_type: Optional[AlertType] = None,
        since: Optional[datetime] = None,
        unacknowledged_only: bool = False
    ) -> List[DriftAlert]:
        """Get alerts matching criteria."""
        with self._lock:
            alerts = list(self._alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledge()
                    return True
        return False

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()


class RetrainingManager:
    """
    Manages automatic model retraining.

    Coordinates retraining requests, prioritization, and
    execution callbacks.
    """

    def __init__(
        self,
        min_interval: timedelta = timedelta(hours=1),
        max_queue_size: int = 10
    ):
        """
        Initialize retraining manager.

        Args:
            min_interval: Minimum time between retraining runs
            max_queue_size: Maximum pending retraining requests
        """
        self.min_interval = min_interval
        self.max_queue_size = max_queue_size

        self._requests: List[RetrainingRequest] = []
        self._last_retrain: Dict[str, datetime] = {}
        self._retrain_callback: Optional[Callable[[RetrainingRequest], bool]] = None
        self._lock = threading.Lock()
        self._request_counter = 0

    def set_retrain_callback(
        self,
        callback: Callable[[RetrainingRequest], bool]
    ) -> None:
        """Set the callback for executing retraining."""
        self._retrain_callback = callback

    def request_retraining(
        self,
        model_id: str,
        reason: str,
        priority: int,
        metrics_snapshot: Dict[str, float]
    ) -> Optional[RetrainingRequest]:
        """
        Request model retraining.

        Args:
            model_id: Model to retrain
            reason: Why retraining is needed
            priority: Priority level (1 = highest)
            metrics_snapshot: Current metrics

        Returns:
            Retraining request, or None if rejected
        """
        now = datetime.now()

        with self._lock:
            # Check minimum interval
            if model_id in self._last_retrain:
                elapsed = now - self._last_retrain[model_id]
                if elapsed < self.min_interval:
                    logger.info(
                        f"Retraining for {model_id} rejected: "
                        f"last retrain was {elapsed} ago"
                    )
                    return None

            # Check queue size
            pending = [r for r in self._requests if r.status == "pending"]
            if len(pending) >= self.max_queue_size:
                logger.warning("Retraining queue full, rejecting request")
                return None

            # Create request
            self._request_counter += 1
            request = RetrainingRequest(
                request_id=f"retrain_{self._request_counter}_{now.strftime('%Y%m%d%H%M%S')}",
                model_id=model_id,
                timestamp=now,
                reason=reason,
                priority=priority,
                metrics_snapshot=metrics_snapshot
            )

            self._requests.append(request)
            logger.info(f"Retraining requested for {model_id}: {reason}")

        return request

    def process_queue(self) -> List[RetrainingRequest]:
        """
        Process pending retraining requests.

        Returns:
            List of processed requests
        """
        if not self._retrain_callback:
            logger.warning("No retrain callback set, cannot process queue")
            return []

        processed = []

        with self._lock:
            # Sort by priority
            pending = sorted(
                [r for r in self._requests if r.status == "pending"],
                key=lambda r: r.priority
            )

        for request in pending:
            request.status = "in_progress"

            try:
                success = self._retrain_callback(request)

                if success:
                    request.status = "completed"
                    with self._lock:
                        self._last_retrain[request.model_id] = datetime.now()
                else:
                    request.status = "failed"

            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                request.status = "failed"

            processed.append(request)

        return processed

    def get_pending_requests(self) -> List[RetrainingRequest]:
        """Get all pending retraining requests."""
        with self._lock:
            return [r for r in self._requests if r.status == "pending"]

    def get_request_status(self, request_id: str) -> Optional[RetrainingRequest]:
        """Get status of a specific request."""
        with self._lock:
            for r in self._requests:
                if r.request_id == request_id:
                    return r
        return None


class ConceptDriftDetector:
    """
    Main concept drift detection system.

    Combines accuracy monitoring, statistical drift tests,
    alerting, and automatic retraining into a unified interface.
    """

    def __init__(
        self,
        model_id: str = "default",
        accuracy_threshold: float = 0.6,
        warning_threshold: float = 0.7,
        drift_sensitivity: float = 0.05,
        window_size: int = 1000,
        min_samples: int = 100,
        enable_auto_retrain: bool = True
    ):
        """
        Initialize concept drift detector.

        Args:
            model_id: Identifier for the model being monitored
            accuracy_threshold: Accuracy below this triggers drift alert
            warning_threshold: Accuracy below this triggers warning
            drift_sensitivity: Sensitivity for statistical tests
            window_size: Sliding window size for monitoring
            min_samples: Minimum samples before detection
            enable_auto_retrain: Whether to auto-trigger retraining
        """
        self.model_id = model_id
        self.accuracy_threshold = accuracy_threshold
        self.warning_threshold = warning_threshold
        self.drift_sensitivity = drift_sensitivity
        self.enable_auto_retrain = enable_auto_retrain

        # Components
        self.accuracy_monitor = AccuracyMonitor(
            window_size=window_size,
            min_samples=min_samples
        )
        self.alert_manager = AlertManager()
        self.retrain_manager = RetrainingManager()

        # Statistical drift detectors
        self._detectors: Dict[str, DriftDetector] = {
            'page_hinkley': PageHinkleyDetector(
                delta=drift_sensitivity,
                threshold=50.0
            ),
            'ddm': DDMDetector(
                warning_level=2.0,
                drift_level=3.0,
                min_samples=min_samples
            ),
            'eddm': EDDMDetector(
                warning_level=0.95,
                drift_level=0.90,
                min_errors=min_samples // 3
            )
        }

        self._drift_detected = False
        self._in_warning = False
        self._lock = threading.Lock()

    def record_prediction(
        self,
        predicted: Any,
        actual: Any,
        probability: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Record a prediction and check for drift.

        Args:
            predicted: Model's prediction
            actual: Actual outcome
            probability: Prediction probability
            features: Input features

        Returns:
            Dictionary with current status and any alerts
        """
        # Record prediction
        record = self.accuracy_monitor.record_prediction(
            predicted=predicted,
            actual=actual,
            probability=probability,
            features=features
        )

        # Update statistical detectors
        is_error = 0.0 if record.is_correct else 1.0
        detector_results = {}

        for name, detector in self._detectors.items():
            drift = detector.update(is_error)
            detector_results[name] = {
                'drift_detected': drift,
                'status': detector.get_status()
            }

        # Check accuracy threshold
        accuracy = self.accuracy_monitor.get_current_accuracy()
        alerts = []

        if accuracy is not None:
            # Check for accuracy drop
            if accuracy < self.accuracy_threshold:
                with self._lock:
                    if not self._drift_detected:
                        self._drift_detected = True

                        alert = self.alert_manager.raise_alert(
                            alert_type=AlertType.ACCURACY_DROP,
                            severity=DriftSeverity.SEVERE,
                            message=f"Model accuracy dropped to {accuracy:.2%}, "
                                    f"below threshold {self.accuracy_threshold:.2%}",
                            metrics=self.accuracy_monitor.get_metrics(),
                            model_id=self.model_id
                        )

                        if alert:
                            alerts.append(alert)

                        # Trigger retraining if enabled
                        if self.enable_auto_retrain:
                            self._trigger_retraining(
                                "Accuracy below threshold",
                                priority=1
                            )

            elif accuracy < self.warning_threshold:
                with self._lock:
                    if not self._in_warning:
                        self._in_warning = True

                        alert = self.alert_manager.raise_alert(
                            alert_type=AlertType.ACCURACY_DROP,
                            severity=DriftSeverity.WARNING,
                            message=f"Model accuracy at {accuracy:.2%}, "
                                    f"approaching threshold {self.accuracy_threshold:.2%}",
                            metrics=self.accuracy_monitor.get_metrics(),
                            model_id=self.model_id
                        )

                        if alert:
                            alerts.append(alert)
            else:
                with self._lock:
                    self._in_warning = False
                    self._drift_detected = False

        # Check statistical detectors
        for name, result in detector_results.items():
            if result['drift_detected']:
                alert = self.alert_manager.raise_alert(
                    alert_type=AlertType.DRIFT_DETECTED,
                    severity=DriftSeverity.MODERATE,
                    message=f"Statistical drift detected by {name} algorithm",
                    metrics={'detector': name, **result['status']},
                    model_id=self.model_id
                )

                if alert:
                    alerts.append(alert)

        return {
            'prediction_correct': record.is_correct,
            'current_accuracy': accuracy,
            'detector_results': detector_results,
            'alerts': alerts,
            'drift_detected': self._drift_detected,
            'in_warning': self._in_warning
        }

    def _trigger_retraining(self, reason: str, priority: int = 2) -> None:
        """Trigger model retraining."""
        request = self.retrain_manager.request_retraining(
            model_id=self.model_id,
            reason=reason,
            priority=priority,
            metrics_snapshot=self.accuracy_monitor.get_metrics()
        )

        if request:
            self.alert_manager.raise_alert(
                alert_type=AlertType.RETRAINING_TRIGGERED,
                severity=DriftSeverity.MODERATE,
                message=f"Automatic retraining triggered: {reason}",
                metrics={'request_id': request.request_id},
                model_id=self.model_id
            )

    def is_drift_detected(self) -> bool:
        """Check if concept drift has been detected."""
        return self._drift_detected

    def is_in_warning(self) -> bool:
        """Check if in warning state."""
        return self._in_warning

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            'model_id': self.model_id,
            'drift_detected': self._drift_detected,
            'in_warning': self._in_warning,
            'accuracy_metrics': self.accuracy_monitor.get_metrics(),
            'accuracy_trend': self.accuracy_monitor.get_accuracy_trend(),
            'detector_status': {
                name: detector.get_status()
                for name, detector in self._detectors.items()
            },
            'pending_retraining': len(self.retrain_manager.get_pending_requests()),
            'recent_alerts': len(self.alert_manager.get_alerts(
                since=datetime.now() - timedelta(hours=1)
            ))
        }

    def set_retrain_callback(
        self,
        callback: Callable[[RetrainingRequest], bool]
    ) -> None:
        """Set callback for executing retraining."""
        self.retrain_manager.set_retrain_callback(callback)

    def register_alert_callback(
        self,
        callback: Callable[[DriftAlert], None]
    ) -> None:
        """Register callback for alerts."""
        self.alert_manager.register_callback(callback)

    def run_statistical_tests(
        self,
        reference_errors: np.ndarray,
        current_errors: np.ndarray
    ) -> List[DriftTestResult]:
        """
        Run statistical drift tests comparing error distributions.

        Args:
            reference_errors: Error rates from reference period
            current_errors: Error rates from current period

        Returns:
            List of test results
        """
        results = []

        # KS Test
        if SCIPY_AVAILABLE:
            ks_stat, ks_pvalue = stats.ks_2samp(reference_errors, current_errors)

            severity = DriftSeverity.NONE
            if ks_pvalue < 0.01:
                severity = DriftSeverity.SEVERE
            elif ks_pvalue < 0.05:
                severity = DriftSeverity.MODERATE
            elif ks_pvalue < 0.1:
                severity = DriftSeverity.WARNING

            results.append(DriftTestResult(
                test_type=DriftTestType.KS_TEST,
                is_drift_detected=ks_pvalue < 0.05,
                statistic=ks_stat,
                p_value=ks_pvalue,
                threshold=0.05,
                severity=severity,
                details={'interpretation': 'Compares cumulative distributions'}
            ))

        # PSI (Population Stability Index)
        psi_value = self._calculate_psi(reference_errors, current_errors)

        severity = DriftSeverity.NONE
        if psi_value >= 0.25:
            severity = DriftSeverity.SEVERE
        elif psi_value >= 0.1:
            severity = DriftSeverity.MODERATE
        elif psi_value >= 0.05:
            severity = DriftSeverity.WARNING

        results.append(DriftTestResult(
            test_type=DriftTestType.PSI,
            is_drift_detected=psi_value >= 0.1,
            statistic=psi_value,
            p_value=None,
            threshold=0.1,
            severity=severity,
            details={
                'interpretation': 'PSI < 0.1: No drift, 0.1-0.25: Moderate, > 0.25: Severe'
            }
        ))

        return results

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return psi

    def reset(self) -> None:
        """Reset all monitoring state."""
        self.accuracy_monitor.reset()
        self.alert_manager.clear_alerts()

        for detector in self._detectors.values():
            detector.reset()

        with self._lock:
            self._drift_detected = False
            self._in_warning = False


# Convenience functions and singleton pattern
_default_detector: Optional[ConceptDriftDetector] = None


def get_drift_detector(model_id: str = "default") -> ConceptDriftDetector:
    """Get or create the default drift detector."""
    global _default_detector

    if _default_detector is None or _default_detector.model_id != model_id:
        _default_detector = ConceptDriftDetector(model_id=model_id)

    return _default_detector


def set_drift_detector(detector: ConceptDriftDetector) -> None:
    """Set the default drift detector."""
    global _default_detector
    _default_detector = detector


def record_prediction(
    predicted: Any,
    actual: Any,
    model_id: str = "default"
) -> Dict[str, Any]:
    """Record a prediction using the default detector."""
    detector = get_drift_detector(model_id)
    return detector.record_prediction(predicted, actual)


def check_drift(model_id: str = "default") -> bool:
    """Check if drift has been detected."""
    detector = get_drift_detector(model_id)
    return detector.is_drift_detected()


def get_model_status(model_id: str = "default") -> Dict[str, Any]:
    """Get model monitoring status."""
    detector = get_drift_detector(model_id)
    return detector.get_status()
