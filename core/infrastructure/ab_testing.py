# -*- coding: utf-8 -*-
"""
A/B Testing Framework - Strategy Comparison & Deployment
=========================================================
Production-grade A/B testing for trading strategies with
statistical significance testing, gradual traffic shifting,
and automatic rollback.

Features:
- Statistical significance testing (t-test, Mann-Whitney, bootstrap)
- Gradual traffic shifting (10% → 50% → 100%)
- Automatic rollback on performance degradation
- Multi-armed bandit for adaptive allocation
- Real-time monitoring and alerts

Example:
    >>> from core.infrastructure.ab_testing import ABTestFramework
    >>>
    >>> # Create A/B test
    >>> framework = ABTestFramework()
    >>> test = framework.create_test(
    ...     name="momentum_v2_test",
    ...     control="momentum_v1",
    ...     treatment="momentum_v2",
    ...     traffic_schedule=[0.1, 0.25, 0.5, 1.0]
    ... )
    >>>
    >>> # Get strategy for next trade
    >>> strategy = test.get_variant()
    >>>
    >>> # Record results
    >>> test.record_outcome(strategy, pnl=150.0)
    >>>
    >>> # Check if we can conclude
    >>> if test.is_significant():
    ...     winner = test.get_winner()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Deque
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging
import random
import math
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"
    ROLLED_BACK = "rolled_back"


class SignificanceMethod(Enum):
    """Statistical significance testing methods."""
    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY = "mann_whitney"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"


class TrafficAllocation(Enum):
    """Traffic allocation strategies."""
    FIXED = "fixed"
    SCHEDULED = "scheduled"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"  # Upper Confidence Bound


class RollbackReason(Enum):
    """Reasons for automatic rollback."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    TIMEOUT = "timeout"


@dataclass
class TrafficSchedule:
    """Traffic shifting schedule."""
    stages: List[float]  # Traffic percentages [0.1, 0.25, 0.5, 1.0]
    stage_duration: timedelta  # Duration per stage
    current_stage: int = 0
    stage_start_time: Optional[datetime] = None
    min_samples_per_stage: int = 100

    def get_current_traffic(self) -> float:
        """Get current traffic percentage for treatment."""
        if self.current_stage >= len(self.stages):
            return self.stages[-1]
        return self.stages[self.current_stage]

    def should_advance(self, sample_count: int) -> bool:
        """Check if should advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False

        if sample_count < self.min_samples_per_stage:
            return False

        if self.stage_start_time:
            elapsed = datetime.now() - self.stage_start_time
            return elapsed >= self.stage_duration

        return False

    def advance(self) -> float:
        """Advance to next stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.stage_start_time = datetime.now()
            logger.info(
                f"Traffic advanced to stage {self.current_stage}: "
                f"{self.get_current_traffic():.0%}"
            )
        return self.get_current_traffic()


@dataclass
class VariantMetrics:
    """Metrics for a test variant."""
    variant_id: str
    sample_count: int = 0
    total_pnl: float = 0.0
    pnl_values: List[float] = field(default_factory=list)
    win_count: int = 0
    loss_count: int = 0
    error_count: int = 0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    last_updated: Optional[datetime] = None

    @property
    def mean_pnl(self) -> float:
        """Calculate mean P&L."""
        if not self.pnl_values:
            return 0.0
        return sum(self.pnl_values) / len(self.pnl_values)

    @property
    def std_pnl(self) -> float:
        """Calculate standard deviation of P&L."""
        if len(self.pnl_values) < 2:
            return 0.0
        mean = self.mean_pnl
        variance = sum((x - mean) ** 2 for x in self.pnl_values) / (len(self.pnl_values) - 1)
        return math.sqrt(variance)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.0
        return self.win_count / total

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if self.std_pnl == 0:
            return 0.0
        # Assuming daily observations
        return (self.mean_pnl / self.std_pnl) * math.sqrt(252)

    def record(self, pnl: float, is_error: bool = False) -> None:
        """Record an outcome."""
        self.sample_count += 1
        self.total_pnl += pnl
        self.pnl_values.append(pnl)
        self.last_updated = datetime.now()

        if is_error:
            self.error_count += 1
        elif pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        # Track drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        else:
            drawdown = (self.peak_pnl - self.total_pnl) / max(self.peak_pnl, 1)
            self.max_drawdown = max(self.max_drawdown, drawdown)


@dataclass
class SignificanceResult:
    """Result of statistical significance test."""
    method: SignificanceMethod
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    effect_size: float
    power: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'is_significant': self.is_significant,
            'effect_size': self.effect_size,
            'power': self.power,
            'recommendation': self.recommendation
        }


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    event_id: str
    test_id: str
    timestamp: datetime
    reason: RollbackReason
    metrics_at_rollback: Dict[str, Any]
    treatment_traffic: float
    message: str


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    name: str
    control_id: str
    treatment_id: str
    significance_level: float = 0.05
    min_samples: int = 100
    max_duration: Optional[timedelta] = None
    traffic_allocation: TrafficAllocation = TrafficAllocation.SCHEDULED
    traffic_schedule: Optional[List[float]] = None
    stage_duration: timedelta = timedelta(hours=24)
    rollback_threshold: float = -0.1  # Rollback if treatment 10% worse
    max_drawdown_threshold: float = 0.15
    error_rate_threshold: float = 0.05


class SignificanceTester:
    """
    Statistical significance testing for A/B tests.

    Implements multiple testing methods with proper corrections.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 30
    ):
        """
        Initialize significance tester.

        Args:
            significance_level: Alpha level for significance
            min_samples: Minimum samples per variant
        """
        self.significance_level = significance_level
        self.min_samples = min_samples

    def test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        method: SignificanceMethod = SignificanceMethod.WELCH_T_TEST
    ) -> SignificanceResult:
        """
        Test for statistical significance.

        Args:
            control: Control variant metrics
            treatment: Treatment variant metrics
            method: Statistical test method

        Returns:
            Significance test result
        """
        if method == SignificanceMethod.T_TEST:
            return self._t_test(control, treatment, equal_var=True)
        elif method == SignificanceMethod.WELCH_T_TEST:
            return self._t_test(control, treatment, equal_var=False)
        elif method == SignificanceMethod.MANN_WHITNEY:
            return self._mann_whitney(control, treatment)
        elif method == SignificanceMethod.BOOTSTRAP:
            return self._bootstrap_test(control, treatment)
        elif method == SignificanceMethod.BAYESIAN:
            return self._bayesian_test(control, treatment)
        else:
            return self._t_test(control, treatment, equal_var=False)

    def _t_test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        equal_var: bool = False
    ) -> SignificanceResult:
        """Perform t-test."""
        if not SCIPY_AVAILABLE:
            return self._fallback_test(control, treatment)

        control_data = control.pnl_values
        treatment_data = treatment.pnl_values

        if len(control_data) < self.min_samples or len(treatment_data) < self.min_samples:
            return SignificanceResult(
                method=SignificanceMethod.WELCH_T_TEST if not equal_var else SignificanceMethod.T_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                recommendation="Insufficient samples for reliable test"
            )

        # Perform t-test
        statistic, p_value = stats.ttest_ind(
            treatment_data, control_data, equal_var=equal_var
        )

        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((len(control_data) - 1) * control.std_pnl ** 2 +
             (len(treatment_data) - 1) * treatment.std_pnl ** 2) /
            (len(control_data) + len(treatment_data) - 2)
        )
        effect_size = (treatment.mean_pnl - control.mean_pnl) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for difference
        diff_mean = treatment.mean_pnl - control.mean_pnl
        se = math.sqrt(
            control.std_pnl ** 2 / len(control_data) +
            treatment.std_pnl ** 2 / len(treatment_data)
        )
        t_crit = stats.t.ppf(1 - self.significance_level / 2,
                            len(control_data) + len(treatment_data) - 2)
        ci = (diff_mean - t_crit * se, diff_mean + t_crit * se)

        is_significant = p_value < self.significance_level

        # Generate recommendation
        if not is_significant:
            recommendation = "No significant difference detected. Continue testing."
        elif treatment.mean_pnl > control.mean_pnl:
            recommendation = f"Treatment significantly better (effect size: {effect_size:.2f}). Consider promoting."
        else:
            recommendation = f"Control significantly better. Consider rolling back treatment."

        return SignificanceResult(
            method=SignificanceMethod.WELCH_T_TEST if not equal_var else SignificanceMethod.T_TEST,
            statistic=statistic,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=is_significant,
            effect_size=effect_size,
            recommendation=recommendation
        )

    def _mann_whitney(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics
    ) -> SignificanceResult:
        """Perform Mann-Whitney U test (non-parametric)."""
        if not SCIPY_AVAILABLE:
            return self._fallback_test(control, treatment)

        control_data = control.pnl_values
        treatment_data = treatment.pnl_values

        if len(control_data) < self.min_samples or len(treatment_data) < self.min_samples:
            return SignificanceResult(
                method=SignificanceMethod.MANN_WHITNEY,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                recommendation="Insufficient samples"
            )

        statistic, p_value = stats.mannwhitneyu(
            treatment_data, control_data, alternative='two-sided'
        )

        # Effect size (rank-biserial correlation)
        n1, n2 = len(control_data), len(treatment_data)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        is_significant = p_value < self.significance_level

        return SignificanceResult(
            method=SignificanceMethod.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            confidence_interval=(0.0, 0.0),  # Not applicable for Mann-Whitney
            is_significant=is_significant,
            effect_size=effect_size,
            recommendation="Significant difference" if is_significant else "No significant difference"
        )

    def _bootstrap_test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics,
        n_bootstrap: int = 10000
    ) -> SignificanceResult:
        """Perform bootstrap significance test."""
        if not NUMPY_AVAILABLE:
            return self._fallback_test(control, treatment)

        control_data = np.array(control.pnl_values)
        treatment_data = np.array(treatment.pnl_values)

        if len(control_data) < self.min_samples or len(treatment_data) < self.min_samples:
            return SignificanceResult(
                method=SignificanceMethod.BOOTSTRAP,
                statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                recommendation="Insufficient samples"
            )

        # Observed difference
        observed_diff = np.mean(treatment_data) - np.mean(control_data)

        # Bootstrap under null hypothesis
        combined = np.concatenate([control_data, treatment_data])
        n_control = len(control_data)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            np.random.shuffle(combined)
            boot_control = combined[:n_control]
            boot_treatment = combined[n_control:]
            bootstrap_diffs.append(np.mean(boot_treatment) - np.mean(boot_control))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value (two-sided)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Confidence interval via percentile method
        ci_low = np.percentile(bootstrap_diffs, 2.5)
        ci_high = np.percentile(bootstrap_diffs, 97.5)

        # Effect size
        pooled_std = np.std(combined)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        is_significant = p_value < self.significance_level

        return SignificanceResult(
            method=SignificanceMethod.BOOTSTRAP,
            statistic=observed_diff,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_significant,
            effect_size=effect_size,
            recommendation="Significant" if is_significant else "Not significant"
        )

    def _bayesian_test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics
    ) -> SignificanceResult:
        """Perform Bayesian A/B test."""
        if not NUMPY_AVAILABLE:
            return self._fallback_test(control, treatment)

        # Simple Bayesian comparison using normal approximation
        control_mean = control.mean_pnl
        control_std = control.std_pnl / math.sqrt(max(1, control.sample_count))
        treatment_mean = treatment.mean_pnl
        treatment_std = treatment.std_pnl / math.sqrt(max(1, treatment.sample_count))

        # Probability that treatment > control
        if control_std > 0 or treatment_std > 0:
            diff_mean = treatment_mean - control_mean
            diff_std = math.sqrt(control_std ** 2 + treatment_std ** 2)
            if diff_std > 0:
                z_score = diff_mean / diff_std
                prob_treatment_better = 1 - stats.norm.cdf(0, diff_mean, diff_std) if SCIPY_AVAILABLE else 0.5
            else:
                z_score = 0
                prob_treatment_better = 0.5
        else:
            z_score = 0
            prob_treatment_better = 0.5

        # Credible interval
        ci = (diff_mean - 1.96 * diff_std, diff_mean + 1.96 * diff_std) if diff_std > 0 else (0, 0)

        is_significant = prob_treatment_better > 0.95 or prob_treatment_better < 0.05

        return SignificanceResult(
            method=SignificanceMethod.BAYESIAN,
            statistic=prob_treatment_better,
            p_value=1 - prob_treatment_better if prob_treatment_better > 0.5 else prob_treatment_better,
            confidence_interval=ci,
            is_significant=is_significant,
            effect_size=z_score,
            recommendation=f"P(treatment > control) = {prob_treatment_better:.1%}"
        )

    def _fallback_test(
        self,
        control: VariantMetrics,
        treatment: VariantMetrics
    ) -> SignificanceResult:
        """Fallback when scipy not available."""
        diff = treatment.mean_pnl - control.mean_pnl
        return SignificanceResult(
            method=SignificanceMethod.T_TEST,
            statistic=diff,
            p_value=0.5,
            confidence_interval=(0, 0),
            is_significant=False,
            effect_size=0,
            recommendation="Statistical libraries not available"
        )


class TrafficAllocator:
    """
    Allocates traffic between variants.

    Supports fixed, scheduled, and adaptive allocation strategies.
    """

    def __init__(
        self,
        allocation: TrafficAllocation = TrafficAllocation.FIXED,
        epsilon: float = 0.1
    ):
        """
        Initialize allocator.

        Args:
            allocation: Allocation strategy
            epsilon: Exploration rate for epsilon-greedy
        """
        self.allocation = allocation
        self.epsilon = epsilon

        # For Thompson Sampling
        self._successes: Dict[str, int] = defaultdict(int)
        self._failures: Dict[str, int] = defaultdict(int)

        # For UCB
        self._pulls: Dict[str, int] = defaultdict(int)
        self._total_pulls = 0

    def get_variant(
        self,
        control_id: str,
        treatment_id: str,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics,
        traffic_pct: float = 0.5
    ) -> str:
        """
        Get which variant to use.

        Args:
            control_id: Control variant ID
            treatment_id: Treatment variant ID
            control_metrics: Control metrics
            treatment_metrics: Treatment metrics
            traffic_pct: Treatment traffic percentage (for fixed/scheduled)

        Returns:
            Selected variant ID
        """
        if self.allocation == TrafficAllocation.FIXED:
            return treatment_id if random.random() < traffic_pct else control_id

        elif self.allocation == TrafficAllocation.SCHEDULED:
            return treatment_id if random.random() < traffic_pct else control_id

        elif self.allocation == TrafficAllocation.EPSILON_GREEDY:
            return self._epsilon_greedy(
                control_id, treatment_id,
                control_metrics, treatment_metrics
            )

        elif self.allocation == TrafficAllocation.THOMPSON_SAMPLING:
            return self._thompson_sampling(
                control_id, treatment_id,
                control_metrics, treatment_metrics
            )

        elif self.allocation == TrafficAllocation.UCB:
            return self._ucb(
                control_id, treatment_id,
                control_metrics, treatment_metrics
            )

        return control_id

    def _epsilon_greedy(
        self,
        control_id: str,
        treatment_id: str,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics
    ) -> str:
        """Epsilon-greedy selection."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice([control_id, treatment_id])
        else:
            # Exploit
            if treatment_metrics.mean_pnl > control_metrics.mean_pnl:
                return treatment_id
            return control_id

    def _thompson_sampling(
        self,
        control_id: str,
        treatment_id: str,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics
    ) -> str:
        """Thompson Sampling selection."""
        if not NUMPY_AVAILABLE:
            return self._epsilon_greedy(
                control_id, treatment_id,
                control_metrics, treatment_metrics
            )

        # Use beta distribution for win rates
        control_sample = np.random.beta(
            max(1, control_metrics.win_count),
            max(1, control_metrics.loss_count)
        )
        treatment_sample = np.random.beta(
            max(1, treatment_metrics.win_count),
            max(1, treatment_metrics.loss_count)
        )

        return treatment_id if treatment_sample > control_sample else control_id

    def _ucb(
        self,
        control_id: str,
        treatment_id: str,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics
    ) -> str:
        """Upper Confidence Bound selection."""
        self._total_pulls += 1

        # Ensure each variant is tried at least once
        if self._pulls[control_id] == 0:
            self._pulls[control_id] += 1
            return control_id
        if self._pulls[treatment_id] == 0:
            self._pulls[treatment_id] += 1
            return treatment_id

        # Calculate UCB scores
        def ucb_score(variant_id: str, metrics: VariantMetrics) -> float:
            exploitation = metrics.mean_pnl
            exploration = math.sqrt(
                2 * math.log(self._total_pulls) / self._pulls[variant_id]
            )
            return exploitation + exploration

        control_score = ucb_score(control_id, control_metrics)
        treatment_score = ucb_score(treatment_id, treatment_metrics)

        selected = treatment_id if treatment_score > control_score else control_id
        self._pulls[selected] += 1

        return selected

    def record_outcome(self, variant_id: str, is_success: bool) -> None:
        """Record outcome for adaptive methods."""
        if is_success:
            self._successes[variant_id] += 1
        else:
            self._failures[variant_id] += 1


class RollbackMonitor:
    """
    Monitors test health and triggers automatic rollback.

    Watches for performance degradation and other issues.
    """

    def __init__(
        self,
        performance_threshold: float = -0.1,
        drawdown_threshold: float = 0.15,
        error_rate_threshold: float = 0.05,
        check_interval: int = 50  # Check every N samples
    ):
        """
        Initialize rollback monitor.

        Args:
            performance_threshold: Relative performance threshold
            drawdown_threshold: Maximum drawdown threshold
            error_rate_threshold: Maximum error rate
            check_interval: Samples between checks
        """
        self.performance_threshold = performance_threshold
        self.drawdown_threshold = drawdown_threshold
        self.error_rate_threshold = error_rate_threshold
        self.check_interval = check_interval

        self._sample_count = 0
        self._rollback_triggered = False
        self._rollback_reason: Optional[RollbackReason] = None
        self._callbacks: List[Callable[[RollbackReason, Dict], None]] = []

    def check(
        self,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics
    ) -> Tuple[bool, Optional[RollbackReason], str]:
        """
        Check if rollback should be triggered.

        Returns:
            Tuple of (should_rollback, reason, message)
        """
        self._sample_count += 1

        if self._sample_count % self.check_interval != 0:
            return False, None, ""

        # Check performance degradation
        if control_metrics.sample_count > 0 and treatment_metrics.sample_count > 0:
            if control_metrics.mean_pnl != 0:
                relative_perf = (
                    (treatment_metrics.mean_pnl - control_metrics.mean_pnl) /
                    abs(control_metrics.mean_pnl)
                )
            else:
                relative_perf = treatment_metrics.mean_pnl

            if relative_perf < self.performance_threshold:
                return (
                    True,
                    RollbackReason.PERFORMANCE_DEGRADATION,
                    f"Treatment underperforming by {abs(relative_perf):.1%}"
                )

        # Check drawdown
        if treatment_metrics.max_drawdown > self.drawdown_threshold:
            return (
                True,
                RollbackReason.EXCESSIVE_DRAWDOWN,
                f"Treatment drawdown {treatment_metrics.max_drawdown:.1%} exceeds threshold"
            )

        # Check error rate
        if treatment_metrics.sample_count > 0:
            error_rate = treatment_metrics.error_count / treatment_metrics.sample_count
            if error_rate > self.error_rate_threshold:
                return (
                    True,
                    RollbackReason.ERROR_RATE,
                    f"Treatment error rate {error_rate:.1%} exceeds threshold"
                )

        return False, None, ""

    def register_callback(
        self,
        callback: Callable[[RollbackReason, Dict], None]
    ) -> None:
        """Register rollback callback."""
        self._callbacks.append(callback)

    def trigger_rollback(
        self,
        reason: RollbackReason,
        metrics: Dict[str, Any]
    ) -> None:
        """Trigger rollback and notify callbacks."""
        self._rollback_triggered = True
        self._rollback_reason = reason

        for callback in self._callbacks:
            try:
                callback(reason, metrics)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")


class ABTest:
    """
    Individual A/B test instance.

    Manages a single comparison between control and treatment.
    """

    def __init__(self, config: ABTestConfig):
        """
        Initialize A/B test.

        Args:
            config: Test configuration
        """
        self.config = config
        self.test_id = f"test_{config.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.control = VariantMetrics(variant_id=config.control_id)
        self.treatment = VariantMetrics(variant_id=config.treatment_id)

        self.status = TestStatus.DRAFT
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.winner: Optional[str] = None

        # Traffic schedule
        self.traffic_schedule = TrafficSchedule(
            stages=config.traffic_schedule or [0.1, 0.25, 0.5, 1.0],
            stage_duration=config.stage_duration,
            min_samples_per_stage=config.min_samples // 4
        )

        # Components
        self.significance_tester = SignificanceTester(
            significance_level=config.significance_level,
            min_samples=config.min_samples // 2
        )
        self.allocator = TrafficAllocator(allocation=config.traffic_allocation)
        self.rollback_monitor = RollbackMonitor(
            performance_threshold=config.rollback_threshold,
            drawdown_threshold=config.max_drawdown_threshold,
            error_rate_threshold=config.error_rate_threshold
        )

        self._lock = threading.Lock()
        self._rollback_events: List[RollbackEvent] = []

    def start(self) -> None:
        """Start the test."""
        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()
        self.traffic_schedule.stage_start_time = datetime.now()
        logger.info(f"A/B test started: {self.test_id}")

    def pause(self) -> None:
        """Pause the test."""
        self.status = TestStatus.PAUSED
        logger.info(f"A/B test paused: {self.test_id}")

    def resume(self) -> None:
        """Resume the test."""
        self.status = TestStatus.RUNNING
        logger.info(f"A/B test resumed: {self.test_id}")

    def get_variant(self) -> str:
        """
        Get which variant to use for next observation.

        Returns:
            Variant ID to use
        """
        if self.status != TestStatus.RUNNING:
            return self.config.control_id

        traffic_pct = self.traffic_schedule.get_current_traffic()

        return self.allocator.get_variant(
            self.config.control_id,
            self.config.treatment_id,
            self.control,
            self.treatment,
            traffic_pct
        )

    def record_outcome(
        self,
        variant_id: str,
        pnl: float,
        is_error: bool = False
    ) -> Optional[RollbackEvent]:
        """
        Record outcome for a variant.

        Args:
            variant_id: Variant that produced outcome
            pnl: P&L result
            is_error: Whether an error occurred

        Returns:
            RollbackEvent if rollback triggered
        """
        if self.status != TestStatus.RUNNING:
            return None

        with self._lock:
            # Record to appropriate variant
            if variant_id == self.config.control_id:
                self.control.record(pnl, is_error)
            else:
                self.treatment.record(pnl, is_error)

            # Update allocator
            self.allocator.record_outcome(variant_id, pnl > 0)

            # Check for traffic advancement
            total_samples = self.control.sample_count + self.treatment.sample_count
            if self.traffic_schedule.should_advance(total_samples):
                self.traffic_schedule.advance()

            # Check for rollback
            should_rollback, reason, message = self.rollback_monitor.check(
                self.control, self.treatment
            )

            if should_rollback:
                return self._trigger_rollback(reason, message)

            # Check test completion
            self._check_completion()

        return None

    def _trigger_rollback(
        self,
        reason: RollbackReason,
        message: str
    ) -> RollbackEvent:
        """Trigger test rollback."""
        self.status = TestStatus.ROLLED_BACK
        self.end_time = datetime.now()
        self.winner = self.config.control_id

        event = RollbackEvent(
            event_id=f"rollback_{len(self._rollback_events)}",
            test_id=self.test_id,
            timestamp=datetime.now(),
            reason=reason,
            metrics_at_rollback={
                'control': {
                    'mean_pnl': self.control.mean_pnl,
                    'samples': self.control.sample_count
                },
                'treatment': {
                    'mean_pnl': self.treatment.mean_pnl,
                    'samples': self.treatment.sample_count,
                    'max_drawdown': self.treatment.max_drawdown
                }
            },
            treatment_traffic=self.traffic_schedule.get_current_traffic(),
            message=message
        )

        self._rollback_events.append(event)

        logger.warning(f"A/B test rolled back: {self.test_id} - {message}")

        return event

    def _check_completion(self) -> None:
        """Check if test should be concluded."""
        # Check max duration
        if self.config.max_duration and self.start_time:
            if datetime.now() - self.start_time > self.config.max_duration:
                self.conclude()
                return

        # Check if reached full traffic with enough samples
        if (self.traffic_schedule.current_stage >= len(self.traffic_schedule.stages) - 1 and
            self.control.sample_count >= self.config.min_samples and
            self.treatment.sample_count >= self.config.min_samples):

            result = self.significance_tester.test(self.control, self.treatment)
            if result.is_significant:
                self.conclude()

    def conclude(self, winner: Optional[str] = None) -> None:
        """Conclude the test."""
        self.status = TestStatus.CONCLUDED
        self.end_time = datetime.now()

        if winner:
            self.winner = winner
        else:
            # Determine winner based on performance
            if self.treatment.mean_pnl > self.control.mean_pnl:
                self.winner = self.config.treatment_id
            else:
                self.winner = self.config.control_id

        logger.info(f"A/B test concluded: {self.test_id}, winner: {self.winner}")

    def get_significance(
        self,
        method: SignificanceMethod = SignificanceMethod.WELCH_T_TEST
    ) -> SignificanceResult:
        """Get statistical significance result."""
        return self.significance_tester.test(
            self.control, self.treatment, method
        )

    def is_significant(self) -> bool:
        """Check if results are statistically significant."""
        result = self.get_significance()
        return result.is_significant

    def get_winner(self) -> Optional[str]:
        """Get the winning variant."""
        if self.winner:
            return self.winner

        if not self.is_significant():
            return None

        return (
            self.config.treatment_id
            if self.treatment.mean_pnl > self.control.mean_pnl
            else self.config.control_id
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive test status."""
        significance = self.get_significance()

        return {
            'test_id': self.test_id,
            'name': self.config.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'winner': self.winner,
            'current_traffic': self.traffic_schedule.get_current_traffic(),
            'traffic_stage': self.traffic_schedule.current_stage,
            'control': {
                'id': self.config.control_id,
                'samples': self.control.sample_count,
                'mean_pnl': self.control.mean_pnl,
                'total_pnl': self.control.total_pnl,
                'win_rate': self.control.win_rate,
                'sharpe': self.control.sharpe_ratio
            },
            'treatment': {
                'id': self.config.treatment_id,
                'samples': self.treatment.sample_count,
                'mean_pnl': self.treatment.mean_pnl,
                'total_pnl': self.treatment.total_pnl,
                'win_rate': self.treatment.win_rate,
                'sharpe': self.treatment.sharpe_ratio,
                'max_drawdown': self.treatment.max_drawdown
            },
            'significance': significance.to_dict(),
            'rollback_events': len(self._rollback_events)
        }


class ABTestFramework:
    """
    Main A/B testing framework.

    Manages multiple A/B tests and provides unified interface.
    """

    def __init__(self):
        """Initialize A/B testing framework."""
        self._tests: Dict[str, ABTest] = {}
        self._active_tests: Dict[str, str] = {}  # strategy_pair -> test_id
        self._lock = threading.Lock()
        self._rollback_callbacks: List[Callable[[ABTest, RollbackEvent], None]] = []

    def create_test(
        self,
        name: str,
        control: str,
        treatment: str,
        traffic_schedule: Optional[List[float]] = None,
        significance_level: float = 0.05,
        min_samples: int = 100,
        max_duration: Optional[timedelta] = None,
        allocation: TrafficAllocation = TrafficAllocation.SCHEDULED,
        **kwargs
    ) -> ABTest:
        """
        Create a new A/B test.

        Args:
            name: Test name
            control: Control strategy ID
            treatment: Treatment strategy ID
            traffic_schedule: List of traffic percentages
            significance_level: Alpha for significance
            min_samples: Minimum samples required
            max_duration: Maximum test duration
            allocation: Traffic allocation strategy

        Returns:
            Created A/B test
        """
        config = ABTestConfig(
            name=name,
            control_id=control,
            treatment_id=treatment,
            significance_level=significance_level,
            min_samples=min_samples,
            max_duration=max_duration,
            traffic_allocation=allocation,
            traffic_schedule=traffic_schedule or [0.1, 0.25, 0.5, 1.0],
            **kwargs
        )

        test = ABTest(config)

        with self._lock:
            self._tests[test.test_id] = test
            pair_key = f"{control}:{treatment}"
            self._active_tests[pair_key] = test.test_id

        logger.info(f"Created A/B test: {test.test_id}")

        return test

    def start_test(self, test_id: str) -> bool:
        """Start a test."""
        test = self._tests.get(test_id)
        if not test:
            return False

        test.start()
        return True

    def get_variant(
        self,
        control: str,
        treatment: str
    ) -> str:
        """
        Get which variant to use for a strategy pair.

        Args:
            control: Control strategy ID
            treatment: Treatment strategy ID

        Returns:
            Strategy ID to use
        """
        pair_key = f"{control}:{treatment}"

        with self._lock:
            test_id = self._active_tests.get(pair_key)
            if not test_id:
                return control

            test = self._tests.get(test_id)
            if not test or test.status != TestStatus.RUNNING:
                return control

        return test.get_variant()

    def record_outcome(
        self,
        variant_id: str,
        pnl: float,
        is_error: bool = False
    ) -> None:
        """
        Record outcome for any active test involving this variant.

        Args:
            variant_id: Variant that produced outcome
            pnl: P&L result
            is_error: Whether error occurred
        """
        with self._lock:
            for test in self._tests.values():
                if test.status != TestStatus.RUNNING:
                    continue

                if (variant_id == test.config.control_id or
                    variant_id == test.config.treatment_id):

                    rollback = test.record_outcome(variant_id, pnl, is_error)

                    if rollback:
                        self._notify_rollback(test, rollback)

    def _notify_rollback(self, test: ABTest, event: RollbackEvent) -> None:
        """Notify rollback callbacks."""
        for callback in self._rollback_callbacks:
            try:
                callback(test, event)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")

    def on_rollback(
        self,
        callback: Callable[[ABTest, RollbackEvent], None]
    ) -> None:
        """Register rollback callback."""
        self._rollback_callbacks.append(callback)

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test by ID."""
        return self._tests.get(test_id)

    def list_tests(
        self,
        status: Optional[TestStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all tests with optional status filter."""
        tests = list(self._tests.values())

        if status:
            tests = [t for t in tests if t.status == status]

        return [t.get_status() for t in tests]

    def conclude_test(
        self,
        test_id: str,
        winner: Optional[str] = None
    ) -> bool:
        """Manually conclude a test."""
        test = self._tests.get(test_id)
        if not test:
            return False

        test.conclude(winner)

        # Remove from active
        with self._lock:
            pair_key = f"{test.config.control_id}:{test.config.treatment_id}"
            if pair_key in self._active_tests:
                del self._active_tests[pair_key]

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get framework summary."""
        tests = list(self._tests.values())

        return {
            'total_tests': len(tests),
            'running': sum(1 for t in tests if t.status == TestStatus.RUNNING),
            'concluded': sum(1 for t in tests if t.status == TestStatus.CONCLUDED),
            'rolled_back': sum(1 for t in tests if t.status == TestStatus.ROLLED_BACK),
            'active_pairs': list(self._active_tests.keys())
        }


# Convenience functions
_default_framework: Optional[ABTestFramework] = None


def get_ab_framework() -> ABTestFramework:
    """Get default A/B testing framework."""
    global _default_framework
    if _default_framework is None:
        _default_framework = ABTestFramework()
    return _default_framework


def set_ab_framework(framework: ABTestFramework) -> None:
    """Set default A/B testing framework."""
    global _default_framework
    _default_framework = framework


def create_ab_test(
    name: str,
    control: str,
    treatment: str,
    **kwargs
) -> ABTest:
    """Create A/B test using default framework."""
    return get_ab_framework().create_test(name, control, treatment, **kwargs)


def get_variant(control: str, treatment: str) -> str:
    """Get variant using default framework."""
    return get_ab_framework().get_variant(control, treatment)


def record_ab_outcome(variant_id: str, pnl: float, is_error: bool = False) -> None:
    """Record outcome using default framework."""
    get_ab_framework().record_outcome(variant_id, pnl, is_error)
