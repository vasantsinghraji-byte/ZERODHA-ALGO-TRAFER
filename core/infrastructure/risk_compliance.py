# -*- coding: utf-8 -*-
"""
Risk Compliance - Regulatory Compliance Engine
===============================================
Production-grade compliance engine for Indian markets
enforcing SEBI regulations, position limits, and circuit breakers.

Features:
- Position limit enforcement (exchange and client level)
- Circuit breaker compliance (price bands, trading halts)
- SEBI regulatory requirements
- Real-time compliance monitoring
- Violation alerts and blocking

Example:
    >>> from core.infrastructure.risk_compliance import ComplianceEngine
    >>>
    >>> # Create compliance engine
    >>> engine = ComplianceEngine()
    >>>
    >>> # Check order compliance
    >>> result = engine.check_order_compliance(
    ...     symbol="RELIANCE",
    ...     side="buy",
    ...     quantity=1000,
    ...     price=2500.0
    ... )
    >>>
    >>> if not result.is_compliant:
    ...     print(f"Blocked: {result.violation_reason}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Set, Tuple
from datetime import datetime, date, time, timedelta
from collections import defaultdict
import threading
import logging
import json

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of compliance violations."""
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    TURNOVER_LIMIT_EXCEEDED = "turnover_limit_exceeded"
    ORDER_VALUE_LIMIT = "order_value_limit"
    CIRCUIT_BREAKER_HIT = "circuit_breaker_hit"
    TRADING_HALTED = "trading_halted"
    PRICE_BAND_VIOLATION = "price_band_violation"
    MARKET_CLOSED = "market_closed"
    SYMBOL_SUSPENDED = "symbol_suspended"
    MARGIN_INSUFFICIENT = "margin_insufficient"
    ORDER_FREQUENCY_LIMIT = "order_frequency_limit"
    SEBI_RULE_VIOLATION = "sebi_rule_violation"


class ComplianceAction(Enum):
    """Actions to take on violation."""
    BLOCK = "block"
    WARN = "warn"
    LOG_ONLY = "log_only"
    REDUCE_SIZE = "reduce_size"


class CircuitBreakerLevel(Enum):
    """Circuit breaker levels as per SEBI."""
    NONE = "none"
    LEVEL_1 = "level_1"  # 10% movement
    LEVEL_2 = "level_2"  # 15% movement
    LEVEL_3 = "level_3"  # 20% movement


class MarketStatus(Enum):
    """Market trading status."""
    PRE_OPEN = "pre_open"
    OPEN = "open"
    POST_CLOSE = "post_close"
    CLOSED = "closed"
    HALTED = "halted"


class Exchange(Enum):
    """Indian stock exchanges."""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NCDEX = "NCDEX"


@dataclass
class PositionLimit:
    """Position limit configuration."""
    symbol: str
    max_quantity: int
    max_value: float
    max_pct_of_oi: float  # Max % of open interest (for F&O)
    exchange: Exchange
    segment: str = "EQ"  # EQ, FO, CD
    client_level: bool = True
    market_wide: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'max_quantity': self.max_quantity,
            'max_value': self.max_value,
            'max_pct_of_oi': self.max_pct_of_oi,
            'exchange': self.exchange.value,
            'segment': self.segment
        }


@dataclass
class PriceBand:
    """Price band (circuit filter) for a symbol."""
    symbol: str
    exchange: Exchange
    reference_price: float
    lower_limit: float
    upper_limit: float
    lower_pct: float
    upper_pct: float
    last_updated: datetime

    def is_within_band(self, price: float) -> bool:
        """Check if price is within band."""
        return self.lower_limit <= price <= self.upper_limit

    def get_violation_pct(self, price: float) -> float:
        """Get how much price violates band (0 if within)."""
        if price < self.lower_limit:
            return (self.lower_limit - price) / self.reference_price * 100
        elif price > self.upper_limit:
            return (price - self.upper_limit) / self.reference_price * 100
        return 0.0


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status for index/market."""
    index_name: str
    current_level: CircuitBreakerLevel
    trigger_time: Optional[datetime]
    halt_duration_minutes: int
    resume_time: Optional[datetime]
    trigger_price: float
    reference_price: float
    movement_pct: float

    def is_halted(self) -> bool:
        """Check if trading is halted."""
        if self.current_level == CircuitBreakerLevel.NONE:
            return False
        if self.resume_time and datetime.now() >= self.resume_time:
            return False
        return True


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    is_compliant: bool
    violations: List[ViolationType]
    violation_reason: str
    action: ComplianceAction
    adjusted_quantity: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_compliant': self.is_compliant,
            'violations': [v.value for v in self.violations],
            'violation_reason': self.violation_reason,
            'action': self.action.value,
            'adjusted_quantity': self.adjusted_quantity,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ComplianceAlert:
    """Alert for compliance violation."""
    alert_id: str
    timestamp: datetime
    violation_type: ViolationType
    severity: str  # high, medium, low
    symbol: Optional[str]
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False


@dataclass
class TradingSession:
    """Trading session timing."""
    exchange: Exchange
    segment: str
    pre_open_start: time
    pre_open_end: time
    market_open: time
    market_close: time
    post_close_end: time


class PositionTracker:
    """
    Tracks positions for limit enforcement.

    Maintains real-time position state for compliance checking.
    """

    def __init__(self):
        self._positions: Dict[str, int] = defaultdict(int)  # symbol -> quantity
        self._position_values: Dict[str, float] = defaultdict(float)
        self._daily_turnover: Dict[str, float] = defaultdict(float)
        self._order_count: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._last_reset: date = date.today()

    def update_position(
        self,
        symbol: str,
        quantity_change: int,
        price: float
    ) -> None:
        """Update position after trade."""
        with self._lock:
            self._check_daily_reset()

            self._positions[symbol] += quantity_change
            self._position_values[symbol] = self._positions[symbol] * price
            self._daily_turnover[symbol] += abs(quantity_change * price)
            self._order_count[symbol] += 1

    def get_position(self, symbol: str) -> int:
        """Get current position quantity."""
        with self._lock:
            return self._positions.get(symbol, 0)

    def get_position_value(self, symbol: str) -> float:
        """Get current position value."""
        with self._lock:
            return self._position_values.get(symbol, 0.0)

    def get_daily_turnover(self, symbol: str) -> float:
        """Get daily turnover for symbol."""
        with self._lock:
            self._check_daily_reset()
            return self._daily_turnover.get(symbol, 0.0)

    def get_total_turnover(self) -> float:
        """Get total daily turnover."""
        with self._lock:
            self._check_daily_reset()
            return sum(self._daily_turnover.values())

    def get_order_count(self, symbol: str) -> int:
        """Get order count for symbol."""
        with self._lock:
            self._check_daily_reset()
            return self._order_count.get(symbol, 0)

    def would_exceed_limit(
        self,
        symbol: str,
        quantity_change: int,
        limit: PositionLimit
    ) -> Tuple[bool, str]:
        """Check if order would exceed position limit."""
        with self._lock:
            new_position = self._positions.get(symbol, 0) + quantity_change

            if abs(new_position) > limit.max_quantity:
                return True, f"Position {new_position} exceeds limit {limit.max_quantity}"

            return False, ""

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today > self._last_reset:
            self._daily_turnover.clear()
            self._order_count.clear()
            self._last_reset = today

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._positions.clear()
            self._position_values.clear()
            self._daily_turnover.clear()
            self._order_count.clear()


class CircuitBreakerMonitor:
    """
    Monitors circuit breakers as per SEBI guidelines.

    SEBI circuit breaker rules:
    - 10% movement: 45 min halt (before 1pm), 15 min (after 1pm)
    - 15% movement: 1h45m halt (before 1pm), 45 min (after 1pm)
    - 20% movement: Trading halted for the day
    """

    # SEBI circuit breaker thresholds
    LEVEL_1_PCT = 10.0
    LEVEL_2_PCT = 15.0
    LEVEL_3_PCT = 20.0

    def __init__(self):
        self._index_status: Dict[str, CircuitBreakerStatus] = {}
        self._price_bands: Dict[str, PriceBand] = {}
        self._halted_symbols: Set[str] = set()
        self._lock = threading.Lock()

    def update_index_price(
        self,
        index_name: str,
        current_price: float,
        reference_price: float
    ) -> Optional[CircuitBreakerStatus]:
        """
        Update index price and check for circuit breaker.

        Args:
            index_name: Index name (NIFTY, SENSEX)
            current_price: Current index value
            reference_price: Previous close

        Returns:
            Circuit breaker status if triggered
        """
        movement_pct = ((current_price - reference_price) / reference_price) * 100
        abs_movement = abs(movement_pct)

        # Determine level
        if abs_movement >= self.LEVEL_3_PCT:
            level = CircuitBreakerLevel.LEVEL_3
            halt_minutes = 999  # Rest of day
        elif abs_movement >= self.LEVEL_2_PCT:
            level = CircuitBreakerLevel.LEVEL_2
            halt_minutes = self._get_halt_duration(2)
        elif abs_movement >= self.LEVEL_1_PCT:
            level = CircuitBreakerLevel.LEVEL_1
            halt_minutes = self._get_halt_duration(1)
        else:
            level = CircuitBreakerLevel.NONE
            halt_minutes = 0

        with self._lock:
            current_status = self._index_status.get(index_name)

            # Only trigger if new level is higher
            if current_status and current_status.current_level.value >= level.value:
                return current_status

            if level != CircuitBreakerLevel.NONE:
                now = datetime.now()
                status = CircuitBreakerStatus(
                    index_name=index_name,
                    current_level=level,
                    trigger_time=now,
                    halt_duration_minutes=halt_minutes,
                    resume_time=now + timedelta(minutes=halt_minutes) if halt_minutes < 999 else None,
                    trigger_price=current_price,
                    reference_price=reference_price,
                    movement_pct=movement_pct
                )
                self._index_status[index_name] = status

                logger.warning(
                    f"Circuit breaker {level.value} triggered for {index_name}: "
                    f"{movement_pct:.2f}% movement"
                )

                return status

        return None

    def _get_halt_duration(self, level: int) -> int:
        """Get halt duration based on time of day."""
        now = datetime.now().time()
        one_pm = time(13, 0)

        if level == 1:
            return 45 if now < one_pm else 15
        elif level == 2:
            return 105 if now < one_pm else 45  # 1h45m or 45m
        return 999

    def set_price_band(
        self,
        symbol: str,
        exchange: Exchange,
        reference_price: float,
        lower_pct: float,
        upper_pct: float
    ) -> PriceBand:
        """Set price band for a symbol."""
        band = PriceBand(
            symbol=symbol,
            exchange=exchange,
            reference_price=reference_price,
            lower_limit=reference_price * (1 - lower_pct / 100),
            upper_limit=reference_price * (1 + upper_pct / 100),
            lower_pct=lower_pct,
            upper_pct=upper_pct,
            last_updated=datetime.now()
        )

        with self._lock:
            self._price_bands[f"{exchange.value}:{symbol}"] = band

        return band

    def check_price_band(
        self,
        symbol: str,
        exchange: Exchange,
        price: float
    ) -> Tuple[bool, Optional[PriceBand]]:
        """
        Check if price is within band.

        Returns:
            Tuple of (is_within_band, price_band)
        """
        key = f"{exchange.value}:{symbol}"

        with self._lock:
            band = self._price_bands.get(key)

        if not band:
            return True, None  # No band set, assume OK

        return band.is_within_band(price), band

    def is_trading_halted(self) -> Tuple[bool, Optional[str]]:
        """Check if market-wide trading is halted."""
        with self._lock:
            for index_name, status in self._index_status.items():
                if status.is_halted():
                    return True, f"Circuit breaker {status.current_level.value} on {index_name}"
        return False, None

    def halt_symbol(self, symbol: str, reason: str) -> None:
        """Halt trading in a specific symbol."""
        with self._lock:
            self._halted_symbols.add(symbol)
        logger.warning(f"Symbol halted: {symbol} - {reason}")

    def resume_symbol(self, symbol: str) -> None:
        """Resume trading in a symbol."""
        with self._lock:
            self._halted_symbols.discard(symbol)

    def is_symbol_halted(self, symbol: str) -> bool:
        """Check if symbol is halted."""
        with self._lock:
            return symbol in self._halted_symbols

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                'index_status': {
                    k: {
                        'level': v.current_level.value,
                        'movement_pct': v.movement_pct,
                        'is_halted': v.is_halted()
                    }
                    for k, v in self._index_status.items()
                },
                'halted_symbols': list(self._halted_symbols),
                'price_bands_count': len(self._price_bands)
            }


class SEBIComplianceRules:
    """
    SEBI regulatory compliance rules.

    Implements key SEBI regulations for algo trading:
    - Order-to-trade ratio limits
    - Daily turnover limits
    - Position limits for F&O
    - Algo order tagging requirements
    """

    # SEBI mandated limits
    MAX_ORDER_TO_TRADE_RATIO = 50  # Max orders per executed trade
    MAX_ORDERS_PER_SECOND = 100  # Per symbol
    MIN_ORDER_VALUE = 0  # Minimum order value (can be exchange specific)

    # F&O position limits as % of market-wide position limit
    CLIENT_LEVEL_FO_LIMIT_PCT = 5.0

    def __init__(self):
        self._order_counts: Dict[str, List[datetime]] = defaultdict(list)
        self._trade_counts: Dict[str, int] = defaultdict(int)
        self._algo_orders: Dict[str, str] = {}  # order_id -> algo_id
        self._lock = threading.Lock()

    def check_order_frequency(
        self,
        symbol: str,
        window_seconds: int = 1
    ) -> Tuple[bool, str]:
        """Check if order frequency is within limits."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)

        with self._lock:
            # Clean old orders
            self._order_counts[symbol] = [
                t for t in self._order_counts[symbol]
                if t > cutoff
            ]

            count = len(self._order_counts[symbol])

            if count >= self.MAX_ORDERS_PER_SECOND:
                return False, f"Order frequency {count}/sec exceeds limit {self.MAX_ORDERS_PER_SECOND}"

            self._order_counts[symbol].append(now)

        return True, ""

    def check_order_to_trade_ratio(self, symbol: str) -> Tuple[bool, str]:
        """Check order-to-trade ratio."""
        with self._lock:
            orders = len(self._order_counts.get(symbol, []))
            trades = self._trade_counts.get(symbol, 0)

            if trades == 0:
                if orders > 100:  # Allow some orders before first trade
                    return False, f"High order count ({orders}) with no trades"
                return True, ""

            ratio = orders / trades

            if ratio > self.MAX_ORDER_TO_TRADE_RATIO:
                return False, f"Order-to-trade ratio {ratio:.1f} exceeds limit {self.MAX_ORDER_TO_TRADE_RATIO}"

        return True, ""

    def record_trade(self, symbol: str) -> None:
        """Record a trade execution."""
        with self._lock:
            self._trade_counts[symbol] += 1

    def register_algo_order(self, order_id: str, algo_id: str) -> None:
        """Register algo order for SEBI tagging requirement."""
        with self._lock:
            self._algo_orders[order_id] = algo_id

    def get_algo_id(self, order_id: str) -> Optional[str]:
        """Get algo ID for an order."""
        with self._lock:
            return self._algo_orders.get(order_id)

    def check_fo_position_limit(
        self,
        symbol: str,
        quantity: int,
        market_wide_oi: int
    ) -> Tuple[bool, str]:
        """Check F&O position limit as per SEBI."""
        max_allowed = int(market_wide_oi * self.CLIENT_LEVEL_FO_LIMIT_PCT / 100)

        if quantity > max_allowed:
            return False, f"Position {quantity} exceeds {self.CLIENT_LEVEL_FO_LIMIT_PCT}% of OI ({max_allowed})"

        return True, ""

    def reset_daily(self) -> None:
        """Reset daily counters."""
        with self._lock:
            self._order_counts.clear()
            self._trade_counts.clear()


class MarketHoursChecker:
    """
    Checks market hours and trading sessions.

    Enforces trading only during valid market hours.
    """

    # Default NSE/BSE equity timings
    DEFAULT_SESSIONS = {
        (Exchange.NSE, "EQ"): TradingSession(
            exchange=Exchange.NSE,
            segment="EQ",
            pre_open_start=time(9, 0),
            pre_open_end=time(9, 8),
            market_open=time(9, 15),
            market_close=time(15, 30),
            post_close_end=time(16, 0)
        ),
        (Exchange.BSE, "EQ"): TradingSession(
            exchange=Exchange.BSE,
            segment="EQ",
            pre_open_start=time(9, 0),
            pre_open_end=time(9, 8),
            market_open=time(9, 15),
            market_close=time(15, 30),
            post_close_end=time(16, 0)
        ),
        (Exchange.NSE, "FO"): TradingSession(
            exchange=Exchange.NSE,
            segment="FO",
            pre_open_start=time(9, 0),
            pre_open_end=time(9, 8),
            market_open=time(9, 15),
            market_close=time(15, 30),
            post_close_end=time(16, 0)
        ),
    }

    # Indian market holidays (sample - should be updated annually)
    HOLIDAYS_2024 = {
        date(2024, 1, 26),   # Republic Day
        date(2024, 3, 8),    # Mahashivratri
        date(2024, 3, 25),   # Holi
        date(2024, 3, 29),   # Good Friday
        date(2024, 4, 11),   # Id-ul-Fitr
        date(2024, 4, 14),   # Ambedkar Jayanti
        date(2024, 4, 17),   # Ram Navami
        date(2024, 4, 21),   # Mahavir Jayanti
        date(2024, 5, 1),    # May Day
        date(2024, 5, 23),   # Buddha Purnima
        date(2024, 6, 17),   # Id-ul-Adha
        date(2024, 7, 17),   # Muharram
        date(2024, 8, 15),   # Independence Day
        date(2024, 10, 2),   # Gandhi Jayanti
        date(2024, 11, 1),   # Diwali
        date(2024, 11, 15),  # Guru Nanak Jayanti
        date(2024, 12, 25),  # Christmas
    }

    def __init__(self):
        self._sessions = dict(self.DEFAULT_SESSIONS)
        self._holidays: Set[date] = set(self.HOLIDAYS_2024)
        self._special_timings: Dict[date, TradingSession] = {}

    def get_market_status(
        self,
        exchange: Exchange = Exchange.NSE,
        segment: str = "EQ"
    ) -> MarketStatus:
        """Get current market status."""
        now = datetime.now()
        today = now.date()
        current_time = now.time()

        # Check if holiday or weekend
        if today in self._holidays:
            return MarketStatus.CLOSED
        if today.weekday() >= 5:  # Saturday or Sunday
            return MarketStatus.CLOSED

        # Get session timing
        session = self._sessions.get((exchange, segment))
        if not session:
            return MarketStatus.CLOSED

        # Check special timing for today
        if today in self._special_timings:
            session = self._special_timings[today]

        # Determine status based on time
        if current_time < session.pre_open_start:
            return MarketStatus.CLOSED
        elif current_time < session.pre_open_end:
            return MarketStatus.PRE_OPEN
        elif current_time < session.market_open:
            return MarketStatus.PRE_OPEN
        elif current_time < session.market_close:
            return MarketStatus.OPEN
        elif current_time < session.post_close_end:
            return MarketStatus.POST_CLOSE
        else:
            return MarketStatus.CLOSED

    def is_trading_allowed(
        self,
        exchange: Exchange = Exchange.NSE,
        segment: str = "EQ"
    ) -> Tuple[bool, str]:
        """Check if trading is currently allowed."""
        status = self.get_market_status(exchange, segment)

        if status == MarketStatus.OPEN:
            return True, ""
        elif status == MarketStatus.PRE_OPEN:
            return False, "Market in pre-open session"
        elif status == MarketStatus.POST_CLOSE:
            return False, "Market in post-close session"
        elif status == MarketStatus.HALTED:
            return False, "Market halted"
        else:
            return False, "Market closed"

    def add_holiday(self, holiday: date) -> None:
        """Add a holiday."""
        self._holidays.add(holiday)

    def set_special_timing(
        self,
        special_date: date,
        session: TradingSession
    ) -> None:
        """Set special timing for a date (e.g., Muhurat trading)."""
        self._special_timings[special_date] = session

    def is_holiday(self, check_date: date) -> bool:
        """Check if date is a holiday."""
        return check_date in self._holidays or check_date.weekday() >= 5


class ComplianceEngine:
    """
    Main compliance engine combining all compliance checks.

    Provides unified interface for order compliance validation.
    """

    def __init__(
        self,
        max_order_value: float = 10_000_000,  # 1 crore
        max_daily_turnover: float = 100_000_000,  # 10 crore
        enable_market_hours_check: bool = True,
        enable_circuit_breaker_check: bool = True,
        enable_sebi_rules: bool = True
    ):
        """
        Initialize compliance engine.

        Args:
            max_order_value: Maximum single order value
            max_daily_turnover: Maximum daily turnover
            enable_market_hours_check: Check market hours
            enable_circuit_breaker_check: Check circuit breakers
            enable_sebi_rules: Enable SEBI compliance rules
        """
        self.max_order_value = max_order_value
        self.max_daily_turnover = max_daily_turnover
        self.enable_market_hours_check = enable_market_hours_check
        self.enable_circuit_breaker_check = enable_circuit_breaker_check
        self.enable_sebi_rules = enable_sebi_rules

        # Components
        self.position_tracker = PositionTracker()
        self.circuit_breaker = CircuitBreakerMonitor()
        self.sebi_rules = SEBIComplianceRules()
        self.market_hours = MarketHoursChecker()

        # Position limits
        self._position_limits: Dict[str, PositionLimit] = {}

        # Alerts
        self._alerts: List[ComplianceAlert] = []
        self._alert_callbacks: List[Callable[[ComplianceAlert], None]] = []
        self._lock = threading.Lock()
        self._alert_counter = 0

    def set_position_limit(self, limit: PositionLimit) -> None:
        """Set position limit for a symbol."""
        key = f"{limit.exchange.value}:{limit.symbol}"
        self._position_limits[key] = limit

    def check_order_compliance(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        exchange: Exchange = Exchange.NSE,
        segment: str = "EQ",
        order_type: str = "LIMIT",
        algo_id: Optional[str] = None
    ) -> ComplianceResult:
        """
        Check if order is compliant with all rules.

        Args:
            symbol: Trading symbol
            side: buy or sell
            quantity: Order quantity
            price: Order price
            exchange: Exchange
            segment: Market segment
            order_type: Order type
            algo_id: Algo identifier for SEBI tagging

        Returns:
            Compliance check result
        """
        violations = []
        details = {}
        adjusted_quantity = None

        # 1. Market hours check
        if self.enable_market_hours_check:
            is_open, reason = self.market_hours.is_trading_allowed(exchange, segment)
            if not is_open:
                violations.append(ViolationType.MARKET_CLOSED)
                details['market_status'] = reason

        # 2. Circuit breaker check
        if self.enable_circuit_breaker_check:
            is_halted, reason = self.circuit_breaker.is_trading_halted()
            if is_halted:
                violations.append(ViolationType.CIRCUIT_BREAKER_HIT)
                details['circuit_breaker'] = reason

            # Check symbol-specific halt
            if self.circuit_breaker.is_symbol_halted(symbol):
                violations.append(ViolationType.SYMBOL_SUSPENDED)
                details['symbol_halt'] = f"{symbol} trading suspended"

            # Check price band
            is_within, band = self.circuit_breaker.check_price_band(symbol, exchange, price)
            if not is_within and band:
                violations.append(ViolationType.PRICE_BAND_VIOLATION)
                details['price_band'] = {
                    'price': price,
                    'lower': band.lower_limit,
                    'upper': band.upper_limit
                }

        # 3. Order value check
        order_value = quantity * price
        if order_value > self.max_order_value:
            violations.append(ViolationType.ORDER_VALUE_LIMIT)
            details['order_value'] = {
                'value': order_value,
                'limit': self.max_order_value
            }
            # Calculate adjusted quantity
            adjusted_quantity = int(self.max_order_value / price)

        # 4. Position limit check
        key = f"{exchange.value}:{symbol}"
        limit = self._position_limits.get(key)

        if limit:
            quantity_change = quantity if side.lower() == 'buy' else -quantity
            exceeds, reason = self.position_tracker.would_exceed_limit(
                symbol, quantity_change, limit
            )
            if exceeds:
                violations.append(ViolationType.POSITION_LIMIT_EXCEEDED)
                details['position_limit'] = reason

        # 5. Daily turnover check
        new_turnover = self.position_tracker.get_total_turnover() + order_value
        if new_turnover > self.max_daily_turnover:
            violations.append(ViolationType.TURNOVER_LIMIT_EXCEEDED)
            details['turnover'] = {
                'current': self.position_tracker.get_total_turnover(),
                'order': order_value,
                'limit': self.max_daily_turnover
            }

        # 6. SEBI rules check
        if self.enable_sebi_rules:
            # Order frequency
            freq_ok, reason = self.sebi_rules.check_order_frequency(symbol)
            if not freq_ok:
                violations.append(ViolationType.ORDER_FREQUENCY_LIMIT)
                details['order_frequency'] = reason

            # Order-to-trade ratio
            otr_ok, reason = self.sebi_rules.check_order_to_trade_ratio(symbol)
            if not otr_ok:
                violations.append(ViolationType.SEBI_RULE_VIOLATION)
                details['order_to_trade'] = reason

        # Determine action
        if violations:
            # Critical violations that should block
            blocking_violations = {
                ViolationType.MARKET_CLOSED,
                ViolationType.CIRCUIT_BREAKER_HIT,
                ViolationType.SYMBOL_SUSPENDED,
                ViolationType.TRADING_HALTED
            }

            if any(v in blocking_violations for v in violations):
                action = ComplianceAction.BLOCK
            elif ViolationType.ORDER_VALUE_LIMIT in violations and adjusted_quantity:
                action = ComplianceAction.REDUCE_SIZE
            else:
                action = ComplianceAction.WARN

            # Create alert
            self._create_alert(
                violations[0],
                symbol,
                f"Order compliance violation: {violations[0].value}",
                details
            )

            return ComplianceResult(
                is_compliant=False,
                violations=violations,
                violation_reason="; ".join(v.value for v in violations),
                action=action,
                adjusted_quantity=adjusted_quantity,
                details=details
            )

        return ComplianceResult(
            is_compliant=True,
            violations=[],
            violation_reason="",
            action=ComplianceAction.LOG_ONLY,
            details={'checks_passed': True}
        )

    def record_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: Optional[str] = None,
        algo_id: Optional[str] = None
    ) -> None:
        """Record order execution for tracking."""
        quantity_change = quantity if side.lower() == 'buy' else -quantity
        self.position_tracker.update_position(symbol, quantity_change, price)
        self.sebi_rules.record_trade(symbol)

        if order_id and algo_id:
            self.sebi_rules.register_algo_order(order_id, algo_id)

    def _create_alert(
        self,
        violation_type: ViolationType,
        symbol: Optional[str],
        message: str,
        details: Dict[str, Any]
    ) -> ComplianceAlert:
        """Create and store compliance alert."""
        with self._lock:
            self._alert_counter += 1

            # Determine severity
            high_severity = {
                ViolationType.CIRCUIT_BREAKER_HIT,
                ViolationType.POSITION_LIMIT_EXCEEDED,
                ViolationType.MARGIN_INSUFFICIENT
            }
            severity = "high" if violation_type in high_severity else "medium"

            alert = ComplianceAlert(
                alert_id=f"alert_{self._alert_counter}",
                timestamp=datetime.now(),
                violation_type=violation_type,
                severity=severity,
                symbol=symbol,
                message=message,
                details=details
            )

            self._alerts.append(alert)

            # Notify callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        return alert

    def register_alert_callback(
        self,
        callback: Callable[[ComplianceAlert], None]
    ) -> None:
        """Register callback for compliance alerts."""
        self._alert_callbacks.append(callback)

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        violation_type: Optional[ViolationType] = None,
        unacknowledged_only: bool = False
    ) -> List[ComplianceAlert]:
        """Get compliance alerts."""
        with self._lock:
            alerts = list(self._alerts)

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if violation_type:
            alerts = [a for a in alerts if a.violation_type == violation_type]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get compliance engine status."""
        return {
            'position_tracker': {
                'total_turnover': self.position_tracker.get_total_turnover()
            },
            'circuit_breaker': self.circuit_breaker.get_status(),
            'market_status': self.market_hours.get_market_status().value,
            'alerts_count': len(self._alerts),
            'unacknowledged_alerts': len([a for a in self._alerts if not a.acknowledged])
        }

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.position_tracker.reset()
        self.sebi_rules.reset_daily()


# Convenience functions
_default_engine: Optional[ComplianceEngine] = None


def get_compliance_engine() -> ComplianceEngine:
    """Get default compliance engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = ComplianceEngine()
    return _default_engine


def set_compliance_engine(engine: ComplianceEngine) -> None:
    """Set default compliance engine."""
    global _default_engine
    _default_engine = engine


def check_compliance(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    **kwargs
) -> ComplianceResult:
    """Check order compliance using default engine."""
    return get_compliance_engine().check_order_compliance(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs
    )


def is_market_open(
    exchange: Exchange = Exchange.NSE,
    segment: str = "EQ"
) -> bool:
    """Check if market is open."""
    status = get_compliance_engine().market_hours.get_market_status(exchange, segment)
    return status == MarketStatus.OPEN
