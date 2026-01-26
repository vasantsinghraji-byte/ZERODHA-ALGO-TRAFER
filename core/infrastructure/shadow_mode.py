# -*- coding: utf-8 -*-
"""
Shadow Trading Engine - Paper Trading Validation System
========================================================
Production-grade shadow trading for validating strategies
before deploying with real money.

Shadow mode runs strategies in parallel with live trading,
receiving real market data but simulating order execution
internally without sending orders to the broker.

Features:
- Parallel strategy execution (shadow + live)
- Simulated order execution with realistic fills
- Real-time P&L comparison
- Statistical validation for go-live decision
- Risk-free strategy testing with live data

Example:
    >>> from core.infrastructure.shadow_mode import ShadowEngine
    >>>
    >>> # Create shadow engine
    >>> engine = ShadowEngine()
    >>>
    >>> # Register shadow strategy
    >>> engine.register_shadow_strategy("momentum_v2", strategy)
    >>>
    >>> # Process live ticks
    >>> engine.on_tick(tick_data)
    >>>
    >>> # Compare performance
    >>> comparison = engine.compare_performance()
    >>> if comparison.is_shadow_better():
    ...     engine.promote_to_live()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Deque
from datetime import datetime, timedelta, date
from collections import defaultdict, deque
import threading
import logging
import uuid
import copy

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


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ShadowMode(Enum):
    """Shadow engine modes."""
    DISABLED = "disabled"
    SHADOW_ONLY = "shadow_only"
    PARALLEL = "parallel"  # Both shadow and live
    VALIDATION = "validation"  # Collecting validation data


class ValidationStatus(Enum):
    """Validation gate status."""
    PENDING = "pending"
    COLLECTING = "collecting"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ShadowOrder:
    """Simulated order in shadow mode."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]  # Limit price
    stop_price: Optional[float]  # Stop trigger price
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    filled_price: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    strategy_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED
        )

    @property
    def remaining_quantity(self) -> int:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class ShadowPosition:
    """Simulated position in shadow mode."""
    symbol: str
    quantity: int
    avg_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: datetime
    updated_at: datetime
    strategy_id: str = ""

    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        """Get cost basis."""
        return self.quantity * self.avg_price


@dataclass
class ShadowTrade:
    """Record of a simulated trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    strategy_id: str
    pnl: float = 0.0  # Realized P&L for closing trades


@dataclass
class PerformanceMetrics:
    """Performance metrics for comparison."""
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_pnl: float
    daily_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_trade_pnl': self.avg_trade_pnl
        }


@dataclass
class PerformanceComparison:
    """Comparison between shadow and live performance."""
    shadow_metrics: PerformanceMetrics
    live_metrics: PerformanceMetrics
    comparison_period: timedelta
    start_time: datetime
    end_time: datetime
    correlation: Optional[float]
    tracking_error: Optional[float]
    p_value: Optional[float]  # Statistical significance
    is_statistically_significant: bool

    def is_shadow_better(self, min_improvement: float = 0.0) -> bool:
        """Check if shadow strategy outperforms live."""
        return (
            self.shadow_metrics.total_pnl >
            self.live_metrics.total_pnl * (1 + min_improvement)
        )

    def get_pnl_difference(self) -> float:
        """Get P&L difference (shadow - live)."""
        return self.shadow_metrics.total_pnl - self.live_metrics.total_pnl

    def get_sharpe_difference(self) -> float:
        """Get Sharpe ratio difference."""
        return self.shadow_metrics.sharpe_ratio - self.live_metrics.sharpe_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'shadow': self.shadow_metrics.to_dict(),
            'live': self.live_metrics.to_dict(),
            'comparison_period_hours': self.comparison_period.total_seconds() / 3600,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'correlation': self.correlation,
            'tracking_error': self.tracking_error,
            'p_value': self.p_value,
            'is_statistically_significant': self.is_statistically_significant,
            'pnl_difference': self.get_pnl_difference(),
            'sharpe_difference': self.get_sharpe_difference()
        }


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    status: ValidationStatus
    metrics: PerformanceMetrics
    comparison: Optional[PerformanceComparison]
    validation_period: timedelta
    min_trades_required: int
    actual_trades: int
    criteria_results: Dict[str, bool]
    recommendation: str
    confidence: float  # 0-1

    def is_ready_for_production(self) -> bool:
        """Check if strategy passed all validation criteria."""
        return self.status == ValidationStatus.PASSED


class ShadowBroker:
    """
    Simulated broker for shadow trading.

    Executes orders internally without sending to real broker.
    Simulates realistic fills with slippage and commissions.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,  # 5 basis points
        commission_per_order: float = 20.0,
        fill_probability: float = 0.98,
        partial_fill_enabled: bool = True
    ):
        """
        Initialize shadow broker.

        Args:
            slippage_bps: Slippage in basis points
            commission_per_order: Commission per order
            fill_probability: Probability of order fill
            partial_fill_enabled: Allow partial fills
        """
        self.slippage_bps = slippage_bps
        self.commission_per_order = commission_per_order
        self.fill_probability = fill_probability
        self.partial_fill_enabled = partial_fill_enabled

        self._orders: Dict[str, ShadowOrder] = {}
        self._positions: Dict[str, ShadowPosition] = {}
        self._trades: List[ShadowTrade] = []
        self._pending_orders: List[str] = []
        self._market_prices: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Callbacks
        self._on_fill: Optional[Callable[[ShadowOrder, ShadowTrade], None]] = None
        self._on_order_update: Optional[Callable[[ShadowOrder], None]] = None

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> ShadowOrder:
        """
        Place a simulated order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            stop_price: Stop trigger price
            strategy_id: Strategy identifier
            tags: Custom tags

        Returns:
            Created order
        """
        order_id = f"shadow_{uuid.uuid4().hex[:12]}"

        order = ShadowOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            strategy_id=strategy_id,
            tags=tags or {}
        )

        with self._lock:
            self._orders[order_id] = order

            # Market orders fill immediately
            if order_type == OrderType.MARKET:
                self._try_fill_order(order)
            else:
                self._pending_orders.append(order_id)
                order.status = OrderStatus.OPEN

        logger.debug(f"Shadow order placed: {order_id} {side.value} {quantity} {symbol}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        with self._lock:
            if order_id not in self._orders:
                return False

            order = self._orders[order_id]
            if order.is_complete:
                return False

            order.status = OrderStatus.CANCELLED

            if order_id in self._pending_orders:
                self._pending_orders.remove(order_id)

        return True

    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update market price and check pending orders.

        Args:
            symbol: Symbol
            price: Current market price
        """
        with self._lock:
            self._market_prices[symbol] = price

            # Update position mark-to-market
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.market_price = price
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity
                pos.updated_at = datetime.now()

            # Check pending orders
            orders_to_check = [
                oid for oid in self._pending_orders
                if self._orders[oid].symbol == symbol
            ]

            for order_id in orders_to_check:
                order = self._orders[order_id]
                self._check_order_trigger(order, price)

    def _check_order_trigger(self, order: ShadowOrder, price: float) -> None:
        """Check if order should be triggered/filled."""
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and price <= order.price:
                self._try_fill_order(order, fill_price=order.price)
            elif order.side == OrderSide.SELL and price >= order.price:
                self._try_fill_order(order, fill_price=order.price)

        elif order.order_type == OrderType.STOP_LOSS:
            if order.side == OrderSide.BUY and price >= order.stop_price:
                self._try_fill_order(order)
            elif order.side == OrderSide.SELL and price <= order.stop_price:
                self._try_fill_order(order)

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY and price >= order.stop_price:
                # Convert to limit order
                order.order_type = OrderType.LIMIT
            elif order.side == OrderSide.SELL and price <= order.stop_price:
                order.order_type = OrderType.LIMIT

    def _try_fill_order(
        self,
        order: ShadowOrder,
        fill_price: Optional[float] = None
    ) -> None:
        """Attempt to fill an order."""
        import random

        # Check fill probability
        if random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            return

        symbol = order.symbol
        market_price = fill_price or self._market_prices.get(symbol, 0)

        if market_price <= 0:
            return

        # Calculate slippage
        slippage_factor = self.slippage_bps / 10000
        if order.side == OrderSide.BUY:
            actual_price = market_price * (1 + slippage_factor)
        else:
            actual_price = market_price * (1 - slippage_factor)

        # Determine fill quantity
        if self.partial_fill_enabled and random.random() < 0.1:
            fill_qty = max(1, int(order.remaining_quantity * random.uniform(0.5, 0.9)))
        else:
            fill_qty = order.remaining_quantity

        # Update order
        order.filled_quantity += fill_qty
        order.filled_price = (
            (order.filled_price * (order.filled_quantity - fill_qty) +
             actual_price * fill_qty) / order.filled_quantity
        )
        order.slippage = abs(actual_price - market_price) * fill_qty
        order.commission = self.commission_per_order
        order.filled_at = datetime.now()

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            if order.order_id in self._pending_orders:
                self._pending_orders.remove(order.order_id)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Create trade record
        trade = self._create_trade(order, fill_qty, actual_price)

        # Update position
        self._update_position(order, fill_qty, actual_price)

        # Callbacks
        if self._on_fill:
            self._on_fill(order, trade)
        if self._on_order_update:
            self._on_order_update(order)

    def _create_trade(
        self,
        order: ShadowOrder,
        quantity: int,
        price: float
    ) -> ShadowTrade:
        """Create trade record."""
        trade = ShadowTrade(
            trade_id=f"trade_{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=order.commission,
            timestamp=datetime.now(),
            strategy_id=order.strategy_id
        )

        self._trades.append(trade)
        return trade

    def _update_position(
        self,
        order: ShadowOrder,
        quantity: int,
        price: float
    ) -> None:
        """Update position after fill."""
        symbol = order.symbol
        now = datetime.now()

        if symbol not in self._positions:
            self._positions[symbol] = ShadowPosition(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                market_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                opened_at=now,
                updated_at=now,
                strategy_id=order.strategy_id
            )

        pos = self._positions[symbol]

        if order.side == OrderSide.BUY:
            # Buying increases position
            new_qty = pos.quantity + quantity
            if new_qty != 0:
                pos.avg_price = (
                    (pos.avg_price * pos.quantity + price * quantity) / new_qty
                )
            pos.quantity = new_qty

        else:  # SELL
            # Selling decreases position, realize P&L
            if pos.quantity > 0:
                close_qty = min(quantity, pos.quantity)
                realized = (price - pos.avg_price) * close_qty
                pos.realized_pnl += realized

                # Update last trade with realized P&L
                if self._trades:
                    self._trades[-1].pnl = realized

            pos.quantity -= quantity

        pos.market_price = price
        pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity if pos.quantity else 0
        pos.updated_at = now

        # Remove closed positions
        if pos.quantity == 0:
            del self._positions[symbol]

    def get_position(self, symbol: str) -> Optional[ShadowPosition]:
        """Get current position for symbol."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, ShadowPosition]:
        """Get all positions."""
        with self._lock:
            return dict(self._positions)

    def get_order(self, order_id: str) -> Optional[ShadowOrder]:
        """Get order by ID."""
        with self._lock:
            return self._orders.get(order_id)

    def get_trades(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[ShadowTrade]:
        """Get trades with optional filters."""
        with self._lock:
            trades = list(self._trades)

        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if since:
            trades = [t for t in trades if t.timestamp >= since]

        return trades

    def get_total_pnl(self) -> Tuple[float, float]:
        """Get total realized and unrealized P&L."""
        with self._lock:
            realized = sum(t.pnl for t in self._trades)
            unrealized = sum(p.unrealized_pnl for p in self._positions.values())

        return realized, unrealized

    def on_fill(self, callback: Callable[[ShadowOrder, ShadowTrade], None]) -> None:
        """Register fill callback."""
        self._on_fill = callback

    def on_order_update(self, callback: Callable[[ShadowOrder], None]) -> None:
        """Register order update callback."""
        self._on_order_update = callback

    def reset(self) -> None:
        """Reset broker state."""
        with self._lock:
            self._orders.clear()
            self._positions.clear()
            self._trades.clear()
            self._pending_orders.clear()
            self._market_prices.clear()


class ShadowStrategy:
    """
    Wrapper for running a strategy in shadow mode.

    Intercepts order placement and routes to shadow broker.
    """

    def __init__(
        self,
        strategy_id: str,
        strategy: Any,
        shadow_broker: ShadowBroker
    ):
        """
        Initialize shadow strategy.

        Args:
            strategy_id: Unique identifier
            strategy: The actual strategy object
            shadow_broker: Shadow broker for simulated execution
        """
        self.strategy_id = strategy_id
        self.strategy = strategy
        self.shadow_broker = shadow_broker

        self._enabled = True
        self._tick_count = 0
        self._signal_count = 0

    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """Process tick data."""
        if not self._enabled:
            return

        self._tick_count += 1

        # Update shadow broker with price
        symbol = tick_data.get('symbol', '')
        ltp = tick_data.get('ltp', 0)

        if symbol and ltp:
            self.shadow_broker.update_market_price(symbol, ltp)

        # Call strategy's on_tick
        if hasattr(self.strategy, 'on_tick'):
            self.strategy.on_tick(tick_data)

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        **kwargs
    ) -> ShadowOrder:
        """Place order through shadow broker."""
        self._signal_count += 1

        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType(kwargs.get('order_type', 'market'))

        return self.shadow_broker.place_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            order_type=order_type,
            price=kwargs.get('price'),
            stop_price=kwargs.get('stop_price'),
            strategy_id=self.strategy_id,
            tags=kwargs.get('tags')
        )

    def get_position(self, symbol: str) -> Optional[ShadowPosition]:
        """Get current position."""
        return self.shadow_broker.get_position(symbol)

    def enable(self) -> None:
        """Enable shadow strategy."""
        self._enabled = True

    def disable(self) -> None:
        """Disable shadow strategy."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if enabled."""
        return self._enabled


class PnLTracker:
    """
    Tracks P&L over time for comparison.

    Records daily returns and calculates performance metrics.
    """

    def __init__(self, name: str):
        """
        Initialize P&L tracker.

        Args:
            name: Tracker name (e.g., "shadow", "live")
        """
        self.name = name

        self._daily_pnl: Dict[date, float] = {}
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._trades: List[Dict[str, Any]] = []
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._max_drawdown = 0.0
        self._lock = threading.Lock()

    def record_trade(
        self,
        pnl: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a trade result."""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self._trades.append({
                'pnl': pnl,
                'timestamp': timestamp
            })

            # Update daily P&L
            trade_date = timestamp.date()
            self._daily_pnl[trade_date] = self._daily_pnl.get(trade_date, 0) + pnl

            # Update equity curve
            self._current_equity += pnl
            self._equity_curve.append((timestamp, self._current_equity))

            # Update peak and drawdown
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity
            else:
                drawdown = (self._peak_equity - self._current_equity) / max(self._peak_equity, 1)
                self._max_drawdown = max(self._max_drawdown, drawdown)

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update unrealized P&L."""
        with self._lock:
            total = sum(t['pnl'] for t in self._trades) + unrealized_pnl
            self._equity_curve.append((datetime.now(), total))

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        with self._lock:
            if not self._trades:
                return PerformanceMetrics(
                    total_pnl=0, realized_pnl=0, unrealized_pnl=0,
                    num_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                    profit_factor=0, sharpe_ratio=0, max_drawdown=0,
                    avg_trade_pnl=0, daily_returns=[]
                )

            trade_pnls = [t['pnl'] for t in self._trades]
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]

            total_pnl = sum(trade_pnls)
            num_trades = len(trade_pnls)
            win_rate = len(wins) / num_trades if num_trades > 0 else 0

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calculate Sharpe ratio from daily returns
            daily_returns = list(self._daily_pnl.values())
            if len(daily_returns) > 1 and NUMPY_AVAILABLE:
                returns_array = np.array(daily_returns)
                sharpe_ratio = (
                    np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                    if np.std(returns_array) > 0 else 0
                )
            else:
                sharpe_ratio = 0

            return PerformanceMetrics(
                total_pnl=total_pnl,
                realized_pnl=total_pnl,
                unrealized_pnl=0,  # Updated separately
                num_trades=num_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self._max_drawdown,
                avg_trade_pnl=total_pnl / num_trades if num_trades > 0 else 0,
                daily_returns=daily_returns
            )

    def get_daily_returns(self) -> List[float]:
        """Get list of daily returns."""
        with self._lock:
            return list(self._daily_pnl.values())

    def reset(self) -> None:
        """Reset tracker."""
        with self._lock:
            self._daily_pnl.clear()
            self._equity_curve.clear()
            self._trades.clear()
            self._peak_equity = 0.0
            self._current_equity = 0.0
            self._max_drawdown = 0.0


class ValidationGate:
    """
    Validates shadow strategy before production deployment.

    Applies statistical tests and criteria checking.
    """

    def __init__(
        self,
        min_trades: int = 50,
        min_days: int = 5,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.15,
        min_win_rate: float = 0.4,
        confidence_level: float = 0.95
    ):
        """
        Initialize validation gate.

        Args:
            min_trades: Minimum trades required
            min_days: Minimum trading days
            min_sharpe: Minimum Sharpe ratio
            max_drawdown: Maximum allowed drawdown
            min_win_rate: Minimum win rate
            confidence_level: Statistical confidence level
        """
        self.min_trades = min_trades
        self.min_days = min_days
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.confidence_level = confidence_level

    def validate(
        self,
        shadow_tracker: PnLTracker,
        live_tracker: Optional[PnLTracker] = None,
        validation_start: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate shadow strategy performance.

        Args:
            shadow_tracker: Shadow P&L tracker
            live_tracker: Live P&L tracker for comparison
            validation_start: Start of validation period

        Returns:
            Validation result
        """
        shadow_metrics = shadow_tracker.get_metrics()
        validation_period = timedelta(days=self.min_days)

        if validation_start:
            validation_period = datetime.now() - validation_start

        # Check criteria
        criteria = {}

        criteria['min_trades'] = shadow_metrics.num_trades >= self.min_trades
        criteria['min_sharpe'] = shadow_metrics.sharpe_ratio >= self.min_sharpe
        criteria['max_drawdown'] = shadow_metrics.max_drawdown <= self.max_drawdown
        criteria['min_win_rate'] = shadow_metrics.win_rate >= self.min_win_rate
        criteria['profitable'] = shadow_metrics.total_pnl > 0

        # Compare with live if available
        comparison = None
        if live_tracker:
            comparison = self._compare_performance(shadow_tracker, live_tracker)
            criteria['outperforms_live'] = comparison.is_shadow_better()
            criteria['statistically_significant'] = comparison.is_statistically_significant

        # Determine status
        all_passed = all(criteria.values())
        num_passed = sum(criteria.values())
        num_criteria = len(criteria)

        if shadow_metrics.num_trades < self.min_trades:
            status = ValidationStatus.COLLECTING
            recommendation = f"Need {self.min_trades - shadow_metrics.num_trades} more trades"
            confidence = 0.0
        elif all_passed:
            status = ValidationStatus.PASSED
            recommendation = "Strategy validated. Ready for production."
            confidence = num_passed / num_criteria
        elif num_passed >= num_criteria * 0.7:
            status = ValidationStatus.INCONCLUSIVE
            failed = [k for k, v in criteria.items() if not v]
            recommendation = f"Borderline. Failed criteria: {', '.join(failed)}"
            confidence = num_passed / num_criteria
        else:
            status = ValidationStatus.FAILED
            failed = [k for k, v in criteria.items() if not v]
            recommendation = f"Validation failed: {', '.join(failed)}"
            confidence = num_passed / num_criteria

        return ValidationResult(
            status=status,
            metrics=shadow_metrics,
            comparison=comparison,
            validation_period=validation_period,
            min_trades_required=self.min_trades,
            actual_trades=shadow_metrics.num_trades,
            criteria_results=criteria,
            recommendation=recommendation,
            confidence=confidence
        )

    def _compare_performance(
        self,
        shadow_tracker: PnLTracker,
        live_tracker: PnLTracker
    ) -> PerformanceComparison:
        """Compare shadow vs live performance."""
        shadow_metrics = shadow_tracker.get_metrics()
        live_metrics = live_tracker.get_metrics()

        shadow_returns = shadow_tracker.get_daily_returns()
        live_returns = live_tracker.get_daily_returns()

        correlation = None
        tracking_error = None
        p_value = None
        is_significant = False

        if NUMPY_AVAILABLE and len(shadow_returns) > 5 and len(live_returns) > 5:
            # Align returns
            min_len = min(len(shadow_returns), len(live_returns))
            shadow_arr = np.array(shadow_returns[-min_len:])
            live_arr = np.array(live_returns[-min_len:])

            # Correlation
            if np.std(shadow_arr) > 0 and np.std(live_arr) > 0:
                correlation = np.corrcoef(shadow_arr, live_arr)[0, 1]

            # Tracking error
            diff = shadow_arr - live_arr
            tracking_error = np.std(diff) * np.sqrt(252)

            # Statistical test
            if SCIPY_AVAILABLE:
                _, p_value = stats.ttest_rel(shadow_arr, live_arr)
                is_significant = p_value < (1 - self.confidence_level)

        return PerformanceComparison(
            shadow_metrics=shadow_metrics,
            live_metrics=live_metrics,
            comparison_period=timedelta(days=len(shadow_returns)),
            start_time=datetime.now() - timedelta(days=len(shadow_returns)),
            end_time=datetime.now(),
            correlation=correlation,
            tracking_error=tracking_error,
            p_value=p_value,
            is_statistically_significant=is_significant
        )


class ShadowEngine:
    """
    Main shadow trading engine.

    Manages parallel execution of shadow and live strategies.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission: float = 20.0
    ):
        """
        Initialize shadow engine.

        Args:
            slippage_bps: Simulated slippage in basis points
            commission: Simulated commission per order
        """
        self.slippage_bps = slippage_bps
        self.commission = commission

        self._shadow_broker = ShadowBroker(
            slippage_bps=slippage_bps,
            commission_per_order=commission
        )

        self._shadow_strategies: Dict[str, ShadowStrategy] = {}
        self._shadow_trackers: Dict[str, PnLTracker] = {}
        self._live_tracker: Optional[PnLTracker] = None
        self._validation_gate = ValidationGate()

        self._mode = ShadowMode.DISABLED
        self._validation_start: Optional[datetime] = None
        self._lock = threading.Lock()

        # Wire up callbacks
        self._shadow_broker.on_fill(self._on_shadow_fill)

    def set_mode(self, mode: ShadowMode) -> None:
        """Set shadow engine mode."""
        self._mode = mode
        logger.info(f"Shadow engine mode: {mode.value}")

    def register_shadow_strategy(
        self,
        strategy_id: str,
        strategy: Any
    ) -> ShadowStrategy:
        """
        Register a strategy to run in shadow mode.

        Args:
            strategy_id: Unique identifier
            strategy: Strategy object

        Returns:
            Shadow strategy wrapper
        """
        shadow = ShadowStrategy(
            strategy_id=strategy_id,
            strategy=strategy,
            shadow_broker=self._shadow_broker
        )

        tracker = PnLTracker(f"shadow_{strategy_id}")

        with self._lock:
            self._shadow_strategies[strategy_id] = shadow
            self._shadow_trackers[strategy_id] = tracker

        logger.info(f"Registered shadow strategy: {strategy_id}")

        return shadow

    def unregister_shadow_strategy(self, strategy_id: str) -> None:
        """Unregister a shadow strategy."""
        with self._lock:
            if strategy_id in self._shadow_strategies:
                del self._shadow_strategies[strategy_id]
            if strategy_id in self._shadow_trackers:
                del self._shadow_trackers[strategy_id]

    def set_live_tracker(self, tracker: PnLTracker) -> None:
        """Set live P&L tracker for comparison."""
        self._live_tracker = tracker

    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Process tick data for all shadow strategies.

        Args:
            tick_data: Market tick data
        """
        if self._mode == ShadowMode.DISABLED:
            return

        with self._lock:
            strategies = list(self._shadow_strategies.values())

        for shadow in strategies:
            try:
                shadow.on_tick(tick_data)
            except Exception as e:
                logger.error(f"Shadow strategy error: {e}")

    def record_live_trade(self, pnl: float) -> None:
        """Record a live trade for comparison."""
        if self._live_tracker:
            self._live_tracker.record_trade(pnl)

    def _on_shadow_fill(
        self,
        order: ShadowOrder,
        trade: ShadowTrade
    ) -> None:
        """Handle shadow fill."""
        strategy_id = order.strategy_id

        if strategy_id in self._shadow_trackers:
            self._shadow_trackers[strategy_id].record_trade(trade.pnl)

    def start_validation(self) -> None:
        """Start validation period."""
        self._mode = ShadowMode.VALIDATION
        self._validation_start = datetime.now()
        logger.info("Shadow validation started")

    def get_validation_result(
        self,
        strategy_id: str
    ) -> Optional[ValidationResult]:
        """Get validation result for a strategy."""
        if strategy_id not in self._shadow_trackers:
            return None

        tracker = self._shadow_trackers[strategy_id]
        return self._validation_gate.validate(
            tracker,
            self._live_tracker,
            self._validation_start
        )

    def compare_performance(
        self,
        strategy_id: Optional[str] = None
    ) -> Dict[str, PerformanceComparison]:
        """
        Compare shadow vs live performance.

        Args:
            strategy_id: Specific strategy or all if None

        Returns:
            Performance comparisons
        """
        results = {}

        strategies = (
            [strategy_id] if strategy_id
            else list(self._shadow_strategies.keys())
        )

        for sid in strategies:
            if sid not in self._shadow_trackers:
                continue

            shadow_tracker = self._shadow_trackers[sid]

            if self._live_tracker:
                comparison = self._validation_gate._compare_performance(
                    shadow_tracker,
                    self._live_tracker
                )
                results[sid] = comparison

        return results

    def get_shadow_positions(
        self,
        strategy_id: Optional[str] = None
    ) -> Dict[str, ShadowPosition]:
        """Get shadow positions."""
        positions = self._shadow_broker.get_all_positions()

        if strategy_id:
            positions = {
                k: v for k, v in positions.items()
                if v.strategy_id == strategy_id
            }

        return positions

    def get_shadow_trades(
        self,
        strategy_id: Optional[str] = None
    ) -> List[ShadowTrade]:
        """Get shadow trades."""
        return self._shadow_broker.get_trades(strategy_id=strategy_id)

    def get_shadow_pnl(
        self,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get shadow P&L.

        Returns:
            Dict of strategy_id -> (realized, unrealized)
        """
        results = {}

        for sid, tracker in self._shadow_trackers.items():
            if strategy_id and sid != strategy_id:
                continue

            metrics = tracker.get_metrics()
            results[sid] = (metrics.realized_pnl, metrics.unrealized_pnl)

        return results

    def promote_to_live(self, strategy_id: str) -> bool:
        """
        Promote a validated shadow strategy to live.

        Args:
            strategy_id: Strategy to promote

        Returns:
            True if promotion successful
        """
        validation = self.get_validation_result(strategy_id)

        if not validation or not validation.is_ready_for_production():
            logger.warning(f"Strategy {strategy_id} not ready for promotion")
            return False

        # In real implementation, this would swap the strategy
        logger.info(f"Strategy {strategy_id} promoted to live trading")

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'mode': self._mode.value,
            'validation_start': (
                self._validation_start.isoformat()
                if self._validation_start else None
            ),
            'shadow_strategies': list(self._shadow_strategies.keys()),
            'total_shadow_trades': len(self._shadow_broker.get_trades()),
            'shadow_pnl': self.get_shadow_pnl()
        }

    def reset(self) -> None:
        """Reset engine state."""
        self._shadow_broker.reset()

        for tracker in self._shadow_trackers.values():
            tracker.reset()

        if self._live_tracker:
            self._live_tracker.reset()

        self._validation_start = None
        self._mode = ShadowMode.DISABLED


# Convenience functions
_default_engine: Optional[ShadowEngine] = None


def get_shadow_engine() -> ShadowEngine:
    """Get default shadow engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = ShadowEngine()
    return _default_engine


def set_shadow_engine(engine: ShadowEngine) -> None:
    """Set default shadow engine."""
    global _default_engine
    _default_engine = engine


def register_shadow(strategy_id: str, strategy: Any) -> ShadowStrategy:
    """Register strategy for shadow trading."""
    return get_shadow_engine().register_shadow_strategy(strategy_id, strategy)


def shadow_on_tick(tick_data: Dict[str, Any]) -> None:
    """Process tick in shadow mode."""
    get_shadow_engine().on_tick(tick_data)


def validate_shadow(strategy_id: str) -> Optional[ValidationResult]:
    """Validate a shadow strategy."""
    return get_shadow_engine().get_validation_result(strategy_id)
