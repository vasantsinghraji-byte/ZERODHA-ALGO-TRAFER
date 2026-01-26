# -*- coding: utf-8 -*-
"""
RiskManager - Portfolio risk monitoring and enforcement
Monitors positions, P&L, and enforces risk limits

Uses Decimal for all financial calculations to avoid floating-point
precision errors that can accumulate over thousands of trades.
"""
from typing import Dict, List, Optional, Union
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import logging
from .order_manager import Order, OrderStatus

logger = logging.getLogger(__name__)


def to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """
    Convert value to Decimal with proper precision.
    Uses string conversion to avoid float representation errors.

    Args:
        value: Number to convert

    Returns:
        Decimal representation
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class RiskLimits:
    """Risk limit configuration using Decimal for precision"""

    def __init__(self, account_balance: Union[float, Decimal]):
        self.account_balance = to_decimal(account_balance)

        # Per-trade limits (as Decimal for precise calculations)
        self.max_risk_per_trade = Decimal("0.01")   # 1% of account
        self.max_position_size = Decimal("0.10")    # 10% of account per position

        # Portfolio limits
        self.max_daily_loss = Decimal("0.02")       # 2% of account
        self.max_drawdown = Decimal("0.05")         # 5% from peak
        self.max_open_positions = 5                  # Maximum concurrent positions

        # Trade limits
        self.max_trades_per_day = 20
        self.min_risk_reward = Decimal("1.5")       # Minimum R:R ratio


class RiskManager:
    """
    Manages portfolio risk and enforces limits

    Responsibilities:
    - Monitor P&L (realized and unrealized)
    - Track drawdown from peak
    - Enforce position size limits
    - Enforce daily loss limits
    - Circuit breaker on excessive losses
    - Risk-reward validation
    - Position concentration limits
    """

    def __init__(self, risk_limits: RiskLimits):
        """
        Initialize RiskManager

        Args:
            risk_limits: Risk limit configuration
        """
        self.limits = risk_limits

        # P&L tracking (using Decimal for precision)
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.daily_pnl = Decimal("0")
        self.peak_account_value = risk_limits.account_balance  # Already Decimal
        self.current_account_value = risk_limits.account_balance

        # Drawdown tracking (Decimal for precise percentage calculations)
        self.max_drawdown_hit = Decimal("0")
        self.current_drawdown = Decimal("0")

        # Trade tracking
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None

        # Risk alerts
        self.risk_alerts: List[str] = []

        # Session start
        self.session_start = datetime.now()

    def validate_trade(self, entry_price: Union[float, Decimal],
                       stop_loss: Union[float, Decimal],
                       target: Union[float, Decimal],
                       quantity: int, current_positions: int) -> tuple[bool, str]:
        """
        Validate if trade meets risk criteria

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            target: Target price
            quantity: Position size
            current_positions: Number of current open positions

        Returns:
            (is_valid, reason)
        """
        # Convert inputs to Decimal for precise calculations
        entry = to_decimal(entry_price)
        sl = to_decimal(stop_loss)
        tgt = to_decimal(target)
        qty = Decimal(str(quantity))

        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, f"Circuit breaker active: {self.circuit_breaker_reason}"

        # Check max positions
        if current_positions >= self.limits.max_open_positions:
            return False, f"Max positions reached ({self.limits.max_open_positions})"

        # Check max trades per day
        if self.trades_today >= self.limits.max_trades_per_day:
            return False, f"Max daily trades reached ({self.limits.max_trades_per_day})"

        # Calculate risk per trade (using Decimal)
        risk_per_share = abs(entry - sl)
        total_risk = risk_per_share * qty

        max_risk = self.limits.account_balance * self.limits.max_risk_per_trade

        if total_risk > max_risk:
            return False, f"Risk too high: ₹{total_risk:.2f} > ₹{max_risk:.2f}"

        # Calculate position size (using Decimal)
        position_value = entry * qty
        max_position = self.limits.account_balance * self.limits.max_position_size

        if position_value > max_position:
            return False, f"Position too large: ₹{position_value:.2f} > ₹{max_position:.2f}"

        # Check risk-reward ratio (using Decimal)
        reward_per_share = abs(tgt - entry)
        if risk_per_share > 0:
            rr_ratio = reward_per_share / risk_per_share

            if rr_ratio < self.limits.min_risk_reward:
                return False, f"R:R too low: {rr_ratio:.2f} < {self.limits.min_risk_reward}"

        # Check daily loss limit
        max_daily_loss = self.limits.account_balance * self.limits.max_daily_loss

        if abs(self.daily_pnl) >= max_daily_loss and self.daily_pnl < Decimal("0"):
            self._activate_circuit_breaker(f"Daily loss limit hit: ₹{abs(self.daily_pnl):.2f}")
            return False, "Daily loss limit exceeded"

        return True, "Trade validated"

    def update_pnl(self, realized_pnl: Union[float, Decimal] = 0,
                   unrealized_pnl: Union[float, Decimal] = 0):
        """
        Update P&L values using Decimal for precision

        Args:
            realized_pnl: Realized P&L from closed trades
            unrealized_pnl: Unrealized P&L from open positions
        """
        # Convert to Decimal for precise accumulation
        self.realized_pnl += to_decimal(realized_pnl)
        self.unrealized_pnl = to_decimal(unrealized_pnl)
        self.daily_pnl = self.realized_pnl + self.unrealized_pnl

        # Update account value
        self.current_account_value = self.limits.account_balance + self.daily_pnl

        # Update peak
        if self.current_account_value > self.peak_account_value:
            self.peak_account_value = self.current_account_value

        # Calculate drawdown (Decimal division maintains precision)
        if self.peak_account_value > 0:
            self.current_drawdown = (self.peak_account_value - self.current_account_value) / self.peak_account_value
        else:
            self.current_drawdown = Decimal("0")

        if self.current_drawdown > self.max_drawdown_hit:
            self.max_drawdown_hit = self.current_drawdown

        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            self._activate_circuit_breaker(
                f"Max drawdown hit: {float(self.current_drawdown)*100:.1f}%"
            )

        # Check daily loss
        max_daily_loss = self.limits.account_balance * self.limits.max_daily_loss
        if abs(self.daily_pnl) >= max_daily_loss and self.daily_pnl < Decimal("0"):
            self._activate_circuit_breaker(
                f"Daily loss limit: ₹{abs(self.daily_pnl):.2f}"
            )

        logger.info(f"P&L updated - Realized: ₹{self.realized_pnl:.2f}, "
                   f"Unrealized: ₹{self.unrealized_pnl:.2f}, "
                   f"Daily: ₹{self.daily_pnl:.2f}")

    def record_trade(self, pnl: Union[float, Decimal]):
        """
        Record completed trade

        Args:
            pnl: Trade P&L
        """
        self.trades_today += 1

        # Convert to Decimal for comparison
        pnl_decimal = to_decimal(pnl)
        if pnl_decimal > Decimal("0"):
            self.winning_trades += 1
        elif pnl_decimal < Decimal("0"):
            self.losing_trades += 1

        logger.info(f"Trade recorded: ₹{pnl_decimal:.2f} (Total today: {self.trades_today})")

    def _activate_circuit_breaker(self, reason: str):
        """
        Activate circuit breaker to stop trading

        Args:
            reason: Reason for activation
        """
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = reason
            self.risk_alerts.append(f"[{datetime.now()}] CIRCUIT BREAKER: {reason}")

            logger.critical(f"🚨 CIRCUIT BREAKER ACTIVATED: {reason}")

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        logger.info("Circuit breaker reset")

    def calculate_position_size(self, entry_price: Union[float, Decimal],
                                stop_loss: Union[float, Decimal]) -> int:
        """
        Calculate optimal position size based on risk limits using Decimal

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Recommended quantity (0 if invalid inputs, capped at MAX_QUANTITY)
        """
        # Maximum quantity cap to prevent integer overflow
        MAX_QUANTITY = 100000
        # Minimum stop loss distance as percentage of entry price
        MIN_STOP_DISTANCE_PCT = Decimal("0.001")  # 0.1% minimum
        # Maximum position value limit
        MAX_POSITION_VALUE = Decimal("10000000")  # 1 crore max position

        # Input validation: prices must be positive
        if entry_price is None or stop_loss is None:
            logger.warning("Position size calculation failed: None price values")
            return 0

        # Convert to Decimal for precise calculation
        try:
            entry = to_decimal(entry_price)
            sl = to_decimal(stop_loss)
        except Exception as e:
            logger.warning(f"Position size calculation failed: invalid price format - {e}")
            return 0

        # Validate positive prices
        if entry <= Decimal("0"):
            logger.warning(f"Position size calculation failed: entry price must be positive, got {entry}")
            return 0

        if sl <= Decimal("0"):
            logger.warning(f"Position size calculation failed: stop loss must be positive, got {sl}")
            return 0

        risk_per_share = abs(entry - sl)

        # Enforce minimum stop loss distance to prevent division by tiny numbers
        min_stop_distance = entry * MIN_STOP_DISTANCE_PCT
        if risk_per_share < min_stop_distance:
            logger.warning(f"Position size calculation: stop loss too close to entry "
                          f"(distance: {risk_per_share}, minimum: {min_stop_distance})")
            return 0

        # Max risk per trade (Decimal arithmetic)
        max_risk = self.limits.account_balance * self.limits.max_risk_per_trade

        # Calculate quantity (convert to int after Decimal division)
        quantity = int(max_risk / risk_per_share)

        # Ensure doesn't exceed max position size by account percentage
        max_position = self.limits.account_balance * self.limits.max_position_size
        if entry > Decimal("0"):
            max_qty_by_position = int(max_position / entry)
            quantity = min(quantity, max_qty_by_position)

        # Enforce maximum position value limit
        if entry > Decimal("0"):
            max_qty_by_value = int(MAX_POSITION_VALUE / entry)
            quantity = min(quantity, max_qty_by_value)

        # Apply absolute quantity cap to prevent integer overflow
        quantity = min(quantity, MAX_QUANTITY)

        # Never return negative (defensive check)
        if quantity < 0:
            logger.error(f"Position size calculation produced negative value: {quantity}")
            return 0

        return quantity

    def get_risk_report(self) -> Dict:
        """
        Generate risk report

        Returns:
            Dictionary with risk metrics (floats for JSON serialization)
        """
        win_rate = 0.0
        if self.trades_today > 0:
            win_rate = (self.winning_trades / self.trades_today) * 100

        # Convert Decimals to float for JSON serialization
        return {
            'account_balance': float(self.limits.account_balance),
            'current_value': float(self.current_account_value),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'daily_pnl': float(self.daily_pnl),
            'peak_value': float(self.peak_account_value),
            'current_drawdown': float(self.current_drawdown * 100),  # %
            'max_drawdown_hit': float(self.max_drawdown_hit * 100),  # %
            'trades_today': self.trades_today,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'risk_alerts': self.risk_alerts
        }

    def get_stats(self) -> Dict:
        """Get RiskManager statistics (floats for JSON serialization)"""
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'trades_today': self.trades_today,
            'daily_pnl': float(self.daily_pnl),
            'current_drawdown_pct': float(self.current_drawdown * 100)
        }

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)"""
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.session_start = datetime.now()

        # Don't reset peak value or max drawdown hit (lifetime metrics)

        logger.info("Daily risk statistics reset")
