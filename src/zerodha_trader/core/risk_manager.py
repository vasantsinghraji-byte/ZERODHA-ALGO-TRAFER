# -*- coding: utf-8 -*-
"""
RiskManager - Portfolio risk monitoring and enforcement
Monitors positions, P&L, and enforces risk limits
"""
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .order_manager import Order, OrderStatus

logger = logging.getLogger(__name__)


class RiskLimits:
    """Risk limit configuration"""

    def __init__(self, account_balance: float):
        self.account_balance = account_balance

        # Per-trade limits
        self.max_risk_per_trade = 0.01  # 1% of account
        self.max_position_size = 0.10   # 10% of account per position

        # Portfolio limits
        self.max_daily_loss = 0.02      # 2% of account
        self.max_drawdown = 0.05        # 5% from peak
        self.max_open_positions = 5     # Maximum concurrent positions

        # Trade limits
        self.max_trades_per_day = 20
        self.min_risk_reward = 1.5      # Minimum R:R ratio


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

        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_account_value = risk_limits.account_balance
        self.current_account_value = risk_limits.account_balance

        # Drawdown tracking
        self.max_drawdown_hit = 0.0
        self.current_drawdown = 0.0

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

    def validate_trade(self, entry_price: float, stop_loss: float, target: float,
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
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, f"Circuit breaker active: {self.circuit_breaker_reason}"

        # Check max positions
        if current_positions >= self.limits.max_open_positions:
            return False, f"Max positions reached ({self.limits.max_open_positions})"

        # Check max trades per day
        if self.trades_today >= self.limits.max_trades_per_day:
            return False, f"Max daily trades reached ({self.limits.max_trades_per_day})"

        # Calculate risk per trade
        risk_per_share = abs(entry_price - stop_loss)
        total_risk = risk_per_share * quantity

        max_risk = self.limits.account_balance * self.limits.max_risk_per_trade

        if total_risk > max_risk:
            return False, f"Risk too high: ₹{total_risk:.2f} > ₹{max_risk:.2f}"

        # Calculate position size
        position_value = entry_price * quantity
        max_position = self.limits.account_balance * self.limits.max_position_size

        if position_value > max_position:
            return False, f"Position too large: ₹{position_value:.2f} > ₹{max_position:.2f}"

        # Check risk-reward ratio
        reward_per_share = abs(target - entry_price)
        if risk_per_share > 0:
            rr_ratio = reward_per_share / risk_per_share

            if rr_ratio < self.limits.min_risk_reward:
                return False, f"R:R too low: {rr_ratio:.2f} < {self.limits.min_risk_reward}"

        # Check daily loss limit
        max_daily_loss = self.limits.account_balance * self.limits.max_daily_loss

        if abs(self.daily_pnl) >= max_daily_loss and self.daily_pnl < 0:
            self._activate_circuit_breaker(f"Daily loss limit hit: ₹{abs(self.daily_pnl):.2f}")
            return False, "Daily loss limit exceeded"

        return True, "Trade validated"

    def update_pnl(self, realized_pnl: float = 0, unrealized_pnl: float = 0):
        """
        Update P&L values

        Args:
            realized_pnl: Realized P&L from closed trades
            unrealized_pnl: Unrealized P&L from open positions
        """
        self.realized_pnl += realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.daily_pnl = self.realized_pnl + self.unrealized_pnl

        # Update account value
        self.current_account_value = self.limits.account_balance + self.daily_pnl

        # Update peak
        if self.current_account_value > self.peak_account_value:
            self.peak_account_value = self.current_account_value

        # Calculate drawdown
        self.current_drawdown = (self.peak_account_value - self.current_account_value) / self.peak_account_value

        if self.current_drawdown > self.max_drawdown_hit:
            self.max_drawdown_hit = self.current_drawdown

        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            self._activate_circuit_breaker(
                f"Max drawdown hit: {self.current_drawdown*100:.1f}%"
            )

        # Check daily loss
        max_daily_loss = self.limits.account_balance * self.limits.max_daily_loss
        if abs(self.daily_pnl) >= max_daily_loss and self.daily_pnl < 0:
            self._activate_circuit_breaker(
                f"Daily loss limit: ₹{abs(self.daily_pnl):.2f}"
            )

        logger.info(f"P&L updated - Realized: ₹{self.realized_pnl:.2f}, "
                   f"Unrealized: ₹{self.unrealized_pnl:.2f}, "
                   f"Daily: ₹{self.daily_pnl:.2f}")

    def record_trade(self, pnl: float):
        """
        Record completed trade

        Args:
            pnl: Trade P&L
        """
        self.trades_today += 1

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

        logger.info(f"Trade recorded: ₹{pnl:.2f} (Total today: {self.trades_today})")

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

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate optimal position size based on risk limits

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Recommended quantity
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0

        # Max risk per trade
        max_risk = self.limits.account_balance * self.limits.max_risk_per_trade

        # Calculate quantity
        quantity = int(max_risk / risk_per_share)

        # Ensure doesn't exceed max position size
        max_position = self.limits.account_balance * self.limits.max_position_size
        max_qty_by_position = int(max_position / entry_price)

        quantity = min(quantity, max_qty_by_position)

        return quantity

    def get_risk_report(self) -> Dict:
        """
        Generate risk report

        Returns:
            Dictionary with risk metrics
        """
        win_rate = 0
        if self.trades_today > 0:
            win_rate = (self.winning_trades / self.trades_today) * 100

        return {
            'account_balance': self.limits.account_balance,
            'current_value': self.current_account_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'peak_value': self.peak_account_value,
            'current_drawdown': self.current_drawdown * 100,  # %
            'max_drawdown_hit': self.max_drawdown_hit * 100,  # %
            'trades_today': self.trades_today,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'risk_alerts': self.risk_alerts
        }

    def get_stats(self) -> Dict:
        """Get RiskManager statistics"""
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'trades_today': self.trades_today,
            'daily_pnl': self.daily_pnl,
            'current_drawdown_pct': self.current_drawdown * 100
        }

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)"""
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.session_start = datetime.now()

        # Don't reset peak value or max drawdown hit (lifetime metrics)

        logger.info("Daily risk statistics reset")
