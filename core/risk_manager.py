# -*- coding: utf-8 -*-
"""
Risk Manager - Your Safety Net!
================================
Protects your capital from big losses.

Think of it like car safety features:
- Stop Loss = Airbag (protects from crashes)
- Position Sizing = Seatbelt (limits damage)
- Daily Limits = Speed Limits (prevents reckless driving)

NEVER TRADE WITHOUT RISK MANAGEMENT!
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import numpy as np

logger = logging.getLogger(__name__)


def _to_money(value: Union[float, int, Decimal, None]) -> Decimal:
    """
    Convert value to Decimal for precise monetary calculations.

    Prevents floating-point errors in risk calculations that could cause:
    - Incorrect position sizing leading to over/under exposure
    - Wrong stop loss prices rejected by brokers
    - Accumulated P&L tracking errors
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ============== ENUMS ==============

class StopLossType(Enum):
    """Types of stop loss"""
    FIXED = "FIXED"           # Fixed price level
    PERCENTAGE = "PERCENTAGE"  # % below entry
    ATR = "ATR"               # Based on volatility
    TRAILING = "TRAILING"     # Moves with price


class RiskLevel(Enum):
    """Risk appetite levels"""
    CONSERVATIVE = "CONSERVATIVE"  # 1% risk per trade
    MODERATE = "MODERATE"          # 2% risk per trade
    AGGRESSIVE = "AGGRESSIVE"      # 3% risk per trade


# ============== CONFIG ==============

@dataclass
class RiskConfig:
    """Risk management configuration"""

    # Position sizing
    max_risk_per_trade_pct: float = 2.0      # Max % of capital to risk per trade
    max_position_size_pct: float = 10.0     # Max % of capital in single position
    max_positions: int = 5                   # Max number of open positions

    # Daily limits
    max_daily_loss_pct: float = 5.0         # Stop trading if daily loss exceeds
    max_daily_trades: int = 20              # Max trades per day
    max_daily_profit_pct: float = 10.0      # Optional: stop if profit target hit

    # Stop loss defaults
    default_stop_loss_pct: float = 2.0      # Default stop loss %
    default_target_pct: float = 4.0         # Default target % (2:1 reward)
    use_trailing_stop: bool = False         # Enable trailing stops
    trailing_stop_pct: float = 1.5          # Trailing stop distance %

    # Risk limits
    max_drawdown_pct: float = 15.0          # Max allowed drawdown
    margin_safety_pct: float = 20.0         # Keep 20% as safety margin


# ============== RISK METRICS ==============

@dataclass
class RiskMetrics:
    """Calculated risk metrics"""
    # P&L
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Win/Loss
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0

    # Exposure
    total_exposure: float = 0.0
    exposure_pct: float = 0.0

    # Sharpe/Sortino
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0


# ============== RISK MANAGER ==============

class RiskManager:
    """
    The Guardian of Your Capital!

    Features:
    1. Position Sizing - How much to buy
    2. Stop Loss - When to cut losses
    3. Daily Limits - When to stop trading
    4. Risk Metrics - Track performance

    Example:
        rm = RiskManager(capital=100000)
        quantity = rm.calculate_position_size("RELIANCE", 2500, 2450)
        print(f"Buy {quantity} shares")
    """

    def __init__(
        self,
        capital: Union[float, Decimal] = 100000,
        config: RiskConfig = None
    ):
        """
        Initialize Risk Manager with precise Decimal capital tracking.

        Args:
            capital: Starting capital (must be > 0)
            config: Risk configuration
        """
        # Validate capital to prevent division by zero errors
        if capital is None or capital <= 0:
            logger.warning(f"Invalid capital ({capital}), defaulting to Rs.100,000")
            capital = 100000

        # Use Decimal for precise capital tracking
        capital_decimal = _to_money(capital)
        self.initial_capital: Decimal = capital_decimal
        self.current_capital: Decimal = capital_decimal
        self.config = config or RiskConfig()

        # Daily tracking with Decimal precision
        self._daily_start_capital: Decimal = capital_decimal
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_trades = 0
        self._daily_date = date.today()

        # Trade history
        self._trades_history: List[Dict] = []
        self._equity_curve: List[Decimal] = [capital_decimal]
        self._peak_capital: Decimal = capital_decimal

        # Alerts
        self._alerts: List[str] = []

        logger.info(f"RiskManager initialized with Rs.{capital_decimal:,.2f}")

    # ============== POSITION SIZING ==============

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: float = None
    ) -> int:
        """
        Calculate how many shares to buy based on risk using precise Decimal arithmetic.

        The GOLDEN FORMULA:
        Position Size = (Capital × Risk%) / (Entry - Stop Loss)

        Uses Decimal to prevent floating-point errors that could cause:
        - Over-exposure due to incorrect position sizing
        - Under-exposure missing profit opportunities

        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_pct: % of capital to risk (default from config)

        Returns:
            Number of shares to buy (0 if inputs are invalid)

        Example:
            >>> rm.calculate_position_size("RELIANCE", 2500, 2450)
            40  # Buy 40 shares
        """
        # Validate entry price to prevent ZeroDivisionError
        if entry_price <= 0:
            logger.error(f"Invalid entry price {entry_price} for {symbol} - cannot calculate position size")
            return 0

        risk_pct = risk_pct or self.config.max_risk_per_trade_pct

        # Use Decimal for precise calculations
        entry = _to_money(entry_price)
        stop_loss = _to_money(stop_loss_price)

        # Calculate risk per share
        risk_per_share = abs(entry - stop_loss)

        if risk_per_share <= 0:
            logger.warning(f"Invalid stop loss for {symbol} - using default 2%")
            risk_per_share = entry * Decimal("0.02")

        # Calculate risk amount using Decimal
        risk_amount = self.current_capital * Decimal(str(risk_pct)) / Decimal("100")

        # Calculate quantity (round down to avoid over-exposure)
        quantity = int((risk_amount / risk_per_share).to_integral_value(rounding=ROUND_DOWN))

        # Check max position size limit
        max_position_pct = Decimal(str(self.config.max_position_size_pct))
        max_position_value = self.current_capital * max_position_pct / Decimal("100")
        max_quantity = int((max_position_value / entry).to_integral_value(rounding=ROUND_DOWN))

        quantity = min(quantity, max_quantity)

        logger.info(f"Position size for {symbol}: {quantity} shares "
                   f"(Risk: Rs.{risk_amount:.2f}, Per share: Rs.{risk_per_share:.2f})")

        return max(1, quantity)

    def calculate_stop_loss(
        self,
        entry_price: float,
        stop_type: StopLossType = StopLossType.PERCENTAGE,
        atr: float = None
    ) -> float:
        """
        Calculate stop loss price using precise Decimal arithmetic.

        Prevents floating-point errors that could cause broker rejection
        due to invalid tick sizes (e.g., 2450.00000001 instead of 2450.00).

        Args:
            entry_price: Entry price
            stop_type: Type of stop loss
            atr: ATR value (for ATR-based stops)

        Returns:
            Stop loss price (precise to 2 decimal places)
        """
        entry = _to_money(entry_price)
        sl_pct = Decimal(str(self.config.default_stop_loss_pct)) / Decimal("100")

        if stop_type == StopLossType.PERCENTAGE:
            result = entry * (Decimal("1") - sl_pct)

        elif stop_type == StopLossType.ATR:
            if atr:
                atr_decimal = _to_money(atr)
                result = entry - (Decimal("2") * atr_decimal)
            else:
                result = entry * Decimal("0.98")

        elif stop_type == StopLossType.FIXED:
            result = entry - (entry * sl_pct)

        else:  # TRAILING - return percentage-based as default
            result = entry * (Decimal("1") - sl_pct)

        return float(_to_money(result))

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate target price based on risk-reward ratio using precise Decimal arithmetic.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            reward_ratio: Risk:Reward ratio (default 1:2)

        Returns:
            Target price (precise to 2 decimal places)
        """
        entry = _to_money(entry_price)
        sl = _to_money(stop_loss)
        ratio = Decimal(str(reward_ratio))

        risk = abs(entry - sl)
        reward = risk * ratio
        result = entry + reward

        return float(_to_money(result))

    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        entry_price: float
    ) -> float:
        """
        Calculate trailing stop loss using precise Decimal arithmetic.

        Args:
            current_price: Current market price
            highest_price: Highest price since entry
            entry_price: Original entry price

        Returns:
            New stop loss price (precise to 2 decimal places)
        """
        highest = _to_money(highest_price)
        entry = _to_money(entry_price)
        trail_pct = Decimal(str(self.config.trailing_stop_pct)) / Decimal("100")
        sl_pct = Decimal(str(self.config.default_stop_loss_pct)) / Decimal("100")

        # Trailing stop is X% below highest price
        trailing_stop = highest * (Decimal("1") - trail_pct)

        # Never lower than original stop
        original_stop = entry * (Decimal("1") - sl_pct)

        return float(_to_money(max(trailing_stop, original_stop)))

    # ============== RISK CHECKS ==============

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            (allowed, reason)
        """
        # Reset daily tracking if new day
        self._check_new_day()

        # Check daily loss limit (guard against zero capital)
        if self._daily_pnl < 0 and self._daily_start_capital > 0:
            loss_pct = abs(self._daily_pnl) / self._daily_start_capital * 100
            if loss_pct >= self.config.max_daily_loss_pct:
                msg = f"Daily loss limit reached ({loss_pct:.1f}%)"
                logger.warning(msg)
                return False, msg

        # Check daily trade limit
        if self._daily_trades >= self.config.max_daily_trades:
            msg = f"Daily trade limit reached ({self._daily_trades})"
            logger.warning(msg)
            return False, msg

        # Check drawdown limit
        if self._check_drawdown():
            msg = f"Max drawdown limit reached ({self.config.max_drawdown_pct}%)"
            logger.warning(msg)
            return False, msg

        # Check daily profit target (optional, guard against zero capital)
        if self._daily_pnl > 0 and self._daily_start_capital > 0:
            profit_pct = self._daily_pnl / self._daily_start_capital * 100
            if profit_pct >= self.config.max_daily_profit_pct:
                msg = f"Daily profit target reached ({profit_pct:.1f}%)"
                logger.info(msg)
                return False, msg

        return True, "OK"

    def can_open_position(
        self,
        current_positions: int,
        position_value: float
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened.

        Args:
            current_positions: Number of open positions
            position_value: Value of new position

        Returns:
            (allowed, reason)
        """
        # Check max positions
        if current_positions >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        # Check position size
        max_value = self.current_capital * self.config.max_position_size_pct / 100
        if position_value > max_value:
            return False, f"Position too large (max: Rs.{max_value:,.0f})"

        # Check margin safety
        min_capital = self.initial_capital * self.config.margin_safety_pct / 100
        if self.current_capital < min_capital:
            return False, "Capital below safety margin"

        return True, "OK"

    # ============== DAILY TRACKING ==============

    def _check_new_day(self):
        """Reset daily counters if new day"""
        today = date.today()
        if today != self._daily_date:
            self._daily_date = today
            self._daily_start_capital = self.current_capital
            self._daily_pnl = 0.0
            self._daily_trades = 0
            logger.info(f"New trading day: {today}")

    def _check_drawdown(self) -> bool:
        """Check if max drawdown exceeded"""
        if self.current_capital < self._peak_capital:
            drawdown = self._peak_capital - self.current_capital
            drawdown_pct = drawdown / self._peak_capital * 100
            return drawdown_pct >= self.config.max_drawdown_pct
        return False

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float
    ):
        """
        Record a completed trade.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Shares traded
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/Loss
        """
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0
        }

        self._trades_history.append(trade)
        self._daily_trades += 1
        self._daily_pnl += pnl
        self.current_capital += pnl

        # Update peak
        if self.current_capital > self._peak_capital:
            self._peak_capital = self.current_capital

        # Update equity curve
        self._equity_curve.append(self.current_capital)

        logger.info(f"Trade recorded: {symbol} P&L: Rs.{pnl:+.0f}")

    def update_capital(self, new_capital: float):
        """Update current capital"""
        self.current_capital = new_capital
        if new_capital > self._peak_capital:
            self._peak_capital = new_capital

    # ============== RISK METRICS ==============

    def get_metrics(self) -> RiskMetrics:
        """
        Calculate and return all risk metrics.

        Returns:
            RiskMetrics object
        """
        metrics = RiskMetrics()

        # P&L
        metrics.total_pnl = self.current_capital - self.initial_capital
        metrics.daily_pnl = self._daily_pnl

        # Drawdown
        if self.current_capital < self._peak_capital:
            metrics.current_drawdown = self._peak_capital - self.current_capital
            metrics.current_drawdown_pct = metrics.current_drawdown / self._peak_capital * 100

        # Calculate max drawdown from equity curve
        if len(self._equity_curve) > 1:
            equity = np.array(self._equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = peak - equity
            metrics.max_drawdown = drawdown.max()
            metrics.max_drawdown_pct = (drawdown / peak).max() * 100

        # Trade stats
        if self._trades_history:
            metrics.total_trades = len(self._trades_history)

            winners = [t for t in self._trades_history if t['pnl'] > 0]
            losers = [t for t in self._trades_history if t['pnl'] <= 0]

            metrics.winning_trades = len(winners)
            metrics.losing_trades = len(losers)

            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100

            if winners:
                metrics.avg_win = sum(t['pnl'] for t in winners) / len(winners)

            if losers:
                metrics.avg_loss = abs(sum(t['pnl'] for t in losers) / len(losers))

            # Profit factor
            total_profit = sum(t['pnl'] for t in winners) if winners else 0
            total_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0

            if total_loss > 0:
                metrics.profit_factor = total_profit / total_loss

            # Risk/Reward
            if metrics.avg_loss > 0:
                metrics.risk_reward_ratio = metrics.avg_win / metrics.avg_loss

        # Exposure
        metrics.exposure_pct = (self.initial_capital - self.current_capital) / self.initial_capital * 100

        # Sharpe (simplified)
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / self._equity_curve[:-1]
            if returns.std() > 0:
                metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        return metrics

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get today's trading stats"""
        return {
            'date': self._daily_date,
            'start_capital': self._daily_start_capital,
            'current_capital': self.current_capital,
            'daily_pnl': self._daily_pnl,
            'daily_pnl_pct': (self._daily_pnl / self._daily_start_capital * 100) if self._daily_start_capital > 0 else 0,
            'trades_today': self._daily_trades,
            'trades_remaining': self.config.max_daily_trades - self._daily_trades,
            'loss_limit_remaining': self.config.max_daily_loss_pct - (abs(self._daily_pnl) / self._daily_start_capital * 100) if (self._daily_pnl < 0 and self._daily_start_capital > 0) else self.config.max_daily_loss_pct
        }

    def print_risk_report(self):
        """Print detailed risk report"""
        metrics = self.get_metrics()
        daily = self.get_daily_stats()

        print("\n" + "=" * 60)
        print("RISK MANAGEMENT REPORT")
        print("=" * 60)

        print(f"\n{'--- CAPITAL ---':^60}")
        print(f"Initial Capital:    Rs.{self.initial_capital:>12,.0f}")
        print(f"Current Capital:    Rs.{self.current_capital:>12,.0f}")
        print(f"Total P&L:          Rs.{metrics.total_pnl:>+12,.0f}")
        print(f"Peak Capital:       Rs.{self._peak_capital:>12,.0f}")

        print(f"\n{'--- TODAY ---':^60}")
        print(f"Daily P&L:          Rs.{daily['daily_pnl']:>+12,.0f} ({daily['daily_pnl_pct']:+.1f}%)")
        print(f"Trades Today:       {daily['trades_today']:>12}")
        print(f"Trades Remaining:   {daily['trades_remaining']:>12}")

        print(f"\n{'--- RISK METRICS ---':^60}")
        print(f"Current Drawdown:   Rs.{metrics.current_drawdown:>12,.0f} ({metrics.current_drawdown_pct:.1f}%)")
        print(f"Max Drawdown:       Rs.{metrics.max_drawdown:>12,.0f} ({metrics.max_drawdown_pct:.1f}%)")
        print(f"Win Rate:           {metrics.win_rate:>12.1f}%")
        print(f"Profit Factor:      {metrics.profit_factor:>12.2f}")
        print(f"Risk/Reward:        {metrics.risk_reward_ratio:>12.2f}")
        print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>12.2f}")

        print(f"\n{'--- LIMITS ---':^60}")
        print(f"Max Risk/Trade:     {self.config.max_risk_per_trade_pct:>12.1f}%")
        print(f"Max Position Size:  {self.config.max_position_size_pct:>12.1f}%")
        print(f"Max Positions:      {self.config.max_positions:>12}")
        print(f"Max Daily Loss:     {self.config.max_daily_loss_pct:>12.1f}%")

        # Trading status
        can_trade, reason = self.can_trade()
        status = "ALLOWED" if can_trade else f"BLOCKED: {reason}"
        print(f"\n{'--- STATUS ---':^60}")
        print(f"Trading Status:     {status}")

        print("=" * 60)

    # ============== ALERTS ==============

    def check_alerts(self) -> List[str]:
        """Check and return any risk alerts"""
        alerts = []

        metrics = self.get_metrics()

        # Check drawdown
        if metrics.current_drawdown_pct > self.config.max_drawdown_pct * 0.8:
            alerts.append(f"WARNING: Approaching max drawdown ({metrics.current_drawdown_pct:.1f}%)")

        # Check daily loss (guard against zero capital)
        if self._daily_pnl < 0 and self._daily_start_capital > 0:
            loss_pct = abs(self._daily_pnl) / self._daily_start_capital * 100
            if loss_pct > self.config.max_daily_loss_pct * 0.8:
                alerts.append(f"WARNING: Approaching daily loss limit ({loss_pct:.1f}%)")

        # Check capital erosion (guard against zero initial capital)
        if self.initial_capital > 0:
            capital_lost = (self.initial_capital - self.current_capital) / self.initial_capital * 100
            if capital_lost > 10:
                alerts.append(f"ALERT: Capital down {capital_lost:.1f}% from initial")

        return alerts


# ============== QUICK HELPERS ==============

def calculate_risk_reward(entry: float, stop_loss: float, target: float) -> float:
    """
    Calculate risk-reward ratio.

    Args:
        entry: Entry price
        stop_loss: Stop loss price
        target: Target price

    Returns:
        Risk:Reward ratio
    """
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    return reward / risk if risk > 0 else 0


def calculate_position_size_fixed(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float
) -> int:
    """
    Quick position size calculation.

    Args:
        capital: Available capital
        risk_pct: % to risk
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Quantity to buy (0 if inputs are invalid)
    """
    # Validate inputs to prevent ZeroDivisionError
    if entry_price <= 0:
        logger.error(f"Invalid entry price {entry_price} - cannot calculate position size")
        return 0

    risk_amount = capital * risk_pct / 100
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share <= 0:
        return 0

    return int(risk_amount / risk_per_share)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("RISK MANAGER - Test")
    print("=" * 50)

    # Create risk manager
    rm = RiskManager(capital=100000)

    # Calculate position size
    qty = rm.calculate_position_size("RELIANCE", 2500, 2450)
    print(f"\nPosition size for RELIANCE: {qty} shares")

    # Calculate stop loss and target
    sl = rm.calculate_stop_loss(2500, StopLossType.PERCENTAGE)
    target = rm.calculate_target(2500, sl)
    print(f"Stop Loss: Rs.{sl:.2f}")
    print(f"Target: Rs.{target:.2f}")
    print(f"Risk:Reward = 1:{calculate_risk_reward(2500, sl, target):.1f}")

    # Check if can trade
    can_trade, reason = rm.can_trade()
    print(f"\nCan trade: {can_trade} ({reason})")

    # Record some trades
    rm.record_trade("RELIANCE", "BUY", 40, 2500, 2600, 4000)
    rm.record_trade("TCS", "BUY", 10, 3500, 3400, -1000)
    rm.record_trade("INFY", "BUY", 50, 1500, 1550, 2500)

    # Print report
    rm.print_risk_report()

    print("\n" + "=" * 50)
    print("Risk Manager ready!")
    print("=" * 50)
