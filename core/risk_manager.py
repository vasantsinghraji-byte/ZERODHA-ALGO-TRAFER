"""
Risk Manager Module

Handles position sizing, risk calculations, and risk limits.
Ensures trades comply with risk management rules.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date


@dataclass
class Position:
    """Position data class"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    target: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    entry_time: Optional[datetime] = None
    sl_type: str = "Fixed"

    def update_pnl(self, current_price: float):
        """Update PnL based on current price"""
        self.current_price = current_price
        self.pnl = (current_price - self.entry_price) * self.quantity
        if self.entry_price > 0:
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100


class RiskManager:
    """
    Manages risk and position sizing.

    Responsibilities:
    - Position size calculation
    - Risk limit enforcement
    - Daily loss tracking
    - Position tracking
    - Stop loss management
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade_pct: float = 1.0,