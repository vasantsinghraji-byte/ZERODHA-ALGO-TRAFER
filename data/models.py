from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DepthItem(BaseModel):
    """Single level of order book depth."""
    price: float = 0.0
    quantity: int = 0
    orders: int = 0  # Number of orders at this level


class MarketDepth(BaseModel):
    """Level 2 order book depth (Top 5 bids/asks)."""
    buy: List[DepthItem] = Field(default_factory=list)
    sell: List[DepthItem] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to dict format for OrderFlowAnalyzer."""
        return {
            'buy': [{'price': d.price, 'quantity': d.quantity, 'orders': d.orders} for d in self.buy],
            'sell': [{'price': d.price, 'quantity': d.quantity, 'orders': d.orders} for d in self.sell]
        }


class Tick(BaseModel):
    instrument_token: int
    symbol: Optional[str] = None  # Trading symbol (e.g., "NSE:SBIN")
    timestamp: datetime
    last_price: float
    volume: int
    buy_quantity: int
    sell_quantity: int
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

    # Level 2 Market Depth (from MODE_QUOTE)
    depth: Optional[MarketDepth] = None

    # Additional quote fields
    average_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    last_traded_quantity: Optional[int] = None
    oi: Optional[int] = None  # Open Interest (for F&O)
    oi_day_high: Optional[int] = None
    oi_day_low: Optional[int] = None

class OHLCV(BaseModel):
    timestamp: datetime
    instrument_token: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
