from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

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
