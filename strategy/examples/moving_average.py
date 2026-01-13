from typing import Optional
import pandas as pd

from ..base import Strategy
from data.models import OHLCV

class MovingAverageStrategy(Strategy):
    def __init__(self, name: str, symbols: List[str], 
                 short_window: int = 10, long_window: int = 20):
        super().__init__(name, symbols)
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signal(self, symbol: str, data: OHLCV) -> Optional[dict]:
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame([data])
            
            # Calculate moving averages
            short_ma = df['close'].rolling(self.short_window).mean()
            long_ma = df['close'].rolling(self.long_window).mean()
            
            # Generate signals
            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': data.close,
                    'quantity': 1,
                    'timestamp': data.timestamp
                }
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': data.close,
                    'quantity': 1,
                    'timestamp': data.timestamp
                }
                
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            
        return None
