from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import logging

from data.models import OHLCV, Tick
from infrastructure.cache import redis_manager

logger = logging.getLogger(__name__)

class Strategy(ABC):
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.active_positions: Dict[str, dict] = {}
        
    @abstractmethod
    def generate_signal(self, symbol: str, data: OHLCV) -> Optional[dict]:
        """Generate trading signals based on market data"""
        pass
        
    def on_tick(self, tick: Tick) -> None:
        """Handle real-time tick data"""
        if tick.instrument_token not in self.symbols:
            return
            
        # Get cached market data
        cached_data = redis_manager.get_market_data(tick.instrument_token)
        if cached_data:
            signal = self.generate_signal(tick.instrument_token, cached_data)
            if signal:
                self.process_signal(signal)
                
    def process_signal(self, signal: dict) -> None:
        """Process generated trading signals"""
        try:
            # Save signal to database
            from data.repository import MarketDataRepository
            repo = MarketDataRepository()
            repo.save_signal(
                strategy_name=self.name,
                symbol=signal['symbol'],
                signal_type=signal['action'],
                price=signal['price'],
                quantity=signal['quantity']
            )
            
            # Emit signal event
            self.emit_signal_event(signal)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            
    def emit_signal_event(self, signal: dict) -> None:
        """Emit signal event for strategy manager"""
        redis_manager.set_market_data(
            f"signal:{self.name}:{signal['symbol']}",
            signal,
            ttl=300
        )
