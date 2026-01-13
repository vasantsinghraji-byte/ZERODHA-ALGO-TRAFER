import logging
from typing import Dict, Type, List
from concurrent.futures import ThreadPoolExecutor

from .base import Strategy
from data.collector import MarketDataCollector
from infrastructure.cache import redis_manager

logger = logging.getLogger(__name__)

class StrategyManager:
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.data_collector = MarketDataCollector()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def register_strategy(self, strategy: Strategy) -> None:
        """Register a new strategy"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
        
    async def start(self) -> None:
        """Start all registered strategies"""
        try:
            # Get unique symbols from all strategies
            symbols = set()
            for strategy in self.strategies.values():
                symbols.update(strategy.symbols)
                
            # Start data collection
            await self.data_collector.start(list(symbols))
            
            # Subscribe to market data
            self.data_collector.stream.callbacks['strategy'] = self._handle_tick
            logger.info("Strategy manager started")
            
        except Exception as e:
            logger.error(f"Error starting strategy manager: {e}")
            raise
            
    def _handle_tick(self, tick: dict) -> None:
        """Handle incoming tick data"""
        for strategy in self.strategies.values():
            self.executor.submit(strategy.on_tick, tick)
