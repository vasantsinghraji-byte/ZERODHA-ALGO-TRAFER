import logging
from typing import List, Set
import asyncio

from config.config import settings
from .zerodha import ZerodhaClient
from .processor import TickProcessor
from .websocket import MarketDataStream

logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self):
        # Validate credentials
        if not settings.ZERODHA_API_KEY:
            raise ValueError("ZERODHA_API_KEY is not set in configuration")
        if not settings.ZERODHA_ACCESS_TOKEN:
            raise ValueError("ZERODHA_ACCESS_TOKEN is not set in configuration")
            
        self.client = ZerodhaClient()
        self.processor = TickProcessor()
        self.stream = MarketDataStream(
            api_key=settings.ZERODHA_API_KEY,
            access_token=settings.ZERODHA_ACCESS_TOKEN,
            processor=self.processor
        )
        self.active_symbols: Set[str] = set()

    async def start(self, symbols: List[str]) -> None:
        """Start collecting market data"""
        try:
            loop = asyncio.get_running_loop()

            # Initialize historical data
            # Offload blocking I/O to thread pool to avoid freezing the event loop
            for symbol in symbols:
                if symbol not in self.active_symbols:
                    token = await loop.run_in_executor(
                        None, self.client.get_instrument_token, symbol
                    )
                    historical = await loop.run_in_executor(
                        None, self.client.get_historical, symbol
                    )
                    await loop.run_in_executor(
                        None, self.processor.process_historical, symbol, historical, token
                    )
                    self.active_symbols.add(symbol)

                    # Respect Zerodha rate limit (~3 requests/second)
                    await asyncio.sleep(0.5)

            # Start real-time stream
            self.stream.subscribe([int(token) for token in symbols])
            self.stream.start()

        except Exception as e:
            logger.error(f"Error in market data collection: {e}")
            raise

    async def stop(self) -> None:
        """Stop data collection"""
        self.stream.stop()
        self.processor.stop()
        self.active_symbols.clear()
