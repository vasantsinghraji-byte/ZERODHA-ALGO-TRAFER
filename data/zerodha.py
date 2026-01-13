from typing import List, Dict, Any, cast, Union
import logging
from datetime import datetime, timedelta

from kiteconnect import KiteConnect
from config.config import settings
from infrastructure.cache import cache_market_data

logger = logging.getLogger(__name__)

class ZerodhaClient:
    def __init__(self):
        if not settings.ZERODHA_API_KEY:
            raise ValueError("ZERODHA_API_KEY is not set in configuration")
        if not settings.ZERODHA_ACCESS_TOKEN:
            raise ValueError("ZERODHA_ACCESS_TOKEN is not set in configuration")
            
        self.kite = KiteConnect(api_key=settings.ZERODHA_API_KEY)
        self.kite.set_access_token(settings.ZERODHA_ACCESS_TOKEN)
        self._instruments_cache: Dict[str, int] = {}

    def get_instrument_token(self, symbol: str) -> int:
        """Get instrument token for a symbol"""
        if symbol in self._instruments_cache:
            return self._instruments_cache[symbol]

        try:
            # Kite quote returns instrument_token
            quote = self.kite.quote(symbol)
            if symbol in quote and isinstance(quote[symbol], dict):
                symbol_data = cast(Dict[str, Any], quote[symbol])
                instrument_token = int(symbol_data.get('instrument_token', 0))
                if instrument_token:
                    self._instruments_cache[symbol] = instrument_token
                    return instrument_token
            raise ValueError(f"Symbol {symbol} not found in quote response")
        except Exception as e:
            logger.error(f"Failed to get instrument token for {symbol}: {e}")
            raise

    def get_instrument_tokens(self, symbols: List[str]) -> List[int]:
        """Get instrument tokens for multiple symbols"""
        return [self.get_instrument_token(symbol) for symbol in symbols]

    def get_token_symbol_mapping(self, symbols: List[str]) -> Dict[int, str]:
        """Get a mapping of instrument_token -> symbol for multiple symbols"""
        mapping = {}
        for symbol in symbols:
            token = self.get_instrument_token(symbol)
            mapping[token] = symbol
        return mapping

    @cache_market_data(ttl=300)
    def get_quote(self, symbol: str) -> Any:
        """Get real-time quote for a symbol"""
        try:
            return self.kite.quote(symbol)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            raise

    def get_historical(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical data"""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            return self.kite.historical_data(
                symbol,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise
