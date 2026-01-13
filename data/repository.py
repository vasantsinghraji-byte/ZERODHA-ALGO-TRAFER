from datetime import datetime
from typing import List, Optional, Dict

from infrastructure.db import execute_query, execute_batch_query
from infrastructure.cache import redis_manager, cache_market_data
from .models import OHLCV, Tick

class MarketDataRepository:
    @staticmethod
    def save_ticks(ticks: List[Tick]) -> None:
        query = """
        INSERT INTO market_data.ticks (
            instrument_token, timestamp, last_price, 
            volume, buy_quantity, sell_quantity
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = [
            (t.instrument_token, t.timestamp, t.last_price,
             t.volume, t.buy_quantity, t.sell_quantity)
            for t in ticks
        ]
        execute_batch_query(query, params)

    @staticmethod
    def save_ohlcv(data: OHLCV) -> None:
        query = """
        INSERT INTO market_data.ohlcv (
            timestamp, symbol, open, high, low, close, volume
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(query, (
            data.timestamp, data.symbol, data.open,
            data.high, data.low, data.close, data.volume
        ))

    @staticmethod
    @cache_market_data(ttl=300)
    def get_latest_price(symbol: str) -> Optional[float]:
        query = """
        SELECT close 
        FROM market_data.ohlcv 
        WHERE symbol = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        result = execute_query(query, (symbol,))
        return result[0]['close'] if result else None

    @staticmethod
    def update_realtime_data(symbol: str, price: float, volume: int) -> None:
        """Update both database and cache"""
        redis_manager.update_ticker(symbol, price, volume)
        # Then update database
        query = """
        INSERT INTO market_data.ticks (symbol, price, volume, timestamp)
        VALUES (%s, %s, %s, NOW())
        """
        execute_query(query, (symbol, price, volume))

    @classmethod
    def save_ticks_batch(cls, ticks: List[Tick]) -> None:
        """Save batch of ticks to database"""
        try:
            query = """
            INSERT INTO market_data.ticks (
                instrument_token, timestamp, last_price,
                volume, buy_quantity, sell_quantity,
                open, high, low, close
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (instrument_token, timestamp) DO UPDATE SET
                last_price = EXCLUDED.last_price,
                volume = EXCLUDED.volume,
                buy_quantity = EXCLUDED.buy_quantity,
                sell_quantity = EXCLUDED.sell_quantity
            """
            params = [
                (
                    t.instrument_token, t.timestamp, t.last_price,
                    t.volume, t.buy_quantity, t.sell_quantity,
                    t.open, t.high, t.low, t.close
                )
                for t in ticks
            ]
            execute_batch_query(query, params)
            
        except Exception as e:
            logger.error(f"Error saving tick batch: {e}")
            raise

    @classmethod
    def get_latest_ticks(cls, instrument_token: int, limit: int = 100) -> List[Dict]:
        """Get latest ticks for an instrument"""
        try:
            query = """
            SELECT * FROM market_data.ticks
            WHERE instrument_token = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            return execute_query(query, (instrument_token, limit))
            
        except Exception as e:
            logger.error(f"Error fetching ticks: {e}")
            return []

    @classmethod
    def save_ohlcv_batch(cls, ohlcv_list: List[OHLCV]) -> None:
        """Save batch of OHLCV records to database"""
        try:
            query = """
            INSERT INTO market_data.ohlcv (
                timestamp, instrument_token, open, high, low, close, volume
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (instrument_token, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """
            params = [
                (
                    ohlcv.timestamp,
                    ohlcv.instrument_token,
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume
                )
                for ohlcv in ohlcv_list
            ]
            execute_batch_query(query, params)
            
        except Exception as e:
            logger.error(f"Error saving OHLCV batch: {e}")
            raise

    @classmethod
    def get_ohlcv(cls, instrument_token: int, start_date, end_date, interval: str = '1minute') -> List[Dict]:
        """Get OHLCV data for a time range"""
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data.ohlcv
            WHERE instrument_token = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp ASC
            """
            return execute_query(query, (instrument_token, start_date, end_date))
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return []
