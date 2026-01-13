from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import threading
import time

from .repository import MarketDataRepository
from .models import OHLCV, Tick
from infrastructure.cache import redis_manager

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    def __init__(self):
        self.repo = MarketDataRepository()
        
    def process_tick(self, tick_data: Dict[str, Any]) -> None:
        """Process incoming tick data"""
        try:
            tick = Tick(**tick_data)
            self.repo.save_ticks([tick])

            # Update real-time cache (only if symbol is available)
            if tick.symbol:
                redis_manager.update_ticker(
                    tick.symbol,
                    tick.last_price,
                    tick.volume
                )

        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            raise

    def process_historical(self, symbol: str, data: List[Dict[str, Any]]) -> None:
        """Process historical data"""
        try:
            ohlcv_data = [
                OHLCV(
                    timestamp=entry['date'],
                    symbol=symbol,
                    open=entry['open'],
                    high=entry['high'],
                    low=entry['low'],
                    close=entry['close'],
                    volume=entry['volume']
                )
                for entry in data
            ]
            
            for ohlcv in ohlcv_data:
                self.repo.save_ohlcv(ohlcv)
                
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            raise

class TickProcessor:
    """Process and store incoming tick data"""
    
    def __init__(self, batch_size: int = 100, batch_timeout: float = 5.0):
        self.repository = MarketDataRepository()
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batching
        self._tick_batch: List[Tick] = []
        self._batch_lock = threading.Lock()
        self._last_flush: Optional[datetime] = None
        
        # Start flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._auto_flush, daemon=True)
        self._flush_thread.start()
        
        # OHLCV aggregation
        self._ohlcv_cache: dict = {}

    def process_tick(self, tick: Tick) -> None:
        """Process a single tick"""
        try:
            # Validate tick
            if not self._validate_tick(tick):
                logger.debug(f"Invalid tick rejected: {tick.instrument_token}")
                return
            
            # Update cache
            self._update_cache(tick)
            
            # Add to batch
            with self._batch_lock:
                self._tick_batch.append(tick)
                
                # Flush if batch is full
                if len(self._tick_batch) >= self.batch_size:
                    self._flush_batch()
                    
            # Update OHLCV aggregation
            self._update_ohlcv(tick)
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}", exc_info=True)

    def _validate_tick(self, tick: Tick) -> bool:
        """Validate tick data"""
        try:
            # Check required fields
            if not tick.instrument_token:
                return False
            if not tick.timestamp:
                return False
            if tick.last_price <= 0:
                return False
                
            # Check timestamp is not too far in future
            if tick.timestamp > datetime.now() + timedelta(seconds=5):
                logger.warning(f"Tick timestamp in future: {tick.timestamp}")
                return False
                
            # Check timestamp is not too old
            if tick.timestamp < datetime.now() - timedelta(hours=1):
                logger.warning(f"Tick timestamp too old: {tick.timestamp}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Tick validation error: {e}")
            return False

    def _update_cache(self, tick: Tick) -> None:
        """Update Redis cache with latest tick data"""
        try:
            redis_manager.update_ticker(
                str(tick.instrument_token),
                tick.last_price,
                tick.volume
            )
        except Exception as e:
            logger.error(f"Cache update error: {e}")

    def _update_ohlcv(self, tick: Tick) -> None:
        """Update OHLCV aggregation"""
        try:
            key = f"{tick.instrument_token}_{tick.timestamp.strftime('%Y%m%d%H%M')}"
            
            if key not in self._ohlcv_cache:
                self._ohlcv_cache[key] = {
                    'instrument_token': tick.instrument_token,
                    'open': tick.last_price,
                    'high': tick.last_price,
                    'low': tick.last_price,
                    'close': tick.last_price,
                    'volume': tick.volume,
                    'timestamp': tick.timestamp
                }
            else:
                ohlcv = self._ohlcv_cache[key]
                ohlcv['high'] = max(ohlcv['high'], tick.last_price)
                ohlcv['low'] = min(ohlcv['low'], tick.last_price)
                ohlcv['close'] = tick.last_price
                ohlcv['volume'] += tick.volume
                
        except Exception as e:
            logger.error(f"OHLCV update error: {e}")

    def _flush_batch(self) -> None:
        """Flush tick batch to database"""
        if not self._tick_batch:
            return
            
        try:
            ticks_to_save = self._tick_batch.copy()
            self._tick_batch.clear()
            self._last_flush = datetime.now()
            
            # Save to database
            self.repository.save_ticks_batch(ticks_to_save)
            logger.info(f"Flushed {len(ticks_to_save)} ticks to database")
            
        except Exception as e:
            logger.error(f"Batch flush error: {e}", exc_info=True)

    def _auto_flush(self) -> None:
        """Auto-flush thread"""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.batch_timeout)
                
                with self._batch_lock:
                    if self._tick_batch:
                        self._flush_batch()
                        
            except Exception as e:
                logger.error(f"Auto-flush error: {e}")

    def flush_ohlcv(self) -> None:
        """Flush OHLCV cache to database"""
        try:
            for key, data in self._ohlcv_cache.items():
                ohlcv = OHLCV(
                    timestamp=data['timestamp'],
                    instrument_token=data['instrument_token'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume']
                )
                self.repository.save_ohlcv(ohlcv)
                
            self._ohlcv_cache.clear()
            logger.info("OHLCV data flushed")
            
        except Exception as e:
            logger.error(f"OHLCV flush error: {e}")

    def stop(self) -> None:
        """Stop the processor"""
        self._stop_event.set()
        with self._batch_lock:
            self._flush_batch()
        self.flush_ohlcv()

    def process_historical(self, symbol: str, historical_data: List[dict]) -> None:
        """Process historical OHLCV data"""
        try:
            if not historical_data:
                logger.warning(f"No historical data for {symbol}")
                return
                
            ohlcv_records = []
            for candle in historical_data:
                try:
                    ohlcv = OHLCV(
                        timestamp=candle['date'],
                        instrument_token=int(symbol) if symbol.isdigit() else 0,
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=int(candle['volume'])
                    )
                    ohlcv_records.append(ohlcv)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error parsing candle data: {e}")
                    continue
            
            # Batch save to database
            if ohlcv_records:
                self.repository.save_ohlcv_batch(ohlcv_records)
                logger.info(f"Saved {len(ohlcv_records)} historical candles for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing historical data: {e}", exc_info=True)
