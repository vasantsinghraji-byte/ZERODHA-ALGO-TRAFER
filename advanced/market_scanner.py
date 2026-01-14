# -*- coding: utf-8 -*-
"""
Market Scanner - Find Trading Opportunities!
=============================================
Scans the market to find stocks that match your criteria.

Like a radar that finds the best stocks for you!

Features:
- Momentum scanner (trending stocks)
- Breakout scanner (stocks about to move)
- Oversold/Overbought scanner
- Volume scanner (unusual activity)
- Custom filter scanner
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of market scans"""
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    VOLUME_SPIKE = "volume_spike"
    TREND_REVERSAL = "trend_reversal"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_BREAK = "resistance_break"
    CUSTOM = "custom"


@dataclass
class ScanResult:
    """Result from scanning a single stock"""
    symbol: str
    scan_type: ScanType
    score: float                    # 0 to 100
    signal: str                     # BUY, SELL, WATCH
    current_price: float
    change_percent: float
    volume_ratio: float             # Volume vs average
    rsi: float
    trend: str                      # UP, DOWN, SIDEWAYS
    reason: str                     # Why this stock was flagged
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def emoji(self) -> str:
        """Get emoji based on signal"""
        if self.signal == "BUY":
            return "🟢"
        elif self.signal == "SELL":
            return "🔴"
        return "🟡"

    @property
    def strength(self) -> str:
        """Get strength description"""
        if self.score >= 80:
            return "STRONG"
        elif self.score >= 60:
            return "MODERATE"
        return "WEAK"


@dataclass
class ScanFilter:
    """Filter criteria for scanning"""
    min_price: float = 0
    max_price: float = float('inf')
    min_volume: int = 0
    min_change_percent: float = -100
    max_change_percent: float = 100
    min_rsi: float = 0
    max_rsi: float = 100
    min_score: float = 50
    sectors: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)


class StockAnalyzer:
    """
    Analyze a single stock for various signals.
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all indicators for a stock"""
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        if len(data) < 20:
            return {}

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', pd.Series([0] * len(data)))

        indicators = {}

        # Current values
        indicators['current_price'] = close.iloc[-1]
        indicators['prev_close'] = close.iloc[-2]
        indicators['change'] = close.iloc[-1] - close.iloc[-2]
        indicators['change_percent'] = (indicators['change'] / indicators['prev_close']) * 100

        # Moving Averages
        indicators['sma_5'] = close.rolling(5).mean().iloc[-1]
        indicators['sma_10'] = close.rolling(10).mean().iloc[-1]
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(data) >= 50 else close.mean()

        # EMA
        indicators['ema_9'] = close.ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = close.ewm(span=21).mean().iloc[-1]

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']

        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        indicators['bb_upper'] = (bb_sma + 2 * bb_std).iloc[-1]
        indicators['bb_lower'] = (bb_sma - 2 * bb_std).iloc[-1]
        indicators['bb_middle'] = bb_sma.iloc[-1]
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / bb_range if bb_range > 0 else 0.5

        # Volume
        indicators['volume'] = volume.iloc[-1]
        indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1

        # ATR
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean().iloc[-1]
        indicators['atr_percent'] = (indicators['atr'] / close.iloc[-1]) * 100

        # Trend
        indicators['above_sma20'] = close.iloc[-1] > indicators['sma_20']
        indicators['above_sma50'] = close.iloc[-1] > indicators['sma_50']
        indicators['sma_20_slope'] = indicators['sma_20'] - close.rolling(20).mean().iloc[-5]

        # Highs and Lows
        indicators['high_20d'] = high.tail(20).max()
        indicators['low_20d'] = low.tail(20).min()
        indicators['high_52w'] = high.tail(252).max() if len(data) >= 252 else high.max()
        indicators['low_52w'] = low.tail(252).min() if len(data) >= 252 else low.min()

        # Distance from highs/lows
        indicators['pct_from_high'] = ((close.iloc[-1] / indicators['high_20d']) - 1) * 100
        indicators['pct_from_low'] = ((close.iloc[-1] / indicators['low_20d']) - 1) * 100

        return indicators

    @staticmethod
    def get_trend(indicators: Dict[str, float]) -> str:
        """Determine trend from indicators"""
        bullish_signals = 0

        if indicators.get('above_sma20', False):
            bullish_signals += 1
        if indicators.get('above_sma50', False):
            bullish_signals += 1
        if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            bullish_signals += 1
        if indicators.get('sma_20_slope', 0) > 0:
            bullish_signals += 1

        if bullish_signals >= 3:
            return "UP"
        elif bullish_signals <= 1:
            return "DOWN"
        return "SIDEWAYS"


class MomentumScanner:
    """Scan for stocks with strong momentum"""

    @staticmethod
    def scan(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Scan for momentum.

        Looks for:
        - Strong upward price movement
        - RSI in bullish zone (50-70)
        - Above key moving averages
        - Increasing volume
        """
        indicators = StockAnalyzer.calculate_indicators(df)
        if not indicators:
            return None

        score = 0
        reasons = []

        # Price momentum
        change_pct = indicators.get('change_percent', 0)
        if change_pct > 2:
            score += 25
            reasons.append(f"Strong gain +{change_pct:.1f}%")
        elif change_pct > 0.5:
            score += 15
            reasons.append(f"Positive +{change_pct:.1f}%")

        # RSI momentum
        rsi = indicators.get('rsi', 50)
        if 50 < rsi < 70:
            score += 25
            reasons.append(f"Bullish RSI {rsi:.0f}")
        elif 40 < rsi <= 50:
            score += 10

        # Above moving averages
        if indicators.get('above_sma20', False):
            score += 15
            reasons.append("Above SMA20")
        if indicators.get('above_sma50', False):
            score += 10
            reasons.append("Above SMA50")

        # MACD bullish
        if indicators.get('macd_histogram', 0) > 0:
            score += 15
            reasons.append("MACD bullish")

        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 10
            reasons.append(f"High volume {volume_ratio:.1f}x")

        if score < 40:
            return None

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.MOMENTUM,
            score=min(score, 100),
            signal="BUY" if score >= 60 else "WATCH",
            current_price=indicators.get('current_price', 0),
            change_percent=change_pct,
            volume_ratio=volume_ratio,
            rsi=rsi,
            trend=StockAnalyzer.get_trend(indicators),
            reason=" | ".join(reasons)
        )


class BreakoutScanner:
    """Scan for stocks breaking out"""

    @staticmethod
    def scan(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Scan for breakouts.

        Looks for:
        - Price breaking above 20-day high
        - Strong volume confirmation
        - Momentum indicators aligned
        """
        indicators = StockAnalyzer.calculate_indicators(df)
        if not indicators:
            return None

        score = 0
        reasons = []

        current_price = indicators.get('current_price', 0)
        high_20d = indicators.get('high_20d', current_price)
        pct_from_high = indicators.get('pct_from_high', -10)

        # Breakout detection
        if pct_from_high >= -1:  # Within 1% of 20-day high
            score += 30
            reasons.append("Near 20-day high")

            if pct_from_high >= 0:  # New high
                score += 20
                reasons.append("NEW HIGH!")

        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            score += 25
            reasons.append(f"Breakout volume {volume_ratio:.1f}x")
        elif volume_ratio > 1.5:
            score += 15
            reasons.append(f"High volume {volume_ratio:.1f}x")

        # RSI not overbought
        rsi = indicators.get('rsi', 50)
        if 50 < rsi < 75:
            score += 15
            reasons.append(f"RSI {rsi:.0f} has room")

        # MACD confirmation
        if indicators.get('macd_histogram', 0) > 0:
            score += 10
            reasons.append("MACD bullish")

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.BREAKOUT,
            score=min(score, 100),
            signal="BUY" if score >= 70 else "WATCH",
            current_price=current_price,
            change_percent=indicators.get('change_percent', 0),
            volume_ratio=volume_ratio,
            rsi=rsi,
            trend=StockAnalyzer.get_trend(indicators),
            reason=" | ".join(reasons)
        )


class OversoldScanner:
    """Scan for oversold stocks (potential bounce)"""

    @staticmethod
    def scan(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Scan for oversold conditions.

        Looks for:
        - RSI below 30
        - Price near Bollinger Band lower
        - Potential reversal signs
        """
        indicators = StockAnalyzer.calculate_indicators(df)
        if not indicators:
            return None

        score = 0
        reasons = []

        rsi = indicators.get('rsi', 50)
        bb_position = indicators.get('bb_position', 0.5)

        # RSI oversold
        if rsi < 30:
            score += 35
            reasons.append(f"RSI oversold {rsi:.0f}")
        elif rsi < 40:
            score += 20
            reasons.append(f"RSI low {rsi:.0f}")

        # Bollinger Band position
        if bb_position < 0.1:
            score += 30
            reasons.append("At lower Bollinger Band")
        elif bb_position < 0.2:
            score += 20
            reasons.append("Near lower Bollinger Band")

        # Price near support (20-day low)
        pct_from_low = indicators.get('pct_from_low', 10)
        if pct_from_low < 2:
            score += 20
            reasons.append("Near 20-day low")

        # Volume spike (potential capitulation)
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            score += 15
            reasons.append(f"High volume {volume_ratio:.1f}x")

        if score < 50:
            return None

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.OVERSOLD,
            score=min(score, 100),
            signal="BUY" if score >= 70 else "WATCH",
            current_price=indicators.get('current_price', 0),
            change_percent=indicators.get('change_percent', 0),
            volume_ratio=volume_ratio,
            rsi=rsi,
            trend=StockAnalyzer.get_trend(indicators),
            reason=" | ".join(reasons)
        )


class VolumeSpikeScanner:
    """Scan for unusual volume activity"""

    @staticmethod
    def scan(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """
        Scan for volume spikes.

        Looks for:
        - Volume significantly above average
        - Price movement confirmation
        """
        indicators = StockAnalyzer.calculate_indicators(df)
        if not indicators:
            return None

        volume_ratio = indicators.get('volume_ratio', 1)

        if volume_ratio < 2:
            return None

        score = 0
        reasons = []

        # Volume spike magnitude
        if volume_ratio > 5:
            score += 40
            reasons.append(f"Massive volume {volume_ratio:.1f}x")
        elif volume_ratio > 3:
            score += 30
            reasons.append(f"Very high volume {volume_ratio:.1f}x")
        else:
            score += 20
            reasons.append(f"High volume {volume_ratio:.1f}x")

        # Price movement direction
        change_pct = indicators.get('change_percent', 0)
        if abs(change_pct) > 3:
            score += 30
            direction = "up" if change_pct > 0 else "down"
            reasons.append(f"Strong move {direction} {abs(change_pct):.1f}%")
        elif abs(change_pct) > 1:
            score += 20
            direction = "up" if change_pct > 0 else "down"
            reasons.append(f"Moving {direction} {abs(change_pct):.1f}%")

        # Determine signal
        signal = "WATCH"
        if change_pct > 2 and volume_ratio > 3:
            signal = "BUY"
        elif change_pct < -2 and volume_ratio > 3:
            signal = "SELL"

        rsi = indicators.get('rsi', 50)

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.VOLUME_SPIKE,
            score=min(score, 100),
            signal=signal,
            current_price=indicators.get('current_price', 0),
            change_percent=change_pct,
            volume_ratio=volume_ratio,
            rsi=rsi,
            trend=StockAnalyzer.get_trend(indicators),
            reason=" | ".join(reasons)
        )


class MarketScanner:
    """
    Main market scanner that combines all scan types.
    """

    def __init__(self, data_provider: Callable = None):
        """
        Initialize scanner.

        Args:
            data_provider: Function that takes symbol and returns DataFrame
        """
        self.data_provider = data_provider
        self.scanners = {
            ScanType.MOMENTUM: MomentumScanner.scan,
            ScanType.BREAKOUT: BreakoutScanner.scan,
            ScanType.OVERSOLD: OversoldScanner.scan,
            ScanType.VOLUME_SPIKE: VolumeSpikeScanner.scan,
        }
        self.results: List[ScanResult] = []

    def scan_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        scan_types: List[ScanType] = None
    ) -> List[ScanResult]:
        """
        Scan a single symbol.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            scan_types: Types of scans to run (default: all)

        Returns:
            List of scan results
        """
        if scan_types is None:
            scan_types = list(self.scanners.keys())

        results = []
        for scan_type in scan_types:
            if scan_type in self.scanners:
                try:
                    result = self.scanners[scan_type](df, symbol)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error scanning {symbol} for {scan_type}: {e}")

        return results

    def scan_watchlist(
        self,
        watchlist: Dict[str, pd.DataFrame],
        scan_types: List[ScanType] = None,
        filter_criteria: ScanFilter = None,
        max_workers: int = 4
    ) -> List[ScanResult]:
        """
        Scan multiple symbols.

        Args:
            watchlist: Dict of symbol -> DataFrame
            scan_types: Types of scans to run
            filter_criteria: Filter to apply to results
            max_workers: Parallel workers

        Returns:
            Sorted list of scan results
        """
        all_results = []

        if filter_criteria is None:
            filter_criteria = ScanFilter()

        def scan_one(item):
            symbol, df = item
            if symbol in filter_criteria.exclude_symbols:
                return []
            return self.scan_symbol(symbol, df, scan_types)

        # Parallel scanning
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scan_one, item): item[0] for item in watchlist.items()}

            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Scan error: {e}")

        # Apply filters
        filtered = self._apply_filters(all_results, filter_criteria)

        # Sort by score
        filtered.sort(key=lambda x: x.score, reverse=True)

        self.results = filtered
        return filtered

    def _apply_filters(
        self,
        results: List[ScanResult],
        criteria: ScanFilter
    ) -> List[ScanResult]:
        """Apply filter criteria to results"""
        filtered = []

        for result in results:
            if result.current_price < criteria.min_price:
                continue
            if result.current_price > criteria.max_price:
                continue
            if result.change_percent < criteria.min_change_percent:
                continue
            if result.change_percent > criteria.max_change_percent:
                continue
            if result.rsi < criteria.min_rsi:
                continue
            if result.rsi > criteria.max_rsi:
                continue
            if result.score < criteria.min_score:
                continue

            filtered.append(result)

        return filtered

    def get_top_results(self, n: int = 10, scan_type: ScanType = None) -> List[ScanResult]:
        """Get top N results, optionally filtered by scan type"""
        results = self.results

        if scan_type:
            results = [r for r in results if r.scan_type == scan_type]

        return results[:n]

    def get_buy_signals(self) -> List[ScanResult]:
        """Get all BUY signals"""
        return [r for r in self.results if r.signal == "BUY"]

    def get_sell_signals(self) -> List[ScanResult]:
        """Get all SELL signals"""
        return [r for r in self.results if r.signal == "SELL"]

    def print_results(self, max_results: int = 20):
        """Print scan results in a nice format"""
        print("\n" + "=" * 80)
        print("MARKET SCAN RESULTS")
        print("=" * 80)

        if not self.results:
            print("No results found matching criteria.")
            return

        for i, result in enumerate(self.results[:max_results], 1):
            print(f"\n{i}. {result.emoji} {result.symbol} - {result.scan_type.value.upper()}")
            print(f"   Score: {result.score:.0f}/100 ({result.strength})")
            print(f"   Price: ₹{result.current_price:,.2f} ({result.change_percent:+.1f}%)")
            print(f"   Volume: {result.volume_ratio:.1f}x avg | RSI: {result.rsi:.0f}")
            print(f"   Trend: {result.trend} | Signal: {result.signal}")
            print(f"   Reason: {result.reason}")

        print("\n" + "=" * 80)
        print(f"Total: {len(self.results)} stocks found")
        print(f"BUY signals: {len(self.get_buy_signals())}")
        print(f"SELL signals: {len(self.get_sell_signals())}")
        print("=" * 80)


# ============== QUICK FUNCTIONS ==============

def quick_scan(
    watchlist: Dict[str, pd.DataFrame],
    scan_type: ScanType = ScanType.MOMENTUM
) -> List[ScanResult]:
    """Quick scan for a specific type"""
    scanner = MarketScanner()
    return scanner.scan_watchlist(watchlist, [scan_type])


def scan_for_buys(watchlist: Dict[str, pd.DataFrame]) -> List[ScanResult]:
    """Scan for buy opportunities"""
    scanner = MarketScanner()
    scanner.scan_watchlist(watchlist)
    return scanner.get_buy_signals()


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("MARKET SCANNER - Test")
    print("=" * 50)

    # Create sample data for multiple stocks
    np.random.seed(42)

    def create_stock_data(base_price: float, trend: float = 0.001) -> pd.DataFrame:
        days = 60
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.randn(days) * 0.02 + trend
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.015)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.015)),
            'close': prices,
            'volume': np.random.randint(100000, 2000000, days)
        }, index=dates)

    # Create watchlist with different trends
    watchlist = {
        'RELIANCE': create_stock_data(2500, 0.003),   # Uptrend
        'TCS': create_stock_data(3500, 0.002),        # Uptrend
        'INFY': create_stock_data(1500, 0.001),       # Slight uptrend
        'HDFC': create_stock_data(2800, -0.001),      # Slight downtrend
        'ICICIBANK': create_stock_data(950, -0.002),  # Downtrend
        'SBIN': create_stock_data(600, 0.004),        # Strong uptrend
        'BHARTIARTL': create_stock_data(1200, 0.002), # Uptrend
        'ITC': create_stock_data(450, 0.001),         # Slight uptrend
        'KOTAKBANK': create_stock_data(1800, -0.001), # Slight downtrend
        'LT': create_stock_data(3200, 0.003),         # Uptrend
    }

    print(f"\nScanning {len(watchlist)} stocks...")

    # Run scanner
    scanner = MarketScanner()
    results = scanner.scan_watchlist(watchlist)

    # Print results
    scanner.print_results()

    print("\n" + "=" * 50)
    print("Market Scanner ready!")
    print("=" * 50)
