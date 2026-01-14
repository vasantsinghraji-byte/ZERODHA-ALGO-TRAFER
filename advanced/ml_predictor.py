# -*- coding: utf-8 -*-
"""
AI/ML Predictions Module - Smart Trading Insights!
===================================================
Uses machine learning to predict price movements.

Like having a crystal ball, but with math!

Features:
- Price direction prediction (Up/Down)
- Trend strength prediction
- Support/Resistance level detection
- Pattern recognition
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions"""
    DIRECTION = "direction"      # Up or Down
    TREND = "trend"              # Trend strength
    VOLATILITY = "volatility"    # Expected volatility
    SUPPORT_RESISTANCE = "sr"    # Support/Resistance levels


class SignalStrength(Enum):
    """How confident is the prediction"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class Prediction:
    """A single prediction result"""
    symbol: str
    prediction_type: PredictionType
    direction: str              # "UP", "DOWN", "NEUTRAL"
    confidence: float           # 0.0 to 1.0
    strength: SignalStrength
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "1D"       # Prediction timeframe
    timestamp: datetime = None
    features_used: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.features_used is None:
            self.features_used = []

    @property
    def emoji(self) -> str:
        """Get emoji for direction"""
        if self.direction == "UP":
            return "📈"
        elif self.direction == "DOWN":
            return "📉"
        return "➡️"

    @property
    def confidence_pct(self) -> str:
        """Confidence as percentage string"""
        return f"{self.confidence * 100:.1f}%"


class FeatureEngineering:
    """
    Create features for ML models.

    Features are the "ingredients" that help predict prices.
    """

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume

        Returns:
            DataFrame with added feature columns
        """
        data = df.copy()

        # Ensure lowercase columns
        data.columns = [c.lower() for c in data.columns]

        # === PRICE FEATURES ===

        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Price changes
        data['price_change'] = data['close'] - data['open']
        data['high_low_range'] = data['high'] - data['low']
        data['close_open_ratio'] = data['close'] / data['open']

        # === MOVING AVERAGES ===

        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # MA crossover signals
        data['sma_5_20_cross'] = (data['sma_5'] > data['sma_20']).astype(int)
        data['sma_10_50_cross'] = (data['sma_10'] > data['sma_50']).astype(int)

        # Price vs MA
        data['price_vs_sma20'] = data['close'] / data['sma_20'] - 1
        data['price_vs_sma50'] = data['close'] / data['sma_50'] - 1

        # === MOMENTUM INDICATORS ===

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # RSI zones
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        data['macd_cross'] = (data['macd'] > data['macd_signal']).astype(int)

        # Rate of Change
        data['roc_5'] = data['close'].pct_change(periods=5)
        data['roc_10'] = data['close'].pct_change(periods=10)

        # === VOLATILITY ===

        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (2 * bb_std)
        data['bb_lower'] = data['bb_middle'] - (2 * bb_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = true_range.rolling(window=14).mean()
        data['atr_pct'] = data['atr'] / data['close']

        # Historical volatility
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_20'] = data['returns'].rolling(window=20).std()

        # === VOLUME FEATURES ===

        if 'volume' in data.columns:
            data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']
            data['volume_change'] = data['volume'].pct_change()

            # On-Balance Volume
            obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            data['obv'] = obv
            data['obv_sma'] = obv.rolling(window=20).mean()

        # === PATTERN FEATURES ===

        # Candle patterns
        data['body_size'] = abs(data['close'] - data['open'])
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        data['is_bullish'] = (data['close'] > data['open']).astype(int)

        # Consecutive patterns
        data['bullish_streak'] = data['is_bullish'].groupby(
            (data['is_bullish'] != data['is_bullish'].shift()).cumsum()
        ).cumsum() * data['is_bullish']

        # === LAGGED FEATURES ===

        for lag in [1, 2, 3, 5]:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag) if 'volume' in data.columns else 0

        # === TARGET VARIABLE ===

        # Future returns (what we want to predict)
        data['future_returns_1d'] = data['close'].shift(-1) / data['close'] - 1
        data['future_returns_5d'] = data['close'].shift(-5) / data['close'] - 1
        data['future_direction'] = (data['future_returns_1d'] > 0).astype(int)

        return data

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature column names for ML"""
        return [
            'returns', 'price_change', 'high_low_range', 'close_open_ratio',
            'sma_5_20_cross', 'sma_10_50_cross', 'price_vs_sma20', 'price_vs_sma50',
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'macd_histogram', 'macd_cross',
            'roc_5', 'roc_10',
            'bb_width', 'bb_position',
            'atr_pct', 'volatility_5', 'volatility_20',
            'volume_ratio', 'volume_change',
            'body_size', 'is_bullish', 'bullish_streak',
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3',
        ]


class SimpleMLPredictor:
    """
    Simple ML predictor using statistical methods.

    No complex libraries needed - uses basic math!
    Works like a voting system where multiple indicators vote.
    """

    def __init__(self):
        self.feature_weights = {
            'trend': {
                'sma_5_20_cross': 1.0,
                'sma_10_50_cross': 1.5,
                'price_vs_sma20': 0.8,
                'macd_cross': 1.2,
            },
            'momentum': {
                'rsi': 1.0,
                'macd_histogram': 0.8,
                'roc_5': 0.6,
            },
            'volatility': {
                'bb_position': 1.0,
                'atr_pct': 0.5,
            },
            'volume': {
                'volume_ratio': 0.7,
            }
        }

        self.trained = False
        self.historical_accuracy = 0.0

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on historical data.

        Actually just calculates optimal weights based on correlations.

        Args:
            df: DataFrame with features and target

        Returns:
            Training metrics
        """
        data = FeatureEngineering.create_features(df)
        data = data.dropna()

        if len(data) < 50:
            logger.warning("Not enough data for training")
            return {'accuracy': 0.0, 'samples': len(data)}

        # Calculate feature correlations with future returns
        features = FeatureEngineering.get_feature_names()
        correlations = {}

        for feature in features:
            if feature in data.columns:
                corr = data[feature].corr(data['future_direction'])
                if not np.isnan(corr):
                    correlations[feature] = abs(corr)

        # Update weights based on correlations
        total_corr = sum(correlations.values()) or 1
        for feature, corr in correlations.items():
            normalized_weight = corr / total_corr * len(correlations)
            # Find which category this feature belongs to
            for category, weights in self.feature_weights.items():
                if feature in weights:
                    self.feature_weights[category][feature] = normalized_weight

        # Calculate historical accuracy
        predictions = self._predict_batch(data)
        correct = (predictions == data['future_direction']).sum()
        self.historical_accuracy = correct / len(predictions)
        self.trained = True

        logger.info(f"Model trained on {len(data)} samples, accuracy: {self.historical_accuracy:.2%}")

        return {
            'accuracy': self.historical_accuracy,
            'samples': len(data),
            'top_features': sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _predict_batch(self, data: pd.DataFrame) -> pd.Series:
        """Predict for multiple rows"""
        scores = pd.Series(0.0, index=data.index)

        for category, weights in self.feature_weights.items():
            for feature, weight in weights.items():
                if feature in data.columns:
                    # Normalize feature values
                    values = data[feature].fillna(0)

                    # Handle different feature types
                    if feature in ['sma_5_20_cross', 'sma_10_50_cross', 'macd_cross', 'is_bullish']:
                        # Binary features: 1 = bullish, 0 = bearish
                        contribution = (values * 2 - 1) * weight
                    elif feature == 'rsi':
                        # RSI: < 30 bullish, > 70 bearish
                        contribution = ((50 - values) / 50) * weight
                    elif feature == 'bb_position':
                        # BB: < 0.2 bullish, > 0.8 bearish
                        contribution = ((0.5 - values) * 2) * weight
                    elif feature in ['price_vs_sma20', 'price_vs_sma50']:
                        # Price vs MA: negative = oversold = bullish
                        contribution = -values * weight * 10
                    else:
                        # Default: positive correlation with direction
                        mean = values.mean()
                        std = values.std() or 1
                        contribution = ((values - mean) / std) * weight * 0.1

                    scores += contribution.fillna(0)

        # Convert scores to predictions
        return (scores > 0).astype(int)

    def predict(self, df: pd.DataFrame, symbol: str = "STOCK") -> Prediction:
        """
        Make a prediction for the latest data point.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            Prediction object
        """
        data = FeatureEngineering.create_features(df)
        data = data.dropna()

        if len(data) < 2:
            return Prediction(
                symbol=symbol,
                prediction_type=PredictionType.DIRECTION,
                direction="NEUTRAL",
                confidence=0.0,
                strength=SignalStrength.WEAK
            )

        # Get latest row
        latest = data.iloc[-1]

        # Calculate scores for each category
        category_scores = {}
        total_score = 0.0
        total_weight = 0.0

        for category, weights in self.feature_weights.items():
            cat_score = 0.0
            cat_weight = 0.0

            for feature, weight in weights.items():
                if feature in data.columns:
                    value = latest[feature]
                    if pd.isna(value):
                        continue

                    # Calculate contribution
                    if feature in ['sma_5_20_cross', 'sma_10_50_cross', 'macd_cross', 'is_bullish']:
                        contribution = (value * 2 - 1) * weight
                    elif feature == 'rsi':
                        contribution = ((50 - value) / 50) * weight
                    elif feature == 'bb_position':
                        contribution = ((0.5 - value) * 2) * weight
                    elif feature in ['price_vs_sma20', 'price_vs_sma50']:
                        contribution = -value * weight * 10
                    else:
                        # Use recent history for normalization
                        recent = data[feature].tail(20)
                        mean = recent.mean()
                        std = recent.std() or 1
                        contribution = ((value - mean) / std) * weight * 0.1

                    cat_score += contribution
                    cat_weight += weight

            if cat_weight > 0:
                category_scores[category] = cat_score / cat_weight
                total_score += cat_score
                total_weight += cat_weight

        # Final prediction
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0

        # Convert to direction and confidence
        if normalized_score > 0.3:
            direction = "UP"
            confidence = min(0.5 + normalized_score * 0.5, 0.95)
        elif normalized_score < -0.3:
            direction = "DOWN"
            confidence = min(0.5 + abs(normalized_score) * 0.5, 0.95)
        else:
            direction = "NEUTRAL"
            confidence = 0.5 - abs(normalized_score)

        # Determine strength
        if confidence >= 0.85:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.70:
            strength = SignalStrength.STRONG
        elif confidence >= 0.55:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate target and stop loss
        current_price = latest['close']
        atr = latest.get('atr', current_price * 0.02)

        if direction == "UP":
            target_price = current_price + (atr * 2)
            stop_loss = current_price - atr
        elif direction == "DOWN":
            target_price = current_price - (atr * 2)
            stop_loss = current_price + atr
        else:
            target_price = current_price
            stop_loss = current_price

        return Prediction(
            symbol=symbol,
            prediction_type=PredictionType.DIRECTION,
            direction=direction,
            confidence=confidence,
            strength=strength,
            target_price=target_price,
            stop_loss=stop_loss,
            features_used=list(category_scores.keys())
        )


class SupportResistanceDetector:
    """
    Detect support and resistance levels.

    Support = Floor (price tends to stop falling)
    Resistance = Ceiling (price tends to stop rising)
    """

    @staticmethod
    def find_levels(
        df: pd.DataFrame,
        num_levels: int = 3,
        sensitivity: float = 0.02
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels.

        Args:
            df: DataFrame with OHLCV data
            num_levels: Number of levels to find
            sensitivity: Price sensitivity for grouping

        Returns:
            Dict with 'support' and 'resistance' lists
        """
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Find local maxima and minima
        resistance_candidates = []
        support_candidates = []

        window = 5
        for i in range(window, len(data) - window):
            # Local maximum (resistance)
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_candidates.append(highs[i])

            # Local minimum (support)
            if lows[i] == min(lows[i-window:i+window+1]):
                support_candidates.append(lows[i])

        # Cluster nearby levels
        def cluster_levels(levels: List[float], sensitivity: float) -> List[float]:
            if not levels:
                return []

            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]

            for level in levels[1:]:
                if level <= current_cluster[-1] * (1 + sensitivity):
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]

            clusters.append(np.mean(current_cluster))
            return clusters

        resistance_levels = cluster_levels(resistance_candidates, sensitivity)
        support_levels = cluster_levels(support_candidates, sensitivity)

        # Filter by relevance (keep levels near current price)
        current_price = closes[-1]
        price_range = current_price * 0.15  # 15% range

        resistance_levels = [
            r for r in resistance_levels
            if current_price < r < current_price + price_range
        ][:num_levels]

        support_levels = [
            s for s in support_levels
            if current_price - price_range < s < current_price
        ][-num_levels:]

        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'current_price': current_price
        }


class TrendAnalyzer:
    """
    Analyze trend strength and direction.
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the current trend.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Trend analysis results
        """
        data = FeatureEngineering.create_features(df)
        latest = data.iloc[-1]

        # Trend direction from moving averages
        ma_signals = 0
        if latest.get('sma_5', 0) > latest.get('sma_20', 0):
            ma_signals += 1
        if latest.get('sma_10', 0) > latest.get('sma_50', 0):
            ma_signals += 1
        if latest.get('close', 0) > latest.get('sma_20', 0):
            ma_signals += 1
        if latest.get('close', 0) > latest.get('sma_50', 0):
            ma_signals += 1

        # Trend strength from ADX-like calculation
        price_changes = data['close'].diff().tail(14)
        up_moves = price_changes.where(price_changes > 0, 0).sum()
        down_moves = abs(price_changes.where(price_changes < 0, 0).sum())
        total_moves = up_moves + down_moves

        if total_moves > 0:
            trend_strength = abs(up_moves - down_moves) / total_moves
        else:
            trend_strength = 0

        # Determine trend
        if ma_signals >= 3:
            trend = "STRONG_UP"
            emoji = "🚀"
        elif ma_signals >= 2:
            trend = "UP"
            emoji = "📈"
        elif ma_signals <= 1:
            trend = "STRONG_DOWN"
            emoji = "📉"
        elif ma_signals <= 2:
            trend = "DOWN"
            emoji = "⬇️"
        else:
            trend = "SIDEWAYS"
            emoji = "➡️"

        return {
            'trend': trend,
            'emoji': emoji,
            'strength': trend_strength,
            'ma_signals': ma_signals,
            'rsi': latest.get('rsi', 50),
            'macd_bullish': latest.get('macd', 0) > latest.get('macd_signal', 0),
            'above_sma20': latest.get('close', 0) > latest.get('sma_20', 0),
            'above_sma50': latest.get('close', 0) > latest.get('sma_50', 0),
        }


class MLPredictor:
    """
    Main ML Predictor class combining all prediction methods.
    """

    def __init__(self):
        self.simple_predictor = SimpleMLPredictor()
        self.sr_detector = SupportResistanceDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the predictor"""
        result = self.simple_predictor.train(df)
        self.trained = True
        return result

    def predict(self, df: pd.DataFrame, symbol: str = "STOCK") -> Prediction:
        """Make a prediction"""
        return self.simple_predictor.predict(df, symbol)

    def get_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Get support and resistance levels"""
        return self.sr_detector.find_levels(df)

    def get_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get trend analysis"""
        return self.trend_analyzer.analyze(df)

    def get_full_analysis(self, df: pd.DataFrame, symbol: str = "STOCK") -> Dict[str, Any]:
        """
        Get complete analysis including prediction, S/R, and trend.
        """
        prediction = self.predict(df, symbol)
        sr_levels = self.get_support_resistance(df)
        trend = self.get_trend(df)

        return {
            'symbol': symbol,
            'prediction': prediction,
            'support_resistance': sr_levels,
            'trend': trend,
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(prediction, sr_levels, trend)
        }

    def _generate_summary(
        self,
        prediction: Prediction,
        sr: Dict[str, List[float]],
        trend: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary"""
        lines = [
            f"{prediction.emoji} Prediction: {prediction.direction} ({prediction.confidence_pct} confidence)",
            f"{trend['emoji']} Trend: {trend['trend']} (strength: {trend['strength']:.0%})",
        ]

        if sr['resistance']:
            lines.append(f"🔴 Resistance: {sr['resistance'][0]:,.2f}")
        if sr['support']:
            lines.append(f"🟢 Support: {sr['support'][-1]:,.2f}")

        if prediction.target_price:
            lines.append(f"🎯 Target: {prediction.target_price:,.2f}")
        if prediction.stop_loss:
            lines.append(f"🛑 Stop Loss: {prediction.stop_loss:,.2f}")

        return "\n".join(lines)


# ============== QUICK FUNCTIONS ==============

def quick_predict(df: pd.DataFrame, symbol: str = "STOCK") -> Prediction:
    """Quick prediction without training"""
    predictor = MLPredictor()
    return predictor.predict(df, symbol)


def quick_analysis(df: pd.DataFrame, symbol: str = "STOCK") -> Dict[str, Any]:
    """Quick full analysis"""
    predictor = MLPredictor()
    return predictor.get_full_analysis(df, symbol)


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("ML PREDICTOR - Test")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    days = 100

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = 100.0
    returns = np.random.randn(days) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)

    print("\nSample data:")
    print(df.tail())

    # Test predictor
    predictor = MLPredictor()

    print("\n--- Training ---")
    metrics = predictor.train(df)
    print(f"Training accuracy: {metrics['accuracy']:.2%}")

    print("\n--- Prediction ---")
    prediction = predictor.predict(df, "TEST")
    print(f"Direction: {prediction.emoji} {prediction.direction}")
    print(f"Confidence: {prediction.confidence_pct}")
    print(f"Strength: {prediction.strength.name}")
    print(f"Target: {prediction.target_price:.2f}")
    print(f"Stop Loss: {prediction.stop_loss:.2f}")

    print("\n--- Support/Resistance ---")
    sr = predictor.get_support_resistance(df)
    print(f"Support levels: {sr['support']}")
    print(f"Resistance levels: {sr['resistance']}")

    print("\n--- Trend Analysis ---")
    trend = predictor.get_trend(df)
    print(f"Trend: {trend['emoji']} {trend['trend']}")
    print(f"Strength: {trend['strength']:.0%}")

    print("\n--- Full Analysis ---")
    analysis = predictor.get_full_analysis(df, "RELIANCE")
    print(analysis['summary'])

    print("\n" + "=" * 50)
    print("ML Predictor ready!")
    print("=" * 50)
