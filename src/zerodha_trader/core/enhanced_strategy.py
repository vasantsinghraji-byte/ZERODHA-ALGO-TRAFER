        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return self._neutral_signal(reason)

        # Calculate all indicators
        vwap_data = self.vwap.calculate(df)
        rsi_data = self.rsi.calculate(df)
        volume_data = self.volume.analyze(df)
        trend_data = self.trend.analyze(df)

        # Evaluate trading opportunity
        signal_type, confidence, reasons = self._evaluate_opportunity(
            df, vwap_data, rsi_data, volume_data, trend_data, market_conditions
        )

        if signal_type == SignalType.NEUTRAL:
            return self._neutral_signal("No clear opportunity")

        # Apply multi-timeframe filter if provided
        if self.use_multi_timeframe and mtf_analysis:
            signal_type, confidence, reasons = self._apply_mtf_filter(
                signal_type, confidence, reasons, mtf_analysis
            )

            if signal_type == SignalType.NEUTRAL:
                return self._neutral_signal("Filtered by multi-timeframe analysis")

        # Generate full signal with entry/exit levels
        return self._create_full_signal(
            signal_type, confidence, reasons, df, vwap_data, rsi_data, trend_data
        )

    def _evaluate_opportunity(self, df: pd.DataFrame, vwap_data: Dict,
                            rsi_data: Dict, volume_data: Dict,
                            trend_data: Dict, market_conditions: Dict) -> Tuple[SignalType, float, List[str]]:
        """Evaluate if there's a trading opportunity"""

        long_score = 0
        short_score = 0
        reasons_long = []
        reasons_short = []

        current_price = df['close'].iloc[-1]

        # === LONG CONDITIONS ===

        # 1. Trend confirmation (30 points)
        if trend_data['bullish_alignment']:
            long_score += 30
            reasons_long.append(f"Bullish EMA alignment (ADX: {trend_data['adx']:.1f})")
        elif trend_data['trend'] == "STRONG_UPTREND":
            long_score += 25
            reasons_long.append("Strong uptrend detected")
        elif trend_data['trend'] == "WEAK_TREND" and current_price > trend_data['ema_21']:
            # ADDED: Give partial points for weak trends if price above EMA21
            long_score += 15
            reasons_long.append(f"Weak uptrend (ADX: {trend_data['adx']:.1f})")

        # 2. VWAP confirmation (25 points)
        if vwap_data['price_above_vwap'] and not vwap_data['extreme_position']:
            long_score += 25
            reasons_long.append(f"Price above VWAP ({vwap_data['distance_pct']:.2f}% above)")
        elif current_price <= vwap_data['lower_1'] and current_price > vwap_data['lower_2']:
            long_score += 20
            reasons_long.append("Price at VWAP lower band (mean reversion setup)")

        # 3. RSI confirmation (25 points)
        if rsi_data['regular_divergence'] == "BULLISH_DIVERGENCE":
            long_score += 25
            reasons_long.append(f"Bullish divergence (RSI: {rsi_data['rsi']:.1f})")
        elif 35 <= rsi_data['rsi'] <= 55 and rsi_data['rsi_momentum'] > 0:
            long_score += 20
            reasons_long.append(f"RSI in buy zone with positive momentum")
        elif rsi_data['oversold']:
            long_score += 15
            reasons_long.append("RSI oversold - bounce expected")

        # 4. Volume confirmation (20 points)
        if volume_data['institutional_buying']:
            long_score += 20
            reasons_long.append(f"Institutional buying detected (Vol ratio: {volume_data['volume_ratio']:.2f}x)")
        elif volume_data['healthy_volume']:
            long_score += 10
            reasons_long.append("Healthy volume confirmation")

        # === SHORT CONDITIONS ===

        # 1. Trend confirmation (30 points)
        if trend_data['bearish_alignment']:
            short_score += 30
            reasons_short.append(f"Bearish EMA alignment (ADX: {trend_data['adx']:.1f})")
        elif trend_data['trend'] == "STRONG_DOWNTREND":
            short_score += 25
            reasons_short.append("Strong downtrend detected")
        elif trend_data['trend'] == "WEAK_TREND" and current_price < trend_data['ema_21']:
            # ADDED: Give partial points for weak trends if price below EMA21
            short_score += 15
            reasons_short.append(f"Weak downtrend (ADX: {trend_data['adx']:.1f})")

        # 2. VWAP confirmation (25 points)
        if not vwap_data['price_above_vwap'] and not vwap_data['extreme_position']:
            short_score += 25
            reasons_short.append(f"Price below VWAP ({vwap_data['distance_pct']:.2f}% below)")
        elif current_price >= vwap_data['upper_1'] and current_price < vwap_data['upper_2']:
            short_score += 20
            reasons_short.append("Price at VWAP upper band (mean reversion setup)")

        # 3. RSI confirmation (25 points)
        if rsi_data['regular_divergence'] == "BEARISH_DIVERGENCE":
            short_score += 25
            reasons_short.append(f"Bearish divergence (RSI: {rsi_data['rsi']:.1f})")
        elif 45 <= rsi_data['rsi'] <= 65 and rsi_data['rsi_momentum'] < 0:
            short_score += 20
            reasons_short.append(f"RSI in sell zone with negative momentum")
        elif rsi_data['overbought']:
            short_score += 15
            reasons_short.append("RSI overbought - pullback expected")

        # 4. Volume confirmation (20 points)
        if volume_data['institutional_selling']:
            short_score += 20
            reasons_short.append(f"Institutional selling detected (Vol ratio: {volume_data['volume_ratio']:.2f}x)")
        elif volume_data['healthy_volume']:
            short_score += 10
            reasons_short.append("Healthy volume confirmation")

        # Determine signal
        # FIXED: Lowered from 70 to 55 to match balanced mode threshold
        # Old: 70 points = too strict, rejected good trades
        # New: 55 points = balanced, catches opportunities while maintaining quality
        min_score = 55  # Minimum 55 points needed (was 70)

        if long_score >= min_score and long_score > short_score:
            confidence = min(long_score, 100)
            return SignalType.LONG, confidence, reasons_long
        elif short_score >= min_score and short_score > long_score:
            confidence = min(short_score, 100)
            return SignalType.SHORT, confidence, reasons_short
        else:
            return SignalType.NEUTRAL, 0, ["Insufficient confirmation"]

    def _create_full_signal(self, signal_type: SignalType, confidence: float,
                           reasons: List[str], df: pd.DataFrame,
                           vwap_data: Dict, rsi_data: Dict, trend_data: Dict) -> EnhancedSignal:
        """Create complete trading signal with all parameters"""

        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)

        if signal_type == SignalType.LONG:
            # Entry: Current price
            entry = current_price

            # Stop loss: 2 ATR below or below key support
            stop_loss = min(entry - (2.0 * atr), vwap_data['lower_1'])

            # Take profits: Progressive targets
            take_profit_1 = entry + (1.5 * atr)  # 1.5 R
            take_profit_2 = entry + (2.5 * atr)  # 2.5 R
            take_profit_3 = entry + (4.0 * atr)  # 4 R

            # Trailing stop activation at 2R
            trailing_activation = entry + (2.0 * atr)

        else:  # SHORT
            entry = current_price
            stop_loss = max(entry + (2.0 * atr), vwap_data['upper_1'])

            take_profit_1 = entry - (1.5 * atr)
            take_profit_2 = entry - (2.5 * atr)
            take_profit_3 = entry - (4.0 * atr)

            trailing_activation = entry - (2.0 * atr)

        # Calculate risk-reward
        risk = abs(entry - stop_loss)
        reward = abs(entry - take_profit_2)
        risk_reward = reward / risk if risk > 0 else 0

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            entry, stop_loss, confidence, trend_data['trend_strength']
        )

        return EnhancedSignal(
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            trailing_stop_activation=trailing_activation,
            position_size=position_size,
            risk_reward_ratio=risk_reward,
            reasons=reasons,
            market_condition=self._get_market_condition(trend_data),
            timestamp=datetime.now()
        )
