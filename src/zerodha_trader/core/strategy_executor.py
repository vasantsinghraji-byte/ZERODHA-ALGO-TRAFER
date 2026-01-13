                self.ohlc_data[instrument_token] = pd.DataFrame(columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])

            df = self.ohlc_data[instrument_token]

            # Check if we need to create a new candle (1-minute candles)
            if len(df) == 0:
                # First candle
                new_row = pd.DataFrame([{
                    'timestamp': timestamp,
                    'open': ltp,
                    'high': ltp,
                    'low': ltp,
                    'close': ltp,
                    'volume': volume
                }])
                self.ohlc_data[instrument_token] = pd.concat([df, new_row], ignore_index=True)

            else:
                last_candle_time = df.iloc[-1]['timestamp']

                # Check if same minute (update current candle)
                if timestamp.replace(second=0, microsecond=0) == last_candle_time.replace(second=0, microsecond=0):
                    # Update current candle
                    df.loc[df.index[-1], 'high'] = max(df.iloc[-1]['high'], ltp)
                    df.loc[df.index[-1], 'low'] = min(df.iloc[-1]['low'], ltp)
                    df.loc[df.index[-1], 'close'] = ltp
                    df.loc[df.index[-1], 'volume'] = volume

                else:
                    # New candle
                    new_row = pd.DataFrame([{
                        'timestamp': timestamp,
                        'open': ltp,
                        'high': ltp,
                        'low': ltp,
                        'close': ltp,
                        'volume': volume
                    }])
                    self.ohlc_data[instrument_token] = pd.concat([df, new_row], ignore_index=True)

                    # Keep only last 200 candles
                    if len(self.ohlc_data[instrument_token]) > 200:
                        self.ohlc_data[instrument_token] = self.ohlc_data[instrument_token].iloc[-200:]

        except Exception as e:
            logger.error(f"Error updating OHLC for {instrument_token}: {e}")

    def _can_generate_signal(self, instrument_token: int) -> bool:
        """
        Check if enough time has passed since last signal

        Args:
            instrument_token: Instrument token

        Returns:
            True if can generate signal
        """
        # Check if paused
        if self.paused:
            return False

        if instrument_token not in self.last_signal_time:
            return True

        last_time = self.last_signal_time[instrument_token]
        elapsed = (datetime.now() - last_time).total_seconds()

        return elapsed >= self.signal_cooldown

    def _generate_signal(self, instrument_token: int, df: pd.DataFrame) -> Optional[EnhancedSignal]:
        """
        Generate trading signal from OHLC data

        Args:
            instrument_token: Instrument token
            df: OHLC DataFrame

        Returns:
            EnhancedSignal or None
        """
        try:
            signal = self.strategy.analyze_and_generate_signal(df)
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {instrument_token}: {e}")
            return None

    def _handle_signal(self, signal: EnhancedSignal, instrument_token: int):
        """
        Handle generated signal

        Args:
            signal: Generated signal
            instrument_token: Instrument token
        """
        try:
            # Update last signal time
            self.last_signal_time[instrument_token] = datetime.now()

            # Add to history
            self.signal_history[instrument_token].append(signal)

            # Keep only last 50 signals per instrument
            if len(self.signal_history[instrument_token]) > 50:
                self.signal_history[instrument_token] = self.signal_history[instrument_token][-50:]

            logger.info(f"Signal generated for {instrument_token}: {signal.signal_type.value} "
                       f"@ {signal.entry_price} (confidence: {signal.confidence:.1f}%)")

            # Distribute to callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal, instrument_token)
                except Exception as e:
                    logger.error(f"Error in signal callback {callback.__name__}: {e}")

        except Exception as e:
            logger.error(f"Error handling signal: {e}")

    def register_signal_callback(self, callback: Callable[[EnhancedSignal, int], None]):
        """
        Register callback for signals

        Args:
            callback: Function that accepts (signal, instrument_token)
        """
        if callback not in self.signal_callbacks:
            self.signal_callbacks.append(callback)
            logger.info(f"Registered signal callback: {callback.__name__}")

    def unregister_signal_callback(self, callback: Callable[[EnhancedSignal, int], None]):
        """
        Unregister signal callback

        Args:
            callback: Callback to remove
        """
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
            logger.info(f"Unregistered signal callback: {callback.__name__}")

    def get_ohlc_data(self, instrument_token: int) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for instrument

        Args:
            instrument_token: Instrument token

        Returns:
            DataFrame or None
        """
        return self.ohlc_data.get(instrument_token)

    def get_signal_history(self, instrument_token: int) -> List[EnhancedSignal]:
        """
        Get signal history for instrument

        Args:
            instrument_token: Instrument token

        Returns:
            List of signals
        """
        return self.signal_history.get(instrument_token, [])

    def get_stats(self) -> Dict:
        """
        Get StrategyExecutor statistics

        Returns:
            Dictionary with stats
        """
        total_signals = sum(len(signals) for signals in self.signal_history.values())

        return {
            'monitored_instruments': len(self.monitored_instruments),
            'total_signals_generated': total_signals,
            'registered_callbacks': len(self.signal_callbacks),
            'instruments_with_data': len(self.ohlc_data)
        }
