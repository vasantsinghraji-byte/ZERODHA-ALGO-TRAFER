                    X_seq, y_seq,
                    epochs=20,
                    batch_size=32,
                    verbose=0
                )
            else:
                # Fallback to RandomForest
                from sklearn.ensemble import RandomForestClassifier
                print("TensorFlow not available, using RandomForest fallback")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                X_flat = X_seq.reshape(X_seq.shape[0], -1)
                self.model.fit(X_flat, y_seq)

        self.is_trained = True
        self.training_date = datetime.now()

        # Evaluate
        predictions = self.predict(X_train)

        metrics = ModelMetrics()
        y_actual = y_seq  # Use sequence targets for evaluation
        metrics.accuracy = accuracy_score(y_actual, predictions)
        metrics.precision = precision_score(y_actual, predictions, zero_division=0)
        metrics.recall = recall_score(y_actual, predictions, zero_division=0)
        metrics.f1_score = f1_score(y_actual, predictions, zero_division=0)
        self.metrics = metrics

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions - override to handle sequences"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Add dummy target column (will be ignored in prediction)
        dummy_y = np.zeros(len(X))
        combined_data = np.column_stack([dummy_y, X.values])

        # Prepare sequences
        X_seq, _ = self._prepare_sequences(combined_data)

        # Predict
        if hasattr(self.model, 'predict_proba'):
            # RandomForest fallback
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            return self.model.predict(X_flat)
        else:
            # Keras LSTM
            predictions = self.model.predict(X_seq, verbose=0)
            return (predictions > 0.5).astype(int).flatten()
