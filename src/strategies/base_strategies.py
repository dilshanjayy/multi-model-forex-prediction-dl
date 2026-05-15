import joblib
import numpy as np
import os
from typing import Any, List
from collections import Counter
from backtesting import Strategy


class MLBaseStrategy(Strategy):
    """
    Abstract Base Strategy for ML models.
    Handles model loading and feature scaling common to all ML strategies.
    """

    model_path: str = ""
    artifacts: Any = None
    feature_cols: List[str] = []
    scaler: Any = None
    model: Any = None
    horizon: int = 1  # Default

    # Research Variables (Configurable)
    v_size: float = 0.1  # Volume (v) - Fraction of equity or fixed size
    atr_multiplier: float = 1.0
    conf_threshold: float = 0.50

    def init(self):
        # 1. Load Artifacts
        if self.artifacts is not None:
            artifacts = self.artifacts
            # CRITICAL: During optimization, the model is already trained and passed in artifacts
            if "model" in artifacts:
                self.model = artifacts["model"]
        else:
            artifacts = joblib.load(self.model_path)

        # 2. Reconstruct Model metadata
        from src.models.model_factory import ModelFactory

        self.model_type = artifacts.get("model_type", "RandomForest")
        self.model_params = artifacts.get("model_params", {})
        self.feature_cols = artifacts["feature_cols"]
        self.scaler = artifacts["scaler"]
        self.horizon = artifacts.get("horizon", 1)

        # 3. Instantiate/Load Weights
        if self.model is None:
            # Standalone mode: Reconstruct and load weights
            self.model = ModelFactory.get_model(self.model_type, self.model_params)

            if self.artifacts is None:
                state_path = self.model_path.replace(
                    "model.joblib", "model_state.joblib"
                )
                if os.path.exists(state_path):
                    self.model.load(state_path)
                elif "model" in artifacts:
                    # Scikit-learn fallback
                    self.model = artifacts["model"]

        if self.scaler is None or self.model is None:
            raise ValueError("Model or Scaler failed to load.")

        # Load ATR for Triple Barrier exits (mirroring the labels)
        if "ATRr_14" in self.data.df.columns:
            self.atr = self.data.df["ATRr_14"].values
        else:
            self.atr = None

        if "RSI_14" not in self.data.df.columns and "RSI_14_Z" in self.data.df.columns:
            self.data.df["RSI_14"] = self.data.df["RSI_14_Z"] * 10 + 50

        if (
            "real_volume" in self.feature_cols
            and "real_volume" not in self.data.df.columns
        ):
            self.data.df["real_volume"] = 0.0

        # Pre-calculate features and ALL predictions at once (Bulk Inference)
        # This is the single biggest speed optimization for ML backtesting
        self.prepared_data = self.data.df[self.feature_cols]
        self.scaled_features = self.scaler.transform(self.prepared_data)

        print(f"Running bulk inference on {len(self.scaled_features)} rows...")

        # FEATURE AUDIT: Print a summary of the first row to detect environment drift
        first_row_sum = np.sum(self.scaled_features[self.horizon + 10])  # Skip warm-up
        print(
            f"FEATURE AUDIT: Checksum of row {self.horizon + 10}: {first_row_sum:.6f}"
        )

        # Both Tabular and Deep Learning models expect the 2D scaled_features here.
        # The PyTorchBaseModel internally converts the 2D array into 3D sliding windows
        # and pads the warm-up period to ensure the output length matches the input length.
        self.all_predictions = self.model.predict(self.scaled_features)
        self.all_probas = self.model.predict_proba(self.scaled_features)

        # DEBUG: Print signal distribution
        unique, counts = np.unique(self.all_predictions, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Signal Distribution: {dist}")

        passed_conf = np.sum(
            (self.all_predictions != 2)
            & (np.max(self.all_probas, axis=1) >= self.conf_threshold)
        )
        print(f"Signals passing confidence ({self.conf_threshold}): {passed_conf}")

        self.current_idx = 0

    def get_prediction(self):
        """Helper to get the pre-calculated model prediction."""
        prediction = int(self.all_predictions[self.current_idx])
        confidence = self.all_probas[self.current_idx][prediction]
        self.current_idx += 1
        return prediction, confidence


class TripleBarrierStrategy(MLBaseStrategy):
    """
    RESEARCH-GRADE: Triple Barrier Strategy.
    Exits via:
    1. Take Profit (ATR-based)
    2. Stop Loss (ATR-based)
    3. Vertical Barrier (Time-out after self.horizon bars)
    """

    def next(self):
        prediction, confidence = self.get_prediction()

        # Handle Vertical Barrier (Time-out Exit)
        for trade in self.trades:
            if len(self.data) - trade.entry_bar >= self.horizon:
                trade.close()

        # Get ATR for dynamic TP/SL
        current_atr = (
            self.atr[self.current_idx - 1] if self.atr is not None else 0.0002
        ) * self.atr_multiplier
        price = self.data.Close[-1]

        # Signal-based Entry with Confidence Filter
        if self.position:
            return

        if (
            prediction == 0 and confidence >= self.conf_threshold
        ):  # Profit predicted (Up)
            self.buy(size=self.v_size, tp=price + current_atr, sl=price - current_atr)
        elif (
            prediction == 1 and confidence >= self.conf_threshold
        ):  # Loss predicted (Down)
            self.sell(size=self.v_size, tp=price - current_atr, sl=price + current_atr)
        elif prediction == 2:  # Neutral
            pass


class ContinuousSignalExecutionStrategy(MLBaseStrategy):
    """
    RESEARCH BASELINE 1: The 'Naive Flip' (Absolute Baseline).
    Responds instantly to every model signal change that passes the confidence threshold.
    """

    def next(self):
        prediction, confidence = self.get_prediction()

        if prediction == 0 and confidence >= self.conf_threshold:  # Up
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.v_size)
        elif prediction == 1 and confidence >= self.conf_threshold:  # Down
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.v_size)
        elif prediction == 2:  # Deadband
            self.position.close()


class MajorityVoteStrategy(MLBaseStrategy):
    """
    RESEARCH BASELINE 2: The 'Majority Vote' (Realistic Baseline).
    Automatically uses the model's training horizon as the smoothing window.
    """

    def init(self):
        super().init()
        self.window_size = self.horizon
        self.signal_history = []

    def next(self):
        raw_prediction, _ = self.get_prediction()

        # Seeding & Rolling Window
        if not self.signal_history:
            self.signal_history = [raw_prediction] * self.window_size
        else:
            self.signal_history.pop(0)
            self.signal_history.append(raw_prediction)

        # Calculate Majority (The 'Switch' alpha logic)
        majority_signal = Counter(self.signal_history).most_common(1)[0][0]

        if majority_signal == 0:
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.v_size)
        elif majority_signal == 1:
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.v_size)
        elif majority_signal == 2:
            self.position.close()
