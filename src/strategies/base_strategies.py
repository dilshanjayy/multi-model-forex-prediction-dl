import joblib
from typing import Any, List
from collections import Counter
from backtesting import Strategy


class MLBaseStrategy(Strategy):
    """
    Abstract Base Strategy for ML models.
    Handles model loading and feature scaling common to all ML strategies.
    """

    model_path: str = ""
    feature_cols: List[str] = []
    scaler: Any = None
    model: Any = None
    horizon: int = 1  # Default

    # Research Variables (Configurable)
    v_size: float = 0.1  # Volume (v) - Fraction of equity or fixed size
    atr_multiplier: float = 1.0

    def init(self):
        # Load the model artifacts once
        if not self.model:
            artifacts = joblib.load(self.model_path)
            self.model = artifacts["model"]
            self.scaler = artifacts["scaler"]
            self.feature_cols = artifacts["feature_cols"]
            # Dynamically read the horizon from the model file!
            self.horizon = artifacts.get("horizon", 1)
            print(f"Strategy linked to model with {self.horizon}h horizon.")

        if self.scaler is None or self.model is None:
            raise ValueError("Model or Scaler failed to load.")

        # Load ATR for Triple Barrier exits (mirroring the labels)
        if "ATRr_14" in self.data.df.columns:
            self.atr = self.data.df["ATRr_14"].values
        else:
            self.atr = None

        # Pre-calculate features and ALL predictions at once (Bulk Inference)
        # This is the single biggest speed optimization for ML backtesting
        self.prepared_data = self.data.df[self.feature_cols]
        self.scaled_features = self.scaler.transform(self.prepared_data)

        print(f"Running bulk inference on {len(self.scaled_features)} rows...")
        self.all_predictions = self.model.predict(self.scaled_features)
        self.all_probas = self.model.predict_proba(self.scaled_features)

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

    # Confidence filter: Only trade if model is >50% sure
    conf_threshold: float = 0.40

    def next(self):
        prediction, confidence = self.get_prediction()

        # 1. Handle Vertical Barrier (Time-out Exit)
        # We manually check the age of every open trade
        for trade in self.trades:
            if len(self.data) - trade.entry_bar >= self.horizon:
                trade.close()

        # 2. Get ATR for dynamic TP/SL
        current_atr = (
            self.atr[self.current_idx - 1] if self.atr is not None else 0.0002
        ) * self.atr_multiplier
        price = self.data.Close[-1]

        # 3. Signal-based Entry with Confidence Filter
        if (
            prediction == 0 and confidence >= self.conf_threshold
        ):  # Profit predicted (Up)
            if not self.position.is_long:
                self.position.close()  # Close any existing short
                self.buy(
                    size=self.v_size, tp=price + current_atr, sl=price - current_atr
                )
        elif (
            prediction == 1 and confidence >= self.conf_threshold
        ):  # Loss predicted (Down)
            if not self.position.is_short:
                self.position.close()  # Close any existing long
                self.sell(
                    size=self.v_size, tp=price - current_atr, sl=price + current_atr
                )
        elif prediction == 2:  # Neutral
            # We "do nothing". We don't enter, and we don't close
            # existing trades (they will hit TP/SL or Vertical Barrier).
            pass


class NaiveFlipStrategy(MLBaseStrategy):
    """
    RESEARCH BASELINE 1: The 'Naive Flip' (Absolute Baseline).
    Responds instantly to every model signal change.
    """

    def next(self):
        prediction = self.get_prediction()

        if prediction == 0:  # Up (Alpha=0)
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.v_size)
        elif prediction == 1:  # Down (Alpha=1)
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
        # window_size is no longer hardcoded!
        self.window_size = self.horizon
        self.signal_history = []

    def next(self):
        raw_prediction = self.get_prediction()

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
