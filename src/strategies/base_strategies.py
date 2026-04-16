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
    horizon: int = 1 # Default

    # Research Variables (Configurable)
    v_size: float = 0.1 # Volume (v) - Fraction of equity or fixed size

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

        self.current_idx = 0

    def get_prediction(self):
        """Helper to get the pre-calculated model prediction."""
        prediction = self.all_predictions[self.current_idx]
        self.current_idx += 1
        return prediction

    
class TripleBarrierStrategy(MLBaseStrategy):
    """
    RESEARCH-GRADE: Triple Barrier Strategy.
    Uses dynamic TP/SL based on ATR (Average True Range).
    """
    def next(self):
        prediction = self.get_prediction()
        # Get the ATR for the entry bar
        current_atr = self.atr[self.current_idx-1] if self.atr is not None else 0.0002

        # Current price
        price = self.data.Close[-1]

        if prediction == 0: # Up
            if not self.position.is_long:
                self.position.close()
                # Use ATR for physical distance TP/SL
                self.buy(
                    size=self.v_size,
                    tp=price + current_atr,
                    sl=price - current_atr
                )
        elif prediction == 1: # Down
            if not self.position.is_short:
                self.position.close()
                self.sell(
                    size=self.v_size,
                    tp=price - current_atr,
                    sl=price + current_atr
                )
        elif prediction == 2: # Neutral
            self.position.close()


class NaiveFlipStrategy(MLBaseStrategy):
    """
    RESEARCH BASELINE 1: The 'Naive Flip' (Absolute Baseline).
    Responds instantly to every model signal change.
    """
    def next(self):
        prediction = self.get_prediction()

        if prediction == 0: # Up (Alpha=0)
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.v_size)
        elif prediction == 1: # Down (Alpha=1)
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.v_size)
        elif prediction == 2: # Deadband
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
