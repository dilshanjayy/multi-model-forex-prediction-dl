import joblib
import numpy as np
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel

class RFModel(BaseModel):
    """
    Random Forest Implementation of the BaseModel.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = RandomForestClassifier(**self.config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"Training RandomForest with params: {self.config}")
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    @property
    def feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return {}
