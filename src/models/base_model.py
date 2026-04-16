from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class BaseModel(ABC):
    """
    Abstract Base Class for all models in the pipeline.
    Ensures a consistent interface for training, prediction, and artifact saving.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs inference and returns predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Runs inference and returns prediction probabilities."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the model weights/artifacts to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads model weights/artifacts from the specified path."""
        pass

    @property
    @abstractmethod
    def feature_importance(self) -> Dict[str, float]:
        """Returns a dictionary of feature importances if applicable."""
        pass
