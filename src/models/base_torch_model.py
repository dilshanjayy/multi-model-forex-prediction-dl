import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from src.models.base_model import BaseModel


class PyTorchBaseModel(BaseModel, nn.Module):
    """
    Abstract Base Class for all PyTorch-based models.
    Hides the complexity of training loops and device management.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nn.Module.__init__(self)  # Multiple inheritance init
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Training Parameters
        self.epochs = config.get("epochs", 50)
        self.lr = config.get("learning_rate", 1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.best_state = None  # To store best weights

        # Placeholder for the actual network (to be defined by child classes)
        self.network = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Satisfies BaseModel's abstract fit method.
        PyTorch models should use train_model() with DataLoaders instead to handle memory efficiently.
        """
        raise NotImplementedError(
            "PyTorch models use train_model(train_loader, val_loader) for sequence training."
        )

    def train_model(self, train_loader, val_loader):
        """
        Standard PyTorch Training Loop with Early Stopping and Weight Restoration.
        """
        self.to(self.device)
        print(f"Starting Training on {self.device}...")

        # 1. Balanced Class Weights (Focus on directional signals)
        all_labels = train_loader.dataset.y.cpu().numpy()
        class_counts = np.bincount(all_labels, minlength=3)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        # Ensure Neutral class doesn't get more weight than signals
        weights[2] = min(weights[2], weights[0], weights[1])
        weights = weights / weights.sum() * 3
        class_weights = torch.FloatTensor(weights).to(self.device)

        print(f"Applied Class Weights: {weights}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_val_loss = float("inf")
        self.best_state = self.state_dict()  # Initial state
        patience = 15
        trigger = 0

        for epoch in range(self.epochs):
            # Training Phase
            self.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation Phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    output = self(x_val)
                    loss = self.criterion(output, y_val)
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {avg_val:.4f}"
                )

            # Always track and save the best state
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.state_dict().items()
                }
                trigger = 0
            else:
                trigger += 1
                if trigger >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}. Restoring best weights."
                    )
                    break

        # MANDATORY: Restore best performing weights for inference
        self.load_state_dict(self.best_state)
        self.to(self.device)

    def _process_2d_input(self, X: np.ndarray) -> torch.Tensor:
        lookback = self.config.get("lookback", 60)
        x_tensor = torch.from_numpy(X).float()
        # Create sliding windows: (N - lookback + 1, lookback, Features)
        windows = x_tensor.unfold(0, lookback, 1).transpose(1, 2)
        return windows

    def _batch_inference(
        self, windows: torch.Tensor, batch_size: int = 512
    ) -> torch.Tensor:
        self.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i : i + batch_size].to(self.device)
                out = self(batch)
                outputs.append(out.cpu())
        return torch.cat(outputs, dim=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Sequence inference for backtesting.
        Converts 2D array into 3D sliding windows and predicts the last step.
        """
        lookback = self.config.get("lookback", 60)
        num_samples = len(X)

        # Initialize full predictions with class 2 (Deadband/Neutral) for the warm-up period
        preds = np.full(num_samples, 2, dtype=int)

        if num_samples < lookback:
            return preds

        windows = self._process_2d_input(X)
        logits = self._batch_inference(windows)

        window_preds = torch.argmax(logits, dim=1).numpy()

        # Align predictions to the end of the window
        preds[lookback - 1 :] = window_preds
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lookback = self.config.get("lookback", 60)
        num_samples = len(X)
        num_classes = 3  # Assuming 3 classes (Up, Down, Deadband)

        # Initialize full probabilities with Deadband certainty for the warm-up period
        probs = np.zeros((num_samples, num_classes), dtype=float)
        probs[:, 2] = 1.0  # 100% probability for class 2

        if num_samples < lookback:
            return probs

        windows = self._process_2d_input(X)
        logits = self._batch_inference(windows)

        window_probs = torch.softmax(logits, dim=1).numpy()

        # Align probabilities to the end of the window
        probs[lookback - 1 :] = window_probs
        return probs

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))

    @property
    def expects_sequences(self) -> bool:
        return True

    @property
    def feature_importance(self) -> Any:
        # Neural Networks don't have built-in feature importance like RF.
        # We could implement Permutation Importance later.
        return {}
