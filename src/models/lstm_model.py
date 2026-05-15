import torch.nn as nn
from typing import Dict, Any
from src.models.base_torch_model import PyTorchBaseModel


class LSTMModel(PyTorchBaseModel):
    """
    Standard LSTM architecture for Forex Time-Series.
    Efficiently captures sequential dependencies and market momentum.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Hyperparameters
        input_dim = config.get("input_dim", 23)
        hidden_dim = config.get("hidden_dim", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)

        # LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 classes: Up (0), Down (1), Deadband (2)
        )

    def forward(self, x):
        """
        Input x shape: (Batch, Seq_Len, Features)
        """
        # LSTM returns: (output, (hn, cn))
        # output contains the hidden states for all time steps
        lstm_out, _ = self.lstm(x)

        last_step_output = lstm_out[:, -1, :]

        # Map to class logits
        logits = self.fc_out(last_step_output)
        return logits
