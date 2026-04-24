import torch
import torch.nn as nn
from typing import Dict, Any
from src.models.base_torch_model import PyTorchBaseModel

class CNNLSTMModel(PyTorchBaseModel):
    """
    Hybrid CNN-LSTM architecture inspired by Nguyen et al. (2024).
    Uses 1D Convolutions for pattern extraction and LSTM for temporal sequences.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Hyperparameters
        input_dim = config.get("input_dim", 23)
        cnn_filters_1 = config.get("cnn_filters_1", 32)
        cnn_filters_2 = config.get("cnn_filters_2", 64)
        lstm_units = config.get("lstm_units", 50)
        kernel_size = config.get("kernel_size", 3)
        dropout = config.get("dropout", 0.2)
        
        # 1. CNN Stage (Feature Extraction)
        # We use Conv1d to scan for local price patterns
        self.cnn_block = nn.Sequential(
            # Input: (Batch, Features, Time)
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters_1, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_filters_1, out_channels=cnn_filters_2, kernel_size=kernel_size, padding=1),
            nn.ReLU()
        )
        
        # 2. LSTM Stage (Sequence Tracking)
        self.lstm = nn.LSTM(
            input_size=cnn_filters_2,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0 # num_layers is 1
        )
        
        # 3. Output Head
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, 3) # 3 classes: Up, Down, Neutral
        )

    def forward(self, x):
        """
        Input x shape: (Batch, Seq_Len, Features)
        """
        # CNN expects (Batch, Channels, Time), so we transpose
        # From: (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.transpose(1, 2)
        
        # CNN Feature Extraction
        x = self.cnn_block(x)
        
        # Reshape back for LSTM: (Batch, Features, Time) -> (Batch, Time, Features)
        x = x.transpose(1, 2)
        
        # LSTM Temporal Tracking
        lstm_out, _ = self.lstm(x)
        
        # We only care about the very last time step
        last_step = lstm_out[:, -1, :]
        
        # Map to class logits
        logits = self.fc_out(last_step)
        return logits
