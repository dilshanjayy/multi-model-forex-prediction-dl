import torch
import torch.nn as nn
import math
from typing import Dict, Any
from src.models.base_torch_model import PyTorchBaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(PyTorchBaseModel):
    """
    State-of-the-art Transformer architecture for Forex Time-Series.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Hyperparameters
        input_dim = config.get("input_dim", 23) # Number of technical indicators
        d_model = config.get("d_model", 64)      # Internal embedding size
        nhead = config.get("nhead", 4)           # Number of attention heads
        num_layers = config.get("num_layers", 2) # Depth of the transformer
        dropout = config.get("dropout", 0.1)
        
        # 1. Input Projection (Linear Layer to match d_model size)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Output Head (3 classes: Up, Down, Deadband)
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, x):
        """
        Input x shape: (Batch, Seq_Len, Features)
        """
        # Project features to d_model space and normalize
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Add Positional information
        x = self.pos_encoder(x)
        
        # Pass through Transformer layers
        output = self.transformer_encoder(x)
        
        # We only care about the very last time step's output
        last_step_output = output[:, -1, :]
        
        # Predict class probabilities
        logits = self.fc_out(last_step_output)
        return logits
