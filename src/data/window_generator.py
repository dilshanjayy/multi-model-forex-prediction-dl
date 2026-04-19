import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class ForexWindowDataset(Dataset):
    """
    Professional PyTorch Dataset for Time-Series Windowing.
    Converts 2D DataFrames into 3D Tensors: (Samples, Lookback, Features).
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, lookback: int):
        self.x = torch.from_numpy(x_data.copy()).float()
        self.y = torch.from_numpy(y_data.copy()).long()
        self.lookback = lookback

    def __len__(self):
        return len(self.x) - self.lookback + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract a window of 'lookback' steps
        x_window = self.x[idx : idx + self.lookback]
        # The target is the label at the VERY LAST step of the window
        y_label = self.y[idx + self.lookback - 1]
        return x_window, y_label

class TimeSeriesWindowGenerator:
    """
    Orchestrator for creating PyTorch DataLoaders from the Feature Store.
    Handles scaling, splitting, and sequential windowing.
    """
    def __init__(
        self, 
        lookback: int = 60, 
        batch_size: int = 64,
        feature_cols: List[str] = []
    ):
        self.lookback = lookback
        self.batch_size = batch_size
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()

    def prepare_loaders(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        target_col: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Fits scaler on training data and returns DataLoader triplets.
        """
        # 1. Feature Selection & Sorting (Ensure Order Locking)
        self.feature_cols.sort()
        
        # 2. Fit and Transform Scaling
        # We fit ONLY on training data to prevent leakage
        x_train = self.scaler.fit_transform(train_df[self.feature_cols])
        x_val = self.scaler.transform(val_df[self.feature_cols])
        x_test = self.scaler.transform(test_df[self.feature_cols])

        # 3. Extract Targets
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # 4. Create PyTorch Datasets
        train_ds = ForexWindowDataset(x_train, y_train, self.lookback)
        val_ds = ForexWindowDataset(x_val, y_val, self.lookback)
        test_ds = ForexWindowDataset(x_test, y_test, self.lookback)

        # 5. Create DataLoaders
        # Shuffle=True for training to break temporal correlation in batches
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        print(f"DataLoaders Ready: Train ({len(train_ds)}), Val ({len(val_ds)}), Test ({len(test_ds)}) windows.")
        return train_loader, val_loader, test_loader
