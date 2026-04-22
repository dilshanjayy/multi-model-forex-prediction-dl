# type: ignore
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.data_processor import generate_features

def fetch_live_data(symbol: str, timeframe_str: str, count: int = 300) -> pd.DataFrame | None:
    """
    Fetches the latest N bars from MT5 and returns an engineered DataFrame.
    If MT5 is not available (e.g., Linux/Colab), falls back to the most recent Parquet data.
    """
    if mt5 is None or not mt5.initialize():
        print("MT5 Initialization failed or unsupported OS. Falling back to local Parquet data.")
        try:
            import os
            processed_dir = "data/processed_market"
            features = pd.read_parquet(os.path.join(processed_dir, "technical_features.parquet"))
            metadata = pd.read_parquet(os.path.join(processed_dir, "metadata.parquet"))
            if "time" in metadata.columns:
                metadata.set_index("time", inplace=True)
            df = features.join(metadata, how="inner")
            # Reset index to match MT5 format where 'time' is a column before generate_features
            df = df.reset_index()
            # If we don't need to run generate_features (since it's already engineered):
            return df.tail(count)
        except Exception as e:
            print(f"Fallback failed: {e}")
            return None

    # Map timeframe string to MT5 enum
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    tf = tf_map.get(timeframe_str.upper())
    if tf is None:
        return None

    # Fetch last 'count' bars
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    # Convert broker time (seconds) to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Ensure it is treated as UTC (MT5 server time is usually UTC+2/3, 
    # but the API allows us to treat it as UTC for consistent charting)
    df["time"] = df["time"].dt.tz_localize('UTC')
    
    # Feature Engineering in-memory
    enriched_df = generate_features(df)
    
    return enriched_df
