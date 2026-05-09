# type: ignore
import MetaTrader5 as mt5
import pandas as pd
from src.data.data_processor import run_pipeline

_MT5_INITIALIZED = False

def fetch_live_data(
    symbol: str, timeframe_str: str, count: int = 300, pipeline_name: str = "default"
) -> pd.DataFrame | None:
    """
    Fetches the latest N bars from MT5 and returns an engineered DataFrame.
    Returns None if MT5 is not available or if the market is closed (no rates fetched).
    """
    global _MT5_INITIALIZED
    if not _MT5_INITIALIZED:
        if mt5 is None or not mt5.initialize():
            print("MT5 Initialization failed or unsupported OS.")
            return None
        _MT5_INITIALIZED = True

    # Map timeframe string to MT5 enum
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
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

    # 1. Shift broker time back 7 hours to get the equivalent New York time
    df["time"] = df["time"] - pd.Timedelta(hours=7)

    # 2. Localize as US/Eastern (which handles US DST perfectly)
    df["time"] = df["time"].dt.tz_localize("US/Eastern", ambiguous='infer')

    # 3. Convert to true UTC to match model training features
    df["time"] = df["time"].dt.tz_convert('UTC')

    # Feature Engineering in-memory using named pipeline
    enriched_df = run_pipeline(pipeline_name, df)

    return enriched_df
