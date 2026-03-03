import talib
import pandas as pd
import numpy as np


def engineer_technical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
    """
    df = dataframe.copy()

    # ensure data is sorted by time
    df = df.sort_values("time").reset_index(drop=True)

    # numpy arrays for TA-Lib
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    # trend indicators
    df["EMA_20"] = talib.EMA(close, timeperiod=20)

    # momentum indicators
    df["RSI_14"] = talib.RSI(close, timeperiod=14)

    # volatility indicators
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)

    # target variable: next period's close price
    df["target"] = df["close"].shift(-1)

    # drop rows with NaN values (due to indicator calculations)
    clean_df = df.dropna().copy()

    return clean_df


