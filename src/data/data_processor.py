import pandas as pd
import pandas_ta as ta
import numpy as np


def add_triple_barrier_labels(
    df: pd.DataFrame, horizon: int = 5, atr_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Implements the Triple Barrier Method (TBM) using ATR.
    Labels: 0 (Profit), 1 (Loss), 2 (Time-out)
    """
    # Ensure ATR column exists
    atr_col = "ATRr_14"
    if atr_col not in df.columns:
        raise ValueError(f"Required column {atr_col} not found in DataFrame.")

    labels = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df[atr_col].values

    # Triple Barrier Race
    for i in range(len(df)):
        if i + horizon >= len(df) or np.isnan(atrs[i]):
            labels.append(np.nan)
            continue

        entry_price = closes[i]
        current_atr = atrs[i]

        # Define barriers using ATR (Physical price distance)
        upper_barrier = entry_price + (current_atr * atr_multiplier)
        lower_barrier = entry_price - (current_atr * atr_multiplier)

        # Look ahead 'horizon' periods using High and Low
        future_highs = highs[i + 1 : i + horizon + 1]
        future_lows = lows[i + 1 : i + horizon + 1]

        found = False
        for h, l in zip(future_highs, future_lows):
            # Conservative trading rule: If both barriers are hit in the same hour, 
            # assume Stop Loss was hit first to prevent false optimism.
            if l <= lower_barrier:
                labels.append(1)  # Loss hit
                found = True
                break
            elif h >= upper_barrier:
                labels.append(0)  # Profit hit
                found = True
                break

        if not found:
            labels.append(2)  # Time-out

    # Final Labeling
    col_name = f"Target_{horizon}h_{atr_multiplier}x_TBM"
    df[col_name] = labels
    return df


def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data, computes technical indicators, and cleans the DataFrame.
    Following Nguyen et al. (2024) philosophy: Combining OHLCV returns + 14 indicators.
    """
    df = dataframe.copy()

    # --- NUCLEAR ALIGNMENT ---
    # We must align the STARTING POINT before any math happens.
    # This ensures recursive indicators (EMA, TSI) start from the exact same bar.
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True, drop=False)
    df.sort_index(inplace=True)
    df = df[df.index >= "2021-05-01"].copy() # Hard start 1 month before your 2021-06-01 goal
    # --------------------------

    # 1. Standardize column names
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        },
        inplace=True,
    )

    # 2. RELATIVE OHLCV (Essential for LSTM/CNN)
    # Nguyen et al. use OHLCV, but for H1 we must use relative returns to ensure stationarity
    df["Ret_Open"] = (df["Open"] - df["Open"].shift(1)) / df["Open"].shift(1)
    df["Ret_High"] = (df["High"] - df["Open"]) / df["Open"] # Wick size
    df["Ret_Low"] = (df["Open"] - df["Low"]) / df["Open"]   # Wick size
    df["Ret_Close"] = (df["Close"] - df["Open"]) / df["Open"] # Body size
    df["Ret_Vol"] = df["Volume"].pct_change()

    # 3. NGUYEN ET AL. INDICATORS (14 total)
    baseline_strategy = ta.Study( # type: ignore
        name="Nguyen_Baseline",
        cores=1, # FORCE single core to prevent non-deterministic Windows/Linux multi-processing bugs
        ta=[
            {"kind": "trix"},
            {"kind": "vwap"}, # Paper uses VWAMA, VWAP is more robust for H1
            {"kind": "mom"},
            {"kind": "roc"},
            {"kind": "rsi"},
            {"kind": "atr"},
            {"kind": "mfi"},
            {"kind": "efi"},
            {"kind": "bbands"},
            {"kind": "cci"},
            {"kind": "tsi"},
            {"kind": "stochrsi"},
            {"kind": "adx"},
            {"kind": "stoch"},
        ],
    )
    df.ta.study(baseline_strategy)

    # --- ADVANCED CONTEXT FEATURES ---
    # Fix A: Volatility Normalization (Z-Score for Momentum)
    # Tells the AI if an RSI of 70 is actually "abnormal" relative to the last 4 days (100 hours)
    if "RSI_14" in df.columns:
        rolling_mean = df["RSI_14"].rolling(window=100).mean()
        rolling_std = df["RSI_14"].rolling(window=100).std()
        df["RSI_14_Z"] = (df["RSI_14"] - rolling_mean) / (rolling_std + 1e-8)
        df.drop(columns=["RSI_14"], inplace=True) # Replace original

    # Fix B: Cyclical Time (Session Context)
    # Neural networks struggle with 0-23 hours. Sine/Cosine makes the transition from 23 to 0 continuous.
    hours = df["time"].dt.hour
    df["Hour_Sin"] = np.sin(2 * np.pi * hours / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * hours / 24)

    # Fix C: Distance from "True Value" (Elasticity)
    # How far has the price stretched away from the daily VWAP and the 50-hour trend?
    if "VWAP_D" in df.columns:
        df["Dist_VWAP"] = (df["Close"] - df["VWAP_D"]) / df["VWAP_D"]
    
    ema_50 = ta.ema(df["Close"], length=50)
    if ema_50 is not None:
        df["Dist_EMA_50"] = (df["Close"] - ema_50) / ema_50

    # --- DEEP ALPHA UPGRADE ---
    # Fix D: Multi-Timeframe Momentum (Context from Big Players)
    # Tells the AI if we are trading WITH or AGAINST the Daily/Weekly trend
    df["Mom_D1"] = (df["Close"] - df["Close"].shift(24)) / df["Close"].shift(24) # 1 Day
    df["Mom_W1"] = (df["Close"] - df["Close"].shift(120)) / df["Close"].shift(120) # 1 Week

    # Fix E: Institutional Baseline (The 'Global' Support/Resistance)
    ema_200 = ta.ema(df["Close"], length=200)
    if ema_200 is not None:
        df["Dist_EMA_200"] = (df["Close"] - ema_200) / ema_200

    # Fix F: Volatility-Adjusted Volume (Denoising Activity)
    v_mean = df["Volume"].rolling(window=100).mean()
    v_std = df["Volume"].rolling(window=100).std()
    df["Vol_Z"] = (df["Volume"] - v_mean) / (v_std + 1e-8)
    # ---------------------------

    # 4. Cleaning
    df.dropna(inplace=True)
    df = df[df["Volume"] > 0]
    df.ffill(inplace=True)

    # Final Crop to the user-requested start date
    df = df[df["time"] >= "2021-06-01"].copy()

    # 5. PURGE (Keep paper-specific features and engineered returns)
    df = purge_redundant_features(df)

    return df


def add_nguyen_labels(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    Implements the Nguyen et al. (2024) 3-class labeling method dynamically
    using a rolling window to prevent future data leakage (lookahead bias).
    Top 35% of returns -> 0 (Up)
    Bottom 30% of returns -> 1 (Down)
    Middle 35% -> 2 (Unknown/Hold)
    """
    # 1. Calculate future returns
    future_ret = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    
    # 2. To avoid lookahead bias, our quantiles must be based ONLY on returns
    # that have already "finished" by the current bar.
    # A return starting at t-horizon finished at t.
    past_rets = future_ret.shift(horizon)
    
    # 3. Calculate rolling thresholds (e.g., over the last 1000 bars ~ 2 months of H1)
    rolling_window = 1000
    
    # We use min_periods=100 so it can start labeling relatively early
    thresh_down = past_rets.rolling(window=rolling_window, min_periods=100).quantile(0.30)
    thresh_up = past_rets.rolling(window=rolling_window, min_periods=100).quantile(0.65)
    
    # For the very beginning of the dataset before min_periods, backfill to retain rows
    thresh_down.bfill(inplace=True)
    thresh_up.bfill(inplace=True)
    
    # 4. Apply dynamic labels
    labels = np.full(len(df), np.nan)
    labels[future_ret > thresh_up] = 0
    labels[future_ret <= thresh_down] = 1
    labels[(future_ret > thresh_down) & (future_ret <= thresh_up)] = 2
    
    col_name = f"Target_{horizon}h_Nguyen"
    df[col_name] = labels
    return df


def generate_targets(
    df: pd.DataFrame,
    horizons: list = [5, 12, 24],
    atr_multipliers: list = [1.0, 2.0, 3.0]
) -> pd.DataFrame:
    """
    Adds raw future log returns and Triple Barrier labels to the DataFrame.
    Generates a grid of targets for all combinations of horizons and multipliers.
    """
    # add multiple raw target horizons (e.g., 5, 12, 24 periods ahead)
    df = add_multi_horizon_log_returns(df, horizons=horizons)

    # add Triple Barrier Labeling and Nguyen Labeling for all combinations
    for h in horizons:
        # Add Nguyen et al. Quantile Labels
        df = add_nguyen_labels(df, horizon=h)
        
        # Add TBM Labels
        for m in atr_multipliers:
            df = add_triple_barrier_labels(df, horizon=h, atr_multiplier=m)

    return df.dropna()


def split_components(df: pd.DataFrame) -> dict:
    """
    Splits the fully processed DataFrame into Feature Store components.
    """
    target_cols = [c for c in df.columns if "Target" in c or "LogRet" in c]
    # Price columns can also be useful as metadata, but not as features
    metadata_cols = ["time", "Open", "High", "Low", "Close", "Volume"]

    feature_cols = [
        c for c in df.columns if c not in target_cols and c not in metadata_cols
    ]

    return {
        "technical_features": df[feature_cols + ["time"]],
        "targets": df[target_cols + ["time"]],
        "metadata": df[metadata_cols],
    }


def add_multi_horizon_log_returns(
    df: pd.DataFrame, horizons: list = [5, 12, 24]
) -> pd.DataFrame:
    """
    Adds raw future log returns for multiple time horizons.
    The final classification labeling (quantiles) should be done
    during training to avoid data leakage and allow for flexible tuning.
    """
    for h in horizons:
        col_name = f"LogRet_{h}h"
        df[col_name] = np.log(df["Close"].shift(-h) / df["Close"])

    # drop the last rows where we don't have future values for the longest horizon
    df.dropna(inplace=True)

    return df


def purge_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    strips highly correlated (>0.90) technical indicators to optimize
    the feature space for Conv1D kernels.
    """
    cols_to_drop = [
        "TRIXs_30_9",  # TRIX signal line
        "TSIs_13_25_13",  # TSI signal line
        "STOCHRSId_14_14_3_3",  # StochRSI %D line
        "STOCHd_14_3_3",  # Stoch %D line
        "MOM_10",  # 1.0 correlation with ROC
        "BBL_5_2.0_2.0",  # 1.0 correlation with BBM
        "BBU_5_2.0_2.0",  # 1.0 correlation with BBM
        "BBM_5_2.0_2.0",  # 1.0 correlation with VWAP_D
        "KCLe_20_2",
        "KCUe_20_2",
        "KCMe_20_2",  # Purge price-level Keltner components
        "ADXR_14_2",  # 0.99 correlation with ADX
        "spread",  # not a technical indicator and may introduce noise
    ]

    existing_cols = [col for col in df.columns if col in cols_to_drop]
    df.drop(columns=existing_cols, inplace=True)

    return df
