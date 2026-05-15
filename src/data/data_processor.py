import pandas as pd
import pandas_ta as ta
import numpy as np


def add_triple_barrier_labels(
    df: pd.DataFrame, horizon: int = 5, atr_multiplier: float = 3.0, unit: str = "h"
) -> pd.DataFrame:
    """
    Implements the Triple Barrier Method (TBM) using ATR.
    Labels: 0 (Profit), 1 (Loss), 2 (Time-out)
    """
    atr_col = "ATRr_14"
    if atr_col not in df.columns:
        raise ValueError(f"Required column {atr_col} not found in DataFrame.")

    labels = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df[atr_col].values

    for i in range(len(df)):
        if i + horizon >= len(df) or np.isnan(atrs[i]):
            labels.append(np.nan)
            continue

        entry_price = closes[i]
        current_atr = atrs[i]

        upper_barrier = entry_price + (current_atr * atr_multiplier)
        lower_barrier = entry_price - (current_atr * atr_multiplier)

        future_highs = highs[i + 1 : i + horizon + 1]
        future_lows = lows[i + 1 : i + horizon + 1]

        found = False
        for h, l in zip(future_highs, future_lows):  # noqa: E741
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

    col_name = f"Target_{horizon}{unit}_{atr_multiplier}x_TBM"
    df[col_name] = labels
    return df


def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data, computes technical indicators, and cleans the DataFrame.
    Following Nguyen et al. (2024) philosophy: Combining OHLCV returns + 14 indicators.
    """
    df = dataframe.copy()

    # --- DYNAMIC ALIGNMENT ---
    # Ensure index is sorted to prevent drift; preserves full history.
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True, drop=False)
    df.sort_index(inplace=True)
    # --------------------------

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

    if "real_volume" in df.columns:
        df.drop(columns=["real_volume"], inplace=True)

    # RELATIVE OHLCV (Essential for LSTM/CNN)
    df["Ret_Open"] = (df["Open"] - df["Open"].shift(1)) / df["Open"].shift(1)
    df["Ret_High"] = (df["High"] - df["Open"]) / df["Open"]
    df["Ret_Low"] = (df["Open"] - df["Low"]) / df["Open"]
    df["Ret_Close"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Ret_Vol"] = df["Volume"].pct_change()

    # NGUYEN ET AL. INDICATORS (14 total)
    df.ta.trix(append=True)
    df.ta.vwap(append=True)
    df.ta.mom(append=True)
    df.ta.roc(append=True)
    df.ta.rsi(append=True)
    df.ta.atr(append=True)
    df.ta.mfi(append=True)
    df.ta.efi(append=True)
    df.ta.bbands(append=True)
    df.ta.cci(append=True)
    df.ta.tsi(append=True)
    df.ta.stochrsi(append=True)
    df.ta.adx(append=True)
    df.ta.stoch(append=True)
    df.ta.chop(append=True)
    df.ta.er(append=True)
    df.ta.kc(append=True)
    df.ta.stc(append=True)

    # --- ADVANCED CONTEXT FEATURES ---
    # Volatility Normalization (Z-Score for Momentum)
    if "RSI_14" in df.columns:
        rolling_mean = df["RSI_14"].rolling(window=100).mean()
        rolling_std = df["RSI_14"].rolling(window=100).std()
        df["RSI_14_Z"] = (df["RSI_14"] - rolling_mean) / (rolling_std + 1e-8)
        df.drop(columns=["RSI_14"], inplace=True)

    # Cyclical Time (Session Context)
    hours = df["time"].dt.hour
    df["Hour_Sin"] = np.sin(2 * np.pi * hours / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * hours / 24)

    # Distance from "True Value" (Elasticity)
    if "VWAP_D" in df.columns:
        df["Dist_VWAP"] = (df["Close"] - df["VWAP_D"]) / df["VWAP_D"]

    ema_50 = ta.ema(df["Close"], length=50) # type: ignore
    if ema_50 is not None:
        df["Dist_EMA_50"] = (df["Close"] - ema_50) / ema_50

    # --- DEEP ALPHA UPGRADE ---
    # Multi-Timeframe Momentum
    df["Mom_D1"] = (df["Close"] - df["Close"].shift(24)) / df["Close"].shift(24)
    df["Mom_W1"] = (df["Close"] - df["Close"].shift(120)) / df["Close"].shift(120)

    # Institutional Baseline
    ema_200 = ta.ema(df["Close"], length=200) # type: ignore
    if ema_200 is not None:
        df["Dist_EMA_200"] = (df["Close"] - ema_200) / ema_200

    # Volatility-Adjusted Volume
    v_mean = df["Volume"].rolling(window=100).mean()
    v_std = df["Volume"].rolling(window=100).std()
    df["Vol_Z"] = (df["Volume"] - v_mean) / (v_std + 1e-8)
    # ---------------------------

    df.dropna(inplace=True)
    df = df[df["Volume"] > 0]
    df.ffill(inplace=True)

    df = purge_redundant_features(df)

    return df


def add_nguyen_labels(df: pd.DataFrame, horizon: int = 24, unit: str = "h") -> pd.DataFrame:
    """
    Implements the Nguyen et al. (2024) 3-class labeling method dynamically
    using a rolling window to prevent future data leakage.
    Top 35% of returns -> 0 (Up)
    Bottom 30% of returns -> 1 (Down)
    Middle 35% -> 2 (Unknown/Hold)
    """
    future_ret = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    past_rets = future_ret.shift(horizon)

    rolling_window = 1000
    thresh_down = past_rets.rolling(window=rolling_window, min_periods=100).quantile(0.30)
    thresh_up = past_rets.rolling(window=rolling_window, min_periods=100).quantile(0.65)

    thresh_down.bfill(inplace=True)
    thresh_up.bfill(inplace=True)

    labels = np.full(len(df), np.nan)
    labels[future_ret > thresh_up] = 0
    labels[future_ret <= thresh_down] = 1
    labels[(future_ret > thresh_down) & (future_ret <= thresh_up)] = 2

    col_name = f"Target_{horizon}{unit}_Nguyen"
    df[col_name] = labels
    return df


def generate_targets(
    df: pd.DataFrame,
    horizons: list = [5, 12, 24],
    atr_multipliers: list = [1.0, 2.0, 3.0],
    unit: str = "h"
) -> pd.DataFrame:
    """
    Adds raw future log returns and Triple Barrier labels to the DataFrame.
    """
    df = add_multi_horizon_log_returns(df, horizons=horizons, unit=unit)

    for h in horizons:
        df = add_nguyen_labels(df, horizon=h, unit=unit)
        for m in atr_multipliers:
            df = add_triple_barrier_labels(df, horizon=h, atr_multiplier=m, unit=unit)

    return df.dropna()


def split_components(df: pd.DataFrame) -> dict:
    """
    Splits the fully processed DataFrame into Feature Store components.
    """
    target_cols = [c for c in df.columns if "Target" in c or "LogRet" in c]
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
    df: pd.DataFrame, horizons: list = [5, 12, 24], unit: str = "h"
) -> pd.DataFrame:
    """
    Adds raw future log returns for multiple time horizons.
    """
    for h in horizons:
        col_name = f"LogRet_{h}{unit}"
        df[col_name] = np.log(df["Close"].shift(-h) / df["Close"])

    df.dropna(inplace=True)
    return df


def purge_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips highly correlated (>0.90) technical indicators to optimize
    the feature space for Conv1D kernels.
    """
    cols_to_drop = [
        "TRIXs_30_9",
        "TSIs_13_25_13",
        "STOCHRSId_14_14_3_3",
        "STOCHd_14_3_3",
        "MOM_10",
        "BBL_5_2.0_2.0",
        "BBU_5_2.0_2.0",
        "BBM_5_2.0_2.0",
        "KCLe_20_2",
        "KCUe_20_2",
        "ADXR_14_2",
        "spread",
    ]

    existing_cols = [col for col in df.columns if col in cols_to_drop]
    df.drop(columns=existing_cols, inplace=True)

    return df


def generate_nguyen_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT REPLICATION: Produces exactly the 19 features from Nguyen et al. (2024).
    """
    df = dataframe.copy()

    # --- DYNAMIC ALIGNMENT ---
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True, drop=False)
    df.sort_index(inplace=True)
    # --------------------------

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

    df.ta.trix(append=True)
    df.ta.vwap(append=True)
    df.ta.mom(append=True)
    df.ta.roc(append=True)
    df.ta.rsi(append=True)
    df.ta.atr(append=True)
    df.ta.mfi(append=True)
    df.ta.efi(append=True)
    df.ta.bbands(append=True)
    df.ta.cci(append=True)
    df.ta.tsi(append=True)
    df.ta.stochrsi(append=True)
    df.ta.adx(append=True)
    df.ta.stoch(append=True)

    bbm_cols = [c for c in df.columns if "BBM" in c]
    bbu_cols = [c for c in df.columns if "BBU" in c]
    bbl_cols = [c for c in df.columns if "BBL" in c]

    if bbm_cols and bbu_cols and bbl_cols:
        df["BBWidth"] = (df[bbu_cols[0]] - df[bbl_cols[0]]) / df[bbm_cols[0]]
    else:
        df["BBWidth"] = 0.0

    mapping = {
        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume",
        "TRIX": "TRIX_30_9",
        "VWAP": "VWAP_D",
        "MOM": "MOM_10",
        "ROC": "ROC_10",
        "RSI": "RSI_14",
        "ATR": "ATR_14",
        "MFI": "MFI_14",
        "EFI": "EFI_13",
        "BBWidth": "BBWidth",
        "CCI": "CCI_14_0.015",
        "TSI": "TSI_13_25_13",
        "STOCHRSIk": "STOCHRSIk_14_14_3_3",
        "ADX": "ADX_14",
        "STOCHk": "STOCHk_14_3_3"
    }

    final_cols = []
    for key, preferred in mapping.items():
        if preferred in df.columns:
            final_cols.append(preferred)
        else:
            matches = [c for c in df.columns if key in c]
            if matches:
                final_cols.append(matches[0])
            else:
                print(f"Warning: Could not find feature for {key}")

    df = df[final_cols + ["time"]]
    df.dropna(inplace=True)
    return df


FEATURE_PIPELINES = {
    "default": generate_features,
    "nguyen_2024": generate_nguyen_features,
}


def run_pipeline(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely dispatches to a named feature generation function.
    """
    pipeline_func = FEATURE_PIPELINES.get(name, FEATURE_PIPELINES["default"])
    if name not in FEATURE_PIPELINES:
        print(f"Warning: Feature pipeline '{name}' not found. Falling back to 'default'.")
    return pipeline_func(df)
