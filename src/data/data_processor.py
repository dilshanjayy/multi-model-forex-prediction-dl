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
    prices = df["Close"].values
    atrs = df[atr_col].values

    # Triple Barrier Race
    for i in range(len(df)):
        if i + horizon >= len(df) or np.isnan(atrs[i]):
            labels.append(np.nan)
            continue

        entry_price = prices[i]
        current_atr = atrs[i]

        # Define barriers using ATR (Physical price distance)
        upper_barrier = entry_price + (current_atr * atr_multiplier)
        lower_barrier = entry_price - (current_atr * atr_multiplier)

        # Look ahead 'horizon' periods
        future_prices = prices[i + 1 : i + horizon + 1]

        found = False
        for p in future_prices:
            if p >= upper_barrier:
                labels.append(0)  # Profit hit
                found = True
                break
            elif p <= lower_barrier:
                labels.append(1)  # Loss hit
                found = True
                break

        if not found:
            labels.append(2)  # Time-out

    # Final Labeling
    df["Target"] = labels
    return df.dropna()


def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data, computes technical indicators, and cleans the DataFrame.
    Returns an enriched DataFrame ready for labeling.
    """
    df = dataframe.copy()

    # standardize column names for pandas_ta
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

    # ensure time column is in datetime format and set as index
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True, drop=False)
    df.sort_index(inplace=True)

    # drop real_volume if it exists, as it's not needed for modeling
    if "real_volume" in df.columns:
        df.drop(columns=["real_volume"], inplace=True)

    # strip any existing dead hours recorded by the broker
    df = df[df["Volume"] > 0]

    # forward-fill any internal NaNs just in case a price packet dropped but volume registered
    df.ffill(inplace=True)

    # define a comprehensive set of technical indicators to compute
    baseline_strategy = ta.Study(  # type: ignore
        name="Tech_Baseline",
        cores=0,
        ta=[
            {"kind": "trix"},
            {"kind": "vwap"},
            {"kind": "mom"},
            {"kind": "roc"},
            {"kind": "rsi"},
            {"kind": "atr"},
            {"kind": "mfi"},
            {"kind": "efi"},
            {"kind": "bbands"},
            {"kind": "kc"},  # Keltner Channels
            {"kind": "cci"},
            {"kind": "tsi"},
            {"kind": "stochrsi"},
            {"kind": "adx"},
            {"kind": "stoch"},
            {"kind": "chop"},  # Choppiness Index
            {"kind": "stc"},  # Schaff Trend Cycle
            {"kind": "er"},  # Efficiency Ratio
        ],
    )

    df.ta.study(baseline_strategy)

    # drop rows with any NaN values that may have been introduced by the indicators
    df.dropna(inplace=True)

    # remove redundant features based on correlation analysis
    df = purge_redundant_features(df)

    return df


def generate_targets(
    df: pd.DataFrame, horizon: int = 5, atr_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Adds raw future log returns and Triple Barrier labels to the DataFrame.
    """
    # add multiple raw target horizons (e.g., 5, 12, 24 periods ahead)
    df = add_multi_horizon_log_returns(df, horizons=[5, 12, 24])

    # add Triple Barrier Labeling for specified horizon
    df = add_triple_barrier_labels(df, horizon=horizon, atr_multiplier=atr_multiplier)

    return df


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
        "MOM_10",  # 1.00 correlation with ROC
        "BBL_5_2.0_2.0",  # 1.00 correlation with BBM
        "BBU_5_2.0_2.0",  # 1.00 correlation with BBM
        "BBM_5_2.0_2.0",  # 1.00 correlation with VWAP_D
        "KCLe_20_2",
        "KCUe_20_2",
        "KCMe_20_2",  # Purge price-level Keltner components
        "ADXR_14_2",  # 0.99 correlation with ADX
        "spread",  # not a technical indicator and may introduce noise
    ]

    existing_cols = [col for col in df.columns if col in cols_to_drop]
    df.drop(columns=existing_cols, inplace=True)

    return df
