import pandas as pd
import pandas_ta as ta
import numpy as np


def engineer_technical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
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
            {"kind": "cci"},
            {"kind": "tsi"},
            {"kind": "stochrsi"},
            {"kind": "adx"},
            {"kind": "stoch"},
        ],
    )

    df.ta.study(baseline_strategy)

    # drop rows with any NaN values that may have been introduced by the indicators
    df.dropna(inplace=True)

    # remove redundant features based on correlation analysis
    df = purge_redundant_features(df)

    # add multiple raw target horizons (e.g., 5, 12, 24 periods ahead)
    # this saves the 'Fact' (raw return) instead of the 'Opinion' (class labels)
    df = add_multi_horizon_targets(df, horizons=[5, 12, 24])

    return df


def add_multi_horizon_targets(
    df: pd.DataFrame, horizons: list = [5, 12, 24]
) -> pd.DataFrame:
    """
    Adds raw future log returns for multiple time horizons.
    The final classification labeling (quantiles) should be done
    during training to avoid data leakage and allow for flexible tuning.
    """
    for h in horizons:
        col_name = f"Target_{h}h_Return"
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
        "ADXR_14_2",  # 0.99 correlation with ADX
        "spread",  # not a technical indicator and may introduce noise
    ]

    existing_cols = [col for col in df.columns if col in cols_to_drop]
    df.drop(columns=existing_cols, inplace=True)

    return df
