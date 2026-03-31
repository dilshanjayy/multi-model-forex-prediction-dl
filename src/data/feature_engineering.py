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

    # create target labels using the Fixed-Time Horizon labeling approach
    df = create_labels(df, future_periods=5)

    return df


def create_labels(df: pd.DataFrame, future_periods: int = 5) -> pd.DataFrame:
    """
    Implements Fixed-Time Horizon labeling using dynamic quantiles.
    Class 0: Up Trend (Top 35% of future returns)
    Class 1: Down Trend (Bottom 30% of future returns)
    Class 2: Unknown / Deadband (Middle 35% of future returns)
    """

    # calculate future log returns
    df["Future_Log_Return"] = np.log(df["Close"].shift(-future_periods) / df["Close"])
    df.dropna(inplace=True)

    # calculate dynamic quantiles for labeling
    lower_threshold = df["Future_Log_Return"].quantile(0.30)
    upper_threshold = df["Future_Log_Return"].quantile(0.65)

    # assign classes based on quantile thresholds
    conditions = [
        df["Future_Log_Return"] > upper_threshold,  # Class 0: Up Trend
        df["Future_Log_Return"] < lower_threshold,  # Class 1: Down Trend
    ]
    choices = [0, 1]

    # Class 2 (Unknown / Deadband) is assigned by default to any returns that fall between the upper and lower thresholds
    df["Target"] = np.select(
        conditions, choices, default=2
    )  # Class 2: Unknown / Deadband

    # drop the Future_Log_Return column as it's no longer needed for modeling
    df.drop(columns=["Future_Log_Return"], inplace=True)

    # remove redundant features based on correlation analysis
    df = purge_redundant_features(df)

    return df


def purge_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips highly correlated (>0.90) technical indicators to optimize
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
        "spread",  # not a technical indicator and may introduce noise due to broker-specific spread variations
    ]

    existing_cols = [col for col in df.columns if col in cols_to_drop]
    df.drop(columns=existing_cols, inplace=True)

    return df
