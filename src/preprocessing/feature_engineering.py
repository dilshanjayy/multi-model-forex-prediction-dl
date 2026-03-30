import pandas as pd
import pandas_ta as ta


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

    return df
