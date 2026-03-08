# type: ignore
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import os

# map MetaTrader5 timeframes to human-readable labels for file naming
TIMEFRAME_ENUMS = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def save_market_data_to_csv(
    symbol: str,
    timeframe: str,
    from_date_utc: datetime,
    to_date_utc: datetime,
) -> str | None:
    """
    Fetches market data from MetaTrader5 and saves it to a CSV file.
    """

    target_dir = "data/raw_market"
    os.makedirs(target_dir, exist_ok=True)

    timeframe_enum = TIMEFRAME_ENUMS.get(timeframe.upper())

    if timeframe_enum is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    file_name = f"{symbol}_{timeframe.upper()}_{from_date_utc.strftime('%Y%m%d')}_{to_date_utc.strftime('%Y%m%d')}.csv"
    file_path = os.path.join(target_dir, file_name)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping data collection.")
        return file_path

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        mt5.shutdown()
        return

    try:
        rates = mt5.copy_rates_range(symbol, timeframe_enum, from_date_utc, to_date_utc)

        if rates is None or len(rates) == 0:
            print(f"Warning: No data found for {symbol} in this range.")
            return

        rates_df = pd.DataFrame(rates)

        # convert 'time' from seconds to datetime
        rates_df["time"] = pd.to_datetime(rates_df["time"], unit="s", utc=True)

        # save to CSV
        rates_df.to_csv(file_path, index=False)
        print(f"Successfully saved {len(rates_df)} bars to {file_path}")

        return file_path

    except Exception as e:
        print(f"An error occurred while fetching data for {symbol}: {e}")

    finally:
        mt5.shutdown()
