import MetaTrader5 as mt5
import pytz
from datetime import datetime
from src.data_collection.market_data_collector import save_market_data_to_csv

utc_tz = pytz.timezone("UTC")
utc_from = datetime(2020, 1, 1, 0, 0, tzinfo=utc_tz)
utc_to = datetime(2026, 1, 31, 23, 59, tzinfo=utc_tz)

symbols = ["EURUSD", "GBPUSD"]

# collect hourly data for each symbol in the specified date range
for symbol in symbols:
    save_market_data_to_csv(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
