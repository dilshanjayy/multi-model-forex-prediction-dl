import argparse
import sys
import os
import pytz
from datetime import datetime
import pandas as pd

# import modular components
from src.data_collection import save_market_data_to_csv
from src.preprocessing.feature_engineering import engineer_technical_features


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Forex Prediction Pipeline"
    )

    # add subparsers for different pipeline commands
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")
    subparsers.required = True

    # subcommand: collect
    parser_collect = subparsers.add_parser("collect", help="Collect market data")
    parser_collect.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Currency pair symbol to process (e.g., EURUSD)",
    )
    parser_collect.add_argument(
        "--timeframe",
        type=str,
        default="H1",
        help="Timeframe for data collection (e.g., H1)",
    )
    parser_collect.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date for data collection (YYYY-MM-DD)",
    )
    parser_collect.add_argument(
        "--end",
        type=str,
        default="2026-01-31",
        help="End date for data collection (YYYY-MM-DD)",
    )

    # subcommand: process
    parser_process = subparsers.add_parser(
        "process", help="Process raw data into features"
    )
    parser_process.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing raw market data",
    )
    parser_process.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_market",
        help="Directory to save the processed CSV file with engineered features",
    )

    # execute the appropriate pipeline steps based on the command
    args = parser.parse_args()

    # collect
    if args.command in ["collect", "all"]:
        print("\n--- Starting Data Collection ---")

        # convert input dates to UTC datetime objects
        utc_tz = pytz.timezone("UTC")

        try:
            utc_from = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=utc_tz)
            utc_to = datetime.strptime(args.end, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=utc_tz
            )
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)

        print(
            f"Collecting {args.symbol} data from {utc_from.date()} to {utc_to.date()}..."
        )

        raw_path = save_market_data_to_csv(
            args.symbol, args.timeframe, utc_from, utc_to
        )

        if raw_path is None:
            print("Pipeline halted: Data collection failed.")
            sys.exit(1)

        print("--- Data Collection Step Complete ---")

    # process
    if args.command in ["process", "all"]:
        print("\n--- Starting Data Processing ---")
        print(
            f"Processing raw data from {args.input} and saving to {args.output_dir}..."
        )

        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)

        print("Reading raw data and processing features...")
        df = pd.read_csv(args.input, parse_dates=["time"])
        processed_df = engineer_technical_features(df)

        # ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # construct output file path
        base_name = os.path.basename(args.input)
        output_path = os.path.join(args.output_dir, f"processed_{base_name}")

        # save processed data to CSV
        processed_df.to_csv(output_path, index=False)
        print(f"Successfully saved processed data to {output_path}")
        print("--- Data Processing Step Complete ---")


if __name__ == "__main__":
    main()
