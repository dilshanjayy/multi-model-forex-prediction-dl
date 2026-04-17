import argparse
import sys
import os
import pytz
import yaml
from datetime import datetime
import pandas as pd

# import modular components
from src.data.market_data_collector import save_market_data_to_csv
from src.data.data_processor import (
    generate_features,
    generate_targets,
    split_components,
)
from src.data.data_module import DataModule
from src.models.baseline_trainer import run_baseline_training
from src.evaluation.backtester import run_backtest_session


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Forex Prediction Pipeline"
    )

    # add subparsers for different pipeline commands
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # --- SUBCOMMAND: collect ---
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

    # --- SUBCOMMAND: process ---
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
        help="Directory to save the processed Parquet components",
    )
    parser_process.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Horizon for Triple Barrier labeling (e.g., 5 for 5h)",
    )

    # --- SUBCOMMAND: train ---
    parser_train = subparsers.add_parser("train", help="Train a baseline model")
    parser_train.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing processed Parquet components",
    )
    parser_train.add_argument(
        "--target",
        type=str,
        default="Target_5h_TBM",
        choices=["Target_5h_TBM", "LogRet_5h", "LogRet_12h", "LogRet_24h"],
        help="The target column to use for labeling",
    )
    parser_train.add_argument(
        "--split-date",
        type=str,
        default="2025-01-01",
        help="Date to split train and test sets (YYYY-MM-DD)",
    )
    parser_train.add_argument(
        "--output-model",
        type=str,
        default="src/models/baseline_rf.joblib",
        help="Path to save the trained model artifacts",
    )

    # --- SUBCOMMAND: backtest ---
    parser_bt = subparsers.add_parser("backtest", help="Backtest a trained model")
    parser_bt.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model.joblib artifact",
    )
    parser_bt.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config.yaml",
    )

    # --- SUBCOMMAND: run ---
    parser_exp = subparsers.add_parser(
        "run", help="Run an experiment from a config file"
    )
    parser_exp.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )

    # execute the appropriate pipeline steps based on the command
    args = parser.parse_args()

    # if no command provided, print help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # --- run ---
    if args.command == "run":
        config = load_config(args.config)
        print(f"\n--- Starting Experiment: {config["project"]["name"]} ---")

        # generate unique experiment name and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{config["project"]["name"]}"
        experiment_dir = os.path.join("experiments", exp_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # process
        print("\n--- Step 1: Data Processing ---")
        raw_input = config["data"]["raw_input"]
        if not os.path.exists(raw_input):
            print(f"Error: Raw input {raw_input} not found.")
            sys.exit(1)

        df = pd.read_csv(raw_input, parse_dates=["time"])
        
        # Modular Pipeline
        enriched_df = generate_features(df)
        final_df = generate_targets(
            enriched_df, 
            horizon=config['data']['horizon'],
            atr_multiplier=config['data'].get('atr_multiplier', 3.0)
        )
        feature_dict = split_components(final_df)
        
        # Save modular components as Parquet
        processed_dir = config["data"]["processed_dir"]
        DataModule.save_features(feature_dict, processed_dir)

        # train
        print("\n--- Step 2: Model Training ---")
        run_baseline_training(
            processed_dir,
            config["model"]["target"],
            config["data"]["split_date"],
            experiment_dir,
            config=config,
        )

        # backtest
        if "backtest" in config:
            print("\n--- Step 3: Backtesting ---")
            run_backtest_session(
                model_path=os.path.join(experiment_dir, "model.joblib"),
                processed_dir=processed_dir,
                split_date=config["data"]["split_date"],
                strategy_name=config["backtest"]["strategy"],
                commission=config["backtest"].get("commission", 0.0001),
                cash=config["backtest"].get("cash", 10000.0),
                v_size=config["backtest"].get("v_size", 0.1),
                atr_multiplier=config["backtest"].get("atr_multiplier", 1.0)
            )

        print(f"\n--- Experiment {config['project']['name']} Complete ---")
        print(f"All artifacts are in: {experiment_dir}")
        return

    # --- collect ---
    if args.command == "collect":
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

    # --- process ---
    if args.command == "process":
        print("\n--- Starting Data Processing ---")
        print(
            f"Processing raw data from {args.input} and saving to {args.output_dir}..."
        )

        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)

        print("Reading raw data and processing features...")
        df = pd.read_csv(args.input, parse_dates=["time"])
        
        # Modular Pipeline
        enriched_df = generate_features(df)
        final_df = generate_targets(enriched_df, horizon=args.horizon)
        feature_dict = split_components(final_df)

        # Save modular components as Parquet
        DataModule.save_features(feature_dict, args.output_dir)
        print(f"Successfully saved processed components to {args.output_dir}")
        print("--- Data Processing Step Complete ---")

    # --- train ---
    if args.command == "train":
        print("\n--- Starting Baseline Training ---")
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist.")
            sys.exit(1)

        run_baseline_training(
            args.input_dir, args.target, args.split_date, args.output_model
        )
        print("--- Training Step Complete ---")

    # --- backtest ---
    if args.command == "backtest":
        config = load_config(args.config)
        print(f"\n--- Starting Standalone Backtest ---")
        
        run_backtest_session(
            model_path=args.model,
            processed_dir=config["data"].get("processed_dir", "data/processed_market"),
            split_date=config["data"].get("split_date", "2025-01-01"),
            strategy_name=config["backtest"].get("strategy", "TripleBarrier"),
            commission=config["backtest"].get("commission", 0.0001),
            cash=config["backtest"].get("cash", 10000.0),
            v_size=config["backtest"].get("v_size", 0.1),
            atr_multiplier=config["backtest"].get("atr_multiplier", 1.0)
        )
        print("--- Backtest Step Complete ---")


if __name__ == "__main__":
    main()
