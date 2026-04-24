import argparse
import sys
import os
import pytz
import yaml
from datetime import datetime
import pandas as pd

# import modular components
from src.utils.reproducibility import set_seed
from src.data.market_data_collector import save_market_data_to_csv
from src.data.data_processor import (
    generate_features,
    generate_targets,
    split_components,
)
from src.data.data_module import DataModule
from src.models.baseline_trainer import run_baseline_training
from src.evaluation.backtester import run_backtest_session
from src.evaluation.optimizer import run_optimization_study


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    set_seed(42)
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
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 12, 24],
        help="List of horizons for Triple Barrier labeling (e.g., 5 12 24)",
    )
    parser_process.add_argument(
        "--atr-multipliers",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="List of ATR multipliers for Triple Barrier labeling (e.g., 1.0 2.0 3.0)",
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
        "--val-split-date",
        type=str,
        default="2024-01-01",
        help="Date to split train and val sets (YYYY-MM-DD)",
    )
    parser_train.add_argument(
        "--test-split-date",
        type=str,
        default="2025-01-01",
        help="Date to split val and test sets (YYYY-MM-DD)",
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

    # --- SUBCOMMAND: optimize ---
    parser_opt = subparsers.add_parser(
        "optimize", help="Run hyperparameter optimization via Optuna"
    )
    parser_opt.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser_opt.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run",
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

    # --- optimize ---
    if args.command == "optimize":
        run_optimization_study(args.config, n_trials=args.trials)

    # --- run ---
    if args.command == "run":
        config = load_config(args.config)
        print(f"\n--- Starting Experiment: {config['project']['name']} ---")

        # generate unique experiment name and directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{config['project']['name']}"
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
            horizons=config["data"].get("horizons", [5, 12, 24]),
            atr_multipliers=config["data"].get("atr_multipliers", [1.0, 2.0, 3.0]),
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
            config["data"].get("val_split_date", "2024-01-01"),
            config["data"].get("test_split_date", "2025-01-01"),
            experiment_dir,
            config=config,
        )

        # backtest
        if "backtest" in config:
            print("\n--- Step 3: Backtesting (Validation Set) ---")
            run_backtest_session(
                model_path=os.path.join(experiment_dir, "model.joblib"),
                processed_dir=processed_dir,
                start_date=config["data"].get("val_split_date", "2024-01-01"),
                end_date=config["data"].get("test_split_date", "2025-01-01"),
                strategy_name=config["backtest"]["strategy"],
                commission=config["backtest"].get("commission", 0.0001),
                cash=config["backtest"].get("cash", 10000.0),
                v_size=config["backtest"].get("v_size", 0.1),
                atr_multiplier=config["backtest"].get("atr_multiplier", 1.0),
                margin=config["backtest"].get("margin", 0.02),
                conf_threshold=config["backtest"].get("conf_threshold", 0.40),
                output_dir=experiment_dir,
                suffix="Validation",
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
        final_df = generate_targets(
            enriched_df,
            horizons=args.horizons,
            atr_multipliers=args.atr_multipliers,
        )
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
            args.input_dir,
            args.target,
            args.val_split_date,
            args.test_split_date,
            args.output_model,
        )
        print("--- Training Step Complete ---")

    # --- backtest ---
    if args.command == "backtest":
        config = load_config(args.config)
        print("\n--- Starting Standalone Backtest (Test Set) ---")

        # Save results in the same directory as the model
        output_dir = os.path.dirname(args.model)

        run_backtest_session(
            model_path=args.model,
            processed_dir=config["data"].get("processed_dir", "data/processed_market"),
            start_date=config["data"].get("test_split_date", "2025-01-01"),
            end_date=None,  # Run until the end of available data
            strategy_name=config["backtest"].get("strategy", "TripleBarrier"),
            commission=config["backtest"].get("commission", 0.0001),
            cash=config["backtest"].get("cash", 10000.0),
            v_size=config["backtest"].get("v_size", 0.1),
            atr_multiplier=config["backtest"].get("atr_multiplier", 1.0),
            margin=config["backtest"].get("margin", 0.02),
            conf_threshold=config["backtest"].get("conf_threshold", 0.40),
            output_dir=output_dir,
            suffix="TestSet_Final",
        )
        print(f"--- Backtest Step Complete. Results saved to: {output_dir} ---")


if __name__ == "__main__":
    main()
