import os
import argparse
import sys

# Add project root to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backtesting import Backtest
from src.data.data_module import DataModule
from src.strategies.base_strategies import (
    NaiveFlipStrategy,
    MajorityVoteStrategy,
    TripleBarrierStrategy,
)


def run_backtest_session(
    model_path: str,
    processed_dir: str,
    split_date: str,
    strategy_name: str,
    commission: float = 0.0001,
    cash: float = 10000.0,
    v_size: float = 0.1,
    atr_multiplier: float = 1.0,
):
    """
    Generic runner for the 'Model Tournament'.
    Loads data via DataModule, selects strategy, and runs the Backtest.
    """
    print("\n--- BACKTEST EXECUTION ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Strategy: {strategy_name}")
    print(f"Fee (epsilon): {commission / 0.0001:.1f} pips")

    # 1. Load Data via DataModule
    print(f"Loading modular data from {processed_dir}...")
    dm = DataModule(processed_dir)
    # Join features and metadata (price) for the backtester
    df = dm.prepare_dataset(components=["technical_features", "metadata"])
    df.sort_index(inplace=True)

    # 2. Filter for Test Period
    test_df = df[df.index >= split_date].copy()
    if len(test_df) == 0:
        print("Error: No test data found.")
        return

    # 3. Strategy Mapping
    strategies = {
        "NaiveFlip": NaiveFlipStrategy,
        "MajorityVote": MajorityVoteStrategy,
        "TripleBarrier": TripleBarrierStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Strategy {strategy_name} not found.")

    selected_strategy = strategies[strategy_name]

    # 4. Execute Backtest
    bt = Backtest(
        test_df,
        selected_strategy,
        cash=cash,
        commission=commission,
        margin=0.02,
        trade_on_close=False,
    )

    stats = bt.run(model_path=model_path, v_size=v_size, atr_multiplier=atr_multiplier)
    print("\n--- RESULTS ---")
    print(stats)

    # 5. Save Report
    report_name = (
        f"experiments/report_{strategy_name}_{os.path.basename(model_path)}.html"
    )
    bt.plot(filename=report_name, open_browser=False)
    print(f"\nReport saved to: {report_name}")

    return stats


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Professional Model Backtester")
    parser.add_argument("--model", type=str, help="Path to the model.joblib artifact")
    parser.add_argument("--config", type=str, help="Path to the experiment config.yaml")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the directory containing processed Parquet components",
    )
    parser.add_argument("--split-date", type=str, help="Date to split train and test sets")
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name",
        choices=["NaiveFlip", "MajorityVote", "TripleBarrier"],
    )

    args = parser.parse_args()

    # Default values
    model_path = args.model
    data_dir = args.data_dir or "data/processed_market"
    split_date = args.split_date or "2025-01-01"
    strategy = args.strategy or "TripleBarrier"
    epsilon = 0.0001
    v_size = 10000.0
    atr_mult = 3.0

    # Override with config if provided
    if args.config:
        print(f"Loading parameters from config: {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            data_dir = config["data"].get("processed_dir", data_dir)
            split_date = config["data"].get("split_date", split_date)
            if "backtest" in config:
                strategy = config["backtest"].get("strategy", strategy)
                epsilon = config["backtest"].get("commission", epsilon)
                v_size = config["backtest"].get("v_size", v_size)
                atr_mult = config["backtest"].get("atr_multiplier", atr_mult)

    if model_path and os.path.exists(model_path):
        run_backtest_session(
            model_path=model_path,
            processed_dir=data_dir,
            split_date=split_date,
            strategy_name=strategy,
            commission=epsilon,
            v_size=v_size,
            atr_multiplier=atr_mult,
        )
    else:
        print("Error: Model path is required and must exist. Use --model <path>")
