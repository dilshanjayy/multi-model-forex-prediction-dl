import os
import argparse
import sys
import json
import yaml
import pandas as pd

# Add project root to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backtesting import Backtest
from src.data.data_module import DataModule
from src.strategies.base_strategies import (
    ContinuousSignalExecutionStrategy,
    MajorityVoteStrategy,
    TripleBarrierStrategy,
)


def run_backtest_session(
    model_path: str,
    processed_dir: str,
    start_date: str,
    end_date: str | None,
    strategy_name: str,
    commission: float = 0.0001,
    cash: float = 10000.0,
    v_size: float = 0.1,
    atr_multiplier: float = 1.0,
    margin: float = 0.02,
    conf_threshold: float = 0.40,
    output_dir: str | None = None,
    suffix: str = "",
    config: dict | None = None,
):
    """
    Generic runner for the 'Model Tournament'.
    Loads data via DataModule, selects strategy, and runs the Backtest.
    """
    print(f"\n--- BACKTEST EXECUTION ({suffix if suffix else 'Custom'}) ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Strategy: {strategy_name}")
    print(f"Fee (epsilon): {commission / 0.0001:.1f} pips")
    print(f"Confidence Threshold: {conf_threshold:.2f}")

    # 1. Load Data via DataModule
    print(f"Loading modular data from {processed_dir}...")
    dm = DataModule(processed_dir)
    # Join features and metadata (price) for the backtester
    df = dm.prepare_dataset(components=["technical_features", "metadata"])

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    # 2. Filter for Test Period
    if config and "train_split_pct" in config.get("data", {}):
        train_pct = config["data"]["train_split_pct"]
        val_pct = config["data"].get("val_split_pct", (1.0 - train_pct) / 2)
        n_samples = len(df)
        train_end = int(n_samples * train_pct)
        val_end = int(n_samples * (train_pct + val_pct))

        if suffix == "Validation":
            test_df = df.iloc[train_end:val_end].copy()
            print("Filtering data by percentage (Validation Set)...")
        else:
            test_df = df.iloc[val_end:].copy()
            print("Filtering data by percentage (Test Set)...")
    else:
        # Ensure start_date and end_date are timezone-aware to match the index
        print(
            f"Filtering data from {start_date} to {end_date if end_date else 'End'}..."
        )

        # Convert string dates to UTC-aware Timestamps and ensure normalized precision
        start_ts = pd.to_datetime(start_date, utc=True).floor("s")
        end_ts = pd.to_datetime(end_date, utc=True).floor("s") if end_date else None

        if end_ts:
            test_df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()
        else:
            test_df = df[df.index >= start_ts].copy()

    print(f"Backtest dataset size: {len(test_df)} bars")
    if len(test_df) == 0:
        print(
            f"Error: No data found for the specified range. Index Range: {df.index[0]} to {df.index[-1]}"
        )
        return
    # 3. Strategy Mapping
    strategies = {
        "ContinuousSignalExecution": ContinuousSignalExecutionStrategy,
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
        margin=margin,
        trade_on_close=False,
    )

    stats = bt.run(
        model_path=model_path,
        v_size=v_size,
        atr_multiplier=atr_multiplier,
        conf_threshold=conf_threshold,
    )

    print("\n--- RESULTS ---")
    print(stats)

    # 5. Save Results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save HTML Report
        report_path = os.path.join(output_dir, f"report_{strategy_name}_{suffix}.html")
        bt.plot(filename=report_path, open_browser=False)

        # Save Numeric Stats to JSON (clean strings for serializability)
        clean_stats = {
            str(k): str(v) if not isinstance(v, (int, float, list, dict)) else v
            for k, v in stats.items()
            if isinstance(k, str) and not k.startswith("_")
        }
        stats_path = os.path.join(output_dir, f"stats_{suffix.lower()}.json")
        with open(stats_path, "w") as f:
            json.dump(clean_stats, f, indent=4)

        print(f"Artifacts saved: {report_path} and {stats_path}")

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
    parser.add_argument("--start-date", type=str, help="Start date for the backtest")
    parser.add_argument(
        "--end-date", type=str, help="End date for the backtest (optional)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name",
        choices=["ContinuousSignalExecution", "MajorityVote", "TripleBarrier"],
    )

    args = parser.parse_args()

    # Default values
    model_path = args.model
    data_dir = args.data_dir or "data/processed_market"
    start_date = args.start_date or "2024-01-01"
    end_date = args.end_date or "2025-01-01"
    strategy = args.strategy or "TripleBarrier"
    epsilon = 0.0001
    v_size = 10000.0
    atr_mult = 3.0
    margin = 0.02

    # Override with config if provided
    if args.config:
        print(f"Loading parameters from config: {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            if "data" in config:
                data_dir = config["data"].get("processed_dir", data_dir)
                # Only use config dates if CLI didn't provide them
                if not args.start_date:
                    start_date = config["data"].get("val_split_date", start_date)
                if not args.end_date:
                    end_date = config["data"].get("test_split_date", end_date)
            if "backtest" in config:
                strategy = config["backtest"].get("strategy", strategy)
                epsilon = config["backtest"].get("commission", epsilon)
                v_size = config["backtest"].get("v_size", v_size)
                atr_mult = config["backtest"].get("atr_multiplier", atr_mult)
                margin = config["backtest"].get("margin", margin)
    if model_path and os.path.exists(model_path):
        run_backtest_session(
            model_path=model_path,
            processed_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy,
            commission=epsilon,
            v_size=v_size,
            atr_multiplier=atr_mult,
        )
    else:
        print("Error: Model path is required and must exist. Use --model <path>")
