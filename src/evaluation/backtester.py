import pandas as pd
import os
import argparse
import sys

# Add project root to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backtesting import Backtest
from src.strategies.base_strategies import (
    NaiveFlipStrategy,
    MajorityVoteStrategy,
    TripleBarrierStrategy,
)


def run_backtest_session(
    model_path: str,
    data_path: str,
    split_date: str,
    strategy_name: str,
    commission: float = 0.001,  # Research Standard (epsilon) 0.1%
    cash: float = 10000.0,
):
    """
    Generic runner for the 'Model Tournament'.
    Loads data, selects strategy, and runs the Backtest.
    """
    print("\n--- TOURNAMENT SESSION ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Strategy: {strategy_name}")
    print(f"Fee (epsilon): {commission * 100}%")

    # 1. Load Data
    df = pd.read_csv(data_path, parse_dates=["time"])
    df.set_index("time", inplace=True)
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
        trade_on_close=True,
    )

    stats = bt.run(model_path=model_path)
    print("\n--- RESULTS ---")
    print(stats)

    # 5. Save Report
    report_name = (
        f"src/evaluation/report_{strategy_name}_{os.path.basename(model_path)}.html"
    )
    bt.plot(filename=report_name, open_browser=False)
    print(f"\nReport saved to: {report_name}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Model Backtester")
    parser.add_argument("--model", type=str, default="src/models/baseline_rf.joblib")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed_market/processed_EURUSD_H1_20200101_20260131.csv",
    )
    parser.add_argument("--split-date", type=str, default="2025-01-01")
    parser.add_argument(
        "--strategy",
        type=str,
        default="TripleBarrier",
        choices=["NaiveFlip", "MajorityVote", "TripleBarrier"],
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Broker fee percentage (e.g., 0.001 for 0.1%)",
    )

    args = parser.parse_args()

    if os.path.exists(args.model) and os.path.exists(args.data):
        run_backtest_session(
            model_path=args.model,
            data_path=args.data,
            split_date=args.split_date,
            strategy_name=args.strategy,
            commission=args.epsilon,
        )
    else:
        print("Required model or data files missing.")
