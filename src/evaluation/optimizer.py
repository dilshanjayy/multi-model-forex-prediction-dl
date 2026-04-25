import os
import yaml
import numpy as np
import pandas as pd
import optuna
from typing import Any, Dict
from sklearn.preprocessing import RobustScaler
from backtesting import Backtest
from src.data.data_module import DataModule
from src.models.model_factory import ModelFactory
from src.strategies.base_strategies import TripleBarrierStrategy


def run_optimization_study(config_path: str, n_trials: int = 50, metric: str = "profit"):
    # Set seed once globally for the study (fast mode)
    from src.utils.reproducibility import set_seed
    set_seed(42, strict=False)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    val_split_date = config["data"].get("val_split_date", "2024-01-01")
    test_split_date = config["data"].get("test_split_date", "2025-01-01")

    horizons = config["data"].get("horizons", [5, 12, 24])
    atr_multipliers = config["data"].get("atr_multipliers", [1.0, 2.0, 3.0])

    print(f"Loading data from {processed_dir} for optimization...")
    dm = DataModule(processed_dir)
    # Load all components including metadata for backtesting prices
    df = dm.prepare_dataset(components=["technical_features", "targets", "metadata"])
    df.sort_index(inplace=True)

    if "train_split_pct" in config.get("data", {}):
        train_pct = config["data"]["train_split_pct"]
        val_pct = config["data"].get("val_split_pct", (1.0 - train_pct) / 2)
        
        n_samples = len(df)
        train_end = int(n_samples * train_pct)
        val_end = int(n_samples * (train_pct + val_pct))
        
        train_df = df.iloc[:train_end].copy()
        val_df_full = df.iloc[train_end:val_end].copy()
    else:
        train_df = df[df.index < val_split_date].copy()
        val_df_full = df[(df.index >= val_split_date) & (df.index < test_split_date)].copy()

    if len(train_df) == 0 or len(val_df_full) == 0:
        raise ValueError("Train or Validation set is empty. Check your split dates or percentages.")

    # Strict Feature Selection (Exact match with baseline_trainer.py)
    target_cols = [c for c in df.columns if "Target" in c or "LogRet" in c]
    exclude_cols = [
        "Open", "High", "Low", "Close", "Volume", 
        "open", "high", "low", "close", "tick_volume", 
        "time", "spread"
    ]
    feature_cols = [
        c for c in df.columns 
        if c not in target_cols and c not in exclude_cols and c != "Target"
    ]
    feature_cols.sort()

    # Pull Backtest constraints from config
    cash = config.get("backtest", {}).get("cash", 10000.0)
    commission = config.get("backtest", {}).get("commission", 0.0001)
    v_size = config.get("backtest", {}).get("v_size", 10000.0)
    margin = config.get("backtest", {}).get("margin", 0.02)

    # Pull Search Space and Model Type from config
    model_type = config.get("model", {}).get("type", "RandomForest")
    search_space = config.get("optimization", {}).get("search_space", {})
    
    # --- TURBO STAGE 2: Pre-training ---
    # If we are optimizing for profit, we train the brain ONCE and reuse it.
    pretrained_model = None
    shared_scaler = None
    target_col = config["model"]["target"]

    if metric == "profit":
        print(f"\n--- Turbo Mode: Pre-training the Stage 1 Brain ---")
        model_params = config["model"]["params"].copy()
        model_params["input_dim"] = len(feature_cols)
        model_params["random_state"] = 42
        
        # Safely extract lookback falling back to data config
        lookback = model_params.get("lookback", config.get("data", {}).get("lookback", 60))
        if "lookback" not in model_params:
            model_params["lookback"] = lookback
        
        # Prepare Data
        y_series = train_df[target_col].dropna()
        X_train_raw = train_df.loc[y_series.index, feature_cols]
        shared_scaler = RobustScaler()
        X_train_scaled = shared_scaler.fit_transform(X_train_raw)
        
        pretrained_model = ModelFactory.get_model(model_type, model_params)
        
        from src.models.base_torch_model import PyTorchBaseModel
        if isinstance(pretrained_model, PyTorchBaseModel):
            from src.data.window_generator import TimeSeriesWindowGenerator
            generator = TimeSeriesWindowGenerator(
                lookback=lookback,
                batch_size=config["model"].get("batch_size", 64),
                feature_cols=list(feature_cols),
                scaler=shared_scaler
            )
            train_loader, val_loader, _ = generator.prepare_loaders(
                train_df.loc[y_series.index], val_df_full, train_df.loc[y_series.index],
                target_col=target_col
            )
            pretrained_model.train_model(train_loader, val_loader)
        else:
            pretrained_model.fit(X_train_scaled, y_series.to_numpy())
        
        print(f"--- Pre-training Complete. Starting Strategy Search ---\n")

    def objective(trial):
        # 0. Set Seed for Reproducibility
        from src.utils.reproducibility import set_seed
        set_seed(42)

        # 1. Hyperparameters for Data (STRICTLY LOCKED)
        # target_col is already defined above from config
        import re
        h_match = re.search(r"(\d+)h", target_col)
        h = int(h_match.group(1)) if h_match else 24

        # 2. Hyperparameters for Model
        if metric == "profit":
            # REUSE the pre-trained brain
            model_wrapper = pretrained_model
            scaler = shared_scaler
        else:
            # Stage 1: Tune architecture and train every time
            model_params: Dict[str, Any] = {"random_state": 42}
            # ... (keep existing tuning logic for other models)
            if model_type == "CNN-LSTM":
                model_params["cnn_filters_1"] = trial.suggest_categorical("cnn_filters_1", [16, 32, 64])
                model_params["cnn_filters_2"] = trial.suggest_categorical("cnn_filters_2", [32, 64, 128])
                model_params["lstm_units"] = trial.suggest_categorical("lstm_units", [32, 50, 100])
                model_params["kernel_size"] = trial.suggest_int("kernel_size", 2, 5)
                model_params["dropout"] = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
                model_params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                model_params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
                model_params["epochs"] = config["model"]["params"].get("epochs", 50)
                model_params["lookback"] = trial.suggest_categorical("lookback", [60, 120, 240])
            
            # 4. ALIGN DATA & TRAIN (Stage 1 only)
            y_series_trial = train_df[target_col].dropna()
            X_train_raw_trial = train_df.loc[y_series_trial.index, feature_cols]
            scaler = RobustScaler()
            X_train_scaled_trial = scaler.fit_transform(X_train_raw_trial)
            
            model_params["input_dim"] = len(feature_cols)
            model_wrapper = ModelFactory.get_model(model_type, model_params)

            from src.models.base_torch_model import PyTorchBaseModel
            if isinstance(model_wrapper, PyTorchBaseModel):
                from src.data.window_generator import TimeSeriesWindowGenerator
                generator = TimeSeriesWindowGenerator(
                    lookback=model_params["lookback"], 
                    batch_size=config["model"].get("batch_size", 64), 
                    feature_cols=list(feature_cols),
                    scaler=scaler
                )
                train_loader, val_loader, _ = generator.prepare_loaders(
                    train_df.loc[y_series_trial.index], val_df_full, train_df.loc[y_series_trial.index], 
                    target_col=target_col
                )
                best_val_loss = model_wrapper.train_model(train_loader, val_loader, trial=trial)
                if metric == "loss": return best_val_loss
            else:
                model_wrapper.fit(X_train_scaled_trial, y_series_trial.to_numpy())
                # ... (keep sklearn loss logic)

        # 3. Hyperparameters for Strategy (The only thing tuned in Stage 2)
        exit_range = search_space.get("exit_atr_multiplier", [1.0, 5.0, 0.5])
        exit_atr_multiplier = trial.suggest_float("exit_atr_multiplier", exit_range[0], exit_range[1], step=exit_range[2] if len(exit_range) > 2 else None)
        
        conf_range = search_space.get("conf_threshold", [0.35, 0.55, 0.05])
        conf_threshold = trial.suggest_float("conf_threshold", conf_range[0], conf_range[1], step=conf_range[2] if len(conf_range) > 2 else None)

        # 6. BACKTEST
        artifacts = {
            "model": model_wrapper,
            "scaler": scaler,
            "feature_cols": list(feature_cols),
            "horizon": h
        }

        bt = Backtest(
            val_df_full,
            TripleBarrierStrategy,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=False,
        )

        stats = bt.run(
            model_path="", 
            artifacts=artifacts,
            v_size=v_size,
            atr_multiplier=exit_atr_multiplier,
            conf_threshold=conf_threshold
        )

        # Smart Objective: Maximize Profit Factor, but penalize bad Sharpe Ratios
        pf = stats.get("Profit Factor", np.nan)
        sharpe = stats.get("Sharpe Ratio", 0.0)
        num_trades = stats.get("# Trades", 0)
        
        # Force statistical significance: Reject models with fewer than 30 trades
        if pd.isna(pf) or pd.isna(sharpe) or num_trades < 30:
            return 0.0

        # If Sharpe is negative, it reduces the score. If positive, it boosts it.
        # This prevents the optimizer from picking a "lucky" model with a jagged equity curve.
        adjusted_score = pf * (1.0 + (sharpe / 10.0))
        return adjusted_score

    # Use MedianPruner to kill the bottom 50% of trials after epoch 10
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    
    # Decide direction based on metric
    direction = "minimize" if metric == "loss" else "maximize"
    study = optuna.create_study(direction=direction, pruner=pruner)
    print(f"\n--- Starting Optuna Optimization ({n_trials} trials | Metric: {metric}) ---")

    # Suppress backtesting prints during trials
    import sys

    original_stdout = sys.stdout
    with open(os.devnull, "w") as f:
        sys.stdout = f
        try:
            study.optimize(objective, n_trials=n_trials)
        finally:
            sys.stdout = original_stdout

    print("\n--- Optimization Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Profit Factor: {trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- AUTO-SAVE OPTIMIZED CONFIG ---
    optimized_config = config.copy()
    
    # Update Data Section (Only if tuned)
    if "horizon" in trial.params:
        best_h = trial.params["horizon"]
        base_target = config.get("model", {}).get("target", "TBM")
        if "Nguyen" in base_target:
            optimized_config["model"]["target"] = f"Target_{best_h}h_Nguyen"
        else:
            best_m = trial.params.get("label_atr_multiplier", config["data"].get("atr_multipliers", [2.0])[0])
            optimized_config["model"]["target"] = f"Target_{best_h}h_{best_m}x_TBM"
    
    # Update Model Section (Only during Stage 1 - Loss)
    if metric == "loss":
        if model_type == "RandomForest":
            optimized_config["model"]["params"]["n_estimators"] = trial.params["n_estimators"]
            optimized_config["model"]["params"]["max_depth"] = trial.params["max_depth"]
        elif model_type == "Transformer":
            optimized_config["model"]["params"]["d_model"] = trial.params["d_model"]
            optimized_config["model"]["params"]["nhead"] = trial.params["nhead"]
            optimized_config["model"]["params"]["num_layers"] = trial.params["num_layers"]
            optimized_config["model"]["params"]["dropout"] = trial.params["dropout"]
            optimized_config["model"]["params"]["learning_rate"] = trial.params["learning_rate"]
            optimized_config["model"]["params"]["lookback"] = trial.params["lookback"]
        elif model_type == "LSTM":
            optimized_config["model"]["params"]["hidden_dim"] = trial.params["hidden_dim"]
            optimized_config["model"]["params"]["num_layers"] = trial.params["num_layers"]
            optimized_config["model"]["params"]["dropout"] = trial.params["dropout"]
            optimized_config["model"]["params"]["learning_rate"] = trial.params["learning_rate"]
            optimized_config["model"]["params"]["lookback"] = trial.params["lookback"]
        elif model_type == "CNN-LSTM":
            optimized_config["model"]["params"]["cnn_filters_1"] = trial.params["cnn_filters_1"]
            optimized_config["model"]["params"]["cnn_filters_2"] = trial.params["cnn_filters_2"]
            optimized_config["model"]["params"]["lstm_units"] = trial.params["lstm_units"]
            optimized_config["model"]["params"]["kernel_size"] = trial.params["kernel_size"]
            optimized_config["model"]["params"]["dropout"] = trial.params["dropout"]
            optimized_config["model"]["params"]["weight_decay"] = trial.params["weight_decay"]
            optimized_config["model"]["params"]["learning_rate"] = trial.params["learning_rate"]
            optimized_config["model"]["params"]["lookback"] = trial.params["lookback"]
    
    # Update Backtest Section (Always updated in Profit mode)
    if metric == "profit":
        optimized_config["backtest"]["atr_multiplier"] = trial.params["exit_atr_multiplier"]
        optimized_config["backtest"]["conf_threshold"] = trial.params["conf_threshold"]

    # Save to file
    base_name = os.path.basename(config_path).replace(".yaml", "")
    output_path = f"configs/optimized_{base_name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(optimized_config, f, default_flow_style=False)

    print(f"\n[SUCCESS] Optimized configuration saved to: {output_path}")
    print(f"You can now run: python main.py run --config {output_path}")

    return study
