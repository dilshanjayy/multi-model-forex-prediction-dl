import pandas as pd
import os
import joblib
import json
import yaml
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import RobustScaler

from src.data.data_module import DataModule
from src.models.model_factory import ModelFactory


def run_baseline_training(
    processed_dir: str,
    target_col: str,
    val_split_date: str,
    test_split_date: str,
    experiment_dir: str,
    config: dict | None = None,
):
    """
    Industry-Standard Trainer using the Pluggable Model Factory.
    """
    print(f"Loading data from Feature Store at {processed_dir}...")
    dm = DataModule(processed_dir)
    df = dm.prepare_dataset(components=["technical_features", "targets"])
    df.sort_index(inplace=True)

    # 1. Temporal Split
    if config and "train_split_pct" in config.get("data", {}):
        train_pct = config["data"]["train_split_pct"]
        val_pct = config["data"].get("val_split_pct", (1.0 - train_pct) / 2)
        test_pct = 1.0 - train_pct - val_pct
        
        n_samples = len(df)
        train_end = int(n_samples * train_pct)
        val_end = int(n_samples * (train_pct + val_pct))
        
        print(f"Splitting data chronologically by percentage: Train ({train_pct:.0%}), Val ({val_pct:.0%}), Test ({test_pct:.0%})...")
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"--- Data Split Audit ---")
        print(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} rows)")
        print(f"Val:   {val_df.index[0]} to {val_df.index[-1]} ({len(val_df)} rows)")
        print(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} rows)")
        print(f"------------------------")
    else:
        print(
            f"Splitting data into Train (< {val_split_date}), Val ({val_split_date} to {test_split_date}), and Test (>= {test_split_date})..."
        )
        train_df = df[df.index < val_split_date].copy()
        val_df = df[(df.index >= val_split_date) & (df.index < test_split_date)].copy()
        test_df = df[df.index >= test_split_date].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("Error: Split dates result in empty train, val, or test set.")
        return

    # 2. Target Identification
    print(f"Loading target labels from '{target_col}' column...")
    train_df["Target"] = train_df[target_col]
    val_df["Target"] = val_df[target_col]
    test_df["Target"] = test_df[target_col]
    lower_threshold, upper_threshold = 0, 0

    # 3. Prepare Features
    target_cols = [c for c in df.columns if "Target" in c or "LogRet" in c]
    # Explicitly exclude metadata and time-based columns
    exclude_cols = [
        "Open", "High", "Low", "Close", "Volume", 
        "open", "high", "low", "close", "tick_volume", 
        "time", "spread"
    ]
    feature_cols = [
        c
        for c in df.columns
        if c not in target_cols and c not in exclude_cols and c != "Target"
    ]
    feature_cols.sort() # Ensure deterministic order for scaler synchronization

    X_train = train_df[feature_cols]
    y_train = train_df["Target"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["Target"].to_numpy()

    # 4. Feature Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Model Instantiation & Training
    model_config = config.get("model", {}) if config else {}
    model_type = model_config.get("type", "RandomForest")
    model_params = model_config.get("params", {})

    # Add input_dim and sequences configs for DL models
    model_params["input_dim"] = len(feature_cols)
    if config:
        data_config = config.get("data", {})
        if "lookback" not in model_params:
            model_params["lookback"] = data_config.get("lookback", 60)
        model_params["batch_size"] = model_config.get("batch_size", 64)

    # Use Factory to get model
    model_wrapper = ModelFactory.get_model(model_type, model_params)

    # CHECK: Is this a Deep Learning (PyTorch) model?
    from src.models.base_torch_model import PyTorchBaseModel
    from src.data.window_generator import TimeSeriesWindowGenerator

    if isinstance(model_wrapper, PyTorchBaseModel):
        print(f"\n--- Deep Learning Mode: Training {model_type} ---")
        # Extract values safely for Pylance
        lookback = model_params.get("lookback", 60)
        batch_size = model_params.get("batch_size", 64)
        
        # 5a. Create Sequential DataLoaders
        generator = TimeSeriesWindowGenerator(
            lookback=lookback, 
            batch_size=batch_size, 
            feature_cols=list(feature_cols),
            scaler=scaler
        )
        train_loader, val_loader, test_loader = generator.prepare_loaders(
            train_df, val_df, test_df,
            target_col="Target"
        )
        
        # 5b. Train using PyTorch training loop
        model_wrapper.train_model(train_loader, val_loader)
        
        # Final evaluation on Validation Set (Sequence Mode)
        # For simplicity in evaluation report, we use numpy predictions
        # but modern models usually stay in DataLoader mode.
    else:
        # Standard Tabular Training
        model_wrapper.fit(X_train_scaled, y_train)

    # 6. Evaluation (Common Interface)
    # We evaluate on the Validation set for the console report
    y_val_pred = model_wrapper.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    report = classification_report(
        y_val,
        y_val_pred,
        target_names=["Up (0)", "Down (1)", "Deadband (2)"],
        output_dict=True,
    )
    print(f"\nModel Evaluation (Validation Set) - Accuracy: {acc:.4f}")

    # 7. Save Artifacts
    os.makedirs(experiment_dir, exist_ok=True)

    # Get horizon from target column name
    horizon_match = re.search(r"(\d+)h", target_col)
    horizon = int(horizon_match.group(1)) if horizon_match else 1

    # Save standardized artifact for inference
    inference_artifacts = {
        "model_type": model_type,
        "model_params": model_params,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "horizon": horizon,
        "thresholds": {"lower": lower_threshold, "upper": upper_threshold},
    }

    # Save Model Weights/State using its own interface (torch.save)
    model_state_path = os.path.join(experiment_dir, "model_state.joblib")
    model_wrapper.save(model_state_path)

    # Save unified model.joblib for metadata
    joblib.dump(inference_artifacts, os.path.join(experiment_dir, "model.joblib"))

    # Save Metrics
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "feature_importance": pd.Series(
            model_wrapper.feature_importance, index=feature_cols
        )
        .sort_values(ascending=False)
        .head(20)
        .to_dict(),
    }
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Config
    if config:
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    print(f"\nExperiment complete. Artifacts saved to: {experiment_dir}")
    return model_wrapper
