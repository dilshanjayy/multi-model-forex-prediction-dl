import pandas as pd
import numpy as np
import os
import joblib
import json
import yaml
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from src.data.data_module import DataModule
from src.models.model_factory import ModelFactory


def run_baseline_training(
    processed_dir: str,
    target_col: str,
    split_date: str,
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
    print(f"Splitting data at {split_date}...")
    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: Split date results in empty train or test set.")
        return

    # 2. Labeling Logic
    if target_col.endswith("_TBM"):
        print(f"Using pre-computed Triple Barrier Labels from {target_col}...")
        train_df["Target"] = train_df[target_col]
        test_df["Target"] = test_df[target_col]
        lower_threshold, upper_threshold = 0, 0
    else:
        print(f"Calculating quantile thresholds on training data for {target_col}...")
        lower_threshold = train_df[target_col].quantile(0.30)
        upper_threshold = train_df[target_col].quantile(0.65)

        def apply_labels(data):
            conditions = [
                data[target_col] > upper_threshold,
                data[target_col] < lower_threshold,
            ]
            return np.select(conditions, [0, 1], default=2)

        train_df["Target"] = apply_labels(train_df)
        test_df["Target"] = apply_labels(test_df)

    # 3. Prepare Features
    target_cols = [c for c in df.columns if "Target" in c or "LogRet" in c]
    exclude_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Vol",
        "Triple_Barrier_Label",
    ]
    feature_cols = [
        c
        for c in df.columns
        if c not in target_cols and c not in exclude_cols and c != "Target"
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["Target"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["Target"].to_numpy()

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Instantiation & Training
    model_type = config["model"]["type"] if config else "RandomForest"
    model_params = config["model"]["params"] if config else {}

    # Use Factory to get model
    model_wrapper = ModelFactory.get_model(model_type, model_params)
    model_wrapper.train(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = model_wrapper.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Up (0)", "Down (1)", "Deadband (2)"],
        output_dict=True,
    )
    print(f"\nModel Evaluation (Test Set) - Accuracy: {acc:.4f}")

    # 7. Save Artifacts
    os.makedirs(experiment_dir, exist_ok=True)

    # Bundle metadata with model
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

    # Save Model Weights/State using its own interface
    model_wrapper.save(os.path.join(experiment_dir, "model_state.joblib"))

    # Save Metadata separately
    joblib.dump(inference_artifacts, os.path.join(experiment_dir, "metadata.joblib"))

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
