import joblib
import pandas as pd


def run_baseline_inference(model_path: str, input_df: pd.DataFrame):
    """
    Loads a trained baseline model and runs inference on new data.
    """
    print(f"Loading model artifacts from {model_path}...")
    artifacts = joblib.load(model_path)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    # ensure all required features are present
    missing_features = [col for col in feature_cols if col not in input_df.columns]
    if missing_features:
        raise ValueError(
            f"Input DataFrame is missing required features: {missing_features}"
        )

    # extract features and scale
    X = input_df[feature_cols]
    X_scaled = scaler.transform(X)

    # run prediction
    print("Running inference...")
    y_pred = model.predict(X_scaled)
    y_probs = model.predict_proba(X_scaled)

    return y_pred, y_probs


if __name__ == "__main__":
    # simple test if run directly
    import os

    MODEL_PATH = "src/models/baseline_rf.joblib"
    DATA_PATH = "data/processed_market/processed_EURUSD_H1_20200101_20260131.csv"

    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).tail(5)  # test on last 5 rows
        predictions, probabilities = run_baseline_inference(MODEL_PATH, df)

        print("\n--- Inference Test Results (Last 5 Rows) ---")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = ["Up (0)", "Down (1)", "Deadband (2)"][pred]
            print(f"Row {i}: Prediction={label}, Probabilities={prob}")
    else:
        print("Model or data file not found for inference test.")
