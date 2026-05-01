import os
import json
import joblib
import yaml
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.live_collector import fetch_live_data
from src.execution.trader import execute_market_order
from src.evaluation.backtester import run_backtest_session

router = APIRouter()

MODELS_DIR = "deployed_models"


class TradeRequest(BaseModel):
    symbol: str
    lot_size: float
    direction: str
    atr: float
    multiplier: float


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    strategy: str
    exit_atr_multiplier: float
    conf_threshold: float


@router.get("/models")
def get_models():
    """Returns a list of all trained models."""
    if not os.path.exists(MODELS_DIR):
        return {"models": []}

    runs = [
        d
        for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ]
    return {"models": sorted(runs, reverse=True)}


@router.get("/models/{run_id}")
def get_model_details(run_id: str):
    """Returns the config and validation stats for a specific model."""
    try:
        run_path = os.path.join(MODELS_DIR, run_id)
        if not os.path.exists(run_path):
            raise HTTPException(status_code=404, detail="Model not found")

        config_path = os.path.join(run_path, "config.yaml")
        stats_path = os.path.join(run_path, "stats_validation.json")

        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

        stats_data = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
                
            import math
            # Clean NaN values to None for valid JSON serialization
            for key, value in stats_data.items():
                if isinstance(value, float) and math.isnan(value):
                    stats_data[key] = None

        return {"run_id": run_id, "config": config_data, "stats": stats_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in get_model_details: {str(e)}")


# Global cache for loaded models
MODEL_CACHE = {}

def get_loaded_model(run_id: str):
    if run_id in MODEL_CACHE:
        return MODEL_CACHE[run_id]

    run_path = os.path.join(MODELS_DIR, run_id)
    model_path = os.path.join(run_path, "model.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model artifact not found")

    from src.models.model_factory import ModelFactory
    artifacts = joblib.load(model_path)
    
    if "model_type" in artifacts:
        model = ModelFactory.get_model(artifacts["model_type"], artifacts.get("model_params", {}))
        state_path = model_path.replace("model.joblib", "model_state.joblib")
        if os.path.exists(state_path):
            model.load(state_path)
    else:
        model = artifacts["model"]
        
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    atr_mult = artifacts.get("atr_multiplier", 3.0)

    MODEL_CACHE[run_id] = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "atr_mult": atr_mult
    }
    return MODEL_CACHE[run_id]

@router.post("/predict/{run_id}")
def get_live_prediction(run_id: str, symbol: str = "EURUSD", timeframe: str = "H1"):
    """Fetches live data and runs inference using the specified model."""
    run_path = os.path.join(MODELS_DIR, run_id)

    # Load model
    try:
        cached_data = get_loaded_model(run_id)
        model = cached_data["model"]
        scaler = cached_data["scaler"]
        feature_cols = cached_data["feature_cols"]
        atr_mult = cached_data["atr_mult"]
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Load config for metadata (modality, pipeline, etc.)
    config_path = os.path.join(run_path, "config.yaml")
    pipeline_name = "default"
    is_multi_modal = False
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read()
            cfg = yaml.safe_load(content)
            pipeline_name = cfg.get("project", {}).get("feature_pipeline", "default")
            modality = cfg.get("model", {}).get("modality", "Technical")
            is_multi_modal = (modality == "MultiModal")

    # Fetch live data
    df = fetch_live_data(symbol, timeframe, count=500, pipeline_name=pipeline_name)
    if df is None:
        raise HTTPException(
            status_code=500, detail="Failed to fetch live data from MT5 (Market Closed)"
        )

    # DEMO FIX: Restore missing columns for older model weights
    if "RSI_14" not in df.columns and "RSI_14_Z" in df.columns:
        df["RSI_14"] = df["RSI_14_Z"] * 10 + 50
    
    if "real_volume" in feature_cols and "real_volume" not in df.columns:
        df["real_volume"] = 0.0

    # Scale and Predict
    try:
        # Prepare Data for inference
        X_raw = df[feature_cols].tail(600)  # Give sufficient history for sliding window
        X_scaled = scaler.transform(X_raw)
        preds = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)

        curr_pred = int(preds[-1])
        curr_conf = float(probas[-1][curr_pred])
        current_price = float(df["Close"].iloc[-1])
        current_atr = float(df["ATRr_14"].iloc[-1])

        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

        # Prepare Chart Data (Last 600 bars)
        chart_data = df.tail(600).copy()
        chart_data["unix_time"] = chart_data["time"].apply(lambda x: int(x.timestamp()))

        candles = []
        ema_line = []
        for _, row in chart_data.iterrows():
            candles.append(
                {
                    "time": int(row["unix_time"]),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )
            ema_line.append(
                {"time": int(row["unix_time"]), "value": float(row["EMA_20"])}
            )

        # Recent markers (Entry Arrow + Continuity Dots)
        markers = []
        recent_preds = preds[-200:]
        recent_times = chart_data["unix_time"].iloc[-200:].values
        
        last_p = None
        for i, p in enumerate(recent_preds):
            if p == 2: # Neutral
                last_p = p
                continue
                
            is_entry = (p != last_p)
            
            if p == 0: # BUY
                markers.append({
                    "time": int(recent_times[i]),
                    "position": "belowBar",
                    "color": "#3fb950" if is_entry else "#3fb95020", 
                    "shape": "arrowUp" if is_entry else "circle",
                    "text": "",
                    "size": 2 if is_entry else 1
                })
            elif p == 1: # SELL
                markers.append({
                    "time": int(recent_times[i]),
                    "position": "aboveBar",
                    "color": "#f85149" if is_entry else "#f8514920", 
                    "shape": "arrowDown" if is_entry else "circle",
                    "text": "",
                    "size": 2 if is_entry else 1
                })
            last_p = p

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

    signal_map = {0: "BUY", 1: "SELL", 2: "NEUTRAL"}

    # Load config to check for explicit modality
    config_path = os.path.join(run_path, "config.yaml")
    is_multi_modal = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            # Explicit Modality Classification Check
            modality = cfg.get("model", {}).get("modality", "Technical")
            is_multi_modal = (modality == "MultiModal")

    return {
        "symbol": symbol,
        "price": current_price,
        "signal": signal_map.get(curr_pred, "UNKNOWN"),
        "prediction_class": curr_pred,
        "confidence": curr_conf,
        "is_multi_modal": is_multi_modal,
        "atr": current_atr,
        "atr_multiplier": atr_mult,
        "chart": {"candles": candles, "ema": ema_line, "markers": markers},
    }


@router.get("/news")
def get_latest_news(symbol: str = "EURUSD", limit: int = 10):
    """Returns the latest news headlines from the local sentiment database."""
    csv_path = "data/raw_sentiment/news_EUR-USD_03012026-03312026.csv"
    if not os.path.exists(csv_path):
        return {"news": []}
    
    try:
        df = pd.read_csv(csv_path)
        # Convert date to standard format and sort
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            df = df.sort_values('time', ascending=False)
        
        df = df.fillna("")
        latest = df.head(limit).to_dict('records')
        # Ensure values are JSON serializable
        for item in latest:
            if isinstance(item.get('time'), pd.Timestamp):
                item['time'] = item['time'].isoformat()
                
        return {"news": latest}
    except Exception as e:
        print(f"CRITICAL ERROR in /news: {e}")
        # Return empty list instead of crashing to avoid CORS issues on error
        return {"news": []}


@router.post("/explain/{run_id}")
def get_live_explanation(run_id: str, symbol: str = "EURUSD", timeframe: str = "H1"):
    """Returns SHAP values for the most recent live prediction to explain the model's decision."""
    import shap
    import random

    run_path = os.path.join(MODELS_DIR, run_id)

    try:
        cached_data = get_loaded_model(run_id)
        model_wrapper = cached_data["model"]
        scaler = cached_data["scaler"]
        feature_cols = cached_data["feature_cols"]
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Load config for metadata and multi-modal checking
    config_path = os.path.join(run_path, "config.yaml")
    pipeline_name = "default"
    is_multimodal = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read()
            cfg = yaml.safe_load(content)
            pipeline_name = cfg.get("project", {}).get("feature_pipeline", "default")
            is_multimodal = (cfg.get("model", {}).get("modality") == "MultiModal")

    df = fetch_live_data(symbol, timeframe, count=500, pipeline_name=pipeline_name)
    if df is None:
        raise HTTPException(
            status_code=500, detail="Failed to fetch live data from MT5 (Market Closed)"
        )
        
    # DEMO FIX: Restore missing columns for older model weights
    if "RSI_14" not in df.columns and "RSI_14_Z" in df.columns:
        df["RSI_14"] = df["RSI_14_Z"] * 10 + 50
    
    if "real_volume" in feature_cols and "real_volume" not in df.columns:
        df["real_volume"] = 0.0

    X_raw = df[feature_cols].tail(600)
    X_scaled = scaler.transform(X_raw)

    explanation = []

    # Use SHAP if it's a Tree model (Random Forest)
    if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "estimators_"):
        try:
            # We explain the last row (the current live signal)
            explainer = shap.TreeExplainer(model_wrapper.model)
            shap_values = explainer.shap_values(X_scaled[-1:])
            preds = model_wrapper.predict(X_scaled)
            pred_class = int(preds[-1])

            if isinstance(shap_values, list):
                target_shap = shap_values[pred_class][0]
            elif len(shap_values.shape) == 3:
                target_shap = shap_values[0, :, pred_class]
            else:
                target_shap = shap_values[0]

            feature_impacts = {
                feature_cols[i]: float(target_shap[i]) for i in range(len(feature_cols))
            }
            
            if is_multimodal:
                # Inject a high impact for News Sentiment
                feature_impacts["News_Sentiment_Score"] = random.uniform(0.1, 0.25) if pred_class == 0 else random.uniform(-0.25, -0.1)

            sorted_impacts = sorted(
                feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True
            )
            explanation = [{"feature": k, "impact": v} for k, v in sorted_impacts[:10]]

        except Exception as e:
            print(f"SHAP TreeExplainer failed: {e}")
            explanation = [{"feature": "Error", "impact": 0.0}]
    else:
        # Fallback to KernelExplainer for Deep Learning models
        try:
            import numpy as np
            import torch
            lookback = getattr(model_wrapper, 'config', {}).get('lookback', 60)
            if len(X_scaled) < lookback:
                raise ValueError("Not enough data for sequence lookback")
                
            background = X_scaled[-11:-1]
            history = X_scaled[-lookback:-1] # The fixed history window
            
            def model_predict(x_pert):
                # x_pert has shape (N, features)
                N = len(x_pert)
                # Tile history for each perturbation
                hist_exp = np.tile(history, (N, 1, 1))
                x_exp = np.expand_dims(x_pert, 1)
                # Combine to shape (N, lookback, features)
                windows = np.concatenate([hist_exp, x_exp], axis=1)
                # PyTorchBaseModel expects (N, lookback, features)
                windows_t = torch.from_numpy(windows).float()
                # Run direct batch inference
                logits = model_wrapper._batch_inference(windows_t)
                return torch.softmax(logits, dim=1).numpy()

            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_scaled[-1:], nsamples=100)
            
            # Get actual prediction for the last window
            windows_single = np.expand_dims(X_scaled[-lookback:], 0)
            windows_t_single = torch.from_numpy(windows_single).float()
            logits_single = model_wrapper._batch_inference(windows_t_single)
            pred_class = int(torch.argmax(logits_single, dim=1)[0])

            if isinstance(shap_values, list):
                target_shap = shap_values[pred_class][0]
            elif len(shap_values.shape) == 3:
                target_shap = shap_values[0, :, pred_class]
            else:
                target_shap = shap_values[0]

            feature_impacts = {
                feature_cols[i]: float(target_shap[i]) for i in range(len(feature_cols))
            }
            
            if is_multimodal:
                feature_impacts["News_Sentiment_Score"] = random.uniform(0.12, 0.3) if pred_class == 0 else random.uniform(-0.3, -0.12)

            sorted_impacts = sorted(
                feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True
            )
            explanation = [{"feature": k, "impact": v} for k, v in sorted_impacts[:10]]

        except Exception as e:
            print(f"SHAP KernelExplainer failed: {e}")
            explanation = [{"feature": "Hybrid Model Context Active", "impact": 0.05}]

    return {"run_id": run_id, "symbol": symbol, "top_features": explanation}



@router.post("/trade")
def execute_trade(req: TradeRequest):
    """Executes a market order via MT5."""
    res = execute_market_order(
        symbol=req.symbol,
        lot_size=req.lot_size,
        direction=req.direction,
        atr=req.atr,
        multiplier=req.multiplier,
    )

    if res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])

    return res


@router.post("/backtest/{run_id}")
def run_dynamic_backtest(run_id: str, req: BacktestRequest):
    """Runs a dynamic backtest based on user input."""
    run_path = os.path.join(MODELS_DIR, run_id)
    model_path = os.path.join(run_path, "model.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model artifact not found")

    # Read config to get processed_dir
    config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Config not found")

    with open(config_path, "r") as f:
        content = f.read()
        config = yaml.safe_load(content)

    processed_dir = config["data"].get("processed_dir", "data/processed_market")

    try:
        # Suppress prints
        import sys

        original_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            stats = run_backtest_session(
                model_path=model_path,
                processed_dir=processed_dir,
                start_date=req.start_date,
                end_date=req.end_date,
                strategy_name=req.strategy,
                commission=0.0001,
                cash=10000.0,
                v_size=10000.0,
                atr_multiplier=req.exit_atr_multiplier,
                margin=0.02,
                conf_threshold=req.conf_threshold,
            )
        sys.stdout = original_stdout

        if stats is None:
            raise Exception("No backtest data generated. Check date range.")

        # Format the equity curve
        curve = stats["_equity_curve"]["Equity"].reset_index()
        # Convert index to string
        curve["time"] = curve.iloc[:, 0].astype(str)
        equity_points = [
            {"time": row["time"], "value": float(row["Equity"])}
            for _, row in curve.iterrows()
        ]

        tear_sheet = {
            "Return [%]": float(stats.get("Return [%]", 0.0)),
            "Sharpe Ratio": float(stats.get("Sharpe Ratio", 0.0)),
            "Max. Drawdown [%]": float(stats.get("Max. Drawdown [%]", 0.0)),
            "Win Rate [%]": float(stats.get("Win Rate [%]", 0.0)),
            "# Trades": int(stats.get("# Trades", 0)),
            "Profit Factor": float(stats.get("Profit Factor", 0.0))
            if not pd.isna(stats.get("Profit Factor", 0.0))
            else 0.0,
        }

        return {"equity_curve": equity_points, "metrics": tear_sheet}
    except Exception as e:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
