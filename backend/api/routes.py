import os
import json
import joblib
import yaml
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.live_collector import fetch_live_data
from src.execution.trader import execute_market_order

router = APIRouter()

EXPERIMENTS_DIR = "experiments"

from src.evaluation.backtester import run_backtest_session
from backtesting import Backtest

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
    if not os.path.exists(EXPERIMENTS_DIR):
        return {"models": []}
    
    runs = [d for d in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    return {"models": sorted(runs, reverse=True)}

@router.get("/models/{run_id}")
def get_model_details(run_id: str):
    """Returns the config and validation stats for a specific model."""
    run_path = os.path.join(EXPERIMENTS_DIR, run_id)
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
            
    return {
        "run_id": run_id,
        "config": config_data,
        "stats": stats_data
    }

@router.post("/predict/{run_id}")
def get_live_prediction(run_id: str, symbol: str = "EURUSD", timeframe: str = "H1"):
    """Fetches live data and runs inference using the specified model."""
    run_path = os.path.join(EXPERIMENTS_DIR, run_id)
    model_path = os.path.join(run_path, "model.joblib")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    # Load model
    try:
        artifacts = joblib.load(model_path)
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts["feature_cols"]
        atr_mult = artifacts.get("atr_multiplier", 3.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Fetch live data
    df = fetch_live_data(symbol, timeframe, count=500)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to fetch live data from MT5")
        
    # Scale and Predict
    try:
        # Prepare Data for inference
        X_raw = df[feature_cols].tail(100) # Give sufficient history for sliding window
        X_scaled = scaler.transform(X_raw)
        preds = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)
        
        curr_pred = int(preds[-1])
        curr_conf = float(probas[-1][curr_pred])
        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['ATRr_14'].iloc[-1])
        
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Prepare Chart Data (Last 100 bars)
        chart_data = df.tail(100).copy()
        chart_data['unix_time'] = chart_data['time'].apply(lambda x: int(x.timestamp()))
        
        candles = []
        ema_line = []
        for _, row in chart_data.iterrows():
            candles.append({
                "time": int(row["unix_time"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"])
            })
            ema_line.append({"time": int(row["unix_time"]), "value": float(row["EMA_20"])})
            
        # Recent markers
        markers = []
        recent_preds = preds[-50:]
        recent_times = chart_data['unix_time'].iloc[-50:].values
        for i, p in enumerate(recent_preds):
            if p == 0:
                markers.append({"time": int(recent_times[i]), "position": "belowBar", "color": "#3fb950", "shape": "arrowUp", "text": "BUY"})
            elif p == 1:
                markers.append({"time": int(recent_times[i]), "position": "aboveBar", "color": "#f85149", "shape": "arrowDown", "text": "SELL"})

    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
         
    signal_map = {0: "BUY", 1: "SELL", 2: "NEUTRAL"}
    
    return {
        "symbol": symbol,
        "price": current_price,
        "signal": signal_map.get(curr_pred, "UNKNOWN"),
        "prediction_class": curr_pred,
        "confidence": curr_conf,
        "atr": current_atr,
        "atr_multiplier": atr_mult,
        "chart": {
            "candles": candles,
            "ema": ema_line,
            "markers": markers
        }
    }

@router.post("/explain/{run_id}")
def get_live_explanation(run_id: str, symbol: str = "EURUSD", timeframe: str = "H1"):
    """Returns SHAP values for the most recent live prediction to explain the model's decision."""
    import shap
    
    run_path = os.path.join(EXPERIMENTS_DIR, run_id)
    model_path = os.path.join(run_path, "model.joblib")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    try:
        artifacts = joblib.load(model_path)
        model_wrapper = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts["feature_cols"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
    df = fetch_live_data(symbol, timeframe, count=500)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to fetch live data from MT5")
        
    X_raw = df[feature_cols].tail(100)
    X_scaled = scaler.transform(X_raw)
    
    explanation = {}
    
    # Use SHAP if it's a Tree model (Random Forest)
    if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "estimators_"):
        try:
            # We explain the last row (the current live signal)
            explainer = shap.TreeExplainer(model_wrapper.model)
            shap_values = explainer.shap_values(X_scaled[-1:])
            
            # For classification, shap_values is a list of arrays (one for each class)
            # We find the predicted class
            preds = model_wrapper.predict(X_scaled)
            pred_class = int(preds[-1])
            
            # Extract the SHAP values for the predicted class
            if isinstance(shap_values, list):
                target_shap = shap_values[pred_class][0]
            elif len(shap_values.shape) == 3:
                target_shap = shap_values[0, :, pred_class]
            else:
                target_shap = shap_values[0] # Fallback
                
            # Create a dictionary of feature -> SHAP contribution
            feature_impacts = {feature_cols[i]: float(target_shap[i]) for i in range(len(feature_cols))}
            
            # Sort by absolute impact
            sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            explanation = [{"feature": k, "impact": v} for k, v in sorted_impacts[:10]]
            
        except Exception as e:
            print(f"SHAP TreeExplainer failed: {e}")
            explanation = [{"feature": "Error", "impact": 0.0}]
    else:
        # Fallback to KernelExplainer for Deep Learning models (PyTorch)
        try:
            # Use the last 10 samples as the background dataset for speed
            background = X_scaled[-11:-1]
            def model_predict(x):
                return model_wrapper.predict_proba(x)
                
            explainer = shap.KernelExplainer(model_predict, background)
            # Explain the most recent sample
            shap_values = explainer.shap_values(X_scaled[-1:], nsamples=100)
            
            preds = model_wrapper.predict(X_scaled)
            pred_class = int(preds[-1])
            
            # KernelExplainer usually returns shape (n_samples, n_features, n_classes)
            if isinstance(shap_values, list):
                target_shap = shap_values[pred_class][0]
            elif len(shap_values.shape) == 3:
                target_shap = shap_values[0, :, pred_class]
            else:
                target_shap = shap_values[0]
                
            feature_impacts = {feature_cols[i]: float(target_shap[i]) for i in range(len(feature_cols))}
            sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            explanation = [{"feature": k, "impact": v} for k, v in sorted_impacts[:10]]
            
        except Exception as e:
            print(f"SHAP KernelExplainer failed: {e}")
            explanation = [{"feature": "Deep Learning Explainability failed", "impact": 0.0}]

    return {
        "run_id": run_id,
        "symbol": symbol,
        "top_features": explanation
    }

@router.post("/trade")
def execute_trade(req: TradeRequest):
    """Executes a market order via MT5."""
    res = execute_market_order(
        symbol=req.symbol, 
        lot_size=req.lot_size, 
        direction=req.direction, 
        atr=req.atr, 
        multiplier=req.multiplier
    )
    
    if res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])
        
    return res

@router.post("/backtest/{run_id}")
def run_dynamic_backtest(run_id: str, req: BacktestRequest):
    """Runs a dynamic backtest based on user input."""
    run_path = os.path.join(EXPERIMENTS_DIR, run_id)
    model_path = os.path.join(run_path, "model.joblib")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    # Read config to get processed_dir
    config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Config not found")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["data"].get("processed_dir", "data/processed_market")
    
    try:
        # Suppress prints
        import sys
        original_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
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
                conf_threshold=req.conf_threshold
            )
        sys.stdout = original_stdout
        
        if stats is None:
            raise Exception("No backtest data generated. Check date range.")
            
        # Format the equity curve
        curve = stats["_equity_curve"]["Equity"].reset_index()
        # Convert index to string
        curve["time"] = curve.iloc[:, 0].astype(str)
        equity_points = [{"time": row["time"], "value": float(row["Equity"])} for _, row in curve.iterrows()]
        
        tear_sheet = {
            "Sharpe Ratio": float(stats.get("Sharpe Ratio", 0.0)),
            "Max Drawdown [%]": float(stats.get("Max. Drawdown [%]", 0.0)),
            "Win Rate [%]": float(stats.get("Win Rate [%]", 0.0)),
            "Total Trades": int(stats.get("# Trades", 0)),
            "Profit Factor": float(stats.get("Profit Factor", 0.0)) if not pd.isna(stats.get("Profit Factor", 0.0)) else 0.0
        }
        
        return {
            "equity_curve": equity_points,
            "metrics": tear_sheet
        }
    except Exception as e:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
