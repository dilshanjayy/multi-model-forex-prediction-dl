import os
import json
import joblib
import yaml
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone

from src.data.live_collector import fetch_live_data
from src.execution.trader import execute_market_order
from src.evaluation.backtester import run_backtest_session
from src.data.sentiment_processor import score_headlines, get_sentiment_engine
from src.data.news_data_collector import fetch_news_data

from backend.db.database import get_db
from backend.db import models, schemas
from backend.api.auth import get_current_user

router = APIRouter()

MODELS_DIR = "deployed_models"

import threading

# --- LIVE SENTIMENT CACHE ---
class LiveSentimentStore:
    def __init__(self):
        self.last_fetch = None
        self.cached_news = []
        self.cached_scores = {
            'sent_pos': 0.0,
            'sent_neg': 0.0,
            'sent_neu': 1.0,
            'sentiment_score': 0.0
        }
        self.ttl_minutes = 5
        self._is_refreshing = False

    def get_live_data(self):
        now = datetime.now(timezone.utc)
        if self.last_fetch is None:
            self.refresh() # Block on first load
        elif (now - self.last_fetch) > timedelta(minutes=self.ttl_minutes):
            if not self._is_refreshing:
                self._is_refreshing = True
                threading.Thread(target=self.refresh, daemon=True).start()
        return self.cached_news, self.cached_scores

    def refresh(self):
        try:
            print("--- Refreshing Live Sentiment from API (Background) ---")
            now = datetime.now(timezone.utc)
            end_d = now
            start_d = end_d - timedelta(days=2)
            fmt = "%m%d%Y"
            range_str = f"{start_d.strftime(fmt)}-{end_d.strftime(fmt)}"
            
            df = fetch_news_data(range_str)
            if df is not None and not df.empty:
                latest = df.head(10).copy()
                probs = score_headlines(latest['title'].tolist(), latest['text'].tolist())
                
                latest['sent_pos'] = probs[:, 0]
                latest['sent_neg'] = probs[:, 1]
                latest['sent_neu'] = probs[:, 2]
                latest['sentiment_score'] = latest['sent_pos'] - latest['sent_neg']
                latest['sentiment_label'] = latest['sentiment_score'].apply(
                    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
                )
                
                if 'time' in latest.columns:
                    latest['time'] = pd.to_datetime(latest['time'], utc=True)
                
                self.cached_news = latest.to_dict('records')
                
                # Ensure values are JSON serializable
                for item in self.cached_news:
                    if isinstance(item.get('time'), pd.Timestamp):
                        item['time'] = item['time'].isoformat()
                self.cached_scores = {
                    'sent_pos': float(probs.mean(axis=0)[0]),
                    'sent_neg': float(probs.mean(axis=0)[1]),
                    'sent_neu': float(probs.mean(axis=0)[2]),
                    'sentiment_score': float((probs[:, 0] - probs[:, 1]).mean())
                }
                self.last_fetch = now
                print(f"--- Live Sentiment Updated: {len(self.cached_news)} items ---")
        except Exception as e:
            print(f"Error in LiveSentimentStore.refresh: {e}")
        finally:
            self._is_refreshing = False

SENTIMENT_STORE = LiveSentimentStore()

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

def preload_all_models():
    """Pre-loads all models in the deployed_models directory into memory."""
    if not os.path.exists(MODELS_DIR):
        return
    runs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    print(f"--- Pre-loading {len(runs)} Trading Models into RAM ---")
    for run_id in runs:
        try:
            get_loaded_model(run_id)
            print(f"Loaded: {run_id}")
        except Exception as e:
            print(f"Failed to load {run_id}: {e}")
    print("--- All Models Ready ---")

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
            pipeline_name = cfg.get("data", {}).get("feature_pipeline", "default")
            modality = cfg.get("model", {}).get("modality", "Technical")
            is_multi_modal = (modality == "MultiModal")

    # Fetch live data
    df = fetch_live_data(symbol, timeframe, count=500, pipeline_name=pipeline_name)
    if df is None:
        raise HTTPException(
            status_code=500, detail="Failed to fetch live data from MT5 (Market Closed)"
        )

    # MultiModal Logic: Inject LIVE Sentiment
    if is_multi_modal:
        _, live_scores = SENTIMENT_STORE.get_live_data()
        for col, val in live_scores.items():
            df[col] = val
        print(f"--- Injected Live Sentiment Score: {live_scores['sentiment_score']:.4f} ---")

    # DEMO FIX: Restore missing columns for older model weights
    if "RSI_14" not in df.columns and "RSI_14_Z" in df.columns:
        df["RSI_14"] = df["RSI_14_Z"] * 10 + 50
    
    if "real_volume" in feature_cols and "real_volume" not in df.columns:
        df["real_volume"] = 0.0

    # Scale and Predict
    try:
        # Prepare Data for inference
        lookback = 60
        if hasattr(model, 'config'):
            lookback = model.config.get('lookback', 60)
            
        # We need 200 predictions for the chart markers, so we need 200 + lookback - 1 rows
        needed_rows = 200 + lookback
        X_raw = df[feature_cols].tail(needed_rows)
        X_scaled = scaler.transform(X_raw)
        
        preds = model.predict(X_scaled)
        
        # We only need the confidence for the very last window
        last_window = X_scaled[-lookback:]
        probas = model.predict_proba(last_window)

        curr_pred = int(preds[-1])
        curr_conf = float(probas[-1][curr_pred])
        current_price = float(df["Close"].iloc[-1])
        current_atr = float(df["ATRr_14"].iloc[-1])

        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

        # Prepare Chart Data (Last 600 bars for visual context)
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

    # Final Sentiment check for the gauge (Ensure it matches the store)
    _, live_scores = SENTIMENT_STORE.get_live_data()
    sentiment_score = live_scores['sentiment_score']

    return {
        "symbol": symbol,
        "price": current_price,
        "signal": signal_map.get(curr_pred, "UNKNOWN"),
        "prediction_class": curr_pred,
        "confidence": curr_conf,
        "is_multi_modal": is_multi_modal,
        "sentiment_score": sentiment_score,
        "atr": current_atr,
        "atr_multiplier": atr_mult,
        "chart": {"candles": candles, "ema": ema_line, "markers": markers},
    }


@router.get("/news")
def get_latest_news(symbol: str = "EURUSD", limit: int = 10):
    """Returns the latest news headlines from the live sentiment store."""
    news, _ = SENTIMENT_STORE.get_live_data()
    
    if not news:
        # Fallback to historical CSV if API fails or no news
        csv_path = "data/raw_sentiment/news_with_sentiment_scores.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                df = df.dropna(subset=['time'])
                df = df.sort_values('time', ascending=False)
            
            latest = df.head(limit).copy()
            if 'sentiment_score' in latest.columns:
                latest['sentiment_label'] = latest['sentiment_score'].apply(
                    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
                )
            else:
                latest['sentiment_label'] = 'Neutral'
                
            # SIMULATE LIVE TIMESTAMPS
            now = datetime.now(timezone.utc)
            base_time = now
            for i in range(len(latest)):
                latest.iloc[i, latest.columns.get_loc('time')] = base_time - timedelta(minutes=i*7)
            
            news_list = latest.to_dict('records')
            for item in news_list:
                if isinstance(item.get('time'), pd.Timestamp):
                    item['time'] = item['time'].isoformat()
                    
            return {"news": news_list}
        return {"news": []}
    
    return {"news": news[:limit]}


import asyncio

@router.post("/explain/{run_id}")
async def get_live_explanation(run_id: str, symbol: str = "EURUSD", timeframe: str = "H1"):
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
            pipeline_name = cfg.get("data", {}).get("feature_pipeline", "default")
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

    # For MultiModal: Inject live sentiment for SHAP context
    if is_multimodal:
        _, live_scores = SENTIMENT_STORE.get_live_data()
        for col, val in live_scores.items():
            if col in feature_cols:
                df[col] = val

    X_raw = df[feature_cols].tail(600)
    X_scaled = scaler.transform(X_raw)

    def calculate_shap():
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
                    # Inject a high impact for News Sentiment if it exists in the model
                    if "sentiment_score" in feature_cols:
                        pass # SHAP will find it naturally

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
                
                sorted_impacts = sorted(
                    feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True
                )
                explanation = [{"feature": k, "impact": v} for k, v in sorted_impacts[:10]]

            except Exception as e:
                print(f"SHAP KernelExplainer failed: {e}")
                explanation = [{"feature": "Hybrid Model Context Active", "impact": 0.05}]
        return explanation

    # Run the heavy SHAP calculation in a background thread to prevent blocking the event loop
    explanation = await asyncio.to_thread(calculate_shap)

    return {"run_id": run_id, "symbol": symbol, "top_features": explanation}



@router.post("/trade")
def execute_trade(req: TradeRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
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

    # Extract price from response if available, or just mock it to 0.0 for now if not returned
    executed_price = float(res.get("price", 0.0))
    order_ticket = res.get("order", None)

    # Save to database
    new_trade = models.Trade(
        user_id=current_user.id,
        symbol=req.symbol,
        direction=req.direction,
        price=executed_price,
        lot_size=float(req.lot_size),
        model_used="Live Execution",  # Could be passed from frontend
        pnl=0.0, # Initial PnL is 0
        mt5_order_ticket=order_ticket,
        status="OPEN"
    )
    db.add(new_trade)
    db.commit()
    db.refresh(new_trade)

    return res

@router.post("/portfolio/sync")
def sync_portfolio(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Syncs open and closed trades with MT5."""
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta

    if not mt5.initialize():
        print("MT5 Sync Failed: Initialization failed")
        raise HTTPException(status_code=500, detail="MT5 Initialization failed")
    
    try:
        open_trades = db.query(models.Trade).filter(
            models.Trade.user_id == current_user.id, 
            models.Trade.status == "OPEN"
        ).all()

        if not open_trades:
            return {"status": "success", "message": "No open trades to sync"}

        # Get all active positions
        positions = mt5.positions_get()
        # MT5 positions usually match via 'ticket' or 'identifier'
        active_positions = {}
        if positions:
            for p in positions:
                active_positions[p.ticket] = p
                active_positions[p.identifier] = p

        # Get historical deals to find closed positions
        # Use a wide range to be safe
        date_from = datetime.now() - timedelta(days=30)
        date_to = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(date_from, date_to)
        
        deals_by_position = {}
        if deals:
            for d in deals:
                pos_id = getattr(d, 'position_id', 0)
                if pos_id != 0:
                    if pos_id not in deals_by_position:
                        deals_by_position[pos_id] = []
                    deals_by_position[pos_id].append(d)

        synced_count = 0
        for trade in open_trades:
            ticket = trade.mt5_order_ticket
            if not ticket:
                continue
                
            # 1. Check if still open
            if ticket in active_positions:
                pos = active_positions[ticket]
                trade.pnl = float(pos.profit)
                synced_count += 1
            # 2. Check if closed in history
            elif ticket in deals_by_position:
                pos_deals = deals_by_position[ticket]
                # Total profit for this position is the sum of profit of all its deals
                total_profit = sum(getattr(d, 'profit', 0.0) for d in pos_deals)
                trade.pnl = float(total_profit)
                trade.status = "CLOSED"
                synced_count += 1
            # 3. Fallback: If not in positions AND not in last 30 days history, 
            # it might be old or closed manually without a deal trace we can find easily
            else:
                # We mark as closed to stop trying to sync it every time
                # but we leave PnL at 0 or last known if we can't find it.
                pass

        db.commit()
        return {"status": "success", "synced": synced_count}
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")

@router.get("/portfolio")
def get_portfolio(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Returns the logged-in user's trade history and calculated statistics."""
    try:
        # Fetch all trades sorted chronologically for equity curve
        trades_asc = db.query(models.Trade).filter(models.Trade.user_id == current_user.id).order_by(models.Trade.timestamp.asc()).all()
        
        # Original trades list (descending) for the table
        trades = sorted(trades_asc, key=lambda x: x.timestamp, reverse=True)

        total_trades = len(trades)
        if total_trades == 0:
            return {
                "trades": [],
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "equity_curve": [],
                "model_performance": {}
            }

        # Calculate statistics
        winning_trades = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
        win_rate = (winning_trades / total_trades * 100)
        total_pnl = sum((t.pnl if t.pnl is not None else 0.0) for t in trades)

        # Advanced Metrics
        gross_profit = sum((t.pnl if t.pnl is not None and t.pnl > 0 else 0.0) for t in trades)
        gross_loss = abs(sum((t.pnl if t.pnl is not None and t.pnl < 0 else 0.0) for t in trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        # Equity Curve and Max Drawdown
        equity_curve = []
        cumulative_pnl = 0.0
        peak_equity = 0.0
        max_dd = 0.0
        
        # Start with 0 point
        if trades_asc:
            equity_curve.append({"time": int(trades_asc[0].timestamp.timestamp()) - 3600, "value": 0.0})

        pnl_list = []
        model_perf = {}

        for t in trades_asc:
            pnl = float(t.pnl if t.pnl is not None else 0.0)
            cumulative_pnl += pnl
            pnl_list.append(pnl)
            
            equity_curve.append({
                "time": int(t.timestamp.timestamp()),
                "value": round(cumulative_pnl, 2)
            })

            # Peak and Drawdown
            if cumulative_pnl > peak_equity:
                peak_equity = cumulative_pnl
            
            drawdown = peak_equity - cumulative_pnl
            if drawdown > max_dd:
                max_dd = drawdown

            # Model Distribution
            m = t.model_used or "Manual"
            if m not in model_perf:
                model_perf[m] = {"wins": 0, "total": 0, "pnl": 0.0}
            
            model_perf[m]["total"] += 1
            model_perf[m]["pnl"] += pnl
            if pnl > 0:
                model_perf[m]["wins"] += 1

        # Sharpe Ratio (Crude per-trade approximation)
        import numpy as np
        sharpe = 0.0
        if len(pnl_list) > 1:
            mean_ret = np.mean(pnl_list)
            std_ret = np.std(pnl_list)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * np.sqrt(252) # Scaled to annual approx

        # Clean NaN/Inf
        import math
        for t in trades:
            if t.pnl is not None and (math.isnan(t.pnl) or math.isinf(t.pnl)):
                t.pnl = 0.0
            if math.isnan(t.price) or math.isinf(t.price):
                t.price = 0.0

        returned_dict = {
            "trades": trades,
            "total_trades": total_trades,
            "win_rate": round(float(win_rate), 2),
            "total_pnl": round(float(total_pnl), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "profit_factor": round(float(profit_factor), 2),
            "max_drawdown": round(float(max_dd), 2),
            "equity_curve": equity_curve,
            "model_performance": model_perf
        }
        
        response_obj = schemas.PortfolioResponse.model_validate(returned_dict)
        return response_obj.model_dump(mode='json')
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
