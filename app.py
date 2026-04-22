import streamlit as st
import os
import joblib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from streamlit_lightweight_charts import renderLightweightCharts

# Import project modules
from src.data.live_collector import fetch_live_data
from src.execution.trader import execute_market_order

# --- Dashboard Configuration ---
st.set_page_config(page_title="FX MULTI-MODAL TERMINAL", layout="wide")

# --- Institutional Dark CSS ---
st.markdown(
    """
<style>
    .main { background-color: #06090f; color: #c9d1d9; }
    [data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 4px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
    
    .news-box {
        padding: 12px;
        border-radius: 4px;
        border-left: 4px solid #30363d;
        background-color: #0d1117;
        margin-bottom: 10px;
        font-size: 0.85rem;
    }
    .pos-news { border-left-color: #3fb950; }
    .neg-news { border-left-color: #f85149; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Persistent State ---
if "lot_size" not in st.session_state:
    st.session_state.lot_size = 0.10

# --- Sidebar: Model Registry ---
st.sidebar.markdown("<h2 style='font-size: 1.1rem; letter-spacing: 2px;'>MODEL REGISTRY</h2>", unsafe_allow_html=True)
experiments_dir = "experiments"
all_runs = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
selected_run = st.sidebar.selectbox("ACTIVE INSTANCE", sorted(all_runs, reverse=True))

@st.cache_resource
def load_model_instance(run_name):
    path = os.path.join(experiments_dir, run_name, "model.joblib")
    return joblib.load(path)

# --- Mock NLP Data Generator ---
def get_mock_nlp_data():
    headlines = [
        {"text": "FED Signal Potential Rate Cut in Late 2026", "score": 0.82},
        {"text": "Eurozone Inflation Dips Below 2% Target", "score": -0.45},
        {"text": "US Retail Sales Beat Expectations Amid Labor Strength", "score": 0.65},
        {"text": "Geopolitical Tensions in Middle East Weigh on Risk Sentiment", "score": -0.78}
    ]
    return headlines

# --- Logic: Main Terminal Loop ---
if selected_run:
    artifacts = load_model_instance(selected_run)
    model, scaler, feature_cols = artifacts["model"], artifacts["scaler"], artifacts["feature_cols"]
    atr_mult = artifacts.get("atr_multiplier", 3.0)

    # 1. FETCH REAL DATA
    df = fetch_live_data("EURUSD", "H1", count=500)
    
    if df is not None:
        # 2. RUN INFERENCE
        X_scaled = scaler.transform(df[feature_cols])
        preds = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)
        
        curr_pred = int(preds[-1])
        curr_conf = float(probas[-1][curr_pred])
        current_price = df["Close"].iloc[-1]
        current_atr = df["ATRr_14"].iloc[-1]
        
        # Calculate EMA for Overlay
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # --- ZONE 1: COMMAND CENTER (KPIs) ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        # Get historical stats
        stats_path = os.path.join(experiments_dir, selected_run, "stats_validation.json")
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)

        kpi1.metric("EUR/USD PRICE", f"{current_price:.5f}", delta=f"{current_price - df['Close'].iloc[-2]:.5f}")
        
        sig_label = "STRONG BUY" if curr_pred == 0 and curr_conf > 0.55 else "BUY" if curr_pred == 0 else \
                    "STRONG SELL" if curr_pred == 1 and curr_conf > 0.55 else "SELL" if curr_pred == 1 else "HOLD"
        kpi2.metric("LIVE AI SIGNAL", sig_label)
        
        mock_sentiment = 0.68
        kpi3.metric("FINBERT SENTIMENT", f"{mock_sentiment:.2f}", delta="Bullish")
        
        kpi4.metric("BACKTEST ROI", f"{stats.get('Return [%]', 0):.2f}%", delta=f"{stats.get('Sharpe Ratio', 0):.2f} SR")

        st.markdown("---")

        # --- ZONE 2 & 3: CHART & NLP ---
        col_main, col_side = st.columns([7, 3])

        with col_main:
            with st.container(border=True):
                # Prepare TV Chart Data
                chart_data = df.copy()
                chart_data["unix_time"] = chart_data["time"].apply(lambda x: int(x.timestamp()))
                
                # Candlesticks
                candles = [{"time": int(row["unix_time"]), "open": float(row["Open"]), "high": float(row["High"]), "low": float(row["Low"]), "close": float(row["Close"])} for _, row in chart_data.iterrows()]
                
                # EMA Line
                ema_line = [{"time": int(row["unix_time"]), "value": float(row["EMA_20"])} for _, row in chart_data.iterrows()]
                
                # Markers (Buy/Sell signals on the last 50 bars)
                markers = []
                recent_preds = preds[-50:]
                recent_times = chart_data["unix_time"].iloc[-50:].values
                for i, p in enumerate(recent_preds):
                    if p == 0: # BUY
                        markers.append({"time": int(recent_times[i]), "position": "belowBar", "color": "#3fb950", "shape": "arrowUp", "text": "BUY"})
                    elif p == 1: # SELL
                        markers.append({"time": int(recent_times[i]), "position": "aboveBar", "color": "#f85149", "shape": "arrowDown", "text": "SELL"})

                # TP/SL Dynamic Lines
                dist = float(current_atr * atr_mult)
                tp_val = current_price + dist
                sl_val = current_price - dist
                
                # Create horizontal lines as data series for TV
                tp_line = [{"time": int(d["time"]), "value": tp_val} for d in candles[-15:]]
                sl_line = [{"time": int(d["time"]), "value": sl_val} for d in candles[-15:]]

                chartOptions = {
                    "layout": {"background": {"type": "solid", "color": "#06090f"}, "textColor": "#8b949e"},
                    "grid": {"vertLines": {"visible": False}, "horzLines": {"color": "#1c2128"}},
                    "rightPriceScale": {"borderColor": "#30363d"},
                    "timeScale": {"borderColor": "#30363d", "timeVisible": True},
                    "height": 500,
                }

                seriesConfig = [
                    {"type": "Candlestick", "data": candles, "options": {"upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False, "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"}},
                    {"type": "Line", "data": ema_line, "options": {"color": "#58a6ff", "lineWidth": 1, "title": "EMA 20"}},
                    {"type": "Line", "data": tp_line, "options": {"color": "#3fb950", "lineWidth": 1, "lineStyle": 2, "title": "TP"}},
                    {"type": "Line", "data": sl_line, "options": {"color": "#f85149", "lineWidth": 1, "lineStyle": 2, "title": "SL"}}
                ]
                
                # Note: streamlit-lightweight-charts doesn't explicitly expose 'setMarkers' in the basic render call options easily,
                # but many versions support 'markers' key inside the series data or as a standalone property.
                # If markers fail to render, they are part of the series dictionary.
                seriesConfig[0]["markers"] = markers

                renderLightweightCharts([{"chart": chartOptions, "series": seriesConfig}], key='main_chart')

        with col_side:
            st.markdown("<h3 style='font-size: 1rem; margin-bottom: 20px;'>SENTIMENT HUB</h3>", unsafe_allow_html=True)
            
            # Plotly Gauge for Sentiment
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = mock_sentiment * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "BULLISH SENTIMENT %", 'font': {'size': 14, 'color': '#8b949e'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "#0d1117",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(248,81,73,0.1)'},
                        {'range': [70, 100], 'color': 'rgba(63,185,80,0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': mock_sentiment * 100
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c9d1d9"}, height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<p style='font-size: 0.8rem; color: #8b949e; letter-spacing: 1px;'>LIVE NLP FEED</p>", unsafe_allow_html=True)
            for news in get_mock_nlp_data():
                css_class = "pos-news" if news["score"] > 0 else "neg-news"
                st.markdown(f"""
                    <div class='news-box {css_class}'>
                        {news['text']}<br/>
                        <small style='color: #8b949e;'>FinBERT Polarity: {news['score']:.2f}</small>
                    </div>
                """, unsafe_allow_html=True)

        # --- ZONE 4: PERFORMANCE REPORT ---
        st.markdown("---")
        st.subheader("STRATEGY VALIDATION REPORT")
        
        report_col_chart, report_col_stats = st.columns([2, 1])
        
        with report_col_chart:
            # Synthetic Equity Curve from stats
            # In a real app, we'd parse the full backtest logs. For now, using st.line_chart with random walk from current return.
            base_equity = 10000
            ret = stats.get('Return [%]', 0) / 100
            steps = 100
            strategy_curve = base_equity * (1 + np.linspace(0, ret, steps) + np.random.normal(0, 0.005, steps).cumsum())
            market_curve = base_equity * (1 + np.linspace(0, ret*0.5, steps) + np.random.normal(0, 0.008, steps).cumsum())
            
            curve_df = pd.DataFrame({
                "Strategy": strategy_curve,
                "Buy & Hold": market_curve
            })
            st.line_chart(curve_df, height=300)

        with report_col_stats:
            # Clean Stats Table
            tear_sheet = {
                "Metric": ["Sharpe Ratio", "Max Drawdown", "Win Rate", "Total Trades", "Profit Factor"],
                "Value": [
                    f"{stats.get('Sharpe Ratio', 0):.2f}",
                    f"{stats.get('Max. Drawdown [%]', 0):.2f}%",
                    f"{stats.get('Win Rate [%]', 0):.1f}%",
                    str(stats.get('# Trades', 0)),
                    f"{stats.get('Profit Factor', 0):.2f}"
                ]
            }
            st.table(pd.DataFrame(tear_sheet))

        # Order Panel
        with st.sidebar:
            st.markdown("---")
            st.markdown("<p style='font-size: 0.8rem; color: #8b949e;'>EXECUTION GATEWAY</p>", unsafe_allow_html=True)
            st.session_state.lot_size = st.number_input("VOLUME (LOTS)", 0.01, 1.0, st.session_state.lot_size, 0.01)
            
            cb, cs = st.columns(2)
            if cb.button("EXECUTE BUY", use_container_width=True):
                res = execute_market_order("EURUSD", st.session_state.lot_size, "BUY", current_atr, atr_mult)
                if res["status"] == "success": st.success(f"Buy @ {res['price']}")
                else: st.error(res["message"])

            if cs.button("EXECUTE SELL", use_container_width=True):
                res = execute_market_order("EURUSD", st.session_state.lot_size, "SELL", current_atr, atr_mult)
                if res["status"] == "success": st.success(f"Sell @ {res['price']}")
                else: st.error(res["message"])

    # Auto-Refresh
    import time
    time.sleep(2)
    st.rerun()
else:
    st.info("SELECT A VALID MODEL INSTANCE TO INITIALIZE TERMINAL SESSION.")
