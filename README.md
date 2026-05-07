# Multi-Model Forex Prediction DL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://reactjs.org/)

A professional-grade, multi-modal hybrid deep learning system for Forex price prediction and automated trading. This project combines state-of-the-art neural architectures (Transformers, CNN-LSTMs) with robust financial labeling methods (Triple Barrier, Nguyen et al. 2024) and a modern full-stack dashboard.

## 🚀 Key Features

*   **Hybrid Deep Learning:** Leverages CNN-LSTM models for spatial pattern recognition and temporal sequence tracking.
*   **Advanced Labeling:** Implements the Triple Barrier Method (TBM) with ATR-volatility adjustment and quantile-based labeling from Nguyen et al. (2024).
*   **Modular ML Pipeline:** Config-driven CLI for data collection, feature engineering, model training, and Optuna-based hyperparameter optimization.
*   **Live MT5 Integration:** Real-time data streaming and automated trade execution via MetaTrader 5.
*   **Explainable AI (XAI):** Integrated SHAP support to provide transparency for live model predictions.
*   **Professional Dashboard:** React-based frontend featuring TradingView charts, real-time signal visualization, and portfolio management.

## 🛠️ Tech Stack

*   **Backend:** FastAPI, MT5 Python API, Uvicorn.
*   **ML/AI:** PyTorch (Neural Networks), Scikit-learn (Random Forest), Optuna (HPO), SHAP (Explainability).
*   **Data:** Pandas, Pandas-TA, PyArrow (Parquet).
*   **Frontend:** React 19, Vite, Tailwind CSS, Lightweight Charts (TradingView).

## 📂 Project Structure

```text
├── backend/            # FastAPI server and API routes
├── configs/            # YAML configuration files for experiments
├── data/               # Raw and processed market/news data
├── deployed_models/    # Production-ready model artifacts
├── experiments/        # Experiment logs and trained models
├── frontend/           # React frontend application
├── src/                # Core Python package
│   ├── data/           # Data collectors and processors
│   ├── evaluation/     # Backtesters and optimizers
│   ├── execution/      # MT5 trading logic
│   ├── models/         # Neural network architectures
│   └── strategies/     # Trading strategy implementations
└── main.py             # CLI entry point for the ML pipeline
```

## 🏁 Getting Started

### Prerequisites

*   Python 3.10+
*   Node.js & npm (for frontend)
*   **Windows OS** (required for MetaTrader 5 integration)
*   MetaTrader 5 Terminal installed and logged in.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/multi-model-forex-prediction-dl.git
    cd multi-model-forex-prediction-dl
    ```

2.  **Setup Python Environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # or `env\Scripts\activate` on Windows
    pip install -r requirements.txt
    ```

3.  **Setup Frontend:**
    ```bash
    cd frontend
    npm install
    ```

### Running the System

*   **ML Pipeline:** `python main.py run --config configs/baseline_transformer.yaml`
*   **Web Backend:** `python backend/main.py`
*   **Web Frontend:** `cd frontend && npm run dev`

## 📊 Roadmap

- [x] Hybrid CNN-LSTM Architecture
- [x] ATR-based Triple Barrier Labeling
- [x] SHAP Explainability for Live Inference
- [ ] Full integration of news sentiment signals (In Progress)
- [ ] WebSocket-based real-time data broadcasting
- [ ] Model Arena (Live performance ranking)

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
