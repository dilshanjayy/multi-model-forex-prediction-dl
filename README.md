# Multi-Modal Forex Prediction DL: Quantitative Trading Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Status](https://img.shields.io/badge/Status-Academic_Complete-success)](https://github.com/)

A professional-grade, multi-modal hybrid deep learning system designed for EUR/USD forecasting and execution. This platform fuses state-of-the-art neural architectures (Transformers, CNN-LSTMs) with real-time FinBERT sentiment analysis, providing a transparent "human-in-the-loop" dashboard for institutional-grade quantitative trading.

Developed as a Final Year Project for **NSBM Green University** in partnership with **Plymouth University** (Module: PUSL3190).

## 🚀 Key Features

*   **Multi-Modal Fusion:** Dynamically combines 19 technical market features with unstructured news sentiment vectors extracted via a pre-trained **FinBERT** transformer.
*   **Advanced Architectures:** Evaluates a spectrum of models from Random Forest baselines to hybrid **CNN-LSTM** (spatial-temporal extraction) and **Transformer** (attention-based sequencing) architectures.
*   **Continuous Signal Execution (CSE):** A sophisticated execution strategy that transitions portfolio states dynamically based on model confidence, successfully overcoming high-frequency "commission traps."
*   **Explainable AI (XAI):** Real-time **SHAP Model X-Ray** visualization, utilizing asynchronous threading (`asyncio.to_thread`) to provide deep interpretability without UI latency.
*   **Robustness & Marker-Proofing:** Integrated **News API Fallback** mechanism that automatically switches to a high-fidelity simulation mode with historical data if external APIs expire.
*   **Institutional Dashboard:** A low-latency React 19 interface featuring TradingView charts, persistent portfolio tracking via SQLite, and **Role-Based Access Control (RBAC)** for administrative model management.

## 🛠️ Tech Stack

*   **Backend:** FastAPI (ASGI), SQLAlchemy (ORM), MetaTrader 5 Python API.
*   **AI/ML:** PyTorch 2.1 (Deep Learning), HuggingFace (FinBERT), SHAP (XAI), Optuna (HPO).
*   **Data:** Pandas, Pandas-TA (Technical Indicators), PyArrow (Parquet Storage).
*   **Frontend:** React 19, Zustand (State), Tailwind CSS, Lightweight Charts (TradingView).

## 📁 Project Structure

```text
├── backend/            # FastAPI orchestration and REST API
├── configs/            # YAML-driven experiment configurations
├── data/               # Raw and processed market/sentiment datasets
├── deployed_models/    # Validated model weights and artifacts
├── frontend/           # React 19 SPA (Vite build)
├── report/             # Full Academic Thesis & Documentation (Markdown)
├── src/                # Core Quantitative Library
│   ├── data/           # Market/News collectors and multi-modal processors
│   ├── evaluation/     # Backtester engine and HPO optimizers
│   ├── execution/      # Live MT5 order execution logic
│   ├── models/         # PyTorch architecture definitions
│   └── strategies/     # CSE and Triple-Barrier algorithmic logic
└── main.py             # CLI entry point for the end-to-end ML pipeline
```

## 🏁 Getting Started

### Prerequisites

*   Python 3.10+
*   Node.js & npm
*   **Windows OS** (Required for the `MetaTrader5` runtime library)
*   MT5 Terminal installed and logged into a demo account.

### Quick Start (Live Dashboard)

1.  **Backend Setup:**
    ```bash
    python -m venv env
    .\env\Scripts\activate
    pip install -r requirements.txt
    python backend/main.py
    ```

2.  **Frontend Setup:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  **Access Dashboard:** Navigate to `http://localhost:5173`. Use the Admin credentials to access model hot-reloading features.

## 📊 Academic Evaluation

This project achieved rigorous performance benchmarks:
*   **Nguyen et al. (2024) Replication:** Successfully achieved a **4.65 Profit Factor** on the Daily (D1) validation set.
*   **Multi-Modal Alpha:** The Sentiment Transformer achieved a **+3.62% Return** on an out-of-sample H1 test set, proving the efficacy of NLP sentiment in filtering market noise.

Full detailed chapters, diagrams (UML Class, Use Case, Sequence), and UAT results are available in the `/report` directory.

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
