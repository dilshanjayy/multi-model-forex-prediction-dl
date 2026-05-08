# Sentiment Integration Strategy: From Text to Alpha

This document outlines the strategic roadmap for integrating the FinBERT-scored news dataset into the Multi-Model Forex Prediction system.

---

## 1. Available Integration Methods

Based on the research analyzed (Dakalbab et al. 2025, Singh et al. 2024), we have identified three primary architectural paths:

### Method 1: Feature Concatenation (Baseline)
*   **Concept:** Treat sentiment as just another technical indicator.
*   **Workflow:** Aggregate `sentiment_score` to H1 and append it as a column to the technical feature vector.
*   **Best For:** Random Forest and standard LSTM models.
*   **Pros:** Zero architectural changes required.
*   **Cons:** Fails to capture the "event-driven" nature of news (dilution of signal).

### Method 2: Multi-Modal "Late Fusion" (Hybrid)
*   **Concept:** Parallel processing of price and news.
*   **Workflow:** 
    *   **Stream A (CNN):** Extracts spatial patterns from OHLCV.
    *   **Stream B (LSTM):** Tracks the temporal trend of sentiment.
    *   **Fusion:** Concatenate high-level features from both streams before the final classification head.
*   **Pros:** Model learns specific "Regimes" (e.g., High News Impact vs. Technical Mean Reversion).

### Method 3: Cross-Modal Attention (State-of-the-Art)
*   **Concept:** Dynamic alignment of modalities.
*   **Workflow:** Use a Transformer-based Attention layer to allow the Technical stream to "attend" to the Sentiment stream.
*   **Pros:** Highest accuracy; provides explainability (which news event triggered the trade).
*   **Cons:** Computationally expensive; requires careful temporal synchronization.

---

## 2. Implementation Roadmap

### Phase A: The Dataset (Dynamic Aggregation)
**Status:** Ready to Implement
*   Update `src/data/data_module.py` to handle the `news_with_sentiment_scores.csv`.
*   **Logic:** Implement a `resample_sentiment(timeframe)` function.
*   **Goal:** Ensure that for any timeframe (M15, H1, D1), every price bar has an associated average sentiment probability vector.

### Phase B: The Model (CNNLSTMSentimentHybrid)
**Status:** Concept
*   Implement a new PyTorch class in `src/models/`.
*   **Architecture:** Dual-input head.
    *   Input 1: `(Batch, Lookback, Tech_Features)`
    *   Input 2: `(Batch, Lookback, Sentiment_Features)`
*   **Loss Function:** Implement `SentimentWeightedCrossEntropy` to punish the model more for missing trades during high-impact news events.

---

## 3. Current Dataset Context (As of May 2026)
*   **Headlines:** 20,966 (Cleaned & Filtered).
*   **Model:** `ProsusAI/finbert` (Financial specialized).
*   **Range:** April 2021 – May 2026.
*   **Alignment:** 100% UTC synchronized with MT5 market data.

---

## 4. Key Research References
*   **Dakalbab et al. (2025):** Advancing Forex prediction through multimodal text-driven model and attention mechanisms.
*   **Farimani et al. (2024):** Adaptive weighting for news and market mood.
*   **Nguyen et al. (2024):** CNN-LSTM architectures for stationarity-preserving returns.
