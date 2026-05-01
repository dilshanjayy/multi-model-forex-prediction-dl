# Project Completion Roadmap (2-Week Sprint)

## Phase 1: Core Platform & User Management (Days 1-5)
- [ ] **Database Setup:** Integrate SQLite/PostgreSQL using SQLAlchemy in FastAPI.
- [ ] **User Authentication:** Implement JWT (JSON Web Token) login/registration.
- [ ] **Trade Database:** Create a `trades` table to record Buy/Sell actions with `user_id`, `price`, `timestamp`, and `model_used`.
- [ ] **Portfolio Dashboard (Frontend):** Build a React page to display the user's trade history, Win Rate, and total PnL.

## Phase 2: Multi-Modal Finalization (Days 6-10)
- [ ] **News Pipeline:** Automate news fetching (e.g., via NewsAPI or Finnhub) or finalize the ForexFactory scraper.
- [ ] **Sentiment Extraction:** Run historical headlines through a pre-trained NLP model (like FinBERT or VADER) to generate numerical sentiment scores (-1 to +1).
- [ ] **Final Model Training:** Merge the sentiment scores into `technical_features.parquet`, update the `input_dim`, and train the final `Alpha_V1_Hybrid_CNNLSTM` model.

## Phase 3: "Wow Factor" Polish (Days 11-14)
*Choose 1 or 2 based on remaining time:*
- [ ] **Model Leaderboard:** Create a "Model Arena" tab in the UI that reads `stats_validation.json` and ranks deployed models by Sharpe Ratio and Return.
- [ ] **AI Auto-Pilot:** Add a toggle to automatically execute paper trades in the database when the active model generates a strong signal.
- [ ] **True WebSockets:** Upgrade the frontend from 2-second HTTP polling to a live WebSocket connection using the existing `websockets.py` file.

## Final Review
- [ ] Code cleanup and documentation.
- [ ] Record presentation/demo video.
- [ ] Finalize written thesis/report.
