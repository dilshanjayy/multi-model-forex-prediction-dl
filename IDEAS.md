# Future Research & Experimentation Ideas

This file tracks advanced concepts and architectural changes to test for improving the model's Alpha and Risk/Reward profile.

## 1. Dual-Simulation Labeling (Asymmetric 2:1 R:R)
**Status:** Backlog / High Priority
**Concept:** Instead of a single 1:1 barrier, simulate two independent trades (Long and Short) for every row to identify high-conviction 2:1 setups.

### Logic:
*   **Label 1 (Buy Setup)**: Price hits `+2.0 * ATR` before it hits `-1.0 * ATR` or a 24h timeout.
*   **Label -1 (Sell Setup)**: Price hits `-2.0 * ATR` before it hits `+1.0 * ATR` or a 24h timeout.
*   **Label 0 (Neutral)**: All other cases (hits stop-loss first or times out).

### Pros:
*   Explicitly trains the model to find high-reward, low-risk entries.
*   Filters out "noisy" 1:1 moves that are prone to mean-reversion.

### Cons / Challenges:
*   **Extreme Class Imbalance**: Most rows will be Label 0. Requires downsampling or SMOTE.
*   **Signal Scarcity**: The model will trade much less frequently.

---

## 2. Meta-Labeling (Triple Barrier + Secondary Model)
**Status:** Concept
**Concept:** Use the primary model to predict direction (Label 0/1) and a secondary model to predict "Bet Size" or "Probability of Success."
*   Primary model: "Should I go Long or Short?"
*   Secondary model: "Is this specific 1:1 signal strong enough to run to 2:1?"

---

## 3. Dynamic Horizon Scaling
**Status:** Concept
**Concept:** Instead of a fixed 5h or 24h horizon, scale the vertical barrier based on the **Choppiness Index**.
*   High Chop = Shorter Horizon (get out fast).
*   Low Chop (Trending) = Longer Horizon (let the trade run).

---

## 4. Regime-Specific Training
**Status:** Concept
**Concept:** Train separate model heads (or separate models) for different volatility regimes (e.g., London Open vs. Asian Session).
