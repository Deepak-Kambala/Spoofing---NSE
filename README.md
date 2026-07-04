# MarketGuard: ML-Based Order Book Manipulation Detection

Detects spoofing-like manipulative order placement in electronic markets using Isolation Forest, XGBoost, and SHAP explainability.

## What's in the notebook

1. **Setup** — imports (numpy, pandas, sklearn, xgboost, shap, matplotlib/seaborn).
2. **Synthetic Order Book Simulator** — multi-agent limit order book simulator (`OrderBookSimulator`) generating normal traders and spoofers, each driven by latent traits (size, distance from touch, cancel behavior).
3. **Dataset Generation** — runs 25 simulated trading sessions, saves raw event log to `marketguard_simulated_events.csv`.
4. **Order Lifecycle Reconstruction & Feature Engineering** — rebuilds each order's lifecycle and derives participant-level microstructure features (`cancel_ratio`, `avg_lifetime`, `fill_ratio`, `avg_order_size`, `avg_dist_from_touch`, `avg_imbalance_at_submit`, etc.).
5. **Exploratory Analytics** — feature distributions by participant type, order book depth / mid-price visualization highlighting spoofing bursts.
6. **Unsupervised Detection (Isolation Forest)** — anomaly scoring on unlabeled participant-session profiles, plus a single-feature ROC-AUC sanity check to rule out trivial threshold rules.
7. **Supervised Classification (XGBoost)** — trained on labeled data, benchmarked against Isolation Forest via ROC curves.
8. **Explainable AI Layer (SHAP)** — SHAP summary plots and per-participant explanations for flagged risk scores.
9. **Risk Scoring Dashboard** — static top-20 highest-risk participant visualization.
10. **From-Scratch Models** — numpy-only reimplementation of Isolation Forest and boosting, evaluated on unseen simulated sessions to confirm results aren't an artifact of library internals.

## Requirements

```
numpy pandas matplotlib seaborn scikit-learn xgboost shap
```

## Outputs

- `marketguard_simulated_events.csv` — raw simulated order-level event log
- `shap_summary.png` — SHAP feature importance summary plot
