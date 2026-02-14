# Market Intelligence: WTI Volatility → S&P 500 Prediction

Predictive analysis of how Asian and European market signals propagate to the
US equity market during WTI crude oil volatility events.

## Quick Start

```bash
pip install -r requirements.txt

python -m src.train    --config configs/v1.yaml   # train + CV + holdout → reports/
python -m src.evaluate --config configs/v1.yaml   # 8 figures → reports/fig_*.png
```

## Project Structure

```
market-intelligence/
├── data/
│   ├── raw/                             # Original CSVs (from Google Sheets)
│   ├── processed/                       # Generated: events_clean.parquet|csv
│   └── data_dictionary.md
├── src/
│   ├── config.py                        # YAML loader
│   ├── data_loader.py                   # Load CSV → clean → save
│   ├── features.py                      # Returns, lags, cross-market spreads
│   ├── train.py                         # CV + holdout + threshold tuning
│   └── evaluate.py                      # 8 figures (including PR curve)
├── reports/                             # Generated outputs
│   ├── latest_metrics.md                # Full report with all tables
│   ├── model_summary.csv               # CV summary
│   ├── holdout_results.json             # Final test results
│   ├── thresholds.json                  # Tuned thresholds per model
│   ├── confusion_matrices_cv.json       # Aggregated CMs
│   ├── pr_curves.json                   # PR curve data
│   └── fig_*.png                        # 8 figures
├── configs/v1.yaml                      # All thresholds + hyperparameters
├── .github/workflows/train-model.yml    # CI/CD
└── requirements.txt
```

## Dataset

- **50 WTI volatility events** (|daily change| > 4%), 1986–2025
- **1,436 raw rows** → **1,283 cleaned** → **704 model-ready** (25 events)
- **23 features**: returns, lags, cross-market spreads
- **Target**: `sign(S&P 500 return[t+1])` — next-day direction
- **No same-day leakage**: features at day *t* predict day *t+1*

## Methodology

1. **Validation**: Event-based temporal CV (5 folds on first 20 events) +
   **final holdout** (last 5 events, never seen during CV).
2. **Primary metric**: **macro-F1** (handles 65/35 class imbalance).
   Secondary: F1_up (catch rebounds), AUC (ranking quality).
3. **Threshold tuning**: optimize F1_up on CV, apply to holdout.
4. **Baselines**: naive persistence (today→tomorrow) + majority class.

## Results

### Cross-Validation (20 events)

| Model               | Accuracy | F1 macro | F1 up | AUC   |
|:---------------------|:--------:|:--------:|:-----:|:-----:|
| Logistic Regression  |   0.590  |  **0.521** |  0.357  | 0.567 |
| Random Forest        |   0.664  |  0.516   |  0.248  | **0.594** |
| Majority baseline    |   0.661  |  0.398   |  0.000  |   —   |
| Naive baseline       |   0.504  |  0.443   |  0.259  |   —   |

### Final Holdout (last 5 events — untouched)

| Model               | Accuracy | F1 macro | F1 up | AUC   |
|:---------------------|:--------:|:--------:|:-----:|:-----:|
| Random Forest        |   0.596  |  **0.535** |  0.368  | **0.618** |
| Gradient Boosting    |   0.559  |  0.529   |  0.412  | 0.522 |
| Logistic Regression  |   0.412  |  0.373   |  **0.529** | 0.502 |
| Majority baseline    |   0.618  |  0.382   |  0.000  |   —   |

### Threshold Tuning (optimized F1_up)

With tuned thresholds, all models reach F1_up ≈ 0.53–0.56 on holdout (vs
0.37–0.53 at default threshold 0.5).

## Key Findings

1. During WTI volatility events, the S&P 500 declines in **~65%** of next-day
   observations. The majority baseline wins accuracy by design.

2. **Random Forest** outperforms on macro-F1 and AUC — a modest but real signal
   from VIX returns, STOXX 50 returns, and Nikkei/EUR cross-market indicators.

3. **If the objective is catching rebounds**, Logistic Regression offers a
   better trade-off (F1_up ≈ 0.53 on holdout), at the cost of raw accuracy.

4. Signal is **modest** (AUC ~0.59–0.62), consistent with the difficulty of
   next-day financial direction prediction, but defensible against baselines.

## Author

Diego Salcedo Flores — Universidad de Los Andes, Venezuela
Supervised by Dr. Arturo Sánchez Pineda — INAIT SA, Lausanne
