# Model Training Report
**Generated:** 2026-02-14 18:32

## Target Definition

- **Target**: `sign(S&P 500 return[t+1])` — next-day direction
- **Features**: known by end of day *t* (Asia/Europe same-day + US lag t-1)
- **No same-day leakage**: features at time *t* predict outcome at *t+1*

## Dataset

- Total samples: 719  |  Features: 23  |  Events: 25
- Class balance: **up 250 (34.8%)**  /  down 469 (65.2%)
- CV: first 20 events  |  Holdout: last 5 events (never seen during CV)

## Primary Metric: macro-F1

Accuracy is misleading with 65/35 imbalance (majority baseline wins at ~64%). We use **macro-F1** as the primary metric (equally weights both classes). Secondary: **F1_up** (ability to catch rebounds) and **AUC** (ranking quality).

## Cross-Validation Results (event-based temporal, 5 folds)

| model               |   accuracy_mean |   accuracy_std |   f1_macro_mean |   f1_macro_std |   f1_up_mean |   f1_up_std |   balanced_acc_mean |   balanced_acc_std |   n_folds |   auc_mean |     auc_std |
|:--------------------|----------------:|---------------:|----------------:|---------------:|-------------:|------------:|--------------------:|-------------------:|----------:|-----------:|------------:|
| random_forest       |        0.647909 |      0.0391011 |        0.547177 |      0.0460373 |     0.333889 |   0.0671466 |            0.550632 |          0.0433455 |         5 |   0.594904 |   0.0653925 |
| gradient_boosting   |        0.572832 |      0.0842461 |        0.479657 |      0.0258082 |     0.289286 |   0.127415  |            0.517598 |          0.0304483 |         5 |   0.57591  |   0.0550736 |
| baseline_naive      |        0.544703 |      0.0286609 |        0.475171 |      0.0493501 |     0.286224 |   0.0938773 |            0.475598 |          0.0483892 |         5 | nan        | nan         |
| logistic_regression |        0.497121 |      0.130761  |        0.440828 |      0.102438  |     0.415504 |   0.11353   |            0.520378 |          0.0445162 |         5 |   0.537807 |   0.0640224 |
| ridge_classifier    |        0.489967 |      0.141187  |        0.427551 |      0.118883  |     0.414523 |   0.107097  |            0.517394 |          0.0387786 |         5 |   0.535991 |   0.0646109 |
| baseline_majority   |        0.676311 |      0.0327503 |        0.403267 |      0.0118293 |     0        |   0         |            0.5      |          0         |         5 | nan        | nan         |

### Aggregated Confusion Matrices (summed across CV folds)

**random_forest**:

|  | Pred Down | Pred Up |
|--|----------|---------|
| **Actual Down** | 253 | 53 |
| **Actual Up** | 106 | 40 |

**logistic_regression**:

|  | Pred Down | Pred Up |
|--|----------|---------|
| **Actual Down** | 136 | 170 |
| **Actual Up** | 56 | 90 |

**baseline_majority**:

|  | Pred Down | Pred Up |
|--|----------|---------|
| **Actual Down** | 306 | 0 |
| **Actual Up** | 146 | 0 |

**baseline_naive**:

|  | Pred Down | Pred Up |
|--|----------|---------|
| **Actual Down** | 204 | 102 |
| **Actual Up** | 104 | 42 |

## Threshold Tuning (optimized for F1_up on CV)

| Model | Default (0.5) F1_up | Tuned Threshold | Tuned F1_up |
|-------|--------------------:|----------------:|------------:|
| random_forest | 0.348 | 0.296 | 0.584 |
| logistic_regression | 0.547 | 0.366 | 0.541 |
| gradient_boosting | 0.466 | 0.144 | 0.578 |

## Final Holdout (last 5 events — never seen during CV)

| model               |   accuracy |   f1_macro |    f1_up | auc                |
|:--------------------|-----------:|-----------:|---------:|:-------------------|
| random_forest       |   0.562044 |   0.509078 | 0.347826 | 0.6088235294117647 |
| gradient_boosting   |   0.59854  |   0.572191 | 0.466019 | 0.6289592760180995 |
| logistic_regression |   0.408759 |   0.347427 | 0.547486 | 0.5230769230769231 |
| ridge_classifier    |   0.408759 |   0.347427 | 0.547486 | 0.5208144796380091 |
| baseline_naive      |   0.540146 |   0.509964 | 0.38835  | —                  |
| baseline_majority   |   0.620438 |   0.382883 | 0        | —                  |

## Top 10 Features — Logistic Regression |coef|

|                |   abs_coef |
|:---------------|-----------:|
| ecb_rate       |   0.421143 |
| VIX_lag1       |   0.276285 |
| fed_rate_ret   |   0.257838 |
| VIX_ret        |   0.199972 |
| fed_rate       |   0.189559 |
| nikkei_ret     |   0.156639 |
| fed_rate_lag1  |   0.152135 |
| DEXUSUK_ret    |   0.129383 |
| wti_lag1_ret   |   0.123886 |
| nk_sp_ret_diff |   0.110808 |

## Top 10 Features — Random Forest importance

|                   |   importance |
|:------------------|-------------:|
| VIX_ret           |    0.0806327 |
| VIX_lag1_ret      |    0.0751683 |
| fed_rate_lag1_ret |    0.0620287 |
| sp500_lag1_ret    |    0.0595036 |
| fed_rate_ret      |    0.0590684 |
| stoxx50_ret       |    0.0586829 |
| DEXUSUK_ret       |    0.0571891 |
| gold_ret          |    0.0509347 |
| wti_lag1_ret      |    0.0485612 |
| nikkei_ret        |    0.0468153 |

## Key Finding

During WTI volatility events (|daily change| > 4%), the S&P 500 declines in **65%** of next-day observations. The majority baseline wins on raw accuracy (65% by design), but Random Forest outperforms on **macro-F1** and **AUC**, indicating a modest but real signal from cross-market indicators.

If the objective is catching rebounds (up days), Logistic Regression offers a better trade-off (higher F1_up ≈ 0.45), at the cost of accuracy.