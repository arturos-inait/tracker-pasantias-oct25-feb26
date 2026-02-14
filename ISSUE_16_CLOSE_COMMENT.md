## Closure — Market Intelligence (internship deliverable)

Deliverable committed in `deliverables/market-intelligence/` (pipeline + data + reports).

**Key artifacts**
- `deliverables/market-intelligence/reports/latest_metrics.md` (CV + holdout metrics + threshold suggestions)
- Figures: `fig_model_comparison.png`, `fig_feature_importance.png`, `fig_pr_curve.png`, `fig_confusion_matrix_cv.png`, `fig_confusion_matrix_holdout.png`

**Interpretation (executive)**
- Evidence of predictive signal is **modest** (AUC around ~0.59 on CV, ~0.62 on holdout).
- Accuracy alone is misleading due to class imbalance; macro-F1 is the primary metric.
- Random Forest is the best balanced option (macro-F1 ~0.52 CV; ~0.54 holdout).
- If we care about *capturing 'Up' days*, thresholds from the tuning section can be applied to trade precision vs recall.

**Reproducibility**
Run: `python -m src.train --config configs/v1.yaml` from inside `deliverables/market-intelligence/`.

A CI workflow (`.github/workflows/market-intelligence.yml`) retrains on changes and publishes `reports/` as an artifact.

Marking the internship objective “predictive modeling + evaluation” as completed.
