# Internship closure deliverable — Market Intelligence (Oct 2025 – Feb 2026)

This folder contains the final reproducible pipeline and the final evaluation artifacts for the **Market Intelligence / spillover** project.

## What is included
- `src/` reproducible training + evaluation code (time-series CV + holdout)
- `configs/v1.yaml` configuration (features, splits, target definition)
- `data/` (raw and processed versions used for the final run)
- `reports/` generated figures, metrics, and confusion matrices
- `requirements.txt` pinned dependencies

## Final results (what to cite)
Use these artifacts as the canonical outputs:
- `reports/latest_metrics.md` — consolidated metrics (CV + holdout) and recommended thresholds
- `reports/fig_model_comparison.png` — metric comparison across models
- `reports/fig_feature_importance_rf.csv` / `reports/feature_importance_rf.csv` — model explainability
- `reports/fig_confusion_matrix_cv.png` and `reports/fig_confusion_matrix_holdout.png` — error analysis
- `reports/fig_pr_curve.png` — class 'Up' precision/recall trade-off

## Reproducibility
From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/v1.yaml
```

Outputs are written to `reports/`.

## Notes for the tracker repository
A GitHub Action is provided at the repository root: `.github/workflows/market-intelligence.yml`.
It runs the pipeline from this folder and uploads the `reports/` directory as a CI artifact.