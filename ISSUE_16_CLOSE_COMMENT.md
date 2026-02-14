## Cierre — Inteligencia de Mercado (entregable de pasantía)

Entregable incorporado en `deliverables/market-intelligence/` (pipeline reproducible + datos + reportes).

### 1) Dataset (trabajo original de la pasantía)
- Se parte de **50 eventos** (ventanas alrededor de episodios de alta volatilidad WTI).
- Tras limpieza y criterios mínimos (WTI+S&P presentes por fila y reglas de missingness), quedan **45 eventos utilizables**.
  - Evidencia: `deliverables/market-intelligence/reports/missingness.csv` y `data/processed/events_clean.parquet`.

### 2) ¿Por qué el modelo entrena con 25 eventos y no con los 50?
Para hacer *modeling* reproducible sin “trucos”:
- Se define un **set fijo de features** (retornos, lags y spreads inter-mercado) que debe existir para **todas** las muestras.
- Algunos eventos quedan con huecos (por faltantes en columnas clave o por pérdida en lags al inicio de ventanas).
- En lugar de imputar agresivamente (riesgo de sesgo/leakage), el pipeline filtra a un subconjunto **consistente**:
  - **Model data**: 25 eventos (719 muestras × 23 features).
  - **Validación**: 20 eventos en CV (5 folds, temporal por eventos).
  - **Holdout final**: 5 eventos (intocados), para cierre.

(Estos números se imprimen al correr `python -m src.train --config configs/v1.yaml` y quedan trazados en `reports/`.)

### 3) Evidencia entregada (qué mirar)
- `deliverables/market-intelligence/reports/latest_metrics.md`  
  Métricas de CV + holdout y umbrales sugeridos por modelo (optimización F1↑ en CV).
- Figuras:  
  `fig_model_comparison.png`, `fig_feature_importance.png`, `fig_pr_curve.png`,  
  `fig_confusion_matrix_cv.png`, `fig_confusion_matrix_holdout.png`.

### 4) Conclusión formal de cierre
- La señal predictiva es **moderada** (AUC ~0.59 en CV y ~0.62 en holdout).
- La **exactitud** sola es engañosa por el desbalance (baseline mayoritaria ~0.66); por eso la métrica principal es **F1 macro**.
- Para **balance global**, Random Forest suele rendir mejor en CV.  
  Para **capturar días “Up”**, los umbrales del reporte permiten ajustar precisión vs recall.

### Reproducibilidad
Desde `deliverables/market-intelligence/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/v1.yaml
python -m src.evaluate --config configs/v1.yaml
```

CI: `.github/workflows/market-intelligence.yml` ejecuta el pipeline y publica `reports/` como artefacto.

Marcando como completado el objetivo del proyecto: **modelado predictivo + evaluación reproducible + evidencia de cierre**.
