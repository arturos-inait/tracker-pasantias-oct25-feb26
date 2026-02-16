# Cierre de pasantía — Inteligencia de Mercado (Oct 2025 – Feb 2026)

Esta carpeta contiene el pipeline reproducible y los artefactos finales de evaluación del proyecto
**Inteligencia de Mercado / spillover**.

## Qué incluye
- `src/` código reproducible de entrenamiento y evaluación (CV temporal por eventos + holdout final)
- `configs/v1.yaml` configuración (features, splits, definición del objetivo)
- `data/` (datos *raw* de la pasantía y *processed* utilizados en la corrida final)
- `reports/` figuras, métricas y matrices de confusión generadas
- `requirements.txt` dependencias

## Resultados finales (qué citar)
Artefactos canónicos:
- `reports/latest_metrics.md` — métricas consolidadas (CV + holdout) y umbrales recomendados
- `reports/fig_model_comparison.png` — comparación de modelos (Exactitud / F1 macro / AUC)
- `reports/feature_importance_rf.csv` y `reports/feature_importance_logreg.csv` — interpretabilidad (top features)
- `reports/fig_confusion_matrix_cv.png` y `reports/fig_confusion_matrix_holdout.png` — análisis de errores
- `reports/fig_pr_curve.png` — trade-off precisión/recall para la clase “Up”

## Reproducibilidad (local)
Desde esta carpeta:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/v1.yaml
python -m src.evaluate --config configs/v1.yaml
```

Las salidas se escriben en `reports/`.

## Nota para este repositorio “tracker”
Existe una GitHub Action en la raíz del repo: `.github/workflows/market-intelligence.yml`.
Ejecuta el pipeline desde esta carpeta y sube `reports/` como artefacto de CI.


## Uso de los 50 eventos (trazabilidad)
- **50 eventos** originales (CSV raw).
- **45 eventos** tras limpieza y criterios mínimos (datos suficientes).
- **25 eventos** con cobertura completa para el set de features definido (modelo).
  - De esos 25: **20** se usan en CV y **5** se reservan como holdout final.

Motivo: privilegiar un entrenamiento reproducible con features consistentes y evitar imputaciones agresivas
que introduzcan sesgo o leakage. La justificación detallada queda documentada en:
`reports/missingness.csv` y en el log/artefactos generados por `src.train`.
