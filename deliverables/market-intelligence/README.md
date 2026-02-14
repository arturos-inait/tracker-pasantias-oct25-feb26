# Inteligencia de Mercado: Volatilidad WTI → Predicción S&P 500 (día siguiente)

Análisis predictivo de cómo señales de Asia y Europa se propagan al mercado accionario
de EE.UU. durante episodios de alta volatilidad del petróleo (WTI).

## Inicio rápido

```bash
pip install -r requirements.txt

python -m src.train    --config configs/v1.yaml   # entrenamiento + CV + holdout → reports/
python -m src.evaluate --config configs/v1.yaml   # figuras → reports/fig_*.png
```

## Estructura del proyecto

```
market-intelligence/
├── data/
│   ├── raw/                             # CSVs originales (derivados del trabajo de pasantía)
│   ├── processed/                       # Generado: events_clean.parquet
│   └── data_dictionary.md               # diccionario de variables
├── src/
│   ├── config.py                        # lector YAML
│   ├── data_loader.py                   # carga + limpieza + guardado
│   ├── features.py                      # retornos, lags, spreads inter-mercado
│   ├── train.py                         # CV por eventos + holdout + ajuste de umbral
│   └── evaluate.py                      # figuras (incluye PR curve)
├── reports/                             # salidas generadas
│   ├── latest_metrics.md                # reporte consolidado
│   └── fig_*.png                        # figuras
├── configs/v1.yaml
└── requirements.txt
```

## Documentos de cierre
- `CIERRE.md` — resumen de cierre, evidencia y cómo citar resultados
- `reports/latest_metrics.md` — reporte técnico reproducible (CV + holdout)
