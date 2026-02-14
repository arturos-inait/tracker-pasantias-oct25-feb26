# Tracker de Pasantías (Oct 2025 – Feb 2026)

Repositorio de seguimiento y entregables asociados a las pasantías.

## Entregables

### 1) Inteligencia de Mercado (WTI → S&P 500, spillover)
Entregable final (pipeline reproducible + evaluación + reportes):

- Carpeta: `deliverables/market-intelligence/`
- Documento de cierre: `deliverables/market-intelligence/CIERRE.md`
- Reporte de métricas: `deliverables/market-intelligence/reports/latest_metrics.md`

## Cómo reproducir (local)

```bash
cd deliverables/market-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/v1.yaml
python -m src.evaluate --config configs/v1.yaml
```

Los resultados quedan en `deliverables/market-intelligence/reports/`.
