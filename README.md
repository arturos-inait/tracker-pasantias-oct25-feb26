<div align="center">

# Tracker de Pasantías

**Oct 2025 – Feb 2026 · Venezuela & Suiza**

Repositorio de seguimiento y entregables asociados a las pasantías.

---

[![Guía Interactiva](https://img.shields.io/badge/📖_Guía_Paso_a_Paso-Ver_Online-0d9488?style=for-the-badge&logoColor=white)](https://arturos-inait.github.io/tracker-pasantias-oct25-feb26/)
[![Pipeline](https://img.shields.io/badge/Pipeline-Reproducible-22c55e?style=for-the-badge&logo=python&logoColor=white)](#cómo-reproducir)
[![Status](https://img.shields.io/badge/Estado-✅_Cerrado-blue?style=for-the-badge)](#)

</div>

---

## Entregable: Inteligencia de Mercado

**Pasantía de Diego Salcedo Flores** · Supervisión: Dr. Arturo Sánchez Pineda

> *Cuando el petróleo se mueve fuerte, ¿las señales de Asia y Europa anticipan la dirección del S&P 500 al día siguiente?*

Pipeline reproducible de ML que evalúa el spillover cross-market durante eventos de alta volatilidad del WTI (1986–2025).

### Datos

| Etapa | Cantidad | Detalle |
|:------|:--------:|:--------|
| Eventos identificados | **50** | Trabajo original de la pasantía |
| Eventos utilizables | **45** | Tras filtrar cobertura insuficiente |
| Set de modelado | **25** → 704 filas | 20 eventos CV + 5 holdout (2019–2025) |

### Resultados clave

| Escenario | Modelo | F1 macro | AUC |
|:----------|:-------|:--------:|:---:|
| CV (20 eventos) | Logistic Regression | **0.521** | 0.567 |
| CV (20 eventos) | Random Forest | 0.516 | **0.594** |
| Holdout (5 eventos) | Random Forest | **0.535** | **0.618** |
| *Baseline mayoría* | *Siempre "Down"* | *0.398* | *—* |

### Archivos principales

```
deliverables/market-intelligence/
├── src/                    # Pipeline Python (5 módulos)
├── configs/v1.yaml         # Configuración reproducible
├── data/raw/               # CSVs originales (pasantía)
├── data/processed/         # Parquet limpio
├── reports/                # 8 figuras + métricas
│   └── latest_metrics.md   # Reporte completo
├── CIERRE.md              # Documentación de cierre
└── requirements.txt
```

---

## Cómo reproducir

```bash
cd deliverables/market-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/v1.yaml
```

Los resultados quedan en `reports/` (tablas, métricas, 8 figuras).

---

## Documentación

| Documento | Descripción |
|:----------|:------------|
| [`CIERRE.md`](deliverables/market-intelligence/CIERRE.md) | Narrativa de cierre del proyecto |
| [`latest_metrics.md`](deliverables/market-intelligence/reports/latest_metrics.md) | Reporte técnico completo con tablas y figuras |
| [**Guía interactiva ↗**](https://arturos-inait.github.io/tracker-pasantias-oct25-feb26/) | Workflow visual paso a paso (sin tecnicismos) |

---

<div align="center">
<sub>INAIT SA · Lausanne, Suiza · 2026</sub>
</div>
