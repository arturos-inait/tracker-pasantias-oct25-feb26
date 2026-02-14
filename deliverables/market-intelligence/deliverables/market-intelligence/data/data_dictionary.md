# Data Dictionary

## Dataset: Eventos-Originales.csv

**Source:** Google Sheets (manual collection from FRED, EIA, CBOE, Yahoo Finance)
**Period:** 1986–2025
**Structure:** 50 events × ~21 observations per event (±10 days around WTI volatility spike)
**Event definition:** Calendar day where |daily % change of WTI| > 4%

## Variables

| Column | Alias | Type | Source | Coverage | Description |
|--------|-------|------|--------|----------|-------------|
| Eventos | — | ID | — | 100% | Event identifier (evento 1–50) |
| date | — | datetime | — | 100% | Calendar date |
| init | — | datetime | — | per event | Event window start date |
| end | — | datetime | — | per event | Event window end date |
| DEXJPUS | — | forex | FRED | 43.1% | Japanese Yen per USD |
| DEXUSEU | — | forex | FRED | 31.7% | USD per Euro (post-1999) |
| DEXUSUK | — | forex | FRED | 43.1% | USD per British Pound |
| DEXCHUS | — | forex | FRED | 42.9% | Chinese Yuan per USD |
| DEXHKUS | — | forex | FRED | 43.1% | Hong Kong Dollar per USD |
| DEXSZUS | — | forex | FRED | 43.1% | Swiss Franc per USD |
| WTI-Oil | wti | commodity | EIA/FRED | 55.9% | WTI Crude Oil price (USD/barrel) |
| Dubai-Crude | — | commodity | — | 1.8% | **EXCLUDED** (insufficient coverage) |
| Brent-Crude | — | commodity | — | 1.8% | **EXCLUDED** (insufficient coverage) |
| VIX | — | volatility | CBOE | 43.1% | CBOE Volatility Index (S&P 500) |
| nikkei-volatility | nikkei_vol | volatility | — | 27.3% | Nikkei 225 Volatility Index |
| sp500 | — | index | Yahoo/FRED | 50.2% | S&P 500 Index (close) |
| Effective Federal Funds Rate | fed_rate | rate | FRED | 30.4% | US Federal Funds Rate |
| Índice de Sentimiento... | — | survey | Michigan | 2.5% | **EXCLUDED** (insufficient) |
| Tasa de interés Euro Area | ecb_rate | rate | ECB | 45.8% | ECB Main Refinancing Rate |
| Oro al contado... | gold | commodity | — | 57.2% | Gold spot price (USD/oz) |
| nikkei-index | nikkei | index | Yahoo | 54.8% | Nikkei 225 Index (close) |
| stoxx50-index | stoxx50 | index | Yahoo | 32.0% | Euro STOXX 50 Index (close) |
| stoxx50-volatility | — | volatility | — | 10.4% | **EXCLUDED** (insufficient) |
| noticias | — | text | manual | 20.8% | News headlines (not used in model) |
| descripción | — | text | manual | 4.8% | Event description (not used) |

## Normalization

- **Min-Max normalization** applied per event window: x' = (x - min) / (max - min)
- Available in `Eventos-Normalizados-Completos.csv`
- Model uses raw values + computed returns (pct_change) instead

## Missingness Policy

Variables with < 5% coverage are excluded from modeling.
Remaining variables are forward-filled within each event window.
Rows where the target (sp500) or critical features are still NaN after forward-fill are dropped.
