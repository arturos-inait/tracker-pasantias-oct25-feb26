"""
data_loader.py — Load, clean, and prepare the WTI volatility events dataset.

Input:  data/raw/Eventos-Originales.csv
Output: data/processed/events_clean.{parquet|csv}
        reports/missingness.csv

Config-driven: all thresholds and paths come from configs/v1.yaml.

Missingness policy:
  DROP cols  : Dubai-Crude (1.8%), Brent-Crude (1.8%), stoxx50-volatility (10.4%),
               Michigan Sentiment (2.5%), plus metadata text columns.
  FFILL      : within each event window, forward-fill up to 2 consecutive NaNs.
  REQUIRE    : sp500 + wti present on every kept row.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ── Defaults (overridden by config when available) ────────────────────
DEFAULT_DROP_COLS = [
    "Dubai-Crude", "Brent-Crude", "stoxx50-volatility",
    "Índice de Sentimiento del Consumidor U Michigan",
    "init", "end", "noticias", "descripción",
]

NUMERIC_COLS = [
    "DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCHUS", "DEXHKUS", "DEXSZUS",
    "WTI-Oil", "VIX", "nikkei-volatility",
    "sp500", "Effective Federal Funds Rate", "Tasa de interés Euro Area",
    "Oro al contado Dólar onza troy", "nikkei-index", "stoxx50-index",
]

ALIASES = {
    "WTI-Oil": "wti",
    "Effective Federal Funds Rate": "fed_rate",
    "Tasa de interés Euro Area": "ecb_rate",
    "Oro al contado Dólar onza troy": "gold",
    "nikkei-index": "nikkei",
    "stoxx50-index": "stoxx50",
    "nikkei-volatility": "nikkei_vol",
}

CORE_REQUIRED = ["sp500", "wti"]


# ══════════════════════════════════════════════════════════════════════

def load_raw(path: str = "data/raw/Eventos-Originales.csv",
             drop_cols: list | None = None) -> pd.DataFrame:
    """Read raw CSV, parse dates, coerce numerics, rename columns."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["Eventos"]).copy()                       # trailing blanks
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)

    drop = drop_cols if drop_cols is not None else DEFAULT_DROP_COLS
    # Always also drop the metadata/text cols even if config only lists coverage drops
    always_drop = ["init", "end", "noticias", "descripción"]
    full_drop = list(set(drop) | set(always_drop))
    df = df.drop(columns=[c for c in full_drop if c in df.columns])

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})
    return df


def filter_usable_events(df: pd.DataFrame, min_obs: int = 10) -> pd.DataFrame:
    """Keep events with enough sp500 + wti observations."""
    usable = []
    for ev, grp in df.groupby("Eventos"):
        sp_ok = grp["sp500"].notna().sum() >= min_obs
        wti_ok = grp["wti"].notna().sum() >= min_obs
        if sp_ok and wti_ok:
            usable.append(ev)
    print(f"  Usable events: {len(usable)} / {df['Eventos'].nunique()}")
    return df[df["Eventos"].isin(usable)].copy()


def clean(df: pd.DataFrame, min_obs: int = 10,
          min_feature_coverage: float = 0.0) -> pd.DataFrame:
    """Filter → sort → ffill(limit=2) → drop core-NaN rows → add tau."""
    df = filter_usable_events(df, min_obs)
    df = df.sort_values(["Eventos", "date"]).reset_index(drop=True)

    # Forward-fill within each event (limit=2 for weekends/holidays)
    num = df.select_dtypes(include="number").columns.tolist()
    df[num] = df.groupby("Eventos")[num].transform(lambda s: s.ffill(limit=2))

    # Drop columns below coverage threshold
    if min_feature_coverage and min_feature_coverage > 0:
        feat = [c for c in df.columns if c not in ["Eventos", "date"]]
        cov = df[feat].notna().mean()
        low = cov[cov < min_feature_coverage].index.tolist()
        if low:
            print(f"  Dropped low-coverage columns: {low}")
            df = df.drop(columns=low)

    # Require core columns
    core = [c for c in CORE_REQUIRED if c in df.columns]
    df = df.dropna(subset=core).reset_index(drop=True)

    # tau: position within event window (~-10 … +10)
    taus = []
    for _, grp in df.groupby("Eventos", sort=False):
        n = len(grp)
        mid = n // 2
        taus.extend(range(-mid, n - mid))
    df["tau"] = taus
    return df


def report_missingness(df: pd.DataFrame, out_path: str = "reports/missingness.csv"):
    """Save per-column missingness."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    miss = df.drop(columns=["Eventos", "date"], errors="ignore").isnull().mean()
    miss = miss.sort_values(ascending=False)
    miss.to_csv(out_path, header=["pct_missing"])
    print(f"  Missingness report → {out_path}")


def _save_df(df: pd.DataFrame, out_path: str):
    """Save as parquet if possible, fall back to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        if out_path.lower().endswith(".parquet"):
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
    except ImportError:
        fallback = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(fallback, index=False)
        out_path = fallback
    return out_path


def load_and_clean(
    raw_path: str = "data/raw/Eventos-Originales.csv",
    out_path: str = "data/processed/events_clean.parquet",
    min_obs: int = 10,
    drop_cols: list | None = None,
    min_feature_coverage: float = 0.0,
    report_path: str = "reports/missingness.csv",
) -> pd.DataFrame:
    """Full pipeline: load → clean → save → return."""
    print("Loading data …")
    df = load_raw(raw_path, drop_cols=drop_cols)
    df = clean(df, min_obs=min_obs, min_feature_coverage=min_feature_coverage)
    saved = _save_df(df, out_path)
    report_missingness(df, out_path=report_path)
    print(f"  Clean dataset → {saved}  ({len(df)} rows, {df['Eventos'].nunique()} events)")
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print(f"\nShape : {df.shape}")
    print(f"Cols  : {list(df.columns)}")
    print(df.head(3).to_string())
