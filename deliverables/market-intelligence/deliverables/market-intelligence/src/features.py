"""
features.py — Feature engineering for the market-spillover model.

Core idea
---------
Asian markets (Nikkei, Yen) close ~6 h before NYSE.
European markets (STOXX, EUR, GBP) close ~30 min before NYSE.
So on calendar day *t*, their closing values are *known* before the US close.
We use them — plus lagged US variables from day *t-1* — to predict S&P 500
direction on day *t*.

Features are built WITHIN each event window to prevent cross-event leakage.
"""
import pandas as pd
import numpy as np

# ── Signal groups (aliased column names from data_loader) ─────────────
ASIA       = ["nikkei", "DEXJPUS"]
EUROPE     = ["stoxx50", "DEXUSEU", "DEXUSUK"]
COMMODITY  = ["wti", "gold"]
VOL        = ["VIX", "nikkei_vol"]
RATES      = ["fed_rate", "ecb_rate"]

ALL_SOURCES = ASIA + EUROPE + COMMODITY + VOL + RATES
TARGET      = "sp500"


def _pct(series):
    """Percentage change, replacing inf with NaN."""
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


def _build_one_event(grp):
    """Engineer features for a single event window."""
    g = grp.copy().sort_values("date").reset_index(drop=True)

    # ── Targets ──
    g["ret_sp500"]       = _pct(g[TARGET])
    g["target_dir"]      = (g["ret_sp500"] > 0).astype(int)       # same-day (for EDA)
    g["target_ret_next"] = g["ret_sp500"].shift(-1)                # next-day return
    g["target_dir_next"] = np.where(g["target_ret_next"].isna(), np.nan,
                             (g["target_ret_next"] > 0).astype(int))  # next-day direction
    g["baseline_naive_next"] = g["target_dir"]  # today's dir as forecast for tomorrow

    # ── Same-day returns ──
    for col in ALL_SOURCES:
        if col in g.columns:
            g[f"{col}_ret"] = _pct(g[col])

    # ── Lag-1 for US variables ──
    for col in [TARGET, "VIX", "wti", "fed_rate"]:
        if col in g.columns:
            g[f"{col}_lag1"]     = g[col].shift(1)
            g[f"{col}_lag1_ret"] = _pct(g[col]).shift(1)

    # ── Cross-market spreads ──
    if "nikkei_ret" in g.columns and "ret_sp500" in g.columns:
        g["nk_sp_ret_diff"] = g.get("nikkei_ret", 0) - g["ret_sp500"].shift(1)
    if "VIX" in g.columns and "nikkei_vol" in g.columns:
        g["vix_nkvol_ratio"] = g["VIX"] / g["nikkei_vol"].replace(0, np.nan)

    return g


def build_features(df):
    """Apply feature engineering to every event."""
    parts = [_build_one_event(grp) for _, grp in df.groupby("Eventos", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _select_feature_cols(df):
    """Return feature column names (no targets, no identifiers)."""
    exclude = {"Eventos", "date", "tau",
               "ret_sp500", "target_dir", "target_ret_next",
               "target_dir_next", "baseline_naive_next", TARGET}
    out = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(col.endswith(s) for s in ("_ret", "_lag1", "_lag1_ret")):
            out.append(col)
        elif col in ("tau", "nk_sp_ret_diff", "vix_nkvol_ratio"):
            out.append(col)
        elif col in RATES:
            out.append(col)
    return out


def prepare_model_data(df, target="target_dir", min_coverage=0.30):
    """
    Full pipeline: features → select → prune sparse → ffill → return X, y, meta.
    """
    feat_df = build_features(df)
    feat_cols = _select_feature_cols(feat_df)

    # Drop sparse features
    cov = feat_df[feat_cols].notna().mean()
    keep = cov[cov >= min_coverage].index.tolist()
    dropped = set(feat_cols) - set(keep)
    if dropped:
        print(f"  Dropped (coverage < {min_coverage}): {sorted(dropped)}")
    feat_cols = keep

    # Forward-fill within events
    for _, idx in feat_df.groupby("Eventos").groups.items():
        feat_df.loc[idx, feat_cols] = feat_df.loc[idx, feat_cols].ffill()

    # Keep rows where target + all features are present
    mask = feat_df[target].notna()
    for c in feat_cols:
        mask &= feat_df[c].notna()

    X    = feat_df.loc[mask, feat_cols].reset_index(drop=True)
    y    = feat_df.loc[mask, target].reset_index(drop=True)
    meta_cols = ["Eventos", "date", "tau"]
    if "baseline_naive_next" in feat_df.columns:
        meta_cols.append("baseline_naive_next")
    meta = feat_df.loc[mask, meta_cols].reset_index(drop=True)

    print(f"  Model data: {len(X)} samples × {len(feat_cols)} features, "
          f"{meta['Eventos'].nunique()} events")
    print(f"  Target balance: up={int((y==1).sum())} ({(y==1).mean():.1%})  "
          f"down={int((y==0).sum())} ({(y==0).mean():.1%})")
    return X, y, meta, feat_cols


if __name__ == "__main__":
    from data_loader import load_and_clean
    df = load_and_clean()
    X, y, meta, cols = prepare_model_data(df)
    print(f"\nFeatures ({len(cols)}):")
    for c in cols:
        print(f"  {c}")
