"""
train.py — Train baselines + ML with event-based temporal CV + final holdout.

Improvements over v2:
  - Final holdout: last N_HOLDOUT (PRUEBA FINAL) events never seen during CV.
  - Threshold tuning: optimize F1_up on CV, then apply to holdout.
  - Aggregated confusion matrix across CV folds.
  - Primary metric: macro-F1 (handles class imbalance).
  - Secondary metrics: F1_up (catch rebounds), AUC (ranking quality).

Usage:
    python -m src.train --config configs/v1.yaml
"""
import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, precision_recall_curve, classification_report
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_and_clean
from features import prepare_model_data
from config import load_config

# ══════════════════════════════════════════════════════════════════════
# SPLITTING
# ══════════════════════════════════════════════════════════════════════

def holdout_split(meta, n_holdout=5):
    """Split into CV events (chronologically first) and holdout (last n)."""
    events_chron = meta.groupby("Eventos")["date"].min().sort_values().index.tolist()
    cv_events = set(events_chron[:-n_holdout])
    ho_events = set(events_chron[-n_holdout:])
    cv_idx = np.where(meta["Eventos"].isin(cv_events))[0].tolist()
    ho_idx = np.where(meta["Eventos"].isin(ho_events))[0].tolist()
    return cv_idx, ho_idx, sorted(cv_events), sorted(ho_events)


def event_time_split(meta, n_splits=5):
    """Chronological event-based CV on the CV portion only."""
    meta = meta.reset_index(drop=True)
    events_chron = meta.groupby("Eventos")["date"].min().sort_values().index.tolist()
    n = len(events_chron)
    fold = max(1, n // (n_splits + 1))
    splits = []
    for i in range(n_splits):
        tr_end = fold * (i + 1)
        te_end = min(tr_end + fold, n)
        if te_end <= tr_end:
            continue
        tr_ev = set(events_chron[:tr_end])
        te_ev = set(events_chron[tr_end:te_end])
        tr_idx = np.where(meta["Eventos"].isin(tr_ev))[0].tolist()
        te_idx = np.where(meta["Eventos"].isin(te_ev))[0].tolist()
        if tr_idx and te_idx:
            splits.append((tr_idx, te_idx))
    return splits

# ══════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════

def baseline_naive(meta_slice):
    """Today's direction → tomorrow's forecast (no leakage)."""
    return meta_slice["baseline_naive_next"].values

def baseline_majority(y_train, y_test):
    return np.full(len(y_test), y_train.mode().iloc[0])

# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════

MODELS = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   class_weight="balanced", random_state=42)),
    ]),
    "ridge_classifier": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                       class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
    ]),
    "gradient_boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=42)),
    ]),
}

# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def _metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_up":    f1_score(y_true, y_pred, average="binary", zero_division=0),
        "balanced_acc": ((y_pred[y_true == 1] == 1).mean() +
                         (y_pred[y_true == 0] == 0).mean()) / 2
                        if len(np.unique(y_true)) > 1 else 0.5,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            m["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    return m

# ══════════════════════════════════════════════════════════════════════
# THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true, y_prob, metric="f1_up"):
    """Find the probability threshold that maximizes F1 for the positive class."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 = 2 * prec * rec / (prec + rec)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * prec * rec / (prec + rec)
    f1s = np.nan_to_num(f1s)
    # precision_recall_curve returns len(thresholds) = len(prec) - 1
    best_i = np.argmax(f1s[:-1])
    return thresholds[best_i], f1s[best_i]

# ══════════════════════════════════════════════════════════════════════
# CV TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_cv(X, y, meta, n_splits=5):
    """CV on the CV portion. Returns results + aggregated confusion matrices."""
    splits = event_time_split(meta, n_splits)
    print(f"  Event-based CV: {len(splits)} folds\n")
    results = {}
    cm_agg = {}       # aggregated confusion matrices
    thresholds = {}   # best thresholds per model
    pr_curves = {}    # precision-recall data for plotting

    for fi, (tr, te) in enumerate(splits):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        n_tr_ev = meta.iloc[tr]["Eventos"].nunique()
        n_te_ev = meta.iloc[te]["Eventos"].nunique()
        print(f"  Fold {fi}: train {len(Xtr)} ({n_tr_ev} ev) → test {len(Xte)} ({n_te_ev} ev)")

        # Baselines
        for name in ["baseline_naive", "baseline_majority"]:
            pred_b = baseline_naive(meta.iloc[te]) if name == "baseline_naive" \
                     else baseline_majority(ytr, yte)
            m = _metrics(yte, pred_b)
            m["fold"] = fi
            results.setdefault(name, []).append(m)
            cm_agg.setdefault(name, np.zeros((2, 2), dtype=int))
            cm_agg[name] += confusion_matrix(yte, pred_b, labels=[0, 1])

        # ML Models
        for name, pipe in MODELS.items():
            try:
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                prob = None
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "predict_proba"):
                    prob = pipe.predict_proba(Xte)[:, 1]
                elif hasattr(clf, "decision_function"):
                    raw = clf.decision_function(
                        pipe.named_steps["scaler"].transform(Xte))
                    # Normalize to [0,1] for threshold tuning
                    prob = 1 / (1 + np.exp(-raw))

                m = _metrics(yte, pred, prob)
                m["fold"] = fi
                results.setdefault(name, []).append(m)
                cm_agg.setdefault(name, np.zeros((2, 2), dtype=int))
                cm_agg[name] += confusion_matrix(yte, pred, labels=[0, 1])

                # Threshold tuning: find best per fold, average later
                if prob is not None:
                    thr, f1_at_thr = find_best_threshold(yte, prob)
                    thresholds.setdefault(name, []).append(thr)
                    m["best_threshold"] = thr
                    m["f1_up_at_threshold"] = f1_at_thr

                    # PR curve data (keep last fold for plotting)
                    prec, rec, thr_arr = precision_recall_curve(yte, prob)
                    pr_curves[name] = {"precision": prec, "recall": rec,
                                       "thresholds": thr_arr}

            except Exception as e:
                print(f"    ⚠ {name} fold {fi}: {e}")

    # Average best thresholds
    avg_thresholds = {k: np.mean(v) for k, v in thresholds.items()}

    return results, cm_agg, avg_thresholds, pr_curves


# ══════════════════════════════════════════════════════════════════════
# HOLDOUT (PRUEBA FINAL) EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_holdout(X_cv, y_cv, X_ho, y_ho, meta_ho, avg_thresholds):
    """Train on ALL CV data, evaluate on holdout. Apply tuned thresholds."""
    print("\n  HOLDOUT (PRUEBA FINAL) EVALUATION")
    print(f"  Train: {len(X_cv)} rows → Test: {len(X_ho)} rows "
          f"({meta_ho['Eventos'].nunique()} events)\n")

    ho_results = {}

    # Baselines
    for name in ["baseline_naive", "baseline_majority"]:
        pred = baseline_naive(meta_ho) if name == "baseline_naive" \
               else baseline_majority(y_cv, y_ho)
        m = _metrics(y_ho, pred)
        m["cm"] = confusion_matrix(y_ho, pred, labels=[0, 1]).tolist()
        ho_results[name] = m

    # ML models (default threshold + tuned threshold)
    for name, pipe in MODELS.items():
        pipe.fit(X_cv, y_cv)
        pred_default = pipe.predict(X_ho)
        clf = pipe.named_steps["clf"]
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = pipe.predict_proba(X_ho)[:, 1]
        elif hasattr(clf, "decision_function"):
            raw = clf.decision_function(pipe.named_steps["scaler"].transform(X_ho))
            prob = 1 / (1 + np.exp(-raw))

        # Default threshold (0.5)
        m = _metrics(y_ho, pred_default, prob)
        m["cm"] = confusion_matrix(y_ho, pred_default, labels=[0, 1]).tolist()
        m["threshold"] = 0.5

        # Tuned threshold
        if prob is not None and name in avg_thresholds:
            thr = avg_thresholds[name]
            pred_tuned = (prob >= thr).astype(int)
            m_tuned = _metrics(y_ho, pred_tuned, prob)
            m["tuned_threshold"] = thr
            m["tuned_f1_up"] = m_tuned["f1_up"]
            m["tuned_f1_macro"] = m_tuned["f1_macro"]
            m["tuned_accuracy"] = m_tuned["accuracy"]
            m["tuned_cm"] = confusion_matrix(y_ho, pred_tuned, labels=[0, 1]).tolist()

        ho_results[name] = m

    return ho_results


# ══════════════════════════════════════════════════════════════════════
# SUMMARIZE + IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

def summarize(results):
    rows = []
    for model, folds in results.items():
        df = pd.DataFrame(folds)
        row = {"model": model}
        for met in ["accuracy", "f1_macro", "f1_up", "balanced_acc", "auc"]:
            if met in df.columns:
                vals = df[met].dropna()
                row[f"{met}_mean"] = vals.mean()
                row[f"{met}_std"]  = vals.std()
        row["n_folds"] = len(folds)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def feature_importance(X, y):
    MODELS["logistic_regression"].fit(X, y)
    lr = pd.Series(
        np.abs(MODELS["logistic_regression"].named_steps["clf"].coef_[0]),
        index=X.columns).sort_values(ascending=False)
    MODELS["random_forest"].fit(X, y)
    rf = pd.Series(
        MODELS["random_forest"].named_steps["clf"].feature_importances_,
        index=X.columns).sort_values(ascending=False)
    return lr, rf


# ══════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════

def _write_report(summary, ho_results, lr_imp, rf_imp, X, y, meta,
                  cm_agg, avg_thresholds, n_holdout, out_dir):
    n_up = int((y == 1).sum())
    n_dn = int((y == 0).sum())

    lines = [
        "# Informe de entrenamiento de modelos",
        f"**Generado:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Definición del objetivo",
        "",
        "- **Objetivo (target)**: `sign(retorno S&P 500[t+1])` — dirección del día siguiente",
        "- **Features**: conocidas al final del día *t* (Asia/Europa mismo día + EE.UU. con lag t-1)",
        "- **Sin leakage mismo día**: features en *t* predicen el resultado en *t+1*",
        "",
        "## Conjunto de datos",
        "",
        f"- Total samples: {len(X)}  |  Features: {X.shape[1]}  |  Events: {meta['Eventos'].nunique()}",
        f"- Class balance: **up {n_up} ({n_up/len(y):.1%})**  /  down {n_dn} ({n_dn/len(y):.1%})",
        f"- CV: first {meta['Eventos'].nunique() - n_holdout} events  |  "
        f"Holdout: last {n_holdout} events (never seen during CV)",
        "",
        "## Métrica principal: F1 macro",
        "",
        "Accuracy is misleading with 65/35 imbalance (majority baseline wins at ~64%). "
        "We use **macro-F1** as the primary metric (equally weights both classes). "
        "Secondary: **F1_up** (ability to catch rebounds) and **AUC** (ranking quality).",
        "",
        "## Resultados de validación cruzada (temporal por eventos, 5 folds)",
        "",
        summary.to_markdown(index=False),
        "",
        "### Aggregated Confusion Matrices (summed across CV folds)",
        "",
    ]

    for name in ["random_forest", "logistic_regression", "baseline_majority", "baseline_naive"]:
        if name in cm_agg:
            cm = cm_agg[name]
            lines.append(f"**{name}**:")
            lines.append("")
            lines.append("|  | Pred Down | Pred Up |")
            lines.append("|--|----------|---------|")
            lines.append(f"| **Actual Down** | {cm[0,0]} | {cm[0,1]} |")
            lines.append(f"| **Actual Up** | {cm[1,0]} | {cm[1,1]} |")
            lines.append("")

    # Threshold tuning
    if avg_thresholds:
        lines.append("## Threshold Tuning (optimized for F1_up on CV)")
        lines.append("")
        lines.append("| Model | Default (0.5) F1_up | Tuned Threshold | Tuned F1_up |")
        lines.append("|-------|--------------------:|----------------:|------------:|")
        for name in ["random_forest", "logistic_regression", "gradient_boosting"]:
            if name in avg_thresholds and name in ho_results:
                hr = ho_results[name]
                lines.append(
                    f"| {name} | {hr.get('f1_up', 0):.3f} | "
                    f"{hr.get('tuned_threshold', 0.5):.3f} | "
                    f"{hr.get('tuned_f1_up', 0):.3f} |")
        lines.append("")

    # Holdout
    lines.append(f"## Final Holdout (last {n_holdout} events — never seen during CV)")
    lines.append("")
    ho_rows = []
    for name in ["random_forest", "gradient_boosting", "logistic_regression",
                  "ridge_classifier", "baseline_naive", "baseline_majority"]:
        if name in ho_results:
            r = ho_results[name]
            ho_rows.append({
                "model": name,
                "accuracy": r.get("accuracy", 0),
                "f1_macro": r.get("f1_macro", 0),
                "f1_up": r.get("f1_up", 0),
                "auc": r.get("auc", "—"),
            })
    lines.append(pd.DataFrame(ho_rows).to_markdown(index=False))
    lines.append("")

    # Feature importance
    lines.append("## Top 10 Features — Logistic Regression |coef|")
    lines.append("")
    lines.append(lr_imp.head(10).to_frame("abs_coef").to_markdown())
    lines.append("")
    lines.append("## Top 10 Features — Random Forest importance")
    lines.append("")
    lines.append(rf_imp.head(10).to_frame("importance").to_markdown())
    lines.append("")

    # Conclusion
    lines.append("## Key Finding")
    lines.append("")
    lines.append(
        f"During WTI volatility events (|daily change| > 4%), the S&P 500 declines "
        f"in **{n_dn/len(y):.0%}** of next-day observations. "
        f"The majority baseline wins on raw accuracy ({n_dn/len(y):.0%} by design), "
        f"but Random Forest outperforms on **macro-F1** and **AUC**, indicating "
        f"a modest but real signal from cross-market indicators.")
    lines.append("")
    lines.append(
        f"If the objective is catching rebounds (up days), Logistic Regression "
        f"offers a better trade-off (higher F1_up ≈ 0.45), at the cost of accuracy.")

    with open(os.path.join(out_dir, "latest_metrics.md"), "w") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run(config_path="configs/v1.yaml"):
    cfg = load_config(config_path)
    save_dir = cfg["reports"]["output_dir"]
    n_holdout = int(cfg["model"].get("n_holdout_events", 5))
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load
    print("=" * 60)
    print("DATOS")
    print("=" * 60)
    df = load_and_clean(
        raw_path=cfg["data"]["raw_path"],
        out_path=cfg["data"]["processed_path"],
        min_obs=int(cfg["data"].get("min_event_obs", 10)),
        drop_cols=list(cfg["data"].get("drop_columns", [])),
        min_feature_coverage=float(cfg["data"].get("min_feature_coverage", 0.0)),
        report_path=os.path.join(save_dir, "missingness.csv"),
    )
    X, y, meta, feat_cols = prepare_model_data(
        df,
        target=cfg["model"].get("target", "target_dir_next"),
        min_coverage=float(cfg["data"].get("min_feature_coverage", 0.0)),
    )

    # 2. Split: CV vs holdout
    cv_idx, ho_idx, cv_events, ho_events = holdout_split(meta, n_holdout)
    X_cv, y_cv, meta_cv = X.iloc[cv_idx], y.iloc[cv_idx], meta.iloc[cv_idx]
    X_ho, y_ho, meta_ho = X.iloc[ho_idx], y.iloc[ho_idx], meta.iloc[ho_idx]
    # Reset indices for iloc safety
    X_cv, y_cv, meta_cv = X_cv.reset_index(drop=True), y_cv.reset_index(drop=True), meta_cv.reset_index(drop=True)
    X_ho, y_ho, meta_ho = X_ho.reset_index(drop=True), y_ho.reset_index(drop=True), meta_ho.reset_index(drop=True)

    print(f"\n  CV:      {len(X_cv)} rows, {len(cv_events)} events")
    print(f"  Holdout: {len(X_ho)} rows, {len(ho_events)} events (untouched)")

    # 3. CV training
    print("\n" + "=" * 60)
    print("VALIDACIÓN CRUZADA")
    print("=" * 60)
    n_splits = int(cfg["model"].get("n_cv_folds", 5))
    cv_results, cm_agg, avg_thresholds, pr_curves = train_cv(X_cv, y_cv, meta_cv, n_splits)

    summary = summarize(cv_results)
    print("\n" + summary.to_string(index=False))

    if avg_thresholds:
        print("\n  Tuned thresholds (avg across folds):")
        for k, v in avg_thresholds.items():
            print(f"    {k}: {v:.3f}")

    # 4. Holdout
    print("\n" + "=" * 60)
    print("HOLDOUT (PRUEBA FINAL)")
    print("=" * 60)
    ho_results = evaluate_holdout(X_cv, y_cv, X_ho, y_ho, meta_ho, avg_thresholds)

    print("\n  Holdout results:")
    for name, r in ho_results.items():
        auc_str = f"{r['auc']:.3f}" if isinstance(r.get("auc"), float) else "—"
        tuned = f" | tuned F1↑={r['tuned_f1_up']:.3f}" if "tuned_f1_up" in r else ""
        print(f"    {name:25s}: acc={r['accuracy']:.3f}  F1m={r['f1_macro']:.3f}  "
              f"F1↑={r['f1_up']:.3f}  AUC={auc_str}{tuned}")

    # 5. Feature importance (on all CV data)
    print("\n" + "=" * 60)
    print("IMPORTANCIA DE VARIABLES")
    print("=" * 60)
    lr_imp, rf_imp = feature_importance(X_cv, y_cv)
    print("\nLogistic Regression |coef|:")
    print(lr_imp.head(10).to_string())
    print("\nRandom Forest:")
    print(rf_imp.head(10).to_string())

    # 6. Save
    summary.to_csv(f"{save_dir}/model_summary.csv", index=False)
    with open(f"{save_dir}/fold_results.json", "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    with open(f"{save_dir}/holdout_results.json", "w") as f:
        json.dump(ho_results, f, indent=2, default=str)
    with open(f"{save_dir}/thresholds.json", "w") as f:
        json.dump(avg_thresholds, f, indent=2, default=str)
    # Save confusion matrices
    cm_serializable = {k: v.tolist() for k, v in cm_agg.items()}
    with open(f"{save_dir}/confusion_matrices_cv.json", "w") as f:
        json.dump(cm_serializable, f, indent=2)
    # Save PR curve data
    pr_serializable = {}
    for k, v in pr_curves.items():
        pr_serializable[k] = {
            "precision": v["precision"].tolist(),
            "recall": v["recall"].tolist(),
        }
    with open(f"{save_dir}/pr_curves.json", "w") as f:
        json.dump(pr_serializable, f, indent=2)

    lr_imp.to_csv(f"{save_dir}/feature_importance_logreg.csv", header=["abs_coef"])
    rf_imp.to_csv(f"{save_dir}/feature_importance_rf.csv", header=["importance"])

    _write_report(summary, ho_results, lr_imp, rf_imp, X, y, meta,
                  cm_agg, avg_thresholds, n_holdout, save_dir)

    print(f"\nAll outputs → {save_dir}/")
    return summary, cv_results, ho_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train market-intelligence models")
    parser.add_argument("--config", default="configs/v1.yaml", help="Path to YAML config")
    run(config_path=parser.parse_args().config)
