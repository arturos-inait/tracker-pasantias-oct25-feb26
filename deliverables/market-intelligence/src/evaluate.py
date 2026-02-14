"""
evaluate.py — Genera todas las figuras para el reporte de análisis.

Produces in reports/:
  fig_model_comparison.png            bar chart (CV)
  fig_feature_importance.png          LR + RF top features
  fig_confusion_matrix_cv.png         aggregated CM across CV folds
  fig_confusion_matrix_holdout.png    CM on final holdout
  fig_pr_curve.png                    Precision-Recall curve for class "up"
  fig_event_coverage.png              heatmap of variable coverage
  fig_target_distribution.png         class balance + P(up) by tau
  fig_cross_market_correlation.png    return correlations

Uso:
    python -m src.evaluate --config configs/v1.yaml
"""
import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_and_clean
from features import prepare_model_data, build_features
from train import MODELS, event_time_split, holdout_split
from config import load_config

DEFAULT_CONFIG = "configs/v1.yaml"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10, "figure.dpi": 150,
})
C = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#6366f1", "#64748b"]


def fig_model_comparison(d):
    s = pd.read_csv(f"{d}/model_summary.csv").sort_values("f1_macro_mean")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, met, title in zip(axes,
                               ["accuracy_mean", "f1_macro_mean", "auc_mean"],
                               ["Exactitud", "F1 macro (principal)", "AUC"]):
        vals = s[met].copy()
        is_nan = vals.isna()
        vals = vals.fillna(0)
        stds = s.get(met.replace("_mean", "_std"), pd.Series([0]*len(s))).fillna(0)
        bars = ax.barh(s["model"], vals, xerr=stds, color=C[:len(s)],
                       edgecolor="white", capsize=3)
        ax.set_title(title, fontweight="bold"); ax.set_xlim(0, 1)
        for j, (b, v) in enumerate(zip(bars, vals)):
            label = "n/a" if (met == "auc_mean" and is_nan.iloc[j]) else f"{v:.3f}"
            ax.text(v + 0.02, b.get_y() + b.get_height()/2, label,
                    va="center", fontsize=8)
    plt.suptitle("Resultados de validación cruzada (objetivo día siguiente)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{d}/fig_model_comparison.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_model_comparison.png")


def fig_feature_importance(d):
    lr = pd.read_csv(f"{d}/feature_importance_logreg.csv", index_col=0)
    rf = pd.read_csv(f"{d}/feature_importance_rf.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    t = lr.head(10).sort_values(lr.columns[0])
    a1.barh(t.index, t.iloc[:, 0], color=C[0])
    a1.set_title("Regresión logística |coef|", fontweight="bold")
    t = rf.head(10).sort_values(rf.columns[0])
    a2.barh(t.index, t.iloc[:, 0], color=C[1])
    a2.set_title("Importancia — Random Forest", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{d}/fig_feature_importance.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_feature_importance.png")


def fig_confusion_matrix_cv(d):
    """Aggregated confusion matrices from CV."""
    with open(f"{d}/confusion_matrices_cv.json") as f:
        cms = json.load(f)
    models = ["random_forest", "logistic_regression", "baseline_majority", "baseline_naive"]
    models = [m for m in models if m in cms]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        cm = np.array(cms[name])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Down", "Up"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Down", "Up"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(name.replace("_", " ").title(), fontsize=9, fontweight="bold")
    plt.suptitle("Matrices de confusión agregadas (suma de folds CV)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{d}/fig_confusion_matrix_cv.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_confusion_matrix_cv.png")


def fig_confusion_matrix_holdout(d):
    """Confusion matrix on final holdout."""
    with open(f"{d}/holdout_results.json") as f:
        ho = json.load(f)
    models = ["random_forest", "logistic_regression", "baseline_majority"]
    models = [m for m in models if m in ho and "cm" in ho[m]]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        cm = np.array(ho[name]["cm"])
        ax.imshow(cm, cmap="Oranges")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Down", "Up"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Down", "Up"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(name.replace("_", " ").title(), fontsize=9, fontweight="bold")
    plt.suptitle("Matrices de confusión (holdout: últimos 5 eventos)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{d}/fig_confusion_matrix_holdout.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_confusion_matrix_holdout.png")


def fig_pr_curve(d):
    """Precision-Recall curve for class 'up'."""
    with open(f"{d}/pr_curves.json") as f:
        curves = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(curves.items()):
        ax.plot(data["recall"], data["precision"], color=C[i % len(C)],
                label=name.replace("_", " ").title(), linewidth=2)
    # Baseline: prevalence
    # Read from model_summary to get class balance
    ax.axhline(0.35, color="gray", ls="--", alpha=0.5, label="Prevalencia (~35% Up)")
    ax.set_xlabel("Recall/Cobertura (clase 'Up')", fontsize=11)
    ax.set_ylabel("Precisión (clase 'Up')", fontsize=11)
    ax.set_title("Precision-Recall Curve — Class 'Up' (next-day S&P positive)",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{d}/fig_pr_curve.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_pr_curve.png")


def fig_event_coverage(df, d):
    num = [c for c in df.select_dtypes("number").columns if c != "tau"]
    cov = df.groupby("Eventos")[num].apply(lambda g: g.notna().mean())
    order = df.groupby("Eventos")["date"].min().sort_values().index
    cov = cov.reindex(order)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(cov.values.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cov)))
    ax.set_xticklabels([e.replace("evento ", "") for e in cov.index], rotation=90, fontsize=7)
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=8)
    ax.set_xlabel("Event #")
    ax.set_title("Variable Coverage per Event", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Coverage")
    plt.tight_layout()
    plt.savefig(f"{d}/fig_event_coverage.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_event_coverage.png")


def fig_target_distribution(y, meta, d):
    tmp = meta.copy(); tmp["target"] = y.values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    cts = y.value_counts()
    a1.bar(["Down (0)", "Up (1)"], [cts.get(0, 0), cts.get(1, 0)], color=[C[2], C[1]])
    a1.set_title("Next-Day S&P 500 Direction\n(during WTI volatility events)", fontweight="bold")
    a1.set_ylabel("Count")
    for i, v in enumerate([cts.get(0, 0), cts.get(1, 0)]):
        a1.text(i, v + 3, str(v), ha="center", fontweight="bold")
    tau_up = tmp.groupby("tau")["target"].mean()
    a2.bar(tau_up.index, tau_up.values, color=C[0], alpha=0.7)
    a2.axhline(y.mean(), color=C[2], ls="--", label=f"Avg {y.mean():.2f}")
    a2.set_xlabel("Position in event window (τ)"); a2.set_ylabel("P(S&P up next day)")
    a2.set_title("P(up next day) by τ", fontweight="bold"); a2.legend()
    plt.tight_layout()
    plt.savefig(f"{d}/fig_target_distribution.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_target_distribution.png")


def fig_cross_market_correlation(df, d):
    feat = build_features(df)
    ret = [c for c in feat.columns if c.endswith("_ret") and "lag" not in c]
    corr = feat[ret + ["ret_sp500"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > .5 else "black")
    ax.set_title("Cross-Market Return Correlations", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(f"{d}/fig_cross_market_correlation.png", bbox_inches="tight"); plt.close()
    print("  ✓ fig_cross_market_correlation.png")


def run_all(config_path=DEFAULT_CONFIG):
    cfg = load_config(config_path)
    d = cfg.get("reports", {}).get("output_dir", "reports")
    os.makedirs(d, exist_ok=True)

    print("Loading data …")
    df = load_and_clean(
        raw_path=cfg["data"]["raw_path"],
        out_path=cfg["data"]["processed_path"],
        min_obs=int(cfg["data"].get("min_event_obs", 10)),
        drop_cols=list(cfg["data"].get("drop_columns", [])),
        min_feature_coverage=float(cfg["data"].get("min_feature_coverage", 0.0)),
        report_path=os.path.join(d, "missingness.csv"),
    )
    X, y, meta, _ = prepare_model_data(
        df,
        target=cfg["model"].get("target", "target_dir_next"),
        min_coverage=float(cfg["data"].get("min_feature_coverage", 0.0)),
    )

    print("\nGenerating figures:")
    fig_model_comparison(d)
    fig_feature_importance(d)
    fig_confusion_matrix_cv(d)
    fig_confusion_matrix_holdout(d)
    fig_pr_curve(d)
    fig_event_coverage(df, d)
    fig_target_distribution(y, meta, d)
    fig_cross_market_correlation(df, d)
    print(f"\nAll figures → {d}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate evaluation figures")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    run_all(config_path=p.parse_args().config)
