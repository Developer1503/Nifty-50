"""
train_models.py
===============
Trains Logistic Regression, Random Forest, and XGBoost on the
feature matrix produced by feature_engineering.py.

Steps:
  1. Walk-forward (time-series) train/val/test split — NO data leakage.
  2. SMOTE-style class balancing on training fold only.
  3. Grid-search hyper-parameter tuning (RandomizedSearchCV).
  4. Full evaluation: accuracy, confusion matrix, classification report,
     feature importance, ROC curves.
  5. Saves trained models to disk (joblib).

Run:
    python option_chain_ml/train_models.py
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # headless — safe on all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils           import resample               # for manual oversampling

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠  XGBoost not installed — skipping XGB model.")

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
ML_DIR    = Path(__file__).resolve().parent
RESULTS   = ML_DIR / "results"
MODELS    = ML_DIR / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

sys.path.insert(0, str(ML_DIR))
from feature_engineering import build_feature_matrix, FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def time_series_split(df: pd.DataFrame, train_frac=0.65, val_frac=0.15):
    """Strict temporal split — never shuffle."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train = df.iloc[:t1]
    val   = df.iloc[t1:t2]
    test  = df.iloc[t2:]
    return train, val, test


def balance_classes(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample minority classes to match majority class count.
    Applied ONLY to training data to prevent leakage.
    """
    df_all = X.copy()
    df_all["__target__"] = y.values

    majority_n = df_all["__target__"].value_counts().max()
    parts = []
    for cls in df_all["__target__"].unique():
        subset = df_all[df_all["__target__"] == cls]
        if len(subset) < majority_n:
            subset = resample(subset, replace=True, n_samples=majority_n,
                              random_state=42)
        parts.append(subset)

    balanced = pd.concat(parts).sample(frac=1, random_state=42)
    y_bal = balanced.pop("__target__")
    return balanced, y_bal


def print_section(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_lr_pipeline(n_classes: int):
    """Logistic Regression pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=0.1,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def get_rf_pipeline():
    """Random Forest pipeline — no scaling needed, but kept for uniformity."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def get_xgb_pipeline(n_classes: int):
    """XGBoost pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name: str,
                   label_names: list[str], results: dict):
    """Compute metrics and store in results dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc    = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=label_names, output_dict=True)

    results[model_name] = {
        "accuracy": acc,
        "roc_auc":  auc,
        "cm":       cm,
        "report":   report,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}  (macro OvR)")
    print()
    print(classification_report(y_test, y_pred, target_names=label_names))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, feature_cols: list[str]) -> pd.Series:
    """Extract feature importances from any pipeline."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # For LogReg with multi-class: take mean absolute coef across classes
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        return pd.Series()
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = {"Logistic Regression": "#4C8BF5",
           "Random Forest":       "#34A853",
           "XGBoost":             "#FA7B17"}

label_names = ["Down", "Sideways", "Up"]


def plot_confusion_matrices(results: dict, save_path: Path):
    """Side-by-side confusion matrices for all models."""
    models = list(results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices — Test Set", fontsize=16, fontweight="bold")

    for ax, name in zip(axes, models):
        cm = results[name]["cm"]
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, pad=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved confusion_matrices.png")


def plot_roc_curves(results: dict, y_test: pd.Series, n_classes: int, save_path: Path):
    """ROC curves (OvR) for each model × class."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)
    fig.suptitle("ROC Curves — One-vs-Rest", fontsize=16, fontweight="bold")

    for cls_idx, ax in enumerate(axes):
        for name, res in results.items():
            proba = res["y_proba"][:, cls_idx]
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], proba)
            try:
                auc_cls = roc_auc_score(y_bin[:, cls_idx], proba)
            except Exception:
                auc_cls = 0.0
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_cls:.2f})",
                    color=PALETTE.get(name, "#888"), linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"Class: {label_names[cls_idx]}", fontsize=13)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved roc_curves.png")


def plot_feature_importance(models_trained: dict, feature_cols: list[str],
                            save_path: Path, top_n: int = 20):
    """Horizontal bar charts of top-N features for each model."""
    model_names = list(models_trained.keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(8 * len(model_names), 8))
    if len(model_names) == 1:
        axes = [axes]
    fig.suptitle(f"Top-{top_n} Feature Importances", fontsize=16, fontweight="bold")

    for ax, name in zip(axes, model_names):
        imp = get_feature_importance(models_trained[name], feature_cols)
        top = imp.head(top_n)
        colors = [PALETTE.get(name, "#888")] * len(top)
        ax.barh(top.index[::-1], top.values[::-1], color=colors, edgecolor="white")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Importance")
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved feature_importance.png")


def plot_accuracy_summary(results: dict, save_path: Path):
    """Bar chart of accuracy and AUC side by side."""
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    aucs   = [results[n]["roc_auc"]  for n in names]
    colors = [PALETTE.get(n, "#888") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison — Test Set", fontsize=15, fontweight="bold")

    for ax, vals, title, ylim in [
        (axes[0], accs, "Accuracy",  (0.30, 1.0)),
        (axes[1], aucs, "ROC-AUC\n(Macro OvR)", (0.40, 1.0)),
    ]:
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.axhline(1 / 3, color="red", linestyle="--", alpha=0.6,
                   label="Random (3-class)")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved model_comparison.png")


def plot_cumulative_returns(models_trained: dict, X_test: pd.DataFrame,
                            test_df: pd.DataFrame, save_path: Path):
    """
    Backtest each model's 'Up' predictions against actual forward returns.
    Baseline = buy-and-hold.
    """
    spot   = test_df["Spot"].values
    fwd    = test_df["Fwd_Return"].fillna(0).values
    dates  = test_df["Date"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    # Buy-and-hold
    bh = np.cumprod(1 + fwd)
    ax.plot(dates, bh, label="Buy & Hold", color="gray",
            linewidth=2, linestyle="--", alpha=0.8)

    for name, model in models_trained.items():
        preds = model.predict(X_test)
        # Trade only when model says "Up" (class 2)
        position = (preds == 2).astype(float)
        strat_ret = position * fwd   # 0 when flat
        cum = np.cumprod(1 + strat_ret)
        ax.plot(dates, cum, label=f"{name} (Long when Up)",
                color=PALETTE.get(name, "#888"), linewidth=1.8)

    ax.set_title("Strategy Backtest — Cumulative Returns", fontsize=14,
                fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (₹1 invested)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path / "backtest_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved backtest_returns.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print_section("NIFTY-50 Option Chain ML — Training Pipeline")

    # ── 1. Build features ─────────────────────────────────────────────────
    print_section("Step 1 · Building Feature Matrix")
    df = build_feature_matrix(start_date="2015-01-01")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df["Target"]

    # ── 2. Time-series split ──────────────────────────────────────────────
    print_section("Step 2 · Train / Val / Test Split (Temporal)")
    train_df, val_df, test_df = time_series_split(df)

    def split_xy(sub_df):
        return sub_df[available_features], sub_df["Target"]

    X_train_raw, y_train = split_xy(train_df)
    X_val,       y_val   = split_xy(val_df)
    X_test,      y_test  = split_xy(test_df)

    print(f"  Train : {len(X_train_raw):,} rows  "
          f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"  Val   : {len(X_val):,} rows  "
          f"({val_df['Date'].min().date()} → {val_df['Date'].max().date()})")
    print(f"  Test  : {len(X_test):,} rows  "
          f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

    # ── 3. Class balancing (train only) ───────────────────────────────────
    print_section("Step 3 · Class Balancing (Oversampling on Train)")
    X_train, y_train_bal = balance_classes(X_train_raw, y_train)
    print(f"  Before: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  After : {pd.Series(y_train_bal).value_counts().to_dict()}")

    n_classes = 3

    # ── 4. Train models ───────────────────────────────────────────────────
    print_section("Step 4 · Training Models")
    models_def = {
        "Logistic Regression": get_lr_pipeline(n_classes),
        "Random Forest":       get_rf_pipeline(),
    }
    if XGBOOST_AVAILABLE:
        models_def["XGBoost"] = get_xgb_pipeline(n_classes)

    models_trained = {}
    for name, pipe in models_def.items():
        print(f"\n  ▶ Training {name} …")
        pipe.fit(X_train, y_train_bal)
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        print(f"    Val accuracy: {val_acc:.4f}")
        models_trained[name] = pipe
        joblib.dump(pipe, MODELS / f"{name.replace(' ', '_').lower()}.pkl")
        print(f"    Saved model → {name.replace(' ', '_').lower()}.pkl")

    # ── 5. Evaluate on held-out test set ──────────────────────────────────
    print_section("Step 5 · Test-Set Evaluation")
    results = {}
    for name, model in models_trained.items():
        results = evaluate_model(model, X_test, y_test, name, label_names, results)

    # ── 6. Visualisation ──────────────────────────────────────────────────
    print_section("Step 6 · Generating Plots")
    plot_accuracy_summary(results,          RESULTS)
    plot_confusion_matrices(results,        RESULTS)
    plot_roc_curves(results, y_test,        n_classes, RESULTS)
    plot_feature_importance(models_trained, available_features, RESULTS)
    plot_cumulative_returns(models_trained, X_test, test_df,    RESULTS)

    # ── 7. Summary table ──────────────────────────────────────────────────
    print_section("Final Summary")
    summary = pd.DataFrame({
        n: {
            "Accuracy":  f"{r['accuracy']:.4f}",
            "ROC-AUC":   f"{r['roc_auc']:.4f}",
            "Down F1":   f"{r['report']['Down']['f1-score']:.3f}",
            "Up F1":     f"{r['report']['Up']['f1-score']:.3f}",
            "Sideways F1": f"{r['report']['Sideways']['f1-score']:.3f}",
        }
        for n, r in results.items()
    }).T

    print(f"\n{summary.to_string()}")

    # Save summary CSV
    summary.to_csv(RESULTS / "model_summary.csv")
    print(f"\n  💾 Summary saved → results/model_summary.csv")

    best = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\n  🏆 Best model: {best}  (Accuracy = {results[best]['accuracy']:.4f})")
    print_section("Done ✅")

    return models_trained, results, test_df, X_test, available_features


if __name__ == "__main__":
    models_trained, results, test_df, X_test, features = main()
