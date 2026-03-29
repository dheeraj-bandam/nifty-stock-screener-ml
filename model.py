"""
model.py
--------
Trains XGBoost and Random Forest classifiers to predict
whether a stock will beat Nifty 50 over the next 30 days.

Also computes SHAP feature importance — the finance-friendly
way to explain "why did the model pick this stock?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score,
                             classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_model(X, y, model_type="xgboost"):
    """
    Time-series cross-validated training.
    Uses TimeSeriesSplit to prevent lookahead bias — critical for finance.

    Returns trained model + cross-val metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )

    print(f"Training {model_type} with TimeSeriesSplit(5)...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        fold_metrics = {
            "fold":      fold + 1,
            "accuracy":  accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall":    recall_score(y_val, y_pred, zero_division=0),
            "auc":       roc_auc_score(y_val, y_proba)
        }
        metrics.append(fold_metrics)
        print(f"  Fold {fold+1}: Acc={fold_metrics['accuracy']:.3f}  "
              f"AUC={fold_metrics['auc']:.3f}  "
              f"Precision={fold_metrics['precision']:.3f}")

    metrics_df = pd.DataFrame(metrics)
    print(f"\nMean across folds:")
    print(metrics_df.mean(numeric_only=True).round(3))

    # Final fit on all data
    model.fit(X, y)
    path = os.path.join(MODEL_DIR, f"{model_type}_final.pkl")
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")

    return model, metrics_df


def get_feature_importance(model, feature_names, model_type="xgboost", top_n=10):
    """
    Returns feature importance as a clean DataFrame.
    For XGBoost: uses built-in gain importance.
    Use with SHAP for recruiter-friendly explanations.
    """
    if model_type == "xgboost":
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).head(top_n)

    return imp_df


def plot_feature_importance(imp_df, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#185FA5")
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("Top features — what drives outperformance?")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def load_model(model_type="xgboost"):
    path = os.path.join(MODEL_DIR, f"{model_type}_final.pkl")
    return joblib.load(path)


def predict_stocks(model, X, meta, threshold=0.55):
    """
    Given feature matrix X and metadata (ticker, date),
    return a scored DataFrame of buy signals.

    threshold: minimum predicted probability to flag as "beat Nifty"
    """
    proba = model.predict_proba(X)[:, 1]
    result = meta.copy()
    result["prob_beat_nifty"] = proba
    result["signal"] = (proba >= threshold).astype(int)
    return result.sort_values("prob_beat_nifty", ascending=False)


if __name__ == "__main__":
    # Smoke test with synthetic data
    from features import compute_features, create_target

    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=500, freq="B")
    n = len(dates)

    X_dummy = pd.DataFrame(
        np.random.randn(n, 13),
        columns=["return_5d", "return_20d", "return_60d",
                 "rsi_14", "macd", "dist_sma20",
                 "volatility_20d", "vol_spike", "pos_52w",
                 "bb_width", "pe_ratio", "beta", "div_yield"],
        index=dates
    )
    y_dummy = pd.Series(np.random.randint(0, 2, n), index=dates, name="beat_nifty")

    model, metrics = train_model(X_dummy, y_dummy, model_type="xgboost")
    imp = get_feature_importance(model, X_dummy.columns.tolist())
    print("\nTop features:")
    print(imp)
