"""Model evaluation metrics and diagnostics.

Implements Ranked Probability Score (RPS) as the primary metric,
plus calibration analysis and feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rps(
    y_true_labels: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 3,
) -> float:
    """Compute Ranked Probability Score (RPS) for ordered multiclass predictions.

    RPS is a proper scoring rule for ordered outcomes (away_win < draw < home_win).
    Lower is better. Perfect prediction scores 0.

    Args:
        y_true_labels: Integer labels (0=away_win, 1=draw, 2=home_win).
        y_proba: Array of shape (n, 3) with class probabilities.
        n_classes: Number of outcome classes.

    Returns:
        Mean RPS across all samples.
    """
    n = len(y_true_labels)
    rps_scores = np.zeros(n)

    for i in range(n):
        # Convert label to one-hot
        y_true = np.zeros(n_classes)
        y_true[int(y_true_labels[i])] = 1.0

        # Cumulative distributions
        cum_pred = np.cumsum(y_proba[i])
        cum_true = np.cumsum(y_true)

        rps_scores[i] = np.sum((cum_pred - cum_true) ** 2) / (n_classes - 1)

    return float(np.mean(rps_scores))


def calibration_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability diagram data for calibration assessment.

    Args:
        y_true: True binary outcomes (0 or 1) for the target class.
        y_proba: Predicted probabilities for the target class.
        n_bins: Number of probability bins.

    Returns:
        DataFrame with columns: bin_mean_prob, actual_freq, count.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins[1:-1])

    records = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            records.append({
                "bin_mean_prob": float(y_proba[mask].mean()),
                "actual_freq": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            })

    return pd.DataFrame(records)


def feature_importance_table(model: object) -> pd.DataFrame:
    """Extract XGBoost feature importances.

    Args:
        model: Fitted OutcomeModel instance.

    Returns:
        DataFrame with columns: feature, gain, cover, weight,
        sorted by gain descending.
    """
    xgb = model._model
    importance_types = ["gain", "cover", "weight"]
    records: dict[str, dict] = {}

    for imp_type in importance_types:
        scores = xgb.get_booster().get_score(importance_type=imp_type)
        for feat, score in scores.items():
            if feat not in records:
                records[feat] = {}
            records[feat][imp_type] = score

    df = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df.columns = ["feature"] + importance_types
    df = df.fillna(0).sort_values("gain", ascending=False).reset_index(drop=True)
    return df


def classification_report_dict(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute common classification metrics.

    Args:
        y_true: True labels (0, 1, 2).
        y_proba: Predicted probabilities of shape (n, 3).

    Returns:
        Dict with keys: accuracy, log_loss, rps.
    """
    from sklearn.metrics import accuracy_score, log_loss

    y_pred = np.argmax(y_proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "rps": compute_rps(y_true, y_proba),
    }
