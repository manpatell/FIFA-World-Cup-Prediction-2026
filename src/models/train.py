"""Model training pipeline with chronological train/test split.

Uses time-series cross-validation (never random split) to prevent
data leakage from future matches into training.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config, get_model_path, get_processed_path
from src.features.match_features import FEATURE_COLUMNS
from src.models.evaluate import classification_report_dict
from src.models.goals_model import GoalsModel
from src.models.outcome_model import OutcomeModel

logger = logging.getLogger(__name__)


def time_series_split(
    match_dataset: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and test sets.

    Args:
        match_dataset: Labelled match dataset with a 'date' column.
        test_size: Fraction of most-recent matches to use as test set.

    Returns:
        Tuple of (train_df, test_df) split by date.
    """
    df = match_dataset.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_cross_validation(
    match_dataset: pd.DataFrame,
    cfg: Config,
) -> dict[str, list[float]]:
    """Time-series cross-validation with rolling expanding window.

    Each fold: train on all data up to split, evaluate on next period.

    Args:
        match_dataset: Labelled match dataset.
        cfg: Config object.

    Returns:
        Dict with keys 'accuracy', 'log_loss', 'rps', each mapping
        to a list of per-fold metric values.
    """
    df = match_dataset.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_folds = cfg.model.cv_folds
    fold_size = n // (n_folds + 1)

    metrics: dict[str, list[float]] = {"accuracy": [], "log_loss": [], "rps": []}

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]

        if len(train) < 100 or len(val) < 20:
            continue

        model = OutcomeModel(cfg.model)
        x_train = train[FEATURE_COLUMNS]
        y_train = train["label"]
        weights = train["sample_weight"].values
        x_val = val[FEATURE_COLUMNS]
        y_val = val["label"]

        try:
            model.fit(x_train, y_train, weights, x_val, y_val)
            proba = model.predict_proba(x_val)
            fold_metrics = classification_report_dict(y_val.values, proba)
            for k, v in fold_metrics.items():
                metrics[k].append(v)
            logger.info(
                "CV fold %d/%d: acc=%.3f log_loss=%.3f rps=%.3f",
                fold + 1, n_folds,
                fold_metrics["accuracy"],
                fold_metrics["log_loss"],
                fold_metrics["rps"],
            )
        except Exception as exc:
            logger.warning("CV fold %d failed: %s", fold + 1, exc)

    return metrics


def train_all_models(cfg: Config) -> tuple[OutcomeModel, GoalsModel]:
    """Full training pipeline: load data → split → train → evaluate → save.

    Args:
        cfg: Config object.

    Returns:
        Tuple of (fitted OutcomeModel, fitted GoalsModel).
    """
    match_dataset_path = get_processed_path(cfg, "match_dataset")
    logger.info("Loading match dataset from %s", match_dataset_path)
    match_dataset = pd.read_parquet(match_dataset_path)
    logger.info("Dataset shape: %s", match_dataset.shape)

    # Chronological split
    train_df, test_df = time_series_split(match_dataset, cfg.model.test_size)
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    weights = train_df["sample_weight"].values
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    # ── Outcome model ────────────────────────────────────────────────────────
    logger.info("Training OutcomeModel...")
    outcome_model = OutcomeModel(cfg.model)
    outcome_model.fit(x_train, y_train, weights, x_test, y_test)

    train_proba = outcome_model.predict_proba(x_train)
    test_proba = outcome_model.predict_proba(x_test)
    train_metrics = classification_report_dict(y_train.values, train_proba)
    test_metrics = classification_report_dict(y_test.values, test_proba)
    logger.info("OutcomeModel train: %s", train_metrics)
    logger.info("OutcomeModel test : %s", test_metrics)

    # ── Goals model ──────────────────────────────────────────────────────────
    logger.info("Training GoalsModel...")
    goals_model = GoalsModel(cfg.model)
    goals_model.fit(
        x_train,
        train_df["home_score"].astype(float),
        train_df["away_score"].astype(float),
        weights,
    )

    # ── Save models ──────────────────────────────────────────────────────────
    outcome_path = get_model_path(cfg, "outcome_model")
    home_goals_path = get_model_path(cfg, "goals_model_home")
    away_goals_path = get_model_path(cfg, "goals_model_away")

    outcome_model.save(outcome_path)
    goals_model.save(home_goals_path, away_goals_path)
    logger.info("Models saved to %s", cfg.paths.models_dir)

    return outcome_model, goals_model
