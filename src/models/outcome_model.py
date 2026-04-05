"""XGBoost multiclass classifier for match outcome prediction.

Predicts probabilities for: 0=away_win, 1=draw, 2=home_win.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import ModelConfig


class OutcomeModel:
    """XGBoost multiclass classifier for match win/draw/loss prediction.

    Labels: 0 = away_win, 1 = draw, 2 = home_win.
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        """Initialize the model with config hyperparameters.

        Args:
            model_cfg: Model configuration from config.yaml.
        """
        xgb_cfg = model_cfg.xgboost
        self._cfg = model_cfg
        self._model = XGBClassifier(
            n_estimators=xgb_cfg.n_estimators,
            max_depth=xgb_cfg.max_depth,
            learning_rate=xgb_cfg.learning_rate,
            subsample=xgb_cfg.subsample,
            colsample_bytree=xgb_cfg.colsample_bytree,
            eval_metric=xgb_cfg.eval_metric,
            early_stopping_rounds=xgb_cfg.early_stopping_rounds,
            random_state=model_cfg.random_seed,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
        )
        self._is_fitted = False
        self.feature_names_: list[str] = []

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train the XGBoost model with early stopping on validation loss.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels (0, 1, or 2).
            sample_weight: Per-sample weights for training.
            x_val: Validation feature matrix.
            y_val: Validation labels.
        """
        self.feature_names_ = list(x_train.columns)
        self._model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        self._is_fitted = True

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for a batch of matches.

        Args:
            x: Feature matrix with FEATURE_COLUMNS.

        Returns:
            Array of shape (n, 3): [p_away_win, p_draw, p_home_win].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self._model.predict_proba(x)

    def predict_match(
        self,
        feature_vector: pd.Series,
    ) -> dict[str, float]:
        """Predict probabilities for a single match.

        Args:
            feature_vector: Series with index = FEATURE_COLUMNS.

        Returns:
            Dict with keys: away_win_prob, draw_prob, home_win_prob.
        """
        x = feature_vector.to_frame().T
        proba = self.predict_proba(x)[0]
        return {
            "away_win_prob": float(proba[0]),
            "draw_prob": float(proba[1]),
            "home_win_prob": float(proba[2]),
        }

    def save(self, path: Path) -> None:
        """Serialize the fitted model to disk.

        Args:
            path: Absolute path for the .pkl file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "OutcomeModel":
        """Load a serialized OutcomeModel from disk.

        Args:
            path: Absolute path to the .pkl file.

        Returns:
            Loaded OutcomeModel instance.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)
