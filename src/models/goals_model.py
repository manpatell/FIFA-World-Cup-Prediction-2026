"""Poisson GLM pair for expected goals prediction.

Two separate Poisson regression models — one for home goals and one
for away goals — capturing asymmetric scoring patterns.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor

from src.config import ModelConfig

# Subset of FEATURE_COLUMNS used for goals prediction
# H2H features excluded (noisier for goal counts)
GOALS_FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "ranking_diff",
    "home_ranking_points",
    "away_ranking_points",
    "log_squad_value_ratio",
    "home_injury_adj_value_log",
    "away_injury_adj_value_log",
    "home_form_points",
    "away_form_points",
    "home_win_rate",
    "away_win_rate",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "is_neutral",
    "tournament_weight",
]


class GoalsModel:
    """Poisson GLM pair for predicting expected home and away goals.

    Uses sklearn TweedieRegressor(power=1) which implements Poisson regression
    via GLM with log link. Separate models for home and away goals.
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        """Initialize both Poisson models with config hyperparameters.

        Args:
            model_cfg: Model configuration from config.yaml.
        """
        self._cfg = model_cfg
        alpha = model_cfg.poisson.alpha
        max_iter = model_cfg.poisson.max_iter
        self._home_model = TweedieRegressor(
            power=1, alpha=alpha, max_iter=max_iter, link="log"
        )
        self._away_model = TweedieRegressor(
            power=1, alpha=alpha, max_iter=max_iter, link="log"
        )
        self._is_fitted = False

    def fit(
        self,
        x_train: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
        sample_weight: np.ndarray,
    ) -> None:
        """Train both Poisson models on goal count targets.

        Args:
            x_train: Training feature matrix (GOALS_FEATURE_COLUMNS subset).
            y_home: Home goals target series.
            y_away: Away goals target series.
            sample_weight: Per-sample weights.
        """
        x_goals = x_train[GOALS_FEATURE_COLUMNS]
        self._home_model.fit(x_goals, y_home, sample_weight=sample_weight)
        self._away_model.fit(x_goals, y_away, sample_weight=sample_weight)
        self._is_fitted = True

    def predict_goals(
        self,
        feature_vector: pd.Series,
    ) -> tuple[float, float]:
        """Predict expected goals for a single match.

        Args:
            feature_vector: Series with at least GOALS_FEATURE_COLUMNS.

        Returns:
            Tuple of (expected_home_goals, expected_away_goals), both >= 0.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        x = feature_vector[GOALS_FEATURE_COLUMNS].to_frame().T
        home_goals = float(max(0.0, self._home_model.predict(x)[0]))
        away_goals = float(max(0.0, self._away_model.predict(x)[0]))
        return home_goals, away_goals

    def predict_goals_batch(
        self,
        x: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goals for a batch of matches.

        Args:
            x: Feature matrix.

        Returns:
            Tuple of (home_goals_array, away_goals_array), clipped to >= 0.
        """
        x_goals = x[GOALS_FEATURE_COLUMNS]
        home = np.maximum(0.0, self._home_model.predict(x_goals))
        away = np.maximum(0.0, self._away_model.predict(x_goals))
        return home, away

    def save(self, home_path: Path, away_path: Path) -> None:
        """Serialize both Poisson models to disk.

        Args:
            home_path: Path for home goals model .pkl.
            away_path: Path for away goals model .pkl.
        """
        for path in [home_path, away_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(home_path, "wb") as fh:
            pickle.dump(self._home_model, fh)
        with open(away_path, "wb") as fh:
            pickle.dump(self._away_model, fh)

    @classmethod
    def load(cls, model_cfg: ModelConfig, home_path: Path, away_path: Path) -> "GoalsModel":
        """Load serialized GoalsModel from disk.

        Args:
            model_cfg: Model configuration.
            home_path: Path to home goals model .pkl.
            away_path: Path to away goals model .pkl.

        Returns:
            Loaded GoalsModel instance.
        """
        instance = cls(model_cfg)
        with open(home_path, "rb") as fh:
            instance._home_model = pickle.load(fh)
        with open(away_path, "rb") as fh:
            instance._away_model = pickle.load(fh)
        instance._is_fitted = True
        return instance
