"""Tests for outcome_model, goals_model, and evaluate modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import EloConfig, FeatureConfig, ModelConfig, PoissonConfig, XGBoostConfig
from src.features.match_features import FEATURE_COLUMNS
from src.models.evaluate import calibration_analysis, compute_rps
from src.models.goals_model import GOALS_FEATURE_COLUMNS, GoalsModel
from src.models.outcome_model import OutcomeModel
from src.models.train import time_series_split

MODEL_CFG = ModelConfig(
    random_seed=42,
    test_size=0.2,
    cv_folds=3,
    training_data_cutoff="2010-01-01",
    xgboost=XGBoostConfig(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=5,
    ),
    poisson=PoissonConfig(alpha=0.1, max_iter=100),
)


def _make_synthetic_dataset(n: int = 200) -> pd.DataFrame:
    """Generate a synthetic match dataset for model testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
    labels = rng.integers(0, 3, size=n)
    home_scores = rng.poisson(1.5, size=n)
    away_scores = rng.poisson(1.2, size=n)

    data = {col: rng.uniform(-1, 1, size=n) for col in FEATURE_COLUMNS}
    data["is_neutral"] = rng.integers(0, 2, size=n).astype(float)
    data["tournament_weight"] = rng.uniform(1.0, 3.0, size=n)
    data["label"] = labels
    data["sample_weight"] = np.ones(n)
    data["home_score"] = home_scores.astype(float)
    data["away_score"] = away_scores.astype(float)
    data["date"] = dates
    data["home_team"] = "TeamA"
    data["away_team"] = "TeamB"
    data["tournament"] = "Friendly"

    return pd.DataFrame(data)


class TestOutcomeModel:
    def _fit_model(self):
        ds = _make_synthetic_dataset(200)
        train, test = ds.iloc[:160], ds.iloc[160:]
        model = OutcomeModel(MODEL_CFG)
        model.fit(
            train[FEATURE_COLUMNS],
            train["label"],
            np.ones(160),
            test[FEATURE_COLUMNS],
            test["label"],
        )
        return model, test

    def test_proba_sum_to_one(self):
        model, test = self._fit_model()
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_proba_shape(self):
        model, test = self._fit_model()
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        assert proba.shape == (len(test), 3)

    def test_proba_in_range(self):
        model, test = self._fit_model()
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_match_keys(self):
        model, test = self._fit_model()
        fv = test[FEATURE_COLUMNS].iloc[0]
        result = model.predict_match(fv)
        for key in ["away_win_prob", "draw_prob", "home_win_prob"]:
            assert key in result

    def test_predict_match_sums_to_one(self):
        model, test = self._fit_model()
        fv = test[FEATURE_COLUMNS].iloc[0]
        result = model.predict_match(fv)
        total = result["away_win_prob"] + result["draw_prob"] + result["home_win_prob"]
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_save_load(self, tmp_path):
        model, test = self._fit_model()
        path = tmp_path / "outcome.pkl"
        model.save(path)
        loaded = OutcomeModel.load(path)
        orig_proba = model.predict_proba(test[FEATURE_COLUMNS])
        loaded_proba = loaded.predict_proba(test[FEATURE_COLUMNS])
        np.testing.assert_allclose(orig_proba, loaded_proba, atol=1e-6)

    def test_unfitted_raises(self):
        model = OutcomeModel(MODEL_CFG)
        with pytest.raises(RuntimeError):
            model.predict_proba(pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}]))


class TestGoalsModel:
    def _fit_goals_model(self):
        ds = _make_synthetic_dataset(200)
        # Make features non-negative for Poisson GLM log-link stability
        x = ds[FEATURE_COLUMNS].abs() + 0.1
        x["is_neutral"] = ds["is_neutral"]
        x["tournament_weight"] = ds["tournament_weight"]
        train_x = x.iloc[:160]
        model = GoalsModel(MODEL_CFG)
        model.fit(
            train_x,
            ds["home_score"].iloc[:160],
            ds["away_score"].iloc[:160],
            np.ones(160),
        )
        return model, x.iloc[160:]

    def test_predict_nonnegative(self):
        model, test_x = self._fit_goals_model()
        fv = test_x.iloc[0]
        # Build proper Series with GOALS_FEATURE_COLUMNS
        fv_goals = fv[GOALS_FEATURE_COLUMNS] if all(c in fv.index for c in GOALS_FEATURE_COLUMNS) else pd.Series({c: abs(fv.get(c, 0.1)) for c in GOALS_FEATURE_COLUMNS})
        home_g, away_g = model.predict_goals(fv_goals)
        assert home_g >= 0
        assert away_g >= 0

    def test_unfitted_raises(self):
        model = GoalsModel(MODEL_CFG)
        fv = pd.Series({c: 0.5 for c in GOALS_FEATURE_COLUMNS})
        with pytest.raises(RuntimeError):
            model.predict_goals(fv)

    def test_save_load(self, tmp_path):
        ds = _make_synthetic_dataset(200)
        x = ds[FEATURE_COLUMNS].abs() + 0.1
        train_x = x.iloc[:160]
        model = GoalsModel(MODEL_CFG)
        model.fit(
            train_x,
            ds["home_score"].iloc[:160],
            ds["away_score"].iloc[:160],
            np.ones(160),
        )
        model.save(tmp_path / "home.pkl", tmp_path / "away.pkl")
        loaded = GoalsModel.load(MODEL_CFG, tmp_path / "home.pkl", tmp_path / "away.pkl")
        fv = pd.Series({c: 0.5 for c in GOALS_FEATURE_COLUMNS})
        orig = model.predict_goals(fv)
        loaded_result = loaded.predict_goals(fv)
        assert orig == pytest.approx(loaded_result, abs=1e-6)


class TestEvaluate:
    def test_rps_perfect_prediction(self):
        y_true = np.array([0, 1, 2])
        y_proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        assert compute_rps(y_true, y_proba) == pytest.approx(0.0)

    def test_rps_worst_prediction(self):
        # Predicting the exact opposite of truth
        y_true = np.array([2])
        y_proba = np.array([[1.0, 0.0, 0.0]])
        rps = compute_rps(y_true, y_proba)
        assert rps > 0

    def test_rps_uniform_prediction(self):
        y_true = np.array([0, 1, 2] * 10)
        y_proba = np.full((30, 3), 1 / 3)
        rps = compute_rps(y_true, y_proba)
        # RPS for uniform = 1/3 * (sum of squared cumulative diffs)
        assert 0 < rps < 1

    def test_calibration_analysis_returns_df(self):
        y_true = np.array([0, 1, 0, 1, 1, 0] * 10)
        y_proba = np.random.uniform(0, 1, 60)
        df = calibration_analysis(y_true, y_proba)
        assert isinstance(df, pd.DataFrame)
        assert "bin_mean_prob" in df.columns
        assert "actual_freq" in df.columns


class TestTimeSeriesSplit:
    def test_no_data_leakage(self):
        ds = _make_synthetic_dataset(100)
        train, test = time_series_split(ds, test_size=0.2)
        assert train["date"].max() <= test["date"].min()

    def test_correct_sizes(self):
        ds = _make_synthetic_dataset(100)
        train, test = time_series_split(ds, test_size=0.2)
        assert len(train) + len(test) == 100
        assert len(test) == pytest.approx(20, abs=2)

    def test_chronological_order(self):
        ds = _make_synthetic_dataset(100)
        train, test = time_series_split(ds, test_size=0.2)
        assert (train["date"].diff().dropna() >= pd.Timedelta(0)).all()
        assert (test["date"].diff().dropna() >= pd.Timedelta(0)).all()
