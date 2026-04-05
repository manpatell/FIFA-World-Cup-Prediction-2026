"""Tests for form, h2h, squad, and match_features modules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.form import (
    build_results_long,
    compute_form,
    compute_match_result,
)
from src.features.h2h import compute_h2h
from src.features.match_features import FEATURE_COLUMNS, build_match_feature_vector
from src.features.squad import identify_injured_players


# ─── Form tests ───────────────────────────────────────────────────────────────

class TestComputeMatchResult:
    def test_home_win(self):
        assert compute_match_result(3, 1) == ("W", "L")

    def test_away_win(self):
        assert compute_match_result(0, 2) == ("L", "W")

    def test_draw(self):
        assert compute_match_result(1, 1) == ("D", "D")

    def test_zero_zero(self):
        assert compute_match_result(0, 0) == ("D", "D")


class TestBuildResultsLong:
    def test_doubles_rows(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        assert len(long_df) == len(mini_results) * 2

    def test_required_columns(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        for col in ["team", "opponent", "goals_for", "goals_against", "result", "is_home"]:
            assert col in long_df.columns

    def test_result_values(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        assert set(long_df["result"].unique()).issubset({"W", "D", "L"})


class TestComputeForm:
    def test_returns_required_keys(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        form = compute_form(long_df, "France", pd.Timestamp("2023-01-01"), window=10)
        for key in ["form_points", "win_rate", "draw_rate", "loss_rate",
                    "goals_scored_avg", "goals_conceded_avg", "goal_diff_avg", "n_matches"]:
            assert key in form

    def test_rates_sum_to_one(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        form = compute_form(long_df, "France", pd.Timestamp("2023-01-01"), window=10)
        total = form["win_rate"] + form["draw_rate"] + form["loss_rate"]
        assert total == pytest.approx(1.0) or form["n_matches"] == 0

    def test_no_matches_returns_zeros(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        form = compute_form(long_df, "France", pd.Timestamp("1800-01-01"), window=10)
        assert form["n_matches"] == 0
        assert form["form_points"] == 0.0

    def test_window_respected(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        form = compute_form(long_df, "France", pd.Timestamp("2025-01-01"), window=3)
        assert form["n_matches"] <= 3

    def test_form_points_formula(self, mini_results):
        """form_points should equal 3*wins + draws."""
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        form = compute_form(long_df, "France", pd.Timestamp("2023-01-01"), window=10)
        n = form["n_matches"]
        expected = form["win_rate"] * n * 3 + form["draw_rate"] * n
        assert form["form_points"] == pytest.approx(expected, abs=0.01)


# ─── H2H tests ────────────────────────────────────────────────────────────────

class TestComputeH2H:
    def test_returns_required_keys(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        h2h = compute_h2h(long_df, "France", "Germany", pd.Timestamp("2025-01-01"))
        for key in ["h2h_win_rate_a", "h2h_draw_rate", "h2h_win_rate_b",
                    "h2h_goal_diff_avg", "h2h_n_matches"]:
            assert key in h2h

    def test_no_history_returns_neutral(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        h2h = compute_h2h(long_df, "France", "Germany", pd.Timestamp("1800-01-01"))
        assert h2h["h2h_win_rate_a"] == pytest.approx(0.5)
        assert h2h["h2h_win_rate_b"] == pytest.approx(0.5)
        assert h2h["h2h_n_matches"] == 0

    def test_win_rates_sum_to_one_minus_draw(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        h2h = compute_h2h(long_df, "France", "Germany", pd.Timestamp("2025-01-01"))
        total = h2h["h2h_win_rate_a"] + h2h["h2h_draw_rate"] + h2h["h2h_win_rate_b"]
        assert total == pytest.approx(1.0)

    def test_n_matches_capped_at_window(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        long_df = build_results_long(mini_results)
        h2h = compute_h2h(long_df, "France", "Germany", pd.Timestamp("2025-01-01"), n_matches=2)
        assert h2h["h2h_n_matches"] <= 2


# ─── Squad tests ──────────────────────────────────────────────────────────────

class TestIdentifyInjuredPlayers:
    def _make_injury_df(self):
        return pd.DataFrame({
            "player_id": [1, 2, 3, 4],
            "season_name": ["24/25"] * 4,
            "injury_reason": ["Knee", "Ankle", "Hamstring", "Back"],
            "from_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2024-01-01", "2025-03-01"]),
            "end_date": pd.to_datetime([None, "2025-03-01", "2024-06-01", "2025-12-31"]),
            "days_missed": [0, 30, 60, 90],
            "games_missed": [0, 3, 5, 8],
        })

    def test_ongoing_injury_detected(self):
        injuries = self._make_injury_df()
        ref_date = pd.Timestamp("2025-04-05")
        injured = identify_injured_players(injuries, [1, 2, 3, 4], ref_date, 90)
        assert 1 in injured  # ongoing (end_date is null)

    def test_recent_injury_detected(self):
        injuries = self._make_injury_df()
        ref_date = pd.Timestamp("2025-04-05")
        injured = identify_injured_players(injuries, [1, 2, 3, 4], ref_date, 90)
        assert 2 in injured  # ended 2025-03-01, within 90 days

    def test_old_injury_not_detected(self):
        injuries = self._make_injury_df()
        ref_date = pd.Timestamp("2025-04-05")
        injured = identify_injured_players(injuries, [1, 2, 3, 4], ref_date, 90)
        assert 3 not in injured  # ended 2024-06-01, more than 90 days ago

    def test_empty_player_list(self):
        injuries = self._make_injury_df()
        injured = identify_injured_players(injuries, [], pd.Timestamp("2025-04-05"), 90)
        assert injured == set()

    def test_returns_set(self):
        injuries = self._make_injury_df()
        result = identify_injured_players(injuries, [1], pd.Timestamp("2025-04-05"), 90)
        assert isinstance(result, set)


# ─── Match features tests ─────────────────────────────────────────────────────

class TestBuildMatchFeatureVector:
    def _make_inputs(self, mini_results):
        mini_results = mini_results.copy()
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        from src.features.form import build_results_long
        results_long = build_results_long(mini_results)

        elo_ratings = pd.Series({
            "France": 1700.0, "Germany": 1650.0, "Brazil": 1680.0,
            "Argentina": 1720.0, "Spain": 1690.0, "England": 1640.0,
        })

        rankings = pd.DataFrame({
            "team_name_canonical": ["France", "Germany", "Brazil", "Argentina", "Spain", "England"],
            "country": ["France", "Germany", "Brazil", "Argentina", "Spain", "England"],
            "rank": [1, 10, 6, 3, 2, 4],
            "total_points": [1877.0, 1730.0, 1761.0, 1874.0, 1876.0, 1826.0],
        })

        squad_features = pd.DataFrame({
            "injury_adj_value": [1e9, 8e8, 9e8, 7.5e8, 1.1e9, 1.2e9],
            "injury_pct_value_lost": [0.0, 0.1, 0.05, 0.15, 0.0, 0.08],
            "avg_age": [26.0, 28.0, 27.0, 29.0, 26.5, 27.5],
            "avg_caps": [40.0, 35.0, 50.0, 60.0, 38.0, 42.0],
        }, index=["France", "Germany", "Brazil", "Argentina", "Spain", "England"])

        return results_long, elo_ratings, rankings, squad_features

    def test_returns_series_correct_length(self, mini_results):
        results_long, elo_ratings, rankings, squad_features = self._make_inputs(mini_results)
        fv = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            True, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        assert isinstance(fv, pd.Series)
        assert len(fv) == len(FEATURE_COLUMNS)

    def test_correct_column_names(self, mini_results):
        results_long, elo_ratings, rankings, squad_features = self._make_inputs(mini_results)
        fv = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            True, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        assert list(fv.index) == FEATURE_COLUMNS

    def test_no_nans(self, mini_results):
        results_long, elo_ratings, rankings, squad_features = self._make_inputs(mini_results)
        fv = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            True, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        assert not fv.isna().any(), f"NaN found in features: {fv[fv.isna()].index.tolist()}"

    def test_is_neutral_flag(self, mini_results):
        results_long, elo_ratings, rankings, squad_features = self._make_inputs(mini_results)
        fv_neutral = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            True, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        fv_home = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            False, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        assert fv_neutral["is_neutral"] == 1.0
        assert fv_home["is_neutral"] == 0.0

    def test_elo_diff_direction(self, mini_results):
        results_long, elo_ratings, rankings, squad_features = self._make_inputs(mini_results)
        # France (1700) vs Germany (1650): home team stronger → positive diff
        fv = build_match_feature_vector(
            "France", "Germany", pd.Timestamp("2025-01-01"),
            True, "Friendly", elo_ratings, rankings, squad_features, results_long,
        )
        assert fv["elo_diff"] == pytest.approx(1700.0 - 1650.0)

    def test_feature_columns_length(self):
        assert len(FEATURE_COLUMNS) == 30
