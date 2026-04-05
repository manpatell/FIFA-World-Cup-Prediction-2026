"""Tests for src/features/elo.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import EloConfig
from src.features.elo import (
    build_elo_history,
    compute_expected_score,
    get_current_elo,
    get_k_factor,
    update_elo,
)

K_FACTORS = {
    "FIFA World Cup": 60,
    "Friendly": 20,
    "default": 30,
}

ELO_CFG = EloConfig(
    initial_rating=1500.0,
    k_factors=K_FACTORS,
    home_advantage=100.0,
    margin_of_victory_mult=True,
)


class TestComputeExpectedScore:
    def test_equal_ratings_give_half(self):
        assert compute_expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_wins_more(self):
        assert compute_expected_score(1600, 1500) > 0.5

    def test_lower_rating_wins_less(self):
        assert compute_expected_score(1400, 1500) < 0.5

    def test_sum_to_one(self):
        e_a = compute_expected_score(1600, 1400)
        e_b = compute_expected_score(1400, 1600)
        assert e_a + e_b == pytest.approx(1.0)

    def test_extreme_difference(self):
        e = compute_expected_score(2000, 1000)
        assert e > 0.99


class TestGetKFactor:
    def test_wc_k_factor(self):
        k = get_k_factor("FIFA World Cup", 1, 0.0, K_FACTORS, margin_mult=False)
        assert k == pytest.approx(60.0)

    def test_friendly_k_factor(self):
        k = get_k_factor("Friendly", 1, 0.0, K_FACTORS, margin_mult=False)
        assert k == pytest.approx(20.0)

    def test_default_k_factor(self):
        k = get_k_factor("Unknown Tournament", 1, 0.0, K_FACTORS, margin_mult=False)
        assert k == pytest.approx(30.0)

    def test_margin_mult_increases_k_for_large_diff(self):
        k_base = get_k_factor("Friendly", 1, 0.0, K_FACTORS, margin_mult=False)
        k_mult = get_k_factor("Friendly", 4, 0.0, K_FACTORS, margin_mult=True)
        assert k_mult > k_base

    def test_no_margin_mult_no_change(self):
        k1 = get_k_factor("Friendly", 3, 200.0, K_FACTORS, margin_mult=False)
        k2 = get_k_factor("Friendly", 0, 200.0, K_FACTORS, margin_mult=False)
        assert k1 == k2


class TestUpdateElo:
    def test_winner_gains_loser_loses(self):
        new_h, new_a = update_elo(
            1500, 1500, 2, 0, "Friendly", True, 100.0, K_FACTORS, False
        )
        assert new_h > 1500
        assert new_a < 1500

    def test_draw_equal_teams_no_change(self):
        new_h, new_a = update_elo(
            1500, 1500, 1, 1, "Friendly", True, 100.0, K_FACTORS, False
        )
        # Draw between equal teams → no rating change
        assert new_h == pytest.approx(1500.0)
        assert new_a == pytest.approx(1500.0)

    def test_sum_conserved(self):
        """Total Elo should be conserved across a match."""
        h, a = 1600.0, 1400.0
        new_h, new_a = update_elo(h, a, 1, 0, "Friendly", True, 100.0, K_FACTORS, False)
        assert new_h + new_a == pytest.approx(h + a)

    def test_home_advantage_applied_on_non_neutral(self):
        """Home team should gain less when winning at home vs neutral."""
        # Home team wins at home venue
        h_home, _ = update_elo(1500, 1500, 2, 1, "Friendly", False, 100.0, K_FACTORS, False)
        # Home team wins at neutral venue
        h_neutral, _ = update_elo(1500, 1500, 2, 1, "Friendly", True, 100.0, K_FACTORS, False)
        # At home, the home team was already expected to win more → gains less
        assert h_home < h_neutral

    def test_upset_winner_gains_more(self):
        """An underdog winning should gain more than a favourite winning."""
        # Underdog (away) wins
        _, new_a_upset = update_elo(1700, 1300, 0, 1, "Friendly", True, 100.0, K_FACTORS, False)
        # Favourite (home) wins
        new_h_fav, _ = update_elo(1700, 1300, 1, 0, "Friendly", True, 100.0, K_FACTORS, False)
        upset_gain = new_a_upset - 1300
        fav_gain = new_h_fav - 1700
        assert upset_gain > fav_gain


class TestBuildEloHistory:
    def test_returns_dataframe(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        assert isinstance(history, pd.DataFrame)

    def test_row_count_matches_valid_rows(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        assert len(history) == len(mini_results)

    def test_required_columns(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        for col in ["home_elo_before", "away_elo_before", "home_elo_after", "away_elo_after"]:
            assert col in history.columns

    def test_initial_ratings_are_default(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        # First match: both teams should start at initial_rating
        first_row = history.iloc[0]
        assert first_row["home_elo_before"] == pytest.approx(ELO_CFG.initial_rating)
        assert first_row["away_elo_before"] == pytest.approx(ELO_CFG.initial_rating)

    def test_elo_after_differs_from_before(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        # At least some matches should change the Elo (non-draws)
        changed = (history["home_elo_before"] != history["home_elo_after"]).any()
        assert changed


class TestGetCurrentElo:
    def test_returns_series(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        current = get_current_elo(history)
        assert isinstance(current, pd.Series)

    def test_all_teams_present(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        current = get_current_elo(history)
        all_teams = set(mini_results["home_team"]) | set(mini_results["away_team"])
        for team in all_teams:
            assert team in current.index

    def test_values_differ_from_initial(self, mini_results):
        mini_results["home_team_canonical"] = mini_results["home_team"]
        mini_results["away_team_canonical"] = mini_results["away_team"]
        history = build_elo_history(mini_results, ELO_CFG)
        current = get_current_elo(history)
        # After 20 matches, not all teams should be exactly at 1500
        assert not (current == ELO_CFG.initial_rating).all()
