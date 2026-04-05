"""Tests for simulation modules: shootout, bracket, monte_carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig, PoissonConfig, XGBoostConfig
from src.simulation.bracket import (
    GroupStandings,
    MatchResult,
    rank_group,
    select_best_third_place_teams,
    simulate_knockout_match,
)
from src.simulation.monte_carlo import run_single_simulation
from src.simulation.shootout import ShootoutModel, compute_shootout_win_rates

MODEL_CFG = ModelConfig(
    random_seed=42,
    test_size=0.2,
    cv_folds=3,
    training_data_cutoff="2010-01-01",
    xgboost=XGBoostConfig(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=3,
    ),
    poisson=PoissonConfig(alpha=0.1, max_iter=50),
)


def _make_dummy_models():
    """Create minimal fitted OutcomeModel and GoalsModel for testing."""
    import numpy as np
    import pandas as pd
    from src.features.match_features import FEATURE_COLUMNS
    from src.models.goals_model import GOALS_FEATURE_COLUMNS, GoalsModel
    from src.models.outcome_model import OutcomeModel

    rng = np.random.default_rng(42)
    n = 300

    data = {col: rng.uniform(-1, 1, size=n) for col in FEATURE_COLUMNS}
    data["is_neutral"] = rng.integers(0, 2, size=n).astype(float)
    data["tournament_weight"] = rng.uniform(1.0, 3.0, size=n)
    df = pd.DataFrame(data)
    x_train, x_val = df.iloc[:250], df.iloc[250:]
    labels = rng.integers(0, 3, size=n)

    outcome = OutcomeModel(MODEL_CFG)
    outcome.fit(x_train, pd.Series(labels[:250]), np.ones(250), x_val, pd.Series(labels[250:]))

    goals = GoalsModel(MODEL_CFG)
    goals.fit(df.iloc[:250].abs() + 0.1, pd.Series(rng.poisson(1.5, 250).astype(float)), pd.Series(rng.poisson(1.2, 250).astype(float)), np.ones(250))

    return outcome, goals


def _make_dummy_feature_kwargs():
    """Create minimal feature_kwargs for simulation."""
    teams = ["France", "Germany", "Brazil", "Argentina", "Spain", "England",
             "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
             "Mexico", "USA", "Morocco", "Senegal"]
    elo_ratings = pd.Series({t: 1500.0 + i * 10 for i, t in enumerate(teams)})
    rankings = pd.DataFrame({
        "team_name_canonical": teams,
        "rank": list(range(1, len(teams) + 1)),
        "total_points": [1877.0 - i * 10 for i in range(len(teams))],
    })
    squad_features = pd.DataFrame({
        "injury_adj_value": [1e9] * len(teams),
        "injury_pct_value_lost": [0.0] * len(teams),
        "avg_age": [26.0] * len(teams),
        "avg_caps": [40.0] * len(teams),
    }, index=teams)
    results_long = pd.DataFrame(columns=["date", "team", "opponent", "goals_for", "goals_against", "result", "tournament", "neutral", "is_home"])

    return {
        "elo_ratings": elo_ratings,
        "rankings": rankings,
        "squad_features": squad_features,
        "results_long": results_long,
        "form_window": 10,
    }


# ─── Shootout tests ───────────────────────────────────────────────────────────

class TestShootoutModel:
    def _make_shootouts(self):
        return pd.DataFrame({
            "home_team": ["France", "Germany", "Brazil", "France"],
            "away_team": ["Germany", "Brazil", "France", "Brazil"],
            "winner": ["France", "Germany", "Brazil", "France"],
            "date": pd.to_datetime(["2018-07-01"] * 4),
            "first_shooter": ["France", "Germany", "Brazil", "France"],
        })

    def test_win_rates_in_range(self):
        shootouts = self._make_shootouts()
        rates = compute_shootout_win_rates(shootouts)
        for team, rate in rates.items():
            assert 0 <= rate <= 1, f"{team} win rate {rate} out of range"

    def test_all_participants_have_rate(self):
        shootouts = self._make_shootouts()
        rates = compute_shootout_win_rates(shootouts)
        for team in ["France", "Germany", "Brazil"]:
            assert team in rates

    def test_predict_winner_returns_one_of_two(self):
        shootouts = self._make_shootouts()
        model = ShootoutModel(shootouts)
        rng = np.random.default_rng(42)
        winner = model.predict_winner("France", "Germany", rng)
        assert winner in ["France", "Germany"]

    def test_unknown_team_uses_global_mean(self):
        shootouts = self._make_shootouts()
        model = ShootoutModel(shootouts)
        p = model.get_win_probability("UnknownTeamXYZ", "France")
        assert 0 <= p <= 1

    def test_probability_sum_to_one(self):
        shootouts = self._make_shootouts()
        model = ShootoutModel(shootouts)
        p_a = model.get_win_probability("France", "Germany")
        p_b = model.get_win_probability("Germany", "France")
        assert p_a + p_b == pytest.approx(1.0)


# ─── Bracket tests ────────────────────────────────────────────────────────────

class TestGroupStandings:
    def test_update_home_win(self):
        s = GroupStandings(group="A", teams=["France", "Germany"])
        result = MatchResult("France", "Germany", 2, 1)
        s.update(result)
        assert s.points["France"] == 3
        assert s.points["Germany"] == 0

    def test_update_draw(self):
        s = GroupStandings(group="A", teams=["France", "Germany"])
        result = MatchResult("France", "Germany", 1, 1)
        s.update(result)
        assert s.points["France"] == 1
        assert s.points["Germany"] == 1

    def test_goal_diff_updated(self):
        s = GroupStandings(group="A", teams=["France", "Germany"])
        s.update(MatchResult("France", "Germany", 3, 1))
        assert s.goal_diff["France"] == 2
        assert s.goal_diff["Germany"] == -2


class TestRankGroup:
    def test_order_by_points(self):
        s = GroupStandings(group="A", teams=["France", "Germany", "Brazil"])
        s.update(MatchResult("France", "Germany", 2, 0))
        s.update(MatchResult("Brazil", "France", 1, 0))
        s.update(MatchResult("Brazil", "Germany", 2, 0))
        ranked = rank_group(s)
        assert ranked[0] == "Brazil"  # 6 points

    def test_tiebreak_by_goal_diff(self):
        s = GroupStandings(group="A", teams=["France", "Germany"])
        s.points = {"France": 3, "Germany": 3}
        s.goal_diff = {"France": 2, "Germany": 1}
        s.goals_for = {"France": 2, "Germany": 1}
        ranked = rank_group(s)
        assert ranked[0] == "France"


class TestSelectBestThirdPlace:
    def test_returns_correct_count(self):
        thirds = []
        for i, team in enumerate(["A3", "B3", "C3", "D3", "E3", "F3",
                                   "G3", "H3", "I3", "J3", "K3", "L3"]):
            s = GroupStandings(group=chr(65 + i), teams=[team])
            s.points[team] = 12 - i
            s.goal_diff[team] = 5 - i
            s.goals_for[team] = 8 - i
            thirds.append((team, s))
        best = select_best_third_place_teams(thirds, n=8)
        assert len(best) == 8

    def test_best_teams_selected(self):
        thirds = []
        for i, team in enumerate(["A3", "B3", "C3"]):
            s = GroupStandings(group=chr(65 + i), teams=[team])
            s.points[team] = 9 - i * 3
            s.goal_diff[team] = 3 - i
            s.goals_for[team] = 3 - i
            thirds.append((team, s))
        best = select_best_third_place_teams(thirds, n=2)
        assert "A3" in best  # highest points


# ─── Monte Carlo tests ────────────────────────────────────────────────────────

class TestRunSingleSimulation:
    def _make_mini_teams(self):
        teams = ["France", "Germany", "Brazil", "Argentina",
                 "Spain", "England", "Portugal", "Netherlands",
                 "Belgium", "Italy", "Croatia", "Uruguay",
                 "Mexico", "USA", "Morocco", "Senegal"]
        groups = list("ABCDEFGHIJKL") + ["A", "B", "C", "D"]
        rows = []
        for i, team in enumerate(teams):
            rows.append({
                "id": i + 1,
                "team_name": team,
                "fifa_code": team[:3].upper(),
                "group_letter": chr(65 + (i % 12)),
                "is_placeholder": False,
            })
        return pd.DataFrame(rows)

    def test_exactly_one_winner(self):
        outcome, goals = _make_dummy_models()
        shootouts = pd.DataFrame({
            "home_team": ["France"],
            "away_team": ["Germany"],
            "winner": ["France"],
            "date": pd.to_datetime(["2018-07-01"]),
            "first_shooter": ["France"],
        })
        shootout_model = ShootoutModel(shootouts)
        teams_df = self._make_mini_teams()
        rng = np.random.default_rng(42)
        fk = _make_dummy_feature_kwargs()
        # Expand feature_kwargs to cover all teams
        all_teams = list(teams_df["team_name"])
        fk["elo_ratings"] = pd.Series({t: 1500.0 for t in all_teams})
        fk["rankings"] = pd.DataFrame({
            "team_name_canonical": all_teams,
            "rank": list(range(1, len(all_teams) + 1)),
            "total_points": [1877.0 - i * 10 for i in range(len(all_teams))],
        })
        fk["squad_features"] = pd.DataFrame({
            "injury_adj_value": [1e9] * len(all_teams),
            "injury_pct_value_lost": [0.0] * len(all_teams),
            "avg_age": [26.0] * len(all_teams),
            "avg_caps": [40.0] * len(all_teams),
        }, index=all_teams)

        result = run_single_simulation(
            teams_df, outcome, goals, shootout_model, fk, rng
        )

        winners = [t for t, stage in result.items() if stage == "winner"]
        assert len(winners) == 1

    def test_all_teams_have_stage(self):
        outcome, goals = _make_dummy_models()
        shootouts = pd.DataFrame({
            "home_team": ["France"],
            "away_team": ["Germany"],
            "winner": ["France"],
            "date": pd.to_datetime(["2018-07-01"]),
            "first_shooter": ["France"],
        })
        shootout_model = ShootoutModel(shootouts)
        teams_df = self._make_mini_teams()
        rng = np.random.default_rng(99)
        fk = _make_dummy_feature_kwargs()
        all_teams = list(teams_df["team_name"])
        fk["elo_ratings"] = pd.Series({t: 1500.0 for t in all_teams})
        fk["rankings"] = pd.DataFrame({
            "team_name_canonical": all_teams,
            "rank": list(range(1, len(all_teams) + 1)),
            "total_points": [1877.0 - i * 10 for i in range(len(all_teams))],
        })
        fk["squad_features"] = pd.DataFrame({
            "injury_adj_value": [1e9] * len(all_teams),
            "injury_pct_value_lost": [0.0] * len(all_teams),
            "avg_age": [26.0] * len(all_teams),
            "avg_caps": [40.0] * len(all_teams),
        }, index=all_teams)

        result = run_single_simulation(
            teams_df, outcome, goals, shootout_model, fk, rng
        )
        for team in teams_df["team_name"]:
            assert team in result, f"{team} missing from simulation result"
