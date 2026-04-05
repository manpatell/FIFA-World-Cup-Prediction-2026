"""2026 FIFA World Cup bracket simulation.

Implements the 48-team, 12-group format with:
- Group stage (72 matches across groups A-L)
- Best 8 of 12 third-placed teams advance to Round of 32
- Single-elimination knockout from Round of 32 → Final
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.match_features import FEATURE_COLUMNS, build_match_feature_vector
from src.models.goals_model import GoalsModel
from src.models.outcome_model import OutcomeModel
from src.simulation.shootout import ShootoutModel

# 2026 Round of 32 seeding rules: which 3rd-place groups qualify each slot
# Format: match_label → (team_slot_a_description, team_slot_b_description)
# Derived from matches.csv match_label column
_R32_MATCH_LABELS = [
    "2A vs 2B",
    "1C vs 2F",
    "1E vs 3ABCDF",
    "1F vs 2C",
    "2E vs 2I",
    "1I vs 3CDFGH",
    "1A vs 3CEFHI",
    "1L vs 3EHIJK",
    "1G vs 3AEHIJ",
    "1D vs 3BEFIJ",
    "1H vs 2J",
    "2K vs 2L",
    "1B vs 3EFGIJ",
    "2D vs 2G",
    "1J vs 2H",
    "1K vs 3DEIJL",
]

STAGE_NAMES = {
    1: "Group Stage",
    2: "Round of 32",
    3: "Round of 16",
    4: "Quarterfinals",
    5: "Semifinals",
    6: "Third Place Playoff",
    7: "Final",
}

STAGE_EXIT_LABELS = {
    1: "group_exit",
    2: "round_of_32_exit",
    3: "round_of_16_exit",
    4: "quarterfinal_exit",
    5: "semifinal_exit",
    6: "third_place",
    7: "runner_up",
}


@dataclass
class MatchResult:
    """Result of a single simulated match."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    went_to_shootout: bool = False
    shootout_winner: str | None = None

    @property
    def winner(self) -> str | None:
        """Return winning team, or shootout winner if applicable."""
        if self.shootout_winner:
            return self.shootout_winner
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return None  # Draw (group stage)


@dataclass
class GroupStandings:
    """Standings for a single group."""
    group: str
    teams: list[str]
    points: dict[str, int] = field(default_factory=dict)
    goal_diff: dict[str, int] = field(default_factory=dict)
    goals_for: dict[str, int] = field(default_factory=dict)
    goals_against: dict[str, int] = field(default_factory=dict)
    matches_played: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for team in self.teams:
            self.points.setdefault(team, 0)
            self.goal_diff.setdefault(team, 0)
            self.goals_for.setdefault(team, 0)
            self.goals_against.setdefault(team, 0)
            self.matches_played.setdefault(team, 0)

    def update(self, result: MatchResult) -> None:
        """Apply a match result to the group standings."""
        h, a = result.home_team, result.away_team
        hs, as_ = result.home_score, result.away_score

        self.goals_for[h] = self.goals_for.get(h, 0) + hs
        self.goals_for[a] = self.goals_for.get(a, 0) + as_
        self.goals_against[h] = self.goals_against.get(h, 0) + as_
        self.goals_against[a] = self.goals_against.get(a, 0) + hs
        self.goal_diff[h] = self.goals_for[h] - self.goals_against[h]
        self.goal_diff[a] = self.goals_for[a] - self.goals_against[a]
        self.matches_played[h] = self.matches_played.get(h, 0) + 1
        self.matches_played[a] = self.matches_played.get(a, 0) + 1

        if hs > as_:
            self.points[h] = self.points.get(h, 0) + 3
        elif as_ > hs:
            self.points[a] = self.points.get(a, 0) + 3
        else:
            self.points[h] = self.points.get(h, 0) + 1
            self.points[a] = self.points.get(a, 0) + 1


def rank_group(standings: GroupStandings) -> list[str]:
    """Rank teams in a group by FIFA tiebreaker rules.

    Criteria: points → goal diff → goals for → alphabetical (placeholder).

    Args:
        standings: GroupStandings after all group matches played.

    Returns:
        List of team names ordered 1st to last.
    """
    return sorted(
        standings.teams,
        key=lambda t: (
            -standings.points.get(t, 0),
            -standings.goal_diff.get(t, 0),
            -standings.goals_for.get(t, 0),
            t,  # alphabetical as final tiebreaker
        ),
    )


def select_best_third_place_teams(
    all_third_place: list[tuple[str, GroupStandings]],
    n: int = 8,
) -> list[str]:
    """Select the best N third-placed teams from all groups.

    Applies FIFA criteria: points → goal diff → goals for → alphabetical.

    Args:
        all_third_place: List of (team_name, standings) for each group's 3rd-place team.
        n: Number of third-place teams to advance (8 for 2026 format).

    Returns:
        List of N best third-placed team names.
    """
    sorted_third = sorted(
        all_third_place,
        key=lambda x: (
            -x[1].points.get(x[0], 0),
            -x[1].goal_diff.get(x[0], 0),
            -x[1].goals_for.get(x[0], 0),
            x[0],
        ),
    )
    return [team for team, _ in sorted_third[:n]]


def simulate_group_match(
    home_team: str,
    away_team: str,
    outcome_model: OutcomeModel,
    goals_model: GoalsModel,
    feature_kwargs: dict,
    rng: np.random.Generator,
    feature_cache: dict | None = None,
) -> MatchResult:
    """Simulate a single group stage match.

    Samples an outcome from the classifier, then samples goal counts
    from the Poisson goals model, ensuring consistency with the sampled outcome.

    Args:
        home_team: Canonical home team name.
        away_team: Canonical away team name.
        outcome_model: Fitted OutcomeModel.
        goals_model: Fitted GoalsModel.
        feature_kwargs: Dict of kwargs for build_match_feature_vector
            (elo_ratings, rankings, squad_features, results_long, form_window).
        rng: NumPy random generator.
        feature_cache: Optional pre-computed feature vector cache for speed.

    Returns:
        MatchResult with home_score, away_score.
    """
    if feature_cache is not None and (home_team, away_team) in feature_cache:
        fv = feature_cache[(home_team, away_team)]
    else:
        fv = build_match_feature_vector(
            home_team=home_team,
            away_team=away_team,
            match_date=pd.Timestamp("2026-06-15"),
            is_neutral=True,
            tournament="FIFA World Cup",
            **feature_kwargs,
        )

    probs = outcome_model.predict_match(fv)
    p_arr = np.array([probs["home_win_prob"], probs["draw_prob"], probs["away_win_prob"]])
    p_arr = np.clip(p_arr, 0.0, 1.0)
    p_arr = p_arr / p_arr.sum()  # normalize to exactly 1.0

    # Sample outcome
    outcome = rng.choice(["home_win", "draw", "away_win"], p=p_arr)

    # Sample goal counts from Poisson (cap lambda to prevent overflow)
    exp_home, exp_away = goals_model.predict_goals(fv)
    exp_home = min(max(0.1, exp_home), 10.0)
    exp_away = min(max(0.1, exp_away), 10.0)

    # Sample then correct to match outcome
    max_tries = 5
    for _ in range(max_tries):
        h = int(rng.poisson(exp_home))
        a = int(rng.poisson(exp_away))
        if outcome == "home_win" and h > a:
            break
        if outcome == "draw" and h == a:
            break
        if outcome == "away_win" and a > h:
            break
    else:
        # Force consistency
        if outcome == "home_win":
            h, a = max(1, h), max(0, h - 1)
        elif outcome == "away_win":
            a, h = max(1, a), max(0, a - 1)
        else:
            h = a = max(h, a)  # use higher of two for draw

    return MatchResult(home_team=home_team, away_team=away_team, home_score=h, away_score=a)


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    outcome_model: OutcomeModel,
    goals_model: GoalsModel,
    shootout_model: ShootoutModel,
    feature_kwargs: dict,
    rng: np.random.Generator,
    feature_cache: dict | None = None,
) -> str:
    """Simulate a knockout match, always producing a winner.

    If the match ends in a draw, simulates a penalty shootout.

    Args:
        home_team: Team A.
        away_team: Team B.
        outcome_model: Fitted OutcomeModel.
        goals_model: Fitted GoalsModel.
        shootout_model: ShootoutModel for penalty outcomes.
        feature_kwargs: Feature builder keyword arguments.
        rng: NumPy random generator.

    Returns:
        Canonical name of the winning team.
    """
    result = simulate_group_match(
        home_team, away_team, outcome_model, goals_model, feature_kwargs, rng,
        feature_cache=feature_cache,
    )

    if result.home_score > result.away_score:
        return home_team
    if result.away_score > result.home_score:
        return away_team

    # Draw → penalty shootout
    winner = shootout_model.predict_winner(home_team, away_team, rng)
    return winner
