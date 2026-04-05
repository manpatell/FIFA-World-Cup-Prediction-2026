"""Penalty shootout probability engine.

Computes per-team historical shootout win rates from 675 historical
shootouts and uses them to simulate penalty outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shootout_win_rates(shootouts: pd.DataFrame) -> dict[str, float]:
    """Compute per-team shootout win rate from historical data.

    Args:
        shootouts: Normalized shootouts DataFrame with
            home_team_canonical, away_team_canonical, winner_canonical.

    Returns:
        Dict mapping canonical team name → historical win rate (0–1).
    """
    team_col = "home_team_canonical" if "home_team_canonical" in shootouts.columns else "home_team"
    away_col = "away_team_canonical" if "away_team_canonical" in shootouts.columns else "away_team"
    winner_col = "winner_canonical" if "winner_canonical" in shootouts.columns else "winner"

    participated: dict[str, int] = {}
    won: dict[str, int] = {}

    for row in shootouts.itertuples(index=False):
        home = getattr(row, team_col.replace("_canonical", "") if team_col == "home_team_canonical" else team_col, None)
        away = getattr(row, away_col.replace("_canonical", "") if away_col == "away_team_canonical" else away_col, None)
        winner = getattr(row, winner_col.replace("_canonical", "") if winner_col == "winner_canonical" else winner_col, None)

        # Use canonical columns if available
        if "home_team_canonical" in shootouts.columns:
            home = getattr(row, "home_team_canonical", home)
        if "away_team_canonical" in shootouts.columns:
            away = getattr(row, "away_team_canonical", away)
        if "winner_canonical" in shootouts.columns:
            winner = getattr(row, "winner_canonical", winner)

        for team in [home, away]:
            if not isinstance(team, str) or not team:
                continue
            participated[team] = participated.get(team, 0) + 1

        if isinstance(winner, str) and winner:
            won[winner] = won.get(winner, 0) + 1

    win_rates: dict[str, float] = {}
    for team, n in participated.items():
        w = won.get(team, 0)
        win_rates[team] = w / n if n > 0 else 0.5

    return win_rates


def compute_global_shootout_win_rate(shootouts: pd.DataFrame) -> float:
    """Compute the baseline first-shooter/overall average win rate.

    Args:
        shootouts: Shootouts DataFrame.

    Returns:
        Average historical win rate (close to 0.5).
    """
    return 0.5  # Theoretical and empirical average


class ShootoutModel:
    """Encapsulates penalty shootout probability logic.

    Uses per-team historical win rates where available, falls back
    to the global mean for teams with no shootout history.
    """

    def __init__(self, shootouts: pd.DataFrame) -> None:
        """Build per-team win rates from historical shootout data.

        Args:
            shootouts: Normalized shootouts DataFrame.
        """
        self._win_rates = compute_shootout_win_rates(shootouts)
        self._global_mean = compute_global_shootout_win_rate(shootouts)

    def get_win_probability(self, team_a: str, team_b: str) -> float:
        """Compute P(team_a wins) in a penalty shootout vs team_b.

        Uses the Bradley-Terry model: P(A) = r_A / (r_A + r_B).

        Args:
            team_a: Canonical name of team A.
            team_b: Canonical name of team B.

        Returns:
            Probability that team_a wins the shootout (0–1).
        """
        rate_a = self._win_rates.get(team_a, self._global_mean)
        rate_b = self._win_rates.get(team_b, self._global_mean)
        total = rate_a + rate_b
        if total == 0:
            return 0.5
        return rate_a / total

    def predict_winner(
        self,
        team_a: str,
        team_b: str,
        rng: np.random.Generator,
    ) -> str:
        """Sample the shootout winner.

        Args:
            team_a: Canonical name of team A.
            team_b: Canonical name of team B.
            rng: NumPy random generator for reproducibility.

        Returns:
            Canonical name of the winning team.
        """
        p_a = self.get_win_probability(team_a, team_b)
        return team_a if rng.random() < p_a else team_b
