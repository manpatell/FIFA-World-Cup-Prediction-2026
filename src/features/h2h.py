"""Head-to-head record computation between two national teams."""

from __future__ import annotations

import pandas as pd


_NEUTRAL_H2H = {
    "h2h_win_rate_a": 0.5,
    "h2h_draw_rate": 0.0,
    "h2h_win_rate_b": 0.5,
    "h2h_goal_diff_avg": 0.0,
    "h2h_n_matches": 0,
}


def compute_h2h(
    results_long: pd.DataFrame,
    team_a: str,
    team_b: str,
    as_of_date: pd.Timestamp,
    n_matches: int = 10,
) -> dict[str, float]:
    """Compute head-to-head record between team_a and team_b.

    Args:
        results_long: Long-format results from build_results_long().
        team_a: First team (treated as the 'home' perspective).
        team_b: Second team.
        as_of_date: Only use matches strictly before this date.
        n_matches: Maximum number of recent H2H matches to use.

    Returns:
        Dict with keys: h2h_win_rate_a, h2h_draw_rate, h2h_win_rate_b,
        h2h_goal_diff_avg, h2h_n_matches.
        Returns neutral values (0.5, 0, 0.5) if no H2H history.
    """
    mask = (
        (results_long["team"] == team_a) &
        (results_long["opponent"] == team_b) &
        (results_long["date"] < as_of_date)
    )
    h2h = results_long[mask].sort_values("date").tail(n_matches)

    n = len(h2h)
    if n == 0:
        return dict(_NEUTRAL_H2H)

    wins_a = (h2h["result"] == "W").sum()
    draws = (h2h["result"] == "D").sum()
    wins_b = (h2h["result"] == "L").sum()
    goal_diff = (h2h["goals_for"] - h2h["goals_against"]).mean()

    return {
        "h2h_win_rate_a": float(wins_a / n),
        "h2h_draw_rate": float(draws / n),
        "h2h_win_rate_b": float(wins_b / n),
        "h2h_goal_diff_avg": float(goal_diff),
        "h2h_n_matches": int(n),
    }


def build_h2h_lookup(
    results_long: pd.DataFrame,
    match_pairs: list[tuple[str, str, pd.Timestamp]],
    n_matches: int = 10,
) -> dict[tuple[str, str, pd.Timestamp], dict[str, float]]:
    """Pre-compute H2H for all (team_a, team_b, date) triples.

    Args:
        results_long: Long-format results from build_results_long().
        match_pairs: List of (team_a, team_b, as_of_date) tuples.
        n_matches: Maximum recent H2H matches to use.

    Returns:
        Dict mapping (team_a, team_b, date) → H2H metrics dict.
    """
    lookup: dict[tuple[str, str, pd.Timestamp], dict[str, float]] = {}
    for team_a, team_b, date in match_pairs:
        key = (team_a, team_b, date)
        lookup[key] = compute_h2h(results_long, team_a, team_b, date, n_matches)
    return lookup
