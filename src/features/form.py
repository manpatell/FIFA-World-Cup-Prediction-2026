"""Recent form feature computation for national teams.

Reshapes match results to long format and computes rolling form
metrics (win rate, goals, form points) for any team/date combination.
"""

from __future__ import annotations

import pandas as pd


def compute_match_result(home_score: int, away_score: int) -> tuple[str, str]:
    """Determine match result for home and away teams.

    Args:
        home_score: Goals scored by home team.
        away_score: Goals scored by away team.

    Returns:
        Tuple of (home_result, away_result) where each is 'W', 'D', or 'L'.
    """
    if home_score > away_score:
        return "W", "L"
    if home_score < away_score:
        return "L", "W"
    return "D", "D"


def build_results_long(results: pd.DataFrame) -> pd.DataFrame:
    """Reshape results from wide to long format (one row per team per match).

    Args:
        results: Normalized results DataFrame with canonical team name columns.

    Returns:
        DataFrame with columns: date, team, opponent, goals_for,
        goals_against, result, tournament, neutral, is_home.
    """
    home_df = results[
        ["date", "home_team_canonical", "away_team_canonical",
         "home_score", "away_score", "tournament", "neutral"]
    ].copy()
    home_df = home_df.rename(columns={
        "home_team_canonical": "team",
        "away_team_canonical": "opponent",
        "home_score": "goals_for",
        "away_score": "goals_against",
    })
    home_df["is_home"] = True

    away_df = results[
        ["date", "away_team_canonical", "home_team_canonical",
         "away_score", "home_score", "tournament", "neutral"]
    ].copy()
    away_df = away_df.rename(columns={
        "away_team_canonical": "team",
        "home_team_canonical": "opponent",
        "away_score": "goals_for",
        "home_score": "goals_against",
    })
    away_df["is_home"] = False

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values("date").reset_index(drop=True)

    results_col = long_df.apply(
        lambda r: compute_match_result(
            int(r["goals_for"]), int(r["goals_against"])
        )[0],
        axis=1,
    )
    long_df["result"] = results_col

    # Convert scores to float for safe arithmetic
    long_df["goals_for"] = pd.to_numeric(long_df["goals_for"], errors="coerce")
    long_df["goals_against"] = pd.to_numeric(long_df["goals_against"], errors="coerce")

    return long_df


def compute_form(
    results_long: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    window: int = 10,
) -> dict[str, float]:
    """Compute recent form metrics for a team before a given date.

    Args:
        results_long: Long-format results from build_results_long().
        team: Canonical team name.
        as_of_date: Compute form using only matches strictly before this date.
        window: Number of most recent matches to include.

    Returns:
        Dict with keys: form_points, win_rate, draw_rate, loss_rate,
        goals_scored_avg, goals_conceded_avg, goal_diff_avg, n_matches.
    """
    team_matches = results_long[
        (results_long["team"] == team) &
        (results_long["date"] < as_of_date)
    ].sort_values("date").tail(window)

    n = len(team_matches)
    if n == 0:
        return {
            "form_points": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "goals_scored_avg": 0.0,
            "goals_conceded_avg": 0.0,
            "goal_diff_avg": 0.0,
            "n_matches": 0,
        }

    wins = (team_matches["result"] == "W").sum()
    draws = (team_matches["result"] == "D").sum()
    losses = (team_matches["result"] == "L").sum()
    goals_scored = team_matches["goals_for"].sum()
    goals_conceded = team_matches["goals_against"].sum()

    return {
        "form_points": float(wins * 3 + draws),
        "win_rate": float(wins / n),
        "draw_rate": float(draws / n),
        "loss_rate": float(losses / n),
        "goals_scored_avg": float(goals_scored / n),
        "goals_conceded_avg": float(goals_conceded / n),
        "goal_diff_avg": float((goals_scored - goals_conceded) / n),
        "n_matches": int(n),
    }


def build_form_lookup(
    results_long: pd.DataFrame,
    teams: list[str],
    dates: list[pd.Timestamp],
    window: int = 10,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Pre-compute form snapshots for all (team, date) pairs.

    Avoids O(n²) recomputation during training set construction by
    building all needed (team, date) pairs in a single pass.

    Args:
        results_long: Long-format results from build_results_long().
        teams: List of team names to compute form for.
        dates: List of dates at which to compute form snapshots.
        window: Number of recent matches to use.

    Returns:
        Dict mapping (team, date) → form metrics dict.
    """
    lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for team in teams:
        for date in dates:
            key = (team, date)
            lookup[key] = compute_form(results_long, team, date, window)
    return lookup
