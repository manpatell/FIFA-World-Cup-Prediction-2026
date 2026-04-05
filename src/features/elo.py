"""Rolling Elo rating engine for international football.

Processes all historical results in chronological order, maintaining
a running dict of team ratings. Applies margin-of-victory scaling
and home advantage correction.
"""

from __future__ import annotations

import math

import pandas as pd

from src.config import EloConfig


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for team A against team B.

    Args:
        rating_a: Elo rating of team A.
        rating_b: Elo rating of team B.

    Returns:
        Probability of team A winning (0–1).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def get_k_factor(
    tournament: str,
    goal_diff: int,
    rating_diff: float,
    k_factors: dict[str, float],
    margin_mult: bool,
) -> float:
    """Compute K-factor for a match, optionally scaled by margin of victory.

    Uses the 538-style margin-of-victory multiplier to prevent Elo
    inflation from blowout results.

    Args:
        tournament: Tournament name string.
        goal_diff: Absolute goal difference for the match.
        rating_diff: Elo rating difference (winner - loser) before the match.
        k_factors: Dict mapping tournament name → base K-factor.
        margin_mult: Whether to apply margin-of-victory scaling.

    Returns:
        Adjusted K-factor for this match.
    """
    base_k = k_factors.get(tournament, k_factors.get("default", 30.0))
    if not margin_mult or goal_diff <= 0:
        return base_k
    # 538-style multiplier: accounts for goal difference and rating difference
    mov_mult = math.log(abs(goal_diff) + 1) * (2.2 / (abs(rating_diff) * 0.001 + 2.2))
    return base_k * mov_mult


def update_elo(
    rating_home: float,
    rating_away: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
    home_advantage: float,
    k_factors: dict[str, float],
    margin_mult: bool,
) -> tuple[float, float]:
    """Compute updated Elo ratings after a single match.

    Args:
        rating_home: Current Elo of the home team.
        rating_away: Current Elo of the away team.
        home_score: Goals scored by home team.
        away_score: Goals scored by away team.
        tournament: Tournament name (for K-factor lookup).
        neutral: True if played at a neutral venue.
        home_advantage: Elo points added to home team in non-neutral matches.
        k_factors: Dict of tournament → base K-factor.
        margin_mult: Whether to apply margin-of-victory scaling.

    Returns:
        Tuple of (new_home_elo, new_away_elo).
    """
    # Apply home advantage for non-neutral venues
    adj_home = rating_home + (0.0 if neutral else home_advantage)

    expected_home = compute_expected_score(adj_home, rating_away)
    expected_away = 1.0 - expected_home

    # Actual scores: 1=win, 0.5=draw, 0=loss
    if home_score > away_score:
        actual_home, actual_away = 1.0, 0.0
    elif home_score < away_score:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5

    goal_diff = abs(home_score - away_score)
    # Rating diff from winner's perspective for multiplier
    if home_score > away_score:
        winner_rating_diff = adj_home - rating_away
    elif away_score > home_score:
        winner_rating_diff = rating_away - adj_home
    else:
        winner_rating_diff = 0.0

    k = get_k_factor(tournament, goal_diff, winner_rating_diff, k_factors, margin_mult)

    new_home = rating_home + k * (actual_home - expected_home)
    new_away = rating_away + k * (actual_away - expected_away)
    return new_home, new_away


def build_elo_history(
    results: pd.DataFrame,
    elo_cfg: EloConfig,
) -> pd.DataFrame:
    """Build full Elo history by iterating results chronologically.

    Processes all 49k+ rows sequentially using a plain Python dict for
    running state (faster than per-row DataFrame access).

    Args:
        results: Normalized results DataFrame with columns:
            date, home_team_canonical, away_team_canonical, home_score,
            away_score, tournament, neutral.
        elo_cfg: Elo configuration from config.yaml.

    Returns:
        DataFrame with columns: date, home_team, away_team,
        home_elo_before, away_elo_before, home_elo_after, away_elo_after,
        tournament, neutral.
    """
    ratings: dict[str, float] = {}
    records: list[dict] = []

    # Sort chronologically
    df = results.sort_values("date").reset_index(drop=True)

    for row in df.itertuples(index=False):
        home = row.home_team_canonical
        away = row.away_team_canonical

        # Skip rows with missing team or score data
        if not isinstance(home, str) or not isinstance(away, str):
            continue
        try:
            h_score = int(row.home_score)
            a_score = int(row.away_score)
        except (TypeError, ValueError):
            continue

        h_elo = ratings.get(home, elo_cfg.initial_rating)
        a_elo = ratings.get(away, elo_cfg.initial_rating)

        new_h, new_a = update_elo(
            rating_home=h_elo,
            rating_away=a_elo,
            home_score=h_score,
            away_score=a_score,
            tournament=row.tournament,
            neutral=bool(row.neutral),
            home_advantage=elo_cfg.home_advantage,
            k_factors=elo_cfg.k_factors,
            margin_mult=elo_cfg.margin_of_victory_mult,
        )

        records.append({
            "date": row.date,
            "home_team": home,
            "away_team": away,
            "home_elo_before": h_elo,
            "away_elo_before": a_elo,
            "home_elo_after": new_h,
            "away_elo_after": new_a,
            "tournament": row.tournament,
            "neutral": bool(row.neutral),
        })

        ratings[home] = new_h
        ratings[away] = new_a

    return pd.DataFrame(records)


def get_current_elo(elo_history: pd.DataFrame) -> pd.Series:
    """Extract the most recent Elo rating per team.

    Args:
        elo_history: DataFrame from build_elo_history().

    Returns:
        Series indexed by canonical team name, values = current Elo rating.
    """
    # Get the last recorded Elo for each team from both home and away columns
    home_last = (
        elo_history[["date", "home_team", "home_elo_after"]]
        .rename(columns={"home_team": "team", "home_elo_after": "elo"})
    )
    away_last = (
        elo_history[["date", "away_team", "away_elo_after"]]
        .rename(columns={"away_team": "team", "away_elo_after": "elo"})
    )
    combined = pd.concat([home_last, away_last], ignore_index=True)
    # Keep the most recent entry per team
    latest = combined.sort_values("date").groupby("team")["elo"].last()
    return latest
