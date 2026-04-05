"""Match-level feature vector assembly for model training and inference.

Combines Elo ratings, FIFA rankings, squad features, form, and H2H
into a single feature vector per match.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import Config
from src.features.form import build_results_long, compute_form
from src.features.h2h import compute_h2h

# Tournament weight mapping for the tournament_weight feature
_TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 3.0,
    "FIFA World Cup qualification": 1.5,
    "UEFA Euro": 2.5,
    "UEFA Euro qualification": 1.5,
    "Copa América": 2.5,
    "Copa America": 2.5,
    "African Cup of Nations": 2.0,
    "African Cup of Nations qualification": 1.5,
    "UEFA Nations League": 2.0,
    "CONCACAF Nations League": 2.0,
    "AFC Asian Cup": 2.0,
    "Gold Cup": 1.8,
    "FIFA Confederations Cup": 2.0,
    "Friendly": 1.0,
    "FIFA Series": 1.2,
}
_DEFAULT_TOURNAMENT_WEIGHT = 1.5

FEATURE_COLUMNS = [
    # Elo
    "elo_diff",
    "home_elo",
    "away_elo",
    "elo_ratio",
    # FIFA rankings (rank: lower is better → negate for model)
    "ranking_diff",
    "home_ranking_points",
    "away_ranking_points",
    # Squad value (log-transformed)
    "log_squad_value_ratio",
    "home_injury_adj_value_log",
    "away_injury_adj_value_log",
    "home_injury_pct_lost",
    "away_injury_pct_lost",
    # Form
    "home_form_points",
    "away_form_points",
    "home_win_rate",
    "away_win_rate",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    # H2H
    "h2h_win_rate_home",
    "h2h_draw_rate",
    "h2h_goal_diff_avg",
    "h2h_n_matches_clipped",
    # Squad demographics
    "home_avg_age",
    "away_avg_age",
    "home_avg_caps",
    "away_avg_caps",
    # Match context
    "is_neutral",
    "tournament_weight",
]


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    """Compute log(numerator/denominator) safely, returning 0 if either is 0."""
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return math.log(numerator / denominator)


def _safe_log(value: float) -> float:
    """Compute log(value) safely, returning 0 for non-positive values."""
    if value <= 0:
        return 0.0
    return math.log(value)


def build_match_feature_vector(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    is_neutral: bool,
    tournament: str,
    elo_ratings: pd.Series,
    rankings: pd.DataFrame,
    squad_features: pd.DataFrame,
    results_long: pd.DataFrame,
    form_window: int = 10,
) -> pd.Series:
    """Assemble the full feature vector for a single match.

    Args:
        home_team: Canonical home team name.
        away_team: Canonical away team name.
        match_date: Date of the match.
        is_neutral: True if played at a neutral venue.
        tournament: Tournament name string.
        elo_ratings: Series of current Elo ratings indexed by team name.
        rankings: Normalized rankings DataFrame with team_name_canonical column.
        squad_features: DataFrame from build_squad_features_all_teams().
        results_long: Long-format results for form/H2H lookups.
        form_window: Number of recent matches for form computation.

    Returns:
        pd.Series with index = FEATURE_COLUMNS (length 30).
    """
    initial_elo = 1500.0

    # ── Elo ──────────────────────────────────────────────────────────────────
    home_elo = float(elo_ratings.get(home_team, initial_elo))
    away_elo = float(elo_ratings.get(away_team, initial_elo))
    elo_diff = home_elo - away_elo
    elo_ratio = home_elo / away_elo if away_elo > 0 else 1.0

    # ── Rankings ─────────────────────────────────────────────────────────────
    rank_df = rankings.set_index("team_name_canonical") if "team_name_canonical" in rankings.columns else rankings.set_index("country")
    home_rank_pts = float(rank_df.loc[home_team, "total_points"]) if home_team in rank_df.index else 1500.0
    away_rank_pts = float(rank_df.loc[away_team, "total_points"]) if away_team in rank_df.index else 1500.0
    home_rank = int(rank_df.loc[home_team, "rank"]) if home_team in rank_df.index else 100
    away_rank = int(rank_df.loc[away_team, "rank"]) if away_team in rank_df.index else 100
    # Positive ranking_diff means home team has better rank (lower number)
    ranking_diff = float(away_rank - home_rank)

    # ── Squad features ───────────────────────────────────────────────────────
    def _sq(team: str, col: str, default: float = 0.0) -> float:
        if team in squad_features.index:
            val = squad_features.loc[team, col]
            return float(val) if pd.notna(val) else default
        return default

    home_adj_val = _sq(home_team, "injury_adj_value", 1e8)
    away_adj_val = _sq(away_team, "injury_adj_value", 1e8)
    log_squad_ratio = _safe_log_ratio(home_adj_val, away_adj_val)
    home_inj_pct = _sq(home_team, "injury_pct_value_lost")
    away_inj_pct = _sq(away_team, "injury_pct_value_lost")
    home_avg_age = _sq(home_team, "avg_age", 26.0)
    away_avg_age = _sq(away_team, "avg_age", 26.0)
    home_avg_caps = _sq(home_team, "avg_caps")
    away_avg_caps = _sq(away_team, "avg_caps")

    # ── Form ─────────────────────────────────────────────────────────────────
    home_form = compute_form(results_long, home_team, match_date, form_window)
    away_form = compute_form(results_long, away_team, match_date, form_window)

    # ── H2H ──────────────────────────────────────────────────────────────────
    h2h = compute_h2h(results_long, home_team, away_team, match_date)

    # ── Tournament weight ────────────────────────────────────────────────────
    t_weight = _TOURNAMENT_WEIGHTS.get(tournament, _DEFAULT_TOURNAMENT_WEIGHT)

    values = [
        elo_diff,
        home_elo,
        away_elo,
        elo_ratio,
        ranking_diff,
        home_rank_pts,
        away_rank_pts,
        log_squad_ratio,
        _safe_log(home_adj_val),
        _safe_log(away_adj_val),
        home_inj_pct,
        away_inj_pct,
        home_form["form_points"],
        away_form["form_points"],
        home_form["win_rate"],
        away_form["win_rate"],
        home_form["goals_scored_avg"],
        away_form["goals_scored_avg"],
        home_form["goals_conceded_avg"],
        away_form["goals_conceded_avg"],
        h2h["h2h_win_rate_a"],
        h2h["h2h_draw_rate"],
        h2h["h2h_goal_diff_avg"],
        min(float(h2h["h2h_n_matches"]), 10.0),  # clip to 10
        home_avg_age,
        away_avg_age,
        home_avg_caps,
        away_avg_caps,
        float(is_neutral),
        t_weight,
    ]

    return pd.Series(values, index=FEATURE_COLUMNS)


def build_training_dataset(
    results: pd.DataFrame,
    elo_history: pd.DataFrame,
    rankings: pd.DataFrame,
    squad_features: pd.DataFrame,
    results_long: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Build the full labelled training dataset from historical matches.

    Labels: 0 = away_win, 1 = draw, 2 = home_win.
    Sample weights: exponential recency decay × WC match multiplier.

    Args:
        results: Normalized results with canonical team columns.
        elo_history: DataFrame from build_elo_history().
        rankings: Normalized rankings DataFrame.
        squad_features: DataFrame from build_squad_features_all_teams().
        results_long: Long-format results from build_results_long().
        cfg: Config object.

    Returns:
        DataFrame with FEATURE_COLUMNS + label, sample_weight,
        home_score, away_score, date, home_team, away_team.
    """
    cutoff = pd.Timestamp(cfg.model.training_data_cutoff)
    form_window = cfg.features.form_window
    wc_weight = cfg.features.wc_match_weight
    half_life = cfg.features.recent_years_weight  # years

    # Reference date for recency weighting
    ref_date = results["date"].max()

    # Build pre-match Elo lookup: (date, home_team, away_team) → elo values
    elo_lookup: dict[tuple, tuple[float, float]] = {}
    for row in elo_history.itertuples(index=False):
        key = (row.home_team, row.away_team, row.date)
        elo_lookup[key] = (row.home_elo_before, row.away_elo_before)

    # Filter to training range
    train_results = results[
        (results["date"] >= cutoff) &
        results["home_score"].notna() &
        results["away_score"].notna()
    ].copy()

    feature_rows: list[dict] = []

    for row in train_results.itertuples(index=False):
        home = row.home_team_canonical
        away = row.away_team_canonical

        if not isinstance(home, str) or not isinstance(away, str):
            continue

        # Get pre-match Elo from history (fallback to current)
        elo_key = (home, away, row.date)
        if elo_key in elo_lookup:
            h_elo, a_elo = elo_lookup[elo_key]
        else:
            h_elo = 1500.0
            a_elo = 1500.0

        # Patch current elo ratings with pre-match values for this row
        match_elo = pd.Series({home: h_elo, away: a_elo})
        # Fill missing teams from full elo ratings
        for team in [home, away]:
            if team not in match_elo.index:
                match_elo[team] = 1500.0

        try:
            fv = build_match_feature_vector(
                home_team=home,
                away_team=away,
                match_date=row.date,
                is_neutral=bool(row.neutral),
                tournament=row.tournament,
                elo_ratings=match_elo,
                rankings=rankings,
                squad_features=squad_features,
                results_long=results_long,
                form_window=form_window,
            )
        except Exception:
            continue

        # Label
        h_score, a_score = int(row.home_score), int(row.away_score)
        if h_score > a_score:
            label = 2
        elif h_score < a_score:
            label = 0
        else:
            label = 1

        # Sample weight: exponential decay + WC multiplier
        years_ago = (ref_date - row.date).days / 365.25
        decay = math.exp(-math.log(2) / half_life * years_ago)
        is_wc = "FIFA World Cup" in row.tournament and "qualification" not in row.tournament.lower()
        weight = decay * (wc_weight if is_wc else 1.0)

        record = fv.to_dict()
        record["label"] = label
        record["sample_weight"] = weight
        record["home_score"] = h_score
        record["away_score"] = a_score
        record["date"] = row.date
        record["home_team"] = home
        record["away_team"] = away
        record["tournament"] = row.tournament
        feature_rows.append(record)

    df = pd.DataFrame(feature_rows)
    return df.reset_index(drop=True)
