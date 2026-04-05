"""Monte Carlo tournament simulator for FIFA WC 2026.

Runs N simulations of the full 48-team, 12-group tournament and
aggregates win probabilities across all teams.

Optimization: pre-compute outcome probabilities and expected goals for all
2256 team pairs in one batch inference call before any simulations run.
Each simulation then consists only of dict lookups + numpy sampling.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.h2h import compute_h2h
from src.features.match_features import (
    FEATURE_COLUMNS,
    _TOURNAMENT_WEIGHTS,
    _DEFAULT_TOURNAMENT_WEIGHT,
    _safe_log,
    _safe_log_ratio,
)
from src.models.goals_model import GOALS_FEATURE_COLUMNS, GoalsModel
from src.models.outcome_model import OutcomeModel
from src.simulation.bracket import (
    GroupStandings,
    MatchResult,
    rank_group,
    select_best_third_place_teams,
)
from src.simulation.shootout import ShootoutModel

logger = logging.getLogger(__name__)


# ── Fast feature pre-computation ──────────────────────────────────────────────

def _precompute_team_form(
    results_long: pd.DataFrame,
    teams: list[str],
    as_of_date: pd.Timestamp,
    window: int,
) -> dict[str, dict[str, float]]:
    """Compute form for all teams in one pass over results_long.

    Builds a per-team view of results_long once, then computes form
    for each team with O(1) tail() instead of O(n) full scan per team.

    Args:
        results_long: Long-format results DataFrame.
        teams: List of canonical team names.
        as_of_date: Reference date (strictly before).
        window: Number of recent matches.

    Returns:
        Dict mapping team → form metrics dict.
    """
    past = results_long[results_long["date"] < as_of_date]
    # Group by team once
    grouped = {
        team: grp.sort_values("date").tail(window)
        for team, grp in past.groupby("team")
    }

    form_cache: dict[str, dict[str, float]] = {}
    for team in teams:
        matches = grouped.get(team, pd.DataFrame())
        n = len(matches)
        if n == 0:
            form_cache[team] = {
                "form_points": 0.0, "win_rate": 0.0, "draw_rate": 0.0,
                "loss_rate": 0.0, "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0, "goal_diff_avg": 0.0, "n_matches": 0,
            }
            continue
        wins = (matches["result"] == "W").sum()
        draws = (matches["result"] == "D").sum()
        losses = (matches["result"] == "L").sum()
        gf = matches["goals_for"].sum()
        ga = matches["goals_against"].sum()
        form_cache[team] = {
            "form_points": float(wins * 3 + draws),
            "win_rate": float(wins / n),
            "draw_rate": float(draws / n),
            "loss_rate": float(losses / n),
            "goals_scored_avg": float(gf / n),
            "goals_conceded_avg": float(ga / n),
            "goal_diff_avg": float((gf - ga) / n),
            "n_matches": int(n),
        }
    return form_cache


def _build_feature_row(
    home: str,
    away: str,
    elo_ratings: pd.Series,
    rank_idx: pd.DataFrame,
    squad_features: pd.DataFrame,
    form_cache: dict[str, dict],
    results_long: pd.DataFrame,
    match_date: pd.Timestamp,
) -> list[float]:
    """Build one feature row for a (home, away) pair.

    Uses pre-computed form_cache to avoid per-pair results_long scans.

    Args:
        home: Canonical home team.
        away: Canonical away team.
        elo_ratings: Current Elo ratings Series.
        rank_idx: Rankings DataFrame indexed by team_name_canonical.
        squad_features: Squad features DataFrame indexed by team_name.
        form_cache: Pre-computed form dict for all teams.
        results_long: For H2H lookup only.
        match_date: Tournament reference date.

    Returns:
        List of float values matching FEATURE_COLUMNS order.
    """
    initial_elo = 1500.0

    # Elo
    h_elo = float(elo_ratings.get(home, initial_elo))
    a_elo = float(elo_ratings.get(away, initial_elo))
    elo_diff = h_elo - a_elo
    elo_ratio = h_elo / a_elo if a_elo > 0 else 1.0

    # Rankings
    h_pts = float(rank_idx.loc[home, "total_points"]) if home in rank_idx.index else 1500.0
    a_pts = float(rank_idx.loc[away, "total_points"]) if away in rank_idx.index else 1500.0
    h_rank = int(rank_idx.loc[home, "rank"]) if home in rank_idx.index else 100
    a_rank = int(rank_idx.loc[away, "rank"]) if away in rank_idx.index else 100
    ranking_diff = float(a_rank - h_rank)

    # Squad
    def _sq(team: str, col: str, default: float = 0.0) -> float:
        if team in squad_features.index:
            v = squad_features.loc[team, col]
            return float(v) if pd.notna(v) else default
        return default

    h_val = _sq(home, "injury_adj_value", 1e8)
    a_val = _sq(away, "injury_adj_value", 1e8)
    log_ratio = _safe_log_ratio(h_val, a_val)
    h_inj = _sq(home, "injury_pct_value_lost")
    a_inj = _sq(away, "injury_pct_value_lost")
    h_age = _sq(home, "avg_age", 26.0)
    a_age = _sq(away, "avg_age", 26.0)
    h_caps = _sq(home, "avg_caps")
    a_caps = _sq(away, "avg_caps")

    # Form (from cache — no results_long scan)
    hf = form_cache.get(home, {})
    af = form_cache.get(away, {})

    # H2H (only lookup, still scans results_long but only 2256 times total)
    h2h = compute_h2h(results_long, home, away, match_date)

    t_weight = _TOURNAMENT_WEIGHTS.get("FIFA World Cup", _DEFAULT_TOURNAMENT_WEIGHT)

    return [
        elo_diff, h_elo, a_elo, elo_ratio,
        ranking_diff, h_pts, a_pts,
        log_ratio, _safe_log(h_val), _safe_log(a_val), h_inj, a_inj,
        hf.get("form_points", 0.0), af.get("form_points", 0.0),
        hf.get("win_rate", 0.0), af.get("win_rate", 0.0),
        hf.get("goals_scored_avg", 0.0), af.get("goals_scored_avg", 0.0),
        hf.get("goals_conceded_avg", 0.0), af.get("goals_conceded_avg", 0.0),
        h2h["h2h_win_rate_a"], h2h["h2h_draw_rate"], h2h["h2h_goal_diff_avg"],
        min(float(h2h["h2h_n_matches"]), 10.0),
        h_age, a_age, h_caps, a_caps,
        1.0,   # is_neutral (all WC matches)
        t_weight,
    ]


def precompute_match_predictions(
    teams_df: pd.DataFrame,
    feature_kwargs: dict,
    outcome_model: OutcomeModel,
    goals_model: GoalsModel,
) -> dict[tuple[str, str], tuple[float, float, float, float, float]]:
    """Pre-compute outcome probabilities and expected goals for all team pairs.

    Strategy:
    1. Pre-compute form for all 48 teams in one grouped pass (fast)
    2. Build 2256-row feature matrix
    3. Single batch XGBoost predict_proba call
    4. Single batch Poisson predict call (home) + (away)

    Args:
        teams_df: 48 qualified teams.
        feature_kwargs: Dict with elo_ratings, rankings, squad_features,
            results_long, form_window.
        outcome_model: Fitted OutcomeModel.
        goals_model: Fitted GoalsModel.

    Returns:
        Dict mapping (home, away) →
        (p_home_win, p_draw, p_away_win, lambda_home, lambda_away).
    """
    elo_ratings = feature_kwargs["elo_ratings"]
    rankings = feature_kwargs["rankings"]
    squad_features = feature_kwargs["squad_features"]
    results_long = feature_kwargs["results_long"]
    form_window = feature_kwargs.get("form_window", 10)
    match_date = pd.Timestamp("2026-06-15")

    teams = [
        row["team_name"] for _, row in teams_df.iterrows()
        if not row.get("is_placeholder", False)
    ]

    # Step 1: pre-compute form for all teams at once
    logger.info("Pre-computing form for %d teams...", len(teams))
    form_cache = _precompute_team_form(results_long, teams, match_date, form_window)

    # Rankings index
    rank_col = "team_name_canonical" if "team_name_canonical" in rankings.columns else "country"
    rank_idx = rankings.set_index(rank_col)

    # Step 2: build feature matrix for all (home, away) pairs
    logger.info("Building feature matrix for all team pairs...")
    pairs: list[tuple[str, str]] = []
    rows: list[list[float]] = []

    for home in teams:
        for away in teams:
            if home == away:
                continue
            pairs.append((home, away))
            rows.append(_build_feature_row(
                home, away, elo_ratings, rank_idx, squad_features,
                form_cache, results_long, match_date,
            ))

    feature_matrix = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    # Replace any NaN/inf
    feature_matrix = feature_matrix.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

    # Step 3: batch model inference
    logger.info("Running batch model inference on %d pairs...", len(pairs))
    outcome_proba = outcome_model.predict_proba(feature_matrix)  # shape (n, 3)
    # Normalize rows to sum to 1.0
    outcome_proba = np.clip(outcome_proba, 0.0, 1.0)
    outcome_proba = outcome_proba / outcome_proba.sum(axis=1, keepdims=True)

    lambda_home, lambda_away = goals_model.predict_goals_batch(feature_matrix)
    lambda_home = np.clip(lambda_home, 0.1, 10.0)
    lambda_away = np.clip(lambda_away, 0.1, 10.0)

    # Step 4: build predictions dict
    predictions: dict[tuple[str, str], tuple[float, float, float, float, float]] = {}
    for i, pair in enumerate(pairs):
        p = outcome_proba[i]
        predictions[pair] = (
            float(p[2]),  # p_home_win
            float(p[1]),  # p_draw
            float(p[0]),  # p_away_win
            float(lambda_home[i]),
            float(lambda_away[i]),
        )

    logger.info("Pre-computation complete: %d pairs cached", len(predictions))
    return predictions


# ── Fast simulation functions ─────────────────────────────────────────────────

def _fast_match(
    home: str,
    away: str,
    predictions: dict[tuple[str, str], tuple],
    rng: np.random.Generator,
    knockout: bool = False,
    shootout_model: ShootoutModel | None = None,
) -> MatchResult:
    """Simulate one match using pre-computed probabilities.

    Args:
        home: Home team name.
        away: Away team name.
        predictions: Pre-computed (p_home, p_draw, p_away, λ_home, λ_away).
        rng: Random generator.
        knockout: If True, must produce a winner (no draws allowed).
        shootout_model: Required when knockout=True.

    Returns:
        MatchResult with score and optional shootout winner.
    """
    key = (home, away)
    if key in predictions:
        p_home, p_draw, p_away, lam_h, lam_a = predictions[key]
    else:
        # Fallback: equal probabilities
        p_home, p_draw, p_away = 1 / 3, 1 / 3, 1 / 3
        lam_h, lam_a = 1.3, 1.1

    p_arr = np.array([p_home, p_draw, p_away], dtype=np.float64)
    p_arr /= p_arr.sum()

    outcome_idx = rng.choice(3, p=p_arr)  # 0=home_win, 1=draw, 2=away_win

    # Sample Poisson goal counts, enforce consistency with sampled outcome
    for _ in range(5):
        h = int(rng.poisson(lam_h))
        a = int(rng.poisson(lam_a))
        if outcome_idx == 0 and h > a:
            break
        if outcome_idx == 1 and h == a:
            break
        if outcome_idx == 2 and a > h:
            break
    else:
        if outcome_idx == 0:
            h, a = max(1, h), max(0, h - 1)
        elif outcome_idx == 2:
            a, h = max(1, a), max(0, a - 1)
        else:
            h = a = max(h, a)

    # Knockout: handle draw via penalty shootout
    shootout_winner = None
    if knockout and h == a:
        assert shootout_model is not None
        shootout_winner = shootout_model.predict_winner(home, away, rng)

    return MatchResult(
        home_team=home, away_team=away,
        home_score=h, away_score=a,
        went_to_shootout=(knockout and h == a),
        shootout_winner=shootout_winner,
    )


def _knockout_winner(result: MatchResult) -> str:
    """Return the winner of a knockout match."""
    if result.shootout_winner:
        return result.shootout_winner
    return result.home_team if result.home_score > result.away_score else result.away_team


# ── Simulation core ───────────────────────────────────────────────────────────

def _run_single_fast(
    teams_df: pd.DataFrame,
    predictions: dict[tuple[str, str], tuple],
    shootout_model: ShootoutModel,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Run one complete tournament simulation using pre-computed predictions.

    Args:
        teams_df: 48 qualified teams with group_letter.
        predictions: Pre-computed match predictions cache.
        shootout_model: For penalty shootout resolution.
        rng: Random generator.

    Returns:
        Dict mapping team → furthest stage reached.
    """
    # Build groups
    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in teams_df.iterrows():
        if not row.get("is_placeholder", False):
            groups[row["group_letter"]].append(row["team_name"])

    # ── Group stage ───────────────────────────────────────────────────────────
    standings: dict[str, GroupStandings] = {
        g: GroupStandings(group=g, teams=ts) for g, ts in groups.items()
    }
    for g, ts in groups.items():
        for i, home in enumerate(ts):
            for away in ts[i + 1:]:
                result = _fast_match(home, away, predictions, rng)
                standings[g].update(result)

    ranked = {g: rank_group(s) for g, s in standings.items()}

    # ── Knockout stage ────────────────────────────────────────────────────────
    team_stage: dict[str, str] = {
        t: "group_exit" for ts in ranked.values() for t in ts
    }

    thirds = [
        (ranked[g][2], standings[g])
        for g in sorted(standings)
        if len(ranked[g]) >= 3
    ]
    best_thirds = select_best_third_place_teams(thirds, n=8)

    advancing: list[str] = []
    for g in sorted(ranked):
        if len(ranked[g]) >= 2:
            advancing.extend([ranked[g][0], ranked[g][1]])
    advancing = advancing[:24] + best_thirds

    def sim_round(teams: list[str], exit_label: str) -> list[str]:
        winners = []
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                winners.append(teams[i])
                continue
            a, b = teams[i], teams[i + 1]
            result = _fast_match(a, b, predictions, rng, knockout=True, shootout_model=shootout_model)
            w = _knockout_winner(result)
            loser = b if w == a else a
            team_stage[loser] = exit_label
            winners.append(w)
        return winners

    r32 = sim_round(advancing, "round_of_32_exit")
    for t in r32:
        team_stage[t] = "round_of_16_exit"

    r16 = sim_round(r32, "round_of_16_exit")
    for t in r16:
        team_stage[t] = "quarterfinal_exit"

    qf = sim_round(r16, "quarterfinal_exit")
    for t in qf:
        team_stage[t] = "semifinal_exit"

    # Semifinals
    sf_losers, finalists = [], []
    for i in range(0, len(qf), 2):
        if i + 1 >= len(qf):
            finalists.append(qf[i])
            continue
        a, b = qf[i], qf[i + 1]
        result = _fast_match(a, b, predictions, rng, knockout=True, shootout_model=shootout_model)
        w = _knockout_winner(result)
        loser = b if w == a else a
        team_stage[loser] = "semifinal_exit"
        sf_losers.append(loser)
        finalists.append(w)

    # Third place playoff
    if len(sf_losers) >= 2:
        result = _fast_match(sf_losers[0], sf_losers[1], predictions, rng, knockout=True, shootout_model=shootout_model)
        w = _knockout_winner(result)
        loser = sf_losers[1] if w == sf_losers[0] else sf_losers[0]
        team_stage[w] = "third_place"
        team_stage[loser] = "fourth_place"

    # Final
    if len(finalists) >= 2:
        result = _fast_match(finalists[0], finalists[1], predictions, rng, knockout=True, shootout_model=shootout_model)
        champion = _knockout_winner(result)
        runner_up = finalists[1] if champion == finalists[0] else finalists[0]
        team_stage[champion] = "winner"
        team_stage[runner_up] = "runner_up"
    elif len(finalists) == 1:
        team_stage[finalists[0]] = "winner"

    return team_stage


def _worker(
    idx: int,
    teams_df: pd.DataFrame,
    predictions: dict,
    shootout_model: ShootoutModel,
    base_seed: int,
) -> dict[str, str]:
    """Parallel worker: one simulation run."""
    rng = np.random.default_rng(base_seed + idx)
    return _run_single_fast(teams_df, predictions, shootout_model, rng)


# ── Public API ────────────────────────────────────────────────────────────────

def run_single_simulation(
    teams_df: pd.DataFrame,
    outcome_model: OutcomeModel,
    goals_model: GoalsModel,
    shootout_model: ShootoutModel,
    feature_kwargs: dict,
    rng: np.random.Generator,
    feature_cache: dict | None = None,
) -> dict[str, str]:
    """Run one complete tournament simulation.

    Accepts feature_cache for API compatibility with tests, but uses the
    fast prediction-based path when possible.

    Args:
        teams_df: 48-team DataFrame.
        outcome_model: Fitted OutcomeModel.
        goals_model: Fitted GoalsModel.
        shootout_model: ShootoutModel.
        feature_kwargs: Feature builder kwargs.
        rng: Random generator.
        feature_cache: Unused (kept for test compatibility).

    Returns:
        Dict mapping team → furthest stage reached.
    """
    predictions = precompute_match_predictions(
        teams_df, feature_kwargs, outcome_model, goals_model
    )
    return _run_single_fast(teams_df, predictions, shootout_model, rng)


def run_monte_carlo(
    n_simulations: int,
    teams_df: pd.DataFrame,
    outcome_model: OutcomeModel,
    goals_model: GoalsModel,
    shootout_model: ShootoutModel,
    feature_kwargs: dict,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run N Monte Carlo tournament simulations.

    Pre-computes all match predictions once, then runs simulations in
    parallel with each worker doing only dict lookups + numpy sampling.

    Args:
        n_simulations: Number of simulations to run.
        teams_df: 48-team DataFrame.
        outcome_model: Fitted OutcomeModel.
        goals_model: Fitted GoalsModel.
        shootout_model: ShootoutModel.
        feature_kwargs: Feature builder kwargs.
        seed: Base random seed for reproducibility.
        n_jobs: Parallel workers (default 1 — Windows multiprocessing
            overhead makes single-threaded faster for <10k runs).

    Returns:
        DataFrame sorted by win_prob descending with stage probability columns.
    """
    logger.info("Running %d Monte Carlo simulations...", n_simulations)

    # Pre-compute all match predictions once (the expensive step)
    predictions = precompute_match_predictions(
        teams_df, feature_kwargs, outcome_model, goals_model
    )

    # Run simulations — single-threaded to avoid Windows joblib overhead
    if n_jobs == 1:
        results = [
            _worker(i, teams_df, predictions, shootout_model, seed)
            for i in range(n_simulations)
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(i, teams_df, predictions, shootout_model, seed)
            for i in range(n_simulations)
        )

    # Aggregate
    stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sim_result in results:
        for team, stage in sim_result.items():
            stage_counts[team][stage] += 1

    records = []
    for team in teams_df["team_name"]:
        counts = stage_counts.get(team, {})
        records.append({
            "team_name": team,
            "win_prob": counts.get("winner", 0) / n_simulations,
            "runner_up_prob": counts.get("runner_up", 0) / n_simulations,
            "third_place_prob": counts.get("third_place", 0) / n_simulations,
            "fourth_place_prob": counts.get("fourth_place", 0) / n_simulations,
            "semifinal_exit_prob": counts.get("semifinal_exit", 0) / n_simulations,
            "quarterfinal_exit_prob": counts.get("quarterfinal_exit", 0) / n_simulations,
            "round_of_16_exit_prob": counts.get("round_of_16_exit", 0) / n_simulations,
            "round_of_32_exit_prob": counts.get("round_of_32_exit", 0) / n_simulations,
            "group_exit_prob": counts.get("group_exit", 0) / n_simulations,
        })

    df = pd.DataFrame(records).sort_values("win_prob", ascending=False).reset_index(drop=True)
    logger.info(
        "Monte Carlo complete. Top: %s (%.1f%%)",
        df.iloc[0]["team_name"], df.iloc[0]["win_prob"] * 100,
    )
    return df
