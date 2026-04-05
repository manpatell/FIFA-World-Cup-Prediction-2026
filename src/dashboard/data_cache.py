"""Cached data loaders for the Streamlit dashboard.

All heavy I/O is wrapped in @st.cache_data to prevent reloading on
every interaction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import Config, get_processed_path, get_raw_path
from src.ingestion.loader import load_rankings, load_teams
from src.ingestion.normalizer import (
    build_name_resolver,
    normalize_rankings,
)


@st.cache_data
def load_simulation_results(_cfg: Config) -> pd.DataFrame:
    """Load Monte Carlo simulation results from processed parquet.

    Args:
        _cfg: Config object (underscore prefix skips st.cache_data hashing).

    Returns:
        DataFrame with win_prob, stage probabilities per team.
    """
    path = get_processed_path(_cfg, "simulation_results")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_squad_features(_cfg: Config) -> pd.DataFrame:
    """Load squad features from processed parquet.

    Args:
        _cfg: Config object.

    Returns:
        DataFrame indexed by team_name with squad feature columns.
    """
    path = get_processed_path(_cfg, "squad_features")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_elo_ratings(_cfg: Config) -> pd.Series:
    """Load current Elo ratings from processed parquet.

    Args:
        _cfg: Config object.

    Returns:
        Series indexed by canonical team name with Elo values.
    """
    path = get_processed_path(_cfg, "elo_ratings")
    if not path.exists():
        return pd.Series(dtype=float)
    history = pd.read_parquet(path)
    from src.features.elo import get_current_elo
    return get_current_elo(history)


@st.cache_data
def load_current_rankings(_cfg: Config) -> pd.DataFrame:
    """Load normalized FIFA rankings.

    Args:
        _cfg: Config object.

    Returns:
        Rankings DataFrame with team_name_canonical column.
    """
    from src.ingestion.loader import load_former_names, load_name_mapping
    teams = load_teams(get_raw_path(_cfg, "teams"))
    nm = load_name_mapping(get_raw_path(_cfg, "name_mapping"))
    fn = load_former_names(get_raw_path(_cfg, "former_names"))
    resolver = build_name_resolver(nm, fn, teams)
    rankings = load_rankings(get_raw_path(_cfg, "rankings"))
    return normalize_rankings(rankings, resolver)


@st.cache_data
def load_teams_df(_cfg: Config) -> pd.DataFrame:
    """Load the 48 qualified teams.

    Args:
        _cfg: Config object.

    Returns:
        Teams DataFrame with team_name, group_letter columns.
    """
    return load_teams(get_raw_path(_cfg, "teams"))


@st.cache_data(show_spinner="Running ML predictions for all matchups...")
def load_match_predictions(_cfg: Config) -> dict:
    """Load trained models and pre-compute predictions for all team pairs.

    Uses the same XGBoost + Poisson pipeline as the Monte Carlo simulation.
    Results are cached so the bracket tab loads instantly after first visit.

    Args:
        _cfg: Config object.

    Returns:
        Dict mapping (home_team, away_team) →
        (p_home_win, p_draw, p_away_win, lambda_home, lambda_away).
        Returns empty dict if models or processed data are not available.
    """
    from src.config import get_model_path
    from src.models.outcome_model import OutcomeModel
    from src.models.goals_model import GoalsModel
    from src.simulation.monte_carlo import precompute_match_predictions
    from src.features.form import build_results_long
    from src.features.elo import get_current_elo
    from src.ingestion.loader import load_former_names, load_name_mapping

    outcome_path = get_model_path(_cfg, "outcome_model")
    home_path    = get_model_path(_cfg, "goals_model_home")
    away_path    = get_model_path(_cfg, "goals_model_away")
    results_path = get_processed_path(_cfg, "results_clean")
    elo_path     = get_processed_path(_cfg, "elo_ratings")
    squad_path   = get_processed_path(_cfg, "squad_features")

    for p in (outcome_path, home_path, away_path, results_path, elo_path, squad_path):
        if not p.exists():
            return {}

    outcome_model = OutcomeModel.load(outcome_path)
    goals_model   = GoalsModel.load(_cfg.model, home_path, away_path)

    results_long   = build_results_long(pd.read_parquet(results_path))
    current_elo    = get_current_elo(pd.read_parquet(elo_path))
    squad_features = pd.read_parquet(squad_path)

    teams    = load_teams(get_raw_path(_cfg, "teams"))
    nm       = load_name_mapping(get_raw_path(_cfg, "name_mapping"))
    fn       = load_former_names(get_raw_path(_cfg, "former_names"))
    resolver = build_name_resolver(nm, fn, teams)
    rankings = normalize_rankings(load_rankings(get_raw_path(_cfg, "rankings")), resolver)

    feature_kwargs = {
        "elo_ratings":   current_elo,
        "rankings":      rankings,
        "squad_features": squad_features,
        "results_long":  results_long,
        "form_window":   _cfg.features.form_window,
    }

    return precompute_match_predictions(teams, feature_kwargs, outcome_model, goals_model)


def clear_all_caches() -> None:
    """Clear all st.cache_data caches.

    Call this after pipeline reset or after a simulation completes
    so display tabs pick up freshly written parquet files.
    """
    st.cache_data.clear()


def load_all_dashboard_data(cfg: Config) -> dict:
    """Load all data needed by the dashboard in one call.

    Args:
        cfg: Config object.

    Returns:
        Dict with keys: sim_results, squad_features, elo_ratings,
        rankings, teams.
    """
    return {
        "sim_results": load_simulation_results(cfg),
        "squad_features": load_squad_features(cfg),
        "elo_ratings": load_elo_ratings(cfg),
        "rankings": load_current_rankings(cfg),
        "teams": load_teams_df(cfg),
    }
