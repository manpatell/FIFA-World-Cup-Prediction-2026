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
