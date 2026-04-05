"""Shared pytest fixtures for FIFA WC 2026 prediction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Load the real project config (session-scoped for speed)."""
    root = Path(__file__).parent.parent
    return load_config(root / "config.yaml")


@pytest.fixture
def mini_results() -> pd.DataFrame:
    """20-row results DataFrame with known outcomes for deterministic testing."""
    rows = [
        ("2018-01-01", "France", "Germany", 2, 1, "FIFA World Cup", "Paris", "France", False),
        ("2018-01-08", "Spain", "Brazil", 1, 1, "Friendly", "Madrid", "Spain", False),
        ("2018-01-15", "Argentina", "England", 3, 0, "FIFA World Cup", "Moscow", "Russia", True),
        ("2018-01-22", "Germany", "France", 0, 2, "UEFA Euro", "Berlin", "Germany", False),
        ("2018-02-01", "Brazil", "Argentina", 2, 2, "Copa América", "Sao Paulo", "Brazil", False),
        ("2019-01-01", "France", "Spain", 1, 0, "FIFA World Cup qualification", "Paris", "France", False),
        ("2019-01-08", "England", "Germany", 2, 1, "Friendly", "London", "England", False),
        ("2019-01-15", "Brazil", "France", 0, 1, "Friendly", "Rio", "Brazil", False),
        ("2019-01-22", "Argentina", "Spain", 1, 2, "Copa América", "Buenos Aires", "Argentina", False),
        ("2019-02-01", "Germany", "England", 3, 2, "UEFA Nations League", "Berlin", "Germany", False),
        ("2020-01-01", "France", "Argentina", 2, 0, "Friendly", "Paris", "France", False),
        ("2020-01-08", "Spain", "England", 1, 1, "UEFA Euro qualification", "Madrid", "Spain", False),
        ("2020-01-15", "Brazil", "Germany", 2, 1, "Friendly", "Rio", "Brazil", False),
        ("2020-01-22", "Argentina", "France", 0, 2, "FIFA World Cup", "Neutral", "Neutral", True),
        ("2020-02-01", "England", "Spain", 2, 0, "Friendly", "London", "England", False),
        ("2021-01-01", "France", "Brazil", 1, 1, "FIFA World Cup", "Neutral", "Neutral", True),
        ("2021-01-08", "Germany", "Argentina", 1, 2, "Friendly", "Berlin", "Germany", False),
        ("2021-01-15", "Spain", "France", 0, 1, "UEFA Euro", "Madrid", "Spain", False),
        ("2022-01-01", "Brazil", "England", 3, 1, "Friendly", "Rio", "Brazil", False),
        ("2022-01-08", "Argentina", "Germany", 2, 0, "FIFA World Cup", "Neutral", "Neutral", True),
    ]
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score",
                 "away_score", "tournament", "city", "country", "neutral"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    return df


@pytest.fixture
def mini_teams() -> pd.DataFrame:
    """4-team mini teams DataFrame for bracket testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "team_name": ["France", "Germany", "Brazil", "Argentina"],
        "fifa_code": ["FRA", "GER", "BRA", "ARG"],
        "group_letter": ["A", "A", "A", "A"],
        "is_placeholder": [False, False, False, False],
    })


@pytest.fixture
def mini_name_mapping() -> pd.DataFrame:
    """Minimal name_mapping for testing."""
    return pd.DataFrame({
        "teams_csv_name": ["USA", "IR Iran"],
        "standardized_name": ["United States", "Iran"],
        "fifa_code": ["USA", "IRN"],
        "notes": ["", ""],
    })


@pytest.fixture
def mini_former_names() -> pd.DataFrame:
    """Minimal former_names for testing."""
    return pd.DataFrame({
        "current": ["Germany", "Russia"],
        "former": ["West Germany", "Soviet Union"],
        "start_date": pd.to_datetime(["1990-10-03", "1992-01-01"]),
        "end_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
    })
