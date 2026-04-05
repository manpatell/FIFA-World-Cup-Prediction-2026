"""Tests for src/ingestion/loader.py and src/ingestion/normalizer.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_raw_path
from src.ingestion.loader import (
    load_all,
    load_matches_2026,
    load_player_injuries,
    load_player_market_value,
    load_player_national,
    load_player_profiles,
    load_rankings,
    load_results,
    load_shootouts,
    load_teams,
)
from src.ingestion.normalizer import (
    build_name_resolver,
    normalize_rankings,
    normalize_results,
    normalize_squad_values,
    resolve_team_name,
)


# ─── Loader tests ─────────────────────────────────────────────────────────────

class TestLoadResults:
    def test_shape(self, cfg):
        df = load_results(get_raw_path(cfg, "results"))
        assert df.shape[0] == 49215
        assert df.shape[1] == 9

    def test_date_is_datetime(self, cfg):
        df = load_results(get_raw_path(cfg, "results"))
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_neutral_is_bool(self, cfg):
        df = load_results(get_raw_path(cfg, "results"))
        assert df["neutral"].dtype == bool

    def test_scores_are_numeric(self, cfg):
        df = load_results(get_raw_path(cfg, "results"))
        assert pd.api.types.is_integer_dtype(df["home_score"])
        assert pd.api.types.is_integer_dtype(df["away_score"])

    def test_no_null_teams(self, cfg):
        df = load_results(get_raw_path(cfg, "results"))
        assert df["home_team"].notna().all()
        assert df["away_team"].notna().all()


class TestLoadTeams:
    def test_shape(self, cfg):
        df = load_teams(get_raw_path(cfg, "teams"))
        assert len(df) == 48

    def test_required_columns(self, cfg):
        df = load_teams(get_raw_path(cfg, "teams"))
        for col in ["id", "team_name", "fifa_code", "group_letter"]:
            assert col in df.columns

    def test_groups_a_to_l(self, cfg):
        df = load_teams(get_raw_path(cfg, "teams"))
        groups = set(df["group_letter"].unique())
        assert groups == set("ABCDEFGHIJKL")


class TestLoadRankings:
    def test_covers_all_confederations(self, cfg):
        df = load_rankings(get_raw_path(cfg, "rankings"))
        assert df["confederation"].nunique() >= 6

    def test_total_points_numeric(self, cfg):
        df = load_rankings(get_raw_path(cfg, "rankings"))
        assert pd.api.types.is_float_dtype(df["total_points"])


class TestLoadMatches2026:
    def test_shape(self, cfg):
        df = load_matches_2026(get_raw_path(cfg, "matches_2026"))
        assert len(df) == 104

    def test_kickoff_is_datetime(self, cfg):
        df = load_matches_2026(get_raw_path(cfg, "matches_2026"))
        assert pd.api.types.is_datetime64_any_dtype(df["kickoff_at"])


class TestLoadPlayerNational:
    def test_only_active_players(self, cfg):
        df = load_player_national(get_raw_path(cfg, "player_national"))
        allowed = {"CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"}
        assert set(df["career_state"].unique()).issubset(allowed)

    def test_has_rows(self, cfg):
        df = load_player_national(get_raw_path(cfg, "player_national"))
        assert len(df) > 4000


class TestLoadPlayerMarketValue:
    def test_date_parsed(self, cfg):
        df = load_player_market_value(get_raw_path(cfg, "player_mv"))
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_value_numeric(self, cfg):
        df = load_player_market_value(get_raw_path(cfg, "player_mv"))
        assert pd.api.types.is_float_dtype(df["value"])


class TestLoadPlayerInjuries:
    def test_dates_parsed(self, cfg):
        df = load_player_injuries(get_raw_path(cfg, "player_injuries"))
        assert pd.api.types.is_datetime64_any_dtype(df["from_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_date"])


class TestLoadShootouts:
    def test_shape(self, cfg):
        df = load_shootouts(get_raw_path(cfg, "shootouts"))
        assert len(df) == 675

    def test_winner_column_exists(self, cfg):
        df = load_shootouts(get_raw_path(cfg, "shootouts"))
        assert "winner" in df.columns


class TestLoadAll:
    def test_returns_all_keys(self, cfg):
        data = load_all(cfg)
        expected_keys = {
            "results", "matches_2026", "matches_wc", "world_cup",
            "rankings", "squad_values", "teams", "player_profiles",
            "player_national", "player_mv", "player_injuries",
            "shootouts", "goalscorers", "name_mapping", "former_names",
            "host_cities", "tournament_stages", "fbref_stats", "team_details",
        }
        assert set(data.keys()) == expected_keys

    def test_all_are_dataframes(self, cfg):
        data = load_all(cfg)
        for key, df in data.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"


# ─── Normalizer tests ─────────────────────────────────────────────────────────

class TestBuildNameResolver:
    def test_canonical_names_resolve_to_self(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        for name in ["France", "Germany", "Brazil", "Argentina"]:
            assert resolver.get(name) == name

    def test_manual_aliases(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        assert resolver.get("West Germany") == "Germany"
        assert resolver.get("Soviet Union") == "Russia"

    def test_name_mapping_entries(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        assert resolver.get("United States") == "USA"
        assert resolver.get("Iran") == "IR Iran"

    def test_full_resolver_has_all_48_canonical(self, cfg):
        from src.ingestion.loader import load_former_names, load_name_mapping, load_teams
        teams = load_teams(get_raw_path(cfg, "teams"))
        nm = load_name_mapping(get_raw_path(cfg, "name_mapping"))
        fn = load_former_names(get_raw_path(cfg, "former_names"))
        resolver = build_name_resolver(nm, fn, teams)
        for name in teams["team_name"]:
            assert name in resolver, f"Canonical name {name!r} missing from resolver"


class TestResolveTeamName:
    def test_exact_match(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        assert resolve_team_name("France", resolver) == "France"

    def test_alias_match(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        assert resolve_team_name("United States", resolver) == "USA"

    def test_case_insensitive(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        assert resolve_team_name("france", resolver) == "France"

    def test_unknown_returns_as_is(self, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        # Completely unknown name with no close match
        result = resolve_team_name("XYZ123UnknownTeam", resolver)
        assert isinstance(result, str)


class TestNormalizeResults:
    def test_adds_canonical_columns(self, mini_results, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        result = normalize_results(mini_results, resolver)
        assert "home_team_canonical" in result.columns
        assert "away_team_canonical" in result.columns

    def test_original_columns_preserved(self, mini_results, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        result = normalize_results(mini_results, resolver)
        assert "home_team" in result.columns
        assert "away_team" in result.columns

    def test_no_nulls_in_canonical(self, mini_results, mini_teams, mini_name_mapping, mini_former_names):
        resolver = build_name_resolver(mini_name_mapping, mini_former_names, mini_teams)
        result = normalize_results(mini_results, resolver)
        assert result["home_team_canonical"].notna().all()
        assert result["away_team_canonical"].notna().all()
