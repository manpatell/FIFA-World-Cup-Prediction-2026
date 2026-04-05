"""Raw CSV loaders for all data sources.

Each function loads one CSV, applies only dtype coercion and datetime
parsing, and returns a typed DataFrame. No transformations are done here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config, get_raw_path


def load_results(path: Path) -> pd.DataFrame:
    """Load results.csv — 49k+ international match results.

    Args:
        path: Absolute path to results.csv.

    Returns:
        DataFrame with columns: date, home_team, away_team, home_score,
        away_score, tournament, city, country, neutral.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    return df


def load_matches_2026(path: Path) -> pd.DataFrame:
    """Load matches.csv — 2026 World Cup fixture schedule.

    Args:
        path: Absolute path to matches.csv.

    Returns:
        DataFrame with columns: id, match_number, home_team_id, away_team_id,
        city_id, stage_id, kickoff_at, match_label.
    """
    df = pd.read_csv(path)
    df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True)
    df["home_team_id"] = pd.to_numeric(df["home_team_id"], errors="coerce").astype("Int64")
    df["away_team_id"] = pd.to_numeric(df["away_team_id"], errors="coerce").astype("Int64")
    return df


def load_matches_wc(path: Path) -> pd.DataFrame:
    """Load matches_1930_2022.csv — World Cup match detail with xG.

    Args:
        path: Absolute path to matches_1930_2022.csv.

    Returns:
        DataFrame with key columns including home/away scores, xG,
        red cards, penalties, Year, Round.
    """
    cols = [
        "home_team", "away_team", "home_score", "away_score",
        "home_xg", "away_xg", "home_penalty", "away_penalty",
        "home_red_card", "away_red_card",
        "Round", "Date", "Year",
    ]
    df = pd.read_csv(path, encoding="utf-8", usecols=lambda c: c in cols, low_memory=False)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_teams(path: Path) -> pd.DataFrame:
    """Load teams.csv — 48 qualified nations with group assignments.

    Args:
        path: Absolute path to teams.csv.

    Returns:
        DataFrame with columns: id, team_name, fifa_code, group_letter,
        is_placeholder.
    """
    df = pd.read_csv(path)
    df["is_placeholder"] = df["is_placeholder"].astype(bool)
    return df


def load_world_cup(path: Path) -> pd.DataFrame:
    """Load world_cup.csv — tournament summary since 1930.

    Args:
        path: Absolute path to world_cup.csv.

    Returns:
        DataFrame with columns: Year, Host, Teams, Champion, Runner-Up,
        TopScorrer, Attendance, AttendanceAvg, Matches.
    """
    return pd.read_csv(path)


def load_rankings(path: Path) -> pd.DataFrame:
    """Load fifa_men_rankings_current.csv — April 2026 FIFA rankings.

    Args:
        path: Absolute path to fifa_men_rankings_current.csv.

    Returns:
        DataFrame with columns: rank, country, country_code, confederation,
        total_points, previous_rank, ranking_movement.
    """
    df = pd.read_csv(path)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    return df


def load_squad_values(path: Path) -> pd.DataFrame:
    """Load transfermarkt_squad_market_values.csv — squad market values.

    Args:
        path: Absolute path to transfermarkt_squad_market_values.csv.

    Returns:
        DataFrame with columns: nation, squad_size, avg_age,
        total_market_value_eur, confederation.
    """
    df = pd.read_csv(path)
    df["total_market_value_eur"] = pd.to_numeric(
        df["total_market_value_eur"], errors="coerce"
    )
    return df


def load_player_profiles(path: Path) -> pd.DataFrame:
    """Load player_profiles.csv — 92k player metadata.

    Args:
        path: Absolute path to player_profiles.csv.

    Returns:
        DataFrame with columns: player_id, player_name, citizenship,
        position, main_position, date_of_birth, current_club_name.
    """
    cols = [
        "player_id", "player_name", "citizenship", "position",
        "main_position", "date_of_birth", "current_club_name",
        "current_club_id",
    ]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    return df


def load_player_national(path: Path) -> pd.DataFrame:
    """Load player_national_performances.csv — caps and goals per player.

    Filters to active players (CURRENT and RECENT national players).

    Args:
        path: Absolute path to player_national_performances.csv.

    Returns:
        DataFrame with columns: player_id, team_id, matches, goals,
        shirt_number, career_state.
    """
    df = pd.read_csv(path)
    active_states = {"CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"}
    df = df[df["career_state"].isin(active_states)].copy()
    df = df.reset_index(drop=True)
    return df


def load_player_market_value(path: Path) -> pd.DataFrame:
    """Load player_market_value.csv — historical market value per player.

    Args:
        path: Absolute path to player_market_value.csv.

    Returns:
        DataFrame with columns: player_id, date, value.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"date_unix": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_player_injuries(path: Path) -> pd.DataFrame:
    """Load player_injuries.csv — injury history per player.

    Args:
        path: Absolute path to player_injuries.csv.

    Returns:
        DataFrame with columns: player_id, season_name, injury_reason,
        from_date, end_date, days_missed, games_missed.
    """
    df = pd.read_csv(path)
    df["from_date"] = pd.to_datetime(df["from_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def load_shootouts(path: Path) -> pd.DataFrame:
    """Load shootouts.csv — penalty shootout outcomes.

    Args:
        path: Absolute path to shootouts.csv.

    Returns:
        DataFrame with columns: date, home_team, away_team, winner,
        first_shooter.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_goalscorers(path: Path) -> pd.DataFrame:
    """Load goalscorers.csv — individual goal records.

    Args:
        path: Absolute path to goalscorers.csv.

    Returns:
        DataFrame with columns: date, home_team, away_team, team,
        scorer, minute, own_goal, penalty.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["own_goal"] = df["own_goal"].astype(bool)
    df["penalty"] = df["penalty"].astype(bool)
    return df


def load_name_mapping(path: Path) -> pd.DataFrame:
    """Load name_mapping.csv — manual team name normalization entries.

    Args:
        path: Absolute path to name_mapping.csv.

    Returns:
        DataFrame with columns: teams_csv_name, standardized_name,
        fifa_code, notes.
    """
    return pd.read_csv(path)


def load_former_names(path: Path) -> pd.DataFrame:
    """Load former_names.csv — historical country name changes.

    Args:
        path: Absolute path to former_names.csv.

    Returns:
        DataFrame with columns: current, former, start_date, end_date.
    """
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def load_host_cities(path: Path) -> pd.DataFrame:
    """Load host_cities.csv — 2026 host venue information.

    Args:
        path: Absolute path to host_cities.csv.

    Returns:
        DataFrame with columns: id, city_name, country, venue_name,
        region_cluster, airport_code.
    """
    return pd.read_csv(path)


def load_tournament_stages(path: Path) -> pd.DataFrame:
    """Load tournament_stages.csv — stage id/name mapping.

    Args:
        path: Absolute path to tournament_stages.csv.

    Returns:
        DataFrame with columns: id, stage_name, stage_order.
    """
    return pd.read_csv(path)


def load_fbref_stats(path: Path) -> pd.DataFrame:
    """Load fbref_advanced_player_stats.csv — advanced player stats.

    Note: Only covers 40 players from limited sources. Use as
    supplementary signal only.

    Args:
        path: Absolute path to fbref_advanced_player_stats.csv.

    Returns:
        DataFrame with columns: player, nation, position, age,
        matches_played, minutes, xg, npxg, xag, progressive_passes,
        progressive_carries.
    """
    cols = [
        "player", "nation", "position", "age", "matches_played",
        "minutes", "xg", "npxg", "xag", "progressive_passes",
        "progressive_carries",
    ]
    return pd.read_csv(path, usecols=cols)


def load_team_details(path: Path) -> pd.DataFrame:
    """Load team_details.csv — Transfermarkt club metadata.

    Useful for mapping current_club_id → club country/league.

    Args:
        path: Absolute path to team_details.csv.

    Returns:
        DataFrame with columns: club_id, club_name, country_name,
        competition_name.
    """
    cols = ["club_id", "club_slug", "club_name", "country_name",
            "competition_name", "club_division"]
    return pd.read_csv(path, usecols=cols, low_memory=False)


def load_all(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load every source CSV and return a keyed dictionary.

    Args:
        cfg: Loaded Config object.

    Returns:
        Dictionary mapping source name → DataFrame for all 19 source files.
    """
    return {
        "results": load_results(get_raw_path(cfg, "results")),
        "matches_2026": load_matches_2026(get_raw_path(cfg, "matches_2026")),
        "matches_wc": load_matches_wc(get_raw_path(cfg, "matches_wc")),
        "world_cup": load_world_cup(get_raw_path(cfg, "world_cup")),
        "rankings": load_rankings(get_raw_path(cfg, "rankings")),
        "squad_values": load_squad_values(get_raw_path(cfg, "squad_values")),
        "teams": load_teams(get_raw_path(cfg, "teams")),
        "player_profiles": load_player_profiles(get_raw_path(cfg, "player_profiles")),
        "player_national": load_player_national(get_raw_path(cfg, "player_national")),
        "player_mv": load_player_market_value(get_raw_path(cfg, "player_mv")),
        "player_injuries": load_player_injuries(get_raw_path(cfg, "player_injuries")),
        "shootouts": load_shootouts(get_raw_path(cfg, "shootouts")),
        "goalscorers": load_goalscorers(get_raw_path(cfg, "goalscorers")),
        "name_mapping": load_name_mapping(get_raw_path(cfg, "name_mapping")),
        "former_names": load_former_names(get_raw_path(cfg, "former_names")),
        "host_cities": load_host_cities(get_raw_path(cfg, "host_cities")),
        "tournament_stages": load_tournament_stages(get_raw_path(cfg, "tournament_stages")),
        "fbref_stats": load_fbref_stats(get_raw_path(cfg, "fbref_stats")),
        "team_details": load_team_details(get_raw_path(cfg, "team_details")),
    }
