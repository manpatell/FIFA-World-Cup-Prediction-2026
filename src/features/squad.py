"""Squad-level feature aggregation from player-level data.

Joins player market values, national performances, and injury data
to produce per-team feature vectors for all 48 qualified nations.
"""

from __future__ import annotations

import pandas as pd

from src.config import FeatureConfig


# Map citizenship string fragments → canonical team names (top-level only)
# Full resolution handled by the normalizer; this is a quick secondary pass.
_CITIZENSHIP_OVERRIDES: dict[str, str] = {
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "United States": "USA",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Iran": "IR Iran",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Congo, Democratic Republic of the": "DR Congo",
}


def _primary_citizenship(citizenship: str) -> str:
    """Extract the primary nationality from a citizenship string.

    Transfermarkt stores dual citizenships as "Country1  Country2".
    Take the first one as the primary national team affiliation.

    Args:
        citizenship: Raw citizenship string.

    Returns:
        Primary country name string.
    """
    if not isinstance(citizenship, str) or not citizenship.strip():
        return ""
    parts = citizenship.split()
    # Citizenship strings are space-separated but country names can be multi-word.
    # We just return the full string and let the resolver handle it.
    # For "United States  Burundi" style dual citizenship, take the first country.
    # Double-space is used as a separator in this dataset.
    if "  " in citizenship:
        primary = citizenship.split("  ")[0].strip()
    else:
        primary = citizenship.strip()
    return _CITIZENSHIP_OVERRIDES.get(primary, primary)


def get_latest_player_values(
    player_mv: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.Series:
    """Get the most recent market value per player on or before snapshot_date.

    Args:
        player_mv: DataFrame from load_player_market_value().
        snapshot_date: Reference date for value lookup.

    Returns:
        Series indexed by player_id, values = market value in EUR.
    """
    valid = player_mv[player_mv["date"] <= snapshot_date].copy()
    latest = valid.sort_values("date").groupby("player_id")["value"].last()
    return latest


def identify_current_squad(
    player_national: pd.DataFrame,
    player_profiles: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    """Return active national players for a given team.

    Uses citizenship from player_profiles to map players to teams.
    Takes the row with the most caps if a player has multiple entries.

    Args:
        player_national: Active national players DataFrame.
        player_profiles: Player profiles DataFrame.
        team_name: Canonical team name.

    Returns:
        Merged DataFrame of active squad players with profile info.
    """
    # Join national performances with profiles
    merged = player_national.merge(
        player_profiles[["player_id", "player_name", "citizenship",
                          "main_position", "date_of_birth"]],
        on="player_id",
        how="inner",
    )

    # Map citizenship to canonical team name
    merged["primary_nation"] = merged["citizenship"].apply(_primary_citizenship)

    # Filter to the requested team
    team_players = merged[
        merged["primary_nation"].str.lower() == team_name.lower()
    ].copy()

    # If multiple rows per player, keep the one with most caps
    team_players = (
        team_players.sort_values("matches", ascending=False)
        .drop_duplicates(subset="player_id", keep="first")
    )
    return team_players.reset_index(drop=True)


def identify_injured_players(
    player_injuries: pd.DataFrame,
    player_ids: list[int],
    reference_date: pd.Timestamp,
    lookback_days: int,
) -> set[int]:
    """Return set of player_ids with recent or current injuries.

    A player is considered injured if:
    - injury end_date is null (still ongoing), OR
    - injury end_date is within lookback_days before reference_date

    Args:
        player_injuries: DataFrame from load_player_injuries().
        player_ids: List of player IDs to check.
        reference_date: Date to evaluate injury status against.
        lookback_days: Injuries ending within this many days count.

    Returns:
        Set of player_ids considered currently or recently injured.
    """
    if not player_ids:
        return set()

    cutoff = reference_date - pd.Timedelta(days=lookback_days)
    squad_injuries = player_injuries[
        player_injuries["player_id"].isin(player_ids)
    ].copy()

    # Ongoing: end_date is null
    ongoing = squad_injuries[squad_injuries["end_date"].isna()]

    # Recent: ended after cutoff
    recent = squad_injuries[
        squad_injuries["end_date"].notna() &
        (squad_injuries["end_date"] >= cutoff)
    ]

    injured_ids = set(ongoing["player_id"]) | set(recent["player_id"])
    return injured_ids


def aggregate_squad_features(
    player_national: pd.DataFrame,
    player_profiles: pd.DataFrame,
    player_mv: pd.DataFrame,
    player_injuries: pd.DataFrame,
    team_name: str,
    feat_cfg: FeatureConfig,
) -> dict[str, float]:
    """Compute full squad feature vector for one team.

    Args:
        player_national: Active national players.
        player_profiles: Player profiles.
        player_mv: Historical market values.
        player_injuries: Injury history.
        team_name: Canonical team name.
        feat_cfg: Feature configuration.

    Returns:
        Dict with squad features: total_squad_value, top23_value,
        injury_adj_value, injury_pct_value_lost, avg_age, avg_caps,
        avg_goals_per_cap, n_players, n_injured, top_player_value.
    """
    snapshot_date = pd.Timestamp(feat_cfg.mv_snapshot_date)
    squad = identify_current_squad(player_national, player_profiles, team_name)

    if len(squad) == 0:
        return _empty_squad_features()

    # Get latest market values for squad players
    latest_values = get_latest_player_values(player_mv, snapshot_date)
    squad = squad.copy()
    squad["market_value"] = squad["player_id"].map(latest_values).fillna(0.0)

    # Top N players by market value
    top_n = squad.nlargest(feat_cfg.squad_top_n, "market_value")
    top23_value = float(top_n["market_value"].sum())

    # Identify injured players
    injured_ids = identify_injured_players(
        player_injuries,
        list(top_n["player_id"]),
        snapshot_date,
        feat_cfg.injury_lookback_days,
    )
    injured_value = float(
        top_n[top_n["player_id"].isin(injured_ids)]["market_value"].sum()
    )
    injury_adj_value = top23_value - injured_value

    # Age calculation
    squad["age"] = (snapshot_date - squad["date_of_birth"]).dt.days / 365.25
    avg_age = float(squad["age"].mean()) if squad["age"].notna().any() else 26.0

    # Experience
    avg_caps = float(squad["matches"].mean()) if len(squad) > 0 else 0.0
    total_goals = float(squad["goals"].sum())
    total_caps = float(squad["matches"].sum())
    avg_goals_per_cap = float(total_goals / total_caps) if total_caps > 0 else 0.0

    return {
        "total_squad_value": float(squad["market_value"].sum()),
        "top23_value": top23_value,
        "injury_adj_value": injury_adj_value,
        "injury_pct_value_lost": float(injured_value / top23_value)
        if top23_value > 0 else 0.0,
        "avg_age": avg_age,
        "avg_caps": avg_caps,
        "avg_goals_per_cap": avg_goals_per_cap,
        "n_players": len(squad),
        "n_injured": len(injured_ids),
        "top_player_value": float(top_n["market_value"].max()) if len(top_n) > 0 else 0.0,
    }


def _empty_squad_features() -> dict[str, float]:
    """Return zero-filled squad features for teams with no data."""
    return {
        "total_squad_value": 0.0,
        "top23_value": 0.0,
        "injury_adj_value": 0.0,
        "injury_pct_value_lost": 0.0,
        "avg_age": 26.0,
        "avg_caps": 0.0,
        "avg_goals_per_cap": 0.0,
        "n_players": 0,
        "n_injured": 0,
        "top_player_value": 0.0,
    }


def build_squad_features_all_teams(
    player_national: pd.DataFrame,
    player_profiles: pd.DataFrame,
    player_mv: pd.DataFrame,
    player_injuries: pd.DataFrame,
    teams: pd.DataFrame,
    feat_cfg: FeatureConfig,
) -> pd.DataFrame:
    """Compute squad features for all 48 qualified teams.

    Args:
        player_national: Active national players.
        player_profiles: Player profiles.
        player_mv: Historical market values.
        player_injuries: Injury history.
        teams: Teams DataFrame (defines canonical team names).
        feat_cfg: Feature configuration.

    Returns:
        DataFrame indexed by canonical team_name with squad feature columns.
    """
    records: list[dict] = []
    for team_name in teams["team_name"]:
        feats = aggregate_squad_features(
            player_national, player_profiles, player_mv,
            player_injuries, team_name, feat_cfg,
        )
        feats["team_name"] = team_name
        records.append(feats)

    df = pd.DataFrame(records).set_index("team_name")
    return df
