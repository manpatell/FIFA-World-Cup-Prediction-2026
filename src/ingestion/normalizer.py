"""Team name normalization across all data sources.

Builds a resolver that maps any alias (from results.csv, rankings, etc.)
to a single canonical team_name as used in teams.csv.
"""

from __future__ import annotations

import difflib

import pandas as pd


# Hard-coded aliases that are not covered by name_mapping.csv or former_names.csv
_MANUAL_ALIASES: dict[str, str] = {
    # results.csv vs teams.csv
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Congo": "Congo",
    "Iran": "IR Iran",
    # FIFA rankings / Transfermarkt variants
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "United States of America": "USA",
    "Curacao": "Curaçao",
    "Cura\u00e7ao": "Curaçao",
    # Historical — merged into current nation
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Yugoslavia": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Netherlands Antilles": "Curaçao",
    "Zaire": "DR Congo",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "Eire": "Republic of Ireland",
    "Irish Free State": "Republic of Ireland",
    "Burma": "Myanmar",
    "Western Samoa": "Samoa",
    # Misc
    "Chinese Taipei": "Chinese Taipei",
    "China PR": "China",
    "China P.R.": "China",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Macau": "Macao",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "St. Lucia": "Saint Lucia",
    "Timor-Leste": "East Timor",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def build_name_resolver(
    name_mapping: pd.DataFrame,
    former_names: pd.DataFrame,
    teams: pd.DataFrame,
) -> dict[str, str]:
    """Build a lookup dict mapping any alias → canonical team_name.

    Priority order:
    1. Exact match to canonical team_name (identity)
    2. Manual hard-coded aliases (_MANUAL_ALIASES)
    3. name_mapping.csv entries (standardized_name and teams_csv_name)
    4. former_names.csv entries (former → current)
    5. Fuzzy fallback (difflib, threshold 0.85)

    Args:
        name_mapping: DataFrame from load_name_mapping().
        former_names: DataFrame from load_former_names().
        teams: DataFrame from load_teams() — defines canonical names.

    Returns:
        Dictionary {alias: canonical_team_name}.
    """
    canonical_names: set[str] = set(teams["team_name"].dropna().unique())
    resolver: dict[str, str] = {}

    # 1. Identity for all canonical names
    for name in canonical_names:
        resolver[name] = name

    # 2. Manual aliases
    for alias, canonical in _MANUAL_ALIASES.items():
        resolver[alias] = canonical

    # 3. name_mapping.csv — both columns map to the canonical teams_csv_name
    for _, row in name_mapping.iterrows():
        teams_csv = str(row.get("teams_csv_name", "")).strip()
        standardized = str(row.get("standardized_name", "")).strip()
        # teams_csv_name is the canonical (from teams.csv)
        if teams_csv in canonical_names:
            if standardized:
                resolver[standardized] = teams_csv
            # Also map fifa_code if present
            code = str(row.get("fifa_code", "")).strip()
            if code:
                resolver[code] = teams_csv

    # 4. former_names.csv — map former → current
    for _, row in former_names.iterrows():
        current = str(row.get("current", "")).strip()
        former = str(row.get("former", "")).strip()
        if former and current:
            resolver[former] = current

    return resolver


def resolve_team_name(name: str, resolver: dict[str, str]) -> str:
    """Map a single team name string to its canonical form.

    Tries exact lookup first, then case-insensitive, then fuzzy matching.

    Args:
        name: Raw team name string.
        resolver: Dict built by build_name_resolver().

    Returns:
        Canonical team name string.

    Raises:
        ValueError: If no match is found above the fuzzy threshold (0.80).
    """
    if not isinstance(name, str) or not name.strip():
        return name

    name = name.strip()

    # Exact match
    if name in resolver:
        return resolver[name]

    # Case-insensitive match
    lower_map = {k.lower(): v for k, v in resolver.items()}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    # Fuzzy match
    candidates = list(resolver.keys())
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.80)
    if matches:
        return resolver[matches[0]]

    # Return as-is if nothing found (avoids crashes on unknown teams)
    return name


def normalize_results(
    results: pd.DataFrame,
    resolver: dict[str, str],
) -> pd.DataFrame:
    """Apply resolver to home_team and away_team columns.

    Args:
        results: DataFrame from load_results().
        resolver: Dict from build_name_resolver().

    Returns:
        results with added columns home_team_canonical, away_team_canonical.
    """
    df = results.copy()
    df["home_team_canonical"] = df["home_team"].map(
        lambda x: resolve_team_name(x, resolver)
    )
    df["away_team_canonical"] = df["away_team"].map(
        lambda x: resolve_team_name(x, resolver)
    )
    return df


def normalize_rankings(
    rankings: pd.DataFrame,
    resolver: dict[str, str],
) -> pd.DataFrame:
    """Map rankings 'country' column to canonical team names.

    Args:
        rankings: DataFrame from load_rankings().
        resolver: Dict from build_name_resolver().

    Returns:
        rankings with added column team_name_canonical.
    """
    df = rankings.copy()
    df["team_name_canonical"] = df["country"].map(
        lambda x: resolve_team_name(x, resolver)
    )
    return df


def normalize_squad_values(
    squad_values: pd.DataFrame,
    resolver: dict[str, str],
) -> pd.DataFrame:
    """Map squad_values 'nation' column to canonical team names.

    Args:
        squad_values: DataFrame from load_squad_values().
        resolver: Dict from build_name_resolver().

    Returns:
        squad_values with added column team_name_canonical.
    """
    df = squad_values.copy()
    df["team_name_canonical"] = df["nation"].map(
        lambda x: resolve_team_name(x, resolver)
    )
    return df


def normalize_shootouts(
    shootouts: pd.DataFrame,
    resolver: dict[str, str],
) -> pd.DataFrame:
    """Map shootout home_team/away_team/winner columns to canonical names.

    Args:
        shootouts: DataFrame from load_shootouts().
        resolver: Dict from build_name_resolver().

    Returns:
        shootouts with added columns *_canonical for team columns.
    """
    df = shootouts.copy()
    for col in ["home_team", "away_team", "winner"]:
        df[f"{col}_canonical"] = df[col].map(
            lambda x: resolve_team_name(x, resolver) if pd.notna(x) else x
        )
    return df
