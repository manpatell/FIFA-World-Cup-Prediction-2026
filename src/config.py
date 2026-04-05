"""Configuration loader for the FIFA WC 2026 prediction project.

Parses config.yaml into typed dataclasses and resolves all file paths
to absolute Path objects. All other modules import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_raw: str
    data_processed: str
    models_dir: str
    notebooks_dir: str
    # raw source files
    results: str
    matches_2026: str
    matches_wc: str
    world_cup: str
    rankings: str
    squad_values: str
    teams: str
    player_profiles: str
    player_national: str
    player_mv: str
    player_injuries: str
    goalscorers: str
    shootouts: str
    team_details: str
    name_mapping: str
    former_names: str
    host_cities: str
    tournament_stages: str
    fbref_stats: str
    # processed outputs
    results_clean: str
    elo_ratings: str
    team_features: str
    squad_features: str
    match_dataset: str
    simulation_results: str
    # model files
    outcome_model: str
    goals_model_home: str
    goals_model_away: str


@dataclass
class EloConfig:
    initial_rating: float
    k_factors: dict[str, float]
    home_advantage: float
    margin_of_victory_mult: bool


@dataclass
class FeatureConfig:
    form_window: int
    recent_years_weight: int
    wc_match_weight: float
    squad_top_n: int
    mv_snapshot_date: str
    injury_lookback_days: int


@dataclass
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    eval_metric: str
    early_stopping_rounds: int


@dataclass
class PoissonConfig:
    alpha: float
    max_iter: int


@dataclass
class ModelConfig:
    random_seed: int
    test_size: float
    cv_folds: int
    training_data_cutoff: str
    xgboost: XGBoostConfig
    poisson: PoissonConfig


@dataclass
class SimulationConfig:
    n_simulations: int
    random_seed: int
    group_advance_top_n: int
    group_advance_3rd_n: int
    third_place_playoff: bool


@dataclass
class DashboardConfig:
    title: str
    top_n_teams_display: int


@dataclass
class Config:
    paths: PathConfig
    elo: EloConfig
    features: FeatureConfig
    model: ModelConfig
    simulation: SimulationConfig
    dashboard: DashboardConfig
    # resolved absolute root (set after construction)
    root: Path = field(default_factory=Path.cwd)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load and parse config.yaml into a typed Config object.

    Args:
        config_path: Path to config.yaml. Relative paths are resolved
            against the current working directory.

    Returns:
        Fully populated Config dataclass with all sub-configs.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    root = config_path.parent

    paths = PathConfig(**raw["paths"])

    elo_raw = raw["elo"]
    elo = EloConfig(
        initial_rating=float(elo_raw["initial_rating"]),
        k_factors={k: float(v) for k, v in elo_raw["k_factors"].items()},
        home_advantage=float(elo_raw["home_advantage"]),
        margin_of_victory_mult=bool(elo_raw["margin_of_victory_mult"]),
    )

    feat_raw = raw["features"]
    features = FeatureConfig(
        form_window=int(feat_raw["form_window"]),
        recent_years_weight=int(feat_raw["recent_years_weight"]),
        wc_match_weight=float(feat_raw["wc_match_weight"]),
        squad_top_n=int(feat_raw["squad_top_n"]),
        mv_snapshot_date=str(feat_raw["mv_snapshot_date"]),
        injury_lookback_days=int(feat_raw["injury_lookback_days"]),
    )

    model_raw = raw["model"]
    model_cfg = ModelConfig(
        random_seed=int(model_raw["random_seed"]),
        test_size=float(model_raw["test_size"]),
        cv_folds=int(model_raw["cv_folds"]),
        training_data_cutoff=str(model_raw["training_data_cutoff"]),
        xgboost=XGBoostConfig(**model_raw["xgboost"]),
        poisson=PoissonConfig(**model_raw["poisson"]),
    )

    sim_raw = raw["simulation"]
    simulation = SimulationConfig(
        n_simulations=int(sim_raw["n_simulations"]),
        random_seed=int(sim_raw["random_seed"]),
        group_advance_top_n=int(sim_raw["group_advance_top_n"]),
        group_advance_3rd_n=int(sim_raw["group_advance_3rd_n"]),
        third_place_playoff=bool(sim_raw["third_place_playoff"]),
    )

    dashboard = DashboardConfig(**raw["dashboard"])

    cfg = Config(
        paths=paths,
        elo=elo,
        features=features,
        model=model_cfg,
        simulation=simulation,
        dashboard=dashboard,
        root=root,
    )
    return cfg


def get_raw_path(cfg: Config, key: str) -> Path:
    """Return absolute path to a raw data file by config key.

    Args:
        cfg: Loaded Config object.
        key: Attribute name on PathConfig (e.g. ``"results"``).

    Returns:
        Absolute Path to the raw file.
    """
    filename = getattr(cfg.paths, key)
    return cfg.root / cfg.paths.data_raw / filename


def get_processed_path(cfg: Config, key: str) -> Path:
    """Return absolute path to a processed data file by config key.

    Args:
        cfg: Loaded Config object.
        key: Attribute name on PathConfig (e.g. ``"results_clean"``).

    Returns:
        Absolute Path to the processed file.
    """
    filename = getattr(cfg.paths, key)
    return cfg.root / cfg.paths.data_processed / filename


def get_model_path(cfg: Config, key: str) -> Path:
    """Return absolute path to a saved model file by config key.

    Args:
        cfg: Loaded Config object.
        key: Attribute name on PathConfig (e.g. ``"outcome_model"``).

    Returns:
        Absolute Path to the model file.
    """
    filename = getattr(cfg.paths, key)
    return cfg.root / cfg.paths.models_dir / filename
