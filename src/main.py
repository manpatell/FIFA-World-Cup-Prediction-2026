"""FIFA WC 2026 Prediction Pipeline Orchestrator.

Runs the full ML pipeline end-to-end or specific steps on demand.

Usage:
    python src/main.py                          # full pipeline
    python src/main.py --steps elo features     # partial run
    python src/main.py --steps simulate         # re-run simulation only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.config import Config, get_processed_path, get_raw_path, load_config
from src.features.elo import build_elo_history, get_current_elo
from src.features.form import build_results_long
from src.features.match_features import build_training_dataset
from src.features.squad import build_squad_features_all_teams
from src.ingestion.loader import load_all
from src.ingestion.normalizer import (
    build_name_resolver,
    normalize_rankings,
    normalize_results,
    normalize_shootouts,
    normalize_squad_values,
)
from src.models.goals_model import GoalsModel
from src.models.outcome_model import OutcomeModel
from src.models.train import train_all_models
from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.shootout import ShootoutModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion(cfg: Config, data: dict) -> dict:
    """Step 1: Normalize names and save cleaned results parquet.

    Args:
        cfg: Config object.
        data: Raw data dict from load_all().

    Returns:
        Updated data dict with normalized DataFrames.
    """
    logger.info("Step 1: Ingestion & normalization")
    resolver = build_name_resolver(
        data["name_mapping"], data["former_names"], data["teams"]
    )
    data["results_norm"] = normalize_results(data["results"], resolver)
    data["rankings_norm"] = normalize_rankings(data["rankings"], resolver)
    data["squad_values_norm"] = normalize_squad_values(data["squad_values"], resolver)
    data["shootouts_norm"] = normalize_shootouts(data["shootouts"], resolver)
    data["resolver"] = resolver

    out_path = get_processed_path(cfg, "results_clean")
    data["results_norm"].to_parquet(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path.name, len(data["results_norm"]))
    return data


def run_elo(cfg: Config, data: dict) -> dict:
    """Step 2: Build Elo history and save parquet.

    Args:
        cfg: Config object.
        data: Data dict with results_norm.

    Returns:
        Updated data dict with elo_history and current_elo.
    """
    logger.info("Step 2: Building Elo ratings")
    elo_history = build_elo_history(data["results_norm"], cfg.elo)
    current_elo = get_current_elo(elo_history)

    out_path = get_processed_path(cfg, "elo_ratings")
    elo_history.to_parquet(out_path, index=False)
    logger.info(
        "Saved %s (%d rows). Top teams: %s",
        out_path.name,
        len(elo_history),
        current_elo.nlargest(5).index.tolist(),
    )
    data["elo_history"] = elo_history
    data["current_elo"] = current_elo
    return data


def run_squad_features(cfg: Config, data: dict) -> dict:
    """Step 3: Build squad features for all 48 teams.

    Args:
        cfg: Config object.
        data: Data dict.

    Returns:
        Updated data dict with squad_features DataFrame.
    """
    logger.info("Step 3: Building squad features")
    squad_features = build_squad_features_all_teams(
        player_national=data["player_national"],
        player_profiles=data["player_profiles"],
        player_mv=data["player_mv"],
        player_injuries=data["player_injuries"],
        teams=data["teams"],
        feat_cfg=cfg.features,
    )
    out_path = get_processed_path(cfg, "squad_features")
    squad_features.to_parquet(out_path)
    logger.info("Saved %s (%d teams)", out_path.name, len(squad_features))
    data["squad_features"] = squad_features
    return data


def run_feature_engineering(cfg: Config, data: dict) -> dict:
    """Step 4: Build labelled training dataset.

    Args:
        cfg: Config object.
        data: Data dict.

    Returns:
        Updated data dict with match_dataset DataFrame.
    """
    logger.info("Step 4: Feature engineering — building training dataset")
    results_long = build_results_long(data["results_norm"])
    data["results_long"] = results_long

    match_dataset = build_training_dataset(
        results=data["results_norm"],
        elo_history=data["elo_history"],
        rankings=data["rankings_norm"],
        squad_features=data["squad_features"],
        results_long=results_long,
        cfg=cfg,
    )
    out_path = get_processed_path(cfg, "match_dataset")
    match_dataset.to_parquet(out_path, index=False)
    logger.info(
        "Saved %s (%d rows, %d features)",
        out_path.name,
        len(match_dataset),
        len(match_dataset.columns),
    )
    data["match_dataset"] = match_dataset
    return data


def run_training(cfg: Config, data: dict) -> dict:
    """Step 5: Train XGBoost outcome model and Poisson goals models.

    Args:
        cfg: Config object.
        data: Data dict.

    Returns:
        Updated data dict with fitted models.
    """
    logger.info("Step 5: Model training")
    outcome_model, goals_model = train_all_models(cfg)
    data["outcome_model"] = outcome_model
    data["goals_model"] = goals_model
    return data


def run_simulation(cfg: Config, data: dict) -> pd.DataFrame:
    """Step 6: Run Monte Carlo tournament simulation.

    Loads models from disk if not already in data dict.

    Args:
        cfg: Config object.
        data: Data dict.

    Returns:
        Simulation results DataFrame.
    """
    logger.info("Step 6: Monte Carlo simulation (%d runs)", cfg.simulation.n_simulations)

    # Load models if not already trained in this run
    if "outcome_model" not in data:
        from src.config import get_model_path
        outcome_path = get_model_path(cfg, "outcome_model")
        home_path = get_model_path(cfg, "goals_model_home")
        away_path = get_model_path(cfg, "goals_model_away")
        data["outcome_model"] = OutcomeModel.load(outcome_path)
        data["goals_model"] = GoalsModel.load(cfg.model, home_path, away_path)

    if "results_long" not in data:
        results_clean = pd.read_parquet(get_processed_path(cfg, "results_clean"))
        data["results_long"] = build_results_long(results_clean)

    if "current_elo" not in data:
        elo_history = pd.read_parquet(get_processed_path(cfg, "elo_ratings"))
        data["current_elo"] = get_current_elo(elo_history)

    if "squad_features" not in data:
        data["squad_features"] = pd.read_parquet(get_processed_path(cfg, "squad_features"))

    if "rankings_norm" not in data:
        raw_data = {
            "name_mapping": load_all(cfg)["name_mapping"],
            "former_names": load_all(cfg)["former_names"],
            "teams": data.get("teams", load_all(cfg)["teams"]),
            "rankings": load_all(cfg)["rankings"],
        }
        resolver = build_name_resolver(
            raw_data["name_mapping"], raw_data["former_names"], raw_data["teams"]
        )
        data["rankings_norm"] = normalize_rankings(raw_data["rankings"], resolver)

    shootout_model = ShootoutModel(
        data.get("shootouts_norm", data.get("shootouts", pd.DataFrame()))
    )

    feature_kwargs = {
        "elo_ratings": data["current_elo"],
        "rankings": data["rankings_norm"],
        "squad_features": data["squad_features"],
        "results_long": data["results_long"],
        "form_window": cfg.features.form_window,
    }

    sim_results = run_monte_carlo(
        n_simulations=cfg.simulation.n_simulations,
        teams_df=data["teams"],
        outcome_model=data["outcome_model"],
        goals_model=data["goals_model"],
        shootout_model=shootout_model,
        feature_kwargs=feature_kwargs,
        seed=cfg.simulation.random_seed,
    )

    out_path = get_processed_path(cfg, "simulation_results")
    sim_results.to_parquet(out_path, index=False)
    logger.info("Saved %s", out_path.name)
    logger.info("\nTop 10 predicted winners:")
    for _, row in sim_results.head(10).iterrows():
        logger.info("  %s: %.1f%%", row["team_name"], row["win_prob"] * 100)

    return sim_results


def main(
    steps: list[str] | None = None,
    config_path: str = "config.yaml",
) -> None:
    """Orchestrate the full prediction pipeline.

    Args:
        steps: Optional list of steps to run. If None, runs all steps.
            Valid values: ingestion, elo, squad, features, train, simulate.
        config_path: Path to config.yaml.
    """
    all_steps = ["ingestion", "elo", "squad", "features", "train", "simulate"]
    if steps is None:
        steps = all_steps

    cfg = load_config(config_path)
    logger.info("Pipeline starting. Steps: %s", steps)

    # Always load raw data
    logger.info("Loading raw data...")
    data = load_all(cfg)
    data["teams"] = data["teams"]  # ensure present

    if "ingestion" in steps:
        data = run_ingestion(cfg, data)
    else:
        results_clean_path = get_processed_path(cfg, "results_clean")
        if results_clean_path.exists():
            data["results_norm"] = pd.read_parquet(results_clean_path)
        # Always build resolver and normalize rankings (lightweight)
        resolver = build_name_resolver(
            data["name_mapping"], data["former_names"], data["teams"]
        )
        data["results_norm"] = data.get("results_norm", normalize_results(data["results"], resolver))
        data["rankings_norm"] = normalize_rankings(data["rankings"], resolver)
        data["shootouts_norm"] = normalize_shootouts(data["shootouts"], resolver)

    if "elo" in steps:
        data = run_elo(cfg, data)
    else:
        elo_path = get_processed_path(cfg, "elo_ratings")
        if elo_path.exists():
            data["elo_history"] = pd.read_parquet(elo_path)
            data["current_elo"] = get_current_elo(data["elo_history"])

    if "squad" in steps:
        data = run_squad_features(cfg, data)
    else:
        squad_path = get_processed_path(cfg, "squad_features")
        if squad_path.exists():
            data["squad_features"] = pd.read_parquet(squad_path)

    if "features" in steps:
        data = run_feature_engineering(cfg, data)

    if "train" in steps:
        data = run_training(cfg, data)

    if "simulate" in steps:
        run_simulation(cfg, data)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIFA WC 2026 Prediction Pipeline")
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Pipeline steps to run (ingestion, elo, squad, features, train, simulate). "
             "Omit to run all.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()
    main(steps=args.steps, config_path=args.config)
