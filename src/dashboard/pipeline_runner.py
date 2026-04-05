"""Pipeline execution engine for the Streamlit dashboard.

Provides log-capturing step execution, hyperparam overrides,
and pipeline reset without touching config.yaml on disk.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import streamlit as st

from src.config import Config
from src.ingestion.loader import load_all
from src.main import (
    run_feature_engineering,
    run_ingestion,
    run_elo,
    run_simulation,
    run_squad_features,
    run_training,
)

# ── Step metadata ─────────────────────────────────────────────────────────────

STEPS: list[str] = ["ingestion", "elo", "squad", "features", "train", "simulate"]

STEP_METADATA: dict[str, dict] = {
    "ingestion": {
        "label": "Ingestion & Normalization",
        "description": "Normalize team names, clean match results from 49k historical matches.",
        "num": 1,
    },
    "elo": {
        "label": "Elo Ratings",
        "description": "Build rolling Elo history with K-factor scaling and margin-of-victory adjustments.",
        "num": 2,
    },
    "squad": {
        "label": "Squad Features",
        "description": "Compute market value, injury-adjusted value, average age and caps per team.",
        "num": 3,
    },
    "features": {
        "label": "Feature Engineering",
        "description": "Build labelled training dataset with 30 features per match (Elo, form, H2H, squad).",
        "num": 4,
    },
    "train": {
        "label": "Train Models",
        "description": "XGBoost multiclass + Poisson GLM with chronological train/test split.",
        "num": 5,
    },
    "simulate": {
        "label": "Monte Carlo Simulation",
        "description": "Simulate the full 48-team bracket. ~3 minutes for 10,000 runs.",
        "num": 6,
    },
}

_STEP_FUNCTIONS = {
    "ingestion": run_ingestion,
    "elo": run_elo,
    "squad": run_squad_features,
    "features": run_feature_engineering,
    "train": run_training,
    "simulate": run_simulation,
}


# ── Log capture ───────────────────────────────────────────────────────────────

class _StreamlitLogHandler(logging.Handler):
    """Logging handler that appends formatted records to a list."""

    def __init__(self, log_lines: list[str]) -> None:
        super().__init__()
        self.log_lines = log_lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass


# ── Core runner ───────────────────────────────────────────────────────────────

def run_pipeline_step(
    step_name: str,
    cfg: Config,
    data: dict,
) -> tuple[dict, list[str], Exception | None]:
    """Execute one pipeline step with log capture.

    Args:
        step_name: One of STEPS (ingestion, elo, squad, features, train, simulate).
        cfg: Base Config object. apply_overrides() is called internally for
            train/simulate using st.session_state["hyperparams"].
        data: Current pipeline data dict (mutated and returned).

    Returns:
        Tuple of (updated_data, log_lines, error_or_None).
    """
    log_lines: list[str] = []
    handler = _StreamlitLogHandler(log_lines)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    err: Exception | None = None
    try:
        # Ensure raw data is loaded for step 1
        if not data:
            data = load_all(cfg)

        # Apply hyperparam overrides for train/simulate steps
        step_cfg = cfg
        if step_name in ("train", "simulate"):
            overrides = st.session_state.get("hyperparams", {})
            if overrides:
                step_cfg = apply_overrides(cfg, overrides)

        step_fn = _STEP_FUNCTIONS[step_name]

        if step_name == "simulate":
            # run_simulation returns a DataFrame, not dict
            result_df = step_fn(step_cfg, data)
            data["sim_results"] = result_df
        else:
            data = step_fn(step_cfg, data)

    except Exception as exc:  # noqa: BLE001
        err = exc
        log_lines.append(f"ERROR: {exc}")
    finally:
        root_logger.removeHandler(handler)

    return data, log_lines, err


# ── Hyperparam overrides ──────────────────────────────────────────────────────

def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Return a deep-copied Config with overridden hyperparameters.

    Never touches config.yaml on disk.

    Args:
        cfg: Base Config to copy from.
        overrides: Dict of hyperparam values from UI widgets.

    Returns:
        New Config instance with override fields applied.
    """
    c = copy.deepcopy(cfg)

    if "n_estimators" in overrides:
        c.model.xgboost.n_estimators = int(overrides["n_estimators"])
    if "max_depth" in overrides:
        c.model.xgboost.max_depth = int(overrides["max_depth"])
    if "learning_rate" in overrides:
        c.model.xgboost.learning_rate = float(overrides["learning_rate"])
    if "subsample" in overrides:
        c.model.xgboost.subsample = float(overrides["subsample"])
    if "colsample_bytree" in overrides:
        c.model.xgboost.colsample_bytree = float(overrides["colsample_bytree"])
    if "poisson_alpha" in overrides:
        c.model.poisson.alpha = float(overrides["poisson_alpha"])
    if "n_simulations" in overrides:
        c.simulation.n_simulations = int(overrides["n_simulations"])
    if "form_window" in overrides:
        c.features.form_window = int(overrides["form_window"])
    if "home_advantage" in overrides:
        c.elo.home_advantage = float(overrides["home_advantage"])

    return c


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_pipeline(cfg: Config) -> list[str]:
    """Delete all processed parquets and model pkl files, clear Streamlit cache.

    Args:
        cfg: Config with resolved paths.

    Returns:
        List of deleted filenames (for display).
    """
    deleted: list[str] = []

    processed_dir = cfg.root / cfg.paths.data_processed
    for f in Path(processed_dir).glob("*.parquet"):
        f.unlink()
        deleted.append(f.name)

    models_dir = cfg.root / cfg.paths.models_dir
    for f in Path(models_dir).glob("*.pkl"):
        f.unlink()
        deleted.append(f.name)

    st.cache_data.clear()
    return deleted
