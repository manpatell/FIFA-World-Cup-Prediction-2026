# FIFA World Cup 2026 — AI Winner Prediction

> **Predicted Winner: Argentina (16.4%)**
> France 13.3% · Spain 11.0% · Brazil 7.5%

A full end-to-end machine learning pipeline that predicts the FIFA World Cup 2026 winner. Trained on 49,000+ international matches (2010–2026), the system engineers 30 features per matchup, trains XGBoost + Poisson regression models with chronological cross-validation, runs 10,000 Monte Carlo tournament simulations, and presents results in an interactive Streamlit dashboard with an animated bracket, team deep-dives, and a suspense reveal.

---

## Dashboard Preview

| Tab | What it shows |
|-----|--------------|
| **Win Probabilities** | Animated winner reveal + all-teams probability chart |
| **Stage Heatmap** | Probability of reaching each round per team |
| **Bracket** | Full R32→Final bracket with ML-predicted scores, zoom controls, reveal button |
| **Group Stage** | Group-by-group breakdown with advancement probabilities |
| **Team Analysis** | Per-team deep-dive: Elo, FIFA ranking, squad value, injury impact, tournament path flowchart |
| **Model Info** | Feature importances, accuracy metrics, calibration |
| **Pipeline** | Run all 6 pipeline steps, tune hyperparameters, reset data — all from the browser |

---

## Results

| Team | Win Probability | Reach Final | Reach Semi |
|------|:-:|:-:|:-:|
| Argentina | **16.4%** | 25.6% | 50.2% |
| France | 13.3% | 22.7% | 48.1% |
| Spain | 11.0% | 18.9% | 44.3% |
| Brazil | 7.5% | 15.8% | 38.6% |
| Belgium | 6.8% | 14.2% | 35.4% |
| Portugal | 6.6% | 12.0% | 29.5% |

> Based on 10,000 Monte Carlo simulations. Probabilities reflect the 2026 format: 48 teams, 12 groups, Round of 32.

---

## Model Performance

| Metric | Value | Baseline |
|--------|-------|---------|
| Test accuracy | **60.8%** | 33.3% (random) |
| Ranked Probability Score | **0.1646** | 0.333 (random) |
| Home win accuracy | 87.0% | — |
| Away win accuracy | 66.2% | — |
| Draw accuracy | 1.5% | — |
| Test period | Jun 2023 – Mar 2026 | 3,060 matches |

> Draw prediction is a known weakness across all football models — draws are near-random events. The 60.8% overall accuracy is competitive with commercial sports betting models (typical range: 55–65%).

---

## Architecture

```
data/raw/              ← 19 source CSVs (matches, rankings, squad values, injuries…)
data/processed/        ← Parquet outputs at each pipeline step
models/                ← Trained model files (.pkl)
src/
  config.py            ← Typed Config dataclass, path resolution
  main.py              ← CLI entry point (runs all 6 steps)
  ingestion/
    loader.py          ← One loader per CSV, no transformation
    normalizer.py      ← Name resolver: aliases → canonical team names (difflib fallback)
  features/
    elo.py             ← Sequential Elo history (49k matches), K-factor + MoV multiplier
    form.py            ← Rolling form window (10 matches), pre-computed lookup
    squad.py           ← Squad market value, top-23, injury-adjusted value
    h2h.py             ← Head-to-head records (last 10 meetings)
    match_features.py  ← Assembles 30-feature vector per matchup
  models/
    outcome_model.py   ← XGBoost multiclass (away win / draw / home win)
    goals_model.py     ← Two Poisson GLMs (home goals, away goals separately)
    train.py           ← Chronological train/test split, 5-fold time-series CV
    evaluate.py        ← RPS, log-loss, accuracy, calibration, feature importance
  simulation/
    bracket.py         ← 2026 bracket format, group ranking rules, seeding logic
    monte_carlo.py     ← 10,000 parallel simulations (joblib), precompute predictions
    shootout.py        ← Penalty shootout model from 675 historical shootouts
  dashboard/
    app.py             ← Streamlit app, CSS dark theme, all tabs wired
    data_cache.py      ← @st.cache_data loaders, ML prediction cache
    pipeline_runner.py ← In-browser pipeline execution, log capture, hyperparam overrides
    components/
      win_probs.py     ← Win probability chart + stage heatmap
      bracket.py       ← Animated HTML/SVG bracket with zoom, reveal, ML scores
      team_card.py     ← Per-team metrics, funnel chart
      team_path.py     ← Animated tournament path flowchart
      group_table.py   ← Group stage table
```

---

## Data Sources (19 CSVs)

| File | Description | Rows |
|------|-------------|------|
| `results.csv` | International match results (all-time) | 49,215 |
| `matches.csv` | 2026 fixture schedule | 104 |
| `matches_1930_2022.csv` | World Cup match history | 900 |
| `fifa_men_rankings_current.csv` | Current FIFA world rankings | 210 |
| `transfermarkt_squad_market_values.csv` | Squad market values by nation | 207 |
| `player_profiles.csv` | Player demographics + citizenship | ~30k |
| `player_national_performances.csv` | International caps + goals | ~50k |
| `player_market_value.csv` | Individual player valuations | 901k |
| `player_injuries.csv` | Injury records with dates | ~15k |
| `shootouts.csv` | Penalty shootout outcomes (historical) | 675 |
| `teams.csv` | 48 qualified nations + group draw | 48 |
| `name_mapping.csv` | Country name normalisation | ~200 |
| `former_names.csv` | Historical name changes (e.g. West Germany) | ~30 |

---

## Feature Engineering (30 features)

| Category | Features |
|----------|----------|
| **Elo** | `elo_home`, `elo_away`, `elo_diff` |
| **Rankings** | `rank_home`, `rank_away`, `ranking_diff`, `ranking_points_diff` |
| **Squad value** | `squad_value_home`, `squad_value_away`, `squad_value_ratio`, `injury_adj_ratio` |
| **Form** | `form_pts_home/away`, `win_rate_home/away`, `goals_scored_avg_home/away`, `goals_conceded_avg_home/away` |
| **H2H** | `h2h_win_rate`, `h2h_draw_rate`, `h2h_goal_diff_avg`, `h2h_n_matches` |
| **Squad** | `avg_age_home/away`, `avg_caps_home/away` |
| **Context** | `is_neutral`, `tournament_weight`, `home_advantage_flag` |

Sample weights: exponential decay (5-year half-life) × 3× multiplier for World Cup matches.

---

## Pipeline Steps

```
Step 1 — Ingestion & Normalisation   load CSVs, resolve team name aliases
Step 2 — Elo Ratings                 build Elo history for all 49k matches
Step 3 — Squad Features              aggregate market value, injuries, caps per team
Step 4 — Feature Engineering        assemble 30-feature training dataset (15,299 rows)
Step 5 — Model Training              XGBoost outcome model + dual Poisson goals models
Step 6 — Monte Carlo Simulation     10,000 full tournament simulations → probabilities
```

Run the full pipeline:
```bash
python src/main.py
```

Or run each step individually from the **Pipeline tab** in the dashboard.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/FIFA-World-Cup-2026.git
cd FIFA-World-Cup-2026

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add data

Place all 19 CSV files into `data/raw/`. The expected filenames are listed in `config.yaml` under `paths`.

### 3. Run the pipeline

```bash
python src/main.py
```

This runs all 6 steps and saves outputs to `data/processed/` and `models/`.

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Fine-tuning Hyperparameters

No need to edit files. Open the **Pipeline tab** → expand **Hyperparameter Configuration**:

| Parameter | Range | Default |
|-----------|-------|---------|
| XGBoost n_estimators | 100 – 2000 | 500 |
| XGBoost max_depth | 2 – 10 | 5 |
| XGBoost learning_rate | 0.001 – 0.5 | 0.05 |
| XGBoost subsample | 0.5 – 1.0 | 0.8 |
| Poisson alpha | 0.0 – 10.0 | 0.1 |
| Simulations | 1k / 2.5k / 5k / 10k | 10,000 |
| Elo home advantage | 0 – 300 | 100 |
| Form window | 3 – 20 matches | 10 |

Click **Apply to Next Training Run**, then re-run Steps 5 and 6.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, transformation, feature assembly |
| `xgboost` | Match outcome classifier (3-class) |
| `scikit-learn` | Poisson GLM (goals), cross-validation, metrics |
| `joblib` | Parallel Monte Carlo simulation |
| `streamlit` | Interactive dashboard |
| `plotly` | Charts and heatmaps |
| `difflib` | Fuzzy team name matching |

Python 3.11+ required.

---

## Key Design Decisions

**Chronological CV only** — Random train/test splits leak future results into training for time-series data. All splits are strictly chronological.

**Two separate Poisson models** — One model for home goals, one for away goals. This captures asymmetric scoring patterns that a single model misses.

**Team joins via `citizenship`, not `team_id`** — Transfermarkt's internal `team_id` is incompatible with the canonical team names in `teams.csv`. Players are linked to national teams through the `citizenship` field + name normalisation.

**Sim-results-consistent bracket** — The bracket advances teams using their overall tournament win probability from 10,000 simulations (not per-match ML probability). This ensures the bracket champion matches the win probability tab.

**Pre-computed feature caches** — Form and H2H lookups are built in a single pass to avoid O(n²) per-row recomputation across 49k matches.

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## License

MIT
