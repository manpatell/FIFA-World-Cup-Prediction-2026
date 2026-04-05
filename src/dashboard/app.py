"""FIFA World Cup 2026 Predictor — Streamlit Dashboard.

Entry point: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import load_config
from src.dashboard.components.bracket_view import render_predicted_bracket
from src.dashboard.components.group_table import render_group_table
from src.dashboard.components.team_card import render_team_card
from src.dashboard.components.team_path import render_team_path_flowchart
from src.dashboard.components.win_probs import (
    render_stage_probability_heatmap,
    render_win_probability_chart,
)
from src.dashboard.data_cache import load_all_dashboard_data
from src.dashboard.pipeline_runner import (
    STEP_METADATA,
    STEPS,
    reset_pipeline,
    run_pipeline_step,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #131929;
    --surface-2: #1a2338;
    --border: #1e2d4a;
    --accent: #1e6efa;
    --accent-dim: #1452c8;
    --success: #00c16e;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e8eaf0;
    --muted: #6b7a99;
    --font: 'Inter', system-ui, -apple-system, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background-color: var(--bg) !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s, border-color 0.15s !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background-color: transparent !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Step cards */
.step-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
}

.step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.25rem;
}

.step-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
}

.step-desc {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.15rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-not-started { background: #1e2d4a; color: #6b7a99; }
.badge-running     { background: #3d2b00; color: #f59e0b; }
.badge-complete    { background: #003d28; color: #00c16e; }
.badge-failed      { background: #3d0a0a; color: #ef4444; }

/* Log output */
[data-testid="stCode"] {
    background-color: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    max-height: 280px !important;
    overflow-y: auto !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background-color: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: background-color 0.15s !important;
}

[data-testid="stButton"] > button:hover {
    background-color: var(--accent-dim) !important;
}

/* Danger button override — use key class */
.danger-btn > button {
    background-color: var(--danger) !important;
}

/* Dividers */
hr {
    border-color: var(--border) !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Select / input widgets */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div {
    color: var(--text) !important;
}

/* Sidebar widgets */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSlider"] div {
    color: var(--text) !important;
}

/* Success / warning / error boxes */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
}

/* ── Winner Reveal ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 30px 8px rgba(30,110,250,0.25); }
    50%       { box-shadow: 0 0 60px 20px rgba(30,110,250,0.55); }
}

@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

@keyframes bounce-dot {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%           { transform: translateY(-10px); opacity: 1; }
}

@keyframes reveal-entrance {
    0%   { opacity: 0; transform: scale(0.6) translateY(30px); filter: blur(12px); }
    60%  { opacity: 1; transform: scale(1.04) translateY(-4px); filter: blur(0); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes name-glow {
    0%, 100% { text-shadow: 0 0 20px rgba(30,110,250,0.4), 0 0 60px rgba(30,110,250,0.2); }
    50%       { text-shadow: 0 0 40px rgba(30,110,250,0.9), 0 0 80px rgba(30,110,250,0.5), 0 0 120px rgba(100,180,255,0.3); }
}

@keyframes fade-slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes count-pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); }
}

@keyframes trophy-spin {
    0%   { transform: rotateY(0deg); }
    100% { transform: rotateY(360deg); }
}

.mystery-card {
    position: relative;
    background: linear-gradient(135deg, #0d1529 0%, #131929 50%, #0d1529 100%);
    border: 1px solid rgba(30,110,250,0.35);
    border-radius: 20px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 640px;
    overflow: hidden;
    animation: pulse-glow 2.4s ease-in-out infinite;
}

.mystery-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg,
        transparent 0%, rgba(30,110,250,0.06) 50%, transparent 100%);
    background-size: 400px 100%;
    animation: shimmer 2.2s linear infinite;
}

.mystery-q {
    font-size: 6rem;
    font-weight: 700;
    color: rgba(30,110,250,0.5);
    line-height: 1;
    margin-bottom: 1rem;
    user-select: none;
    filter: blur(1.5px);
}

.mystery-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.mystery-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 1.6rem;
}

.mystery-dots {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
}

.mystery-dots span {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: bounce-dot 1.4s ease-in-out infinite;
}

.mystery-dots span:nth-child(2) { animation-delay: 0.2s; }
.mystery-dots span:nth-child(3) { animation-delay: 0.4s; }

/* Reveal button — override Streamlit's button for this one */
.reveal-btn-wrap {
    text-align: center;
    margin: 1rem auto 0.5rem;
}

/* Winner card */
.winner-card {
    background: linear-gradient(145deg, #0a1628 0%, #0f1e3d 40%, #091224 100%);
    border: 1px solid rgba(30,110,250,0.5);
    border-radius: 20px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 680px;
    animation: reveal-entrance 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    position: relative;
    overflow: hidden;
}

.winner-card::after {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(30,110,250,0.3), transparent 40%, rgba(30,110,250,0.15));
    pointer-events: none;
    z-index: 0;
}

.winner-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    position: relative; z-index: 1;
}

.winner-name {
    font-size: 3.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.01em;
    line-height: 1;
    margin-bottom: 0.6rem;
    animation: name-glow 2.5s ease-in-out infinite;
    position: relative; z-index: 1;
}

.winner-prob {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 2rem;
    animation: count-pulse 2s ease-in-out infinite;
    position: relative; z-index: 1;
}

.runner-up-list {
    border-top: 1px solid var(--border);
    padding-top: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    position: relative; z-index: 1;
}

.runner-up-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.runner-up-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    opacity: 0;
    animation: fade-slide-up 0.5s ease forwards;
}

.runner-up-rank { color: var(--muted); font-size: 0.8rem; min-width: 1.5rem; }
.runner-up-name { font-weight: 600; font-size: 0.9rem; color: var(--text); flex: 1; text-align: left; margin-left: 0.75rem; }
.runner-up-pct  { color: var(--accent); font-size: 0.85rem; font-weight: 600; font-variant-numeric: tabular-nums; }

.lock-icon {
    font-size: 3.5rem;
    margin-bottom: 0.75rem;
    display: block;
}

</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config(_ROOT / "config.yaml")

# ── Session state ─────────────────────────────────────────────────────────────
for _step in STEPS:
    st.session_state.setdefault(f"step_{_step}_status", "not_started")
    st.session_state.setdefault(f"step_{_step}_logs", [])
st.session_state.setdefault("pipeline_data", {})
st.session_state.setdefault("hyperparams", {})
st.session_state.setdefault("reset_confirm", False)
st.session_state.setdefault("winner_revealed", False)

# ── Dashboard display data (cached) ───────────────────────────────────────────
data = load_all_dashboard_data(cfg)
sim_results = data["sim_results"]
squad_features = data["squad_features"]
elo_ratings = data["elo_ratings"]
rankings = data["rankings"]
teams = data["teams"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FIFA WC 2026")
    st.caption("Monte Carlo Prediction Engine")
    st.divider()

    if sim_results.empty:
        st.error("No predictions found. Run the pipeline first.")
    else:
        st.success(f"Loaded {len(sim_results)} teams")
        top_team = sim_results.iloc[0]
        st.metric("Predicted Winner", top_team["team_name"], f"{top_team['win_prob']*100:.1f}%")

    st.divider()

    confederations = ["All"]
    if "confederation" in rankings.columns:
        confederations += sorted(rankings["confederation"].dropna().unique().tolist())
    conf_filter = st.selectbox("Filter by Confederation", confederations)

    top_n = st.slider("Teams to display", min_value=8, max_value=48, value=16, step=4)

    st.divider()
    st.caption("Data: April 2026")
    st.caption("Model: XGBoost + Poisson GLM")
    st.caption("10,000 Monte Carlo simulations")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_win, tab_heat, tab_group, tab_team, tab_info, tab_pipe = st.tabs([
    "Win Probabilities",
    "Stage Heatmap",
    "Group Stage",
    "Team Analysis",
    "Model Info",
    "Pipeline",
])

# ── Tab 1: Win Probabilities ──────────────────────────────────────────────────
with tab_win:

    if sim_results.empty:
        st.warning("No simulation results found. Run the pipeline first.")
    else:
        # ── Winner Reveal ─────────────────────────────────────────────────────
        _winner = sim_results.iloc[0]
        _top5 = sim_results.head(5)

        if not st.session_state["winner_revealed"]:
            # Mystery card
            st.markdown("""
<div class="mystery-card">
  <span class="lock-icon">&#128274;</span>
  <div class="mystery-title">The AI has made its prediction</div>
  <div class="mystery-sub">10,000 simulated World Cup tournaments &mdash; one winner emerged</div>
  <div class="mystery-dots"><span></span><span></span><span></span></div>
</div>
""", unsafe_allow_html=True)

            _r1, _r2, _r3 = st.columns([2, 2, 2])
            with _r2:
                if st.button("Reveal the Winner", key="btn_reveal", use_container_width=True):
                    st.session_state["winner_revealed"] = True
                    st.rerun()

        else:
            # Build runner-up rows HTML
            _runner_rows = ""
            for _i, (_, _row) in enumerate(_top5.iterrows()):
                if _i == 0:
                    continue  # skip winner
                _delay = 0.15 + _i * 0.12
                _runner_rows += f"""
<div class="runner-up-item" style="animation-delay:{_delay:.2f}s">
  <span class="runner-up-rank">{_i+1}</span>
  <span class="runner-up-name">{_row['team_name']}</span>
  <span class="runner-up-pct">{_row['win_prob']*100:.1f}%</span>
</div>"""

            st.markdown(f"""
<div class="winner-card">
  <div class="winner-label">2026 FIFA World Cup &mdash; Predicted Winner</div>
  <div class="winner-name">{_winner['team_name']}</div>
  <div class="winner-prob">{_winner['win_prob']*100:.1f}% win probability</div>
  <div class="runner-up-list">
    <div class="runner-up-label">Runners-up</div>
    {_runner_rows}
  </div>
</div>
""", unsafe_allow_html=True)

            _h1, _h2, _h3 = st.columns([2, 1, 2])
            with _h2:
                if st.button("Reset Reveal", key="btn_hide_winner", use_container_width=True):
                    st.session_state["winner_revealed"] = False
                    st.rerun()

        # ── Full probability chart ─────────────────────────────────────────────
        st.divider()
        st.subheader("All Teams — Win Probability")
        render_win_probability_chart(sim_results, rankings, top_n=top_n, confederation_filter=conf_filter)

# ── Tab 2: Stage Heatmap ──────────────────────────────────────────────────────
with tab_heat:
    st.header("Stage-Reaching Probability Heatmap")
    render_stage_probability_heatmap(sim_results, top_n=top_n)

# ── Tab 3: Group Stage ────────────────────────────────────────────────────────
with tab_group:
    st.header("Group Stage Breakdown")
    if not teams.empty:
        groups = sorted(teams["group_letter"].unique())
        cols = st.columns(3)
        for i, group in enumerate(groups):
            with cols[i % 3]:
                st.write(f"**Group {group}**")
                render_group_table(group, sim_results, teams)
    else:
        st.warning("Teams data not available.")

# ── Tab 4: Team Analysis ──────────────────────────────────────────────────────
with tab_team:
    st.header("Team Analysis")
    if not sim_results.empty:
        team_list = sim_results["team_name"].tolist()
        selected_team = st.selectbox("Select a team", team_list)
        if selected_team:
            render_team_card(selected_team, sim_results, squad_features, elo_ratings, rankings)
            st.divider()
            render_team_path_flowchart(selected_team, sim_results)
    else:
        st.warning("Run the pipeline to enable team analysis.")

# ── Tab 5: Model Info ─────────────────────────────────────────────────────────
with tab_info:
    st.header("Model Information")
    st.markdown("""
### How predictions are made

**1. Elo Rating System**
Rolling Elo ratings computed from 49,215 international matches (1872–2026),
with K-factors by tournament type and margin-of-victory scaling.

**2. Feature Engineering (30 features per match)**
- Elo rating difference and ratio
- FIFA ranking points difference
- Squad market value ratio (injury-adjusted, log-transformed)
- Recent form: last 10 matches (win rate, goals scored/conceded)
- Head-to-head record (last 10 encounters)
- Squad average age and caps (experience)
- Match context: neutral venue, tournament type weight

**3. XGBoost Outcome Classifier**
Multiclass model trained on matches from 2010–2026.
Predicts P(home win), P(draw), P(away win).
World Cup matches weighted 3x in training.
Chronological train/test split (no data leakage).

**4. Poisson Goals Model**
Two separate Poisson GLMs (home goals, away goals).
Used for group stage goal difference calculations.

**5. Monte Carlo Simulation (10,000 runs)**
Full tournament bracket simulated 10,000 times.
Group stage: round-robin with Poisson goal sampling.
Knockout rounds: match outcome — extra time — penalty shootout.
Penalty model uses historical shootout win rates from 675 shootouts.

**Data Sources**
- Match history: results.csv (international-football dataset, 1872–2026)
- World Cup details: matches_1930_2022.csv
- Rankings: FIFA (April 2026)
- Squad values: Transfermarkt (April 2026)
- Player data: Transfermarkt (92k players, 143k injuries)
""")

# ── Tab 6: Pipeline ───────────────────────────────────────────────────────────
with tab_pipe:
    st.header("Pipeline Execution")
    st.caption("Run each step sequentially. Steps 1–4 must complete before training.")

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.divider()
    _rcol1, _rcol2 = st.columns([1, 5])

    with _rcol1:
        if st.button("Reset All Data", key="btn_reset_trigger"):
            st.session_state["reset_confirm"] = True

    if st.session_state["reset_confirm"]:
        st.warning("This will permanently delete all processed data and model files. This cannot be undone.")
        _cc1, _cc2, _cc3 = st.columns([1, 1, 6])
        if _cc1.button("Confirm Reset", key="btn_reset_confirm"):
            with st.spinner("Deleting files..."):
                deleted = reset_pipeline(cfg)
            for _s in STEPS:
                st.session_state[f"step_{_s}_status"] = "not_started"
                st.session_state[f"step_{_s}_logs"] = []
            st.session_state["pipeline_data"] = {}
            st.session_state["reset_confirm"] = False
            if deleted:
                st.success(f"Reset complete. Deleted: {', '.join(deleted)}")
            else:
                st.info("Reset complete. No files were found to delete.")
            st.rerun()
        if _cc2.button("Cancel", key="btn_reset_cancel"):
            st.session_state["reset_confirm"] = False
            st.rerun()

    st.divider()

    # ── Step cards ────────────────────────────────────────────────────────────
    _STATUS_BADGE = {
        "not_started": '<span class="badge badge-not-started">Not Started</span>',
        "running":     '<span class="badge badge-running">Running</span>',
        "complete":    '<span class="badge badge-complete">Complete</span>',
        "failed":      '<span class="badge badge-failed">Failed</span>',
    }

    for _step in STEPS:
        _meta = STEP_METADATA[_step]
        _status = st.session_state[f"step_{_step}_status"]
        _logs = st.session_state[f"step_{_step}_logs"]

        st.markdown(f"""
<div class="step-card">
  <div class="step-header">
    <span class="step-title">Step {_meta['num']}: {_meta['label']}</span>
    {_STATUS_BADGE[_status]}
  </div>
  <div class="step-desc">{_meta['description']}</div>
</div>
""", unsafe_allow_html=True)

        _btn_disabled = _status == "running"
        _btn_label = f"Run Step {_meta['num']}: {_meta['label']}"

        if st.button(_btn_label, key=f"run_{_step}", disabled=_btn_disabled):
            with st.status(f"Running {_meta['label']}...", expanded=True) as _status_widget:
                st.write(f"Starting Step {_meta['num']}...")
                _updated_data, _logs, _err = run_pipeline_step(
                    _step, cfg, st.session_state["pipeline_data"]
                )

            st.session_state[f"step_{_step}_logs"] = _logs
            if _err:
                st.session_state[f"step_{_step}_status"] = "failed"
                st.error(f"Step failed: {_err}")
            else:
                st.session_state[f"step_{_step}_status"] = "complete"
                st.session_state["pipeline_data"] = _updated_data
                # After simulation, refresh display caches
                if _step == "simulate":
                    st.cache_data.clear()

            st.rerun()

        if _logs:
            with st.expander("View logs", expanded=False):
                st.code("\n".join(_logs), language=None)

        st.write("")  # spacing

    # ── Hyperparameter Configuration ──────────────────────────────────────────
    st.divider()
    with st.expander("Hyperparameter Configuration", expanded=False):
        st.caption("Changes apply to the next Train Models or Simulation run. Config file is not modified.")

        _hp = st.session_state.get("hyperparams", {})

        _hc1, _hc2 = st.columns(2)

        with _hc1:
            st.markdown("**XGBoost**")
            _n_est = st.number_input(
                "n_estimators", min_value=100, max_value=2000, step=50,
                value=_hp.get("n_estimators", cfg.model.xgboost.n_estimators),
                key="hp_n_estimators",
            )
            _depth = st.slider(
                "max_depth", min_value=2, max_value=10, step=1,
                value=_hp.get("max_depth", cfg.model.xgboost.max_depth),
                key="hp_max_depth",
            )
            _lr = st.number_input(
                "learning_rate", min_value=0.001, max_value=0.5, step=0.001, format="%.3f",
                value=float(_hp.get("learning_rate", cfg.model.xgboost.learning_rate)),
                key="hp_lr",
            )
            _sub = st.slider(
                "subsample", min_value=0.5, max_value=1.0, step=0.05,
                value=float(_hp.get("subsample", cfg.model.xgboost.subsample)),
                key="hp_subsample",
            )
            _col = st.slider(
                "colsample_bytree", min_value=0.5, max_value=1.0, step=0.05,
                value=float(_hp.get("colsample_bytree", cfg.model.xgboost.colsample_bytree)),
                key="hp_colsample",
            )

        with _hc2:
            st.markdown("**Poisson GLM**")
            _alpha = st.number_input(
                "alpha (regularization)", min_value=0.0, max_value=10.0, step=0.01, format="%.2f",
                value=float(_hp.get("poisson_alpha", cfg.model.poisson.alpha)),
                key="hp_alpha",
            )

            st.markdown("**Simulation**")
            _n_sim = st.select_slider(
                "n_simulations",
                options=[1000, 2500, 5000, 10000],
                value=_hp.get("n_simulations", cfg.simulation.n_simulations),
                key="hp_n_sim",
            )

            st.markdown("**Elo & Features**")
            _home_adv = st.number_input(
                "home_advantage (Elo pts)", min_value=0, max_value=300, step=10,
                value=int(_hp.get("home_advantage", cfg.elo.home_advantage)),
                key="hp_home_adv",
            )
            _form_win = st.slider(
                "form_window (matches)", min_value=3, max_value=20, step=1,
                value=int(_hp.get("form_window", cfg.features.form_window)),
                key="hp_form_win",
            )

        _bc1, _bc2, _ = st.columns([1, 1, 4])
        if _bc1.button("Apply to Next Run", key="hp_apply"):
            st.session_state["hyperparams"] = {
                "n_estimators": int(_n_est),
                "max_depth": int(_depth),
                "learning_rate": float(_lr),
                "subsample": float(_sub),
                "colsample_bytree": float(_col),
                "poisson_alpha": float(_alpha),
                "n_simulations": int(_n_sim),
                "home_advantage": float(_home_adv),
                "form_window": int(_form_win),
            }
            st.success("Hyperparameters saved. They will be used in the next training/simulation run.")

        if _bc2.button("Reset to Defaults", key="hp_reset"):
            st.session_state["hyperparams"] = {}
            st.info("Hyperparameters reset to config.yaml defaults.")
            st.rerun()
