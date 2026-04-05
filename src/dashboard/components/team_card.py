"""Per-team deep-dive card component for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_team_card(
    team_name: str,
    sim_results: pd.DataFrame,
    squad_features: pd.DataFrame,
    elo_ratings: pd.Series,
    rankings: pd.DataFrame,
) -> None:
    """Render a detailed stats card for a single team.

    Shows win probability, stage probabilities, FIFA ranking, Elo,
    squad market value, and injury impact.

    Args:
        team_name: Canonical team name.
        sim_results: Simulation results DataFrame.
        squad_features: Squad features DataFrame (indexed by team_name).
        elo_ratings: Current Elo ratings Series.
        rankings: Normalized rankings DataFrame.
    """
    st.subheader(f"{team_name}")

    # ── Simulation probabilities ──────────────────────────────────────────────
    team_sim = sim_results[sim_results["team_name"] == team_name]
    if not team_sim.empty:
        row = team_sim.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Win Tournament", f"{row['win_prob']*100:.1f}%")
        cols[1].metric("Reach Final", f"{(row['win_prob']+row['runner_up_prob'])*100:.1f}%")
        cols[2].metric("Reach Semi", f"{(1-row['round_of_16_exit_prob']-row['round_of_32_exit_prob']-row['group_exit_prob'])*100:.1f}%")
        cols[3].metric("Exit Groups", f"{row['group_exit_prob']*100:.1f}%")

        # Stage probability funnel chart
        stages = ["Win", "Final", "3rd Place", "Semi Exit", "QF Exit", "R16 Exit", "R32 Exit", "Group Exit"]
        probs = [
            row.get("win_prob", 0),
            row.get("runner_up_prob", 0),
            row.get("third_place_prob", 0),
            row.get("semifinal_exit_prob", 0),
            row.get("quarterfinal_exit_prob", 0),
            row.get("round_of_16_exit_prob", 0),
            row.get("round_of_32_exit_prob", 0),
            row.get("group_exit_prob", 0),
        ]
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in probs],
            y=stages,
            orientation="h",
            marker_color="#003399",
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            title="Stage Probability Distribution",
            xaxis_title="Probability (%)",
            height=320,
            margin=dict(l=10, r=60, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Team ratings ──────────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns(2)

    elo = elo_ratings.get(team_name, None)
    if elo:
        c1.metric("Elo Rating", f"{elo:.0f}")

    rank_row = None
    if "team_name_canonical" in rankings.columns:
        rank_df = rankings[rankings["team_name_canonical"] == team_name]
        if not rank_df.empty:
            rank_row = rank_df.iloc[0]
    if rank_row is not None:
        c2.metric("FIFA Ranking", f"#{int(rank_row['rank'])}", f"{rank_row['total_points']:.0f} pts")

    # ── Squad features ────────────────────────────────────────────────────────
    if team_name in squad_features.index:
        sf = squad_features.loc[team_name]
        st.divider()
        st.write("**Squad Analysis**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Top 23 Value", f"€{sf.get('top23_value', 0)/1e6:.0f}M")
        sc2.metric("Injury-Adj Value", f"€{sf.get('injury_adj_value', 0)/1e6:.0f}M")
        sc3.metric("Avg Age", f"{sf.get('avg_age', 0):.1f}")
        sc4.metric("Avg Caps", f"{sf.get('avg_caps', 0):.0f}")

        inj_pct = sf.get("injury_pct_value_lost", 0)
        if inj_pct > 0:
            st.warning(f"Warning: {inj_pct*100:.1f}% of squad value is injury-affected")
