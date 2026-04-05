"""Tournament bracket visualization component."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_predicted_bracket(
    sim_results: pd.DataFrame,
    teams: pd.DataFrame,
) -> None:
    """Display the most-likely tournament bracket based on simulations.

    Shows group winners and runners-up sorted by advancement probability.

    Args:
        sim_results: Simulation results DataFrame.
        teams: Teams DataFrame with group_letter column.
    """
    if sim_results.empty:
        st.warning("Run the pipeline to see the predicted bracket.")
        return

    st.write("**Groups — Predicted Advancement Probability**")
    st.caption("Teams sorted by likelihood of advancing from group stage")

    groups = sorted(teams["group_letter"].unique())
    cols_per_row = 4
    for row_start in range(0, len(groups), cols_per_row):
        group_row = groups[row_start:row_start + cols_per_row]
        cols = st.columns(len(group_row))
        for col, g in zip(cols, group_row):
            group_teams = teams[teams["group_letter"] == g]["team_name"].tolist()
            group_sim = sim_results[sim_results["team_name"].isin(group_teams)].copy()
            group_sim["advance_prob"] = 1 - group_sim["group_exit_prob"]
            group_sim = group_sim.sort_values("advance_prob", ascending=False)
            with col:
                st.write(f"**Group {g}**")
                for _, team_row in group_sim.iterrows():
                    prob = team_row["advance_prob"] * 100
                    bar = "█" * int(prob / 10) + "░" * (10 - int(prob / 10))
                    st.write(f"`{team_row['team_name'][:12]:<12}` {prob:4.0f}%")

    st.divider()
    st.write("**Predicted Winner**")
    if not sim_results.empty:
        winner = sim_results.iloc[0]
        st.success(f"**{winner['team_name']}** — {winner['win_prob']*100:.1f}% win probability")

        top5 = sim_results.head(5)
        for _, row in top5.iterrows():
            st.write(f"- {row['team_name']}: {row['win_prob']*100:.1f}%")
