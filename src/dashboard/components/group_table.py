"""Group stage table component for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_group_table(
    group_letter: str,
    sim_results: pd.DataFrame,
    teams: pd.DataFrame,
) -> None:
    """Display average simulated group standings for one group.

    Args:
        group_letter: Single letter identifying the group (e.g. 'A').
        sim_results: Simulation results DataFrame.
        teams: Teams DataFrame with group_letter column.
    """
    if sim_results.empty:
        st.warning("Run the pipeline to see group tables.")
        return

    group_teams = teams[teams["group_letter"] == group_letter]["team_name"].tolist()
    if not group_teams:
        st.write(f"No teams found for Group {group_letter}.")
        return

    group_sim = sim_results[sim_results["team_name"].isin(group_teams)].copy()
    if group_sim.empty:
        st.write(f"No simulation data for Group {group_letter}.")
        return

    group_sim = group_sim.sort_values("group_exit_prob").reset_index(drop=True)
    group_sim["advance_prob"] = (
        (1 - group_sim["group_exit_prob"]) * 100
    ).round(1)
    group_sim["win_pct"] = (group_sim["win_prob"] * 100).round(2)

    display = group_sim[["team_name", "advance_prob", "win_pct"]].rename(columns={
        "team_name": "Team",
        "advance_prob": "Advance % (sim)",
        "win_pct": "Win Tournament %",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)
