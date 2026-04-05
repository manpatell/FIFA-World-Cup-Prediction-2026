"""Win probability visualizations for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_win_probability_chart(
    sim_results: pd.DataFrame,
    rankings: pd.DataFrame,
    top_n: int = 16,
    confederation_filter: str | None = None,
) -> None:
    """Render a horizontal bar chart of top-N win probabilities.

    Args:
        sim_results: Simulation results DataFrame.
        rankings: Normalized rankings with confederation and team_name_canonical.
        top_n: Number of teams to display.
        confederation_filter: Optional confederation to filter by.
    """
    if sim_results.empty:
        st.warning("No simulation results available. Run the pipeline first.")
        return

    df = sim_results.copy()

    # Join confederation from rankings
    if "team_name_canonical" in rankings.columns:
        conf_map = rankings.set_index("team_name_canonical")["confederation"].to_dict()
    else:
        conf_map = {}
    df["confederation"] = df["team_name"].map(conf_map).fillna("Unknown")

    if confederation_filter and confederation_filter != "All":
        df = df[df["confederation"] == confederation_filter]

    df = df.nlargest(top_n, "win_prob").sort_values("win_prob")
    df["win_pct"] = (df["win_prob"] * 100).round(1)

    color_map = {
        "UEFA": "#003399",
        "CONMEBOL": "#FF6600",
        "CONCACAF": "#009900",
        "CAF": "#CC0000",
        "AFC": "#990099",
        "OFC": "#009999",
        "Unknown": "#888888",
    }

    fig = px.bar(
        df,
        x="win_pct",
        y="team_name",
        orientation="h",
        color="confederation",
        color_discrete_map=color_map,
        text="win_pct",
        labels={"win_pct": "Win Probability (%)", "team_name": ""},
        title=f"Top {top_n} Teams — Tournament Win Probability",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=max(400, top_n * 28), showlegend=True, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def render_stage_probability_heatmap(
    sim_results: pd.DataFrame,
    top_n: int = 24,
) -> None:
    """Render a heatmap of stage-reaching probabilities.

    Rows = teams (sorted by win prob), columns = tournament stages.

    Args:
        sim_results: Simulation results DataFrame.
        top_n: Number of top teams to include.
    """
    if sim_results.empty:
        st.warning("No simulation results available.")
        return

    stage_cols = [
        "win_prob", "runner_up_prob", "third_place_prob",
        "semifinal_exit_prob", "quarterfinal_exit_prob",
        "round_of_16_exit_prob", "round_of_32_exit_prob", "group_exit_prob",
    ]
    stage_labels = [
        "Champion", "Runner-Up", "3rd Place",
        "SF Exit", "QF Exit", "R16 Exit", "R32 Exit", "Group Exit",
    ]

    df = sim_results.nlargest(top_n, "win_prob")[["team_name"] + stage_cols].copy()
    df = df.sort_values("win_prob", ascending=False)

    z = df[stage_cols].values * 100  # convert to percentages

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=stage_labels,
        y=df["team_name"].tolist(),
        colorscale="Blues",
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Stage-Reaching Probability Heatmap",
        height=max(500, top_n * 22),
        xaxis_title="Stage",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, use_container_width=True)
