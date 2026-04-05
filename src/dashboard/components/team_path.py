"""Tournament path flowchart component for a single team."""

from __future__ import annotations

import streamlit.components.v1 as components
import pandas as pd
import streamlit as st


def render_team_path_flowchart(
    team_name: str,
    sim_results: pd.DataFrame,
) -> None:
    """Render an animated vertical flowchart showing a team's tournament path.

    Each stage row shows [stage box | exit indicator] side-by-side so nothing
    clips outside the iframe. Rendered as a self-contained HTML component.

    Args:
        team_name: Canonical team name.
        sim_results: Simulation results DataFrame.
    """
    team_df = sim_results[sim_results["team_name"] == team_name]
    if team_df.empty:
        st.warning(f"No simulation data for {team_name}.")
        return

    r = team_df.iloc[0]

    group_exit  = float(r.get("group_exit_prob", 0))
    r32_exit    = float(r.get("round_of_32_exit_prob", 0))
    r16_exit    = float(r.get("round_of_16_exit_prob", 0))
    qf_exit     = float(r.get("quarterfinal_exit_prob", 0))
    sf_exit     = float(r.get("semifinal_exit_prob", 0))
    third_place = float(r.get("third_place_prob", 0))
    runner_up   = float(r.get("runner_up_prob", 0))
    win         = float(r.get("win_prob", 0))

    reach_groups = 1.0
    reach_r32    = reach_groups - group_exit
    reach_r16    = reach_r32   - r32_exit
    reach_qf     = reach_r16   - r16_exit
    reach_sf     = reach_qf    - qf_exit
    reach_final  = reach_sf    - sf_exit
    reach_win    = win

    # (stage_label, reach_prob, exit_label, exit_prob, is_champion)
    stages = [
        ("Group Stage",    reach_groups, "Group Exit",  group_exit, False),
        ("Round of 32",    reach_r32,    "R32 Exit",    r32_exit,   False),
        ("Round of 16",    reach_r16,    "R16 Exit",    r16_exit,   False),
        ("Quarter-finals", reach_qf,     "QF Exit",     qf_exit,    False),
        ("Semi-finals",    reach_sf,     "SF Exit",     sf_exit,    False),
        ("Final",          reach_final,  "Runner-Up",   runner_up,  False),
        ("Champion",       reach_win,    "",            0,          True),
    ]

    def bar_color(p: float, champ: bool) -> str:
        if champ:
            return "#f5c518"
        if p >= 0.5:
            return "#1e6efa"
        if p >= 0.25:
            return "#1452c8"
        if p >= 0.10:
            return "#0f3a8a"
        return "#1e2d4a"

    rows_html = ""
    advance_seq = [reach_r32, reach_r16, reach_qf, reach_sf, reach_final, reach_win]

    for i, (label, reach, exit_label, exit_prob, is_champ) in enumerate(stages):
        delay      = 0.05 + i * 0.12
        bar_col    = bar_color(reach, is_champ)
        border_col = "#f5c518" if is_champ else "#1e6efa"
        bg         = "linear-gradient(135deg,#1c1500,#2a2000)" if is_champ else "linear-gradient(135deg,#0d1529,#131929)"
        name_col   = "#f5c518" if is_champ else "#a0b0d0"
        prob_col   = "#f5c518" if is_champ else "#1e6efa"
        glow       = "0 0 20px 6px rgba(245,197,24,0.30), 0 0 40px 12px rgba(245,197,24,0.15)" if is_champ else "none"
        bar_w      = max(2, int(reach * 100))

        # Exit side box — only if probability is meaningful
        if exit_prob > 0.005 and exit_label:
            exit_pct = f"{exit_prob * 100:.1f}%"
            exit_box = f"""
            <div style="
                display:flex; align-items:center; gap:10px;
                opacity:0; animation:fade-right 0.4s ease {delay+0.25:.2f}s forwards;
                flex-shrink:0;
            ">
              <!-- connector -->
              <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
                <div style="width:28px;height:2px;background:linear-gradient(90deg,rgba(239,68,68,0.15),rgba(239,68,68,0.5))"></div>
              </div>
              <!-- box -->
              <div style="
                background:#1a0d0d; border:1px solid rgba(239,68,68,0.35);
                border-radius:10px; padding:8px 14px; text-align:center; min-width:108px;
              ">
                <div style="font-size:0.6rem;color:#6b7a99;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:3px">{exit_label}</div>
                <div style="font-size:1.15rem;font-weight:800;color:#ef4444;line-height:1">{exit_pct}</div>
              </div>
            </div>"""
        else:
            exit_box = '<div style="width:160px;flex-shrink:0"></div>'

        # Stage node
        stage_box = f"""
        <div style="
            background:{bg};
            border:1.5px solid {border_col};
            border-radius:14px; padding:14px 20px;
            box-shadow:{glow};
            width:260px; flex-shrink:0;
            opacity:0; animation:slide-up 0.45s cubic-bezier(0.34,1.56,0.64,1) {delay:.2f}s forwards;
        ">
          <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                      text-transform:uppercase;color:{name_col};margin-bottom:6px">
            {label}
          </div>
          <div style="font-size:2.1rem;font-weight:800;color:{prob_col};line-height:1;margin-bottom:10px">
            {reach * 100:.1f}%
          </div>
          <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;overflow:hidden">
            <div style="
                height:100%; width:0; border-radius:4px;
                background:{bar_col};
                animation:grow-bar 0.6s ease {delay+0.15:.2f}s forwards;
                --w:{bar_w}%;
            "></div>
          </div>
        </div>"""

        rows_html += f"""
        <div style="display:flex;align-items:center;justify-content:center;gap:0;width:100%">
          {stage_box}
          {exit_box}
        </div>"""

        # Arrow connector between stages
        if i < len(stages) - 1:
            adv      = advance_seq[i]
            adv_pct  = f"{adv * 100:.1f}%"
            arr_delay = delay + 0.08
            rows_html += f"""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            padding-left:0; margin-left:0;
            opacity:0; animation:fade-in 0.3s ease {arr_delay:.2f}s forwards;
        ">
          <!-- left-aligned connector to match stage box -->
          <div style="display:flex; align-items:center; margin-left:-148px; gap:0; flex-direction:column">
            <div style="width:2px;height:14px;background:linear-gradient(180deg,{border_col},rgba(30,110,250,0.3))"></div>
            <div style="
                font-size:0.65rem;font-weight:600;color:#1e6efa;
                background:#0e1828;padding:2px 10px;border-radius:4px;
                border:1px solid rgba(30,110,250,0.25);
                white-space:nowrap;
            ">{adv_pct} advance</div>
            <div style="width:2px;height:14px;background:linear-gradient(180deg,rgba(30,110,250,0.3),#1e6efa)"></div>
            <div style="
                width:0;height:0;
                border-left:7px solid transparent;
                border-right:7px solid transparent;
                border-top:9px solid #1e6efa;
            "></div>
          </div>
        </div>"""

    third_note = ""
    if third_place > 0.005:
        third_note = f"""
        <div style="
            margin-top:16px; text-align:center;
            background:#131929; border:1px dashed rgba(245,158,11,0.3);
            border-radius:10px; padding:10px 20px;
            font-size:0.78rem; color:#6b7a99; max-width:440px; margin-inline:auto;
        ">
          3rd Place finish probability:
          <span style="color:#f59e0b;font-weight:700"> {third_place * 100:.1f}%</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{
  background: #0a0e1a;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  padding: 20px 12px 28px;
  color: #e8eaf0;
}}
@keyframes slide-up {{
  from {{ opacity:0; transform:translateY(22px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fade-in {{
  from {{ opacity:0; }}
  to   {{ opacity:1; }}
}}
@keyframes fade-right {{
  from {{ opacity:0; transform:translateX(-10px); }}
  to   {{ opacity:1; transform:translateX(0); }}
}}
@keyframes grow-bar {{
  from {{ width:0; }}
  to   {{ width:var(--w); }}
}}
</style>
</head>
<body>

<div style="text-align:center; margin-bottom:20px; opacity:0; animation:fade-in 0.3s ease 0s forwards">
  <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.15em;
              text-transform:uppercase;color:#6b7a99;margin-bottom:4px">
    Tournament Path
  </div>
  <div style="font-size:1.3rem;font-weight:800;color:#e8eaf0">{team_name}</div>
  <div style="font-size:0.72rem;color:#6b7a99;margin-top:3px">
    Based on 10,000 simulated tournaments &nbsp;|&nbsp;
    <span style="color:#1e6efa">blue bar</span> = reach probability &nbsp;|&nbsp;
    <span style="color:#ef4444">red box</span> = eliminated here
  </div>
</div>

<div style="display:flex;flex-direction:column;align-items:center;gap:0;width:100%">
  {rows_html}
</div>

{third_note}

</body>
</html>"""

    # Each stage: ~100px node + ~58px arrow gap; header ~100px; third note ~50px
    n = len(stages)
    height = 110 + n * 100 + (n - 1) * 58 + (50 if third_place > 0.005 else 0)
    components.html(html, height=height, scrolling=True)
