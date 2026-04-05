"""Tournament bracket viewer component.

Derives the most-likely 2026 bracket using XGBoost + Poisson ML predictions,
showing predicted scores and win probabilities for every matchup from R32 to Final.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd


# R32 seeding labels from src/simulation/bracket.py (duplicated to avoid circular import)
_R32_MATCH_LABELS = [
    "2A vs 2B",
    "1C vs 2F",
    "1E vs 3ABCDF",
    "1F vs 2C",
    "2E vs 2I",
    "1I vs 3CDFGH",
    "1A vs 3CEFHI",
    "1L vs 3EHIJK",
    "1G vs 3AEHIJ",
    "1D vs 3BEFIJ",
    "1H vs 2J",
    "2K vs 2L",
    "1B vs 3EFGIJ",
    "2D vs 2G",
    "1J vs 2H",
    "1K vs 3DEIJL",
]

_R32_TO_R16_PAIRS = [(i * 2, i * 2 + 1) for i in range(8)]
_R16_TO_QF_PAIRS  = [(i * 2, i * 2 + 1) for i in range(4)]
_QF_TO_SF_PAIRS   = [(0, 1), (2, 3)]
_SF_TO_FINAL      = [(0, 1)]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Elo-based win probability for team A (fallback when ML prediction missing)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _infer_group_finishers(
    sim_results: pd.DataFrame,
    teams: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Infer most-likely 1st, 2nd, 3rd place finishers per group.

    Args:
        sim_results: Simulation results with group_exit_prob, win_prob.
        teams: Teams DataFrame with team_name, group_letter.

    Returns:
        Tuple of (firsts, seconds, thirds) dicts mapping group_letter → team_name.
    """
    firsts: dict[str, str]  = {}
    seconds: dict[str, str] = {}
    thirds: dict[str, str]  = {}

    merged = teams[["team_name", "group_letter"]].merge(
        sim_results[["team_name", "group_exit_prob", "win_prob"]],
        on="team_name", how="left",
    )
    merged["group_exit_prob"] = merged["group_exit_prob"].fillna(1.0)
    merged["win_prob"]        = merged["win_prob"].fillna(0.0)

    for grp, grp_df in merged.groupby("group_letter"):
        ranked = grp_df.sort_values(["group_exit_prob", "win_prob"], ascending=[True, False])
        names  = ranked["team_name"].tolist()
        if len(names) >= 1:
            firsts[grp]  = names[0]
        if len(names) >= 2:
            seconds[grp] = names[1]
        if len(names) >= 3:
            thirds[grp]  = names[2]

    return firsts, seconds, thirds


def _select_third_place_qualifiers(
    thirds: dict[str, str],
    sim_results: pd.DataFrame,
    n: int = 8,
) -> list[str]:
    """Pick the n best third-place teams by group_exit_prob (ascending).

    Args:
        thirds: Mapping group_letter → third-place team name.
        sim_results: Simulation results for probability lookup.
        n: Number of qualifiers (8 in 2026 format).

    Returns:
        List of n team names, ordered best-first.
    """
    probs = sim_results.set_index("team_name")["group_exit_prob"].to_dict()
    ranked = sorted(thirds.values(), key=lambda t: probs.get(t, 1.0))
    return ranked[:n]


def _resolve_slot(
    token: str,
    firsts: dict[str, str],
    seconds: dict[str, str],
    third_qualifiers: list[str],
    third_by_group: dict[str, str],
) -> str:
    """Resolve a bracket slot token to a team name.

    Args:
        token: Slot descriptor, e.g. "1A", "2K", "3ABCDF".
        firsts: Group winners dict.
        seconds: Group runners-up dict.
        third_qualifiers: Ordered list of 8 qualifying third-place teams.
        third_by_group: Mapping group_letter → third-place team name.

    Returns:
        Team name, or "TBD" if not resolvable.
    """
    token = token.strip()
    if token.startswith("1"):
        return firsts.get(token[1:], "TBD")
    if token.startswith("2"):
        return seconds.get(token[1:], "TBD")
    if token.startswith("3"):
        pool_groups = set(token[1:])
        for team in third_qualifiers:
            for g, t in third_by_group.items():
                if t == team and g in pool_groups:
                    return team
    return "TBD"


def _build_bracket(
    sim_results: pd.DataFrame,
    teams: pd.DataFrame,
    elo_ratings: pd.Series,
    match_predictions: dict,
) -> dict:
    """Compute the most-likely tournament bracket using ML predictions.

    Uses XGBoost + Poisson model predictions for win probabilities and
    expected goals. Falls back to Elo formula for missing matchups.

    Args:
        sim_results: Monte Carlo simulation results (48 teams × stage probs).
        teams: Teams DataFrame (team_name, group_letter).
        elo_ratings: Current Elo ratings Series indexed by team_name.
        match_predictions: Dict mapping (home, away) →
            (p_home_win, p_draw, p_away_win, lambda_home, lambda_away).

    Returns:
        Dict with keys r32, r16, qf, sf, final. Each value is a list of
        match dicts: {team_a, team_b, win_prob_a, score_a, score_b, label}.
    """
    firsts, seconds, thirds = _infer_group_finishers(sim_results, teams)
    third_qualifiers = _select_third_place_qualifiers(thirds, sim_results, n=8)

    # Overall simulation win probability — used to decide who advances
    sim_wp = sim_results.set_index("team_name")["win_prob"].to_dict()

    def get_elo(name: str) -> float:
        return float(elo_ratings.get(name, 1500.0))

    def make_match(a: str, b: str, label: str = "") -> dict:
        """Build a match dict using ML predictions where available."""
        score_a: int | None = None
        score_b: int | None = None

        if (a, b) in match_predictions:
            p_hw, p_d, p_aw, lam_h, lam_a = match_predictions[(a, b)]
            # Knockout: draw → extra time/penalties (each side equally likely)
            wa = float(p_hw) + float(p_d) * 0.5
            score_a = int(round(float(lam_h)))
            score_b = int(round(float(lam_a)))
        elif (b, a) in match_predictions:
            # Lookup has b as home, a as away — swap perspective
            p_hw, p_d, p_aw, lam_h, lam_a = match_predictions[(b, a)]
            wa = float(p_aw) + float(p_d) * 0.5
            score_a = int(round(float(lam_a)))  # a was away
            score_b = int(round(float(lam_h)))  # b was home
        else:
            wa = _elo_win_prob(get_elo(a), get_elo(b))

        # Advancement is decided by overall simulation win_prob (consistent with Win Probabilities tab)
        sim_winner = a if sim_wp.get(a, 0) >= sim_wp.get(b, 0) else b

        # Ensure displayed score shows the sim_winner winning
        if score_a is not None and score_b is not None:
            if sim_winner == a and score_a <= score_b:
                score_a = score_b + 1
            elif sim_winner == b and score_b <= score_a:
                score_b = score_a + 1

        return {
            "team_a": a,
            "team_b": b,
            "win_prob_a": wa,
            "sim_winner": sim_winner,
            "score_a": score_a,
            "score_b": score_b,
            "label": label,
        }

    # ── R32 ───────────────────────────────────────────────────────────────────
    r32_matches: list[dict] = []
    used_thirds: set[str] = set()

    def resolve_third(tok: str) -> str:
        """Find best unused 3rd-place qualifier from the given pool groups.

        Falls back to best unused qualifier from any group if pool is exhausted.
        """
        pool = set(tok[1:])
        for t in third_qualifiers:
            if t not in used_thirds:
                for g, tm in thirds.items():
                    if tm == t and g in pool:
                        used_thirds.add(t)
                        return t
        for t in third_qualifiers:
            if t not in used_thirds:
                used_thirds.add(t)
                return t
        return "Unknown"

    for lbl in _R32_MATCH_LABELS:
        left_tok, right_tok = [t.strip() for t in lbl.split(" vs ")]

        def resolve(tok: str) -> str:
            if tok.startswith("3"):
                return resolve_third(tok)
            return _resolve_slot(tok, firsts, seconds, third_qualifiers, thirds)

        team_a = resolve(left_tok)
        team_b = resolve(right_tok)
        r32_matches.append(make_match(team_a, team_b, lbl))

    # ── Propagate rounds ──────────────────────────────────────────────────────
    def propagate(prev_matches: list[dict], pairs: list[tuple[int, int]]) -> list[dict]:
        next_round = []
        for i, j in pairs:
            m1, m2 = prev_matches[i], prev_matches[j]
            winner1 = m1["sim_winner"]
            winner2 = m2["sim_winner"]
            next_round.append(make_match(winner1, winner2))
        return next_round

    r16_matches = propagate(r32_matches, _R32_TO_R16_PAIRS)
    qf_matches  = propagate(r16_matches, _R16_TO_QF_PAIRS)
    sf_matches  = propagate(qf_matches,  _QF_TO_SF_PAIRS)
    final_match = propagate(sf_matches,  _SF_TO_FINAL)

    return {
        "r32":   r32_matches,
        "r16":   r16_matches,
        "qf":    qf_matches,
        "sf":    sf_matches,
        "final": final_match,
    }


# ── HTML rendering ────────────────────────────────────────────────────────────

def render_bracket_viewer(
    sim_results: pd.DataFrame,
    teams: pd.DataFrame,
    elo_ratings: pd.Series,
    match_predictions: dict | None = None,
    revealed: bool = True,
) -> None:
    """Render the full 2026 tournament bracket as an animated HTML component.

    Uses ML model predictions (XGBoost + Poisson) for each matchup to show
    win probabilities and predicted scorelines. Falls back to Elo for any
    matchup not covered by the prediction cache.

    Vertical layout: R32 (16 matches) at top → R16 → QF → SF → Final at bottom.
    SVG lines connect each pair of matches to their next-round result.

    Args:
        sim_results: Simulation results DataFrame.
        teams: Teams DataFrame with group_letter column.
        elo_ratings: Current Elo ratings Series (fallback when no ML prediction).
        match_predictions: Precomputed ML predictions dict mapping
            (home_team, away_team) → (p_home_win, p_draw, p_away_win, lam_h, lam_a).
            Pass None or empty dict to use Elo-only mode.
        revealed: If False, the Final match card and champion banner are hidden
            and replaced with a mystery placeholder.
    """
    if sim_results.empty or teams.empty:
        st.warning("No data available. Run the pipeline first.")
        return

    predictions = match_predictions or {}
    bracket = _build_bracket(sim_results, teams, elo_ratings, predictions)

    using_ml = len(predictions) > 0
    if using_ml:
        st.caption(
            f"Bracket built using XGBoost + Poisson ML predictions "
            f"({len(predictions):,} matchup predictions loaded). "
            "Score shown is predicted scoreline; probability is model win probability."
        )
    else:
        st.caption(
            "ML models not loaded — showing Elo-based win probabilities. "
            "Run the full pipeline (through Training) to enable ML predictions."
        )

    rounds = [
        ("Round of 32",    bracket["r32"],   "#6b7a99"),
        ("Round of 16",    bracket["r16"],   "#6b7a99"),
        ("Quarter-finals", bracket["qf"],    "#6b7a99"),
        ("Semi-finals",    bracket["sf"],    "#6b7a99"),
        ("Final",          bracket["final"], "#f5c518"),
    ]

    # ── Geometry ──────────────────────────────────────────────────────────────
    CARD_W  = 140   # px — wide enough for score display
    CARD_H  = 76    # px — taller to show score row
    H_GAP   = 8     # px — horizontal gap between cards
    ROW_GAP = 56    # px — vertical space between rounds (for connectors)
    LABEL_H = 22    # px — round label height
    PAD     = 20    # px — outer padding

    n_r32   = 16
    total_w = PAD * 2 + n_r32 * CARD_W + (n_r32 - 1) * H_GAP
    total_h = PAD * 2 + len(rounds) * (LABEL_H + CARD_H) + (len(rounds) - 1) * ROW_GAP + 90

    # Card center x per round (each subsequent round centered between source pair)
    def r32_centers() -> list[float]:
        return [PAD + i * (CARD_W + H_GAP) + CARD_W / 2 for i in range(n_r32)]

    all_cx: list[list[float]] = [r32_centers()]
    for _ in range(len(rounds) - 1):
        prev = all_cx[-1]
        all_cx.append([(prev[i * 2] + prev[i * 2 + 1]) / 2 for i in range(len(prev) // 2)])

    row_y: list[float] = []
    y = float(PAD)
    for _ in rounds:
        row_y.append(y)
        y += LABEL_H + CARD_H + ROW_GAP

    # ── Build cards HTML ──────────────────────────────────────────────────────
    cards_html = ""
    for row_idx, (round_name, matches, label_col) in enumerate(rounds):
        is_final = round_name == "Final"
        cx_list  = all_cx[row_idx]
        card_top = row_y[row_idx] + LABEL_H

        # Round label — centered above the row
        row_label_cx = (cx_list[0] + cx_list[-1]) / 2
        cards_html += f"""
<div style="
    position:absolute;
    left:{row_label_cx - 80:.0f}px; top:{row_y[row_idx]:.0f}px;
    width:160px; text-align:center;
    font-size:0.6rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:{label_col};
">
  {round_name}
</div>"""

        for match_idx, match in enumerate(matches):
            card_left = cx_list[match_idx] - CARD_W / 2
            delay     = 0.05 + row_idx * 0.15 + match_idx * 0.025
            card_id   = f"c{row_idx}_{match_idx}"

            # Final card: show mystery placeholder when not revealed
            if is_final and not revealed:
                cards_html += f"""
<div style="
    position:absolute;
    left:{card_left:.0f}px; top:{card_top:.0f}px;
    width:{CARD_W}px; height:{CARD_H}px;
    background:linear-gradient(135deg,#0d1020,#141b2e);
    border:1px dashed rgba(245,197,24,0.4); border-radius:7px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px;
    animation:mystery-pulse 2s ease-in-out infinite;
">
  <div style="font-size:1.1rem;color:rgba(245,197,24,0.5)">&#128274;</div>
  <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
              text-transform:uppercase;color:rgba(245,197,24,0.4)">Final</div>
</div>"""
                continue

            a  = match["team_a"]
            b  = match["team_b"]
            wa = match["win_prob_a"]
            wb = 1.0 - wa
            sa = match.get("score_a")
            sb = match.get("score_b")
            fav_a = match.get("sim_winner", a) == a

            has_score = sa is not None and sb is not None

            def shorten(name: str, limit: int = 14) -> str:
                return name[:limit] if len(name) > limit else name

            def team_row(
                name: str, prob: float, score: int | None,
                is_fav: bool, is_bottom: bool,
            ) -> str:
                nc  = "#e8eaf0" if is_fav else "#6b7a99"
                pc  = "#1e6efa" if is_fav else "#3a4a60"
                fw  = "700"     if is_fav else "400"
                sc_col = "#00c16e" if is_fav else "#4a5568"
                mb  = "0" if is_bottom else "2px"
                score_html = (
                    f'<div style="font-size:0.85rem;font-weight:800;color:{sc_col};'
                    f'min-width:16px;text-align:center">{score}</div>'
                    if score is not None else ""
                )
                return f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:4px 0;gap:4px;margin-bottom:{mb}">
  <div style="flex:1;font-size:0.67rem;font-weight:{fw};color:{nc};
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{shorten(name)}</div>
  {score_html}
  <div style="font-size:0.65rem;font-weight:700;color:{pc};
              min-width:28px;text-align:right">{prob*100:.0f}%</div>
</div>"""

            brd  = "#f5c518" if is_final else "#1e2d4a"
            bg   = "linear-gradient(135deg,#1c1500,#281e00)" if is_final else "#111827"
            glow = "0 0 14px 3px rgba(245,197,24,0.3)" if is_final else "none"

            # Score display between teams
            if has_score:
                score_sep = f"""
<div style="text-align:center;font-size:0.62rem;color:#4a5568;
            letter-spacing:0.06em;margin:1px 0;line-height:1">
  ML Predicted Score
</div>"""
                divider = ""
            else:
                score_sep = ""
                divider = '<div style="height:1px;background:#1a2235;margin:2px 0"></div>'

            cards_html += f"""
<div id="{card_id}" style="
    position:absolute;
    left:{card_left:.0f}px; top:{card_top:.0f}px;
    width:{CARD_W}px;
    background:{bg}; border:1px solid {brd}; border-radius:7px;
    padding:7px 9px; box-shadow:{glow};
    opacity:0; animation:slide-down 0.4s ease {delay:.2f}s forwards;
">
  {team_row(a, wa, sa, fav_a, False)}
  {divider}
  {score_sep}
  {team_row(b, wb, sb, not fav_a, True)}
</div>"""

    # ── SVG connectors ────────────────────────────────────────────────────────
    svg = '<svg style="position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none">'

    for row_idx in range(len(rounds) - 1):
        cx_cur     = all_cx[row_idx]
        cx_next    = all_cx[row_idx + 1]
        card_bot_y = row_y[row_idx] + LABEL_H + CARD_H
        card_top_y = row_y[row_idx + 1] + LABEL_H
        mid_y      = (card_bot_y + card_top_y) / 2

        for nxt_idx in range(len(cx_next)):
            xa = cx_cur[nxt_idx * 2]
            xb = cx_cur[nxt_idx * 2 + 1]
            xc = cx_next[nxt_idx]
            svg += f"""
  <line x1="{xa:.1f}" y1="{card_bot_y}" x2="{xa:.1f}" y2="{mid_y:.1f}"
        stroke="#1e2d4a" stroke-width="1.5"/>
  <line x1="{xb:.1f}" y1="{card_bot_y}" x2="{xb:.1f}" y2="{mid_y:.1f}"
        stroke="#1e2d4a" stroke-width="1.5"/>
  <line x1="{xa:.1f}" y1="{mid_y:.1f}" x2="{xb:.1f}" y2="{mid_y:.1f}"
        stroke="#1e2d4a" stroke-width="1.5"/>
  <line x1="{xc:.1f}" y1="{mid_y:.1f}" x2="{xc:.1f}" y2="{card_top_y:.1f}"
        stroke="#1e6efa" stroke-width="1.5" opacity="0.55"/>"""

    svg += "</svg>"

    # ── Champion banner ────────────────────────────────────────────────────────
    final_m  = bracket["final"][0]
    champion = final_m.get("sim_winner", final_m["team_a"])
    champ_sa = final_m.get("score_a")
    champ_sb = final_m.get("score_b")
    champ_top = row_y[-1] + LABEL_H + CARD_H + 14
    champ_cx  = total_w / 2

    if revealed:
        score_line = f"{champ_sa} – {champ_sb}" if champ_sa is not None else ""
        score_html = (
            f'<div style="font-size:0.85rem;color:rgba(245,197,24,0.7);margin-top:3px">'
            f'Predicted Final: {score_line}</div>'
            if score_line else ""
        )
        champ_html = f"""
<div style="
    position:absolute;
    left:{champ_cx - 120:.0f}px; top:{champ_top:.0f}px;
    width:240px; text-align:center;
    background:linear-gradient(135deg,#1c1500,#2a2000);
    border:1.5px solid #f5c518; border-radius:12px;
    padding:12px 20px;
    animation:champ-pulse 2.2s ease-in-out infinite;
">
  <div style="font-size:0.58rem;color:#f5c518;letter-spacing:0.12em;
              text-transform:uppercase;margin-bottom:6px">2026 World Cup Champion</div>
  <div style="font-size:1.2rem;font-weight:800;color:#f5c518">{champion}</div>
  {score_html}
</div>"""
    else:
        champ_html = f"""
<div style="
    position:absolute;
    left:{champ_cx - 130:.0f}px; top:{champ_top:.0f}px;
    width:260px; text-align:center;
    background:linear-gradient(135deg,#0d1020,#141b2e);
    border:1px dashed rgba(245,197,24,0.35); border-radius:12px;
    padding:12px 20px;
    animation:mystery-pulse 2s ease-in-out infinite;
">
  <div style="font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;
              color:rgba(245,197,24,0.4);margin-bottom:6px">2026 World Cup Champion</div>
  <div style="font-size:1.4rem;color:rgba(245,197,24,0.25)">? ? ?</div>
</div>"""

    total_h += 80
    TOOLBAR_H = 48  # zoom toolbar height

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
html, body {{
  background:#0a0e1a;
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  color:#e8eaf0;
  overflow:hidden;
}}

/* ── Zoom toolbar ── */
#zoom-bar {{
  position:sticky; top:0; z-index:200;
  display:flex; align-items:center; gap:10px;
  padding:8px 16px;
  background:rgba(10,14,26,0.92);
  border-bottom:1px solid #1e2d4a;
  backdrop-filter:blur(8px);
}}
#zoom-bar button {{
  background:#131929; border:1px solid #1e2d4a;
  color:#e8eaf0; border-radius:6px;
  width:32px; height:32px; font-size:1.1rem;
  cursor:pointer; display:flex; align-items:center; justify-content:center;
  transition:background 0.15s, border-color 0.15s;
  flex-shrink:0;
}}
#zoom-bar button:hover {{ background:#1e2d4a; border-color:#1e6efa; }}
#zoom-label {{
  font-size:0.75rem; font-weight:700; color:#6b7a99;
  min-width:40px; text-align:center; letter-spacing:0.06em;
}}
#reset-btn {{
  font-size:0.65rem !important; width:auto !important;
  padding:0 10px; letter-spacing:0.08em; text-transform:uppercase;
  color:#6b7a99 !important;
}}
#reset-btn:hover {{ color:#e8eaf0 !important; }}

/* ── Scrollable bracket canvas ── */
#scroll-outer {{
  overflow:auto;
  width:100%;
  height:calc(100vh - {TOOLBAR_H}px);
}}
#bracket-wrap {{
  transform-origin: top left;
  transition: transform 0.18s cubic-bezier(0.25,0.8,0.25,1);
  display:inline-block;
  padding:0;
}}

@keyframes slide-down {{
  from {{ opacity:0; transform:translateY(-14px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes grow-bar {{
  from {{ width:0; }}
  to   {{ width:var(--w); }}
}}
@keyframes champ-pulse {{
  0%,100% {{ box-shadow:0 0 14px 4px rgba(245,197,24,0.25); }}
  50%      {{ box-shadow:0 0 32px 10px rgba(245,197,24,0.55); }}
}}
@keyframes mystery-pulse {{
  0%,100% {{ border-color:rgba(245,197,24,0.2); box-shadow:none; }}
  50%      {{ border-color:rgba(245,197,24,0.5); box-shadow:0 0 18px 4px rgba(245,197,24,0.12); }}
}}
</style>
</head>
<body>

<!-- Zoom toolbar -->
<div id="zoom-bar">
  <button onclick="adjustZoom(-0.15)" title="Zoom out">&#8722;</button>
  <span id="zoom-label">100%</span>
  <button onclick="adjustZoom(0.15)" title="Zoom in">&#43;</button>
  <button id="reset-btn" onclick="resetZoom()" title="Reset zoom">Reset</button>
  <span style="font-size:0.62rem;color:#3a4a60;margin-left:4px">
    scroll to pan &nbsp;·&nbsp; use +/&#8722; to zoom
  </span>
</div>

<!-- Scrollable outer container -->
<div id="scroll-outer">
  <div id="bracket-wrap">
    <div style="position:relative;width:{total_w}px;height:{total_h}px">
      {svg}
      {cards_html}
      {champ_html}
    </div>
  </div>
</div>

<script>
var scale = 1.0;
var MIN_SCALE = 0.3;
var MAX_SCALE = 2.0;
var wrap = document.getElementById('bracket-wrap');
var label = document.getElementById('zoom-label');

function applyZoom() {{
  wrap.style.transform = 'scale(' + scale + ')';
  // Keep the scrollable area sized to the scaled content
  wrap.style.width  = ({total_w} * scale) + 'px';
  wrap.style.height = ({total_h} * scale) + 'px';
  label.textContent = Math.round(scale * 100) + '%';
}}

function adjustZoom(delta) {{
  scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale + delta));
  applyZoom();
}}

function resetZoom() {{
  scale = 1.0;
  applyZoom();
  document.getElementById('scroll-outer').scrollTo(0, 0);
}}

// Ctrl+Scroll / pinch-zoom support
document.getElementById('scroll-outer').addEventListener('wheel', function(e) {{
  if (e.ctrlKey || e.metaKey) {{
    e.preventDefault();
    adjustZoom(e.deltaY < 0 ? 0.1 : -0.1);
  }}
}}, {{ passive: false }});
</script>

</body>
</html>"""

    components.html(html, height=total_h + TOOLBAR_H + 10, scrolling=False)
