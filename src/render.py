"""HTML/CSS card components for the dashboard, built on the validated dark-mode
reference palette (see dataviz skill: references/palette.md). Fixed roles, not
per-team colors: Home is always the blue slot, Away is always the aqua slot,
Draw is neutral gray — that's a stable visual language across every fixture,
not an attempt to assign identity color per team (there are too many teams for
that to mean anything).

Status colors (good/warning/critical) are reserved for state and always paired
with a text label, never color alone.
"""
import html

# --- palette (dark surface, from the validated reference instance) ---
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
BORDER = "rgba(255,255,255,0.10)"

HOME_COLOR = "#3987e5"  # categorical slot 1 (blue)
AWAY_COLOR = "#199e70"  # categorical slot 2 (aqua)
DRAW_COLOR = "#5b5a54"  # neutral, distinct from both

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

SPORT_COLORS = {
    "Soccer": "#3987e5",       # blue
    "Basketball": "#199e70",   # aqua
    "Baseball": "#c98500",     # yellow (dark-stepped)
    "Cricket": "#008300",      # green
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def inject_css() -> str:
    return f"""
<style>
.fixture-card {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 10px;
}}
.fixture-card.live {{
  border-left: 3px solid {STATUS_CRITICAL};
}}
.fixture-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}}
.fixture-teams {{
  font-size: 1.05rem;
  font-weight: 600;
  color: {INK_PRIMARY};
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 9px;
  border-radius: 999px;
  color: {INK_PRIMARY};
  white-space: nowrap;
}}
.badge-dot {{
  width: 6px; height: 6px; border-radius: 50%; display: inline-block;
}}
.badge-live {{
  background: rgba(208,59,59,0.18); color: #ff8a8a;
}}
.badge-live .badge-dot {{ background: {STATUS_CRITICAL}; animation: pulse 1.4s infinite; }}
@keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.35}} 100%{{opacity:1}} }}
.badge-league {{ background: rgba(255,255,255,0.08); color: {INK_SECONDARY}; }}
.badge-good {{ background: rgba(12,163,12,0.18); color: #7fe07f; }}
.badge-warning {{ background: rgba(250,178,25,0.18); color: #ffcf6b; }}
.badge-neutral {{ background: rgba(57,135,229,0.16); color: #8fbdf2; }}
.meta-line {{
  font-size: 0.78rem; color: {INK_MUTED}; margin-top: 2px;
}}
.prob-bar {{
  display: flex; height: 22px; border-radius: 5px; overflow: hidden;
  margin: 10px 0 4px 0; background: {PAGE};
}}
.prob-seg {{
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 600; color: {INK_PRIMARY};
  min-width: 0; overflow: hidden; white-space: nowrap;
}}
.prob-legend {{
  display: flex; gap: 14px; font-size: 0.74rem; color: {INK_SECONDARY};
  margin-bottom: 8px;
}}
.prob-legend-item {{ display: flex; align-items: center; gap: 5px; }}
.prob-swatch {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; }}
.pick-line {{
  font-size: 0.86rem; color: {INK_SECONDARY}; margin-top: 6px;
}}
.pick-line b {{ color: {INK_PRIMARY}; }}
.stat-tile-row {{ display: flex; gap: 10px; margin: 6px 0 18px 0; flex-wrap: wrap; }}
.stat-tile {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
  padding: 10px 16px; min-width: 120px;
}}
.stat-tile-label {{ font-size: 0.72rem; color: {INK_MUTED}; }}
.stat-tile-value {{ font-size: 1.5rem; font-weight: 700; color: {INK_PRIMARY}; }}
</style>
"""


def stat_tiles(items: list) -> str:
    """items: list of (label, value) tuples."""
    tiles = "".join(
        f'<div class="stat-tile"><div class="stat-tile-label">{_esc(l)}</div>'
        f'<div class="stat-tile-value">{_esc(v)}</div></div>'
        for l, v in items
    )
    return f'<div class="stat-tile-row">{tiles}</div>'


def confidence_badge(stars: int, label_prefix: str = "") -> str:
    if stars >= 4:
        cls, text = "badge-good", "Strong market agreement"
    elif stars == 3:
        cls, text = "badge-neutral", "Moderate agreement"
    else:
        cls, text = "badge-warning", "Thin market"
    prefix = f"{_esc(label_prefix)} " if label_prefix else ""
    return f'<span class="badge {cls}">{prefix}{"★" * stars}{"☆" * (5 - stars)} · {text}</span>'


def prob_bar(home: str, away: str, h2h_outcomes: dict) -> str:
    """Stacked Home/Draw/Away bar. Home = blue role, Away = aqua role, Draw = gray."""
    home_p = (h2h_outcomes.get(home) or {}).get("fair_prob") or 0
    away_p = (h2h_outcomes.get(away) or {}).get("fair_prob") or 0
    draw_p = (h2h_outcomes.get("Draw") or {}).get("fair_prob") or 0
    total = home_p + away_p + draw_p
    if total <= 0:
        return ""
    home_pct, draw_pct, away_pct = (home_p / total * 100, draw_p / total * 100, away_p / total * 100)

    def seg(pct, color):
        if pct <= 0:
            return ""
        label = f"{round(pct)}%" if pct >= 10 else ""
        return f'<div class="prob-seg" style="width:{pct:.2f}%;background:{color};">{label}</div>'

    bar = f'<div class="prob-bar">{seg(home_pct, HOME_COLOR)}{seg(draw_pct, DRAW_COLOR)}{seg(away_pct, AWAY_COLOR)}</div>'
    legend_items = [f'<div class="prob-legend-item"><span class="prob-swatch" style="background:{HOME_COLOR};"></span>{_esc(home)} {round(home_pct)}%</div>']
    if draw_pct > 0:
        legend_items.append(f'<div class="prob-legend-item"><span class="prob-swatch" style="background:{DRAW_COLOR};"></span>Draw {round(draw_pct)}%</div>')
    legend_items.append(f'<div class="prob-legend-item"><span class="prob-swatch" style="background:{AWAY_COLOR};"></span>{_esc(away)} {round(away_pct)}%</div>')
    legend = f'<div class="prob-legend">{"".join(legend_items)}</div>'
    return bar + legend


def fixture_card(
    when: str,
    league: str,
    sport: str,
    home: str,
    away: str,
    h2h: dict | None,
    total_pick: dict | None,
    is_live: bool = False,
    live_score: str | None = None,
) -> str:
    sport_color = SPORT_COLORS.get(sport, HOME_COLOR)
    live_badge = (
        '<span class="badge badge-live"><span class="badge-dot"></span>LIVE</span>'
        if is_live
        else ""
    )
    league_badge = f'<span class="badge badge-league" style="border-left:3px solid {sport_color};">{_esc(league)}</span>'

    bar_html = ""
    if h2h and h2h.get("outcomes"):
        bar_html = prob_bar(home, away, h2h["outcomes"])

    picks_html = ""
    if h2h and h2h.get("outcomes"):
        fav_name, fav_data = max(h2h["outcomes"].items(), key=lambda kv: (kv[1]["fair_prob"] or 0))
        picks_html += (
            f'<div class="pick-line">Moneyline: <b>{_esc(fav_name)}</b> — best price '
            f'<b>{fav_data["best_odds"]}</b> @ {_esc(fav_data["best_book"])} '
            f'{confidence_badge(h2h["stars"])}</div>'
        )
    if total_pick:
        side = total_pick["side"]
        odds = total_pick["best_over_odds"] if side == "Over" else total_pick["best_under_odds"]
        book = total_pick["best_over_book"] if side == "Over" else total_pick["best_under_book"]
        picks_html += (
            f'<div class="pick-line">Totals: <b>{side} {total_pick["point"]}</b> — best price '
            f'<b>{odds}</b> @ {_esc(book)} {confidence_badge(total_pick["stars"])}</div>'
        )
    if not picks_html:
        picks_html = '<div class="pick-line">No markets quoted by tracked books — skip.</div>'

    score_line = f'<div class="meta-line">{_esc(live_score)}</div>' if live_score else ""

    card_class = "fixture-card live" if is_live else "fixture-card"
    return f"""
<div class="{card_class}">
  <div class="fixture-row">
    <div class="fixture-teams">{_esc(home)} vs {_esc(away)}</div>
    <div>{live_badge}{league_badge}</div>
  </div>
  <div class="meta-line">{_esc(when)}</div>
  {score_line}
  {bar_html}
  {picks_html}
</div>
"""
