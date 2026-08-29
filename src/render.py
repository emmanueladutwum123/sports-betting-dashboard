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
        f'<div class="stat-tile"><div class="stat-tile-label">{_esc(label)}</div>'
        f'<div class="stat-tile-value">{_esc(value)}</div></div>'
        for label, value in items
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


def ev_badge(ev_pct: float) -> str:
    """Colour-coded expected value. Negative EV is shown, not hidden -- most
    prices on any board are negative EV, and seeing that is the point."""
    if ev_pct is None:
        return ""
    if ev_pct >= 3:
        colour, label = "#16a34a", f"+{ev_pct:.1f}% EV"
    elif ev_pct > 0:
        colour, label = "#65a30d", f"+{ev_pct:.1f}% EV"
    elif ev_pct > -3:
        colour, label = "#a1a1aa", f"{ev_pct:.1f}% EV"
    else:
        colour, label = "#dc2626", f"{ev_pct:.1f}% EV"
    return (f'<span class="badge" style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}55;">{label}</span>')


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
        # Rank by the value on offer, not by who is most likely to win. The
        # favourite is usually the worst-priced selection on the board.
        best_name, best_data = max(
            h2h["outcomes"].items(), key=lambda kv: kv[1].get("best_ev_pct", -99)
        )
        picks_html += (
            f'<div class="pick-line">Moneyline: <b>{_esc(best_name)}</b> — fair '
            f'<b>{best_data["fair_odds"]}</b>, best <b>{best_data["best_odds"]}</b> '
            f'@ {_esc(best_data["best_book"])} {ev_badge(best_data.get("best_ev_pct", 0))} '
            f'{confidence_badge(h2h["stars"])}</div>'
        )
    if total_pick:
        side = total_pick["side"]
        odds = total_pick.get(f"best_{side.lower()}_odds")
        book = total_pick.get(f"best_{side.lower()}_book")
        picks_html += (
            f'<div class="pick-line">Totals: <b>{side} {total_pick["point"]}</b> — best '
            f'<b>{odds}</b> @ {_esc(book)} {ev_badge(total_pick.get("ev_pct", 0))} '
            f'{confidence_badge(total_pick["stars"])}</div>'
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


METHODOLOGY_MD = """
## What this tool does, and what it cannot do

### The estimate
Every fair probability here is built in four steps:

1. **De-vig each book separately.** A bookmaker's prices imply probabilities
   summing to more than 1; the excess is their margin. Removing it with the
   naive method (divide by the booksum) is *biased* -- it spreads margin in
   proportion to price, when books actually load margin onto longshots. The
   default here is **Shin's method**, which models the book as pricing to
   protect against informed money, and is the best-supported choice in the
   literature. On a lopsided 1X2 the two methods disagree by well over a
   percentage point on the favourite -- which is larger than the entire edge a
   +EV bettor is hunting. Switch methods in the sidebar and watch the numbers
   move.
2. **Pool across books in log-odds space, weighted by book quality.** Averaging
   decimal odds is a real mathematical error: odds are a reciprocal scale, so by
   Jensen's inequality the average price is systematically longer than the price
   implied by the average probability. That manufactures edge that is not there.
3. **Anchor to sharp books.** Pinnacle and the exchanges run thin margins, take
   large limits, and welcome winning bettors, so their price is a market-clearing
   price. A recreational book's price is shaded toward public sentiment. They are
   not interchangeable estimators and are not weighted as if they were.
4. **Exclude the book being judged.** When testing whether a book's price is
   +EV, that book's own number is removed from the benchmark. Leaving it in is
   the same leakage as scoring a model on its training data, and it shrinks
   edges exactly when they are most real.

### The stake
Sizing uses fractional Kelly on a **shrunk** probability, where the shrinkage is
the observed disagreement between books. This matters more than it sounds.
Kelly assumes you *know* the probability; you only have an estimate. Because
Kelly's growth curve falls away faster on the downside than it rises on the
upside, plugging in a point estimate systematically overbets. Past roughly twice
the Kelly fraction, expected log growth turns **negative even when your edge is
real** -- overbetting does not merely reduce your return, it reverses it. The
default quarter-Kelly and the exposure cap exist for that reason.

### The verdict
Results are a poor feedback signal. At a realistic 2-3% edge it takes thousands
of settled bets before profit can be distinguished from luck, and the ledger
tells you the exact number for your own betting. **Closing line value** is the
way out: if the sharp closing price moves toward a bet after you place it, you
bought something the market later agreed was underpriced. It is observable
within hours, far less noisy than P&L, and it is the metric sportsbooks
themselves use to identify winning accounts.

### What the backtest found
Walking a Dixon-Coles model forward across several seasons of real matches and
scoring it against **de-vigged Pinnacle closing prices**, the optimal weight on
the model in the top divisions comes out at or near **zero**. The market wins.
That result is reported rather than tuned away, because a model that is quietly
worse than the closing line will lose money slowly and confidently, and a
backtest that hides this is worse than no backtest.

The honest reading: for liquid 1X2 markets, the reliable sources of edge are
**line shopping** (taking the best of many prices for a bet the sharp consensus
already justifies) and **discipline** (staking correctly, tracking CLV, not
betting when nothing is mispriced) -- not out-forecasting the close. The model
earns its place on thinner markets and derivative prices, where it prices
totals, both-teams-to-score and handicaps from one internally consistent
distribution.

### What this is not
Not a guaranteed-profit system; no such thing exists. Odds move, the best price
may be gone by the time you reach it, winning accounts get limited, and a real
edge still loses over long stretches. Everything here is decision support with
its uncertainty made visible. Stake only what you can afford to lose.
"""
