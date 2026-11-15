"""The daily card: the strongest-evidence selections across many competitions.

What "confidence" means here
----------------------------
Two different things get called confidence, they point in opposite directions,
and conflating them is how bettors lose money while being mostly right.

**Confidence that the selection wins** is just the win probability. It is
maximised by backing short-priced favourites. A 1.20 favourite wins ~83% of the
time -- and at a fair price of 1.25 it carries roughly -4% expected value, so
backing those all season is a slow, high-strike-rate way to go broke.

**Confidence that the price is wrong in your favour** is expected value. It is
maximised by taking prices the market disagrees with, which are frequently
longshots that lose most of the time.

There is a third thing, separate from both, which this module treats as a gate
rather than a ranking: **confidence in the estimate itself**. A fair price
derived from 30 books that agree to within half a point is a tight estimate; one
from 3 books that disagree by 4 points is barely an estimate at all. That has
nothing to do with who wins -- it says how much to trust the other two numbers.

So the card exposes both rankings explicitly (:data:`MODES`) and gates both on
estimate quality. It never claims a selection will win.

Independence and the accumulator
--------------------------------
Because the card deliberately spans different countries and competitions, its
legs are close to statistically independent -- Boca Juniors' result tells you
nothing about the WNBA. That independence is what makes
:func:`parlay_analysis` meaningful, and it is also what makes an accumulator
such a bad idea: independent probabilities *multiply*, and so do the margins.
Ten legs at 75% is 5.6% to land, and ten legs each carrying -3% edge compound to
roughly -26%. The card computes both numbers so the decision is made against
arithmetic rather than optimism.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.devig import DEFAULT_METHOD
from src.edge import ev_per_unit, shrink_probability
from src.market import MarketView, dispersion

# Ranking modes. Each is defensible; they answer different questions.
MODES = {
    "likely": {
        "label": "Most likely to win (highest strike rate)",
        "help": "Ranks by conservative win probability. Expect to win most of these "
                "and still lose money slowly — short prices carry the worst value.",
    },
    "value": {
        "label": "Best value (positive expectation)",
        "help": "Ranks by expected value. Expect to lose more of these than you win, "
                "while making money if the edges are real.",
    },
    "balanced": {
        "label": "Balanced (likely AND fairly priced)",
        "help": "Win probability, restricted to selections not priced against you.",
    },
}
DEFAULT_MODE = "balanced"

# Estimate-quality gates. A selection failing these is not ranked at all: the
# fair price is too uncertain for either ranking to mean anything.
MIN_BOOKS = 8
MAX_DISPERSION = 0.035
ACCEPTABLE_ANCHORS = ("sharp", "consensus")

# Mode-specific price floors, in EV terms.
MIN_EV_BY_MODE = {"likely": -0.04, "balanced": -0.005, "value": 0.01}
# Ignore near-coin-flips in the probability-ranked modes: a 52% pick is not a
# "strong" selection in any useful sense, whatever its price.
MIN_PROB_BY_MODE = {"likely": 0.55, "balanced": 0.55, "value": 0.0}


class Selection:
    """One candidate leg of the daily card."""

    __slots__ = (
        "sport", "league", "event_id", "commence_time", "home", "away", "selection",
        "fair_prob", "shrunk_prob", "odds", "book", "book_key", "ev", "sigma",
        "n_books", "anchor_quality", "fair_odds", "is_home", "opponent",
        "market_prob", "calibration",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def estimate_stars(self) -> int:
        """1-5 on how tight the fair-price estimate is. NOT a claim about winning."""
        score = 0
        score += 2 if self.n_books >= 20 else 1 if self.n_books >= 12 else 0
        score += 2 if self.sigma <= 0.010 else 1 if self.sigma <= 0.020 else 0
        score += 1 if self.anchor_quality == "sharp" else 0
        return max(1, min(5, score))

    def as_row(self) -> dict:
        return {
            "Starts": self.commence_time,
            "Competition": self.league,
            "Match": f"{self.home} vs {self.away}",
            "Backing": self.selection,
            "Win %": round((self.fair_prob or 0) * 100, 1),
            "Market %": round((self.market_prob or 0) * 100, 1),
            "Conservative %": round((self.shrunk_prob or 0) * 100, 1),
            "Best odds": self.odds,
            "Book": self.book,
            "Fair odds": round(self.fair_odds, 2) if self.fair_odds else None,
            "EV %": round((self.ev or 0) * 100, 2),
            "Books": self.n_books,
            "Disagreement pp": round((self.sigma or 0) * 100, 2),
            "Estimate": "★" * self.estimate_stars + "☆" * (5 - self.estimate_stars),
        }


def _parse_iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def candidates_from_event(
    event: dict, sport: str, devig_method: str = DEFAULT_METHOD, shrink_k: float = 1.0
) -> list:
    """Every moneyline/1X2 outcome of one event, evaluated.

    Only the head-to-head market is used. It is the most liquid market on any
    board, which makes its consensus the most trustworthy estimate available --
    exactly what a "strongest selections" card should be built on.
    """
    view = MarketView(event, "h2h", None, devig_method)
    if view.n_books < 2:
        return []

    out = []
    anchor = view.anchor()
    raw_anchor = view.raw_anchor()
    for name in view.outcomes:
        fair = anchor.get(name)
        if not fair:
            continue
        prices = [(q.title, q.key, q.net_price(name)) for q in view.quotes if name in q.prices]
        if not prices:
            continue
        book_title, book_key, price = max(prices, key=lambda x: x[2])
        sigma = dispersion(view.quotes, name)
        home, away = event.get("home_team"), event.get("away_team")
        out.append(
            Selection(
                sport=sport,
                league=event.get("_league"),
                event_id=event.get("id"),
                commence_time=event.get("commence_time"),
                home=home,
                away=away,
                selection=name,
                fair_prob=fair,
                market_prob=raw_anchor.get(name),
                calibration=view.calibration,
                shrunk_prob=shrink_probability(fair, sigma, shrink_k),
                odds=price,
                book=book_title,
                book_key=book_key,
                ev=ev_per_unit(fair, price),
                sigma=sigma,
                n_books=view.n_books,
                anchor_quality=view.anchor_quality(),
                fair_odds=(1.0 / fair) if fair else None,
                is_home=(name == home),
                opponent=(away if name == home else home if name == away else "the field"),
            )
        )
    return out


def passes_quality_gate(sel: Selection) -> bool:
    return (
        sel.n_books >= MIN_BOOKS
        and sel.sigma <= MAX_DISPERSION
        and sel.anchor_quality in ACCEPTABLE_ANCHORS
    )


def build_card(
    events: list,
    sport: str,
    n: int = 5,
    mode: str = DEFAULT_MODE,
    within_hours: int = 24,
    now: datetime | None = None,
    devig_method: str = DEFAULT_METHOD,
    shrink_k: float = 1.0,
    one_per_league: bool = True,
    one_per_event: bool = True,
) -> dict:
    """Top ``n`` selections for one sport inside a time window.

    ``one_per_league`` enforces spread across competitions, which is both what a
    diversified card should look like and what keeps the legs independent enough
    for the accumulator arithmetic to hold.

    The return always reports how many selections were *available* versus asked
    for. Returning four when five were requested is a real answer; padding the
    fifth with something that failed the quality gate would not be.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(MODES)}")
    now = now or datetime.now(UTC)
    horizon = now + timedelta(hours=within_hours)

    pool, in_window = [], 0
    for event in events:
        start = _parse_iso(event.get("commence_time"))
        if start is None or not (now <= start <= horizon):
            continue
        in_window += 1
        pool.extend(candidates_from_event(event, sport, devig_method, shrink_k))

    gated = [s for s in pool if passes_quality_gate(s)]
    min_ev, min_prob = MIN_EV_BY_MODE[mode], MIN_PROB_BY_MODE[mode]
    eligible = [s for s in gated if s.ev >= min_ev and s.fair_prob >= min_prob]

    if mode == "value":
        eligible.sort(key=lambda s: -s.ev)
    else:
        eligible.sort(key=lambda s: -s.shrunk_prob)

    picked, seen_leagues, seen_events = [], set(), set()
    for sel in eligible:
        if one_per_event and sel.event_id in seen_events:
            continue
        if one_per_league and sel.league in seen_leagues:
            continue
        picked.append(sel)
        seen_events.add(sel.event_id)
        seen_leagues.add(sel.league)
        if len(picked) >= n:
            break

    return {
        "sport": sport,
        "mode": mode,
        "requested": n,
        "selections": picked,
        "events_in_window": in_window,
        "candidates": len(pool),
        "passed_quality_gate": len(gated),
        "eligible": len(eligible),
        "short_by": max(0, n - len(picked)),
        "window_hours": within_hours,
    }


# --------------------------------------------------------------------- parlay

def poisson_binomial(probs: list) -> list:
    """Exact distribution of the number of winners among independent legs.

    Each leg has its own probability, so the count is Poisson-binomial rather
    than binomial. Computed by the standard O(n^2) convolution, which is exact
    -- no normal or Poisson approximation, both of which are poor in the tail
    that matters here (all legs winning).
    """
    dist = [1.0]
    for p in probs:
        p = min(max(p, 0.0), 1.0)
        nxt = [0.0] * (len(dist) + 1)
        for k, acc in enumerate(dist):
            nxt[k] += acc * (1.0 - p)
            nxt[k + 1] += acc * p
        dist = nxt
    return dist


def _kelly_growth(prob: float, decimal_odds: float) -> float:
    """Expected log growth of one bet staked at its own Kelly optimum.

    Zero when the bet has no edge, so a set of fairly-priced legs correctly
    scores zero growth either way rather than manufacturing a difference.
    """
    from src.edge import kelly_fraction

    f = kelly_fraction(prob, decimal_odds)
    if f <= 0 or f >= 1:
        return 0.0
    b = decimal_odds - 1.0
    return prob * math.log(1.0 + f * b) + (1.0 - prob) * math.log(1.0 - f)


def parlay_analysis(selections: list) -> dict:
    """What actually happens if the whole card is combined into one accumulator.

    The headline numbers are deliberately blunt. Independent probabilities
    multiply, so a card of individually likely legs becomes an unlikely parlay
    very fast. Margins multiply too: each leg contributes its own vig, so a
    ten-leg accumulator of mildly negative-EV legs is severely negative-EV
    overall. Both effects are invisible until they are computed, which is why
    accumulators are the most profitable product a sportsbook sells.
    """
    if not selections:
        return {"n": 0}

    probs = [s.fair_prob for s in selections]
    conservative = [s.shrunk_prob for s in selections]
    combined_odds = 1.0
    for s in selections:
        combined_odds *= s.odds

    all_win = 1.0
    for p in probs:
        all_win *= p
    all_win_cons = 1.0
    for p in conservative:
        all_win_cons *= p

    dist = poisson_binomial(probs)
    expected_winners = sum(probs)

    # Fair price for the accumulator vs the price the book actually pays.
    fair_parlay_odds = (1.0 / all_win) if all_win > 0 else float("inf")
    parlay_ev = all_win * combined_odds - 1.0

    singles_ev = sum(s.ev for s in selections) / len(selections)

    # Expected log growth, each bet sized at its own Kelly optimum. This is the
    # rigorous comparison, and unlike EV it points the same way every time.
    #
    # Expected value alone does NOT condemn parlays: EV compounds as
    # prod(1 + ev_i) - 1, so combining genuinely +EV legs raises the EV. What it
    # also does is destroy the growth rate, because a parlay converts many
    # independent bets into one all-or-nothing bet and throws away every bit of
    # diversification. Kelly growth is additive across independent bets and
    # brutally concave in a single one, so backing the legs separately dominates
    # parlaying them for any set of legs at all.
    singles_growth = sum(_kelly_growth(s.fair_prob, s.odds) for s in selections)
    parlay_growth = _kelly_growth(all_win, combined_odds)

    return {
        "n": len(selections),
        "combined_odds": combined_odds,
        "all_win_prob": all_win,
        "all_win_prob_conservative": all_win_cons,
        "fair_parlay_odds": fair_parlay_odds,
        "parlay_ev": parlay_ev,
        "expected_winners": expected_winners,
        "distribution": dist,
        "singles_avg_ev": singles_ev,
        "lose_everything_prob": 1.0 - all_win,
        "singles_log_growth": singles_growth,
        "parlay_log_growth": parlay_growth,
        # >1 means backing the legs singly compounds the bankroll faster than
        # parlaying them, at each bet's own optimal stake.
        "growth_ratio": (singles_growth / parlay_growth) if parlay_growth > 0 else None,
    }


def summarise_card(card: dict) -> str:
    """One honest paragraph about what the card is and is not."""
    sels = card["selections"]
    if not sels:
        return (
            f"No {card['sport'].lower()} selection in the next {card['window_hours']}h "
            f"clears the quality gate ({card['events_in_window']} games in window, "
            f"{card['candidates']} candidates). Nothing to bet is a valid answer."
        )
    avg = sum(s.fair_prob for s in sels) / len(sels)
    par = parlay_analysis(sels)
    line = (
        f"{len(sels)} {card['sport'].lower()} selections across "
        f"{len({s.league for s in sels})} competitions, averaging "
        f"{avg * 100:.0f}% win probability. "
        f"All {len(sels)} landing together is {par['all_win_prob'] * 100:.1f}% likely — "
        f"expect about {par['expected_winners']:.1f} winners, not {len(sels)}."
    )
    if card["short_by"]:
        line += (
            f" Only {len(sels)} of the {card['requested']} requested qualified; "
            "the rest of the board did not meet the evidence bar."
        )
    return line


# --------------------------------------------------------------- daily ticket

TICKET_LEGS = 4
# Every leg must be a clear favourite on the fair line, at a price that is not
# stacked against the backer, and not so short that it adds risk without return.
TICKET_MIN_PROB = 0.60
TICKET_MIN_EV = -0.02
TICKET_MIN_ODDS = 1.15
# A four-leg accumulator loses far more often than it wins, so its stake is
# capped well below a single's whatever Kelly says.
TICKET_MAX_STAKE = 0.02
# The betting day runs 08:00 UTC to 08:00 UTC so that a night of NBA games,
# which tips off after midnight UTC, stays on the day it belongs to.
DAY_STARTS_HOUR_UTC = 8
# Stake for a ticket priced about fairly: no edge to size, so a small flat one.
TICKET_FLAT_STAKE = 0.005
NOTES_PATH = Path(__file__).resolve().parents[1] / "research" / "notes.json"


def research_notes(day: str, path: Path | None = None) -> dict:
    """Hand-filed research for one betting day (``YYYY-MM-DD``), or ``{}``.

    The ticket is ranked from prices alone. Team news -- injuries, suspensions,
    rested players, what each side needs from the game -- is researched
    separately and filed here, so the page can show which legs were checked.
    """
    try:
        return json.loads((path or NOTES_PATH).read_text(encoding="utf-8")).get(day, {})
    except (OSError, ValueError):
        return {}


def betting_day(now: datetime | None = None, offset_days: int = 0) -> tuple:
    """``(start, end)`` of the betting day containing ``now``, shifted by days."""
    now = now or datetime.now(UTC)
    start = now.replace(hour=DAY_STARTS_HOUR_UTC, minute=0, second=0, microsecond=0)
    if now < start:
        start -= timedelta(days=1)
    start += timedelta(days=offset_days)
    return start, start + timedelta(days=1)


def build_ticket(
    soccer_events: list,
    basket_events: list,
    n: int = TICKET_LEGS,
    start: datetime | None = None,
    end: datetime | None = None,
    now: datetime | None = None,
    devig_method: str = DEFAULT_METHOD,
    shrink_k: float = 1.0,
) -> dict:
    """The ``n`` best-evidenced favourites across soccer and basketball for one day.

    Legs are ranked by conservative win probability, one per match, and split
    between the two sports as evenly as the day allows: half from each when
    both have enough qualifying games, the shortfall made up from the other
    when one does not. A game that has already started is never used.

    Fewer than ``n`` legs is a real answer. The ticket is never padded with a
    selection that failed the evidence bar.
    """
    now = now or datetime.now(UTC)
    if start is None or end is None:
        start, end = betting_day(now)
    opens = max(start, now)

    ranked, in_window = {}, {}
    for sport, events in (("Soccer", soccer_events), ("Basketball", basket_events)):
        pool, count = [], 0
        for event in events:
            kick = _parse_iso(event.get("commence_time"))
            if kick is None or not (opens <= kick < end):
                continue
            count += 1
            pool.extend(candidates_from_event(event, sport, devig_method, shrink_k))
        eligible = sorted(
            (s for s in pool
             if passes_quality_gate(s) and s.fair_prob >= TICKET_MIN_PROB
             and s.ev >= TICKET_MIN_EV and s.odds >= TICKET_MIN_ODDS),
            key=lambda s: -s.shrunk_prob,
        )
        one_per_match, seen = [], set()
        for sel in eligible:
            if sel.event_id not in seen:
                seen.add(sel.event_id)
                one_per_match.append(sel)
        ranked[sport], in_window[sport] = one_per_match, count

    share = n // 2
    picked = ranked["Soccer"][:share] + ranked["Basketball"][:share]
    rest = sorted(
        ranked["Soccer"][share:] + ranked["Basketball"][share:], key=lambda s: -s.shrunk_prob
    )
    picked += rest[: n - len(picked)]
    reserves = [s for s in rest if s not in picked]
    picked.sort(key=lambda s: str(s.commence_time))

    return {
        "requested": n,
        "selections": picked,
        "reserves": reserves[:n],
        "short_by": max(0, n - len(picked)),
        "games": in_window,
        "qualified": {sport: len(sels) for sport, sels in ranked.items()},
        "start": start,
        "end": end,
    }


def ticket_stake(parlay: dict, kelly_multiplier: float = 0.25) -> float:
    """Fraction of bankroll for the accumulator: fractional Kelly on the
    conservative all-legs-win probability, capped at :data:`TICKET_MAX_STAKE`.
    Zero when the combined price carries no edge."""
    from src.edge import kelly_fraction

    if not parlay.get("n"):
        return 0.0
    full = kelly_fraction(parlay["all_win_prob_conservative"], parlay["combined_odds"])
    return max(0.0, min(TICKET_MAX_STAKE, full * kelly_multiplier))


# ------------------------------------------------------------ over / under

# A side needs this much of the fair probability before it is called a lean,
# and a totals line needs this many books before it is priced at all.
TOTALS_LEAN_PROB = 0.55
TOTALS_MIN_BOOKS = 3
TOTALS_VALUE_EV = 0.01


def main_total(event: dict, devig_method: str = DEFAULT_METHOD, shrink_k: float = 1.0) -> dict | None:
    """The market's main over/under line for one event, with both sides priced.

    The main line is the one most books quote -- the bookmakers' balance point,
    where Over and Under sit closest to even money. Both sides get the fair
    probability, the best available price and its expected value, and the row
    says plainly whether either side is a bet: usually neither is, because a
    balance point is by construction close to a coin flip.
    """
    from src.picks import _totals_lines

    best = None
    for line in _totals_lines(event):
        view = MarketView(event, "totals", line, devig_method)
        fair = view.anchor() if view.n_books >= TOTALS_MIN_BOOKS else None
        if not fair or not fair.get("Over") or not fair.get("Under"):
            continue
        rank = (view.n_books, -abs(fair["Over"] - 0.5))
        if best is None or rank > best[0]:
            best = (rank, line, view, fair)
    if best is None:
        return None
    _, line, view, fair = best

    out = {"point": line, "n_books": view.n_books}
    for side in ("Over", "Under"):
        prices = [(q.net_price(side), q.title) for q in view.quotes if side in q.prices]
        if not prices:
            return None
        price, book = max(prices)
        key = side.lower()
        out[f"fair_{key}"] = fair[side]
        out[f"{key}_odds"], out[f"{key}_book"] = price, book
        out[f"ev_{key}"] = ev_per_unit(fair[side], price)

    side = "Over" if out["ev_over"] >= out["ev_under"] else "Under"
    likelier = "Over" if out["fair_over"] >= out["fair_under"] else "Under"
    key = side.lower()
    if out[f"ev_{key}"] >= TOTALS_VALUE_EV:
        sigma = dispersion(view.quotes, side)
        out.update(
            side=side, verdict="value", odds=out[f"{key}_odds"], book=out[f"{key}_book"],
            prob=out[f"fair_{key}"], ev=out[f"ev_{key}"],
            shrunk_prob=shrink_probability(out[f"fair_{key}"], sigma, shrink_k),
        )
    elif out[f"fair_{likelier.lower()}"] >= TOTALS_LEAN_PROB:
        key = likelier.lower()
        out.update(side=likelier, verdict="lean", odds=out[f"{key}_odds"],
                   book=out[f"{key}_book"], prob=out[f"fair_{key}"], ev=out[f"ev_{key}"])
    else:
        out.update(side=None, verdict="none")
    return out
