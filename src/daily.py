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

import math
from datetime import UTC, datetime, timedelta

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
    for name in view.outcomes:
        anchor = view.anchor()
        fair = anchor.get(name)
        if not fair:
            continue
        prices = [(q.title, q.key, q.prices[name]) for q in view.quotes if name in q.prices]
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
