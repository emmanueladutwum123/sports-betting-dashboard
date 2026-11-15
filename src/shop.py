"""Line shopping: the best price you can actually take, for every selection.

The backtest in this project reached one firm conclusion about where a retail
bettor's edge comes from. It is not forecasting -- the closing line beat the
model in all 36 tests. It is *price*: the same sharp-consensus signal returned
nothing at Pinnacle's own number and +2.1% at the best number available across
books. So the most valuable thing this dashboard can do is answer, for any bet
you are already considering, "who pays the most for it, and is even that
enough?"

Three things make a raw "highest odds" column untrustworthy, and each is
handled here rather than left to the reader:

* **Exchange commission.** 7.60 on Betfair pays 7.27 after 5%. Prices are
  compared net (see :data:`src.market.EXCHANGE_COMMISSION`).
* **Stale quotes.** The highest number on a board is disproportionately often a
  book that stopped updating. Quotes lagging their market are set aside and the
  best *live* price is reported instead.
* **Books you cannot use.** A price at a book that will not take your account
  is not a price. ``books`` restricts the offer side to the ones you hold --
  the fair line is still built from every book in the feed.

And one thing makes a best price worth less than it looks: **the best price is
usually still not a bet.** Most selections are negative-EV at every book. The
table says so on every row instead of implying that "best" means "good".
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime

from src.devig import DEFAULT_METHOD
from src.edge import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_SHRINK_K,
    MAX_PLAUSIBLE_EV,
    MIN_EV_THRESHOLD,
    evaluate,
)
from src.market import MarketView, dispersion
from src.picks import DROPPED_FLAGS, _totals_lines, scan_event, selection_key

# A totals line quoted by fewer books than this has no meaningful "best" price
# -- there is nothing to shop between, and no fair line to judge it against.
MIN_BOOKS_TO_SHOP = 3

VALUE, NOISE, FAIR, SKIP = "✅ Value", "〰️ Edge within noise", "➖ Fair price", "⛔ Overpriced"


def _parse_iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def split_upcoming(events: list, now: datetime | None = None) -> tuple:
    """(not yet started, already started).

    The feed keeps returning a match after kick-off. In play, books reprice at
    different speeds, so comparing them manufactures large "edges" that are
    nothing but one book being four seconds slower than another. Everything
    that recommends a bet works from the first list only.
    """
    now = now or datetime.now(UTC)
    upcoming, started = [], []
    for event in events:
        start = _parse_iso(event.get("commence_time"))
        (started if start is not None and start <= now else upcoming).append(event)
    return upcoming, started


def verdict_for(ev: float, stake_fraction: float, min_ev: float = MIN_EV_THRESHOLD) -> str:
    """Four honest buckets. 'Value' requires the edge to survive shrinkage: a
    positive EV that Kelly sizes at zero once book disagreement is subtracted
    is reported as what it is, an edge the estimate cannot distinguish from
    none."""
    if ev >= min_ev and stake_fraction > 0:
        return VALUE
    if ev >= min_ev:
        return NOISE
    if ev > -0.01:
        return FAIR
    return SKIP


class ShopLine:
    """One selection, with the best takeable price and how it compares."""

    __slots__ = (
        "best", "runner_up", "typical_odds", "n_offers", "stale_better", "verdict",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def shop_gain(self) -> float:
        """Extra return from taking the best price over a typical one.

        This is the part of the edge that is fully in your control and needs no
        forecasting at all: same bet, same outcome, more money back.
        """
        if not self.typical_odds or not self.best.odds:
            return 0.0
        return self.best.odds / self.typical_odds - 1.0

    def as_row(self, bankroll: float | None = None) -> dict:
        o = self.best
        row = {
            "Starts": o.commence_time,
            "League": o.league,
            "Match": f"{o.home} vs {o.away}",
            "Market": "Match result" if o.market == "h2h" else f"Total {o.point}",
            "Selection": o.selection,
            "Best odds": o.odds,
            "Book": o.book_title,
            "Next best": (
                f"{self.runner_up.odds} @ {self.runner_up.book_title}" if self.runner_up else "—"
            ),
            "Typical odds": round(self.typical_odds, 2) if self.typical_odds else None,
            "Shop gain %": round(self.shop_gain * 100, 2),
            "Fair odds": round(o.fair_odds, 2) if o.fair_odds else None,
            "Win %": round((o.fair_prob or 0) * 100, 1),
            "EV %": round((o.ev or 0) * 100, 2),
            "Stake %": round((o.stake_fraction or 0) * 100, 2),
            "Verdict": self.verdict,
            "Books": self.n_offers,
            "Notes": "; ".join(self.notes()),
        }
        if bankroll is not None:
            row["Stake"] = round(bankroll * (o.stake_fraction or 0), 2)
        return row

    def notes(self) -> list:
        notes = list(self.best.flags)
        if self.best.quoted_odds and self.best.quoted_odds != self.best.odds:
            notes.append(f"quoted {self.best.quoted_odds}")
        if self.stale_better:
            notes.append(
                f"ignored stale {self.stale_better.odds} @ {self.stale_better.book_title}"
            )
        return notes


def best_lines(
    event: dict,
    devig_method: str = DEFAULT_METHOD,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    shrink_k: float = DEFAULT_SHRINK_K,
    books: set | None = None,
    min_ev: float = MIN_EV_THRESHOLD,
) -> list:
    """Best takeable price for every selection on one event.

    Every row is judged against a fair line that excludes the book offering it,
    exactly as the +EV board does, so the two views can never disagree about
    whether a given price is value.
    """
    every = scan_event(
        event, devig_method=devig_method, kelly_multiplier=kelly_multiplier,
        shrink_k=shrink_k, include_negative=True,
    )
    grouped = {}
    for o in every:
        grouped.setdefault(selection_key(o), []).append(o)

    lines = []
    for offers in grouped.values():
        if len(offers) < MIN_BOOKS_TO_SHOP:
            continue
        # The typical price is taken over *every* book, so the gain from
        # shopping is measured against the market, not against your own subset.
        typical = statistics.median(o.odds for o in offers)

        mine = [o for o in offers if books is None or o.book_key in books]
        live = [o for o in mine if not any(f.startswith(DROPPED_FLAGS) for f in o.flags)]
        if not live:
            continue
        live.sort(key=lambda o: -o.odds)
        best = live[0]
        # A different *price*, not merely a different book at the same price.
        runner_up = next((o for o in live[1:] if o.odds < best.odds), None)
        stale = [
            o for o in mine
            if o.odds > best.odds and any(f.startswith("stale-quote") for f in o.flags)
        ]
        lines.append(
            ShopLine(
                best=best,
                runner_up=runner_up,
                typical_odds=typical,
                n_offers=len(offers),
                stale_better=max(stale, key=lambda o: o.odds) if stale else None,
                verdict=verdict_for(best.ev or 0.0, best.stake_fraction or 0.0, min_ev),
            )
        )
    lines.sort(key=lambda ln: -(ln.best.ev or 0))
    return lines


def shop_slate(events: list, **kw) -> list:
    """:func:`best_lines` across many events, best expected value first."""
    out = []
    for event in events:
        out.extend(best_lines(event, **kw))
    out.sort(key=lambda ln: -(ln.best.ev or 0))
    return out


def books_in(events: list) -> dict:
    """{book_key: title} for every book quoting anything in the feed."""
    found = {}
    for event in events:
        for bm in event.get("bookmakers", []) or []:
            if bm.get("key"):
                found[bm["key"]] = bm.get("title") or bm["key"]
    return dict(sorted(found.items(), key=lambda kv: kv[1].lower()))


def markets_for(event: dict, devig_method: str = DEFAULT_METHOD) -> list:
    """[(label, market_key, point, [selections])] that can be price-checked."""
    out = []
    view = MarketView(event, "h2h", None, devig_method)
    if view.n_books >= 2 and view.anchor():
        home, away = event.get("home_team"), event.get("away_team")
        order = [n for n in (home, "Draw", away) if n in view.anchor()]
        out.append(("Match result", "h2h", None, order))
    for line in _totals_lines(event):
        view = MarketView(event, "totals", line, devig_method)
        if view.n_books >= MIN_BOOKS_TO_SHOP and view.anchor():
            out.append((f"Total {line}", "totals", line, ["Over", "Under"]))
    return out


def check_price(
    event: dict,
    market_key: str,
    point,
    selection: str,
    offered_odds: float,
    devig_method: str = DEFAULT_METHOD,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    shrink_k: float = DEFAULT_SHRINK_K,
    min_ev: float = MIN_EV_THRESHOLD,
) -> dict | None:
    """Judge a price from a book that is not in the feed.

    The licensed feed does not carry every bookmaker -- SportyBet and bet365 in
    particular are absent. That does not stop their prices being evaluated: the
    fair line needs the *market*, not the book. Type in what your book offers
    and it is measured against the same sharp-anchored fair value as everything
    else here.

    Because the book being judged is not part of the benchmark, the full anchor
    is used with nothing excluded. Returns ``None`` when the market has no
    usable fair line.
    """
    if not offered_odds or offered_odds <= 1:
        return None
    view = MarketView(event, market_key, point, devig_method)
    anchor = view.anchor()
    fair = anchor.get(selection) if anchor else None
    if not fair:
        return None

    sigma = dispersion(view.quotes, selection)
    metrics = evaluate(fair, offered_odds, sigma, kelly_multiplier, shrink_k)
    feed = sorted(
        ((q.net_price(selection), q.title) for q in view.quotes if selection in q.prices),
        reverse=True,
    )
    beaten = sum(1 for price, _ in feed if offered_odds > price)
    return {
        "selection": selection,
        "offered_odds": offered_odds,
        "fair_prob": fair,
        "fair_odds": 1.0 / fair,
        # The lowest price at which this becomes a bet at all.
        "min_odds_for_value": (1.0 + min_ev) / fair,
        "ev": metrics["ev"],
        "edge": metrics["edge"],
        "shrunk_prob": metrics["shrunk_prob"],
        "stake_fraction": metrics["stake_fraction"],
        "sigma": sigma,
        "n_books": view.n_books,
        "anchor_quality": view.anchor_quality(),
        "best_feed_odds": feed[0][0] if feed else None,
        "best_feed_book": feed[0][1] if feed else None,
        "typical_odds": statistics.median(p for p, _ in feed) if feed else None,
        "beats_books": beaten,
        "verdict": verdict_for(metrics["ev"], metrics["stake_fraction"], min_ev),
        # A price far above every book in the feed is far more often a typo, a
        # different market (e.g. draw-no-bet read as 1X2) or a boosted price
        # with stake limits than it is a gift.
        "suspicious": metrics["ev"] > MAX_PLAUSIBLE_EV,
    }
