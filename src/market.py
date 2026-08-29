"""Turning many books' quotes into one fair probability estimate.

Two mistakes are easy to make here, and the original version of this dashboard
made both.

**Mistake 1: averaging decimal odds.** Decimal odds are a reciprocal scale.
By Jensen's inequality ``mean(1/p) != 1/mean(p)``, so averaging prices and then
inverting systematically *overstates* the fair price -- it flatters every
outcome and manufactures phantom edge. Aggregation must happen in probability
(or log-odds) space.

**Mistake 2: de-vigging the aggregate.** Each book has its own margin and its
own margin *shape*. De-vigging once, after pooling, mixes margins together and
leaves a residue that varies with how lopsided the game is. The correct order is
de-vig each book's own complete price vector first, *then* pool the fair
probabilities.

Aggregation happens in log-odds space with a weighted mean, then renormalises.
Log-odds is the natural scale for combining probability estimates: it is
unbounded, roughly symmetric, and a weighted mean there is the maximum-likelihood
pool under a logistic-error model. Averaging raw probabilities instead
over-weights the tails.

Book weighting
--------------
Books are not interchangeable estimators. Pinnacle runs ~2% margins, takes six-
figure limits, and welcomes sharp money -- its price is a market-clearing price.
A recreational book's price is a *marketing* price shaded toward public
sentiment, and it moves only when Pinnacle moves. Weighting them equally throws
away the entire signal, so sharp books carry roughly 4-8x the weight of soft
ones, and the sharp-only anchor is preferred whenever it is available.
"""

from __future__ import annotations

import math
import statistics

from src.devig import DEFAULT_METHOD, devig

# Relative estimator weights. Grounded in margin size, limit size, and whether
# the book restricts winning accounts (a book that bans winners has no
# mechanism to keep its prices honest).
BOOK_WEIGHTS = {
    # Sharp / market-making: low margin, high limits, sharp money welcome.
    "pinnacle": 1.00,
    "circasports": 0.90,
    "betfair_ex_uk": 0.85,
    "betfair_ex_eu": 0.85,
    "betfair_ex_au": 0.80,
    "betfair": 0.80,
    "matchbook": 0.70,
    "smarkets": 0.65,
    "betonlineag": 0.50,
    "lowvig": 0.50,
    "bookmaker": 0.50,
    # Large retail: reasonably efficient, still shaded toward the public.
    "betfair_sb_uk": 0.30,
    "williamhill": 0.30,
    "betvictor": 0.30,
    "unibet_uk": 0.28,
    "unibet_eu": 0.28,
    "marathonbet": 0.28,
    "onexbet": 0.25,
    "fanduel": 0.25,
    "draftkings": 0.25,
    "betmgm": 0.22,
    "espnbet": 0.18,
    "ladbrokes_uk": 0.18,
    "coral": 0.18,
    "paddypower": 0.18,
    "skybet": 0.15,
    "betrivers": 0.15,
    "nordicbet": 0.15,
    "betsson": 0.15,
    "888sport": 0.15,
    "mybookieag": 0.12,
    "betus": 0.10,
    "bovada": 0.10,
}
DEFAULT_BOOK_WEIGHT = 0.15
# Books whose price alone is treated as a fair-value anchor.
SHARP_BOOKS = ("pinnacle", "circasports", "betfair_ex_uk", "betfair_ex_eu", "betfair", "bookmaker")

# Plausible bounds on a real bookmaker's margin. Anything outside this band is
# not a priced market and must never reach the fair-value estimate.
#
# This guard is not theoretical. Betting exchanges report placeholder vectors
# for markets with no matched liquidity -- observed live: Betfair quoting
# 1.04 / 1.04 / 1.04 across a 1X2, a "margin" of 65%. That vector de-vigs to
# exactly 1/3 per outcome, so it drags the fair price of every draw toward 3.00
# and manufactures a ~17% phantom edge on the draw at every other book. Because
# exchanges are weighted as *sharp*, the artefact was setting the anchor rather
# than being outvoted by it -- the worst possible place for bad data to land.
#
# A genuine book runs 1-20% margin; exchanges sit near zero and can go slightly
# negative across regions. Outside that, the market is suspended, unmatched, or
# mis-parsed.
MAX_PLAUSIBLE_MARGIN = 0.25
MIN_PLAUSIBLE_MARGIN = -0.05

_EPS = 1e-9


def book_weight(book_key: str) -> float:
    return BOOK_WEIGHTS.get((book_key or "").lower(), DEFAULT_BOOK_WEIGHT)


def is_sharp(book_key: str) -> bool:
    return (book_key or "").lower() in SHARP_BOOKS


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BookQuote:
    """One book's complete price vector for one market on one event."""

    __slots__ = ("key", "title", "prices", "fair", "margin", "weight")

    def __init__(self, key: str, title: str, prices: dict, method: str = DEFAULT_METHOD):
        self.key = key
        self.title = title
        self.prices = prices  # {outcome_label: decimal_odds}
        names = list(prices.keys())
        fair_list = devig([prices[n] for n in names], method=method)
        self.fair = {n: f for n, f in zip(names, fair_list, strict=True) if f is not None}
        raw_sum = sum(1.0 / o for o in prices.values() if o and o > 1)
        self.margin = 1.0 - 1.0 / raw_sum if raw_sum > 0 else 0.0
        self.weight = book_weight(key)

    @property
    def is_plausible(self) -> bool:
        """False for a vector that is not a real priced market.

        See :data:`MAX_PLAUSIBLE_MARGIN`. Dropping these is strictly better than
        down-weighting them: a placeholder vector carries no information at all,
        so any weight above zero only adds bias.
        """
        return MIN_PLAUSIBLE_MARGIN <= self.margin <= MAX_PLAUSIBLE_MARGIN

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<BookQuote {self.key} margin={self.margin:.3%} {self.prices}>"


def collect_quotes(event: dict, market_key: str, point=None, method: str = DEFAULT_METHOD) -> list:
    """Every book's complete price vector for one market, each de-vigged alone.

    Books quoting an incomplete market (e.g. an Over with no matching Under) are
    dropped: a partial vector has no recoverable margin and de-vigging it would
    invent probability mass.
    """
    quotes = []
    for bm in event.get("bookmakers", []) or []:
        for market in bm.get("markets", []) or []:
            if market.get("key") != market_key:
                continue
            prices = {}
            for outcome in market.get("outcomes", []) or []:
                if point is not None and outcome.get("point") != point:
                    continue
                price = outcome.get("price")
                if price and price > 1:
                    prices[outcome["name"]] = price
            if len(prices) < 2:
                continue
            quote = BookQuote(
                bm.get("key", "?"), bm.get("title", bm.get("key", "?")), prices, method
            )
            # Drop suspended / unmatched / mis-parsed vectors before they can
            # reach the anchor. See MAX_PLAUSIBLE_MARGIN for the live case that
            # motivated this.
            if not quote.is_plausible:
                continue
            quotes.append(quote)
    return quotes


def pool_fair_probs(quotes: list, exclude_book: str | None = None, sharp_only: bool = False) -> dict:
    """Weighted log-odds pool of per-book fair probabilities, renormalised.

    ``exclude_book`` matters more than it looks. When judging whether book X's
    price is +EV you must build the reference line *without* book X, otherwise
    X's own (possibly stale or mispriced) number is inside the benchmark it is
    being measured against, and the edge shrinks toward zero exactly when it is
    most real. This is the same leakage as scoring a model on its training data.
    """
    usable = [
        q
        for q in quotes
        if q.fair
        and (exclude_book is None or q.key != exclude_book)
        and (not sharp_only or is_sharp(q.key))
    ]
    if not usable:
        return {}

    names = sorted({n for q in usable for n in q.fair})
    pooled = {}
    for name in names:
        num = den = 0.0
        for q in usable:
            p = q.fair.get(name)
            if p is None:
                continue
            num += q.weight * _logit(p)
            den += q.weight
        if den > 0:
            pooled[name] = _expit(num / den)

    total = sum(pooled.values())
    if total <= 0:
        return {}
    return {n: p / total for n, p in pooled.items()}


def dispersion(quotes: list, outcome: str) -> float:
    """Standard deviation of books' fair probabilities for one outcome.

    This is the honest empirical estimate of how uncertain the fair price is.
    When books agree to within 0.4pp, the consensus is a tight estimate and a 3%
    edge is probably real. When they disagree by 4pp, a 3% edge is inside the
    noise. :mod:`src.edge` feeds this straight into stake sizing, which is what
    stops the model from betting hardest on exactly the games it understands
    least.
    """
    vals = [q.fair[outcome] for q in quotes if outcome in q.fair]
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def best_price(quotes: list, outcome: str) -> tuple:
    """(decimal_odds, book_key, book_title) of the best available price."""
    best = None
    for q in quotes:
        price = q.prices.get(outcome)
        if price and (best is None or price > best[0]):
            best = (price, q.key, q.title)
    return best or (None, None, None)


class MarketView:
    """Everything needed to judge one market: the fair line and every price."""

    def __init__(self, event: dict, market_key: str, point=None, method: str = DEFAULT_METHOD):
        self.market_key = market_key
        self.point = point
        self.quotes = collect_quotes(event, market_key, point, method)
        self.consensus = pool_fair_probs(self.quotes)
        self.sharp = pool_fair_probs(self.quotes, sharp_only=True)
        self.sharp_books = [q.key for q in self.quotes if is_sharp(q.key)]

    @property
    def n_books(self) -> int:
        return len(self.quotes)

    @property
    def outcomes(self) -> list:
        return sorted(self.consensus)

    def anchor(self, exclude_book: str | None = None) -> dict:
        """The reference fair line, sharp-anchored when a sharp book is present.

        Falls back to the full weighted consensus otherwise. Excluding the book
        under evaluation is only meaningful for the consensus fallback -- if the
        anchor is Pinnacle and you are evaluating Pinnacle, there is by
        definition no edge to find, and :func:`pool_fair_probs` returns empty.
        """
        if self.sharp:
            sharp_ex = pool_fair_probs(self.quotes, exclude_book=exclude_book, sharp_only=True)
            if sharp_ex:
                return sharp_ex
        return pool_fair_probs(self.quotes, exclude_book=exclude_book)

    def anchor_quality(self) -> str:
        if self.sharp:
            return "sharp"
        if self.n_books >= 6:
            return "consensus"
        if self.n_books >= 3:
            return "thin"
        return "unreliable"
