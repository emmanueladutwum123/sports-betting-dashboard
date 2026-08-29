"""Expected value, stake sizing, and the discipline that makes them survivable.

The arithmetic of an edge is trivial. Everything hard is in the two corrections
applied before a stake is sized:

1. **Uncertainty shrinkage.** Kelly assumes you know ``p``. You do not -- you
   have an estimate with a standard error. Kelly is concave in ``p`` on the
   downside and grows unboundedly on the upside, so plugging in a point estimate
   systematically *over*bets: the loss from overestimating ``p`` exceeds the gain
   from underestimating it by the same amount. The fix with the best evidence
   behind it is to bet a conservative lower bound on ``p`` rather than the point
   estimate, with the bound's width taken from how much the books actually
   disagree. Over-betting Kelly is not merely suboptimal, it is ruinous: betting
   above 2x the Kelly fraction has *negative* expected log growth even with a
   genuine edge.

2. **Slate normalisation.** Kelly fractions are derived one bet at a time, but
   bets are placed simultaneously against one bankroll. Twelve independent 3%
   Kelly bets is 36% of bankroll at risk in a single afternoon. Fractions are
   scaled down to respect a total-exposure cap.

The default is quarter-Kelly. Half-Kelly captures 75% of the growth rate at 25%
of the variance; quarter-Kelly is the right default when ``p`` is estimated
rather than known, because the effective Kelly fraction is already inflated by
estimation error before you apply any multiplier at all.
"""

from __future__ import annotations

import math

# Default staking policy.
DEFAULT_KELLY_FRACTION = 0.25
# Number of standard errors to shrink the probability estimate by. 1.0 is a
# ~84% one-sided confidence bound -- deliberately mild, because dispersion
# across books already understates true uncertainty (books copy each other).
DEFAULT_SHRINK_K = 1.0
# Refuse to size anything below this edge: it is inside the noise floor of the
# fair-price estimate, and slippage on the price you actually get will eat it.
MIN_EV_THRESHOLD = 0.01
# Above this, assume a stale line or a data error rather than a gift.
MAX_PLAUSIBLE_EV = 0.20
# Cap on total simultaneous bankroll exposure across one slate.
DEFAULT_EXPOSURE_CAP = 0.10


def ev_per_unit(prob: float, decimal_odds: float) -> float:
    """Expected profit per 1 unit staked. ``0.03`` means +3% EV."""
    if not prob or not decimal_odds or decimal_odds <= 1:
        return 0.0
    return prob * (decimal_odds - 1.0) - (1.0 - prob)


def edge_vs_fair(prob: float, decimal_odds: float) -> float:
    """Probability-space edge: how much better the price is than break-even.

    Equivalent to ``prob - 1/decimal_odds``. Reported alongside EV because they
    rank bets differently -- EV divides by nothing, so it flatters longshots,
    while probability edge flatters favourites. A bet should look good on both.
    """
    if not prob or not decimal_odds or decimal_odds <= 1:
        return 0.0
    return prob - 1.0 / decimal_odds


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll: ``(p*o - 1) / (o - 1)``."""
    if not prob or not decimal_odds or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    f = (prob * decimal_odds - 1.0) / b
    return max(f, 0.0)


def shrink_probability(prob: float, sigma: float, k: float = DEFAULT_SHRINK_K) -> float:
    """Conservative lower bound on ``prob`` given estimator dispersion ``sigma``."""
    if sigma <= 0:
        return prob
    return max(min(prob - k * sigma, 1.0 - 1e-9), 1e-9)


def growth_rate(prob: float, decimal_odds: float, fraction: float) -> float:
    """Expected log growth per bet at stake ``fraction``.

    Negative here means the bet shrinks the bankroll in the long run *even when
    the EV is positive* -- the signature of overbetting. Worth surfacing because
    it is the only number that makes the danger of full-Kelly legible.
    """
    if fraction <= 0 or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    if fraction * b <= -1 or fraction >= 1:
        return float("-inf")
    return prob * math.log(1.0 + fraction * b) + (1.0 - prob) * math.log(1.0 - fraction)


class Opportunity:
    """A single priced selection, fully evaluated."""

    __slots__ = (
        "event_id", "commence_time", "league", "home", "away", "market", "point",
        "selection", "book_key", "book_title", "odds", "fair_prob", "shrunk_prob",
        "sigma", "n_books", "anchor_quality", "ev", "edge", "kelly", "stake_fraction",
        "flags", "model_prob",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))
        self.flags = list(kw.get("flags") or [])

    @property
    def fair_odds(self) -> float | None:
        if not self.fair_prob:
            return None
        return 1.0 / self.fair_prob

    @property
    def price_edge_pct(self) -> float:
        """How much better the taken price is than the fair price, in %."""
        fo = self.fair_odds
        if not fo or not self.odds:
            return 0.0
        return (self.odds / fo - 1.0) * 100.0

    def as_row(self) -> dict:
        return {
            "Starts": self.commence_time,
            "League": self.league,
            "Match": f"{self.home} vs {self.away}",
            "Market": self.market if self.point is None else f"{self.market} {self.point}",
            "Selection": self.selection,
            "Book": self.book_title,
            "Odds": self.odds,
            "Fair odds": round(self.fair_odds, 2) if self.fair_odds else None,
            "Fair %": round((self.fair_prob or 0) * 100, 1),
            "EV %": round((self.ev or 0) * 100, 2),
            "Edge pp": round((self.edge or 0) * 100, 2),
            "Kelly %": round((self.kelly or 0) * 100, 2),
            "Stake %": round((self.stake_fraction or 0) * 100, 2),
            "Books": self.n_books,
            "Anchor": self.anchor_quality,
            "Disagreement pp": round((self.sigma or 0) * 100, 2),
            "Flags": ", ".join(self.flags) if self.flags else "",
        }


def evaluate(
    fair_prob: float,
    decimal_odds: float,
    sigma: float = 0.0,
    kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION,
    shrink_k: float = DEFAULT_SHRINK_K,
) -> dict:
    """Core per-bet arithmetic: raw EV, shrunk EV, and the stake it justifies."""
    shrunk = shrink_probability(fair_prob, sigma, shrink_k)
    return {
        "ev": ev_per_unit(fair_prob, decimal_odds),
        "ev_shrunk": ev_per_unit(shrunk, decimal_odds),
        "edge": edge_vs_fair(fair_prob, decimal_odds),
        "kelly": kelly_fraction(fair_prob, decimal_odds),
        "kelly_shrunk": kelly_fraction(shrunk, decimal_odds),
        "shrunk_prob": shrunk,
        "stake_fraction": kelly_fraction(shrunk, decimal_odds) * kelly_fraction_multiplier,
    }


def classify(ev: float, sigma: float, n_books: int, anchor_quality: str) -> list:
    """Warning flags. These exist to stop the biggest failure mode in +EV
    betting: treating a data artefact as a 12% edge and staking it."""
    flags = []
    if ev > MAX_PLAUSIBLE_EV:
        flags.append("implausible-EV: suspect stale line or bad data")
    if anchor_quality in ("thin", "unreliable"):
        flags.append(f"weak anchor ({anchor_quality})")
    if n_books < 3:
        flags.append("thin market")
    if sigma > 0.03:
        flags.append("books disagree strongly")
    if ev > 0 and sigma > 0 and ev < sigma:
        flags.append("edge smaller than book disagreement")
    return flags


def normalise_slate(opportunities: list, exposure_cap: float = DEFAULT_EXPOSURE_CAP) -> list:
    """Scale stakes so simultaneous exposure stays under ``exposure_cap``.

    A single proportional scaling preserves the *relative* Kelly ordering, which
    is what carries the growth-optimality; only the overall leverage changes.
    """
    total = sum(o.stake_fraction or 0.0 for o in opportunities)
    if total <= exposure_cap or total <= 0:
        return opportunities
    scale = exposure_cap / total
    for o in opportunities:
        o.stake_fraction = (o.stake_fraction or 0.0) * scale
        o.flags.append(f"stake scaled x{scale:.2f} for slate exposure cap")
    return opportunities
