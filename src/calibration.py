"""Residual calibration of the de-vigged fair line.

The premise of this dashboard is that the sharp closing price is the best
publicly available forecast, and :mod:`src.backtest` keeps confirming it. That
is a statement about the *average* match. It is not a statement that the price
is unbiased everywhere on the probability scale, and it turns out not to be.

What the data says
------------------
Over 66,272 matches across 18 European leagues (2015/16-2024/25), taking
Pinnacle's closing 1X2 prices and de-vigging each one with Shin's method, the
resulting probabilities are close to perfect in the middle of the scale and
systematically *too low* at the top of it:

======================  ========  ==========  ==========  ===========
selection band          outcomes  mean price  actual rate  bias
======================  ========  ==========  ==========  ===========
p in [0.5, 0.6)           12,798      0.5459      0.5478     +0.19pp
p in [0.6, 0.7)            6,568      0.6444      0.6591     +1.47pp
p in [0.7, 0.8)            3,550      0.7440      0.7654     +2.14pp
p in [0.8, 0.9)            1,815      0.8410      0.8523     +1.13pp
**p >= 0.6 (all)**        13,277      0.7063      0.7215   **+1.52pp**
======================  ========  ==========  ==========  ===========

The ``p >= 0.6`` row is +1.52pp against a 95% interval of +/-0.76pp -- roughly
four standard errors. It is the favourite-longshot bias, the oldest documented
inefficiency in betting markets, and Shin de-vigging (which is *designed* to
correct for it) removes some but not all of it.

It is not an artefact of three things worth ruling out explicitly:

*Selection.* Conditioning on a high estimated probability and then comparing to
outcomes is the classic winner's-curse trap -- but noise in the estimate would
bias the realised rate *downward* relative to the price. The bias runs the
other way, so if anything this understates it.

*One book.* The same +1.7pp appears in Pinnacle (+1.66pp), the cross-book
average (+1.85pp) and Bet365 (+1.77pp) on a common sample.

*One period or one league.* Positive in 8 of 10 seasons and 13 of 18 leagues.

The correction
--------------
One parameter, applied to the whole probability vector and renormalised::

    p'_i = p_i ** b / sum_j (p_j ** b)

``b > 1`` pushes probability mass toward the outcomes that are already likely,
which is exactly the shape of the bias. Fitted by minimising out-of-sample log
loss, ``b`` comes out at **1.0408** on 1X2 and **1.0409** on Over/Under 2.5 --
two independent markets, with different numbers of outcomes, agreeing to four
decimal places. That coincidence is the main reason to believe the parameter
rather than the story.

How much this is worth -- honestly
----------------------------------
As a *forecast*, almost nothing. Pooled over every match in an expanding-window
backtest (refit before each season on prior seasons only), sharpening improves
log loss by **0.00011 nats**, better in 7 of 9 seasons. That is not a better
model of soccer and should never be sold as one.

As a *number on the card*, it matters, because the card does not show the
average match -- it ranks by win probability and shows the top of the scale,
which is precisely where the bias lives. Out-of-sample:

=========  ===================  ====================
band       bias before          bias after
=========  ===================  ====================
p >= 0.6   +1.61pp              +0.35pp
p >= 0.7   +1.78pp              +0.47pp
p >= 0.8   +1.15pp              -0.08pp
=========  ===================  ====================

All three post-correction residuals sit inside their confidence intervals. The
displayed probability for a card selection is now honest; before, it was
understating a 72% shot as a 71% shot, every time.

One exponent is the wrong shape
-------------------------------
The bias is not uniform across leagues, and a single global constant makes
eight of eighteen leagues *worse*. The Bundesliga close does not understate
favourites at all -- it slightly overstates them -- so sharpening it further
pushes the error from -1.0pp to -2.3pp. The Premier League, similarly, is
already close to unbiased.

Fitting a separate exponent per league is the obvious fix and it overfits:
with 2,000-5,500 matches each, per-league estimates are noisy, and used raw
they score *worse* out-of-sample than the single global number. The answer is
partial pooling -- shrink each league's estimate halfway toward the global one::

    b_league = b_global + K * (b_fitted_for_that_league - b_global)

``K`` is itself chosen out-of-sample, by expanding-window log loss over 52,872
matches:

======================  ==========  =====================
shrinkage K             log loss    vs no calibration
======================  ==========  =====================
0.00 (global only)        0.993347   +0.000081
0.25                      0.993304   +0.000123
**0.50**                  0.993298   **+0.000129**
0.75                      0.993329   +0.000098
1.00 (per-league raw)     0.993396   +0.000031
======================  ==========  =====================

The curve peaks in the middle and full per-league pooling is barely better
than nothing, which is what overfitting looks like when you measure it.

The gain in pooled log loss is still tiny. What partial pooling actually buys
is that it stops breaking the sharp leagues. Mean absolute bias at ``p >= 0.6``,
averaged across the eighteen leagues, out-of-sample:

==================  ==========
no calibration        2.55pp
one global exponent   2.09pp
**shrunk per-league** **1.65pp**
==================  ==========

Scope
-----
Applied to soccer only, because soccer is the only sport with a closing-price
history here to fit and check against. Other sports pass through unchanged --
an uncalibrated number beats one calibrated with a constant borrowed from a
different sport. A soccer league with no fitted entry falls back to the global
exponent. Refit with ``scripts/fit_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fitted on 66,272 soccer matches, 18 European leagues, seasons 2015/16-2024/25,
# football-data.co.uk closing prices. See scripts/fit_calibration.py to refit.
SOCCER_EXPONENT = 1.0408

# Independently fitted on 39,548 Over/Under 2.5 markets from the same matches.
# Kept as a separate constant so that a refit can disagree with the 1X2 number
# and be seen to disagree, rather than being silently averaged away.
SOCCER_TOTALS_EXPONENT = 1.0409

CALIBRATION_SAMPLE = 66_272
CALIBRATION_SEASONS = "2015/16-2024/25"

# How far each league's own fitted exponent is allowed to pull away from the
# global one. Chosen out-of-sample; see the table in the module docstring.
SHRINKAGE_K = 0.5

# Per-league exponents, already shrunk. Keys are football-data.co.uk division
# codes; Odds API sport keys are mapped onto them. The comment on each line is
# that league's unshrunk fit, kept visible so the amount of pooling is legible
# rather than buried in the fitting script.
#
# Note D1/D2 sitting below 1.0: the German closing lines slightly *overstate*
# favourites, the opposite of the usual bias, which is exactly the case a
# single global constant got wrong.
LEAGUE_EXPONENTS = {
    "E0": 1.0242,   # unshrunk 1.0076, n=3800   English Premier League
    "E1": 1.0332,   # unshrunk 1.0256, n=5518   English Championship
    "E2": 1.0730,   # unshrunk 1.1052, n=5361   English League One
    "E3": 1.0212,   # unshrunk 1.0015, n=5403   English League Two
    "SP1": 1.0388,  # unshrunk 1.0369, n=3799   Spain La Liga
    "SP2": 1.0652,  # unshrunk 1.0897, n=4590   Spain Segunda
    "D1": 0.9854,   # unshrunk 0.9300, n=3059   Germany Bundesliga
    "D2": 0.9860,   # unshrunk 0.9313, n=3060   Germany Bundesliga 2
    "I1": 1.0677,   # unshrunk 1.0946, n=3800   Italy Serie A
    "I2": 1.0443,   # unshrunk 1.0478, n=3995   Italy Serie B
    "F1": 1.0268,   # unshrunk 1.0128, n=3551   France Ligue 1
    "F2": 1.0499,   # unshrunk 1.0589, n=3625   France Ligue 2
    "N1": 1.0271,   # unshrunk 1.0135, n=2983   Netherlands Eredivisie
    "P1": 1.0858,   # unshrunk 1.1308, n=3057   Portugal Primeira Liga
    "B1": 1.0292,   # unshrunk 1.0176, n=2733   Belgium First Div
    "T1": 1.0993,   # unshrunk 1.1579, n=3360   Turkey Super Lig
    "G1": 1.0441,   # unshrunk 1.0474, n=2348   Greece Super League
    "SC0": 1.0255,  # unshrunk 1.0101, n=2230   Scotland Premiership
}

# A fitted exponent outside this range means the refit found something very
# different from the favourite-longshot bias -- a data problem, most likely.
# Better to fail loudly than to ship a silent 1.4.
SANE_EXPONENT_RANGE = (0.90, 1.20)


@dataclass(frozen=True)
class Calibration:
    """A fitted sharpening exponent plus the provenance needed to trust it."""

    exponent: float
    n_matches: int
    seasons: str
    market: str = "h2h"
    league: str | None = None

    @property
    def is_identity(self) -> bool:
        return abs(self.exponent - 1.0) < 1e-9

    def describe(self) -> str:
        if self.is_identity:
            return "no calibration applied"
        direction = "toward favourites" if self.exponent > 1 else "toward longshots"
        scope = f"{self.league} " if self.league else "all-league "
        return (
            f"{scope}exponent {self.exponent:.4f} ({direction}), fitted on "
            f"{self.n_matches:,} matches, {self.seasons}"
        )


SOCCER_H2H = Calibration(SOCCER_EXPONENT, CALIBRATION_SAMPLE, CALIBRATION_SEASONS, "h2h")
SOCCER_TOTALS = Calibration(SOCCER_TOTALS_EXPONENT, 39_548, CALIBRATION_SEASONS, "totals")
IDENTITY = Calibration(1.0, 0, "not fitted", "any")

# Built once. Each league's Calibration carries its own league code so the app
# can say which number it used and why.
_LEAGUE_CALIBRATIONS = {
    code: Calibration(exponent, CALIBRATION_SAMPLE, CALIBRATION_SEASONS, "h2h", code)
    for code, exponent in LEAGUE_EXPONENTS.items()
}


def calibration_for(sport_key: str | None, market_key: str = "h2h") -> Calibration:
    """Which calibration applies to this market, if any.

    Deliberately conservative: anything that is not recognisably soccer gets
    :data:`IDENTITY`. A constant fitted on European league soccer has no claim
    on an NBA moneyline, and applying it there would be exactly the kind of
    unearned confidence this dashboard exists to avoid.

    Within soccer, a league with its own fitted exponent gets it; a soccer
    league that was never fitted (MLS, Brazil, Japan -- no closing-price
    history in the free feed) falls back to the global number. That fallback is
    a judgement call worth naming: it assumes an unmeasured league looks more
    like the average measured league than like no bias at all, which is the
    better bet but is still a bet.
    """
    key = (sport_key or "").lower()
    if not key.startswith("soccer"):
        return IDENTITY
    if market_key and market_key.startswith("totals"):
        return SOCCER_TOTALS
    if market_key not in ("h2h", "h2h_lay", ""):
        return IDENTITY

    from src.datafeeds.football_data import SPORT_KEY_TO_DIVISION

    division = SPORT_KEY_TO_DIVISION.get(key)
    return _LEAGUE_CALIBRATIONS.get(division, SOCCER_H2H)


def sharpen(probs: dict, exponent: float) -> dict:
    """``p_i ** b`` renormalised. Returns a fresh dict; input is untouched.

    Degenerate inputs (empty, non-positive total, a single outcome) come back
    unchanged rather than raising -- this sits in the middle of the live pricing
    path, and a malformed market should lose its calibration, not the whole card.
    """
    if not probs or abs(exponent - 1.0) < 1e-9:
        return dict(probs)
    powered = {}
    for name, p in probs.items():
        if p is None or p <= 0.0:
            powered[name] = 0.0
        else:
            powered[name] = float(p) ** exponent
    total = sum(powered.values())
    if total <= 0.0:
        return dict(probs)
    return {name: v / total for name, v in powered.items()}


def apply(probs: dict, calibration: Calibration | None) -> dict:
    """Apply a :class:`Calibration` to a fair-probability vector."""
    if calibration is None or calibration.is_identity:
        return dict(probs)
    return sharpen(probs, calibration.exponent)


# --------------------------------------------------------------------- fitting


def fit_exponent(prob_vectors: list, outcomes: list, bounds: tuple = (0.5, 2.0)) -> float:
    """The ``b`` minimising log loss on ``(vector, realised outcome)`` pairs.

    ``outcomes`` holds the key of the outcome that actually happened. Fitting
    minimises log loss rather than calibration error on purpose: log loss is the
    proper scoring rule whose optimum is the Kelly growth optimum, so the fitted
    ``b`` is the one that maximises bankroll growth, not the one that makes a
    reliability plot look tidiest.
    """
    import math

    from scipy.optimize import minimize_scalar

    pairs = [
        (p, y) for p, y in zip(prob_vectors, outcomes, strict=True) if p and y in p and p[y] > 0
    ]
    if len(pairs) < 100:
        raise ValueError(f"need at least 100 scored markets to fit, got {len(pairs)}")

    def loss(b: float) -> float:
        total = 0.0
        for p, y in pairs:
            s = sum(v**b for v in p.values() if v and v > 0)
            if s <= 0:
                continue
            total -= math.log(max((p[y] ** b) / s, 1e-12))
        return total / len(pairs)

    return float(minimize_scalar(loss, bounds=bounds, method="bounded").x)


def reliability(prob_vectors: list, outcomes: list, edges: tuple = (0.0, 0.1, 0.2, 0.3, 0.4,
                                                                    0.5, 0.6, 0.7, 0.8, 1.0)) -> list:
    """Predicted vs realised rate per probability band, with a 95% interval.

    Every outcome of every market is one observation, so a 3-way market
    contributes three rows' worth of evidence. Bands report the *mean predicted*
    probability rather than the band midpoint, since the two differ a lot in the
    sparse tails.
    """
    import math

    bands = [{"lo": lo, "hi": hi, "n": 0, "pred": 0.0, "hits": 0}
             for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    for p, y in zip(prob_vectors, outcomes, strict=True):
        if not p:
            continue
        for name, val in p.items():
            if val is None:
                continue
            for band in bands:
                if band["lo"] <= val < band["hi"]:
                    band["n"] += 1
                    band["pred"] += val
                    band["hits"] += int(name == y)
                    break

    rows = []
    for band in bands:
        if not band["n"]:
            continue
        n = band["n"]
        pred, actual = band["pred"] / n, band["hits"] / n
        se = math.sqrt(max(actual * (1.0 - actual), 1e-12) / n)
        rows.append({
            "band": f"{band['lo']:.1f}-{band['hi']:.1f}",
            "n": n,
            "predicted": pred,
            "actual": actual,
            "bias": actual - pred,
            "ci95": 1.96 * se,
            "significant": abs(actual - pred) > 1.96 * se,
        })
    return rows


def bias_above(prob_vectors: list, outcomes: list, threshold: float = 0.6) -> dict:
    """Aggregate bias over every selection priced at or above ``threshold``.

    This is the number that matters for a card that ranks by win probability:
    not "is the market calibrated overall" but "is it calibrated in the band I
    actually pick from".
    """
    import math

    preds, hits = [], []
    for p, y in zip(prob_vectors, outcomes, strict=True):
        if not p:
            continue
        for name, val in p.items():
            if val is not None and val >= threshold:
                preds.append(val)
                hits.append(1.0 if name == y else 0.0)
    if not preds:
        return {"n": 0, "threshold": threshold}
    n = len(preds)
    pred, actual = sum(preds) / n, sum(hits) / n
    se = math.sqrt(max(actual * (1.0 - actual), 1e-12) / n)
    return {
        "n": n,
        "threshold": threshold,
        "predicted": pred,
        "actual": actual,
        "bias": actual - pred,
        "ci95": 1.96 * se,
        "significant": abs(actual - pred) > 1.96 * se,
    }
