"""De-vigging: recovering fair (margin-free) probabilities from quoted odds.

Why this module exists
----------------------
A bookmaker's quoted prices imply probabilities that sum to more than 1. That
excess is the *overround* (booksum - 1). Removing it is the single most
important estimation step in the whole pipeline, because every downstream
number -- expected value, Kelly stake, closing-line value -- is a difference
between two probabilities, and a systematic 1-2% bias in the de-vig swallows
the entire realistic edge.

The naive fix (divide every implied probability by the booksum) is the
*multiplicative* method. It is provably biased in the presence of the
favourite-longshot bias: it assumes the bookmaker spreads margin in proportion
to price, when in reality margin is loaded disproportionately onto longshots.
Empirical comparisons consistently rank Shin and power above multiplicative for
predictive accuracy.

Methods implemented
-------------------
Let ``pi_i = 1 / o_i`` be the raw implied probability of outcome *i* and
``B = sum(pi_i)`` the booksum.

multiplicative
    ``p_i = pi_i / B``. Margin proportional to price. Fast, biased.

additive
    ``p_i = pi_i - (B - 1) / n``. Margin spread equally in probability units.
    Can produce negative probabilities on very lopsided books, so we clip and
    renormalise.

power
    Solve for ``k`` such that ``sum(pi_i ** k) = 1``, then ``p_i = pi_i ** k``.
    Because ``k > 1``, small probabilities shrink proportionally more than large
    ones -- exactly the favourite-longshot correction. Good general default.

shin
    Shin (1992, 1993) models the bookmaker as setting prices to protect against
    a proportion ``z`` of insider bettors. Given ``z``::

        p_i = ( sqrt(z^2 + 4(1-z) * pi_i^2 / B) - z ) / (2(1-z))

    and ``z`` is the root of ``sum(p_i) = 1``. Best-supported method in the
    academic literature; the recovered ``z`` is itself informative (high ``z``
    means the book fears informed money on this market).

odds_ratio
    Cheung's method: the fair odds-ratio is a constant multiple ``c`` of the
    quoted odds-ratio, ``p_i / (1 - p_i) = c * pi_i / (1 - pi_i)``, with ``c``
    solved so probabilities sum to 1.

Note on two-outcome markets: Shin and additive coincide exactly, so the method
choice only bites on 1X2 / futures / any market with three or more outcomes,
and on strongly lopsided two-way markets where power still differs.

References
----------
Shin, H. S. (1993). "Measuring the Incidence of Insider Trading in a Market for
State-Contingent Claims." *The Economic Journal*, 103(420), 1141-1153.
Strumbelj, E. (2014). "On determining probability forecasts from betting odds."
*International Journal of Forecasting*, 30(4), 934-943.
Clarke, S., Kovalchik, S., & Ingram, M. (2017). "Adjusting bookmaker's odds to
allow for overround." *American Journal of Sports Science*, 5(6), 45-49.
"""

from __future__ import annotations

import math

METHODS = ("shin", "power", "multiplicative", "additive", "odds_ratio")
DEFAULT_METHOD = "shin"

# Bisection settings. 200 iterations of bisection on a bracket of width < 10
# resolves to < 1e-59, i.e. far below float precision -- the loop always exits
# on the tolerance check long before then; the cap is just a hang guard.
_TOL = 1e-12
_MAX_ITER = 200


class DevigError(ValueError):
    """Raised when a price vector cannot be de-vigged into a valid distribution."""


def implied_prob(decimal_odds: float | None) -> float | None:
    """Raw implied probability of a single decimal price (margin still in it)."""
    if not decimal_odds or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds


def booksum(decimal_odds: list) -> float:
    """Sum of raw implied probabilities. ``booksum - 1`` is the overround."""
    return sum(p for p in (implied_prob(o) for o in decimal_odds) if p is not None)


def margin(decimal_odds: list) -> float:
    """Bookmaker margin as a fraction of turnover: ``1 - 1 / booksum``.

    This is the theoretically correct "hold" -- the share of stake the book keeps
    when action is balanced -- and is what should be compared across markets with
    different numbers of outcomes. ``booksum - 1`` (the overround) systematically
    overstates hold and is not comparable between 2-way and 3-way markets.
    """
    b = booksum(decimal_odds)
    if b <= 0:
        return 0.0
    return 1.0 - 1.0 / b


def _bisect(f, lo: float, hi: float) -> float:
    """Root of a continuous decreasing-or-increasing ``f`` on ``[lo, hi]``."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        # No sign change in the bracket -- caller falls back to multiplicative.
        raise DevigError("root not bracketed")
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < _TOL or (hi - lo) < _TOL:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _normalise(probs: list) -> list:
    total = sum(probs)
    if total <= 0:
        raise DevigError("degenerate probability vector")
    return [p / total for p in probs]


def devig_multiplicative(pis: list) -> list:
    return _normalise(pis)


def devig_additive(pis: list) -> list:
    n = len(pis)
    excess = (sum(pis) - 1.0) / n
    adjusted = [max(p - excess, 1e-9) for p in pis]
    return _normalise(adjusted)


def devig_power(pis: list) -> list:
    """Solve ``sum(pi_i ** k) = 1`` for ``k``.

    ``sum(pi ** k)`` is strictly decreasing in ``k`` for ``pi < 1``, so the root
    is unique and bisection is safe.
    """
    if any(p >= 1.0 for p in pis):
        return devig_multiplicative(pis)

    def f(k: float) -> float:
        return sum(p**k for p in pis) - 1.0

    try:
        k = _bisect(f, 1e-6, 50.0)
    except DevigError:
        return devig_multiplicative(pis)
    return _normalise([p**k for p in pis])


def devig_shin(pis: list) -> list:
    """Shin's method. Returns fair probabilities; see :func:`shin_z` for ``z``."""
    z = shin_z(pis)
    return _normalise(_shin_probs(pis, z))


def _shin_probs(pis: list, z: float) -> list:
    b = sum(pis)
    if z <= 0 or z >= 1:
        return _normalise(pis)
    out = []
    for pi in pis:
        disc = z * z + 4.0 * (1.0 - z) * (pi * pi) / b
        out.append((math.sqrt(max(disc, 0.0)) - z) / (2.0 * (1.0 - z)))
    return out


def shin_z(pis: list) -> float:
    """Estimated proportion of insider money, ``z``, implied by the price vector.

    Useful as a signal in its own right: an unusually high ``z`` on one market
    means the book has widened specifically against informed flow there, which is
    a reason to *distrust* a large apparent edge rather than to bet it.
    """
    if len(pis) < 2 or sum(pis) <= 1.0:
        return 0.0

    def f(z: float) -> float:
        return sum(_shin_probs(pis, z)) - 1.0

    try:
        # At z -> 0 the sum is sqrt(booksum) > 1; it decreases in z. Bracket
        # short of 1.0 where the expression is singular.
        return _bisect(f, 1e-9, 1.0 - 1e-9)
    except DevigError:
        return 0.0


def devig_odds_ratio(pis: list) -> list:
    """Cheung's odds-ratio method: fair OR is a constant multiple of quoted OR."""
    if any(p >= 1.0 for p in pis):
        return devig_multiplicative(pis)

    def p_of(c: float, pi: float) -> float:
        # Invert p/(1-p) = c * pi/(1-pi).
        r = c * pi / (1.0 - pi)
        return r / (1.0 + r)

    def f(c: float) -> float:
        return sum(p_of(c, pi) for pi in pis) - 1.0

    try:
        c = _bisect(f, 1e-6, 1e6)
    except DevigError:
        return devig_multiplicative(pis)
    return _normalise([p_of(c, pi) for pi in pis])


_DISPATCH = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
    "odds_ratio": devig_odds_ratio,
}


def devig(decimal_odds: list, method: str = DEFAULT_METHOD) -> list:
    """Fair probabilities for one complete mutually-exclusive market.

    ``decimal_odds`` must be every outcome of a single market (e.g. home/draw/
    away, or over/under at one line). Entries that are missing or unpriced come
    back as ``None`` and are excluded from the normalisation -- but note that a
    partial market cannot be de-vigged correctly, so callers should require a
    complete price vector before trusting the output.
    """
    if method not in _DISPATCH:
        raise ValueError(f"unknown de-vig method {method!r}; expected one of {METHODS}")

    raw = [implied_prob(o) for o in decimal_odds]
    live_idx = [i for i, p in enumerate(raw) if p is not None]
    if not live_idx:
        return [None] * len(decimal_odds)

    pis = [raw[i] for i in live_idx]
    if sum(pis) <= 1.0:
        # Booksum <= 1 means either an arbitrage or an incomplete market. There
        # is no margin to remove; normalising anyway would invent probability
        # mass. Report the normalised vector but callers detect arbs separately.
        fair = _normalise(pis)
    else:
        try:
            fair = _DISPATCH[method](pis)
        except (DevigError, ValueError, ZeroDivisionError):
            fair = devig_multiplicative(pis)

    out: list = [None] * len(decimal_odds)
    for slot, p in zip(live_idx, fair, strict=True):
        out[slot] = p
    return out


def fair_odds(prob: float | None) -> float | None:
    """Break-even decimal price for a probability. Bet above it, skip below it."""
    if not prob or prob <= 0 or prob >= 1:
        return None
    return 1.0 / prob


# --- Backwards compatibility -------------------------------------------------
# src/probability.py used to own implied_prob/devig with a multiplicative-only
# implementation. It now re-exports from here so old call sites keep working.
