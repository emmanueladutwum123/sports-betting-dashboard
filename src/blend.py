"""Combining an independent model with the market price.

The market is a strong forecaster -- de-vigged closing odds are close to the
best publicly available probability estimate, and any honest backtest of a
hand-built model against them starts out losing. So the goal is not to *replace*
the market number with a model number. It is to ask whether the model carries
any information the market has not already priced, and to move off the market
only by the amount that information justifies.

Blending happens in log-odds space::

    logit(p_blend) = w * logit(p_model) + (1 - w) * logit(p_market)

Log-odds is the right scale for two reasons. It keeps blended probabilities
strictly inside (0, 1) for any weight, and under a logistic error model the
weighted log-odds mean is the maximum-likelihood pool of two estimates -- so
``w`` has a real interpretation as relative information content, not just a
knob. For multi-outcome markets each outcome is blended independently and the
vector renormalised, which is the softmax-consistent generalisation.

Choosing ``w``
--------------
Not by taste. :func:`optimal_weight` grid-searches ``w`` to minimise out-of-
sample log loss on historical matches with their closing prices. Typical honest
answers for a well-specified soccer model against closing Pinnacle prices are
``w`` between 0.05 and 0.25. A backtest returning ``w = 0`` is not a failure --
it is the model correctly reporting that it adds nothing, which is worth far
more than a fitted number that quietly loses money.
"""

from __future__ import annotations

import math

_EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def blend_probs(model: dict, market: dict, weight: float) -> dict:
    """Log-odds blend of two probability vectors over the same outcome labels.

    Outcomes missing from either side fall back to whichever estimate exists.
    """
    if not market:
        return dict(model)
    if not model or weight <= 0:
        return dict(market)
    weight = min(max(weight, 0.0), 1.0)

    out = {}
    for name in set(market) | set(model):
        pm, pk = model.get(name), market.get(name)
        if pm is None:
            out[name] = pk
        elif pk is None:
            out[name] = pm
        else:
            out[name] = _expit(weight * _logit(pm) + (1.0 - weight) * _logit(pk))

    total = sum(v for v in out.values() if v)
    if total <= 0:
        return dict(market)
    return {k: (v / total if v else 0.0) for k, v in out.items()}


def log_loss(probs: list, outcomes: list) -> float:
    """Mean negative log likelihood. The scoring rule that matters for betting.

    Log loss is the proper scoring rule whose optimisation coincides with
    maximising expected log bankroll growth -- it is Kelly's objective in
    disguise. Brier score is easier to read but does not share that property, so
    weight selection is done on log loss.
    """
    if not probs:
        return float("nan")
    total = 0.0
    for p, y in zip(probs, outcomes, strict=True):
        total -= math.log(min(max(p[y], _EPS), 1.0))
    return total / len(probs)


def brier(probs: list, outcomes: list) -> float:
    """Multi-class Brier score: mean squared error over the outcome vector."""
    if not probs:
        return float("nan")
    total = 0.0
    for p, y in zip(probs, outcomes, strict=True):
        for name, val in p.items():
            total += (val - (1.0 if name == y else 0.0)) ** 2
    return total / len(probs)


def optimal_weight(model_probs: list, market_probs: list, outcomes: list, grid: int = 41) -> dict:
    """Grid-search the blend weight that minimises out-of-sample log loss.

    Returns the best weight alongside the model-only and market-only baselines,
    so the improvement (or lack of it) is legible rather than asserted.
    """
    if not model_probs:
        return {"weight": 0.0, "log_loss": float("nan"), "market_log_loss": float("nan"),
                "model_log_loss": float("nan"), "improvement": 0.0}

    best_w, best_ll = 0.0, float("inf")
    for i in range(grid):
        w = i / (grid - 1)
        blended = [blend_probs(m, k, w) for m, k in zip(model_probs, market_probs, strict=True)]
        ll = log_loss(blended, outcomes)
        if ll < best_ll:
            best_w, best_ll = w, ll

    market_ll = log_loss(market_probs, outcomes)
    model_ll = log_loss(model_probs, outcomes)
    return {
        "weight": round(best_w, 4),
        "log_loss": best_ll,
        "market_log_loss": market_ll,
        "model_log_loss": model_ll,
        "improvement": market_ll - best_ll,
    }
