"""Walk-forward evaluation against closing prices.

Design rules, each of which exists because breaking it is the standard way
backtests lie:

**Refit before every prediction, using only earlier matches.** A model fitted
once on the whole history and evaluated on part of it has seen the answers. The
loop here refits at each step on a strictly prior window, and passes the
prediction date as the time-weighting reference so even the decay weights carry
no future information.

**Score against the closing price, not the opening price.** Opening lines are
soft. Beating them measures how fast you are, not how right you are, and the
edge evaporates the moment you try to get real money down at the open.

**Report the market's score next to the model's.** Almost every published sports
model looks impressive until you put the de-vigged market number beside it. If
the model's log loss is worse than the market's, the honest conclusion is that
the model has no standalone edge -- which is the usual result, and is why the
blend weight is fitted rather than assumed.

**Simulate the bankroll with the price you would actually get.** ``best`` uses
the best closing price across all books (what a line-shopper gets), ``sharp``
uses Pinnacle. The gap between those two is itself a finding: for most people
line shopping is a larger and far more reliable source of edge than modelling.
"""

from __future__ import annotations

import math
from datetime import timedelta

from src.blend import blend_probs, brier, log_loss, optimal_weight
from src.datafeeds.football_data import to_matches
from src.devig import DEFAULT_METHOD, devig
from src.edge import ev_per_unit, kelly_fraction
from src.models.dixon_coles import DixonColesModel

_OUTCOMES = ("home", "draw", "away")


def market_probs_from_row(row: str, key: str = "psc", method: str = DEFAULT_METHOD) -> dict | None:
    prices = row[key] if isinstance(row, dict) else None
    if not prices or not all(prices):
        return None
    fair = devig(list(prices), method=method)
    if any(f is None for f in fair):
        return None
    return dict(zip(_OUTCOMES, fair, strict=True))


def walk_forward(
    rows: list,
    min_train: int = 300,
    step: int = 10,
    xi: float = 0.0019,
    max_train_days: int = 1095,
    price_key: str = "psc",
    devig_method: str = DEFAULT_METHOD,
) -> dict:
    """Refit-and-predict across a chronological match list.

    ``step`` batches predictions between refits purely for speed: with step=10
    the model is at most 10 matches stale, which is negligible against a
    multi-season decay window and cuts fit count by an order of magnitude.
    """
    rows = sorted(rows, key=lambda r: r["date"])
    model_preds, market_preds, actuals, kept = [], [], [], []

    i = min_train
    model = None
    last_fit_at = -10**9
    while i < len(rows):
        if i - last_fit_at >= step or model is None:
            cutoff = rows[i]["date"]
            window = [
                r for r in rows[:i] if (cutoff - r["date"]) <= timedelta(days=max_train_days)
            ]
            if len(window) < min_train:
                window = rows[:i]
            try:
                model = DixonColesModel(max_goals=10).fit(
                    to_matches(window), xi=xi, reference_date=cutoff
                )
            except (ValueError, RuntimeError):
                model = None
            last_fit_at = i

        row = rows[i]
        i += 1
        if model is None or not model.knows(row["home_team"], row["away_team"]):
            continue
        market = market_probs_from_row(row, price_key, devig_method)
        if market is None:
            continue

        model_preds.append(model.match_odds(row["home_team"], row["away_team"]))
        market_preds.append(market)
        actuals.append(row["result"])
        kept.append(row)

    if not actuals:
        return {"n": 0, "error": "no evaluable matches"}

    weight_fit = optimal_weight(model_preds, market_preds, actuals)
    blended = [blend_probs(m, k, weight_fit["weight"]) for m, k in zip(model_preds, market_preds, strict=True)]

    return {
        "n": len(actuals),
        "model": {"log_loss": log_loss(model_preds, actuals), "brier": brier(model_preds, actuals)},
        "market": {"log_loss": log_loss(market_preds, actuals), "brier": brier(market_preds, actuals)},
        "blend": {
            "weight": weight_fit["weight"],
            "log_loss": log_loss(blended, actuals),
            "brier": brier(blended, actuals),
            "improvement_vs_market": weight_fit["market_log_loss"] - weight_fit["log_loss"],
        },
        "_rows": kept,
        "_model_preds": model_preds,
        "_market_preds": market_preds,
        "_blended": blended,
        "_actuals": actuals,
    }


def simulate_bankroll(
    result: dict,
    source: str = "blend",
    price_key: str = "maxc",
    min_ev: float = 0.02,
    kelly_multiplier: float = 0.25,
    max_stake: float = 0.02,
    start_bankroll: float = 1000.0,
) -> dict:
    """Flat-forward bankroll simulation over the walk-forward predictions.

    ``source`` picks which probability drives the bet: ``model``, ``market``
    (a control that should lose exactly the vig), or ``blend``.
    """
    preds = {
        "model": result.get("_model_preds"),
        "market": result.get("_market_preds"),
        "blend": result.get("_blended"),
    }[source]
    rows, actuals = result.get("_rows"), result.get("_actuals")
    if not preds:
        return {"n_bets": 0}

    bankroll = start_bankroll
    peak, max_dd = bankroll, 0.0
    staked = pnl = 0.0
    bets = wins = 0
    clv_sum = 0.0
    clv_positive = 0
    log_growth = 0.0
    # Per-unit returns of every settled bet. Kept because the significance of an
    # ROI depends entirely on the variance of these, and that variance is wildly
    # non-constant: a bet at 1.20 has a per-unit return in {-1, +0.2}, one at
    # 9.00 in {-1, +8}. Assuming unit variance (the common shortcut) can inflate
    # an apparent t-statistic several-fold on a longshot-heavy strategy.
    returns = []

    for p, row, actual in zip(preds, rows, actuals, strict=True):
        prices = row.get(price_key)
        ref = row.get("psc")
        if not prices or not all(prices):
            continue
        for name, price in zip(_OUTCOMES, prices, strict=True):
            prob = p.get(name)
            if not prob:
                continue
            ev = ev_per_unit(prob, price)
            if ev < min_ev:
                continue
            f = min(kelly_fraction(prob, price) * kelly_multiplier, max_stake)
            if f <= 0:
                continue
            stake = bankroll * f
            won = actual == name
            profit = stake * (price - 1.0) if won else -stake

            bankroll += profit
            staked += stake
            pnl += profit
            bets += 1
            wins += int(won)
            returns.append((price - 1.0) if won else -1.0)
            log_growth += math.log(max(1.0 + f * (price - 1.0), 1e-9)) if won else math.log(max(1.0 - f, 1e-9))

            # Closing line value: the price taken vs the sharp closing price for
            # the same selection. Positive CLV means the market moved toward the
            # bet after it was placed, which is the only leading indicator of
            # edge that does not need thousands of settled bets to read.
            if ref and all(ref):
                close = dict(zip(_OUTCOMES, ref, strict=True))[name]
                clv = price / close - 1.0
                clv_sum += clv
                clv_positive += int(clv > 0)

            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
            if bankroll <= 0:
                break

    roi = pnl / staked if staked else 0.0
    # t-statistic on the mean per-unit return, using the realised variance of
    # those returns. This answers the only question that matters about a
    # positive backtest ROI: is it more than a couple of standard errors from
    # zero, or is it noise dressed up as a strategy?
    if bets > 1:
        mean_r = sum(returns) / bets
        var_r = sum((r - mean_r) ** 2 for r in returns) / (bets - 1)
        se = math.sqrt(var_r / bets)
        tstat = mean_r / se if se > 0 else float("nan")
    else:
        mean_r, se, tstat = float("nan"), float("nan"), float("nan")
    return {
        "source": source,
        "price_key": price_key,
        "n_bets": bets,
        "hit_rate": wins / bets if bets else 0.0,
        "staked": round(staked, 2),
        "pnl": round(pnl, 2),
        "roi": roi,
        "mean_return": mean_r,
        "return_se": se,
        "roi_tstat": tstat,
        # Bets needed for this edge to clear 2 standard errors, i.e. how long
        # before you could distinguish it from luck. Usually a sobering number.
        "bets_for_significance": (int(4.0 * var_r / mean_r**2) if bets > 1 and mean_r > 0 else None),
        "final_bankroll": round(bankroll, 2),
        "max_drawdown": max_dd,
        "avg_clv": clv_sum / bets if bets else 0.0,
        "clv_hit_rate": clv_positive / bets if bets else 0.0,
        "log_growth_per_bet": log_growth / bets if bets else 0.0,
    }


def tune_xi(rows: list, candidates=(0.0, 0.001, 0.0019, 0.003, 0.005), **kw) -> dict:
    """Pick the time-decay rate by out-of-sample log loss, not in-sample fit."""
    scores = {}
    for xi in candidates:
        res = walk_forward(rows, xi=xi, **kw)
        if res.get("n"):
            scores[xi] = res["model"]["log_loss"]
    if not scores:
        return {"best_xi": None, "scores": {}}
    best = min(scores, key=scores.get)
    return {"best_xi": best, "scores": scores, "half_life_days": (math.log(2) / best) if best else None}


def walk_forward_totals(
    rows: list,
    min_train: int = 380,
    step: int = 10,
    xi: float = 0.0019,
    max_train_days: int = 1095,
    line: float = 2.5,
    devig_method: str = DEFAULT_METHOD,
) -> dict:
    """The same walk-forward, but on Over/Under 2.5 goals.

    This is the hypothesis worth testing. A goal model has no business beating a
    sharp 1X2 close -- that market is the most heavily traded on the board. But
    the model's totals price falls out of the *same* fitted score matrix as its
    1X2 price, for free, and the totals market is thinner and more mechanically
    priced. If a Dixon-Coles fit adds anything anywhere, this is where.
    """
    rows = sorted(rows, key=lambda r: r["date"])
    model_preds, market_preds, actuals, kept = [], [], [], []

    i, model, last_fit_at = min_train, None, -10**9
    while i < len(rows):
        if i - last_fit_at >= step or model is None:
            cutoff = rows[i]["date"]
            window = [r for r in rows[:i] if (cutoff - r["date"]) <= timedelta(days=max_train_days)]
            if len(window) < min_train:
                window = rows[:i]
            try:
                model = DixonColesModel(max_goals=10).fit(
                    to_matches(window), xi=xi, reference_date=cutoff
                )
            except (ValueError, RuntimeError):
                model = None
            last_fit_at = i

        row = rows[i]
        i += 1
        prices = row.get("ou25_psc")
        if model is None or not model.knows(row["home_team"], row["away_team"]):
            continue
        if not prices or not all(prices):
            continue
        fair = devig(list(prices), method=devig_method)
        if any(f is None for f in fair):
            continue

        mp = model.totals(row["home_team"], row["away_team"], line)
        model_preds.append({"Over": mp["Over"], "Under": mp["Under"]})
        market_preds.append({"Over": fair[0], "Under": fair[1]})
        total_goals = row["home_score"] + row["away_score"]
        actuals.append("Over" if total_goals > line else "Under")
        kept.append(row)

    if not actuals:
        return {"n": 0, "error": "no evaluable matches"}

    weight_fit = optimal_weight(model_preds, market_preds, actuals)
    blended = [blend_probs(m, k, weight_fit["weight"]) for m, k in zip(model_preds, market_preds, strict=True)]
    return {
        "n": len(actuals),
        "line": line,
        "model": {"log_loss": log_loss(model_preds, actuals), "brier": brier(model_preds, actuals)},
        "market": {"log_loss": log_loss(market_preds, actuals), "brier": brier(market_preds, actuals)},
        "blend": {
            "weight": weight_fit["weight"],
            "log_loss": log_loss(blended, actuals),
            "brier": brier(blended, actuals),
            "improvement_vs_market": weight_fit["market_log_loss"] - weight_fit["log_loss"],
        },
        "_rows": kept,
        "_model_preds": model_preds,
        "_market_preds": market_preds,
        "_blended": blended,
        "_actuals": actuals,
    }
