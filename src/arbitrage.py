"""Arbitrage and middles -- edge that does not depend on being right.

An arbitrage exists when the best available prices across books imply a booksum
below 1: back every outcome in the right proportion and the payout exceeds the
outlay whatever happens. A *middle* is the same idea in one dimension up -- take
Over at a low line at one book and Under at a higher line at another, and every
result landing strictly between them wins both sides, while everything else
wins one and loses one for roughly the cost of the vig.

These are worth surfacing prominently for a reason that has nothing to do with
their (small, quickly-closed) profit: an apparent arbitrage is far more often a
*data problem* than free money -- a stale quote, a mismatched line, a book that
has already taken the market down. So the same scan that finds them also
functions as a live integrity check on the odds feed. An arb over about 3% on a
liquid market should be read as "something in this row is wrong", not as profit.
"""

from __future__ import annotations

from src.market import best_price, collect_quotes


def stake_split(prices: list, bankroll: float = 1.0) -> list:
    """Stakes that equalise the payout across every outcome."""
    inv = [1.0 / p for p in prices]
    total = sum(inv)
    return [bankroll * i / total for i in inv]


def find_arbitrage(event: dict, market_key: str = "h2h", point=None) -> dict | None:
    """Best-price arbitrage across books for one market, or None."""
    quotes = collect_quotes(event, market_key, point)
    if len(quotes) < 2:
        return None
    outcomes = sorted({n for q in quotes for n in q.prices})
    best = {}
    for name in outcomes:
        price, key, title = best_price(quotes, name)
        if price is None:
            return None  # incomplete market -> no valid arb
        best[name] = (price, key, title)

    booksum = sum(1.0 / v[0] for v in best.values())
    if booksum >= 1.0:
        return None

    profit = 1.0 / booksum - 1.0
    stakes = stake_split([best[n][0] for n in outcomes])
    return {
        "market": market_key,
        "point": point,
        "booksum": round(booksum, 5),
        "profit_pct": round(profit * 100, 3),
        "legs": [
            {
                "selection": n,
                "odds": best[n][0],
                "book": best[n][2],
                "book_key": best[n][1],
                "stake_pct": round(s * 100, 2),
            }
            for n, s in zip(outcomes, stakes, strict=True)
        ],
        "suspect": profit > 0.03,
    }


def find_middles(event: dict, market_key: str = "totals") -> list:
    """Over at a low line vs Under at a higher line, across books.

    ``gap`` is the width of the window that wins both legs. ``cost_pct`` is what
    the position costs when the result lands outside the window and the two legs
    cancel -- negative cost means the middle is also an outright arbitrage, which
    is the strongest version of this and rare.
    """
    by_line = {}
    for bm in event.get("bookmakers", []) or []:
        for market in bm.get("markets", []) or []:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []) or []:
                pt, side, price = outcome.get("point"), outcome.get("name"), outcome.get("price")
                if pt is None or not price or price <= 1:
                    continue
                slot = by_line.setdefault(pt, {"Over": None, "Under": None})
                cur = slot.get(side)
                if cur is None or price > cur[0]:
                    slot[side] = (price, bm.get("title", bm.get("key", "?")))

    lines = sorted(by_line)
    out = []
    for lo in lines:
        over = by_line[lo].get("Over")
        if not over:
            continue
        for hi in lines:
            if hi <= lo:
                continue
            under = by_line[hi].get("Under")
            if not under:
                continue
            booksum = 1.0 / over[0] + 1.0 / under[0]
            out.append(
                {
                    "over_line": lo,
                    "over_odds": over[0],
                    "over_book": over[1],
                    "under_line": hi,
                    "under_odds": under[0],
                    "under_book": under[1],
                    "gap": round(hi - lo, 2),
                    "cost_pct": round((booksum - 1.0) * 100, 2),
                    "arb": booksum < 1.0,
                }
            )
    out.sort(key=lambda m: (-m["gap"], m["cost_pct"]))
    return out
