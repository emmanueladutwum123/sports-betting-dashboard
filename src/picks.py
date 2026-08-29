"""Turning a raw odds feed into ranked, sized, defensible betting opportunities.

What changed and why
--------------------
The first version of this module picked the *balance-point* Over/Under line --
the line closest to a fair 50/50 -- on the reasoning that it avoided both the
cheap near-certain end and the flashy extreme end of the board. That is sound
risk instinct and it is also, unfortunately, an anti-strategy. The balance-point
line is where the market has the *most* information and the least disagreement;
picking it maximises variance while guaranteeing an expected return of exactly
minus the vig. There is no line on a board that is inherently good to bet. Only
a line that is *mispriced relative to its fair value* is good to bet.

So this module now asks a different question. For every selection at every book:

1. Build a fair probability from the other books, de-vigged per book with Shin's
   method, pooled in log-odds space, weighted toward sharp books, and explicitly
   excluding the book being evaluated.
2. Compare the offered price to that fair probability. The difference is the
   edge; the edge times the price is the expected value.
3. Size it with fractional Kelly on a *shrunk* probability, where the shrinkage
   comes from how much the books actually disagree.
4. Flag everything that smells like a data artefact rather than an edge.

The output is ranked by expected value, and it is frequently empty. An empty
board is the correct output on most days: it means no book is currently offering
a price meaningfully better than the market's own consensus, which is the normal
state of a market that works.
"""

from __future__ import annotations

from src.devig import DEFAULT_METHOD
from src.edge import (
    DEFAULT_EXPOSURE_CAP,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_SHRINK_K,
    MIN_EV_THRESHOLD,
    Opportunity,
    classify,
    evaluate,
    normalise_slate,
)
from src.market import MarketView, dispersion, is_sharp
from src.probability import confidence_stars, fmt_stars

# Markets scanned by default. Totals are enumerated per distinct line.
SCAN_MARKETS = ("h2h", "totals")


def _totals_lines(event: dict) -> list:
    lines = set()
    for bm in event.get("bookmakers", []) or []:
        for market in bm.get("markets", []) or []:
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []) or []:
                if outcome.get("point") is not None:
                    lines.add(outcome["point"])
    return sorted(lines)


def scan_event(
    event: dict,
    min_ev: float = MIN_EV_THRESHOLD,
    devig_method: str = DEFAULT_METHOD,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    shrink_k: float = DEFAULT_SHRINK_K,
    model_probs: dict | None = None,
    include_negative: bool = False,
) -> list:
    """Every +EV selection on one event, ranked by expected value.

    ``model_probs`` optionally supplies an independent forecast keyed by market
    (``{"h2h": {...}, "totals_2.5": {...}}``); when present it is already
    expected to have been blended with the market by the caller, so it simply
    replaces the anchor for those outcomes.
    """
    views = [("h2h", None, MarketView(event, "h2h", None, devig_method))]
    for line in _totals_lines(event):
        views.append(("totals", line, MarketView(event, "totals", line, devig_method)))

    found = []
    for market_key, point, view in views:
        if view.n_books < 2:
            continue
        model_key = market_key if point is None else f"{market_key}_{point}"
        model_for_market = (model_probs or {}).get(model_key)

        for quote in view.quotes:
            anchor = view.anchor(exclude_book=quote.key)
            if not anchor:
                continue
            if model_for_market:
                anchor = {k: model_for_market.get(k, v) for k, v in anchor.items()}

            for selection, price in quote.prices.items():
                fair = anchor.get(selection)
                if not fair:
                    continue
                sigma = dispersion(view.quotes, selection)
                metrics = evaluate(fair, price, sigma, kelly_multiplier, shrink_k)
                if not include_negative and metrics["ev"] < min_ev:
                    continue

                found.append(
                    Opportunity(
                        event_id=event.get("id"),
                        commence_time=event.get("commence_time"),
                        league=event.get("_league"),
                        home=event.get("home_team"),
                        away=event.get("away_team"),
                        market=market_key,
                        point=point,
                        selection=selection,
                        book_key=quote.key,
                        book_title=quote.title,
                        odds=price,
                        fair_prob=fair,
                        shrunk_prob=metrics["shrunk_prob"],
                        model_prob=(model_for_market or {}).get(selection),
                        sigma=sigma,
                        n_books=view.n_books,
                        anchor_quality=view.anchor_quality(),
                        ev=metrics["ev"],
                        edge=metrics["edge"],
                        kelly=metrics["kelly"],
                        stake_fraction=metrics["stake_fraction"],
                        flags=classify(metrics["ev"], sigma, view.n_books, view.anchor_quality()),
                    )
                )

    found.sort(key=lambda o: -(o.ev or 0))
    return found


def scan_slate(
    events: list,
    min_ev: float = MIN_EV_THRESHOLD,
    devig_method: str = DEFAULT_METHOD,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    shrink_k: float = DEFAULT_SHRINK_K,
    exposure_cap: float = DEFAULT_EXPOSURE_CAP,
    drop_flagged: bool = True,
) -> list:
    """Scan many events and size the whole slate against one bankroll."""
    out = []
    for event in events:
        out.extend(scan_event(event, min_ev, devig_method, kelly_multiplier, shrink_k))
    if drop_flagged:
        out = [o for o in out if not any(f.startswith("implausible-EV") for f in o.flags)]
    out.sort(key=lambda o: -(o.ev or 0))
    return normalise_slate(out, exposure_cap)


# --------------------------------------------------------------------------
# Per-event summaries used by the fixture cards. Same shapes as before so the
# renderer is unchanged, but the numbers underneath are now built the right way:
# each book de-vigged separately, pooled in log-odds space, sharp-anchored.
# --------------------------------------------------------------------------

def summarize_h2h(event: dict, devig_method: str = DEFAULT_METHOD) -> dict | None:
    view = MarketView(event, "h2h", None, devig_method)
    if not view.quotes:
        return None
    anchor = view.anchor()
    if not anchor:
        return None

    outcomes = {}
    for name, fair in anchor.items():
        prices = [(q.title, q.prices[name]) for q in view.quotes if name in q.prices]
        if not prices:
            continue
        best_book, best_price = max(prices, key=lambda x: x[1])
        outcomes[name] = {
            "fair_prob": round(fair, 4),
            "fair_odds": round(1.0 / fair, 2) if fair else None,
            "avg_odds": round(len(prices) / sum(1.0 / p for _, p in prices), 2),
            "best_odds": best_price,
            "best_book": best_book,
            "book_count": len(prices),
            # Positive means the best available price beats fair value.
            "best_ev_pct": round((best_price * fair - 1.0) * 100, 2),
            "dispersion_pp": round(dispersion(view.quotes, name) * 100, 2),
        }
    if not outcomes:
        return None
    return {
        "outcomes": outcomes,
        "stars": confidence_stars(view.n_books),
        "book_count": view.n_books,
        "anchor": view.anchor_quality(),
        "sharp_books": view.sharp_books,
        "avg_margin_pct": round(
            100 * sum(q.margin for q in view.quotes) / len(view.quotes), 2
        ),
    }


def best_total_bet(event: dict, devig_method: str = DEFAULT_METHOD) -> dict | None:
    """The totals line and side with the best expected value, not the flattest.

    Returns the same keys the fixture card expects, plus the EV that justifies
    the selection. Returns ``None`` when no totals line offers positive EV --
    which is the honest answer far more often than not.
    """
    best = None
    for line in _totals_lines(event):
        view = MarketView(event, "totals", line, devig_method)
        if view.n_books < 2:
            continue
        anchor = view.anchor()
        if not anchor:
            continue

        entry = {"point": line, "book_count": view.n_books, "stars": confidence_stars(view.n_books)}
        for side in ("Over", "Under"):
            prices = [(q.title, q.prices[side]) for q in view.quotes if side in q.prices]
            if not prices:
                break
            book, price = max(prices, key=lambda x: x[1])
            fair = anchor.get(side)
            if not fair:
                break
            entry[f"fair_{side.lower()}"] = round(fair, 4)
            entry[f"best_{side.lower()}_odds"] = price
            entry[f"best_{side.lower()}_book"] = book
            entry[f"ev_{side.lower()}"] = price * fair - 1.0
        else:
            side = "Over" if entry["ev_over"] >= entry["ev_under"] else "Under"
            entry["side"] = side
            entry["ev"] = entry[f"ev_{side.lower()}"]
            if best is None or entry["ev"] > best["ev"]:
                best = entry

    if best is None:
        return None
    best["ev_pct"] = round(best["ev"] * 100, 2)
    best["positive_ev"] = best["ev"] > 0
    return best


# Kept under the old name so existing callers keep working.
pick_total_line = best_total_bet


def build_verdict(event: dict, h2h_summary: dict | None, total_pick: dict | None) -> str:
    """One-line plain-language read of the event."""
    home, away = event.get("home_team"), event.get("away_team")
    parts = []

    if h2h_summary:
        fav_name, fav = max(
            h2h_summary["outcomes"].items(), key=lambda kv: (kv[1]["fair_prob"] or 0)
        )
        pct = round((fav["fair_prob"] or 0) * 100)
        ev = fav["best_ev_pct"]
        verdict = f"+{ev:.1f}% EV at that price" if ev > 0 else "no value at any listed price"
        parts.append(
            f"{fav_name} ~{pct}% fair (anchor: {h2h_summary['anchor']}, "
            f"{h2h_summary['book_count']} books, avg margin {h2h_summary['avg_margin_pct']}%) — "
            f"best {fav['best_odds']} @ {fav['best_book']}, {verdict} "
            f"{fmt_stars(h2h_summary['stars'])}"
        )
    else:
        parts.append("No moneyline/1X2 priced — skip.")

    if total_pick:
        ev = total_pick["ev_pct"]
        if total_pick["positive_ev"]:
            odds = total_pick[f"best_{total_pick['side'].lower()}_odds"]
            book = total_pick[f"best_{total_pick['side'].lower()}_book"]
            parts.append(
                f"{total_pick['side']} {total_pick['point']} is the only +EV total "
                f"(+{ev:.1f}%) — {odds} @ {book}"
            )
        else:
            parts.append(f"No totals line offers value (best is {ev:.1f}% EV) — skip.")
    else:
        parts.append("No totals priced — skip.")

    return f"{home} vs {away}: " + " | ".join(parts)


def sharp_disagreement(event: dict) -> dict | None:
    """Where sharp books and soft books disagree most on the same market.

    A large, persistent gap between the sharp anchor and the recreational
    consensus is the cleanest structural signal available in a public odds feed:
    the soft side is shaded toward public sentiment, and the sharp side is where
    the money is. Bets are found on the soft side of that gap.
    """
    view = MarketView(event, "h2h")
    if not view.sharp or view.n_books < 3:
        return None
    from src.market import pool_fair_probs

    soft = pool_fair_probs([q for q in view.quotes if not is_sharp(q.key)])
    if not soft:
        return None
    gaps = {name: view.sharp[name] - soft.get(name, view.sharp[name]) for name in view.sharp}
    biggest = max(gaps, key=lambda n: abs(gaps[n]))
    return {
        "selection": biggest,
        "sharp_prob": round(view.sharp[biggest], 4),
        "soft_prob": round(soft.get(biggest, 0.0), 4),
        "gap_pp": round(gaps[biggest] * 100, 2),
        "sharp_books": view.sharp_books,
    }
