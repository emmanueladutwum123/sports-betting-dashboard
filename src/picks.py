"""Turns a raw Odds API event into a market summary + a defensible recommended
pick, following the user's saved methodology:
  - never the extreme end of an O/U board — pick the "balance point" line
    (where Over/Under are closest to a fair 50/50), which naturally avoids
    both the cheap near-certain Over and the juicy-looking extreme Under.
  - state a confidence rating from real market liquidity, not vibes.
  - if a market isn't quoted by any book, say so — never invent a number.
"""
from src.probability import confidence_stars, devig, fmt_stars


def _collect_h2h(event: dict) -> dict:
    """{'Team A': [(book_title, price), ...], ...}"""
    per_outcome = {}
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                per_outcome.setdefault(outcome["name"], []).append(
                    (bm.get("title", bm.get("key", "?")), outcome["price"])
                )
    return per_outcome


def _collect_totals(event: dict) -> dict:
    """{point: {'Over': [(book,price)], 'Under': [(book,price)]}}"""
    lines: dict = {}
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                point = outcome.get("point")
                side = outcome["name"]
                lines.setdefault(point, {"Over": [], "Under": []})
                lines[point][side].append(
                    (bm.get("title", bm.get("key", "?")), outcome["price"])
                )
    return lines


def summarize_h2h(event: dict) -> dict | None:
    """Returns {'outcomes': {team: {...}}, 'stars': n, 'book_count': n}, or None
    if no book quotes this event's moneyline/1X2 at all."""
    per_outcome = _collect_h2h(event)
    if not per_outcome:
        return None

    names = list(per_outcome.keys())
    avg_odds = [sum(p for _, p in per_outcome[n]) / len(per_outcome[n]) for n in names]
    fair_probs = devig(avg_odds)

    book_counts = [len(per_outcome[n]) for n in names]
    total_books = max(book_counts) if book_counts else 0

    outcomes = {}
    for name, avg, fair, quotes in zip(names, avg_odds, fair_probs, [per_outcome[n] for n in names]):
        best_book, best_price = max(quotes, key=lambda x: x[1])
        outcomes[name] = {
            "avg_odds": round(avg, 2),
            "fair_prob": round(fair, 3) if fair is not None else None,
            "best_odds": best_price,
            "best_book": best_book,
            "book_count": len(quotes),
        }
    return {"outcomes": outcomes, "stars": confidence_stars(total_books), "book_count": total_books}


def pick_total_line(event: dict) -> dict | None:
    """Balance-point O/U line selection. Returns None if no totals quoted."""
    lines = _collect_totals(event)
    candidates = []
    for point, sides in lines.items():
        if not sides["Over"] or not sides["Under"]:
            continue
        avg_over = sum(p for _, p in sides["Over"]) / len(sides["Over"])
        avg_under = sum(p for _, p in sides["Under"]) / len(sides["Under"])
        fair_over, fair_under = devig([avg_over, avg_under])
        if fair_over is None:
            continue
        best_over = max(sides["Over"], key=lambda x: x[1])
        best_under = max(sides["Under"], key=lambda x: x[1])
        book_count = min(len(sides["Over"]), len(sides["Under"]))
        candidates.append(
            {
                "point": point,
                "fair_over": round(fair_over, 3),
                "fair_under": round(fair_under, 3),
                "avg_over_odds": round(avg_over, 2),
                "avg_under_odds": round(avg_under, 2),
                "best_over_odds": best_over[1],
                "best_over_book": best_over[0],
                "best_under_odds": best_under[1],
                "best_under_book": best_under[0],
                "book_count": book_count,
                "balance_gap": abs(fair_over - 0.5),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["balance_gap"])
    chosen = candidates[0]
    chosen["side"] = "Over" if chosen["fair_over"] >= chosen["fair_under"] else "Under"
    chosen["all_lines_considered"] = sorted(c["point"] for c in candidates)
    chosen["stars"] = confidence_stars(chosen["book_count"])
    return chosen


def build_verdict(event: dict, h2h_summary: dict | None, total_pick: dict | None) -> str:
    """Plain-language verdict line combining the moneyline/1X2 favorite and the
    balance-point total, mirroring the saved analyst methodology's output style."""
    home, away = event.get("home_team"), event.get("away_team")
    parts = []

    if h2h_summary:
        favorite = max(h2h_summary["outcomes"].items(), key=lambda kv: (kv[1]["fair_prob"] or 0))
        fav_name, fav_data = favorite
        pct = round((fav_data["fair_prob"] or 0) * 100)
        parts.append(
            f"{fav_name} favored (~{pct}% market-implied) — best price {fav_data['best_odds']} "
            f"@ {fav_data['best_book']} {fmt_stars(h2h_summary['stars'])}"
        )
    else:
        parts.append("No moneyline/1X2 odds quoted by any tracked book — skip this market.")

    if total_pick:
        side = total_pick["side"]
        pct = round((total_pick["fair_over"] if side == "Over" else total_pick["fair_under"]) * 100)
        odds = total_pick["best_over_odds"] if side == "Over" else total_pick["best_under_odds"]
        book = total_pick["best_over_book"] if side == "Over" else total_pick["best_under_book"]
        parts.append(
            f"{side} {total_pick['point']} the balance-point line (~{pct}% fair) — "
            f"best price {odds} @ {book} {fmt_stars(total_pick['stars'])}"
        )
    else:
        parts.append("No totals market quoted — skip O/U on this one.")

    return f"{home} vs {away}: " + " | ".join(parts)
