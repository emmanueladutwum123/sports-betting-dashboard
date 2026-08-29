import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_event(bookmakers, home="Arsenal", away="Chelsea", event_id="evt1"):
    """Build an Odds API-shaped event from {book_key: {market: {outcome: price}}}."""
    out = []
    for key, markets in bookmakers.items():
        mk = []
        for market_key, spec in markets.items():
            if market_key == "totals":
                outcomes = [
                    {"name": name, "price": price, "point": point}
                    for point, sides in spec.items()
                    for name, price in sides.items()
                ]
            else:
                outcomes = [{"name": n, "price": p} for n, p in spec.items()]
            mk.append({"key": market_key, "outcomes": outcomes})
        out.append({"key": key, "title": key.title(), "markets": mk})
    return {"id": event_id, "home_team": home, "away_team": away,
            "commence_time": "2026-09-01T14:00:00Z", "bookmakers": out}
