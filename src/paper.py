"""Frozen paper-trading protocol. Estimates never authorize cash stakes.

All times are aware UTC. Market dispersion is a heuristic, not a confidence
interval: bookmakers are dependent observations. Protocol changes need a new
version and a new prospective sample.
"""
from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta

from src.edge import ev_per_unit, kelly_fraction, shrink_probability
from src.market import MarketView, dispersion, is_sharp

STRATEGY = "own-price-h2h-v1"
BOOKS = ("SportyBet", "bet365")
MIN_BOOKS = 8
MAX_AGE = timedelta(minutes=15)
MAX_DISPERSION = 0.035
MIN_EV = 0.01
MAX_EV = 0.20
MIN_EVENTS = 300
MIN_DAYS = 60
MAX_CLOSE_LAG = timedelta(minutes=5)

# Scope is fixed before collecting outcomes. No international cups, obscure
# leagues, alternate markets or basketball model claims.
LEAGUES = {
    "soccer_epl": "English Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "basketball_nba": "NBA",
}


def utc(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def sport_key(event: dict) -> str:
    return event.get("sport_key") or event.get("_sport_key") or ""


def fresh_event(event: dict, now: datetime) -> dict:
    """Discard old, future-dated, duplicated and incomplete book vectors."""
    clean, seen = [], set()
    expected = {event.get("home_team"), event.get("away_team")}
    if sport_key(event).startswith("soccer_"):
        expected.add("Draw")
    for book in event.get("bookmakers", []) or []:
        key = book.get("key")
        if not key or key in seen:
            continue
        markets = []
        for market in book.get("markets", []) or []:
            stamp = utc(market.get("last_update") or book.get("last_update"))
            outcomes = market.get("outcomes") or []
            if (market.get("key") != "h2h" or stamp is None
                    or not timedelta(0) <= now - stamp <= MAX_AGE
                    or {o.get("name") for o in outcomes} != expected
                    or len(outcomes) != len(expected)):
                continue
            if not all(isinstance(o.get("price"), (int, float))
                       and math.isfinite(o["price"]) and o["price"] > 1 for o in outcomes):
                continue
            markets.append(market)
        if markets:
            clean.append({**book, "markets": markets[:1]})
            seen.add(key)
    return {**event, "bookmakers": clean}


def assess(event: dict, book: str, selection: str, odds: float,
           observed_at: str, context_checked: bool = False,
           now: datetime | None = None) -> dict:
    """A reproducible assessment against raw fresh market prices.

    The offer is from a manually verified external book, never the best book
    in the feed. Team news is an exclusion checklist, never a probability bump.
    """
    now = now or datetime.now(UTC)
    start, observed = utc(event.get("commence_time")), utc(observed_at)
    reasons = []
    if sport_key(event) not in LEAGUES:
        reasons.append("Competition is outside the paper protocol.")
    if not event.get("id") or start is None or not now < start <= now + timedelta(hours=24):
        reasons.append("Requires an identified match starting within 24 hours.")
    if observed is None or not timedelta(0) <= now - observed <= MAX_AGE:
        reasons.append("Your price is missing, future-dated, or over 15 minutes old.")
    if book not in BOOKS:
        reasons.append("Bookmaker is outside your supported accounts.")
    if not isinstance(odds, (int, float)) or not math.isfinite(odds) or odds <= 1:
        raise ValueError("Decimal odds must be finite and greater than 1.")
    if not context_checked:
        reasons.append("Confirm lineups, motivation, market rules and price first.")

    view = MarketView(fresh_event(event, now), "h2h", calibrate=False)
    fair = view.raw_anchor().get(selection)
    sigma = dispersion(view.quotes, selection) if fair else None
    if view.n_books < MIN_BOOKS:
        reasons.append(f"Needs {MIN_BOOKS} fresh complete bookmaker quotes.")
    if not view.sharp:
        reasons.append("No fresh sharp benchmark is available.")
    if fair is None:
        reasons.append("Selection has no complete reference market.")
    if sigma is not None and sigma > MAX_DISPERSION:
        reasons.append("Bookmakers disagree too much.")
    conservative = shrink_probability(fair, sigma) if fair is not None else None
    ev = ev_per_unit(fair, odds) if fair is not None else None
    ev_low = ev_per_unit(conservative, odds) if conservative is not None else None
    if ev_low is not None and ev_low < MIN_EV:
        reasons.append("Conservative estimated EV is below +1%.")
    if ev is not None and ev > MAX_EV:
        reasons.append("Estimated EV exceeds 20%; suspect a price or market error.")
    return {
        "strategy": STRATEGY, "assessed_at": now.isoformat(),
        "event_id": event.get("id"), "sport_key": sport_key(event),
        "commence_time": event.get("commence_time"),
        "home": event.get("home_team"), "away": event.get("away_team"),
        "market": "h2h", "book": book, "selection": selection, "odds": odds,
        "observed_at": observed_at, "fair_prob": fair, "conservative_prob": conservative,
        "ev": ev, "conservative_ev": ev_low, "n_books": view.n_books, "sigma": sigma,
        "context_checked": context_checked, "eligible": not reasons, "reasons": reasons,
        "cash_stake": 0.0, "event_snapshot": event,
    }


def minimum_stake_check(bankroll: float, minimum: float, probability: float,
                        odds: float) -> dict:
    """Illustrative sizing only; never round a small Kelly stake upward."""
    if not all(math.isfinite(x) for x in (bankroll, minimum, probability, odds)):
        raise ValueError("Inputs must be finite.")
    if bankroll < 0 or minimum <= 0 or not 0 <= probability <= 1 or odds <= 1:
        raise ValueError("Invalid bankroll, minimum stake, probability or odds.")
    amount = math.floor(bankroll * min(0.01, 0.25 * kelly_fraction(probability, odds)) * 100) / 100
    return {"illustrative_stake": amount if amount >= minimum else 0.0,
            "below_minimum": amount < minimum, "cash_stake": 0.0}


def settle_h2h(record: dict, score: dict) -> str | None:
    """Grade only completed supported h2h matches with exact event/team identity.

    Soccer protocol covers league regulation-time 1X2; NBA moneyline includes
    overtime. Missing, cancelled, postponed or ambiguous results stay open.
    """
    if (record.get("market") != "h2h" or record.get("sport_key") not in LEAGUES
            or not score.get("completed") or score.get("id") != record.get("event_id")
            or score.get("home_team") != record.get("home")
            or score.get("away_team") != record.get("away")):
        return None
    try:
        if len(score.get("scores") or []) != 2:
            return None
        scores = {s["name"]: float(s["score"]) for s in score.get("scores") or []}
        h, a = scores[record["home"]], scores[record["away"]]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and x >= 0 and x.is_integer() for x in (h, a)):
        return None
    if h == a and record["sport_key"] == "basketball_nba":
        return None
    winner = record["home"] if h > a else record["away"] if a > h else "Draw"
    if record.get("selection") not in (record["home"], record["away"], "Draw"):
        return None
    return "win" if record["selection"] == winner else "loss"


def closing_benchmark(record: dict, event: dict, now: datetime | None = None) -> dict | None:
    """A sampled near-close, never a historical close invented after kickoff.

    Requires an observation in the final 5 minutes *before* recorded kickoff,
    and a sharp update in that same interval. Records feed/source timestamps.
    """
    now = now or datetime.now(UTC)
    start = utc(record.get("commence_time"))
    entry_time = utc(record.get("assessed_at") or record.get("placed_at"))
    if (start is None or not timedelta(0) < start - now <= MAX_CLOSE_LAG
            or entry_time is None or entry_time > now
            or event.get("id") != record.get("event_id")
            or event.get("commence_time") != record.get("commence_time")):
        return None
    clean = fresh_event(event, now)
    books = []
    for book in clean["bookmakers"]:
        stamp = utc(book["markets"][0].get("last_update") or book.get("last_update"))
        if is_sharp(book["key"]) and stamp and timedelta(0) <= now - stamp <= MAX_CLOSE_LAG:
            books.append(book)
    view = MarketView({**clean, "bookmakers": books}, "h2h", calibrate=False)
    probability = view.raw_anchor().get(record["selection"])
    if not probability:
        return None
    return {"closing_prob": probability,
            "close_observed_at": now.isoformat(), "close_event": event,
            "close_source": ",".join(view.sharp_books),
            "clv": record["odds"] * probability - 1.0}


def evidence(records: list[dict]) -> dict:
    """Descriptive event-clustered CLV screen, never automatic cash approval.

    One mean per event prevents multiple selections/books on one match from
    masquerading as independent evidence. Missing closes remain in coverage.
    A future strategy needs a separate untouched confirmation sample.
    """
    papers = [r for r in records if r.get("kind") == "paper" and r.get("strategy") == STRATEGY]
    events = {r["event_id"] for r in papers}
    grouped = {}
    for r in papers:
        value = r.get("clv")
        if value is not None and math.isfinite(value):
            grouped.setdefault(r["event_id"], []).append(value)
    values = [statistics.mean(v) for v in grouped.values()]
    mean = statistics.mean(values) if values else None
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    tstat = mean / se if se and mean is not None else None
    stamps = [utc(r.get("assessed_at")) for r in papers]
    stamps = [s for s in stamps if s]
    days = (max(stamps) - min(stamps)).days if len(stamps) > 1 else 0
    coverage = len(grouped) / len(events) if events else 0.0
    passed = bool(len(values) >= MIN_EVENTS and days >= MIN_DAYS and coverage >= 0.90
                  and mean is not None and mean > 0 and tstat is not None and tstat > 2)
    return {"events": len(events), "events_with_close": len(values), "days": days,
            "coverage": coverage, "mean_clv": mean, "tstat": tstat,
            "screen_passed": passed, "cash_stake": 0.0,
            "verdict": "Promising paper screen; independent confirmation required."
            if passed else "No demonstrated edge. Continue paper trading only."}


def promotion_ev(probability: float, decimal_odds: float, stake: float,
                 stake_returned: bool, qualifying_cost: float = 0) -> float:
    """Cash value of a genuinely free token, less known qualifying costs.

    Assumes no further rollover, withdrawal restrictions, tax or expiry loss.
    Cash-funded boosted bets use p*odds-1 separately; a token's face value is
    not cash and is never deducted as though it were a paid stake.
    """
    if (not all(math.isfinite(x) for x in (probability, decimal_odds, stake, qualifying_cost))
            or not 0 <= probability <= 1 or decimal_odds <= 1 or min(stake, qualifying_cost) < 0):
        raise ValueError("Invalid promotion inputs.")
    payout = decimal_odds if stake_returned else decimal_odds - 1
    return probability * stake * payout - qualifying_cost
