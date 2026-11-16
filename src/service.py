"""Quota-budgeted polling shared by the dashboard and scheduled worker."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from src import odds_api
from src.paper import LEAGUES, utc

MONTHLY_BUDGET = 400  # Leave headroom within a 500-credit plan.
DAILY_BUDGET = 12


def fetch(store, sport, kind="odds", now=None):
    if sport not in LEAGUES or kind not in ("odds", "scores"):
        raise ValueError("Unsupported feed request.")
    supplied_clock = now is not None
    now = now or datetime.now(UTC)
    cache_key = f"cache:{kind}:{sport}"
    cached = store.get(cache_key)
    ttl = timedelta(minutes=15) if kind == "odds" else timedelta(hours=24)
    if cached and now - utc(cached["fetched_at"]) < ttl:
        if kind == "scores":
            store.sync_scores(cached["events"])
        return cached["events"], cached.get("remaining"), True
    cost = 1 if kind == "odds" else 2
    # A single row lock serializes budget reservations across worker/UI sessions.
    # Failed requests remain charged conservatively; their cost is uncertain.
    try:
        if not store.postgres:
            store.execute("BEGIN IMMEDIATE")
        store._put("quota", "quota", {"month": "", "day": "", "monthly": 0, "daily": 0},
                   replace=False)
        row = store.execute("SELECT payload FROM journal WHERE key='quota'"
                            + (" FOR UPDATE" if store.postgres else "")).fetchone()
        budget = json.loads(row["payload"])
        month, day = now.strftime("%Y-%m"), now.strftime("%Y-%m-%d")
        if budget["month"] != month:
            budget.update(month=month, monthly=0)
        if budget["day"] != day:
            budget.update(day=day, daily=0)
        if budget["monthly"] + cost > MONTHLY_BUDGET or budget["daily"] + cost > DAILY_BUDGET:
            raise odds_api.OddsAPIError("Research quota budget reached; no request sent.")
        budget["monthly"] += cost
        budget["daily"] += cost
        store._put("quota", "quota", budget)
        store.conn.commit()
    except Exception:
        store.conn.rollback()
        raise
    if kind == "odds":
        events, remaining = odds_api.get_odds(sport, regions="eu", markets="h2h")
        events = [{**e, "_sport_key": sport, "_league": LEAGUES[sport]} for e in events]
    else:
        events, remaining = odds_api.get_scores(sport, days_from=3)
    # A response arriving after kickoff must never become a pre-kickoff close.
    received_at = now if supplied_clock else datetime.now(UTC)
    store._put(cache_key, "cache", {"events": events, "remaining": remaining,
               "fetched_at": received_at.isoformat()})
    store.conn.commit()
    if kind == "odds":
        store.capture_closes(events, received_at)
    else:
        store.sync_scores(events)
    return events, remaining, False


def poll(store, now=None):
    """Fetch only due closes/open results. No predictions or actual bets created.

    Near-close capture is best effort. A missing snapshot remains missing;
    delayed scheduler runs never relabel an in-play price as a close.
    """
    supplied_clock = now is not None
    now = now or datetime.now(UTC)
    records = store.rows("record")
    close_sports, score_sports = set(), set()
    for record in records:
        start = utc(record.get("commence_time"))
        sport = record.get("sport_key")
        if start is None or sport not in LEAGUES:
            continue
        if not record.get("closing_prob") and timedelta(0) < start - now <= timedelta(minutes=5):
            close_sports.add(sport)
        if not record.get("result") and timedelta(hours=3) < now - start <= timedelta(days=3):
            score_sports.add(sport)
    report = {"odds": 0, "scores": 0, "errors": []}
    for kind, sports in (("odds", close_sports), ("scores", score_sports)):
        for sport in sorted(sports):
            try:
                # Odds loaded earlier in the day cannot masquerade as a close.
                if kind == "odds":
                    cache = store.get(f"cache:odds:{sport}")
                    if cache and now - utc(cache["fetched_at"]) > timedelta(minutes=1):
                        store.execute("DELETE FROM journal WHERE key=?", (f"cache:odds:{sport}",))
                        store.conn.commit()
                events, _, cached = fetch(store, sport, kind, now if supplied_clock else None)
                if cached and kind == "odds":
                    cache = store.get(f"cache:odds:{sport}")
                    store.capture_closes(events, utc(cache["fetched_at"]))
                report[kind] += not cached
            except Exception:
                # requests/DB errors can embed credentials. Report identifiers only.
                report["errors"].append(f"{kind}:{sport}: request failed; check configuration/quota")
    return report
