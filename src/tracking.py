"""Durable-capable private journal; SQLite locally or PostgreSQL in hosting.

Payloads are JSON to preserve immutable raw snapshots and protocol versions.
No real betting credentials are stored. All UI access is password protected.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import uuid
from datetime import UTC, datetime

from src.paper import BOOKS, STRATEGY, closing_benchmark, settle_h2h, utc


class Store:
    def __init__(self, location: str = "research.db"):
        if not location.strip():
            raise ValueError("An empty storage location would create a temporary database.")
        self.postgres = location.startswith(("postgres://", "postgresql://"))
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row

            self.conn = psycopg.connect(location, row_factory=dict_row, connect_timeout=10)
        else:
            self.conn = sqlite3.connect(location, timeout=15)
            self.conn.row_factory = sqlite3.Row
        self.execute("CREATE TABLE IF NOT EXISTS journal (key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")
        self.conn.commit()

    def execute(self, sql, params=()):
        return self.conn.execute(sql.replace("?", "%s") if self.postgres else sql, params)

    def close(self):
        self.conn.close()

    def _put(self, key, kind, payload, replace=True):
        encoded = json.dumps(payload, allow_nan=False)
        suffix = "DO UPDATE SET payload=excluded.payload" if replace else "DO NOTHING"
        self.execute(f"INSERT INTO journal(key,kind,payload) VALUES(?,?,?) ON CONFLICT(key) {suffix}",
                     (key, kind, encoded))

    def get(self, key, default=None):
        row = self.execute("SELECT payload FROM journal WHERE key=?", (key,)).fetchone()
        return json.loads(row["payload"]) if row else default

    def rows(self, kind):
        return [json.loads(r["payload"]) for r in self.execute(
            "SELECT payload FROM journal WHERE kind=? ORDER BY key", (kind,)).fetchall()]

    def settings(self):
        return self.get("settings", {"bankroll": 0.0, "minimum_stake": 1.0, "books": list(BOOKS)})

    def save_settings(self, value):
        if (not all(math.isfinite(value[k]) for k in ("bankroll", "minimum_stake"))
                or value["bankroll"] < 0 or value["minimum_stake"] <= 0
                or not set(value["books"]).issubset(BOOKS)):
            raise ValueError("Invalid account settings.")
        self._put("settings", "settings", value)
        self.conn.commit()

    def save_assessment(self, assessment):
        """Persist every price/check; freeze first eligible offer per event/book/side."""
        key = "assessment:" + uuid.uuid4().hex
        identity = [STRATEGY, assessment["event_id"], assessment["book"], assessment["selection"]]
        paper_key = "paper:" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        try:
            self._put(key, "assessment", {**assessment, "id": key})
            if assessment["eligible"]:
                self._put(paper_key, "record", {**assessment, "id": paper_key, "kind": "paper",
                          "stake": 1.0, "result": None, "pnl": None}, replace=False)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return paper_key if assessment["eligible"] else None

    def record_actual(self, value, key=None):
        validate_actual(value)
        key = key or "actual:" + uuid.uuid4().hex
        self._put(key, "record", {**value, "id": key, "kind": "actual"}, replace=False)
        self.conn.commit()
        return key

    def settle(self, key, result, source="manual"):
        if result not in ("win", "loss", "push", "void"):
            raise ValueError("Invalid result.")
        r = self.get(key)
        if r is None:
            raise KeyError(key)
        r.update(result=result, pnl=round(r["stake"] * (r["odds"] - 1), 8)
                 if result == "win" else -r["stake"] if result == "loss" else 0.0,
                 settled_at=datetime.now(UTC).isoformat(), settlement_source=source)
        self._put(key, "record", r)
        self.conn.commit()

    def sync_scores(self, scores):
        by_id = {s["id"]: s for s in scores}
        count = 0
        for record in self.rows("record"):
            if record.get("result"):
                continue
            score = by_id.get(record.get("event_id"))
            result = settle_h2h(record, score) if score else None
            if result:
                self.settle(record["id"], result, "Odds API completed score")
                count += 1
        return count

    def capture_closes(self, events, now=None):
        by_id = {e["id"]: e for e in events}
        count = 0
        for record in self.rows("record"):
            event = by_id.get(record.get("event_id"))
            close = closing_benchmark(record, event, now) if event else None
            if close and (not record.get("close_observed_at")
                          or utc(close["close_observed_at"]) > utc(record["close_observed_at"])):
                self._put(record["id"], "record", {**record, **close})
                count += 1
        self.conn.commit()
        return count

    def backup(self):
        return json.dumps({"version": 1, "rows": [dict(r) for r in self.execute(
            "SELECT key,kind,payload FROM journal WHERE kind IN ('settings','record','assessment') ORDER BY key").fetchall()]}, allow_nan=False)

    def restore(self, text):
        data = json.loads(text)
        if data.get("version") != 1 or not isinstance(data.get("rows"), list):
            raise ValueError("Unsupported backup format.")
        rows = data["rows"]
        for row in rows:
            if row.get("kind") not in ("record", "settings", "assessment"):
                raise ValueError("Unknown journal row type.")
            payload = json.loads(row["payload"])
            json.dumps(payload, allow_nan=False)
            if row["kind"] == "record" and payload.get("kind") == "actual":
                validate_actual(payload)
        before = self.execute("SELECT COUNT(*) AS n FROM journal").fetchone()["n"]
        try:
            for row in rows:
                self._put(row["key"], row["kind"], json.loads(row["payload"]), replace=False)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return self.execute("SELECT COUNT(*) AS n FROM journal").fetchone()["n"] - before

    def import_actual_csv(self, text):
        """Atomic, repeatable import; accepts this template or legacy bets.csv.

        Missing results remain open. No results are inferred from reported
        losses or handoff tickets. Identical rows deduplicate; use distinct
        bet_id values for otherwise identical genuine bets.
        """
        parsed = []
        for n, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
            try:
                v = {k: row.get(k) or None for k in (
                    "event_id", "sport_key", "commence_time", "home", "away", "selection", "book")}
                v.update(market=row.get("market") or "h2h",
                         placed_at=row.get("placed_at"), odds=float(row.get("odds") or row.get("odds_taken")),
                         stake=float(row["stake"]), result=row.get("result") or None,
                         notes=row.get("notes") or "")
                validate_actual(v)
                v["pnl"] = (v["stake"] * (v["odds"] - 1) if v["result"] == "win"
                            else -v["stake"] if v["result"] == "loss"
                            else 0.0 if v["result"] else None)
                identity = row.get("bet_id") or row.get("id") or json.dumps(v, sort_keys=True)
                key = "actual:csv:" + hashlib.sha256(identity.encode()).hexdigest()
                parsed.append((key, {**v, "id": key, "kind": "actual"}))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"CSV row {n}: invalid odds, stake, date or result.") from exc
        before = len(self.rows("record"))
        try:
            for key, payload in parsed:
                self._put(key, "record", payload, replace=False)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(self.rows("record")) - before


def validate_actual(value):
    if (not all(isinstance(value.get(k), (int, float)) and math.isfinite(value[k])
                for k in ("odds", "stake")) or value["odds"] <= 1 or value["stake"] <= 0
            or not value.get("selection") or not value.get("book")
            or utc(value.get("placed_at")) is None
            or value.get("result") not in (None, "win", "loss", "push", "void")):
        raise ValueError("Actual bet needs a book, selection, aware date, positive stake and odds >1.")


def profit_summary(records, kind):
    rows = [r for r in records if r.get("kind") == kind]
    settled = [r for r in rows if r.get("result")]
    # Pushes return stake but still consumed turnover; voids are excluded.
    staked = sum(r["stake"] for r in settled if r["result"] != "void")
    pnl = sum(r.get("pnl") or 0 for r in settled)
    return {"count": len(rows), "open": sum(not r.get("result") for r in rows),
            "settled": len(settled), "turnover": staked, "pnl": pnl,
            "roi": pnl / staked if staked else None}
