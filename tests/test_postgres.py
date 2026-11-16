"""CI checks the actual PostgreSQL backend against a disposable service."""
import os
from datetime import timedelta

import pytest
from test_paper_tracking import NOW, assessment, event

from src import paper, service
from src.tracking import Store


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="Disposable PostgreSQL not configured")
def test_postgres_journal_transactions_and_worker(monkeypatch):
    store = Store(os.environ["TEST_POSTGRES_URL"])
    try:
        store.execute("DELETE FROM journal")
        store.conn.commit()
        store.save_settings({"bankroll": 15.0, "minimum_stake": 1.0, "books": ["SportyBet"]})
        key = store.save_assessment(assessment())
        store.save_assessment(assessment())
        assert len(store.rows("record")) == 1
        assert store.get(key)["odds"] == 2.3
        assert store.settings()["bankroll"] == 15
        monkeypatch.setattr(service.odds_api, "get_odds", lambda *a, **kw: ([event()], "400"))
        service.fetch(store, "soccer_epl", now=NOW)
        assert store.get("quota")["monthly"] == 1
        assert service.fetch(store, "soccer_epl", now=NOW)[2]
        close_event = event()
        close_time = paper.utc(close_event["commence_time"]) - timedelta(minutes=1)
        for b in close_event["bookmakers"]:
            b["last_update"] = close_time.isoformat()
        assert store.capture_closes([close_event], close_time) == 1
        assert store.get(key)["clv"] > 0
        store.settle(key, "loss")
        assert store.get(key)["pnl"] == -1
        assert store.restore(store.backup()) == 0
        store.execute("DELETE FROM journal")
        store.conn.commit()
    finally:
        store.close()
