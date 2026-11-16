"""Regression checks for money/data failures, using synthetic timestamped odds."""
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from conftest import make_event

from src import paper, service
from src.market import MarketView
from src.odds_api import OddsAPIError
from src.tracking import Store, profit_summary

NOW = datetime(2026, 11, 15, 12, tzinfo=UTC)


def event():
    books = {"pinnacle": {"h2h": {"Arsenal": 2.0, "Draw": 3.5, "Chelsea": 4.0}}}
    books.update({f"book{i}": deepcopy(books["pinnacle"]) for i in range(7)})
    e = make_event(books)
    e.update(sport_key="soccer_epl", commence_time=(NOW + timedelta(hours=2)).isoformat())
    for book in e["bookmakers"]:
        book["last_update"] = NOW.isoformat()
    return e


def assessment(e=None, **kw):
    return paper.assess(e or event(), kw.get("book", "SportyBet"), "Arsenal",
                        kw.get("odds", 2.3), kw.get("observed_at", NOW.isoformat()),
                        kw.get("context_checked", True), now=NOW)


def test_actual_positive_price_is_paper_only():
    a = assessment()
    assert a["eligible"] and a["conservative_ev"] > 0.01
    assert a["cash_stake"] == 0
    assert a["event_snapshot"] == event()


def test_empty_database_location_cannot_silently_make_a_temporary_ledger():
    with pytest.raises(ValueError, match="temporary database"):
        Store("")


def test_unfitted_totals_line_is_not_calibrated_even_when_research_opts_in():
    e = make_event({"pinnacle": {"totals": {3.5: {"Over": 1.6, "Under": 2.4}}}})
    e["sport_key"] = "soccer_epl"
    view = MarketView(e, "totals", 3.5, calibrate=True)
    assert view.anchor() == view.raw_anchor()


@pytest.mark.parametrize("change", ["old", "future", "missing", "duplicate", "incomplete"])
def test_unusable_reference_prices_fail_closed(change):
    e = event()
    b = e["bookmakers"][0]
    if change == "old":
        b["last_update"] = (NOW - timedelta(minutes=16)).isoformat()
    elif change == "future":
        b["last_update"] = (NOW + timedelta(minutes=1)).isoformat()
    elif change == "missing":
        del b["last_update"]
    elif change == "duplicate":
        e["bookmakers"][-1] = deepcopy(b)
    else:
        b["markets"][0]["outcomes"] = b["markets"][0]["outcomes"][:2]
    assert not assessment(e)["eligible"]


@pytest.mark.parametrize("kwargs", [
    {"odds": 1.5}, {"context_checked": False}, {"book": "Unavailable book"},
    {"observed_at": "2026-11-15T11:00:00Z"}, {"observed_at": "2026-11-15T12:01:00Z"},
    {"observed_at": "2026-11-15T12:00:00"}, {"odds": 8.0},
])
def test_price_context_and_own_book_gates(kwargs):
    a = assessment(**kwargs)
    assert not a["eligible"] and a["reasons"] and a["cash_stake"] == 0


@pytest.mark.parametrize("odds", [float("nan"), float("inf"), 1.0, -1.0])
def test_nonfinite_or_invalid_manual_prices_are_rejected(odds):
    with pytest.raises(ValueError):
        assessment(odds=odds)


def test_unknown_competition_and_started_match_fail():
    e = event()
    e["sport_key"] = "soccer_uefa_nations_league"
    assert not assessment(e)["eligible"]
    e = event()
    e["commence_time"] = NOW.isoformat()
    assert not assessment(e)["eligible"]


def test_below_minimum_never_rounded_up():
    result = paper.minimum_stake_check(15, 1, 0.55, 2.0)
    assert result["below_minimum"] and result["illustrative_stake"] == 0
    assert result["cash_stake"] == 0


def test_repeated_prices_are_saved_but_first_paper_entry_is_frozen(tmp_path):
    path = str(tmp_path / "journal.db")
    store = Store(path)
    key = store.save_assessment(assessment())
    store.save_assessment(assessment(odds=2.35))
    store.save_assessment(assessment(odds=1.4))
    assert len(store.rows("assessment")) == 3
    assert len(store.rows("record")) == 1
    assert store.get(key)["odds"] == 2.3
    store.close()
    store = Store(path)
    assert store.get(key)["odds"] == 2.3
    store.close()


def test_backup_restores_snapshots_once_without_quota_reset():
    store, restored = Store(":memory:"), Store(":memory:")
    store.save_assessment(assessment())
    store._put("quota", "quota", {"monthly": 5})
    store.conn.commit()
    backup = store.backup()
    assert restored.restore(backup) == 2
    assert restored.restore(backup) == 0
    assert restored.rows("record")[0]["event_snapshot"] == event()
    assert restored.get("quota") is None
    store.close()
    restored.close()


def test_completed_results_settle_exact_matches_only():
    store = Store(":memory:")
    key = store.save_assessment(assessment())
    score = {"id": "evt1", "home_team": "Arsenal", "away_team": "Chelsea",
             "completed": False, "scores": [{"name": "Arsenal", "score": "2"},
                                              {"name": "Chelsea", "score": "1"}]}
    assert store.sync_scores([score]) == 0
    score["completed"] = True
    assert store.sync_scores([{**score, "id": "wrong"}]) == 0
    assert store.sync_scores([score]) == 1
    assert store.sync_scores([score]) == 0
    assert store.get(key)["pnl"] == pytest.approx(1.3)
    assert profit_summary(store.rows("record"), "actual")["count"] == 0
    store.close()


def test_missing_and_ambiguous_scores_stay_open():
    r = {**assessment(), "kind": "paper"}
    base = {"id": "evt1", "home_team": "Arsenal", "away_team": "Chelsea", "completed": True}
    for scores in (None, [], [{"name": "Arsenal", "score": "NaN"},
                              {"name": "Chelsea", "score": "1"}]):
        assert paper.settle_h2h(r, {**base, "scores": scores}) is None


def test_closes_are_only_captured_before_kickoff_with_source_times():
    r = {**assessment(), "kind": "paper"}
    e = event()
    start = paper.utc(e["commence_time"])
    close_time = start - timedelta(minutes=2)
    for b in e["bookmakers"]:
        b["last_update"] = close_time.isoformat()
    close = paper.closing_benchmark(r, e, close_time)
    assert close["close_source"] == "pinnacle"
    assert close["clv"] == pytest.approx(r["odds"] * close["closing_prob"] - 1)
    assert paper.closing_benchmark(r, e, start) is None
    assert paper.closing_benchmark(r, e, start + timedelta(seconds=1)) is None
    assert paper.closing_benchmark(r, e, start - timedelta(minutes=6)) is None
    e["bookmakers"][0]["last_update"] = (close_time - timedelta(minutes=6)).isoformat()
    assert paper.closing_benchmark(r, e, close_time) is None


def test_repeated_correlated_rows_do_not_inflate_evidence():
    r = {**assessment(), "kind": "paper", "clv": 0.02}
    ev = paper.evidence([r] * 500)
    assert ev["events_with_close"] == 1 and not ev["screen_passed"]
    assert ev["cash_stake"] == 0


def test_zero_variance_and_missing_close_samples_do_not_pass():
    rows = [{**assessment(), "kind": "paper", "clv": 0.02,
             "event_id": str(i), "assessed_at": (NOW + timedelta(days=i)).isoformat()}
            for i in range(310)]
    assert not paper.evidence(rows)["screen_passed"]
    for i, r in enumerate(rows):
        r["clv"] = (0.01 + (i % 3) * 0.01) if i < 250 else None
    assert not paper.evidence(rows)["screen_passed"]


def test_passing_screen_still_requires_confirmation_and_no_cash():
    rows = [{**assessment(), "kind": "paper", "clv": 0.01 + (i % 3) * 0.01,
             "event_id": str(i), "assessed_at": (NOW + timedelta(days=i // 4)).isoformat()}
            for i in range(310)]
    ev = paper.evidence(rows)
    assert ev["screen_passed"] and ev["cash_stake"] == 0
    assert "confirmation required" in ev["verdict"]


def test_csv_import_is_atomic_validated_and_repeatable():
    store = Store(":memory:")
    header = "bet_id,placed_at,book,selection,odds,stake,result\n"
    good = "abc,2026-11-15T10:00:00Z,SportyBet,France,1.4,5,loss\n"
    assert store.import_actual_csv(header + good) == 1
    assert store.import_actual_csv(header + good) == 0
    assert profit_summary(store.rows("record"), "actual")["pnl"] == -5
    bad = "def,2026-11-15T10:00:00Z,SportyBet,Greece,NaN,5,win\n"
    with pytest.raises(ValueError, match="CSV row"):
        store.import_actual_csv(header + good.replace("abc", "another") + bad)
    assert len(store.rows("record")) == 1
    store.close()


def test_turnover_includes_pushes_excludes_voids():
    rows = [{"kind": "actual", "stake": 10, "result": r, "pnl": p}
            for r, p in (("win", 10), ("push", 0), ("void", 0))]
    summary = profit_summary(rows, "actual")
    assert summary["turnover"] == 20 and summary["roi"] == 0.5


def test_quota_cache_and_daily_limit(monkeypatch):
    store = Store(":memory:")
    calls = []
    monkeypatch.setattr(service.odds_api, "get_odds", lambda *a, **k: (calls.append(k) or [event()], "100"))
    assert not service.fetch(store, "soccer_epl", now=NOW)[2]
    assert service.fetch(store, "soccer_epl", now=NOW + timedelta(minutes=1))[2]
    assert len(calls) == 1 and calls[0] == {"regions": "eu", "markets": "h2h"}
    budget = store.get("quota")
    budget["daily"] = service.DAILY_BUDGET
    store._put("quota", "quota", budget)
    store.conn.commit()
    with pytest.raises(OddsAPIError, match="budget"):
        service.fetch(store, "soccer_epl", now=NOW + timedelta(minutes=16))
    assert len(calls) == 1
    store.close()


def test_failed_fetch_still_reserves_budget(monkeypatch):
    store = Store(":memory:")

    def fail(*args, **kw):
        raise RuntimeError("network failed")

    monkeypatch.setattr(service.odds_api, "get_odds", fail)
    with pytest.raises(RuntimeError):
        service.fetch(store, "soccer_epl", now=NOW)
    assert store.get("quota")["daily"] == 1
    store.close()


def test_worker_does_not_poll_when_no_records_exist(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr(service, "fetch", lambda *a, **k: pytest.fail("No API calls expected"))
    assert service.poll(store, NOW) == {"odds": 0, "scores": 0, "errors": []}
    store.close()


def test_worker_fetches_due_closes_and_never_backfills_late_closes(monkeypatch):
    store = Store(":memory:")
    key = store.save_assessment(assessment())
    start = paper.utc(store.get(key)["commence_time"])
    calls = []

    def fake_fetch(store, sport, kind, now):
        calls.append((sport, kind))
        return [], None, False

    monkeypatch.setattr(service, "fetch", fake_fetch)
    service.poll(store, start - timedelta(minutes=2))
    assert calls == [("soccer_epl", "odds")]
    calls.clear()
    service.poll(store, start + timedelta(minutes=1))
    assert calls == []
    service.poll(store, start + timedelta(hours=4))
    assert calls == [("soccer_epl", "scores")]
    store.close()


def test_free_token_terms_are_distinct_from_paid_boost():
    assert paper.promotion_ev(0.5, 3, 10, False, 2) == 8
    assert paper.promotion_ev(0.5, 3, 10, True, 2) == 13
    with pytest.raises(ValueError):
        paper.promotion_ev(float("nan"), 3, 10, False)


def test_backup_validation_happens_before_writes():
    store = Store(":memory:")
    with pytest.raises(ValueError):
        store.restore(json.dumps({"version": 1, "rows": [
            {"key": "settings", "kind": "settings", "payload": "{}"},
            {"key": "bad", "kind": "unknown", "payload": "{}"}]}))
    assert store.get("settings") is None
    store.close()
