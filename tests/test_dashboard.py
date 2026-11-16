"""The UI must boot without spending quota or exposing a private journal."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import odds_api

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_locked_app_boots_without_feed_or_database(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr("src.ui.dashboard.setting", lambda name, default="": default)
    monkeypatch.setattr(odds_api, "get_odds", lambda *a, **k: (_ for _ in ()).throw(AssertionError("API called")))
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert any("locked" in w.value for w in at.warning)


def test_authenticated_navigation_is_offline_and_stakes_zero(monkeypatch, tmp_path):
    values = {"DASHBOARD_PASSWORD": "test-private", "RESEARCH_DATABASE_URL": str(tmp_path / "research.db")}
    monkeypatch.setattr("src.ui.dashboard.setting", lambda name, default="": values.get(name, default))
    monkeypatch.setattr(odds_api, "get_odds", lambda *a, **k: (_ for _ in ()).throw(AssertionError("API called")))
    at = AppTest.from_file(APP).run()
    at.text_input[0].set_value("wrong")
    at.button[0].click().run()
    assert at.error and not at.exception
    at.text_input[0].set_value("test-private")
    at.button[0].click().run()
    assert not at.exception
    assert any("GHS 0" in w.value for w in at.warning)
    for view in ("Results", "Actual history & backup", "Account", "Calculators", "Method", "Paper prices"):
        at.radio[0].set_value(view).run()
        assert not at.exception, view
