"""Tests for model/market blending and the CLV ledger."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ledger  # noqa: E402
from src.blend import blend_probs, brier, log_loss, optimal_weight  # noqa: E402


def test_blend_endpoints_recover_each_input():
    model = {"home": 0.60, "away": 0.40}
    market = {"home": 0.40, "away": 0.60}
    assert blend_probs(model, market, 0.0)["home"] == pytest.approx(0.40)
    assert blend_probs(model, market, 1.0)["home"] == pytest.approx(0.60)


def test_blend_stays_a_valid_distribution_at_every_weight():
    model = {"home": 0.7, "draw": 0.2, "away": 0.1}
    market = {"home": 0.4, "draw": 0.3, "away": 0.3}
    for i in range(11):
        blended = blend_probs(model, market, i / 10)
        assert sum(blended.values()) == pytest.approx(1.0)
        assert all(0 < p < 1 for p in blended.values())


def test_blend_is_monotone_in_the_weight():
    model, market = {"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}
    seq = [blend_probs(model, market, w / 10)["a"] for w in range(11)]
    assert all(x < y for x, y in zip(seq, seq[1:], strict=False))  # deliberately offset pairwise


def test_blend_survives_extreme_probabilities():
    """Log-odds blending must not overflow or leave the unit interval even when
    a model is effectively certain."""
    blended = blend_probs({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}, 0.5)
    assert 0 < blended["a"] < 1 and sum(blended.values()) == pytest.approx(1.0)


def test_optimal_weight_is_zero_when_the_model_is_noise():
    """The result that keeps the project honest: a model with no information
    must be assigned no weight, not a flattering small one."""
    market = [{"a": 0.7, "b": 0.3}] * 200
    model = [{"a": 0.3, "b": 0.7}] * 200          # systematically wrong
    outcomes = ["a"] * 140 + ["b"] * 60           # market is correctly calibrated
    assert optimal_weight(model, market, outcomes)["weight"] == 0.0


def test_optimal_weight_is_positive_when_the_model_knows_more():
    market = [{"a": 0.5, "b": 0.5}] * 200
    model = [{"a": 0.7, "b": 0.3}] * 200
    outcomes = ["a"] * 140 + ["b"] * 60           # model is the calibrated one
    fit = optimal_weight(model, market, outcomes)
    assert fit["weight"] > 0.5 and fit["improvement"] > 0


def test_log_loss_rewards_calibration_and_punishes_confident_errors():
    assert log_loss([{"a": 0.9, "b": 0.1}], ["a"]) < log_loss([{"a": 0.5, "b": 0.5}], ["a"])
    assert log_loss([{"a": 0.01, "b": 0.99}], ["a"]) > 4.0


def test_brier_is_zero_for_a_perfect_forecast():
    assert brier([{"a": 1.0, "b": 0.0}], ["a"]) == pytest.approx(0.0)


# --- ledger ---------------------------------------------------------------

@pytest.fixture()
def db():
    conn = ledger.connect(":memory:")
    yield conn
    conn.close()


def test_clv_sign_matches_which_way_the_market_moved():
    assert ledger.clv(2.10, 1.95) > 0     # took 2.10, market closed shorter -> good
    assert ledger.clv(1.95, 2.10) < 0
    assert ledger.clv(2.00, 2.00) == pytest.approx(0.0)


def test_settlement_pays_the_price_that_was_taken(db):
    bet = ledger.log_bet(db, market="h2h", selection="A", odds_taken=2.50, stake=40.0)
    ledger.settle(db, bet, "win")
    assert ledger.performance(db)["pnl"] == pytest.approx(60.0)


def test_a_push_returns_the_stake_and_is_excluded_from_roi(db):
    """A push moves no money, so it must not dilute ROI or hit rate."""
    bet = ledger.log_bet(db, market="totals", selection="Over", odds_taken=1.9, stake=10.0)
    ledger.settle(db, bet, "push")
    perf = ledger.performance(db)
    assert ledger.all_bets(db)[0]["pnl"] == pytest.approx(0.0)
    assert perf["n_push_or_void"] == 1
    assert perf["n_settled"] == 0
    assert "roi" not in perf


def test_verdict_refuses_to_judge_a_small_sample(db):
    """Guards against the project's worst failure mode: declaring an edge from
    a handful of lucky bets."""
    for _ in range(5):
        bet = ledger.log_bet(db, market="h2h", selection="A", odds_taken=2.2, stake=10.0)
        ledger.record_close(db, bet, closing_odds=1.9)
    perf = ledger.performance(db)
    assert perf["avg_clv"] > 0.10
    assert "too few bets" in perf["clv_verdict"]


def test_verdict_reports_an_edge_once_the_sample_supports_it(db):
    # Varying prices so the CLV sample has realistic non-zero variance.
    for i in range(60):
        taken = 2.10 + (i % 5) * 0.02
        bet = ledger.log_bet(db, market="h2h", selection="A", odds_taken=taken, stake=10.0)
        ledger.record_close(db, bet, closing_odds=2.00)
    perf = ledger.performance(db)
    assert perf["clv_tstat"] > 2
    assert "strong edge" in perf["clv_verdict"]


def test_negative_clv_is_reported_as_no_edge(db):
    for i in range(60):
        taken = 1.90 + (i % 5) * 0.02
        bet = ledger.log_bet(db, market="h2h", selection="A", odds_taken=taken, stake=10.0)
        ledger.record_close(db, bet, closing_odds=2.10)
    assert "no edge" in ledger.performance(db)["clv_verdict"]


def test_csv_round_trip_preserves_the_ledger(db):
    ledger.log_bet(db, market="h2h", selection="A", odds_taken=2.0, stake=10.0, book="X")
    text = ledger.to_csv(db)
    restored = ledger.connect(":memory:")
    assert ledger.from_csv(restored, text) == 1
    assert ledger.all_bets(restored)[0]["odds_taken"] == 2.0
    restored.close()


def test_open_bets_are_tracked_separately(db):
    a = ledger.log_bet(db, market="h2h", selection="A", odds_taken=2.0, stake=10.0)
    ledger.log_bet(db, market="h2h", selection="B", odds_taken=2.0, stake=10.0)
    ledger.settle(db, a, "win")
    assert len(ledger.open_bets(db)) == 1


def test_uniform_clv_is_not_mistaken_for_a_tiny_sample(db):
    """Zero sample variance must not collapse the t-statistic to NaN, which the
    verdict would misread as 'too few bets' and hide a real signal."""
    for _ in range(40):
        bet = ledger.log_bet(db, market="h2h", selection="A", odds_taken=2.10, stake=10.0)
        ledger.record_close(db, bet, closing_odds=2.00)
    perf = ledger.performance(db)
    assert perf["clv_tstat"] == float("inf")
    assert "strong edge" in perf["clv_verdict"]
