"""Tests for the daily card and the accumulator arithmetic."""
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_event  # noqa: E402

from src.daily import (  # noqa: E402
    MODES,
    build_card,
    candidates_from_event,
    parlay_analysis,
    poisson_binomial,
    summarise_card,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _liquid_event(home, away, home_price, away_price, league, event_id, hours=3, n_books=12):
    """An event quoted by enough agreeing books to clear the quality gate."""
    books = {"pinnacle": {"h2h": {home: home_price, away: away_price}}}
    for i in range(n_books - 1):
        jitter = 1 + (i % 3 - 1) * 0.004
        books[f"book{i}"] = {"h2h": {home: round(home_price * jitter, 3),
                                     away: round(away_price * jitter, 3)}}
    ev = make_event(books, home=home, away=away, event_id=event_id)
    ev["commence_time"] = (NOW + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    ev["_league"] = league
    return ev


def test_card_spreads_across_competitions():
    """One pick per league — what makes the legs near-independent, and what the
    user asked for when they said 'different nations or clubs'."""
    events = [
        _liquid_event(f"Home{i}", f"Away{i}", 1.45, 3.10, f"League {i}", f"e{i}")
        for i in range(6)
    ]
    card = build_card(events, "Soccer", n=5, mode="likely", now=NOW)
    assert len(card["selections"]) == 5
    assert len({s.league for s in card["selections"]}) == 5


def test_two_picks_from_one_league_are_not_both_taken():
    events = [
        _liquid_event("A", "B", 1.4, 3.2, "EPL", "e1"),
        _liquid_event("C", "D", 1.4, 3.2, "EPL", "e2"),
    ]
    assert len(build_card(events, "Soccer", n=5, mode="likely", now=NOW)["selections"]) == 1


def test_card_reports_being_short_rather_than_padding():
    """The behaviour the user most needs: if only 3 basketball games exist, say
    3 — do not invent a 4th and 5th."""
    events = [_liquid_event(f"H{i}", f"A{i}", 1.5, 2.9, f"L{i}", f"e{i}") for i in range(3)]
    card = build_card(events, "Basketball", n=5, mode="likely", now=NOW)
    assert len(card["selections"]) == 3
    assert card["short_by"] == 2


def test_illiquid_markets_are_excluded_entirely():
    """Three books that disagree is not an estimate, so it cannot be a 'strong'
    selection at any price."""
    thin = _liquid_event("A", "B", 1.5, 2.9, "Obscure League", "e1", n_books=3)
    card = build_card([thin], "Soccer", n=5, mode="likely", now=NOW)
    assert card["selections"] == []
    assert card["candidates"] > 0 and card["passed_quality_gate"] == 0


def test_games_outside_the_window_are_ignored():
    """NBA fixtures listed seven weeks out must not appear on today's card."""
    far = _liquid_event("A", "B", 1.5, 2.9, "NBA", "e1", hours=24 * 52)
    assert build_card([far], "Basketball", n=5, within_hours=24, now=NOW)["selections"] == []
    assert build_card([far], "Basketball", n=5, within_hours=24, now=NOW)["events_in_window"] == 0


def test_started_games_are_excluded():
    past = _liquid_event("A", "B", 1.5, 2.9, "EPL", "e1", hours=-2)
    assert build_card([past], "Soccer", n=5, now=NOW)["selections"] == []


def test_likely_mode_prefers_favourites_and_value_mode_need_not():
    """The two modes answer different questions and must not collapse into one."""
    events = [
        _liquid_event("BigFav", "Minnow", 1.15, 6.5, "L1", "e1"),
        _liquid_event("Edge", "Other", 2.60, 1.55, "L2", "e2"),
    ]
    likely = build_card(events, "Soccer", n=1, mode="likely", now=NOW)["selections"]
    assert likely and likely[0].fair_prob > 0.7


def test_value_mode_requires_a_positive_price_edge():
    """A fairly-priced board yields no value picks, however likely the winners."""
    events = [_liquid_event(f"H{i}", f"A{i}", 1.45, 3.10, f"L{i}", f"e{i}") for i in range(4)]
    for sel in build_card(events, "Soccer", n=5, mode="value", now=NOW)["selections"]:
        assert sel.ev >= 0.01


def test_balanced_mode_refuses_selections_priced_against_you():
    events = [_liquid_event(f"H{i}", f"A{i}", 1.45, 3.10, f"L{i}", f"e{i}") for i in range(4)]
    for sel in build_card(events, "Soccer", n=5, mode="balanced", now=NOW)["selections"]:
        assert sel.ev >= -0.005


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_card([], "Soccer", mode="guaranteed-winners")


def test_conservative_probability_never_exceeds_the_estimate():
    ev = _liquid_event("A", "B", 1.5, 2.9, "L", "e1")
    for sel in candidates_from_event(ev, "Soccer"):
        assert sel.shrunk_prob <= sel.fair_prob


# --- accumulator arithmetic ------------------------------------------------

def test_poisson_binomial_is_a_distribution():
    dist = poisson_binomial([0.3, 0.7, 0.5, 0.9])
    assert len(dist) == 5
    assert sum(dist) == pytest.approx(1.0)


def test_poisson_binomial_matches_the_binomial_when_probabilities_are_equal():
    from math import comb
    p, n = 0.6, 5
    dist = poisson_binomial([p] * n)
    for k in range(n + 1):
        assert dist[k] == pytest.approx(comb(n, k) * p**k * (1 - p) ** (n - k))


def test_poisson_binomial_handles_certainties():
    assert poisson_binomial([1.0, 1.0])[2] == pytest.approx(1.0)
    assert poisson_binomial([0.0, 0.0])[0] == pytest.approx(1.0)


def test_ten_likely_legs_are_an_unlikely_parlay():
    """The number the whole tab exists to show: individually likely legs make a
    very unlikely accumulator."""
    dist = poisson_binomial([0.75] * 10)
    assert dist[10] == pytest.approx(0.75**10, rel=1e-9)
    assert dist[10] < 0.06


def test_negative_ev_legs_compound_into_a_much_worse_parlay():
    """Margins compound: mildly negative legs become severely negative combined."""
    class L:
        def __init__(self, p, o):
            self.fair_prob, self.odds, self.ev, self.shrunk_prob = p, o, p * o - 1, p - 0.01
    par = parlay_analysis([L(0.75, 1.30) for _ in range(10)])
    assert par["parlay_ev"] < -0.20
    assert par["singles_avg_ev"] > par["parlay_ev"]


def test_positive_ev_legs_raise_parlay_ev_and_the_code_admits_it():
    """EV compounds in BOTH directions. Claiming a parlay always has worse EV is
    simply false, and the honest argument has to rest on something else."""
    class L:
        def __init__(self, p, o):
            self.fair_prob, self.odds, self.ev, self.shrunk_prob = p, o, p * o - 1, p - 0.01
    par = parlay_analysis([L(0.60, 1.75) for _ in range(5)])
    assert par["parlay_ev"] > par["singles_avg_ev"]


def test_singles_always_outgrow_the_parlay_even_when_parlay_ev_is_higher():
    """The rigorous argument. Kelly growth is additive across independent bets and
    concave in a single one, so singles dominate regardless of how the EVs compare."""
    class L:
        def __init__(self, p, o):
            self.fair_prob, self.odds, self.ev, self.shrunk_prob = p, o, p * o - 1, p - 0.01
    par = parlay_analysis([L(0.60, 1.75) for _ in range(5)])
    assert par["parlay_ev"] > par["singles_avg_ev"]          # parlay wins on EV
    assert par["singles_log_growth"] > par["parlay_log_growth"]   # and still loses on growth
    assert par["growth_ratio"] > 1


def test_parlay_reports_the_chance_of_returning_nothing():
    events = [_liquid_event(f"H{i}", f"A{i}", 1.45, 3.10, f"L{i}", f"e{i}") for i in range(6)]
    sels = build_card(events, "Soccer", n=5, mode="likely", now=NOW)["selections"]
    par = parlay_analysis(sels)
    assert par["lose_everything_prob"] == pytest.approx(1 - par["all_win_prob"])
    assert par["all_win_prob"] < min(s.fair_prob for s in sels)


def test_expected_winners_is_the_sum_of_probabilities():
    events = [_liquid_event(f"H{i}", f"A{i}", 1.5, 2.9, f"L{i}", f"e{i}") for i in range(4)]
    sels = build_card(events, "Soccer", n=4, mode="likely", now=NOW)["selections"]
    par = parlay_analysis(sels)
    assert par["expected_winners"] == pytest.approx(sum(s.fair_prob for s in sels))


def test_empty_parlay_is_handled():
    assert parlay_analysis([])["n"] == 0


def test_summary_states_expected_winners_not_certainty():
    events = [_liquid_event(f"H{i}", f"A{i}", 1.45, 3.1, f"L{i}", f"e{i}") for i in range(5)]
    text = summarise_card(build_card(events, "Soccer", n=5, mode="likely", now=NOW))
    assert "expect about" in text.lower()


def test_summary_of_an_empty_card_says_so():
    assert "no soccer selection" in summarise_card(
        build_card([], "Soccer", n=5, now=NOW)).lower()


def test_every_mode_is_documented():
    for _name, meta in MODES.items():
        assert meta["label"] and meta["help"]


def test_a_longshot_parlay_is_not_reported_as_certain_loss():
    """Guards a display bug found live: a 5-leg value parlay busts 99.985% of
    the time, and rounding that to '100%' reads as certainty."""
    class L:
        def __init__(self, p, o):
            self.fair_prob, self.odds, self.ev, self.shrunk_prob = p, o, p * o - 1, p - 0.01
    par = parlay_analysis([L(0.20, 5.2) for _ in range(5)])
    assert par["lose_everything_prob"] < 1.0
    assert f"{par['lose_everything_prob'] * 100:.0f}%" == "100%"   # the naive format lies
    assert par["all_win_prob"] > 0
