"""Tests for EV, Kelly, shrinkage and slate sizing."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edge import (  # noqa: E402
    Opportunity,
    classify,
    edge_vs_fair,
    evaluate,
    growth_rate,
    kelly_fraction,
    normalise_slate,
    shrink_probability,
)


def test_fair_price_has_zero_ev_and_zero_kelly():
    from src.edge import ev_per_unit
    assert ev_per_unit(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.5, 2.0) == 0.0


def test_kelly_matches_the_closed_form():
    # p=0.6 at 2.0 -> f* = (0.6*2 - 1)/(2 - 1) = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_negative_edge_never_produces_a_stake():
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert evaluate(0.4, 2.0)["stake_fraction"] == 0.0


def test_overbetting_kelly_destroys_the_bankroll():
    """The single most important fact about stake sizing: past 2x Kelly the
    expected log growth turns negative even though the edge is real."""
    p, odds = 0.55, 2.0
    f_star = kelly_fraction(p, odds)
    assert growth_rate(p, odds, f_star) > 0
    assert growth_rate(p, odds, 2 * f_star) < growth_rate(p, odds, f_star)
    assert growth_rate(p, odds, 2.5 * f_star) < 0


def test_kelly_growth_is_maximised_at_the_kelly_fraction():
    p, odds = 0.55, 2.0
    f_star = kelly_fraction(p, odds)
    best = max(
        (f_star * m for m in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        key=lambda f: growth_rate(p, odds, f),
    )
    assert best == pytest.approx(f_star)


def test_shrinkage_reduces_the_stake_when_books_disagree():
    tight = evaluate(0.55, 2.0, sigma=0.002)
    loose = evaluate(0.55, 2.0, sigma=0.030)
    assert loose["stake_fraction"] < tight["stake_fraction"]


def test_shrinkage_can_remove_a_marginal_bet_entirely():
    """A 1% edge inside 3pp of book disagreement should not be staked at all."""
    assert evaluate(0.51, 2.0, sigma=0.03)["stake_fraction"] == 0.0


def test_shrink_probability_stays_in_the_unit_interval():
    assert 0.0 < shrink_probability(0.02, 0.5) < 1.0


def test_edge_and_ev_agree_on_sign():
    assert edge_vs_fair(0.55, 2.0) > 0
    assert edge_vs_fair(0.45, 2.0) < 0


def test_slate_normalisation_caps_exposure_and_keeps_ordering():
    ops = [Opportunity(stake_fraction=f, flags=[]) for f in (0.05, 0.04, 0.03)]
    normalise_slate(ops, exposure_cap=0.06)
    total = sum(o.stake_fraction for o in ops)
    assert total == pytest.approx(0.06)
    assert ops[0].stake_fraction > ops[1].stake_fraction > ops[2].stake_fraction


def test_slate_under_the_cap_is_left_alone():
    ops = [Opportunity(stake_fraction=0.01, flags=[])]
    normalise_slate(ops, exposure_cap=0.10)
    assert ops[0].stake_fraction == 0.01
    assert ops[0].flags == []


def test_implausible_edges_are_flagged_not_bet():
    assert any("implausible" in f for f in classify(0.35, 0.01, 8, "sharp"))


def test_thin_market_and_disagreement_are_flagged():
    flags = classify(0.03, 0.05, 2, "unreliable")
    assert any("thin market" in f for f in flags)
    assert any("disagree" in f for f in flags)
