"""Tests for the de-vig methods.

The properties asserted here are the ones that make downstream EV numbers
trustworthy: every method returns a genuine probability distribution, the
favourite-longshot ordering between methods is the one theory predicts, and the
known equivalences hold exactly.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.devig import (  # noqa: E402
    METHODS,
    devig,
    devig_additive,
    devig_power,
    devig_shin,
    fair_odds,
    margin,
    shin_z,
)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("odds", [[2.10, 1.80], [1.30, 5.50, 11.0], [4.0, 3.6, 2.0], [1.01, 41.0]])
def test_output_is_a_probability_distribution(method, odds):
    probs = devig(odds, method)
    assert all(0.0 < p < 1.0 for p in probs)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_ordering_is_preserved(method):
    """De-vigging must never reorder outcomes: shorter price, higher probability."""
    odds = [1.50, 4.20, 7.00]
    probs = devig(odds, method)
    assert probs[0] > probs[1] > probs[2]


def test_shin_and_additive_agree_on_two_outcome_markets():
    """A documented identity -- with two outcomes Shin reduces to the additive
    method. If this ever fails the Shin root-finder has drifted."""
    pis = [1 / 1.91, 1 / 1.95]
    assert devig_shin(pis) == pytest.approx(devig_additive(pis), abs=1e-6)


def test_shin_and_power_price_the_favourite_above_multiplicative():
    """The favourite-longshot correction, which is the entire reason for using
    anything other than the naive method. Multiplicative spreads margin in
    proportion to price and so under-prices the favourite."""
    odds = [1.30, 5.50, 11.0]
    naive = devig(odds, "multiplicative")
    for method in ("shin", "power", "additive"):
        corrected = devig(odds, method)
        assert corrected[0] > naive[0], f"{method} should raise the favourite"
        assert corrected[-1] < naive[-1], f"{method} should cut the longshot"


def test_correction_is_material_relative_to_a_realistic_edge():
    """The whole argument for this module: the gap between de-vig methods is
    larger than the edge a +EV bettor is hunting for, so the choice is not a
    detail."""
    odds = [1.30, 5.50, 11.0]
    gap = devig(odds, "shin")[0] - devig(odds, "multiplicative")[0]
    assert gap > 0.01


def test_no_margin_to_remove_leaves_a_fair_book_untouched():
    fair = [2.0, 2.0]
    assert devig(fair, "shin") == pytest.approx([0.5, 0.5])


def test_margin_is_reported_as_hold_not_overround():
    """Hold = 1 - 1/booksum, which is comparable across 2-way and 3-way markets;
    booksum - 1 is not."""
    assert margin([1.91, 1.91]) == pytest.approx(1 - 1 / (2 / 1.91), rel=1e-9)


def test_shin_z_is_zero_for_a_fair_book_and_positive_for_a_vigged_one():
    assert shin_z([0.5, 0.5]) == 0.0
    assert shin_z([1 / 1.30, 1 / 5.50, 1 / 11.0]) > 0


def test_missing_prices_pass_through_as_none():
    probs = devig([2.0, None, 3.0], "shin")
    assert probs[1] is None
    assert math.isclose(sum(p for p in probs if p is not None), 1.0, rel_tol=1e-9)


def test_power_method_solves_its_defining_equation():
    pis = [1 / 1.4, 1 / 4.5, 1 / 8.0]
    probs = devig_power(pis)
    k = math.log(probs[0]) / math.log(pis[0])
    assert sum(p**k for p in pis) == pytest.approx(1.0, abs=1e-6)


def test_fair_odds_round_trips():
    assert fair_odds(0.25) == pytest.approx(4.0)
    assert fair_odds(0.0) is None


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        devig([2.0, 2.0], "nonsense")
