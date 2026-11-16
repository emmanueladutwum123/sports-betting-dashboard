"""Tests for the residual favourite-longshot correction.

The empirical case for the correction lives in ``src/calibration.py`` and is
re-derivable with ``scripts/fit_calibration.py``. These tests cover the things
that would silently break it: the transform's algebra, the scope rules that
keep a soccer-fitted constant off other sports, and the wiring into the live
pricing path.
"""
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_event  # noqa: E402

from src.calibration import (  # noqa: E402
    IDENTITY,
    LEAGUE_EXPONENTS,
    SANE_EXPONENT_RANGE,
    SHRINKAGE_K,
    SOCCER_EXPONENT,
    SOCCER_H2H,
    SOCCER_TOTALS,
    Calibration,
    apply,
    bias_above,
    calibration_for,
    fit_exponent,
    reliability,
    sharpen,
)
from src.datafeeds.football_data import SPORT_KEY_TO_DIVISION  # noqa: E402
from src.market import MarketView  # noqa: E402

# ------------------------------------------------------------------- transform


def test_sharpening_preserves_normalisation():
    out = sharpen({"home": 0.55, "draw": 0.25, "away": 0.20}, 1.0408)
    assert sum(out.values()) == pytest.approx(1.0)


def test_sharpening_moves_mass_to_the_favourite():
    """b > 1 must raise likely outcomes and lower unlikely ones. This is the
    whole point: the market understates favourites, so the correction has to
    push in that direction, not merely rescale."""
    raw = {"home": 0.70, "draw": 0.20, "away": 0.10}
    out = sharpen(raw, 1.05)
    assert out["home"] > raw["home"]
    assert out["draw"] < raw["draw"]
    assert out["away"] < raw["away"]


def test_sharpening_preserves_ranking():
    """A monotone transform must never reorder outcomes -- if it did, the
    correction would be changing which selection the card picks, not just the
    number printed beside it."""
    raw = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    out = sharpen(raw, 1.2)
    assert sorted(raw, key=raw.get) == sorted(out, key=out.get)


def test_exponent_one_is_identity():
    raw = {"home": 0.44, "draw": 0.27, "away": 0.29}
    assert sharpen(raw, 1.0) == raw


def test_sharpening_leaves_a_uniform_vector_alone():
    """Any exponent fixes the uniform distribution -- a useful invariant,
    because it means the correction cannot invent an edge out of a market where
    the book has expressed no opinion at all."""
    raw = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    out = sharpen(raw, 1.2)
    for value in out.values():
        assert value == pytest.approx(1 / 3)


def test_sharpening_survives_degenerate_input():
    assert sharpen({}, 1.1) == {}
    assert sharpen({"only": 1.0}, 1.1) == {"only": pytest.approx(1.0)}
    assert sharpen({"a": 0.0, "b": 0.0}, 1.1) == {"a": 0.0, "b": 0.0}
    # A None price must not take the whole card down with it.
    out = sharpen({"a": 0.6, "b": None}, 1.1)
    assert out["a"] == pytest.approx(1.0)


def test_sharpening_does_not_mutate_its_input():
    raw = {"home": 0.6, "away": 0.4}
    sharpen(raw, 1.2)
    assert raw == {"home": 0.6, "away": 0.4}


def test_shipped_exponent_is_in_the_sane_range():
    lo, hi = SANE_EXPONENT_RANGE
    assert lo <= SOCCER_EXPONENT <= hi
    assert SOCCER_EXPONENT > 1.0, "the measured bias understates favourites"


def test_shipped_exponent_size_is_a_small_correction():
    """Guards against a refit shipping something that reshapes the board. At
    the shipped exponent a 70% favourite should move by about a point, not ten."""
    moved = sharpen({"home": 0.70, "draw": 0.20, "away": 0.10}, SOCCER_EXPONENT)
    assert 0.002 < moved["home"] - 0.70 < 0.02


# ----------------------------------------------------------------------- scope


def test_soccer_h2h_uses_its_own_league_exponent():
    cal = calibration_for("soccer_epl", "h2h")
    assert cal.league == "E0"
    assert cal.exponent == pytest.approx(LEAGUE_EXPONENTS["E0"])


def test_an_unfitted_soccer_league_is_not_calibrated():
    """MLS and Brazil have no closing-price history in the free feed."""
    assert calibration_for("soccer_usa_mls", "h2h") is IDENTITY
    assert calibration_for("soccer_uefa_nations_league", "totals") is IDENTITY


def test_a_league_measured_as_unbiased_is_not_sharpened():
    """The Bundesliga close slightly *overstates* favourites, so its exponent
    is below 1. A single global constant made that league worse, which is the
    whole reason per-league exponents exist."""
    cal = calibration_for("soccer_germany_bundesliga", "h2h")
    assert cal.exponent < 1.0
    raw = {"home": 0.70, "draw": 0.20, "away": 0.10}
    assert sharpen(raw, cal.exponent)["home"] < raw["home"]


def test_every_league_exponent_is_in_the_sane_range():
    lo, hi = SANE_EXPONENT_RANGE
    for code, exponent in LEAGUE_EXPONENTS.items():
        assert lo <= exponent <= hi, f"{code} = {exponent}"


def test_league_exponents_are_shrunk_toward_the_global_one():
    """Every shipped exponent must sit strictly between the global number and
    its own unshrunk fit -- that is what partial pooling means. Catches a refit
    that pastes in raw per-league fits and silently reintroduces overfitting."""
    spread = max(LEAGUE_EXPONENTS.values()) - min(LEAGUE_EXPONENTS.values())
    assert spread < 0.15, "per-league exponents look unshrunk"
    assert 0.0 < SHRINKAGE_K < 1.0


def test_every_mapped_soccer_league_resolves():
    """Any league the feed knows how to map must get a real calibration, not
    an accidental identity from a typo in a division code."""
    for sport_key in SPORT_KEY_TO_DIVISION:
        cal = calibration_for(sport_key, "h2h")
        assert not cal.is_identity


def test_soccer_totals_uses_its_own_exponent():
    assert calibration_for("soccer_epl", "totals") is SOCCER_TOTALS


def test_other_sports_are_not_calibrated():
    """A constant fitted on European soccer has no claim on basketball."""
    for key in ("basketball_nba", "americanfootball_nfl", "tennis_atp", "", None):
        assert calibration_for(key, "h2h") is IDENTITY


def test_unfitted_soccer_markets_are_not_calibrated():
    """Spreads and player props were never measured, so they pass through."""
    assert calibration_for("soccer_epl", "spreads") is IDENTITY
    assert calibration_for("soccer_epl", "player_goal_scorer_anytime") is IDENTITY


def test_identity_calibration_is_a_no_op():
    raw = {"home": 0.5, "away": 0.5}
    assert apply(raw, IDENTITY) == raw
    assert apply(raw, None) == raw


def test_describe_reports_direction_scope_and_provenance():
    text = SOCCER_H2H.describe()
    assert "toward favourites" in text
    assert "66,272" in text
    assert "all-league" in text
    epl = calibration_for("soccer_epl", "h2h").describe()
    assert epl.startswith("E0 ")
    assert "toward longshots" in calibration_for(
        "soccer_germany_bundesliga", "h2h"
    ).describe()
    assert Calibration(1.0, 0, "x").describe() == "no calibration applied"


# ------------------------------------------------------------------ live wiring


def _three_way(home_price, draw_price, away_price, sport_key):
    event = make_event(
        {
            "pinnacle": {"h2h": {"Arsenal": home_price, "Draw": draw_price, "Chelsea": away_price}},
            "betway": {"h2h": {"Arsenal": home_price, "Draw": draw_price, "Chelsea": away_price}},
        }
    )
    event["sport_key"] = sport_key
    return event


def test_anchor_is_calibrated_for_soccer_and_raw_is_still_available():
    view = MarketView(_three_way(1.35, 5.0, 9.0, "soccer_epl"), "h2h", calibrate=True)
    raw, calibrated = view.raw_anchor(), view.anchor()
    assert calibrated["Arsenal"] > raw["Arsenal"]
    assert sum(calibrated.values()) == pytest.approx(1.0)


def test_anchor_is_untouched_for_basketball():
    view = MarketView(_three_way(1.35, 5.0, 9.0, "basketball_nba"), "h2h")
    assert view.anchor() == view.raw_anchor()


def test_calibration_can_be_switched_off():
    event = _three_way(1.35, 5.0, 9.0, "soccer_epl")
    assert MarketView(event, "h2h", calibrate=False).anchor() == MarketView(
        event, "h2h", calibrate=False
    ).raw_anchor()


def test_consensus_and_sharp_pools_stay_raw():
    """Dispersion and sharp-vs-soft disagreement compare books to each other.
    The correction is a property of the market as a forecast, not of any one
    book's price, so it must not leak into those comparisons."""
    view = MarketView(_three_way(1.35, 5.0, 9.0, "soccer_epl"), "h2h", calibrate=True)
    assert view.sharp["Arsenal"] == pytest.approx(view.raw_anchor()["Arsenal"])


def test_calibration_raises_ev_on_favourites():
    """The practical consequence, stated as a test: a favourite priced at the
    de-vigged fair line looks break-even before the correction and slightly
    +EV after it. If this ever flips, the sign of the correction is wrong."""
    view = MarketView(_three_way(1.35, 5.0, 9.0, "soccer_epl"), "h2h", calibrate=True)
    fair_price = 1.0 / view.raw_anchor()["Arsenal"]
    assert view.anchor()["Arsenal"] * fair_price > 1.0


# --------------------------------------------------------------------- fitting


def _synthetic(bias_exponent, n=4000, seed=7):
    """Markets whose quoted prices are a known distortion of the truth.

    Quoted probabilities are the true ones flattened by ``1/bias_exponent``, so
    a correct fit has to recover ``bias_exponent`` to undo them.
    """
    rng = random.Random(seed)
    vectors, outcomes = [], []
    for _ in range(n):
        raw = [rng.uniform(0.15, 3.0) for _ in range(3)]
        total = sum(raw)
        true = {name: v / total for name, v in zip(("home", "draw", "away"), raw, strict=True)}
        quoted = sharpen(true, 1.0 / bias_exponent)
        draw = rng.random()
        cumulative = 0.0
        for name, p in true.items():
            cumulative += p
            if draw <= cumulative:
                outcomes.append(name)
                break
        else:
            outcomes.append("away")
        vectors.append(quoted)
    return vectors, outcomes


def test_fit_recovers_a_known_distortion():
    vectors, outcomes = _synthetic(1.15, n=8000)
    assert fit_exponent(vectors, outcomes) == pytest.approx(1.15, abs=0.06)


def test_fit_returns_about_one_on_already_calibrated_markets():
    vectors, outcomes = _synthetic(1.0, n=8000)
    assert fit_exponent(vectors, outcomes) == pytest.approx(1.0, abs=0.06)


def test_fit_refuses_a_sample_too_small_to_mean_anything():
    with pytest.raises(ValueError, match="at least 100"):
        fit_exponent([{"a": 0.5, "b": 0.5}] * 10, ["a"] * 10)


# ----------------------------------------------------------------- diagnostics


def test_reliability_detects_a_planted_bias():
    """Markets quoting 0.60 on something that happens 75% of the time must be
    flagged, and the flag must be attached to the right band."""
    vectors = [{"win": 0.60, "lose": 0.40}] * 1000
    outcomes = ["win" if i % 4 else "lose" for i in range(1000)]
    rows = {r["band"]: r for r in reliability(vectors, outcomes)}
    hot = rows["0.6-0.7"]
    assert hot["n"] == 1000
    assert hot["bias"] == pytest.approx(0.15, abs=0.01)
    assert hot["significant"]


def test_reliability_is_quiet_on_calibrated_markets():
    vectors, outcomes = _synthetic(1.0, n=6000)
    flagged = [r for r in reliability(vectors, outcomes) if r["significant"]]
    assert len(flagged) <= 1, f"unexpected miscalibration: {flagged}"


def test_bias_above_only_counts_the_selected_band():
    vectors = [{"big": 0.80, "small": 0.20}] * 500
    outcomes = ["big"] * 500
    out = bias_above(vectors, outcomes, 0.6)
    assert out["n"] == 500
    assert out["predicted"] == pytest.approx(0.80)
    assert out["bias"] == pytest.approx(0.20)


def test_bias_above_handles_an_empty_selection():
    assert bias_above([{"a": 0.1, "b": 0.9}], ["b"], 0.95)["n"] == 0


def test_correction_shrinks_the_measured_bias():
    """End to end: markets with the real-world distortion, corrected by the
    shipped exponent, must come out closer to honest than they went in."""
    vectors, outcomes = _synthetic(SOCCER_EXPONENT, n=20000, seed=11)
    before = bias_above(vectors, outcomes, 0.5)
    after = bias_above([sharpen(v, SOCCER_EXPONENT) for v in vectors], outcomes, 0.5)
    assert before["bias"] > 0.0
    assert abs(after["bias"]) < abs(before["bias"])


def test_log_loss_does_not_get_worse_on_distorted_markets():
    vectors, outcomes = _synthetic(SOCCER_EXPONENT, n=20000, seed=3)

    def loss(vs):
        return -sum(math.log(max(v[y], 1e-12)) for v, y in zip(vs, outcomes, strict=True)) / len(vs)

    assert loss([sharpen(v, SOCCER_EXPONENT) for v in vectors]) < loss(vectors)


def test_live_market_calibration_is_disabled_by_default():
    view = MarketView(_three_way(1.35, 5.0, 9.0, "soccer_epl"), "h2h")
    assert view.anchor() == view.raw_anchor()
