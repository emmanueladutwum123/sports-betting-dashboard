"""Tests for the goal model and the shot-augmented variant.

The Dixon-Coles fit itself is tested through its invariants rather than against
golden numbers: a maximum-likelihood fit's exact parameters depend on the
optimiser, but the properties below must hold for any correct fit, and each one
fails loudly under a specific plausible bug.
"""
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datafeeds.football_data import shot_conversion, to_matches  # noqa: E402
from src.models.dixon_coles import (  # noqa: E402
    SHOT_WEIGHT,
    DixonColesModel,
    ShotAugmentedDixonColes,
)

STRONG, MIDDLE, WEAK = "Strong", "Middle", "Weak"
TEAMS = (STRONG, MIDDLE, WEAK)
# Goals per match each team scores at home, by (home, away) pairing. Strong
# outscores Weak heavily, which every fit below has to recover.
RATE = {STRONG: 2.4, MIDDLE: 1.3, WEAK: 0.7}


def synthetic_rows(n_rounds=140, seed=5, sot_multiplier=3.0, sot_noise=True):
    """Fixtures generated from known strengths, with shots on target that are a
    noisy multiple of the goal rate. Shot counts are deliberately generated
    from the *same* latent strength as goals, so a correct shot fit must agree
    with the goal fit rather than pull against it."""
    rng = random.Random(seed)
    start = datetime(2021, 8, 1)
    rows, day = [], 0
    for _ in range(n_rounds):
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                day += 3
                lam, mu = RATE[home] * 1.2, RATE[away] * 0.85
                hg = _poisson(rng, lam)
                ag = _poisson(rng, mu)
                if sot_noise:
                    hst = _poisson(rng, lam * sot_multiplier)
                    ast = _poisson(rng, mu * sot_multiplier)
                else:
                    hst, ast = round(lam * sot_multiplier), round(mu * sot_multiplier)
                rows.append({
                    "date": start + timedelta(days=day),
                    "home_team": home, "away_team": away,
                    "home_score": hg, "away_score": ag,
                    "home_sot": hst, "away_sot": ast,
                    "result": "home" if hg > ag else ("away" if ag > hg else "draw"),
                })
    return rows


def _poisson(rng, lam):
    """Knuth's method. Avoids a numpy RNG so the fixture is reproducible
    independently of numpy's version-to-version generator changes."""
    target, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= target:
            return k
        k += 1


# ------------------------------------------------------------------- base fit


def test_fit_recovers_the_strength_ordering():
    model = DixonColesModel().fit(to_matches(synthetic_rows()), xi=0.0)
    ranked = [r["team"] for r in model.team_ratings()]
    assert ranked == [STRONG, MIDDLE, WEAK]


def test_fit_recovers_home_advantage():
    model = DixonColesModel().fit(to_matches(synthetic_rows()), xi=0.0)
    # Generated with home rates 1.2x and away 0.85x, i.e. log(1.2/0.85) = 0.345.
    assert model.home_advantage == pytest.approx(math.log(1.2 / 0.85), abs=0.12)


def test_fit_rejects_a_sample_too_small_to_identify_anything():
    with pytest.raises(ValueError, match="at least 20"):
        DixonColesModel().fit(to_matches(synthetic_rows(n_rounds=2)))


def test_derived_markets_agree_with_the_score_matrix():
    """1X2, totals and BTTS all read off one matrix, so they cannot disagree.
    This is the property that makes the model worth having at all."""
    model = DixonColesModel().fit(to_matches(synthetic_rows()), xi=0.0)
    odds = model.match_odds(STRONG, WEAK)
    assert sum(odds.values()) == pytest.approx(1.0)
    totals = model.totals(STRONG, WEAK, 2.5)
    assert totals["Over"] + totals["Under"] == pytest.approx(1.0)
    btts = model.btts(STRONG, WEAK)
    assert btts["Yes"] + btts["No"] == pytest.approx(1.0)
    # A higher line cannot be more likely to go over.
    assert model.totals(STRONG, WEAK, 3.5)["Over"] < totals["Over"]


def test_the_better_team_is_favoured():
    model = DixonColesModel().fit(to_matches(synthetic_rows()), xi=0.0)
    assert model.match_odds(STRONG, WEAK)["home"] > model.match_odds(WEAK, STRONG)["home"]


# --------------------------------------------------------- shot augmentation


def test_shot_augmented_agrees_with_goals_only_at_weight_zero():
    rows = synthetic_rows()
    combined = ShotAugmentedDixonColes(shot_weight=0.0).fit(rows, xi=0.0)
    plain = DixonColesModel().fit(to_matches(rows), xi=0.0)
    assert combined.expected_goals(STRONG, WEAK) == pytest.approx(
        plain.expected_goals(STRONG, WEAK)
    )


def test_shot_augmented_still_predicts_goal_scale_rates():
    """The conversion factor must bring shot counts back onto the goal scale.
    Forgetting it would triple every expected-goals number, which is the single
    most likely way to get this wrong."""
    rows = synthetic_rows(sot_multiplier=3.0)
    model = ShotAugmentedDixonColes(shot_weight=0.5).fit(rows, xi=0.0)
    lam, mu = model.expected_goals(STRONG, WEAK)
    assert 1.5 < lam < 4.5
    assert 0.2 < mu < 1.6
    assert model.conversion == pytest.approx(1 / 3.0, abs=0.06)


def test_shot_augmented_is_between_its_two_components():
    """A log-space mix must sit between the two rates it mixes."""
    rows = synthetic_rows(seed=9)
    goals_only = ShotAugmentedDixonColes(shot_weight=0.0).fit(rows, xi=0.0)
    shots_only = ShotAugmentedDixonColes(shot_weight=1.0).fit(rows, xi=0.0)
    mixed = ShotAugmentedDixonColes(shot_weight=0.5).fit(rows, xi=0.0)
    lo, hi = sorted([goals_only.expected_goals(STRONG, WEAK)[0],
                     shots_only.expected_goals(STRONG, WEAK)[0]])
    assert lo <= mixed.expected_goals(STRONG, WEAK)[0] <= hi


def test_shot_augmented_falls_back_when_shot_data_is_missing():
    """Real feeds drop shot columns for whole leagues. That must degrade to a
    goals model, not crash and not silently predict nonsense."""
    rows = synthetic_rows()
    for r in rows:
        r["home_sot"] = r["away_sot"] = None
    model = ShotAugmentedDixonColes(shot_weight=0.5).fit(rows, xi=0.0)
    plain = DixonColesModel().fit(to_matches(rows), xi=0.0)
    assert model.shots is None
    assert model.expected_goals(STRONG, WEAK) == pytest.approx(plain.expected_goals(STRONG, WEAK))


def test_shot_augmented_handles_partial_shot_coverage():
    rows = synthetic_rows()
    for r in rows[: len(rows) // 2]:
        r["home_sot"] = r["away_sot"] = None
    model = ShotAugmentedDixonColes(shot_weight=0.5).fit(rows, xi=0.0)
    assert model.shots is not None
    assert all(v > 0 for v in model.expected_goals(STRONG, WEAK))


def test_shot_augmented_exposes_the_same_read_api():
    rows = synthetic_rows()
    model = ShotAugmentedDixonColes().fit(rows, xi=0.0)
    assert sum(model.match_odds(STRONG, WEAK).values()) == pytest.approx(1.0)
    assert sum(model.btts(STRONG, WEAK).values()) == pytest.approx(1.0)
    ah = model.asian_handicap(STRONG, WEAK, -1.0)
    assert ah["Home"] + ah["Away"] == pytest.approx(1.0)
    assert len(model.correct_score(STRONG, WEAK, top=5)) == 5
    assert [r["team"] for r in model.team_ratings()] == [STRONG, MIDDLE, WEAK]
    assert model.knows(STRONG, WEAK) and not model.knows("Nobody")


def test_shot_weight_is_validated():
    with pytest.raises(ValueError, match="shot_weight"):
        ShotAugmentedDixonColes(shot_weight=1.5)


def test_shipped_shot_weight_is_a_genuine_mix():
    assert 0.0 < SHOT_WEIGHT < 1.0


def test_shot_augmented_is_more_accurate_on_noisy_goals():
    """The empirical claim, reproduced on synthetic data: when goals are a
    noisy read on strength and shots are a better-sampled one, mixing beats
    goals alone. Scored out-of-sample on held-back fixtures."""
    rows = synthetic_rows(n_rounds=120, seed=21)
    split = int(len(rows) * 0.7)
    train, test = rows[:split], rows[split:]

    def loss(model):
        total = 0.0
        for r in test:
            probs = model.match_odds(r["home_team"], r["away_team"])
            total -= math.log(max(probs[r["result"]], 1e-12))
        return total / len(test)

    goals_only = loss(ShotAugmentedDixonColes(shot_weight=0.0).fit(train, xi=0.0))
    mixed = loss(ShotAugmentedDixonColes(shot_weight=SHOT_WEIGHT).fit(train, xi=0.0))
    assert mixed <= goals_only + 1e-4


# ------------------------------------------------------------------ feed glue


def test_to_matches_can_read_the_shot_columns():
    rows = synthetic_rows(n_rounds=3)
    goals = to_matches(rows, "goals")
    sot = to_matches(rows, "sot")
    assert len(goals) == len(sot) == len(rows)
    assert goals[0].home_score == rows[0]["home_score"]
    assert sot[0].home_score == rows[0]["home_sot"]


def test_to_matches_drops_rather_than_zero_fills_missing_shots():
    """Zero-filling would assert a team had no shots on target, dragging its
    attack rating down for a reason that is purely a gap in the file."""
    rows = synthetic_rows(n_rounds=3)
    rows[0]["home_sot"] = None
    assert len(to_matches(rows, "sot")) == len(rows) - 1
    assert len(to_matches(rows, "goals")) == len(rows)


def test_to_matches_rejects_an_unknown_count():
    with pytest.raises(ValueError, match="goals"):
        to_matches(synthetic_rows(n_rounds=3), "corners")


def test_shot_conversion_is_goals_per_shot_on_target():
    rows = [
        {"home_score": 2, "away_score": 1, "home_sot": 6, "away_sot": 4},
        {"home_score": 1, "away_score": 0, "home_sot": 5, "away_sot": 5},
    ]
    assert shot_conversion(rows) == pytest.approx(4 / 20)


def test_shot_conversion_falls_back_when_nothing_is_usable():
    assert shot_conversion([{"home_score": 1, "away_score": 1,
                             "home_sot": None, "away_sot": None}]) == pytest.approx(0.30)


def test_a_zero_shot_match_is_kept():
    """0 is a real shot count, not a missing value. An earlier float parser
    treated anything <= 1 as absent, which would have silently deleted every
    genuinely one-sided match from the shot fit."""
    rows = synthetic_rows(n_rounds=3)
    rows[0]["home_sot"] = 0
    assert len(to_matches(rows, "sot")) == len(rows)
