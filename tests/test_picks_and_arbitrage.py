"""End-to-end tests over Odds API-shaped events."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_event  # noqa: E402

from src.arbitrage import find_arbitrage, find_middles, stake_split  # noqa: E402
from src.picks import (  # noqa: E402
    best_total_bet,
    scan_event,
    scan_slate,
    sharp_disagreement,
    summarize_h2h,
)


def test_an_efficient_board_yields_no_bets():
    """The correct output most of the time. Every book agrees; nothing is
    mispriced; the scanner must return empty rather than manufacture a pick."""
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 1.95, "Chelsea": 1.95}},
        "williamhill": {"h2h": {"Arsenal": 1.92, "Chelsea": 1.92}},
        "unibet_uk": {"h2h": {"Arsenal": 1.90, "Chelsea": 1.90}},
    })
    assert scan_event(event) == []


def test_an_outlier_price_is_found_and_sized():
    """One soft book badly out of line with a sharp anchor is the whole point."""
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 2.00, "Chelsea": 2.00}},
        "williamhill": {"h2h": {"Arsenal": 1.95, "Chelsea": 1.95}},
        "bovada": {"h2h": {"Arsenal": 2.40, "Chelsea": 1.65}},
    })
    found = scan_event(event)
    assert found, "a 2.40 against a fair 2.00 must be surfaced"
    top = found[0]
    assert top.selection == "Arsenal"
    assert top.book_key == "bovada"
    assert top.ev > 0.10
    assert 0 < top.stake_fraction < 1


def test_the_book_being_judged_is_excluded_from_its_own_benchmark():
    """Without exclusion the outlier partly sets its own fair value and its edge
    is understated -- the leakage that makes edges vanish exactly when real."""
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 2.00, "Chelsea": 2.00}},
        "bovada": {"h2h": {"Arsenal": 2.40, "Chelsea": 1.65}},
    })
    top = scan_event(event)[0]
    assert top.fair_prob == pytest.approx(0.5, abs=1e-6)


def test_stakes_never_exceed_the_slate_exposure_cap():
    event = make_event({
        "pinnacle": {"h2h": {"A": 2.00, "B": 2.00}},
        "bovada": {"h2h": {"A": 3.00, "B": 1.45}},
    })
    sized = scan_slate([event] * 12, exposure_cap=0.10)
    assert sum(o.stake_fraction for o in sized) <= 0.10 + 1e-9


def test_arbitrage_is_detected_and_stakes_are_balanced():
    event = make_event({
        "booka": {"h2h": {"Arsenal": 2.20, "Chelsea": 1.75}},
        "bookb": {"h2h": {"Arsenal": 1.75, "Chelsea": 2.30}},
    })
    arb = find_arbitrage(event, "h2h")
    assert arb and arb["booksum"] < 1.0 and arb["profit_pct"] > 0
    # Balanced stakes mean identical payout whichever way the game goes.
    payouts = [leg["stake_pct"] * leg["odds"] for leg in arb["legs"]]
    assert payouts[0] == pytest.approx(payouts[1], rel=1e-3)  # stake_pct is rounded to 2dp for display


def test_no_arbitrage_on_a_normal_board():
    event = make_event({
        "booka": {"h2h": {"Arsenal": 1.90, "Chelsea": 1.90}},
        "bookb": {"h2h": {"Arsenal": 1.88, "Chelsea": 1.92}},
    })
    assert find_arbitrage(event, "h2h") is None


def test_large_arbitrage_is_flagged_as_suspect_data():
    """A big arb arises from one book holding a stale line while the other has
    moved. Each book's own vector still carries a normal positive margin -- a
    book quoting a negative margin against itself is a data error, not an
    opportunity, and is filtered out upstream."""
    event = make_event({
        "booka": {"h2h": {"Arsenal": 5.00, "Chelsea": 1.22}},   # margin ~1.9%
        "bookb": {"h2h": {"Arsenal": 1.60, "Chelsea": 2.50}},   # margin ~2.4%, stale
    })
    arb = find_arbitrage(event, "h2h")
    assert arb is not None and arb["profit_pct"] > 3
    assert arb["suspect"] is True


def test_a_book_pricing_against_itself_is_not_treated_as_an_arb():
    """A single book quoting a sub-1.0 booksum is a feed error. Reporting it as
    free money would be the most expensive kind of false positive."""
    event = make_event({"booka": {"h2h": {"Arsenal": 5.00, "Chelsea": 1.60}}})
    assert find_arbitrage(event, "h2h") is None


def test_stake_split_equalises_payouts():
    stakes = stake_split([2.5, 1.8])
    assert stakes[0] * 2.5 == pytest.approx(stakes[1] * 1.8)


def test_middles_are_found_between_books():
    event = make_event({"booka": {"totals": {2.5: {"Over": 2.00, "Under": 1.85}}},
                        "bookb": {"totals": {3.5: {"Over": 2.60, "Under": 1.50}}}})
    middles = find_middles(event)
    assert middles and middles[0]["gap"] == pytest.approx(1.0)
    assert middles[0]["over_line"] < middles[0]["under_line"]


def test_totals_pick_is_ev_ranked_not_balance_point():
    """The behavioural change from the old version: given a flat 50/50 line and
    a mispriced lopsided one, pick the mispriced one."""
    event = make_event({
        "pinnacle": {"totals": {2.5: {"Over": 2.00, "Under": 2.00},
                                3.5: {"Over": 3.00, "Under": 1.45}}},
        "bovada":   {"totals": {2.5: {"Over": 1.98, "Under": 1.98},
                                3.5: {"Over": 3.90, "Under": 1.32}}},
    })
    pick = best_total_bet(event)
    assert pick["point"] == 3.5 and pick["side"] == "Over" and pick["positive_ev"]


def test_summary_reports_anchor_quality_and_margin():
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 1.95, "Draw": 3.70, "Chelsea": 4.30}},
        "bovada": {"h2h": {"Arsenal": 1.85, "Draw": 3.50, "Chelsea": 4.00}},
    })
    summary = summarize_h2h(event)
    assert summary["anchor"] == "sharp"
    assert summary["avg_margin_pct"] > 0
    assert sum(o["fair_prob"] for o in summary["outcomes"].values()) == pytest.approx(1.0, abs=1e-3)


def test_sharp_versus_soft_disagreement_is_reported():
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 2.00, "Chelsea": 2.00}},
        "bovada": {"h2h": {"Arsenal": 1.55, "Chelsea": 2.70}},
        "betmgm": {"h2h": {"Arsenal": 1.57, "Chelsea": 2.65}},
    })
    gap = sharp_disagreement(event)
    assert gap and abs(gap["gap_pp"]) > 5
