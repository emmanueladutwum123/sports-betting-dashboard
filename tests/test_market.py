"""Tests for cross-book aggregation -- the layer where the original code was
mathematically wrong in two separate ways."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_event  # noqa: E402

from src.market import (  # noqa: E402
    BookQuote,
    MarketView,
    book_weight,
    collect_quotes,
    dispersion,
    is_sharp,
    pool_fair_probs,
)


def test_averaging_odds_is_biased_upward_versus_pooling_probabilities():
    """The Jensen's-inequality bug. Two books quoting 1.80 and 2.20 on the same
    outcome average to 2.00 in price space, but the mean of their implied
    probabilities corresponds to a *shorter* price. Averaging odds therefore
    invents value that is not there."""
    prices = [1.80, 2.20]
    mean_price = sum(prices) / 2
    price_from_mean_prob = 1.0 / (sum(1 / p for p in prices) / 2)
    assert mean_price > price_from_mean_prob


def test_sharp_books_outrank_soft_books():
    assert book_weight("pinnacle") > book_weight("draftkings") > book_weight("bovada")
    assert is_sharp("pinnacle") and not is_sharp("draftkings")
    assert book_weight("some_unknown_book") > 0


def test_each_book_is_devigged_separately():
    """A high-margin book and a low-margin book must both contribute fair
    probabilities, not raw ones -- otherwise the wider book drags the pool."""
    tight = BookQuote("pinnacle", "Pinnacle", {"A": 2.02, "B": 1.98})
    wide = BookQuote("bovada", "Bovada", {"A": 1.83, "B": 1.80})
    assert wide.margin > tight.margin
    assert sum(tight.fair.values()) == pytest.approx(1.0)
    assert sum(wide.fair.values()) == pytest.approx(1.0)


def test_pool_excludes_the_book_under_evaluation():
    """Leakage check. A book's own price must not appear in the benchmark used
    to judge that book's price."""
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 2.00, "Chelsea": 2.00}},
        "softbook": {"h2h": {"Arsenal": 3.00, "Chelsea": 1.45}},
    })
    quotes = collect_quotes(event, "h2h")
    with_soft = pool_fair_probs(quotes)
    without_soft = pool_fair_probs(quotes, exclude_book="softbook")
    assert with_soft["Arsenal"] != without_soft["Arsenal"]
    assert without_soft["Arsenal"] == pytest.approx(0.5, abs=1e-6)


def test_pooled_probabilities_sum_to_one():
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 1.90, "Draw": 3.60, "Chelsea": 4.20}},
        "williamhill": {"h2h": {"Arsenal": 1.85, "Draw": 3.50, "Chelsea": 4.00}},
    })
    pooled = pool_fair_probs(collect_quotes(event, "h2h"))
    assert sum(pooled.values()) == pytest.approx(1.0)


def test_incomplete_markets_are_dropped():
    """A lone Over with no Under has no recoverable margin."""
    event = {"bookmakers": [{"key": "b", "title": "B", "markets": [
        {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0}]}]}]}
    assert collect_quotes(event, "h2h") == []


def test_sharp_anchor_is_preferred_over_the_crowd():
    event = make_event({
        "pinnacle": {"h2h": {"Arsenal": 2.00, "Chelsea": 2.00}},
        "softa": {"h2h": {"Arsenal": 1.60, "Chelsea": 2.50}},
        "softb": {"h2h": {"Arsenal": 1.62, "Chelsea": 2.45}},
    })
    view = MarketView(event, "h2h")
    assert view.anchor_quality() == "sharp"
    # Three soft books cannot drag the anchor away from Pinnacle's 50/50.
    assert view.anchor()["Arsenal"] == pytest.approx(0.5, abs=1e-6)


def test_dispersion_measures_book_disagreement():
    agree = make_event({
        "a": {"h2h": {"X": 2.00, "Y": 2.00}}, "b": {"h2h": {"X": 2.01, "Y": 1.99}}})
    differ = make_event({
        "a": {"h2h": {"X": 1.50, "Y": 2.80}}, "b": {"h2h": {"X": 2.60, "Y": 1.55}}})
    assert dispersion(collect_quotes(differ, "h2h"), "X") > dispersion(
        collect_quotes(agree, "h2h"), "X")


def test_anchor_quality_degrades_with_thin_coverage():
    thin = make_event({"softa": {"h2h": {"X": 2.0, "Y": 2.0}}})
    assert MarketView(thin, "h2h").anchor_quality() == "unreliable"


def test_unmatched_exchange_placeholders_are_rejected():
    """Regression, from live data. Betfair reported 1.04/1.04/1.04 on a 1X2 with
    no matched liquidity -- a 65% "margin". That vector de-vigs to exactly 1/3
    per outcome, and because exchanges are weighted *sharp* it set the anchor
    rather than being outvoted by it, manufacturing a ~17% phantom edge on the
    draw at every other book. On one league slate it produced 71 fake +EV
    selections where only 1 was real."""
    placeholder = BookQuote("betfair_ex_eu", "Betfair", {"A": 1.04, "B": 1.04, "Draw": 1.04})
    assert placeholder.margin > 0.6
    assert not placeholder.is_plausible

    event = make_event({
        "betfair_ex_eu": {"h2h": {"Austria": 1.04, "Draw": 1.04, "Tirol": 1.04}},
        "pinnacle": {"h2h": {"Austria": 1.65, "Draw": 3.90, "Tirol": 5.20}},
        "skybet": {"h2h": {"Austria": 1.62, "Draw": 3.60, "Tirol": 4.80}},
    })
    quotes = collect_quotes(event, "h2h")
    assert "betfair_ex_eu" not in [q.key for q in quotes]
    # The anchor must now come from Pinnacle, nowhere near the placeholder's 1/3.
    assert MarketView(event, "h2h").anchor()["Draw"] < 0.30


def test_normal_margins_are_kept():
    """The filter must not throw away real books. Exchanges run near zero and a
    wide retail book runs into the teens; both are legitimate."""
    assert BookQuote("betfair_ex_uk", "Betfair", {"A": 2.02, "B": 2.02}).is_plausible
    assert BookQuote("skybet", "Sky Bet", {"A": 1.70, "B": 1.90}).is_plausible
    assert BookQuote("boylesports", "Boyle", {"A": 1.62, "B": 4.0, "D": 3.6}).is_plausible


def test_a_suspended_market_cannot_set_the_anchor_alone():
    """If every quote is a placeholder the market is unpriced -- return nothing
    rather than a confident wrong number."""
    event = make_event({
        "betfair_ex_eu": {"h2h": {"A": 1.04, "B": 1.04}},
        "betfair_ex_uk": {"h2h": {"A": 1.03, "B": 1.03}},
    })
    assert collect_quotes(event, "h2h") == []
    assert MarketView(event, "h2h").anchor() == {}
