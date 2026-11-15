"""Line shopping, exchange commission, stale quotes and the own-book checker."""

from datetime import UTC, datetime

import pytest

from src import shop
from src.market import BookQuote, best_price, collect_quotes, net_odds, staleness
from src.picks import best_per_selection, one_stake_per_market, scan_event, scan_slate
from tests.conftest import make_event

# A fair coin priced by a sharp book and a few recreational ones; "Soft" is the
# only book paying above fair on A.
BOARD = {
    "pinnacle": {"h2h": {"A": 1.98, "B": 1.98}},
    "williamhill": {"h2h": {"A": 1.91, "B": 1.91}},
    "unibet_uk": {"h2h": {"A": 1.90, "B": 1.92}},
    "skybet": {"h2h": {"A": 1.89, "B": 1.91}},
    "soft": {"h2h": {"A": 2.15, "B": 1.75}},
}


def stamp(event, times):
    """Attach per-book last_update timestamps to an event built by make_event."""
    for bm in event["bookmakers"]:
        if bm["key"] in times:
            for market in bm["markets"]:
                market["last_update"] = times[bm["key"]]
    return event


# ------------------------------------------------------------- commission

def test_bookmaker_prices_are_untouched_by_commission():
    assert net_odds("pinnacle", 2.50) == 2.50
    assert net_odds("some_new_book", 2.50) == 2.50


def test_exchange_price_is_net_of_commission_on_winnings_only():
    # 5% of the 6.60 profit, not 5% of the 7.60 return.
    assert net_odds("betfair_ex_uk", 7.60) == pytest.approx(1 + 6.60 * 0.95)
    assert net_odds("smarkets", 3.00) == pytest.approx(1 + 2.00 * 0.98)


def test_best_price_prefers_the_book_that_pays_more_after_commission():
    quotes = [
        BookQuote("betfair_ex_uk", "Betfair", {"A": 7.60, "B": 1.15}),
        BookQuote("unibet_uk", "Unibet", {"A": 7.50, "B": 1.12}),
    ]
    price, key, _ = best_price(quotes, "A")
    assert key == "unibet_uk" and price == 7.50


def test_commission_does_not_change_the_fair_probability():
    # Commission is a cost of taking the price, not part of the market's view.
    raw = BookQuote("betfair_ex_uk", "Betfair", {"A": 2.02, "B": 2.02})
    assert raw.fair["A"] == pytest.approx(0.5)


def test_scan_evaluates_exchange_at_net_price_and_says_so():
    event = make_event({**BOARD, "smarkets": {"h2h": {"A": 2.20, "B": 1.80}}})
    opp = next(o for o in scan_event(event, include_negative=True)
               if o.book_key == "smarkets" and o.selection == "A")
    assert opp.quoted_odds == 2.20
    assert opp.odds == pytest.approx(1 + 1.20 * 0.98)
    assert any(f.startswith("exchange") for f in opp.flags)


# --------------------------------------------------------------- staleness

def test_staleness_is_relative_to_the_market_not_the_clock():
    event = stamp(make_event(BOARD), {
        "pinnacle": "2026-01-01T12:00:00Z", "williamhill": "2026-01-01T12:01:00Z",
        "unibet_uk": "2026-01-01T12:02:00Z", "skybet": "2026-01-01T12:01:00Z",
        "soft": "2026-01-01T09:00:00Z",
    })
    lag = staleness(collect_quotes(event, "h2h"))
    assert lag["soft"] == pytest.approx(181.0)
    assert lag["pinnacle"] <= 1.0 and lag["unibet_uk"] == 0.0


def test_no_timestamps_means_nothing_is_called_stale():
    assert staleness(collect_quotes(make_event(BOARD), "h2h")) == {}


def test_stale_best_price_is_flagged_and_dropped_from_the_board():
    fresh = {k: "2026-01-01T12:00:00Z" for k in BOARD}
    event = stamp(make_event(BOARD), {**fresh, "soft": "2026-01-01T09:00:00Z"})
    flagged = [o for o in scan_event(event) if o.book_key == "soft"]
    assert flagged and any(f.startswith("stale-quote") for f in flagged[0].flags)
    assert not [o for o in scan_slate([event]) if o.book_key == "soft"]
    assert [o for o in scan_slate([event], drop_flagged=False) if o.book_key == "soft"]


def test_shopping_skips_a_stale_price_and_reports_what_it_ignored():
    fresh = {k: "2026-01-01T12:00:00Z" for k in BOARD}
    event = stamp(make_event(BOARD), {**fresh, "soft": "2026-01-01T09:00:00Z"})
    line = next(ln for ln in shop.best_lines(event) if ln.best.selection == "A")
    assert line.best.book_key == "pinnacle" and line.best.odds == 1.98
    assert line.stale_better.book_key == "soft"
    assert "ignored stale 2.15" in " ".join(line.notes())


# ------------------------------------------------------------ line shopping

def test_best_line_is_the_highest_price_with_runner_up_and_typical():
    line = next(ln for ln in shop.best_lines(make_event(BOARD)) if ln.best.selection == "A")
    assert (line.best.book_key, line.best.odds) == ("soft", 2.15)
    assert (line.runner_up.book_key, line.runner_up.odds) == ("pinnacle", 1.98)
    assert line.typical_odds == 1.91
    assert line.shop_gain == pytest.approx(2.15 / 1.91 - 1)
    assert line.n_offers == 5


def test_best_price_can_still_be_a_bad_bet_and_is_labelled_so():
    line = next(ln for ln in shop.best_lines(make_event(BOARD)) if ln.best.selection == "B")
    assert line.best.odds == 1.98  # the best B on the board...
    assert line.best.ev < 0.01  # ...is still not worth a stake
    assert line.verdict != shop.VALUE


def test_value_verdict_requires_a_stake_that_survives_shrinkage():
    assert shop.verdict_for(0.05, 0.01) == shop.VALUE
    assert shop.verdict_for(0.05, 0.0) == shop.NOISE
    assert shop.verdict_for(0.0, 0.0) == shop.FAIR
    assert shop.verdict_for(-0.05, 0.0) == shop.SKIP


def test_my_books_limits_the_offer_but_not_the_fair_line():
    event = make_event(BOARD)
    everything = next(ln for ln in shop.best_lines(event) if ln.best.selection == "A")
    mine = next(ln for ln in shop.best_lines(event, books={"williamhill", "skybet"})
                if ln.best.selection == "A")
    assert mine.best.book_key == "williamhill" and mine.best.odds == 1.91
    # Typical price and book count still describe the whole market.
    assert mine.typical_odds == everything.typical_odds
    assert mine.n_offers == everything.n_offers


def test_a_market_with_too_few_books_is_not_shopped():
    thin = make_event({"pinnacle": {"h2h": {"A": 1.98, "B": 1.98}},
                       "soft": {"h2h": {"A": 2.15, "B": 1.75}}})
    assert shop.best_lines(thin) == []


def test_shopping_agrees_with_the_ev_board_about_what_is_value():
    event = make_event(BOARD)
    board = {(o.selection, o.book_key): o.ev for o in scan_slate([event], best_only=True)}
    for line in shop.best_lines(event):
        key = (line.best.selection, line.best.book_key)
        if key in board:
            assert line.best.ev == pytest.approx(board[key])


# ------------------------------------------------------------- board dedupe

def test_best_only_keeps_one_row_per_selection():
    board = {**BOARD, "soft2": {"h2h": {"A": 2.10, "B": 1.78}}}
    event = make_event(board)
    every = [o for o in scan_slate([event]) if o.selection == "A"]
    best = [o for o in scan_slate([event], best_only=True) if o.selection == "A"]
    assert len(every) == 2
    assert len(best) == 1 and best[0].book_key == "soft"


def test_best_per_selection_keeps_separate_matches_separate():
    a = make_event(BOARD, home="A", away="B", event_id="e1")
    b = make_event(BOARD, home="A", away="B", event_id="e2")
    assert len(best_per_selection(scan_slate([a, b]))) == 2


def test_only_one_selection_per_market_per_match_is_staked():
    # Both sides priced above fair at different books: they cannot both win.
    board = {**BOARD, "soft2": {"h2h": {"A": 1.75, "B": 2.15}}}
    opps = one_stake_per_market(best_per_selection(scan_event(make_event(board))))
    assert {o.selection for o in opps} == {"A", "B"}
    staked = [o for o in opps if o.stake_fraction > 0]
    assert len(staked) == 1
    loser = next(o for o in opps if o.stake_fraction == 0)
    assert any("overlaps" in f for f in loser.flags)


# ----------------------------------------------------------------- in-play

def test_started_matches_are_separated_from_upcoming_ones():
    now = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
    started = make_event(BOARD, event_id="live")  # kicks off 14:00
    later = make_event(BOARD, event_id="later")
    later["commence_time"] = "2026-09-02T14:00:00Z"
    upcoming, in_play = shop.split_upcoming([started, later], now)
    assert [e["id"] for e in upcoming] == ["later"]
    assert [e["id"] for e in in_play] == ["live"]


def test_unparseable_start_time_is_not_assumed_to_be_in_play():
    event = make_event(BOARD)
    event["commence_time"] = None
    upcoming, in_play = shop.split_upcoming([event])
    assert upcoming and not in_play


# ------------------------------------------------------- own-book checker

def test_check_price_above_fair_is_value_with_a_stake():
    res = shop.check_price(make_event(BOARD), "h2h", None, "A", 2.20)
    assert res["verdict"] == shop.VALUE
    assert res["ev"] > 0 and res["stake_fraction"] > 0
    assert res["offered_odds"] > res["min_odds_for_value"] > res["fair_odds"]


def test_check_price_below_fair_is_overpriced_with_no_stake():
    res = shop.check_price(make_event(BOARD), "h2h", None, "A", 1.70)
    assert res["verdict"] == shop.SKIP
    assert res["ev"] < 0 and res["stake_fraction"] == 0


def test_check_price_reports_how_it_compares_with_the_feed():
    res = shop.check_price(make_event(BOARD), "h2h", None, "A", 2.00)
    assert res["best_feed_odds"] == 2.15 and res["best_feed_book"] == "Soft"
    assert res["beats_books"] == 4 and res["n_books"] == 5


def test_check_price_flags_an_implausibly_good_price():
    assert shop.check_price(make_event(BOARD), "h2h", None, "A", 3.50)["suspicious"]
    assert not shop.check_price(make_event(BOARD), "h2h", None, "A", 2.10)["suspicious"]


def test_check_price_handles_bad_input():
    event = make_event(BOARD)
    assert shop.check_price(event, "h2h", None, "A", 1.0) is None
    assert shop.check_price(event, "h2h", None, "Nobody", 2.0) is None
    assert shop.check_price(event, "totals", 2.5, "Over", 2.0) is None


def test_markets_for_lists_only_markets_that_can_be_priced():
    board = {k: {**v, "totals": {2.5: {"Over": 1.90, "Under": 1.90}}} for k, v in BOARD.items()}
    board["soft"]["totals"][3.5] = {"Over": 3.0, "Under": 1.35}  # one book only
    labels = [m[0] for m in shop.markets_for(make_event(board, home="A", away="B"))]
    assert labels == ["Match result", "Total 2.5"]


def test_books_in_lists_every_book_once_sorted_by_title():
    books = shop.books_in([make_event(BOARD), make_event(BOARD)])
    assert list(books) == sorted(BOARD, key=lambda k: k.title().lower())


# --------------------------------------------------- hand-typed book prices

def _board_with_totals():
    board = {k: {**v, "totals": {2.5: {"Over": 1.90, "Under": 1.90}}} for k, v in BOARD.items()}
    event = make_event(board, home="Arsenal", away="Chelsea")
    # Relabel the outcomes so Home/Away resolve to real team names.
    for bm in event["bookmakers"]:
        for market in bm["markets"]:
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    outcome["name"] = {"A": "Arsenal", "B": "Chelsea"}[outcome["name"]]
    return {"Arsenal vs Chelsea": event}


def _row(selection, odds, market="Match result", match="Arsenal vs Chelsea"):
    return {"match": match, "market": market, "selection": selection, "odds": odds}


def test_manual_price_is_judged_like_any_other_price():
    events = _board_with_totals()
    offers, problems = shop.manual_offers(events, [_row("Home", 2.20)])
    assert problems == []
    opp, res = offers[0]
    direct = shop.check_price(events["Arsenal vs Chelsea"], "h2h", None, "Arsenal", 2.20)
    assert opp.selection == "Arsenal" and opp.odds == 2.20
    assert opp.ev == pytest.approx(direct["ev"])
    assert res["verdict"] == shop.VALUE and opp.stake_fraction > 0
    assert opp.book_title == "SportyBet (yours)"


def test_manual_sides_resolve_to_the_right_team_and_totals_line():
    offers, problems = shop.manual_offers(_board_with_totals(), [
        _row("Away", 1.80), _row("Over", 1.95, market="Total 2.5"),
    ])
    assert problems == []
    assert [(o.market, o.point, o.selection) for o, _ in offers] == [
        ("h2h", None, "Chelsea"), ("totals", 2.5, "Over"),
    ]


def test_manual_rows_that_cannot_be_priced_are_explained_not_dropped():
    offers, problems = shop.manual_offers(_board_with_totals(), [
        _row("Over", 2.0),                           # totals side on a 1X2 market
        _row("Home", 2.0, market="Total 2.5"),       # 1X2 side on a totals market
        _row("Over", 2.0, market="Total 4.5"),       # line nobody quotes
        _row("Home", 2.0, match="Gone vs Missing"),  # match not on the board
        _row("Home", 1.0),                           # not a valid price
        _row("Home", None),                          # incomplete
        {"match": None, "market": None, "selection": None, "odds": None},  # blank row
    ])
    assert offers == []
    assert len(problems) == 6  # the blank row is ignored, the rest explained
    assert all(p.startswith("Row ") for p in problems)


def test_manual_duplicate_selection_is_rejected_once():
    offers, problems = shop.manual_offers(
        _board_with_totals(), [_row("Home", 2.10), _row("Home", 2.20)]
    )
    assert len(offers) == 1 and offers[0][0].odds == 2.10
    assert len(problems) == 1 and "twice" in problems[0]


def test_manual_typo_gets_no_stake_and_cannot_become_best_price():
    events = _board_with_totals()
    offers, _ = shop.manual_offers(events, [_row("Home", 21.0)])  # meant 2.10
    opp, res = offers[0]
    assert res["verdict"] == shop.VERIFY and opp.stake_fraction == 0
    line = next(ln for ln in shop.best_lines(events["Arsenal vs Chelsea"], extra_offers=[opp])
                if ln.best.selection == "Arsenal")
    assert line.best.book_key != shop.MANUAL_BOOK_KEY


def test_manual_price_competes_for_best_without_moving_the_market_numbers():
    events = _board_with_totals()
    event = events["Arsenal vs Chelsea"]
    before = next(ln for ln in shop.best_lines(event) if ln.best.selection == "Arsenal")
    offers, _ = shop.manual_offers(events, [_row("Home", 2.20)])
    after = next(ln for ln in shop.best_lines(event, extra_offers=[offers[0][0]])
                 if ln.best.selection == "Arsenal")
    assert after.best.book_key == shop.MANUAL_BOOK_KEY and after.best.odds == 2.20
    assert after.runner_up.odds == before.best.odds
    assert after.typical_odds == before.typical_odds and after.n_offers == before.n_offers


def test_only_my_manual_book_shows_only_what_i_entered():
    events = _board_with_totals()
    offers, _ = shop.manual_offers(events, [_row("Home", 1.85)])
    lines = shop.best_lines(events["Arsenal vs Chelsea"], books=set(),
                            extra_offers=[offers[0][0]])
    assert [(ln.best.selection, ln.best.odds) for ln in lines] == [("Arsenal", 1.85)]


def test_market_labels_cover_every_quoted_line():
    events = list(_board_with_totals().values())
    assert shop.market_labels(events) == ["Match result", "Total 2.5"]
