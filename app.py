"""Quantitative betting research dashboard.

Tabs, in the order you should actually use them:

Best Odds      who pays the most for every selection, whether even that is
               enough, and a checker for prices from your own bookmaker.
Daily Card     the strongest-evidence selections across many competitions.
+EV Board      every selection currently priced above its fair value, ranked by
               expected value and sized by fractional Kelly.
Fixtures       per-match breakdown: fair line, best price, book disagreement.
Arb & Middles  riskless-ish positions, and a live integrity check on the feed.
Model Lab      fit Dixon-Coles on real historical results and price any matchup.
Ledger & CLV   log what you bet, grade yourself on closing line value.
Methodology    what the numbers mean and what they cannot do.

Run: ./start.sh   (or `streamlit run app.py` with ODDS_API_KEY set)
"""
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import (  # noqa: E402
    arbitrage,
    calibration,
    daily,
    form,
    ledger,
    odds_api,
    picks,
    render,
    shop,
)
from src.devig import METHODS  # noqa: E402
from src.edge import (  # noqa: E402
    DEFAULT_EXPOSURE_CAP,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_SHRINK_K,
    normalise_slate,
)

st.set_page_config(page_title="Quant Betting Research", layout="wide", page_icon="📊")
st.markdown(render.inject_css(), unsafe_allow_html=True)

st.title("📊 Quantitative Betting Research")
st.caption(
    "Fair prices from per-book Shin de-vigging pooled in log-odds space and anchored "
    "to sharp books · expected value and fractional-Kelly sizing · closing-line-value "
    "tracking · Dixon-Coles goal model with a walk-forward backtest."
)

try:
    odds_api.check_api_key()
except odds_api.OddsAPIError as e:
    st.error(str(e))
    st.stop()


# --- Caching. Odds are quota-metered, so cache generously. -------------------
@st.cache_data(ttl=1800, show_spinner="Checking in-season leagues...")
def cached_sports_by_group():
    return odds_api.in_season_sports_by_group()


@st.cache_data(ttl=600, show_spinner=False)
def cached_odds(sport_key: str, regions_arg: str):
    return odds_api.get_odds(sport_key, regions=regions_arg)


@st.cache_data(ttl=300, show_spinner=False)
def cached_scores(sport_key: str):
    return odds_api.get_scores(sport_key, days_from=1)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_form(team_name: str):
    return form.get_form(team_name)


@st.cache_data(ttl=900, show_spinner=False)
def cached_h2h(sport_key: str, regions_arg: str):
    """Head-to-head only. The Odds API bills one credit per region per market
    per league, so restricting the card to h2h and a single region is the
    difference between 1 credit a league and 4."""
    return odds_api.get_odds(sport_key, regions=regions_arg, markets="h2h")


@st.cache_resource(show_spinner="Fetching historical results...")
def cached_history(division: str, years: tuple):
    from src.datafeeds.football_data import fetch_seasons

    return fetch_seasons(division, list(years))


@st.cache_resource(show_spinner="Fitting Dixon-Coles...")
def cached_model(division: str, years: tuple, xi: float, shot_weight: float = None):
    from src.datafeeds.football_data import to_matches
    from src.models.dixon_coles import SHOT_WEIGHT, DixonColesModel, ShotAugmentedDixonColes

    rows = cached_history(division, years)
    if len(rows) < 100:
        return None, rows
    weight = SHOT_WEIGHT if shot_weight is None else shot_weight
    if weight <= 0.0:
        return DixonColesModel(max_goals=10).fit(to_matches(rows), xi=xi), rows
    return ShotAugmentedDixonColes(max_goals=10, shot_weight=weight).fit(rows, xi=xi), rows


try:
    sports_by_group = cached_sports_by_group()
except odds_api.OddsAPIError as e:
    st.error(str(e))
    st.stop()

# --- Sidebar -----------------------------------------------------------------
st.sidebar.header("Data")
st.sidebar.caption(
    "Each league costs API quota on every uncached refresh (free tier: 500/month). "
    "More regions = more books = better fair-value estimates, but more quota."
)
groups = st.sidebar.multiselect("Sports", list(odds_api.SPORT_GROUPS), default=["Soccer"])
regions = st.sidebar.multiselect(
    "Bookmaker regions", ["eu", "uk", "us", "au"], default=["eu", "uk"],
    help="Sharp books (Pinnacle) live in 'eu'. Keep it selected — the fair line depends on it.",
)
regions_str = ",".join(regions) if regions else "eu"

# Preselect the most heavily traded competitions that are in season, rather
# than whatever sorts first alphabetically: more books quote them, so the fair
# line is tighter and there are more prices to shop between.
PREFERRED_LEAGUES = (
    "soccer_epl", "soccer_uefa_champs_league", "soccer_spain_la_liga",
    "soccer_italy_serie_a", "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "basketball_nba", "basketball_euroleague", "baseball_mlb",
)

selected_leagues = {}
for group in groups:
    available = sports_by_group.get(group, [])
    titles = [x["title"] for x in available]
    by_key = {x["key"]: x["title"] for x in available}
    preferred = [by_key[k] for k in PREFERRED_LEAGUES if k in by_key][:3]
    selected_leagues[group] = st.sidebar.multiselect(
        f"{group} leagues", titles, default=preferred or titles[:3]
    )

st.sidebar.header("Model & staking")
devig_method = st.sidebar.selectbox(
    "De-vig method", METHODS, index=0,
    help="Shin models informed money and is the best-supported choice. "
         "Multiplicative is the naive baseline; compare them to see the bias.",
)
min_ev = st.sidebar.slider(
    "Minimum EV to show (%)", 0.0, 15.0, 2.0, 0.5,
    help="Below ~1-2% the edge is inside the noise of the fair-value estimate.",
) / 100.0
kelly_mult = st.sidebar.slider(
    "Kelly fraction", 0.05, 1.0, DEFAULT_KELLY_FRACTION, 0.05,
    help="Fraction of full Kelly. Above 1.0 is mathematically ruinous; 0.25 is "
         "the sane default when probabilities are estimated rather than known.",
)
shrink_k = st.sidebar.slider(
    "Uncertainty shrinkage (σ)", 0.0, 3.0, DEFAULT_SHRINK_K, 0.25,
    help="Standard errors of book disagreement subtracted from the fair "
         "probability before sizing. Higher = more conservative.",
)
exposure_cap = st.sidebar.slider(
    "Max slate exposure (%)", 1.0, 50.0, DEFAULT_EXPOSURE_CAP * 100, 1.0,
    help="Total bankroll at risk across all simultaneous bets.",
) / 100.0
bankroll = st.sidebar.number_input("Bankroll", min_value=1.0, value=1000.0, step=50.0)

if st.sidebar.button("🔄 Force refresh (clears cache)"):
    st.cache_data.clear()


def _pct_near_one(p: float) -> str:
    """Format a probability that may sit very close to 1 without rounding to
    '100%'. A 5-leg longshot parlay busts 99.985% of the time; printing that as
    100% reads as certainty and is the kind of small dishonesty this whole app
    is built to avoid."""
    pct = p * 100
    if pct >= 99.95:
        return f">{min(pct, 99.99):.2f}%"
    return f"{pct:.1f}%"


def parse_iso(ts: str):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def fmt_when(ts: str) -> str:
    dt = parse_iso(ts)
    return dt.astimezone().strftime("%a %d %b %H:%M") if dt else str(ts)


@st.cache_data(ttl=600, show_spinner="Loading odds...")
def load_events(groups_key: tuple, leagues_key: tuple, regions_arg: str):
    """All events across the selected leagues, tagged with league + sport."""
    events, quota = [], None
    for group, titles in leagues_key:
        for league in sports_by_group.get(group, []):
            if league["title"] not in titles:
                continue
            try:
                found, quota = cached_odds(league["key"], regions_arg)
            except odds_api.OddsAPIError as exc:
                st.warning(f"{league['title']}: {exc}")
                continue
            for ev in found:
                ev["_league"] = league["title"]
                ev["_group"] = group
                ev["_sport_key"] = league["key"]
                events.append(ev)
    events.sort(key=lambda e: e.get("commence_time") or "")
    return events, quota


leagues_key = tuple((g, tuple(selected_leagues.get(g, []))) for g in groups)
events, quota_left = ([], None)
if groups and any(t for _, t in leagues_key):
    events, quota_left = load_events(tuple(groups), leagues_key, regions_str)

# Anything that recommends a bet works from matches that have not started. In
# play, books reprice at different speeds and the gaps between them are latency,
# not value.
upcoming, in_play = shop.split_upcoming(events)
in_play_ids = {e.get("id") for e in in_play}

st.sidebar.header("Your bookmakers")
feed_books = shop.books_in(events)
my_book_keys = st.sidebar.multiselect(
    "Books I can bet with", list(feed_books), format_func=lambda k: feed_books[k],
    help="Leave empty to shop every book in the feed. Pick yours and the best price "
         "shown is the best one you can actually take — fair value is still built "
         "from all of them.",
)
my_books = set(my_book_keys) or None
st.sidebar.caption(
    "Book not listed (SportyBet, bet365…)? Type its prices in under **Add your "
    "SportyBet prices** on the Best Odds tab."
)

tab_shop, tab_card, tab_ev, tab_fix, tab_arb, tab_model, tab_ledger, tab_docs = st.tabs(
    ["🛒 Best Odds", "🔥 Daily Card", "🎯 +EV Board", "🗓️ Fixtures", "⚖️ Arb & Middles",
     "🧪 Model Lab", "📒 Ledger & CLV", "📘 Methodology"]
)

# =============================================================== BEST ODDS ===
with tab_shop:
    st.subheader("Best available price for every selection")
    st.caption(
        "Same bet, same outcome, more money back: taking the best price instead of a "
        "typical one is the only edge here that needs no forecasting. Prices are net of "
        "exchange commission, stale quotes are set aside, and every row is judged against "
        "fair value — because the best price on the board is usually still not a bet."
    )
    if not events:
        st.info("Select at least one league in the sidebar to compare prices.")
    elif not upcoming:
        st.info("Every loaded match has already kicked off. Pick another league.")
    else:
        if in_play:
            st.caption(f"{len(in_play)} in-play match(es) excluded — live prices are not comparable.")
        by_label = {
            f"{e.get('home_team')} vs {e.get('away_team')} · {fmt_when(e.get('commence_time'))}": e
            for e in upcoming
        }

        # ---- Prices typed in from a book outside the feed -------------------
        entries = st.session_state.setdefault("manual_entries", [])
        with st.expander("➕ Check and add your SportyBet prices (or any book not in the feed)",
                         expanded=True):
            st.caption(
                "SportyBet and bet365 have no official data feed, so their odds are entered "
                "by hand. Pick the match, market and selection, type the decimal odds exactly "
                "as your book shows them, and the verdict appears straight away. **Add** keeps "
                "the price so it competes in the table below alongside every other book. "
                "Saved prices last for this browser session and do not update themselves — "
                "re-check them before staking."
            )
            m1, m2 = st.columns([1, 2])
            with m1:
                manual_book = (st.text_input("Bookmaker name", "SportyBet", key="manual_book")
                               .strip() or "SportyBet")
            with m2:
                st.write("")
                only_manual = st.checkbox(
                    f"I can only bet at {manual_book} — hide prices I cannot take",
                    key="manual_only",
                    help="The table below then shows only the selections you added. "
                         "Fair value is still built from every book in the feed.",
                )
            p1, p2, p3, p4 = st.columns([3, 2, 2, 1])
            with p1:
                chosen_event = by_label[st.selectbox("Match", list(by_label), key="pc_match")]
            check_markets = shop.markets_for(chosen_event, devig_method)
            if not check_markets:
                st.info("No market on this match has enough books to build a fair price.")
            else:
                with p2:
                    mk_label = st.selectbox("Market", [m[0] for m in check_markets], key="pc_market")
                _, mk_key, mk_point, mk_selections = next(m for m in check_markets if m[0] == mk_label)
                with p3:
                    sel_name = st.selectbox("Selection", mk_selections, key="pc_sel")
                # Start from the typical book's price for this exact selection, and
                # reset whenever the selection changes: a leftover number from a
                # different bet would be judged as if it were an offer on this one.
                typical = shop.check_price(chosen_event, mk_key, mk_point, sel_name, 2.0,
                                           devig_method)
                start_odds = max(1.01, round((typical or {}).get("typical_odds") or 2.0, 2))
                with p4:
                    offered = st.number_input(
                        "Your odds", min_value=1.01, value=start_odds, step=0.01, format="%.2f",
                        key=f"pc_odds_{chosen_event.get('id')}_{mk_label}_{sel_name}",
                    )
                res = shop.check_price(
                    chosen_event, mk_key, mk_point, sel_name, offered, devig_method,
                    kelly_mult, shrink_k, min_ev,
                )
                if res is None:
                    st.info("No fair price available for that selection.")
                else:
                    # An implausible price gets no verdict and no stake until it
                    # has been checked -- sizing a typo is how a bankroll goes.
                    st.markdown(
                        render.stat_tiles([
                            ("Verdict", "⚠️ Verify first" if res["suspicious"] else res["verdict"]),
                            ("Your odds", f"{res['offered_odds']:.2f}"),
                            ("Fair odds", f"{res['fair_odds']:.2f}"),
                            ("Value starts at", f"{res['min_odds_for_value']:.2f}"),
                            ("EV", f"{res['ev'] * 100:+.2f}%"),
                            ("Stake", "—" if res["suspicious"]
                             else f"{bankroll * res['stake_fraction']:.2f}"),
                        ]),
                        unsafe_allow_html=True,
                    )
                    compare = (
                        f"The best price in the feed is **{res['best_feed_odds']:.2f}** at "
                        f"{res['best_feed_book']} (typical book: {res['typical_odds']:.2f}); yours "
                        f"beats {res['beats_books']} of {res['n_books']} books."
                    )
                    basis = (
                        f"Fair win probability **{res['fair_prob'] * 100:.1f}%** from "
                        f"{res['n_books']} books ({res['anchor_quality']} anchor, "
                        f"{res['sigma'] * 100:.1f}pp disagreement)."
                    )
                    if res["suspicious"]:
                        st.warning(
                            f"**{res['ev'] * 100:+.1f}% EV is too good to take at face value.** "
                            "Check that the market matches exactly (1X2 vs draw-no-bet vs double "
                            "chance, the same totals line, 90 minutes vs including extra time) and "
                            f"that the odds are typed correctly. {compare}"
                        )
                    elif res["verdict"] == shop.VALUE:
                        st.success(
                            f"**Value.** {res['offered_odds']:.2f} is above the fair "
                            f"{res['fair_odds']:.2f}. Quarter-Kelly on the conservative probability "
                            f"sizes it at **{res['stake_fraction'] * 100:.2f}% of bankroll "
                            f"({bankroll * res['stake_fraction']:.2f})**. {basis} {compare}"
                        )
                    elif res["verdict"] == shop.NOISE:
                        st.info(
                            f"**Positive on paper, not stakeable.** The edge "
                            f"({res['ev'] * 100:+.2f}%) is smaller than the books' disagreement "
                            f"about the true price, so the sized stake is zero. {basis} {compare}"
                        )
                    elif res["verdict"] == shop.FAIR:
                        st.info(
                            f"**About fair — no edge.** You would need "
                            f"**{res['min_odds_for_value']:.2f} or better** for this to be a bet. "
                            f"{basis} {compare}"
                        )
                    else:
                        st.error(
                            f"**Overpriced — skip, or take it elsewhere.** At "
                            f"{res['offered_odds']:.2f} you give up {abs(res['ev']) * 100:.1f}% of "
                            f"every stake in expectation. It becomes a bet at "
                            f"**{res['min_odds_for_value']:.2f} or better**. {basis} {compare}"
                        )

            if check_markets and res is not None:
                home_name, away_name = chosen_event.get("home_team"), chosen_event.get("away_team")
                side = {home_name: "Home", away_name: "Away"}.get(sel_name, sel_name)
                match_label = next(k for k, v in by_label.items() if v is chosen_event)
                if st.button(f"➕ Add to my {manual_book} prices", type="primary"):
                    # Re-entering a selection replaces its old price.
                    entries[:] = [
                        e for e in entries
                        if (e["match"], e["market"], e["selection"]) != (match_label, mk_label, side)
                    ]
                    entries.append({"match": match_label, "market": mk_label,
                                    "selection": side, "odds": float(offered)})
                    st.rerun()

        manual, manual_problems = shop.manual_offers(
            by_label, entries, manual_book, devig_method, kelly_mult, shrink_k, min_ev,
        )
        for problem in manual_problems:
            st.warning(problem)
        if entries:
            r1, r2, r3 = st.columns([4, 1, 1])
            labels = [
                f"{e['match'].split(' · ')[0]} — {e['selection']} ({e['market']}) @ {e['odds']:.2f}"
                for e in entries
            ]
            with r1:
                to_remove = st.multiselect(
                    "Saved prices", range(len(entries)), format_func=lambda i: labels[i],
                    placeholder=f"{len(entries)} saved — select any to remove",
                    label_visibility="collapsed", key=f"manual_remove_{len(entries)}",
                )
            with r2:
                if st.button("Remove selected", disabled=not to_remove, width="stretch"):
                    entries[:] = [e for i, e in enumerate(entries) if i not in to_remove]
                    st.rerun()
            with r3:
                if st.button("Clear all", width="stretch"):
                    entries.clear()
                    st.rerun()

        shop_books = (set(my_book_keys) if only_manual and manual else my_books)
        lines = shop.shop_slate(
            upcoming, devig_method=devig_method, kelly_multiplier=kelly_mult,
            shrink_k=shrink_k, books=shop_books, min_ev=min_ev,
            extra_offers=[o for o, _ in manual],
        )
        # Every recommended stake on this tab shares one bankroll, whether the
        # price came from the feed or was typed in: at most one per market per
        # match, then scaled to the exposure cap -- the same rules as the +EV
        # board. Sized once, together, so the two tables cannot add up to more
        # than the cap between them.
        staked = {id(ln.best): ln.best for ln in lines if ln.verdict == shop.VALUE}
        staked.update({id(o): o for o, r in manual if r["verdict"] == shop.VALUE})
        normalise_slate(picks.one_stake_per_market(list(staked.values())), exposure_cap)

        if manual:
            st.markdown(f"#### Your {manual_book} prices")
            manual_rows = []
            for opp, res in manual:
                gap = res["best_feed_odds"] - opp.odds if res["best_feed_odds"] else 0.0
                manual_rows.append({
                    "Verdict": res["verdict"],
                    "Match": f"{opp.home} vs {opp.away}",
                    "Market": shop.MATCH_RESULT if opp.market == "h2h"
                    else shop.total_label(opp.point),
                    "Selection": opp.selection,
                    f"{manual_book} odds": opp.odds,
                    "Fair odds": round(res["fair_odds"], 2),
                    "Value starts at": round(res["min_odds_for_value"], 2),
                    "EV %": round(opp.ev * 100, 2),
                    "Stake": round(bankroll * (opp.stake_fraction or 0), 2),
                    "Stake %": round((opp.stake_fraction or 0) * 100, 2),
                    "Best in feed": f"{res['best_feed_odds']:.2f} @ {res['best_feed_book']}",
                    "Vs best": "better here" if gap < 0 else "same" if gap == 0
                    else f"{gap:.2f} better elsewhere",
                    "Win %": round(res["fair_prob"] * 100, 1),
                    "Notes": "; ".join(opp.flags),
                })
            st.dataframe(
                pd.DataFrame(manual_rows), width="stretch", hide_index=True,
                column_config={
                    f"{manual_book} odds": st.column_config.NumberColumn(format="%.2f"),
                    "EV %": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Stake": st.column_config.NumberColumn(format="%.2f"),
                    "Stake %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            n_value = sum(1 for o, r in manual if r["verdict"] == shop.VALUE)
            n_verify = sum(1 for o, r in manual if r["verdict"] == shop.VERIFY)
            st.caption(
                f"**{n_value} of {len(manual)}** {manual_book} prices are value. "
                "**Value starts at** is the lowest odds at which the selection becomes a bet. "
                + (f"**{n_verify}** look too good to be true and get no stake until you have "
                   "re-checked the odds and that the market matches exactly. " if n_verify else "")
                + "Stakes share one bankroll: one per market per match, capped at your "
                "slate exposure."
            )
            st.divider()

        gains = sorted(ln.shop_gain for ln in lines)
        st.markdown(
            render.stat_tiles([
                ("Matches", len(upcoming)),
                ("Selections priced", len(lines)),
                ("Value at best price", sum(1 for ln in lines if ln.verdict == shop.VALUE)),
                ("Median gain from shopping",
                 f"{gains[len(gains) // 2] * 100:+.1f}%" if gains else "—"),
                ("Books compared", len(feed_books) if my_books is None else len(my_books)),
            ]),
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            market_filter = st.selectbox("Market", ["All", "Match result", "Totals"])
        with f2:
            show = st.selectbox("Show", ["Value only", "Value + fair", "Everything"], index=1)
        with f3:
            match_names = sorted({f"{e.get('home_team')} vs {e.get('away_team')}" for e in upcoming})
            match_filter = st.multiselect("Matches", match_names, placeholder="All matches")

        def _keep(ln):
            if market_filter == "Match result" and ln.best.market != "h2h":
                return False
            if market_filter == "Totals" and ln.best.market != "totals":
                return False
            if show == "Value only" and ln.verdict != shop.VALUE:
                return False
            if show == "Value + fair" and ln.verdict == shop.SKIP:
                return False
            return not match_filter or f"{ln.best.home} vs {ln.best.away}" in match_filter

        shown = [ln for ln in lines if _keep(ln)]
        if not shown:
            st.success(
                "Nothing matches that filter. If you asked for value only, that is the "
                "market working — no book is currently paying more than fair for these "
                "selections, so the right stake is zero."
            )
        else:
            rows = []
            for ln in shown:
                row = ln.as_row(bankroll)
                row["Starts"] = fmt_when(row["Starts"])
                rows.append(row)
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True,
                # The decision first; the supporting detail to the right.
                column_order=[
                    "Verdict", "Match", "Market", "Selection", "Best odds", "Book",
                    "Fair odds", "EV %", "Stake", "Stake %", "Next best", "Typical odds",
                    "Shop gain %", "Win %", "Starts", "League", "Books", "Notes",
                ],
                column_config={
                    "Best odds": st.column_config.NumberColumn(format="%.2f"),
                    "Shop gain %": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="Extra return from the best price over the median book's price.",
                    ),
                    "EV %": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="Expected profit per unit staked at the best price, against a "
                             "fair line built without the book offering it.",
                    ),
                    "Stake %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Stake": st.column_config.NumberColumn(format="%.2f"),
                    "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            st.caption(
                "**Best odds** is what a winning bet pays after any exchange commission. "
                "**Typical odds** is the median book. **Fair odds** is the no-margin price; "
                "only a best price *above* it is value. A stake of 0 on a positive-EV row "
                "means the edge is smaller than the books' disagreement about the true "
                "price. Odds move — confirm the price on the book before staking."
            )


# ============================================================== DAILY CARD ===
with tab_card:
    st.subheader("Today's strongest selections")
    st.caption(
        "The best-evidenced head-to-head selections across many competitions, one per "
        "league. Built only from markets where enough books agree that the fair price "
        "means something."
    )

    CARD_SOCCER = {
        "soccer_epl": "EPL", "soccer_germany_bundesliga": "Bundesliga",
        "soccer_spain_la_liga": "La Liga", "soccer_italy_serie_a": "Serie A",
        "soccer_france_ligue_one": "Ligue 1", "soccer_usa_mls": "MLS",
        "soccer_argentina_primera_division": "Argentina Primera",
        "soccer_brazil_campeonato": "Brazil Serie A",
        "soccer_netherlands_eredivisie": "Eredivisie",
        "soccer_portugal_primeira_liga": "Primeira Liga",
        "soccer_mexico_ligamx": "Liga MX", "soccer_efl_champ": "Championship",
        "soccer_turkey_super_league": "Turkey Super Lig",
        "soccer_belgium_first_div": "Belgium First Div",
        "soccer_japan_j_league": "J League", "soccer_spl": "Scottish Premiership",
        "soccer_saudi_arabia_pro_league": "Saudi Pro League",
        "soccer_switzerland_superleague": "Swiss Superleague",
    }
    CARD_BASKET = {
        "basketball_nba": "NBA", "basketball_wnba": "WNBA",
        "basketball_euroleague": "EuroLeague",
    }

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        mode = st.selectbox(
            "Rank selections by", list(daily.MODES),
            index=list(daily.MODES).index(daily.DEFAULT_MODE),
            format_func=lambda m: daily.MODES[m]["label"],
        )
        st.caption(daily.MODES[mode]["help"])
    with c2:
        window = st.selectbox("Time window", [24, 48, 72],
                              format_func=lambda h: f"Next {h}h")
    with c3:
        per_sport = st.number_input("Picks per sport", 1, 10, 5)

    chosen_soccer = st.multiselect(
        "Soccer competitions to scan", list(CARD_SOCCER),
        default=list(CARD_SOCCER)[:10], format_func=lambda k: CARD_SOCCER[k],
    )
    chosen_basket = st.multiselect(
        "Basketball competitions to scan", list(CARD_BASKET),
        default=list(CARD_BASKET), format_func=lambda k: CARD_BASKET[k],
    )

    n_leagues = len(chosen_soccer) + len(chosen_basket)
    card_regions = st.radio(
        "Bookmaker regions for the card", ["eu", "eu,uk", "eu,uk,us"],
        horizontal=True, index=0,
        help="Each extra region multiplies the credit cost. 'eu' alone already "
             "includes Pinnacle and the exchanges, which is what anchors the fair price.",
    )
    cost = n_leagues * len(card_regions.split(","))
    st.caption(
        f"Scanning **{n_leagues} competitions** across **{len(card_regions.split(','))} "
        f"region(s)** costs about **{cost} API credits** per uncached build "
        "(free tier: 500/month, cached 15 min)."
    )

    if st.button("🔥 Build the card", type="primary"):
        st.session_state["card_request"] = (
            tuple(chosen_soccer), tuple(chosen_basket), card_regions, mode, window, per_sport
        )

    if st.session_state.get("card_request"):
        soc_keys, bk_keys, regs, mode_sel, win_sel, n_sel = st.session_state["card_request"]

        def gather(keys, titles):
            out = []
            for key in keys:
                try:
                    found, _ = cached_h2h(key, regs)
                except odds_api.OddsAPIError as exc:
                    st.warning(f"{titles.get(key, key)}: {exc}")
                    continue
                for e in found:
                    e["_league"] = titles.get(key, key)
                    # The feed already carries sport_key, but set it explicitly
                    # so the calibration layer cannot silently fall back to
                    # identity if the feed ever stops sending it.
                    e.setdefault("sport_key", key)
                    e["_sport_key"] = key
                    out.append(e)
            return out

        soccer_events = gather(soc_keys, CARD_SOCCER)
        basket_events = gather(bk_keys, CARD_BASKET)

        cards = [
            ("⚽ Soccer", daily.build_card(soccer_events, "Soccer", n_sel, mode_sel, win_sel)),
            ("🏀 Basketball", daily.build_card(basket_events, "Basketball", n_sel, mode_sel, win_sel)),
        ]

        all_selected = []
        for heading, card in cards:
            st.markdown(f"### {heading}")
            sels = card["selections"]

            if not sels:
                st.warning(
                    f"**No qualifying {card['sport'].lower()} selection in the next "
                    f"{card['window_hours']}h.** {card['events_in_window']} games fell in "
                    f"the window; {card['passed_quality_gate']} of {card['candidates']} "
                    "candidate selections had enough book agreement to price confidently. "
                    "An empty card is a real answer — it is not padded."
                )
                continue

            st.markdown(
                render.stat_tiles([
                    ("Selections", f"{len(sels)}/{card['requested']}"),
                    ("Competitions", len({s.league for s in sels})),
                    ("Avg win prob", f"{sum(s.fair_prob for s in sels) / len(sels) * 100:.0f}%"),
                    ("Avg EV", f"{sum(s.ev for s in sels) / len(sels) * 100:+.2f}%"),
                    ("Games in window", card["events_in_window"]),
                ]),
                unsafe_allow_html=True,
            )
            if card["short_by"]:
                st.info(
                    f"Only **{len(sels)}** of the {card['requested']} requested cleared the "
                    f"evidence bar (needs ≥{daily.MIN_BOOKS} books, ≤"
                    f"{daily.MAX_DISPERSION * 100:.1f}pp disagreement, one pick per league). "
                    "Filling the gap with weaker selections would make the card look "
                    "complete and be worth less."
                )

            rows = []
            for s in sels:
                row = s.as_row()
                row["Starts"] = fmt_when(row["Starts"])
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            for s in sels:
                verdict = (
                    "priced in your favour" if s.ev > 0.01
                    else "priced about fairly" if s.ev > -0.01
                    else "priced against you — you are paying for the safety"
                )
                st.markdown(
                    f"- **{s.selection}** vs {s.opponent} ({s.league}) — "
                    f"**{s.fair_prob * 100:.0f}%** to win "
                    f"(conservatively {s.shrunk_prob * 100:.0f}%), best price **{s.odds}** "
                    f"at {s.book} against a fair {s.fair_odds:.2f}: {verdict}. "
                    f"{s.n_books} books, {s.sigma * 100:.1f}pp disagreement."
                )
            all_selected.extend(sels)
            st.divider()

        if all_selected:
            st.markdown("### 🧮 If you combine them")
            par = daily.parlay_analysis(all_selected)
            st.markdown(
                render.stat_tiles([
                    ("Legs", par["n"]),
                    ("Accumulator odds", f"{par['combined_odds']:.0f}"),
                    ("All legs land", f"{par['all_win_prob'] * 100:.2f}%"),
                    ("Expected winners", f"{par['expected_winners']:.1f}"),
                    ("Parlay EV", f"{par['parlay_ev'] * 100:+.1f}%"),
                ]),
                unsafe_allow_html=True,
            )
            growth_line = (
                f"backing them singly compounds your bankroll about "
                f"**{par['growth_ratio']:.0f}x faster** than the parlay"
                if par.get("growth_ratio") and par["growth_ratio"] > 1
                else "backing them singly is the better-growing option"
            )
            ev_line = (
                f"The accumulator's expected value is **{par['parlay_ev'] * 100:+.1f}%**. "
                "That is *higher* than the legs average individually — expected value "
                "compounds in both directions, so combining genuinely +EV legs does raise "
                "it. That is not a reason to parlay."
                if par["parlay_ev"] > par["singles_avg_ev"]
                else
                f"The accumulator's expected value is **{par['parlay_ev'] * 100:+.1f}%** "
                f"against **{par['singles_avg_ev'] * 100:+.2f}%** for the same legs backed "
                "singly — each leg's margin compounds into the price of the whole bet."
            )
            st.error(
                f"**Back these as {par['n']} separate bets, not one accumulator.** "
                f"The legs sit in different countries and competitions, so they are "
                f"near-independent — and independent probabilities multiply. All "
                f"{par['n']} landing is **{par['all_win_prob'] * 100:.2f}%**; you should "
                f"expect about **{par['expected_winners']:.1f} winners, not {par['n']}**, "
                f"and the parlay returns nothing at all "
                f"**{_pct_near_one(par['lose_everything_prob'])}** of the time.\n\n"
                f"{ev_line}\n\n"
                f"The decisive number is growth, not EV: at each bet's own optimal stake, "
                f"{growth_line}. A parlay throws away the diversification that makes "
                "independent bets compound, which is why accumulators are the most "
                "profitable product a sportsbook sells."
            )

            dist = par["distribution"]
            st.markdown("**Exact distribution of how many legs win** (Poisson-binomial):")
            st.dataframe(
                pd.DataFrame([
                    {"Winners": k, "Probability %": round(p * 100, 2),
                     "At least this many %": round(sum(dist[k:]) * 100, 2)}
                    for k, p in enumerate(dist)
                ]),
                width="stretch", hide_index=True,
            )

            st.markdown("### 💷 Backing them as singles")
            stake_rows = []
            for s in all_selected:
                from src.edge import kelly_fraction
                f = kelly_fraction(s.shrunk_prob, s.odds) * kelly_mult
                stake_rows.append({
                    "Selection": f"{s.selection} ({s.league})",
                    "Odds": s.odds, "EV %": round(s.ev * 100, 2),
                    "Stake %": round(f * 100, 2), "Stake": round(bankroll * f, 2),
                })
            total_f = sum(r["Stake %"] for r in stake_rows) / 100
            st.dataframe(pd.DataFrame(stake_rows), width="stretch", hide_index=True)
            if total_f == 0:
                st.info(
                    "Every stake is zero: at these prices no selection survives the "
                    "uncertainty shrinkage, so Kelly sizes them at nothing. In the "
                    "'most likely to win' mode that is expected — those are the safest "
                    "outcomes, and safety is exactly what the market charges most for."
                )
            else:
                st.caption(
                    f"Total {total_f * 100:.2f}% of bankroll at quarter-Kelly on the "
                    "conservative probability. Sized individually, not as one bet."
                )


# =============================================================== +EV BOARD ===
with tab_ev:
    st.subheader("Selections priced above fair value")
    st.caption(
        "Fair value for each book's price is built from the *other* books — sharp-anchored "
        "where a sharp book is present. An empty board is the normal, correct result: it "
        "means nothing is currently mispriced."
    )
    if not events:
        st.info("Select at least one league in the sidebar to scan.")
    else:
        # One row per selection at its best-paying book, pre-match only.
        opportunities = picks.scan_slate(
            upcoming, min_ev=min_ev, devig_method=devig_method,
            kelly_multiplier=kelly_mult, shrink_k=shrink_k, exposure_cap=exposure_cap,
            best_only=True, books=my_books,
        )
        staked = [o for o in opportunities if (o.stake_fraction or 0) > 0]

        st.markdown(
            render.stat_tiles([
                ("Events scanned", len(upcoming)),
                ("+EV selections", len(opportunities)),
                ("Passing shrinkage", len(staked)),
                ("Total exposure", f"{sum(o.stake_fraction or 0 for o in staked) * 100:.1f}%"),
            ]),
            unsafe_allow_html=True,
        )

        if not opportunities:
            st.success(
                f"No selection currently exceeds {min_ev * 100:.1f}% EV. "
                "That is the market working as expected — do not force a bet."
            )
        else:
            rows = []
            for o in opportunities:
                row = o.as_row()
                row["Starts"] = fmt_when(row["Starts"])
                row["Stake"] = round(bankroll * (o.stake_fraction or 0), 2)
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(
                df, width="stretch", hide_index=True,
                column_config={
                    "EV %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Stake %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Stake": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                "**Edge pp** is the probability-space edge; **EV %** is the return per unit "
                "staked. A bet should look good on both — EV alone flatters longshots. "
                "**Disagreement pp** is how much the books differ; when it exceeds the edge, "
                "the edge is inside the noise."
            )

            with st.expander("Log a bet to the ledger"):
                if staked:
                    labels = {
                        f"{o.home} v {o.away} · {o.selection} "
                        f"{'' if o.point is None else o.point} @ {o.odds} ({o.book_title}) "
                        f"· {o.ev * 100:+.1f}% EV": i
                        for i, o in enumerate(staked)
                    }
                    choice = st.selectbox("Selection", list(labels))
                    pick = staked[labels[choice]]
                    stake_amt = st.number_input(
                        "Stake", min_value=0.0,
                        value=float(round(bankroll * (pick.stake_fraction or 0), 2)),
                    )
                    taken = st.number_input("Price actually taken", min_value=1.01, value=float(pick.odds))
                    if st.button("Log bet"):
                        conn = ledger.connect()
                        ledger.log_bet(
                            conn, event_id=pick.event_id, commence_time=pick.commence_time,
                            league=pick.league, home=pick.home, away=pick.away,
                            market=pick.market, point=pick.point, selection=pick.selection,
                            book=pick.book_title, odds_taken=taken, stake=stake_amt,
                            fair_prob=pick.fair_prob, ev_at_bet=pick.ev,
                            notes=f"anchor={pick.anchor_quality} books={pick.n_books}",
                        )
                        conn.close()
                        st.success("Logged. Record the closing price at kickoff to grade CLV.")
                else:
                    st.caption("Nothing passes the shrinkage filter right now.")

# ================================================================= FIXTURES ===
with tab_fix:
    if not events:
        st.info("Select at least one league in the sidebar.")
    for group in groups:
        group_events = [e for e in events if e.get("_group") == group]
        if not group_events:
            continue
        st.header(group)
        for ev in group_events:
            h2h = picks.summarize_h2h(ev, devig_method)
            total = picks.best_total_bet(ev, devig_method)
            st.markdown(
                render.fixture_card(
                    when=fmt_when(ev.get("commence_time")), league=ev.get("_league"),
                    sport=group, home=ev.get("home_team"), away=ev.get("away_team"),
                    h2h=h2h, total_pick=total, is_live=ev.get("id") in in_play_ids,
                ),
                unsafe_allow_html=True,
            )
            with st.expander(f"Breakdown — {ev.get('home_team')} vs {ev.get('away_team')}"):
                st.write(picks.build_verdict(ev, h2h, total))
                gap = picks.sharp_disagreement(ev)
                if gap:
                    st.markdown(
                        f"**Sharp vs soft:** sharp books price **{gap['selection']}** at "
                        f"{gap['sharp_prob'] * 100:.1f}% against the recreational consensus's "
                        f"{gap['soft_prob'] * 100:.1f}% — a **{gap['gap_pp']:+.1f}pp** gap "
                        f"(sharp: {', '.join(gap['sharp_books'])})."
                    )
                if h2h:
                    st.dataframe(pd.DataFrame(h2h["outcomes"]).T, width="stretch")
                own = [] if ev.get("id") in in_play_ids else picks.best_per_selection(
                    picks.scan_event(ev, min_ev=min_ev, devig_method=devig_method,
                                     kelly_multiplier=kelly_mult, shrink_k=shrink_k,
                                     books=my_books)
                )
                if own:
                    st.markdown("**+EV selections on this match:**")
                    st.dataframe(pd.DataFrame([o.as_row() for o in own]),
                                 width="stretch", hide_index=True)

# =========================================================== ARB & MIDDLES ===
with tab_arb:
    st.subheader("Arbitrage and middles")
    st.caption(
        "Read these sceptically. A genuine arbitrage on a liquid market is rare and small; "
        "a large one almost always means a stale quote or a mismatched line, so this scan "
        "doubles as an integrity check on the odds feed."
    )
    arbs, middles = [], []
    for ev in upcoming:
        found = arbitrage.find_arbitrage(ev, "h2h")
        if found:
            found.update({"match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                          "league": ev.get("_league"), "starts": fmt_when(ev.get("commence_time"))})
            arbs.append(found)
        for m in arbitrage.find_middles(ev)[:2]:
            if m["gap"] >= 0.5 and m["cost_pct"] < 8:
                m.update({"match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                          "league": ev.get("_league")})
                middles.append(m)

    st.markdown("#### Arbitrage")
    if arbs:
        for a in arbs:
            tone = "⚠️ suspicious — verify before staking" if a["suspect"] else "✅ plausible"
            st.markdown(f"**{a['match']}** ({a['league']}, {a['starts']}) — "
                        f"{a['profit_pct']:+.2f}% · booksum {a['booksum']} · {tone}")
            st.dataframe(pd.DataFrame(a["legs"]), width="stretch", hide_index=True)
    else:
        st.caption("No arbitrage across the selected books — the expected result.")

    st.markdown("#### Middles")
    if middles:
        middles.sort(key=lambda m: (-m["gap"], m["cost_pct"]))
        st.dataframe(pd.DataFrame(middles), width="stretch", hide_index=True)
        st.caption("`cost_pct` is what the position costs when the result lands outside the "
                   "window. Negative means it is also an outright arbitrage.")
    else:
        st.caption("No middles worth the spread right now.")

# ================================================================ MODEL LAB ===
with tab_model:
    st.subheader("Dixon-Coles goal model")
    st.caption(
        "Fitted on free historical results from football-data.co.uk. The score matrix prices "
        "1X2, totals, both-teams-to-score and Asian handicaps from one coherent distribution, "
        "so those prices cannot contradict each other the way a book's can."
    )
    from src.datafeeds.football_data import DIVISIONS

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        division = st.selectbox("League", list(DIVISIONS), format_func=lambda d: DIVISIONS[d])
    with col_b:
        n_seasons = st.slider("Seasons of history", 1, 8, 4)
    with col_c:
        xi = st.select_slider(
            "Time decay ξ", options=[0.0, 0.0005, 0.001, 0.0019, 0.003, 0.005], value=0.0019,
            help="Per-day exponential decay. ξ=0.0019 is roughly a one-year half-life.",
        )
    with col_d:
        shot_w = st.select_slider(
            "Shots-on-target weight", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.5,
            help="Fraction of team strength taken from the shots-on-target fit rather "
                 "than the goals fit. 0.5 is the walk-forward optimum: shots are a "
                 "better-sampled measure of control (~9/match vs 2.7), goals carry "
                 "finishing. Set 0 for the classic goals-only Dixon-Coles.",
        )
    years = tuple(range(2026 - n_seasons, 2026))

    if st.button("Fit model"):
        st.session_state["fit_request"] = (division, years, xi, shot_w)

    if st.session_state.get("fit_request"):
        request = st.session_state["fit_request"]
        # A session opened before the shots control existed holds a 3-tuple.
        div_sel, yrs, xi_sel = request[:3]
        shot_sel = request[3] if len(request) > 3 else 0.5
        model, rows = cached_model(div_sel, yrs, xi_sel, shot_sel)
        if model is None:
            st.error(f"Not enough history for {DIVISIONS[div_sel]} in those seasons.")
        else:
            half_life = (0.693 / xi_sel) if xi_sel else None
            st.markdown(render.stat_tiles([
                ("Matches fitted", model.n_matches),
                ("Teams", len(model.teams)),
                ("Home advantage", f"{model.home_advantage:+.3f} log-goals"),
                ("Low-score ρ", f"{model.rho:+.3f}"),
                ("Shots weight", f"{shot_sel:.0%}"),
                ("Decay half-life", f"{half_life:.0f}d" if half_life else "none"),
            ]), unsafe_allow_html=True)
            st.caption(
                f"Home advantage of {model.home_advantage:+.3f} in log-goals means the same "
                f"team scores about {(2.718 ** model.home_advantage - 1) * 100:.0f}% more goals "
                f"at home. ρ={model.rho:+.3f} is the Dixon-Coles correction to low-scoring "
                "draws; a negative ρ means independent Poisson was over-predicting them."
            )

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Team ratings** (net = attack − defence)")
                st.dataframe(pd.DataFrame(model.team_ratings()), width="stretch",
                             hide_index=True, height=420)
                st.caption("Sanity check: this should look roughly like the league table. "
                           "If it does not, the fit is wrong and nothing below is usable.")
            with right:
                st.markdown("**Price a matchup**")
                home = st.selectbox("Home", model.teams, key="mh")
                away = st.selectbox("Away", model.teams,
                                    index=min(1, len(model.teams) - 1), key="ma")
                if home != away:
                    lam, mu = model.expected_goals(home, away)
                    mo = model.match_odds(home, away)
                    st.markdown(f"Expected goals: **{home} {lam:.2f} – {mu:.2f} {away}**")
                    st.dataframe(pd.DataFrame([{
                        "Market": "1X2",
                        f"{home}": f"{mo['home'] * 100:.1f}% ({1 / mo['home']:.2f})",
                        "Draw": f"{mo['draw'] * 100:.1f}% ({1 / mo['draw']:.2f})",
                        f"{away}": f"{mo['away'] * 100:.1f}% ({1 / mo['away']:.2f})",
                    }]), width="stretch", hide_index=True)

                    line = st.select_slider("Totals line",
                                            options=[0.5, 1.5, 2.5, 3.5, 4.5], value=2.5)
                    tot, bt = model.totals(home, away, line), model.btts(home, away)
                    st.dataframe(pd.DataFrame([
                        {"Market": f"Over/Under {line}",
                         "Yes/Over": f"{tot['Over'] * 100:.1f}% ({1 / tot['Over']:.2f})",
                         "No/Under": f"{tot['Under'] * 100:.1f}% ({1 / tot['Under']:.2f})"},
                        {"Market": "Both teams to score",
                         "Yes/Over": f"{bt['Yes'] * 100:.1f}% ({1 / bt['Yes']:.2f})",
                         "No/Under": f"{bt['No'] * 100:.1f}% ({1 / bt['No']:.2f})"},
                    ]), width="stretch", hide_index=True)
                    st.markdown("**Most likely scorelines**")
                    st.dataframe(
                        pd.DataFrame(model.correct_score(home, away, 8),
                                     columns=["Score", "Probability"]),
                        width="stretch", hide_index=True,
                    )

            st.divider()
            st.markdown("#### Walk-forward backtest against closing prices")
            st.caption(
                "Refits before every prediction on prior matches only, then scores against "
                "the de-vigged Pinnacle **closing** line. This is the honest test, and it is "
                "designed to be failable."
            )
            if st.button("Run backtest (slow — refits hundreds of times)"):
                from src.backtest import simulate_bankroll, walk_forward

                with st.spinner("Walking forward..."):
                    res = walk_forward(rows, min_train=380, step=10, xi=xi_sel,
                                       devig_method=devig_method, shot_weight=shot_sel)
                if not res.get("n"):
                    st.error(res.get("error", "no evaluable matches"))
                else:
                    st.dataframe(pd.DataFrame([
                        {"Forecaster": "Dixon-Coles alone", "Log loss": res["model"]["log_loss"],
                         "Brier": res["model"]["brier"]},
                        {"Forecaster": "Market (Pinnacle close, de-vigged)",
                         "Log loss": res["market"]["log_loss"], "Brier": res["market"]["brier"]},
                        {"Forecaster": f"Blend (w={res['blend']['weight']})",
                         "Log loss": res["blend"]["log_loss"], "Brier": res["blend"]["brier"]},
                    ]), width="stretch", hide_index=True)
                    w = res["blend"]["weight"]
                    if w == 0:
                        st.warning(
                            f"Optimal blend weight is **0** over {res['n']} matches: the model "
                            "adds nothing the closing line has not already priced. That is the "
                            "expected result for a liquid 1X2 market, and it is the model "
                            "telling you not to bet it — which is worth more than a fitted "
                            "number that quietly loses money."
                        )
                    else:
                        st.success(
                            f"Optimal blend weight **{w}** improves log loss by "
                            f"{res['blend']['improvement_vs_market']:.5f} over the market alone."
                        )
                    sims = [simulate_bankroll(res, source=s, price_key=p)
                            for s in ("market", "blend") for p in ("psc", "maxc")]
                    sims = [s for s in sims if s.get("n_bets")]
                    if sims:
                        st.markdown("**Bankroll simulation** (quarter-Kelly)")
                        st.dataframe(pd.DataFrame([{
                            "Signal": s["source"],
                            "Price": "Pinnacle" if s["price_key"] == "psc" else "best available",
                            "Bets": s["n_bets"], "ROI %": round(s["roi"] * 100, 2),
                            "t-stat": round(s["roi_tstat"], 2),
                            "Hit %": round(s["hit_rate"] * 100, 1),
                            "Max DD %": round(s["max_drawdown"] * 100, 1),
                            "Bets to significance": s.get("bets_for_significance"),
                        } for s in sims]), width="stretch", hide_index=True)
                        st.caption("A t-stat below 2 means the ROI is not distinguishable from "
                                   "luck, however good it looks.")

# ============================================================= LEDGER & CLV ===
with tab_ledger:
    st.subheader("Bet ledger and closing line value")
    st.caption(
        "Settled profit is a terrible feedback signal over any period you will live through — "
        "at a realistic 2-3% edge it takes thousands of bets to separate skill from luck. "
        "CLV is observable within hours of each bet and is what sportsbooks themselves use to "
        "spot winning accounts. Track it, and let profit catch up."
    )
    conn = ledger.connect()
    perf = ledger.performance(conn)

    if not perf.get("n"):
        st.info("No bets logged yet. Log one from the +EV Board.")
    else:
        tiles = [("Bets", perf["n"]), ("Open", perf["n_open"]), ("Settled", perf["n_settled"])]
        if "avg_clv" in perf:
            tiles += [("Avg CLV", f"{perf['avg_clv'] * 100:+.2f}%"),
                      ("CLV hit rate", f"{perf['clv_hit_rate'] * 100:.0f}%")]
        if "roi" in perf:
            tiles += [("ROI", f"{perf['roi'] * 100:+.2f}%"), ("P&L", f"{perf['pnl']:+.2f}")]
        st.markdown(render.stat_tiles(tiles), unsafe_allow_html=True)

        if "clv_verdict" in perf:
            verdict = perf["clv_verdict"]
            (st.success if "strong" in verdict or "positive edge" in verdict
             else st.error if "no edge" in verdict else st.info)(f"**CLV verdict:** {verdict}")
        if perf.get("bets_for_significance"):
            st.caption(
                f"At the ROI observed so far it would take roughly "
                f"**{perf['bets_for_significance']:,} settled bets** for profit alone to "
                "prove an edge at two standard errors. This is exactly why CLV is the "
                "metric to watch."
            )

        st.markdown("#### Open bets — record the closing price at kickoff")
        for bet in ledger.open_bets(conn):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(
                f"**{bet['home']} v {bet['away']}** · {bet['selection']} "
                f"{'' if bet['point'] is None else bet['point']} @ {bet['odds_taken']} "
                f"({bet['book']}) · stake {bet['stake']}"
            )
            close = c2.number_input("Close", min_value=0.0, value=float(bet["closing_odds"] or 0.0),
                                    key=f"cl{bet['id']}", label_visibility="collapsed")
            result = c3.selectbox("Result", ["—", "win", "loss", "push", "void"],
                                  key=f"rs{bet['id']}", label_visibility="collapsed")
            if c3.button("Save", key=f"sv{bet['id']}"):
                if close > 1:
                    ledger.record_close(conn, bet["id"], close)
                if result != "—":
                    ledger.settle(conn, bet["id"], result)
                st.rerun()

        st.markdown("#### All bets")
        st.dataframe(pd.DataFrame(ledger.all_bets(conn)), width="stretch", hide_index=True)
        st.download_button("⬇️ Export ledger CSV", ledger.to_csv(conn), "bets.csv", "text/csv")
        st.caption(
            "On Streamlit Community Cloud the filesystem resets on redeploy — export "
            "anything you want to keep."
        )

    uploaded = st.file_uploader("Restore a ledger from CSV", type="csv")
    if uploaded is not None and st.button("Import"):
        count = ledger.from_csv(conn, uploaded.getvalue().decode("utf-8"))
        st.success(f"Imported {count} bets.")
        st.rerun()
    conn.close()

# ============================================================== METHODOLOGY ===
with tab_docs:
    st.markdown(render.METHODOLOGY_MD)

    st.markdown("---")
    st.markdown("### How accurate are these probabilities?")
    st.markdown(
        "Measured, not asserted. Every number below comes from **66,272 matches** across "
        "18 European leagues (2015/16–2024/25), taking Pinnacle's *closing* 1X2 prices, "
        "de-vigging each one, and comparing the result to what actually happened. "
        "The calibration is refit with `python -m scripts.fit_calibration`."
    )
    st.markdown(render.stat_tiles([
        ("Matches scored", "66,272"),
        ("Leagues fitted", len(calibration.LEAGUE_EXPONENTS)),
        ("Mean |bias| p≥0.6, before", "2.55pp"),
        ("Mean |bias| p≥0.6, after", "1.65pp"),
    ]), unsafe_allow_html=True)
    st.markdown(
        "**What was wrong.** The de-vigged sharp close is excellent in the middle of the "
        "probability scale and systematically *too low* at the top of it — the "
        "favourite–longshot bias. Across every selection priced at 60% or better, the "
        "market said 70.6% and reality delivered 72.2%. That gap is about four standard "
        "errors, it shows up in Pinnacle, Bet365 and the cross-book average alike, and it "
        "sits exactly in the band this card selects from."
    )
    st.markdown(
        "**What was done.** One parameter per league: the fair-probability vector is raised "
        "to a power and renormalised. The all-league fit is "
        f"**{calibration.SOCCER_EXPONENT:.4f}**, and fitting Over/Under 2.5 separately gives "
        "1.0409 — two different markets with different numbers of outcomes, agreeing to four "
        "decimal places, which is the main reason to believe the parameter rather than the story."
    )
    st.markdown(
        "**One exponent was the wrong shape.** A single global constant helps 10 of 18 leagues "
        "and *hurts* the other 8 — the Bundesliga close slightly **over**states favourites, so "
        "sharpening it further made it worse. Fitting each league separately overfits at "
        "2,000–5,500 matches apiece. So each league's exponent is shrunk halfway toward the "
        f"global one (K={calibration.SHRINKAGE_K}, itself chosen out-of-sample). That cuts mean "
        "absolute bias across leagues from 2.55pp to 1.65pp, where the global constant only "
        "reached 2.09pp. Soccer only; other sports pass through untouched, and an unfitted "
        "soccer league falls back to the global number."
    )
    from src.datafeeds.football_data import DIVISIONS as DIVISION_NAMES

    exps = sorted(calibration.LEAGUE_EXPONENTS.items(), key=lambda kv: kv[1])
    st.dataframe(
        pd.DataFrame(
            [{"League": DIVISION_NAMES.get(code, code), "Exponent": round(val, 4),
              "Direction": "sharpen toward favourites" if val > 1 else "flatten toward longshots"}
             for code, val in exps]
        ),
        width="stretch", hide_index=True, height=360,
        column_config={
            "League": st.column_config.TextColumn(width="medium"),
            "Exponent": st.column_config.NumberColumn(format="%.4f", width="small"),
            "Direction": st.column_config.TextColumn(width="medium"),
        },
    )
    st.caption(
        "Sorted by exponent. Below 1.0 means that league's closing line already "
        "overstates favourites, so the correction runs the other way."
    )
    st.warning(
        "**What this is not.** As a forecast of the average match it is worth almost nothing "
        "— 0.00011 nats of log loss, better in 7 of 9 seasons. It does not make the model "
        "smarter than the market and it is not an edge. It makes the number printed next to "
        "a favourite honest, which matters because that is the number you size a bet from.",
        icon="⚠️",
    )
    st.markdown(
        "**The model, separately.** Mixing shots on target into the Dixon-Coles fit at a "
        "50/50 weight is a genuinely better model — 1X2 log loss 1.0035 → 0.9993, "
        "Over/Under 0.6876 → 0.6814. It still loses to the market on both, and the "
        "honestly-fitted blend weight is zero on 1X2 and negative on totals. Better "
        "ratings and scoreline distributions; no betting edge. The Model Lab uses it."
    )

if quota_left is not None:
    st.caption(f"Odds API quota remaining this month: {quota_left}")
