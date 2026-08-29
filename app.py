"""Quantitative betting research dashboard.

Six tabs, in the order you should actually use them:

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

from src import arbitrage, daily, form, ledger, odds_api, picks, render  # noqa: E402
from src.devig import METHODS  # noqa: E402
from src.edge import DEFAULT_EXPOSURE_CAP, DEFAULT_KELLY_FRACTION, DEFAULT_SHRINK_K  # noqa: E402

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
def cached_model(division: str, years: tuple, xi: float):
    from src.datafeeds.football_data import to_matches
    from src.models.dixon_coles import DixonColesModel

    rows = cached_history(division, years)
    if len(rows) < 100:
        return None, rows
    return DixonColesModel(max_goals=10).fit(to_matches(rows), xi=xi), rows


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

selected_leagues = {}
for group in groups:
    available = sports_by_group.get(group, [])
    titles = [x["title"] for x in available]
    selected_leagues[group] = st.sidebar.multiselect(
        f"{group} leagues", titles, default=titles[:3]
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

tab_card, tab_ev, tab_fix, tab_arb, tab_model, tab_ledger, tab_docs = st.tabs(
    ["🔥 Daily Card", "🎯 +EV Board", "🗓️ Fixtures", "⚖️ Arb & Middles", "🧪 Model Lab",
     "📒 Ledger & CLV", "📘 Methodology"]
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
        opportunities = picks.scan_slate(
            events, min_ev=min_ev, devig_method=devig_method,
            kelly_multiplier=kelly_mult, shrink_k=shrink_k, exposure_cap=exposure_cap,
        )
        staked = [o for o in opportunities if (o.stake_fraction or 0) > 0]

        st.markdown(
            render.stat_tiles([
                ("Events scanned", len(events)),
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
                    h2h=h2h, total_pick=total,
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
                own = picks.scan_event(ev, min_ev=min_ev, devig_method=devig_method,
                                       kelly_multiplier=kelly_mult, shrink_k=shrink_k)
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
    for ev in events:
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

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        division = st.selectbox("League", list(DIVISIONS), format_func=lambda d: DIVISIONS[d])
    with col_b:
        n_seasons = st.slider("Seasons of history", 1, 8, 4)
    with col_c:
        xi = st.select_slider(
            "Time decay ξ", options=[0.0, 0.0005, 0.001, 0.0019, 0.003, 0.005], value=0.0019,
            help="Per-day exponential decay. ξ=0.0019 is roughly a one-year half-life.",
        )
    years = tuple(range(2026 - n_seasons, 2026))

    if st.button("Fit model"):
        st.session_state["fit_request"] = (division, years, xi)

    if st.session_state.get("fit_request"):
        div_sel, yrs, xi_sel = st.session_state["fit_request"]
        model, rows = cached_model(div_sel, yrs, xi_sel)
        if model is None:
            st.error(f"Not enough history for {DIVISIONS[div_sel]} in those seasons.")
        else:
            half_life = (0.693 / xi_sel) if xi_sel else None
            st.markdown(render.stat_tiles([
                ("Matches fitted", model.n_matches),
                ("Teams", len(model.teams)),
                ("Home advantage", f"{model.home_advantage:+.3f} log-goals"),
                ("Low-score ρ", f"{model.rho:+.3f}"),
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
                                       devig_method=devig_method)
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

if quota_left is not None:
    st.caption(f"Odds API quota remaining this month: {quota_left}")
