"""Betting research dashboard — live scores + upcoming fixtures across soccer,
basketball, baseball, cricket, with de-vigged market-implied probabilities and
a defensible recommended pick per game (never the extreme end of an O/U board).

Run: ./start.sh   (or `streamlit run app.py` once .env has ODDS_API_KEY set)
"""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import form, odds_api, picks, render  # noqa: E402  (must follow load_dotenv())
from src.probability import fmt_stars  # noqa: E402

st.set_page_config(page_title="Betting Research Dashboard", layout="wide", page_icon="📊")
st.markdown(render.inject_css(), unsafe_allow_html=True)

st.title("📊 Betting Research Dashboard")
st.caption(
    "Live scores + upcoming fixtures across soccer, basketball, baseball, and cricket, "
    "with market-implied probabilities and a defensible recommended pick per game."
)

with st.expander("⚠️ Read before using", expanded=False):
    st.markdown(
        "- Odds come from **The Odds API**, a licensed aggregator. Bookmaker coverage "
        "depends on region: FanDuel / DraftKings / BetMGM (US), William Hill / Unibet / "
        "Ladbrokes / Pinnacle (UK/EU). **bet365, SportyBet, and Betway are not available "
        "through any legitimate API** — this dashboard does not scrape sportsbooks. "
        "Compare the recommended selection/line against your actual book before staking.\n"
        "- \"Fair probability\" is the **de-vigged market-implied probability** (bookmaker "
        "margin mathematically removed) — not a proprietary prediction model. It reflects "
        "what the market collectively prices, generally the sharpest available estimate.\n"
        "- The O/U pick is always the **balance-point line** (closest to a fair 50/50), "
        "never the cheap near-certain end or the juicy-looking extreme end of the board.\n"
        "- Confidence stars reflect **how many independent books quote and agree on the "
        "line**, not certainty of the outcome. Nothing here is guaranteed. Bet responsibly."
    )

try:
    odds_api.check_api_key()
except odds_api.OddsAPIError as e:
    st.error(str(e))
    st.stop()

@st.cache_data(ttl=1800, show_spinner="Checking in-season leagues...")
def cached_sports_by_group():
    return odds_api.in_season_sports_by_group()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_odds(sport_key: str, regions_arg: str):
    return odds_api.get_odds(sport_key, regions=regions_arg)


@st.cache_data(ttl=300, show_spinner=False)
def cached_scores(sport_key: str):
    return odds_api.get_scores(sport_key, days_from=1)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_form(team_name: str):
    return form.get_form(team_name)


try:
    sports_by_group = cached_sports_by_group()
except odds_api.OddsAPIError as e:
    st.error(str(e))
    st.stop()

# --- Sidebar ---
st.sidebar.header("Filters")
st.sidebar.caption(
    "⚠️ Each league you pick below spends API quota on every uncached refresh "
    "(free tier: 500 credits/month). Start small."
)
groups = st.sidebar.multiselect(
    "Sports", options=list(odds_api.SPORT_GROUPS), default=["Soccer"]
)
regions = st.sidebar.multiselect(
    "Bookmaker regions",
    options=["eu", "uk", "us", "au"],
    default=["eu"],
    help="More regions = more books compared, but costs more API quota per refresh.",
)
show_live = st.sidebar.checkbox("Show live matches", value=True)
show_upcoming = st.sidebar.checkbox("Show upcoming fixtures", value=True)
fetch_form = st.sidebar.checkbox("Include recent-form lookup (slower, best-effort)", value=False)
if st.sidebar.button("🔄 Force refresh (clears cache)"):
    st.cache_data.clear()

regions_str = ",".join(regions) if regions else "eu"

selected_leagues_by_group = {}
for group in groups:
    available = sports_by_group.get(group, [])
    titles = [l["title"] for l in available]
    default_titles = titles[:5]
    chosen = st.sidebar.multiselect(f"{group} leagues", options=titles, default=default_titles)
    selected_leagues_by_group[group] = chosen

now = datetime.now(timezone.utc)


def parse_iso(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


for group in groups:
    chosen_titles = set(selected_leagues_by_group.get(group, []))
    leagues = [l for l in sports_by_group.get(group, []) if l["title"] in chosen_titles]
    st.header(group)
    if not sports_by_group.get(group):
        st.info(f"No {group.lower()} leagues currently in-season/tracked by this API.")
        continue
    if not leagues:
        st.caption(f"No {group.lower()} leagues selected — pick some in the sidebar.")
        continue

    all_events = []
    live_rows = []
    quota_left = None

    for league in leagues:
        sport_key = league["key"]
        try:
            events, quota_left = cached_odds(sport_key, regions_str)
        except odds_api.OddsAPIError as e:
            st.warning(f"{league['title']}: {e}")
            continue
        for ev in events:
            ev["_league"] = league["title"]
            all_events.append(ev)

        if show_live:
            try:
                scores, sc_quota = cached_scores(sport_key)
                quota_left = sc_quota or quota_left
            except odds_api.OddsAPIError:
                scores = []
            for s in scores:
                if s.get("completed"):
                    continue
                ct = parse_iso(s.get("commence_time", ""))
                if ct is None or ct > now:
                    continue  # not started yet -> not live
                score_list = s.get("scores") or []
                score_txt = (
                    " - ".join(f"{sc['name']} {sc['score']}" for sc in score_list)
                    if score_list
                    else "Score not yet reported"
                )
                live_rows.append(
                    {
                        "League": league["title"],
                        "Home": s.get("home_team"),
                        "Away": s.get("away_team"),
                        "Score": score_txt,
                        "Last update": s.get("last_update", ""),
                    }
                )

    st.markdown(
        render.stat_tiles(
            [("Live now", len(live_rows) if show_live else "—"),
             ("Upcoming tracked", len([e for e in all_events if e.get("commence_time")])),
             ("Leagues tracked", len(leagues))]
        ),
        unsafe_allow_html=True,
    )

    if show_live:
        st.subheader("🔴 Live")
        if live_rows:
            for lr in live_rows:
                st.markdown(
                    render.fixture_card(
                        when="In progress",
                        league=lr["League"],
                        sport=group,
                        home=lr["Home"],
                        away=lr["Away"],
                        h2h=None,
                        total_pick=None,
                        is_live=True,
                        live_score=lr["Score"],
                    ),
                    unsafe_allow_html=True,
                )
            with st.expander("Table view"):
                st.dataframe(pd.DataFrame(live_rows), width="stretch", hide_index=True)
        else:
            st.caption("No live games right now.")

    if show_upcoming:
        st.subheader("🗓️ Upcoming")
        upcoming = [e for e in all_events if e.get("commence_time")]
        upcoming.sort(key=lambda e: e["commence_time"])

        if not upcoming:
            st.caption("No upcoming fixtures with quoted odds right now.")
        else:
            rows = []
            for ev in upcoming:
                h2h = picks.summarize_h2h(ev)
                total_pick = picks.pick_total_line(ev)
                verdict = picks.build_verdict(ev, h2h, total_pick)

                home, away = ev.get("home_team"), ev.get("away_team")
                ct = parse_iso(ev.get("commence_time", ""))
                when = ct.astimezone().strftime("%a %d %b, %H:%M") if ct else ev.get("commence_time", "?")

                fav_name, fav_pct, fav_odds, fav_book, fav_stars = "N/A", "N/A", "N/A", "N/A", 0
                if h2h and h2h["outcomes"]:
                    fav_name, fav_data = max(
                        h2h["outcomes"].items(), key=lambda kv: (kv[1]["fair_prob"] or 0)
                    )
                    fav_pct = f"{round((fav_data['fair_prob'] or 0) * 100)}%"
                    fav_odds = fav_data["best_odds"]
                    fav_book = fav_data["best_book"]
                    fav_stars = h2h["stars"]

                ou_txt, ou_odds, ou_book, ou_stars = "N/A", "N/A", "N/A", 0
                if total_pick:
                    side = total_pick["side"]
                    pct = round(
                        (total_pick["fair_over"] if side == "Over" else total_pick["fair_under"]) * 100
                    )
                    ou_txt = f"{side} {total_pick['point']} (~{pct}%)"
                    ou_odds = total_pick["best_over_odds"] if side == "Over" else total_pick["best_under_odds"]
                    ou_book = total_pick["best_over_book"] if side == "Over" else total_pick["best_under_book"]
                    ou_stars = total_pick["stars"]

                row = {
                    "Date": when,
                    "League": ev.get("_league"),
                    "Home": home,
                    "Away": away,
                    "Favorite": fav_name,
                    "Win %": fav_pct,
                    "Moneyline pick": f"{fav_odds} @ {fav_book}" if fav_odds != "N/A" else "N/A",
                    "1X2 confidence": fmt_stars(fav_stars) if fav_stars else "N/A",
                    "O/U pick": ou_txt,
                    "O/U odds": f"{ou_odds} @ {ou_book}" if ou_odds != "N/A" else "N/A",
                    "O/U confidence": fmt_stars(ou_stars) if ou_stars else "N/A",
                }
                if fetch_form:
                    hf = cached_form(home)
                    af = cached_form(away)
                    row["Home form"] = f"{hf['results']} ({hf['score']}%)" if hf else "N/A"
                    row["Away form"] = f"{af['results']} ({af['score']}%)" if af else "N/A"
                rows.append(row)

                st.markdown(
                    render.fixture_card(
                        when=when,
                        league=ev.get("_league"),
                        sport=group,
                        home=home,
                        away=away,
                        h2h=h2h,
                        total_pick=total_pick,
                    ),
                    unsafe_allow_html=True,
                )
                with st.expander(f"Full breakdown — {home} vs {away}"):
                    st.write(verdict)
                    if fetch_form:
                        st.caption(f"Recent form — {row.get('Home form', 'N/A')} vs {row.get('Away form', 'N/A')}")
                    if h2h:
                        st.markdown("**Moneyline / 1X2 — all outcomes:**")
                        st.table(pd.DataFrame(h2h["outcomes"]).T)
                    if total_pick:
                        st.markdown(
                            f"**Totals — lines considered:** {total_pick['all_lines_considered']} "
                            f"→ chosen balance-point line: **{total_pick['point']} {total_pick['side']}**"
                        )

            with st.expander("Table view (all fixtures)"):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if quota_left is not None:
        st.caption(f"Odds API quota remaining this month: {quota_left}")

    st.divider()
