"""Private, paper-first dashboard with explicit API requests only."""
from __future__ import annotations

import hmac
import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from src import ledger, paper, service
from src.tracking import Store, profit_summary


def setting(name, default=""):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get(name, default)


def authenticate():
    password = str(setting("DASHBOARD_PASSWORD"))
    if not password:
        st.warning("Private dashboard is locked. Set DASHBOARD_PASSWORD in .env locally or in hosted Secrets.")
        st.caption("No API requests or personal journal reads occur while locked.")
        st.stop()
    if st.session_state.get("authenticated"):
        if st.sidebar.button("Sign out"):
            st.session_state.clear()
            st.rerun()
        return
    with st.form("login"):
        entered = st.text_input("Dashboard password", type="password")
        submit = st.form_submit_button("Open dashboard")
    if submit:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Password did not match.")
    st.stop()


def table(records):
    fields = ("id", "kind", "assessed_at", "placed_at", "home", "away", "book", "selection",
              "odds", "stake", "fair_prob", "conservative_ev", "result", "pnl", "clv")
    return pd.DataFrame([{k: r.get(k) for k in fields} for r in records])


def account(store):
    st.subheader("Your accounts and bankroll")
    settings = store.settings()
    with st.form("account"):
        bankroll = st.number_input("Actual betting bankroll (GHS)", min_value=0.0,
                                   value=float(settings["bankroll"]), step=1.0)
        minimum = st.number_input("Bookmaker minimum stake (GHS)", min_value=0.01,
                                  value=float(settings["minimum_stake"]), step=0.1)
        books = st.multiselect("Books you can actually use", paper.BOOKS, default=settings["books"])
        if st.form_submit_button("Save account settings"):
            store.save_settings({"bankroll": bankroll, "minimum_stake": minimum, "books": books})
            st.success("Saved. Recommended cash stake remains GHS 0.")
    st.caption("A zero bankroll means unknown. The app never assumes a GHS 1,000 balance.")


def paper_board(store):
    st.subheader("Check your own price")
    st.write("Enter the decimal price you can take at your own book. The feed is a benchmark. "
             "Every submitted check is saved; qualifying prices create a frozen 1-unit paper single.")
    sport = st.selectbox("Competition", list(paper.LEAGUES), format_func=paper.LEAGUES.get)
    st.caption("One EU region, match-result market only. A new odds fetch reserves 1 credit; cached for 15 minutes.")
    if st.button("Load / refresh benchmark odds"):
        try:
            events, remaining, cached = service.fetch(store, sport)
            st.success(f"{'Cached' if cached else 'Fetched'} benchmark. Provider credits remaining: {remaining or 'unknown'}.")
        except Exception:
            st.error("Could not load odds. Check API configuration, network and quota. Credentials are hidden.")
    cache = store.get(f"cache:odds:{sport}")
    if not cache:
        st.info("No benchmark loaded. Nothing is fetched until you press the button.")
        return
    now = datetime.now(UTC)
    events = [e for e in cache["events"] if paper.utc(e.get("commence_time"))
              and now < paper.utc(e["commence_time"]) <= now + timedelta(hours=24)]
    st.caption(f"Benchmark fetched {cache['fetched_at']}. Dates and kickoff times are UTC (Ghana time).")
    if not events:
        st.info("No matches within 24 hours in this snapshot. No bet today.")
        return
    event = st.selectbox("Match", events, format_func=lambda e:
                         f"{e['home_team']} vs {e['away_team']} · {e['commence_time']}")
    names = [event["home_team"], event["away_team"]]
    if sport.startswith("soccer_"):
        names.insert(1, "Draw")
    books = store.settings()["books"]
    if not books:
        st.info("Save at least one usable bookmaker in Account first.")
        return
    with st.form("price"):
        book = st.selectbox("Your bookmaker", books)
        selection = st.selectbox("Selection", names)
        odds = st.number_input("Decimal odds at your book", min_value=1.01, value=2.0, step=0.01)
        observed_at = st.text_input("Time you checked this price (UTC ISO date)",
                                   value=now.isoformat(timespec="seconds"))
        checked = st.checkbox("I checked the actual price and market rules, confirmed relevant lineups, "
                              "and excluded low-information matches / uncertain motivation.")
        save = st.form_submit_button("Save price and paper assessment")
    if save:
        assessment = paper.assess(event, book, selection, odds, observed_at, checked)
        paper_id = store.save_assessment(assessment)
        st.session_state["last_assessment"] = assessment
        st.success("Assessment saved" + ("; paper single recorded (or already frozen)." if paper_id else "."))
    a = st.session_state.get("last_assessment")
    if a:
        st.write(f"Last saved assessment: {a['selection']} at {a['odds']:.2f}, {a['book']}.")
        if a["ev"] is not None:
            st.write(f"Estimated EV {a['ev']:+.2%}; conservative estimate {a['conservative_ev']:+.2%}. "
                     f"Reference probability {a['fair_prob']:.1%}, from {a['n_books']} fresh books.")
        if a["eligible"]:
            st.info("Paper candidate only. This estimate has no demonstrated profitable track record. Cash stake: GHS 0.")
        else:
            st.warning("No paper bet: " + " ".join(a["reasons"]))
    assessments = store.rows("assessment")
    if assessments:
        with st.expander("Saved bookmaker prices and rejected checks"):
            st.dataframe(pd.DataFrame([{k: a.get(k) for k in (
                "assessed_at", "book", "selection", "odds", "eligible", "reasons")}
                for a in assessments]), hide_index=True)


def results(store):
    st.subheader("Paper evidence and actual costs")
    records = store.rows("record")
    ev = paper.evidence(records)
    st.info(ev["verdict"])
    st.write(f"Unique paper events: {ev['events']}; with near-close benchmark: {ev['events_with_close']}; "
             f"coverage {ev['coverage']:.0%}; observation span {ev['days']} days.")
    if ev["mean_clv"] is not None:
        st.write(f"Event-averaged de-vigged benchmark CLV: {ev['mean_clv']:+.2%}. "
                 f"Descriptive t-statistic: {ev['tstat']:.2f}" if ev["tstat"] is not None
                 else f"Mean benchmark CLV {ev['mean_clv']:+.2%}; variance/sample is insufficient for a t-statistic.")
    st.caption("Screen fixed in advance: 300 distinct events, 60 days, ≥90% close coverage, "
               "positive mean CLV and t>2. This is a research screen, not proof or an automatic unlock. "
               "Bookmakers and matches can be dependent; missing closes can bias the sample.")
    for kind, label in (("paper", "Paper singles (units)"), ("actual", "Actual bets (GHS)")):
        summary = profit_summary(records, kind)
        st.markdown(f"**{label}**")
        st.write(f"{summary['count']} recorded · {summary['open']} open · "
                 f"{summary['settled']} settled · profit/loss {summary['pnl']:+.2f}")
        if summary["roi"] is not None:
            st.write(f"Settled turnover {summary['turnover']:.2f}; ROI {summary['roi']:+.2%}.")
    if st.button("Fetch completed results and settle matching records"):
        from src.service import poll

        report = poll(store)
        if report["errors"]:
            st.warning("Some requests failed: " + "; ".join(report["errors"]))
        else:
            st.success("Result sync finished. Refresh this view to see updated totals.")
    st.caption("Scores cover at most the past 3 days. Older, postponed and ambiguous records require manual settlement. "
               "The scheduled worker can capture a benchmark in the final 5 minutes before kickoff; "
               "missed closes remain missing.")
    if records:
        st.dataframe(table(records), hide_index=True)
        record = st.selectbox("Record to settle manually", records,
                              format_func=lambda r: f"{r['kind']} · {r['selection']} · {r['id'][-10:]}")
        with st.form("settle"):
            result = st.selectbox("Verified result", ["win", "loss", "push", "void"])
            verified = st.checkbox("I verified this result and the bookmaker's settlement rules.")
            if st.form_submit_button("Save verified result"):
                if verified:
                    store.settle(record["id"], result)
                    st.rerun()
                else:
                    st.warning("Verify the result before saving.")


CSV_TEMPLATE = "bet_id,placed_at,book,selection,odds,stake,result,event_id,sport_key,commence_time,home,away,market,notes\n"


def history(store):
    st.subheader("Reconstruct actual bet history")
    st.write("Import the bets you actually placed. The app does not invent results for past suggested tickets. "
             "Use one row per actual single or whole accumulator; use market=accumulator for a combined slip.")
    st.download_button("Download history CSV template", CSV_TEMPLATE, "bet-history-template.csv", "text/csv")
    upload = st.file_uploader("Actual history CSV (also accepts the old ledger export)", type="csv")
    if upload and st.button("Import actual history"):
        try:
            count = store.import_actual_csv(upload.getvalue().decode("utf-8-sig"))
            st.success(f"Imported {count} new actual bets. Re-imports do not duplicate matching IDs.")
        except (ValueError, UnicodeError) as exc:
            st.error(str(exc))
    with st.expander("Log an actual bet manually"):
        with st.form("actual"):
            book = st.selectbox("Book", paper.BOOKS)
            selection = st.text_input("Selection / whole slip description")
            placed = st.text_input("Placed at (UTC ISO date)", datetime.now(UTC).isoformat(timespec="seconds"))
            odds = st.number_input("Actual decimal odds", min_value=1.01, value=2.0)
            stake = st.number_input("Actual stake (GHS)", min_value=0.01, value=1.0)
            result = st.selectbox("Result", ["open", "win", "loss", "push", "void"])
            if st.form_submit_button("Log actual bet"):
                try:
                    key = store.record_actual({"book": book, "selection": selection, "placed_at": placed,
                                               "odds": odds, "stake": stake, "market": "manual",
                                               "result": None})
                    if result != "open":
                        store.settle(key, result)
                    st.success("Actual bet logged separately from paper results.")
                except ValueError as exc:
                    st.error(str(exc))
    if os.path.isfile("bets.db") and st.button("Import existing local bets.db (read only)"):
        import sqlite3

        conn = sqlite3.connect("file:bets.db?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            count = store.import_actual_csv(ledger.to_csv(conn))
            st.success(f"Imported {count} existing bets; original ledger unchanged.")
        finally:
            conn.close()
    st.markdown("**Private backup**")
    st.download_button("Export full journal including price snapshots", store.backup(),
                       "private-research-backup.json", "application/json")
    restore = st.file_uploader("Restore journal backup", type="json")
    if restore and st.button("Restore missing records"):
        try:
            count = store.restore(restore.getvalue().decode("utf-8"))
            st.success(f"Restored {count} missing rows. Existing rows preserved.")
        except (ValueError, KeyError, TypeError, UnicodeError):
            st.error("Invalid backup. No valid backup rows were imported.")


def tools():
    st.subheader("Price, promotion and minimum-stake checks")
    st.caption("Scenario calculations only. Probabilities are assumptions, not predictions.")
    p = st.number_input("Assumed win probability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    odds = st.number_input("Offered decimal odds", min_value=1.01, value=2.0, key="calc_odds")
    st.write(f"Cash-funded bet expected return: {paper.ev_per_unit(p, odds):+.2%} per unit staked.")
    token = st.number_input("Genuinely free token value (GHS)", min_value=0.0, value=0.0)
    returned = st.checkbox("Token stake is returned on a win")
    cost = st.number_input("All qualifying costs / expected losses (GHS)", min_value=0.0, value=0.0)
    simple = st.checkbox("I verified there is no remaining rollover, withdrawal restriction, tax, cap or expiry loss")
    if simple:
        value = paper.promotion_ev(p, odds, token, returned, cost)
        st.write(f"Assumption-based token value after qualifying costs: GHS {value:+.2f}.")
    else:
        st.info("Unpriced promotion terms: no positive-EV verdict available.")
    bankroll = st.number_input("Bankroll for stake illustration (GHS)", min_value=0.0, value=0.0)
    minimum = st.number_input("Minimum stake for illustration (GHS)", min_value=0.01, value=1.0)
    sizing = paper.minimum_stake_check(bankroll, minimum, p, odds)
    if sizing["below_minimum"]:
        st.write("Do not bet: illustrative quarter-Kelly sizing (1% cap) is below the minimum. Never round up.")
    else:
        st.write(f"Illustrative size GHS {sizing['illustrative_stake']:.2f}. Paper mode cash recommendation remains GHS 0.")


def main():
    st.set_page_config(page_title="Betting Evidence Journal", page_icon="📒", layout="wide")
    st.title("Betting Evidence Journal")
    st.caption("Paper trading · your actual prices · measured results · Ghana time (UTC)")
    st.warning("No demonstrated profitable strategy. No bet today. Recommended cash stake: GHS 0.")
    authenticate()
    location = str(setting("RESEARCH_DATABASE_URL", "research.db")).strip() or "research.db"
    try:
        store = Store(location)
    except Exception:
        st.error("Cannot open research storage. Check database configuration. No temporary fallback was used.")
        st.stop()
    try:
        if store.postgres:
            st.caption("Journal storage: external PostgreSQL. Keep independent private backups.")
        else:
            st.warning("Journal storage: local SQLite. On hosted Streamlit this can disappear. "
                       "Configure RESEARCH_DATABASE_URL for PostgreSQL before relying on hosted tracking.")
        view = st.radio("Workspace", ["Paper prices", "Results", "Actual history & backup", "Account", "Calculators", "Method"],
                        horizontal=True, label_visibility="collapsed")
        views = {"Paper prices": paper_board, "Results": results,
                 "Actual history & backup": history, "Account": account}
        if view in views:
            views[view](store)
        elif view == "Calculators":
            tools()
        else:
            st.markdown("""
### Research rules

Only fresh prices at your own SportyBet or bet365 account can create paper
singles. Eight complete, fresh books and a sharp reference are required.
The probability adjustment fitted to historical closing prices is disabled.
Book disagreement is an uncertainty heuristic, not a calibrated confidence interval.
Public team news never adds an invented probability or star rating.

The protocol records every submitted price and freezes the first qualifying
price per event, bookmaker and selection. Refreshing never rewrites entry prices.
Paper stakes are one unit and stay separate from actual cash results.
No accumulator recommendations or cash stakes are issued.

Near-close CLV uses odds_taken × de-vigged sharp probability − 1, with observed
timestamps. This is a sampled pre-kickoff benchmark, not a guaranteed closing price.
The scheduled worker must run before kickoff; it cannot recover missed closes
on the free feed. Automatic results use exact event/team matches and completed
scores. Unsupported or ambiguous bets require verified manual settlement.

The existing retrospective research does not establish an actionable edge at
your books. A successful prospective paper screen needs independent confirmation
and a review of executable prices, costs, dependence and missing observations.
Losing streaks never justify increasing stakes.
""")
        budget = store.get("quota")
        if budget:
            st.sidebar.caption(f"Reserved API credits: {budget['daily']} on {budget['day']}; "
                               f"{budget['monthly']} in {budget['month']}. Limits: 12/day, 400/month.")
    finally:
        store.close()
