# Betting Evidence Journal

**No profitable strategy has been demonstrated for this owner's SportyBet or
bet365 prices. The dashboard recommends no cash bets.**

The former favourites/accumulator tip sheet has been replaced by a private,
paper-first journal. A high chance of winning is not evidence that a price is
profitable. There are no daily quotas of picks and no recovery staking.

## What it does

- Checks manually entered SportyBet or bet365 prices against fresh market data.
- Saves every submitted price, timestamp, estimate, reason for rejection and raw
  benchmark snapshot. Prices survive browser reloads in configured storage.
- Freezes the first qualifying offer per event/book/selection as a 1-unit paper
  single. Later checks do not rewrite that paper entry or duplicate it.
- Keeps actual GHS bets, actual losses and paper results separate.
- Settles supported h2h records from completed scores with exact event/team IDs;
  supports verified manual settlement for everything else.
- Samples pre-kickoff sharp benchmarks with timestamps. Missing closes stay
  missing. It cannot reconstruct past closes from the free feed.
- Supports private JSON backups, repeatable actual-history CSV imports, and a
  PostgreSQL journal for hosting. Existing `bets.db` is left untouched and can
  be imported using the history screen.
- Requires login before opening personal records or allowing quota usage. All
  odds and scores requests are explicit or run by the configured worker.

The six workspace views are Paper prices, Results, Actual history & backup,
Account, Calculators and Method. Pure research modules and the historical
backtest scripts remain available from Python. The old UI and claims are
documented in [the archived methodology](docs/legacy-methodology.md).

## Run locally

Python 3.11+:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# Set DASHBOARD_PASSWORD and, for live benchmarks, ODDS_API_KEY in .env.
./start.sh
```

The launcher binds to `127.0.0.1`. This audit generated a local password in the
ignored `.dashboard-password` file and added it to `.env` if none existed.
Neither password nor API key belongs in Git. The journal defaults to the
ignored `research.db`; back it up. A missing API key does not prevent using
the journal, importing history or running calculators.

Bankroll defaults to **unknown/zero**. Enter the real GHS balance and minimum
stake in Account. Illustrative sizing is rounded down and returns zero when
below the minimum; cash recommendations remain zero even if an illustration
is positive. The minimum is a user setting, not an asserted bookmaker rule.

## Paper protocol v1

Scope: EPL, La Liga, Bundesliga, Serie A, Ligue 1 and NBA h2h. Soccer league
1X2 uses regulation time; NBA moneyline includes overtime. No obscure or
international competitions, totals, alternate lines or accumulators enter
the prospective sample. Basketball probabilities come only from the market;
there is no basketball forecasting model.

Before paper entry:

1. The match must have a valid ID and start within 24 hours.
2. The owner's price and eight complete, distinct bookmaker quote vectors
   must be at most 15 minutes old, with aware timestamps. Missing/future
   timestamps and incomplete 1X2 markets are rejected.
3. A fresh sharp benchmark must exist and dispersion must be ≤3.5 percentage
   points. The owner must confirm lineups, motivation and matching rules.
4. Estimated EV after subtracting bookmaker dispersion from probability must
   be at least +1%; raw estimated EV above +20% is rejected as suspect.

Calibration fitted to historical closing prices is **off by default** in live
pricing. Research opt-in calibration is restricted to fitted competitions and
markets. Neither dispersion nor the de-vigged benchmark is a calibrated
confidence interval or a demonstrated betting edge.

The predeclared descriptive screen requires 300 distinct events, at least
60 days, ≥90% near-close coverage, positive event-averaged CLV and t>2. Quotes
on one match count as one event cluster. Zero-variance samples do not produce
infinite significance. Missing closes remain visible in the denominator.
A passing screen requires separate prospective confirmation and examination
of dependence, price execution, costs and missingness; it never enables cash
stakes automatically. The old retrospective line-shopping result (+2.10% on
1,408 bets, reported t≈0.79) did not establish an actionable edge at these books.

## Hosted persistence and access

In Streamlit Community Cloud Secrets, configure:

```toml
ODDS_API_KEY = "your-private-key"
DASHBOARD_PASSWORD = "a-long-random-private-password"
RESEARCH_DATABASE_URL = "postgresql://user:password@host/database?sslmode=require"
```

Use an owner-controlled PostgreSQL database and keep independent private
backups. On connection failure the app stops; it does not silently switch to
an empty local database. Without a database URL, local SQLite is explicitly
labelled temporary on hosting. Without a password the app remains locked.
Hosted database provisioning and secrets must be completed in the owner's
accounts; this code change cannot create credentials or recover old lost bets.

Do not upload private backups, slips or account details to the public repo.
The app uses one shared owner password and one owner journal, not multi-user
account isolation. For wider audiences use platform authentication and review
access controls first.

## Settlement and near-close worker

The worker never creates bets or recommendations:

```sh
.venv/bin/python -m scripts.sync_research
```

It requires `ODDS_API_KEY` and an external `RESEARCH_DATABASE_URL`; without
external storage it skips. The optional GitHub Actions workflow runs every
five minutes after it is deployed on the default branch and both repository
secrets are configured and the Actions repository variable
`RESEARCH_WORKER_ENABLED` is set to `true`. It also supports a manual dispatch.
Scheduled jobs are disabled by default to avoid unconfigured runs. GitHub schedules
can run late; close coverage is best effort, never guaranteed. For reliable
timing use an owner-controlled scheduler.

The worker polls odds only for recorded matches in the final five minutes
before kickoff and result feeds for open supported matches from 3 hours to
3 days after kickoff. Scores are cached for 24 hours per league. Completed
results must be fetched within the provider's three-day window; older and
ambiguous records require manual verification. Near-close snapshots require
a sharp quote updated within the final five minutes; no post-kickoff quote
is accepted. CLV is `entry_odds * de_vigged_sharp_probability - 1`, a sampled
benchmark rather than a claim to the exact final closing line.

## API budget

No API requests happen on initial page load, login or ordinary navigation.
Odds requests use one region (`eu`) and one market (`h2h`), with a 15-minute
shared cache. Completed score requests reserve two credits. Atomic database
reservations enforce 12 credits per UTC day and 400 per calendar month across
the dashboard and worker sharing that database. Failed requests remain
reserved conservatively. This is an application budget, not the provider's
billing cycle or remaining balance; other apps using the key are outside it.
Starting with an empty/recreated database resets this local counter. Monitor
the provider's actual remaining balance too. A budget stop or missing data
means no paper entry or close capture, never a reason to relax a gate.

## History and promotion calculators

Actual-history CSV accepts the downloaded template or the former `bets.csv`
export. Include an aware `placed_at`, book, selection, actual odds/stake and
verified result; leave result empty when unknown. Prices/stakes must be
finite and valid. The full file is validated before import. Stable `bet_id`
values deduplicate repeated imports; use distinct IDs for genuinely separate
identical bets. A whole accumulator is one actual record, not several wins.

Calculators distinguish cash-funded boosted prices from genuinely free
stake-returned/stake-not-returned tokens. They provide assumption-based
arithmetic only. Unknown rollover, withdrawal restrictions, taxes, caps,
qualification costs or expiry loss prevent an offer verdict. There is no
claim that any available promotion has been established profitable.

## Verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests app.py scripts
```

Regression tests cover price freshness, complete soccer markets, duplicate
bookmakers, paper entry immutability, actual-history import, settlement,
close timing, event clustering, quota reservations and offline authenticated
Streamlit navigation. See [the audit](docs/AUDIT.md) for findings and limits.

No scraping, committed API keys, AI attribution lines or Aviator strategy.
