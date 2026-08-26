# 📊 Betting Research Dashboard

A local dashboard that lists live and upcoming fixtures across **soccer,
basketball, baseball, and cricket**, with real bookmaker odds, de-vigged
market-implied win probabilities, and a defensible recommended pick per game
— built around a non-greedy, evidence-first betting methodology (never the
extreme end of an Over/Under board, always a stated confidence level, never
a fabricated stat).

## ⚠️ Read this first

- Odds come from **[The Odds API](https://the-odds-api.com)**, a licensed
  aggregator. Bookmaker coverage depends on region: FanDuel / DraftKings /
  BetMGM (US), William Hill / Unibet / Ladbrokes / Pinnacle (UK/EU).
  **bet365, SportyBet, and Betway are not available through any legitimate
  API**, and this project deliberately does not scrape sportsbook websites
  (against their terms of service, and fragile). Treat the recommended
  selection and line as the pick; sanity-check the exact price on your own
  book before staking.
- "Fair probability" is the odds with the bookmaker's margin mathematically
  removed (de-vigging) — it reflects what the market collectively prices,
  not a proprietary prediction model.
- The Over/Under pick is always the **balance-point line** — the line where
  Over and Under are closest to a fair 50/50 — never the cheapest
  near-certain end or the flashiest extreme end of the board.
- Confidence stars reflect how many independent bookmakers quote/agree on a
  line (market liquidity), **not** certainty that the bet wins. Nothing here
  is guaranteed. Bet responsibly, within what you can afford to lose.
- MotoGP and swimming are intentionally not included — neither sport has
  reliable structured odds/stats coverage in any free or affordable API.

## Setup (one-time)

1. Get a free API key (no credit card) at **https://the-odds-api.com** —
   500 requests/month free tier, plenty for personal use with the built-in
   5-minute cache.
2. Clone this repo and open a terminal in it.
3. Run the app once to generate a `.env` file:
   ```bash
   ./start.sh
   ```
4. Open `.env` and paste your key in place of `your_key_here`.
5. Run `./start.sh` again.

## Deploy it online (Streamlit Community Cloud, free)

Want a shareable URL instead of `localhost`? This repo is deploy-ready:

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**, and fill in:
   - Repository: `emmanueladutwum123/sports-betting-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   ODDS_API_KEY = "your_real_key_here"
   ```
   (You can also add this later from the app's ⋮ menu → Settings → Secrets.)
4. **Deploy.** First build takes a couple of minutes; you get a public URL
   like `https://<app-name>.streamlit.app`.

Notes for the hosted version:

- The key lives in Streamlit's secrets store, never in the repo — `.gitignore`
  already excludes `.streamlit/secrets.toml`. See `.streamlit/secrets.toml.example`.
- **A public app shares your API quota with everyone who opens it.** The free
  Odds API tier is 500 credits/month, so keep the league selection narrow, or
  set the app to private (Settings → Sharing) if the quota starts disappearing.
- Free Cloud apps sleep after ~12 hours idle and wake on the next visit
  (first load after sleeping is slow, then normal).

## Running it

```bash
./start.sh
```

or on macOS, just double-click **`Start Dashboard.command`** in Finder.

Either way, it opens a terminal, installs dependencies into a local `.venv`
on first run, starts the dashboard server, and **automatically opens a
browser tab** at `http://localhost:8501`. Leave the terminal window open
while you use it; `Ctrl+C` in that window stops the server.

## What's inside

```
app.py                  Streamlit dashboard UI
src/odds_api.py         The Odds API client (sport discovery, odds, scores)
src/probability.py      Implied probability, de-vig, confidence stars
src/picks.py            Balance-point O/U line selection + verdict text
src/form.py             Best-effort recent-form lookup (TheSportsDB, optional)
start.sh                One-command launcher (venv setup + run)
```

## Dashboard controls

- **Sports** — toggle soccer / basketball / baseball / cricket. Defaults to
  soccer only — add more deliberately, since each one costs quota.
- **`<Sport>` leagues** — appears per selected sport once you pick it; choose
  specific leagues instead of pulling everything in season. Defaults to the
  first 5 available. **This is the main quota lever** — a sport can have
  15-20+ leagues in season at once, and each one is a separate paid call.
- **Bookmaker regions** — `eu` (Pinnacle, Unibet, ...), `uk` (William Hill,
  Ladbrokes, ...), `us` (FanDuel, DraftKings, ...), `au`. More regions =
  more books compared = more API quota used per refresh.
- **Show live / upcoming** — toggle each section independently. Live pulls a
  second (scores) call per selected league, so turn it off if you only care
  about upcoming fixtures.
- **Include recent-form lookup** — optional, slower; best-effort last-5
  results per team from TheSportsDB (separate free API, doesn't touch Odds
  API quota), skipped (shown as "N/A") if a team can't be matched by name.
- **Force refresh** — clears the cache to pull fresh odds immediately
  (normal auto-refresh is every 30 minutes for odds, 5 minutes for live
  scores, to conserve your monthly quota).

## Free-tier quota notes

The free plan is 500 credits/month; each odds/scores call costs roughly 1
credit per region per market requested. **Selecting leagues in the sidebar,
not sport groups, is what controls spend** — picking "Soccer" alone can mean
15-20+ leagues if you don't narrow the league list. Cached for 5–30 minutes so
repeated page interactions and accidental reruns don't re-spend quota. If you
see "Odds API monthly quota exhausted," that's real — wait for your monthly
reset (check your account at the-odds-api.com for the exact date), upgrade to
a paid tier, or narrow `regions`/leagues next session.
