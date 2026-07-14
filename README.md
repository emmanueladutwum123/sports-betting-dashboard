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

- **Sports** — toggle soccer / basketball / baseball / cricket.
- **Bookmaker regions** — `eu` (Pinnacle, Unibet, ...), `uk` (William Hill,
  Ladbrokes, ...), `us` (FanDuel, DraftKings, ...), `au`. More regions =
  more books compared = more API quota used per refresh.
- **Show live / upcoming** — toggle each section independently.
- **Include recent-form lookup** — optional, slower; best-effort last-5
  results per team from TheSportsDB, skipped (shown as "N/A") if a team
  can't be matched by name.
- **Force refresh** — clears the cache to pull fresh odds immediately
  (normal auto-refresh is every 5 minutes for odds, 3 minutes for live
  scores, to conserve your monthly quota).

## Free-tier quota notes

The free plan is 500 requests/month; each odds/scores call costs 1 credit
per region per market. Cached for 3–5 minutes so repeated page interactions
don't re-spend quota. If you hit the limit, wait for the monthly reset or
narrow your `regions`/`sports` selection.
