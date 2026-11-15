# 📊 Quantitative Betting Research

A betting-research dashboard built around one honest question: **is this price
better than its fair value, and by enough to survive the uncertainty in my own
estimate?**

Most betting tools answer a different, easier question — "who wins?" — and then
recommend a bet regardless. This one frequently recommends nothing, because on
most days nothing on the board is mispriced. That is the market working, and
reporting it is the point.

---

## Why the previous version had no edge

This project began as a fixture list with de-vigged probabilities and a
"balance-point" Over/Under pick — the line closest to a fair 50/50. The
reasoning was risk-averse and sensible-sounding: avoid the cheap near-certain
end of the board, avoid the flashy extreme end, take the middle.

It is an anti-strategy. The balance-point line is precisely where the market has
the most information and the least disagreement. Betting it maximises variance
and returns exactly minus the vig, every time. **No line is inherently good to
bet. Only a mispriced line is good to bet.**

Three further mathematical problems were fixed along the way:

| Bug | Effect |
|---|---|
| Averaging **decimal odds** across books, then de-vigging | Odds are a reciprocal scale; by Jensen's inequality the mean price is longer than the price implied by the mean probability. This *manufactures* edge that does not exist. |
| De-vigging the **pooled** number instead of each book separately | Mixes books' different margins together and leaves a residue that varies with how lopsided the game is. |
| **Proportional** (multiplicative) de-vig | Assumes margin is spread in proportion to price. Books load margin onto longshots. On a lopsided 1X2 this misprices the favourite by **>1 percentage point** — larger than the entire edge a +EV bettor hunts for. |

---

## What it does now

### 1. Fair value, built correctly

Four steps, each fixing one of the problems above:

1. **De-vig each book's own complete price vector** using [Shin's method]
   (Shin 1993) by default, which models the bookmaker as pricing to protect
   against informed money. Power, additive, odds-ratio and multiplicative are
   available in the sidebar so you can watch the bias move.
2. **Pool across books in log-odds space** — the maximum-likelihood pool under a
   logistic error model, and the only scale on which averaging probabilities is
   not biased.
3. **Anchor to sharp books.** Pinnacle and the exchanges run ~2% margins, take
   large limits, and welcome winning bettors, so their prices clear a market. A
   recreational book's price is shaded toward public sentiment. They are weighted
   4–8× accordingly.
4. **Exclude the book under evaluation** from its own benchmark. Leaving it in is
   the same leakage as scoring a model on its training data, and it shrinks
   edges exactly when they are most real.
5. **Correct the residual favourite–longshot bias.** Shin de-vigging removes
   most of it, not all. Measured over 66,272 matches, the de-vigged close is
   excellent in the middle of the probability scale and about **1.5 points too
   low** at the top of it. One fitted exponent fixes that. Details below.

### 1b. Calibration — the number next to a favourite is now honest

Taking Pinnacle's closing 1X2 prices across 18 leagues and 10 seasons, de-vigging
each with Shin, and comparing to what actually happened:

| band | outcomes | mean price | actual | bias |
|---|---:|---:|---:|---:|
| 0.5–0.6 | 12,798 | 0.5459 | 0.5478 | +0.19pp |
| 0.6–0.7 | 6,568 | 0.6444 | 0.6591 | +1.47pp |
| 0.7–0.8 | 3,550 | 0.7440 | 0.7654 | +2.14pp |
| 0.8–0.9 | 1,815 | 0.8410 | 0.8523 | +1.13pp |
| **≥ 0.6** | **13,277** | **0.7063** | **0.7215** | **+1.52pp** |

That last row is ±0.76pp at 95% — about four standard errors. It is not
selection bias (noise in the estimate would push the realised rate *down*, not
up), not one book (Pinnacle +1.66pp, cross-book average +1.85pp, Bet365
+1.77pp on a common sample), and not one period (positive in 8 of 10 seasons
and 13 of 18 leagues).

The fix is one parameter — raise the probability vector to a power and
renormalise. Fitted by out-of-sample log loss it comes out at **1.0408** on
1X2 and **1.0409** on Over/Under 2.5: two different markets, different numbers
of outcomes, agreeing to four decimals.

**Be clear about what this is worth.** As a forecast of the average match,
almost nothing: 0.00011 nats of log loss in an expanding-window backtest,
better in 7 of 9 seasons. It does not beat the market and it is not an edge.
What it fixes is the band the Daily Card actually selects from:

| band | bias before | bias after |
|---|---:|---:|
| p ≥ 0.6 | +1.61pp | +0.35pp |
| p ≥ 0.7 | +1.78pp | +0.47pp |
| p ≥ 0.8 | +1.15pp | −0.08pp |

#### …and one exponent turned out to be the wrong shape

A single global constant helps 10 of 18 leagues and **hurts** the other 8. The
Bundesliga close does not understate favourites — it slightly overstates them —
so sharpening it further pushed its bias from −1.0pp to −2.3pp. The Premier
League is already near-unbiased and got worse too.

Fitting a separate exponent per league overfits: at 2,000–5,500 matches each
the estimates are noisy, and used raw they score *worse* out-of-sample than the
single global number. The answer is partial pooling — shrink each league's
estimate halfway toward the global one, with the shrinkage chosen
out-of-sample:

| shrinkage K | log loss | vs no calibration |
|---|---:|---:|
| 0.00 (global only) | 0.993347 | +0.000081 |
| 0.25 | 0.993304 | +0.000123 |
| **0.50** | **0.993298** | **+0.000129** |
| 0.75 | 0.993329 | +0.000098 |
| 1.00 (per-league raw) | 0.993396 | +0.000031 |

The curve peaks in the middle and full per-league pooling is barely better than
nothing — that is what overfitting looks like once you measure it.

The pooled log-loss gain is still tiny. What partial pooling actually buys is
that it stops breaking the sharp leagues. Mean absolute bias at p ≥ 0.6,
averaged over the 18 leagues, out-of-sample:

| | mean abs. bias |
|---|---:|
| no calibration | 2.55pp |
| one global exponent | 2.09pp |
| **shrunk per-league** | **1.65pp** |

Shipped exponents range from **0.9854** (Bundesliga — sharpening *down*) to
**1.0993** (Turkish Süper Lig). Soccer only; other sports pass through
untouched, and a soccer league with no fitted entry falls back to the global
number. Refit — including the per-league table — with
`python -m scripts.fit_calibration`.

### 1c. The best price you can actually take

The backtest's one positive finding is that edge comes from *price*, not
forecasting (see "So where is the edge?" below). The **Best Odds** tab is built
on that: for every selection it shows who pays the most, the next-best price,
what a typical book pays, and whether even the best price clears fair value.
On a live 58-match board across 42 books, the median selection paid **5.3%
more** at the best book than at the median one — and only 26 of 342 best
prices were actually value. Both numbers matter.

Four things make a raw "highest odds" column wrong, and each is handled:

| Problem | What it did | Fix |
|---|---|---|
| **Exchange commission ignored** | Betfair's 7.60 pays 7.27 after 5%. Raw, the exchange looked like the best price when a bookmaker at 7.50 paid more — and it inflated every arbitrage and middle with an exchange leg. | Prices are compared and evaluated net (`EXCHANGE_COMMISSION` in `src/market.py`); de-vigging still uses the raw price. |
| **Stale quotes** | The highest number on a board is disproportionately a book that stopped updating. | A quote more than 30 minutes behind its market's median update is flagged, dropped from the +EV board, and skipped when choosing the best price. |
| **In-play matches** | The feed keeps returning a match after kick-off; books reprice at different speeds, so comparing them manufactures edges out of latency. | Everything that recommends a bet uses pre-match events only. |
| **The same bet counted many times** | The +EV board listed one row per *book* clearing the threshold and sized each as an independent wager — and could stake Home and Draw on one match. | One row per selection at its best-paying book; one staked selection per market per match. |

**Books you can use.** Pick yours under *Your bookmakers* and the best price
becomes the best one you can take. The fair line is still built from every
book: a price you cannot bet is still evidence about what the right price is.

**Books that are not in the feed.** SportyBet and bet365 are in no licensed
feed. *Check and add your SportyBet or bet365 prices* takes the odds you are
being offered at either book (or any other you name), judges them against the
same fair value, and lets you save them so they compete in the Best Odds table
as "SportyBet (yours)" or "bet365 (yours)" for the session — verdict, the lowest
price at which it becomes a bet, and the stake. A price more than 20% above
fair gets no verdict and no stake until it has been checked, because that is
far more often a typo or a mismatched market than a gift.

### 2. Sizing that survives being wrong

Kelly assumes you *know* the probability. You have an estimate. Because Kelly's
growth curve falls away faster on the downside than it rises on the upside,
plugging in a point estimate systematically **over**bets.

The engine therefore sizes on a *shrunk* probability, with the shrinkage taken
from how much the books actually disagree — so it automatically bets least on
the games it understands least. Default is quarter-Kelly with a slate-wide
exposure cap.

Verified numerically in `tests/test_edge.py`: with a genuine +10% EV edge,
expected log growth at **2× Kelly is negative**. Overbetting does not reduce
your return, it reverses it.

### 3. A backtest designed to fail

`walk_forward()` refits the model before every prediction on strictly prior
matches, passes the prediction date as the time-decay reference so even the
weights carry no lookahead, and scores against **de-vigged Pinnacle closing
prices** — not opening prices, which are soft and flatter any model.

### 4. Closing line value, not P&L

At a realistic 2–3% edge it takes **thousands** of settled bets to distinguish
skill from luck — the ledger computes the exact number for your own betting.
CLV is observable within hours of each bet, far less noisy, and is the metric
sportsbooks themselves use to identify and limit winning accounts.

The ledger refuses to declare an edge on a small sample: verdicts are gated on
both sample size (n ≥ 30) and significance (|t| > 2).

---

## The headline finding

Dixon-Coles walked forward across **six seasons per league in 18 European
divisions** (~32,000 matches), refit before every prediction, scored against
de-vigged Pinnacle **closing** prices. `w` is the optimal weight to place on the
model, fitted out-of-sample against the market:

| League | 1X2 n | mkt LL | model LL | **w** | O/U n | mkt LL | model LL | **w** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English Premier League | 1890 | 0.9553 | 0.9811 | **0** | 1881 | 0.6717 | 0.6890 | **0** |
| English Championship | 2358 | 1.0329 | 1.0632 | **0** | 2187 | 0.6822 | 0.6999 | 0.025 |
| English League One | 2746 | 1.0139 | 1.0539 | **0** | 2572 | 0.6869 | 0.7101 | **0** |
| English League Two | 2795 | 1.0512 | 1.0899 | **0** | 2611 | 0.6802 | 0.7038 | **0** |
| Spain La Liga | 1892 | 0.9740 | 1.0003 | **0** | 1887 | 0.6709 | 0.6886 | **0** |
| Spain Segunda | 2354 | 1.0435 | 1.0835 | **0** | 2275 | 0.6575 | 0.6851 | **0** |
| Germany Bundesliga | 1450 | 0.9749 | 1.0002 | **0** | 1412 | 0.6530 | 0.6686 | **0** |
| Germany Bundesliga 2 | 1444 | 1.0378 | 1.0648 | **0** | 1438 | 0.6768 | 0.6987 | **0** |
| Italy Serie A | 1888 | 0.9610 | 0.9875 | **0** | 1885 | 0.6772 | 0.6896 | 0.050 |
| Italy Serie B | 1835 | 1.0464 | 1.0893 | **0** | 1822 | 0.6817 | 0.7213 | **0** |
| France Ligue 1 | 1716 | 0.9930 | 1.0177 | **0** | 1709 | 0.6775 | 0.6967 | **0** |
| France Ligue 2 | 1781 | 1.0397 | 1.0868 | **0** | 1773 | 0.6763 | 0.7164 | **0** |
| Netherlands Eredivisie | 1374 | 0.9196 | 0.9576 | **0** | 1374 | 0.6613 | 0.6757 | **0** |
| Portugal Primeira Liga | 1448 | 0.9197 | 0.9526 | **0** | 1447 | 0.6797 | 0.7008 | **0** |
| Belgium First Div | 1314 | 0.9733 | 1.0115 | **0** | 1310 | 0.6741 | 0.6833 | 0.150 |
| Turkey Super Lig | 1713 | 0.9851 | 1.0277 | **0** | 1710 | 0.6789 | 0.7057 | **0** |
| Greece Super League | 1051 | 0.9379 | 0.9582 | 0.125 | 1016 | 0.6659 | 0.6865 | **0** |
| Scotland Premiership | 937 | 0.9250 | 0.9468 | **0** | 928 | 0.6730 | 0.6856 | **0** |

**The market wins in every single league, on both markets.** The model's log
loss is worse than the closing line's in all 36 tests without exception, and in
**32 of 36** the optimal weight on the model is exactly **zero**.

The four non-zero weights should be read as noise, not as discoveries. They are
4 hits out of 36 searches, each worth a log-loss improvement of ≤0.0005 — the
textbook signature of multiple comparisons, not of edge. Treating "we found
alpha in the Greek Super League" as a finding is precisely the mistake this
backtest exists to prevent.

Note also what did **not** happen: efficiency does not decay with liquidity.
Closing lines in English League Two are as hard to beat as in the Premier
League. The intuition that small leagues are soft is, on this evidence, wrong —
at least at the close.

### So where is the edge?

Two places, neither of them forecasting:

- **Line shopping.** In the same backtest, betting the sharp consensus at the
  *best available* price across books returned **+2.10%** (1408 bets), where the
  identical signal at Pinnacle's own price produced **no qualifying bets at
  all**. Taking the best of many prices is a larger and far more reliable source
  of edge than out-forecasting the close. Honesty demands the caveat: t ≈ 0.79,
  so even this is not statistically established on one league's sample.
- **Discipline.** Staking correctly, tracking CLV, and not betting when nothing
  is mispriced.

The model still earns its place: it prices totals, both-teams-to-score, Asian
handicaps and correct scores from **one internally consistent** score matrix, so
its markets cannot contradict each other the way a book quoting each
independently can.

### Correction: the totals column above was scored against opening prices

The O/U half of that table is not what it claims to be. The feed looked for a
Pinnacle closing column named `PSCO>2.5`, which does not exist in these files,
and silently fell through to `P>2.5` — the **opening** price. Every totals
number above was therefore measured against a soft line, which is precisely the
failure `src/backtest.py` warns about in its own docstring.

The headline did not change — the model loses on totals either way — but the
three small non-zero totals weights (Championship 0.025, Serie A 0.050, Belgium
0.150) were fitted in-sample against a soft line and should be read as zero.
Re-measured against the true closing price (`PC>2.5`), walk-forward over 10,876
markets in 6 leagues:

| forecaster | log loss |
|---|---:|
| market (Pinnacle close) | **0.672048** |
| Dixon-Coles, goals only | 0.687644 |
| Dixon-Coles, best goals/shots mix | 0.681384 |

and with the blend weight fitted on the first half and scored on the second —
the only honest way to fit it — adding the model to the market **loses**
0.00062 ± 0.00050 nats. So the standing hypothesis that totals is where a goal
model earns its keep is now tested, and it is false.

### The model did get better — it just still does not beat the market

Fitting Dixon-Coles to **shots on target** as well as goals, and mixing the two
in log-rate space, is a real improvement. Walk-forward over 20,368 predictions:

| shots weight | 1X2 log loss |
|---|---:|
| 0% (goals only) | 1.003511 |
| 25% | 1.000200 |
| **50%** | **0.999340** |
| 75% | 1.001021 |
| 100% (shots only) | 1.005441 |

A flat optimum in the middle is what you would expect if the two signals carry
roughly equal independent information: shots on target are a better-sampled
measure of territorial control (~9 per match against 2.7 goals), goals carry
the finishing that shots throw away.

That gain is about **forty times** the size of the calibration effect. It still
is not enough: the market's 0.9781 beats all of them, and the fitted blend
weight against a sharp close remains exactly zero. Better ratings and better
scoreline distributions, no betting edge. The Model Lab defaults to the 50/50
mix and the slider goes back to 0 for the classic goals-only fit.

### A live bug this engine caught

On first run against real data the +EV board showed **71 selections**, nearly all
draws at ~+17% EV. Every one was fake.

Betfair's exchange was returning `1.04 / 1.04 / 1.04` on a 1X2 with no matched
liquidity — a 65% "margin". That vector de-vigs to exactly ⅓ per outcome, so it
dragged the fair price of every draw toward 3.00. Worse, exchanges are weighted
as *sharp*, so the artefact was **setting the anchor** rather than being
outvoted by it.

Adding a plausibility filter on per-book margin (`MIN/MAX_PLAUSIBLE_MARGIN` in
`src/market.py`) collapsed those 71 phantom edges to **1 real one**. It is
covered by regression tests, and it is the reason the arbitrage tab is framed as
a data-integrity check rather than a profit centre.

## Tabs

| Tab | What it is for |
|---|---|
| 🛒 **Best Odds** | Best takeable price for every selection, net of commission, against fair value — plus a checker for prices from a book that is not in the feed. Start here. |
| 🔥 **Daily Card** | The 5 best-evidenced soccer and 5 basketball selections in the next 24–72h, one per competition so the legs stay independent. Three ranking modes (most-likely / best-value / balanced), a liquidity gate, and exact Poisson-binomial accumulator arithmetic. Reports being short rather than padding. |
| 🎯 **+EV Board** | Every selection priced above fair value at its best-paying book, ranked by EV, sized by fractional Kelly, with flags for anything that smells like a data artefact. Often empty. |
| 🗓️ **Fixtures** | Per-match fair line, best price, book disagreement, and where sharp books disagree with the recreational consensus. |
| ⚖️ **Arb & Middles** | Riskless positions — and a live integrity check on the feed, since a large "arbitrage" almost always means a stale quote. |
| 🧪 **Model Lab** | Fit Dixon-Coles on real historical results; price any matchup across five markets; run the walk-forward backtest. |
| 📒 **Ledger & CLV** | Log bets, record closing prices, get graded on CLV with significance gating. |
| 📘 **Methodology** | What every number means and what it cannot do. |

---

## Why the Daily Card will not promise you winners

The card ranks by the strongest evidence available, and it separates three
things that all get called "confidence":

- **Confidence the pick wins** — the win probability. Maximised by short-priced
  favourites, which carry the *worst* value on the board. Backing them all
  season is a high-strike-rate way to lose slowly.
- **Confidence the price is wrong in your favour** — expected value. Maximised
  by selections the market disagrees with, which are usually longshots that lose
  most of the time.
- **Confidence in the estimate** — how many books agree, and how tightly. This
  is a *gate*, not a ranking: ≥8 books, ≤3.5pp disagreement, sharp or consensus
  anchor. It says nothing about who wins, only how much to trust the other two.

You pick which of the first two to rank by. Both are honest; they answer
different questions, and the tab says so in the UI rather than hiding it.

### The accumulator arithmetic

Because the card deliberately spans different countries, its legs are close to
independent — which is exactly what makes combining them a bad idea. Independent
probabilities multiply:

- 10 legs at 75% each → **5.6%** chance all land.
- A real card of 6 legs averaging 73% → **15.7%**, with ~4.4 expected winners.

Expected value alone does **not** condemn parlays, and the tab does not pretend
it does: EV compounds as `∏(1+eᵢ)−1`, so combining genuinely +EV legs *raises*
the EV. The decisive argument is growth. Kelly growth is additive across
independent bets and brutally concave in a single one, so backing the legs
separately dominates parlaying them — measured at **~30× the log-growth rate**
on a real card, even in the case where the parlay's EV was the higher number.

That is why accumulators are the most profitable product a sportsbook sells.

## Setup

```bash
./start.sh                      # generates .env on first run
# paste your key into .env, then:
./start.sh
```

Get a free key (no card) at **[the-odds-api.com](https://the-odds-api.com)** —
500 requests/month, ample with the built-in caching.

Historical results and closing odds come from
**[football-data.co.uk](https://www.football-data.co.uk)** (free, no key) for 18
European leagues.

### Command line

```bash
python scripts/run_backtest.py E0 2019,2020,2021,2022,2023
python scripts/run_backtest.py --totals SP1 2020,2021,2022,2023
```

### Tests

```bash
python -m pytest tests/ -q      # 204 tests
```

---

## Deploy (Streamlit Community Cloud, free)

1. **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub.
2. **Create app** → repo `emmanueladutwum123/sports-betting-dashboard`, branch
   `main`, main file `app.py`.
3. **Advanced settings → Secrets**:
   ```toml
   ODDS_API_KEY = "your_real_key_here"
   ```
4. Deploy.

The ledger uses SQLite, and Community Cloud's filesystem resets on redeploy —
export the CSV to keep anything you care about.

---

## Layout

```
app.py                      eight-tab Streamlit UI
src/
  devig.py                  5 de-vig methods (Shin, power, additive, OR, multiplicative)
  market.py                 per-book de-vig, log-odds pooling, sharp-book weighting
  calibration.py            residual favourite-longshot correction, fitted on 66k matches
  edge.py                   EV, Kelly, uncertainty shrinkage, slate exposure caps
  picks.py                  +EV scanner and per-event summaries
  shop.py                   line shopping, pre-match filter, own-book price checker
  arbitrage.py              arbitrage and middles (doubles as a feed integrity check)
  blend.py                  log-odds model/market blending, weight fitted out-of-sample
  backtest.py               walk-forward vs closing prices, bankroll simulation
  ledger.py                 SQLite bet log, CLV tracking with significance gating
  models/dixon_coles.py     bivariate Poisson goal model (goals + shots on target)
  datafeeds/football_data.py  historical results + closing odds
scripts/run_backtest.py     CLI backtest
scripts/fit_calibration.py  refit the calibration exponent against closing prices
tests/                      204 tests
```

---

## ⚠️ Honest limits

- **Not a guaranteed-profit system.** No such thing exists. The backtest above
  says so explicitly.
- **The calibration is not an edge.** It corrects a measured 1.5-point
  understatement on favourites and is worth 0.0001 nats as a forecast. It makes
  the displayed probability honest; it does not make it beat the market. Any
  reading of it as a source of profit is wrong.
- Odds move; the best price may be gone by the time you reach it.
- Accounts that consistently beat the closing line **get limited or banned** —
  that is what CLV predicts, and books track it too.
- A real edge still loses over long stretches. The max-drawdown column exists
  for a reason.
- Bookmaker coverage is US/UK/EU via a licensed aggregator. **bet365 and
  SportyBet are not available through any legitimate API**, and this project
  deliberately does not scrape sportsbooks; use the price checker on the Best
  Odds tab for those. Betway and 1xBet do appear in the feed, but as their
  UK/EU books — a regional site can quote a different price, so confirm it
  there before staking.
- Exchange commission is netted at each exchange's standard rate (Betfair 5%,
  Smarkets and Matchbook 2%). If your account is on a different rate, the net
  price shown for those rows is off by the difference.
- Everything here is decision support with its uncertainty made visible. Stake
  only what you can afford to lose.

## References

- Shin, H. S. (1993). *Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims.* The Economic Journal, 103(420), 1141–1153.
- Štrumbelj, E. (2014). *On determining probability forecasts from betting odds.* International Journal of Forecasting, 30(4), 934–943.
- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* JRSS-C, 46(2), 265–280.
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.* Bell System Technical Journal.
- Baker, R. D., & McHale, I. G. (2013). *Optimal Betting Under Parameter Uncertainty.* (Kelly shrinkage.)
