# Independent strategy and engineering audit — 15 November 2026

## Conclusion

There is no verified profitable strategy at this owner's actual SportyBet or
bet365 prices in the available record. The existing local ledger contained
**zero bets** when queried read-only during this audit. Reported losses cannot
be quantified without the owner's history. No historical ticket result was
invented or imported as an actual bet.

The recommendation is to stop cash recommendations and collect prospective
paper evidence at executable owner prices. This is a limitation of the
evidence, not proof that every possible strategy is necessarily unprofitable.
Market-based probability estimates are not known true probabilities; even a
positive estimated EV needs validation.

The handoff's phrase "no edge by construction" is stronger than the code
supports: a cross-book price can exceed a market-derived fair estimate. What
is verified is that the former ticket admitted negative estimated EV and that
neither its probability ranking nor the existing record established profit.

## Claims checked against the pre-audit source

| Finding | Verification / qualification | Resolution |
|---|---|---|
| Favourites could have negative EV | `src/daily.py` had `TICKET_MIN_EV=-0.02`, sorted eligible favourites by shrunk win probability | Retired ticket API returns no selections; cash ticket stake always zero |
| Negative/fair totals leans got stakes | Former `app.py` used `bankroll*TICKET_FLAT_STAKE` for a `lean` | Tip-sheet UI removed; legacy totals output carries zero cash stake |
| Ticket lacked liquidity gate | **Not confirmed**: `build_ticket` called `passes_quality_gate`, including `MIN_BOOKS=8` | Preserve this correction; new paper gate also requires fresh, complete distinct vectors |
| Calibration extended to unfitted leagues | `calibration_for` fell back to `SOCCER_H2H` for unknown soccer | Identity for unknown competitions; calibration off in live MarketView by default |
| Owner prices vanished on reload | Former manual price list used Streamlit session state | Submitted quotes/checks saved with immutable raw snapshots |
| Inaccessible books masqueraded as takeable prices | Blank account selection meant all feed books; ticket selected highest feed price | Feed is benchmark only; entry price must be manually supplied at a configured owner book |
| Logging existed without measurement | Ledger provided manual logging/settlement; zero local records | Every submitted assessment is saved; qualifying paper entries automatically recorded; exact-ID settlement worker |
| Opening page consumed quota | Former `load_events(...)` executed during top-level rendering | Login before storage/API access; paid requests only by button or due worker; persistent atomic budget |
| Hosted local storage was unreliable | Existing ledger docstring admitted ephemeral deployment storage | External PostgreSQL option, explicit local-storage warning, private full backup/restore |

Original tests passed: 216. Their success verified the former implementation,
not its profitability. Some old tests explicitly required constructing the
favourites ticket; they were replaced with regression checks that require
retirement and zero cash staking. Calibration math remains independently
testable through explicit research opt-in.

## Additional issues found

- Relative quote-lag detection can accept a board where *every* quote is old.
  The paper protocol checks absolute timestamps against UTC instead.
- A two-outcome soccer vector can omit Draw yet appear complete. Paper entry
  requires Home/Draw/Away for soccer, exactly two teams for NBA, and distinct
  bookmaker keys. Unknown/future timestamps and nonfinite odds are excluded.
- Historical calibration was applied to live quotes by default despite the
  different prediction horizon. It is now off, even in fitted leagues.
- Raw closing odds retain bookmaker margin. New evidence uses the de-vigged
  sharp probability and records source/timing; it does not confuse raw price
  ratios with fair closing expected return.
- Multiple prices on one match are dependent. The evidence screen aggregates
  CLV per event and displays missing-close coverage. It remains descriptive;
  season/day and market-wide dependence still need independent review.
- Treating identical CLV values as infinite statistical significance is unsafe.
  New evidence leaves a zero-variance t-statistic undefined and cannot pass it.
- CSV imports could duplicate history and were not atomic. New imports validate
  all rows before mutation and deduplicate stable IDs; no guessed result or P&L.
- A push still contributes settled turnover. New ROI includes pushes in turnover
  and excludes voids; paper units and actual GHS are never mixed.
- Personal ledgers should not be exposed with a public quota-funded dashboard.
  The new application requires a private password before accessing either.
- The former 08:00 UTC betting-day boundary did not match Ghana calendar dates.
  Compatibility helpers now use midnight UTC; UI shows explicit UTC kickoffs.

## Evaluation protocol and scope

See README and `src/paper.py` for the fixed `own-price-h2h-v1` protocol. It
records no bets until the owner's actual price is entered; no feed price is
substituted. It needs eight fresh books, a sharp reference, acceptable
dispersion, context confirmation and conservative estimated EV ≥1%.
Paper results use flat one-unit singles; all recommended cash stakes are zero.

The 300-event/60-day/90%-coverage/positive-CLV/t>2 screen is a predeclared
screening convention, not a theorem or guarantee. A separate untouched sample,
valid execution, costs, dependence, missingness and bookmaker rules must be
examined before any future decision about cash recommendations. The application
does not automatically unlock cash stakes even if that screen passes.

The historical +2.10% / 1,408-bet / t≈0.79 result and 36 model-vs-market tests
are claims in the existing README/backtest research, preserved in the archived
methodology. The audit verified that those claims are present and that the
existing tests pass; it **did not independently rerun the full historical data
collection/calibration/backtest**. They cannot be upgraded into proof of an
edge at the owner's prices. No live market or paid-feed requests were needed
to test these fixes.

## Deployment and remaining owner data

- Actual history, bankroll and minimum stake have not been supplied. Defaults
  explicitly represent an unknown bankroll; no losses were reconstructed.
- Hosted database credentials and hosted private access secrets must be set in
  the owner's accounts. The repository has no authority to invent them.
- Worker code and an optional five-minute GitHub workflow are provided. A
  scheduled job must actually run, with the same external database, before
  closing coverage can be expected. GitHub can delay cron runs; missing close
  data stays missing. The worker refuses an ephemeral hosted database fallback.
- Result retrieval has a three-day maximum provider window. Unsupported,
  abandoned, postponed, older or ambiguous records need manual verification.
- The new PostgreSQL path needs a configured server for an integration check;
  SQLite tests alone are not evidence of a successful hosted migration.
- A public repository must never receive owner bet history, private backups,
  actual database files, passwords or API keys.

CI includes a disposable PostgreSQL service and an integration test for journal
transactions, quota reservations, near-close capture and settlement. Its result
must be checked on the published branch; it does not validate an owner's
separately hosted database credentials or migration.

## Primary documentation checked

- [The Odds API v4](https://the-odds-api.com/liveapi/guides/v4/): score retrieval
  supports up to three days; historical odds are a separate endpoint, so late
  polls are not relabelled as closing observations.
- [Streamlit connections](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data):
  Community Cloud does not guarantee local file persistence.
- [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management):
  production credentials belong in hosted Secrets, not source control.
