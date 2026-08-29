#!/usr/bin/env python
"""Walk-forward backtest of Dixon-Coles against de-vigged closing prices.

    python scripts/run_backtest.py E0 2019,2020,2021,2022,2023
    python scripts/run_backtest.py --totals SP1 2020,2021,2022,2023

Reports the model, the market, and the optimal blend side by side, then
simulates a quarter-Kelly bankroll at both the Pinnacle close and the best
price available anywhere. The gap between those two columns is usually the most
actionable number in the output.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import simulate_bankroll, walk_forward, walk_forward_totals  # noqa: E402
from src.datafeeds.football_data import DIVISIONS, fetch_seasons  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("division", choices=sorted(DIVISIONS), help="football-data division code")
    ap.add_argument("seasons", help="comma-separated season start years, e.g. 2021,2022,2023")
    ap.add_argument("--totals", action="store_true", help="backtest Over/Under 2.5 instead of 1X2")
    ap.add_argument("--xi", type=float, default=0.0019, help="time-decay rate per day")
    ap.add_argument("--min-ev", type=float, default=0.02, help="minimum EV to place a bet")
    args = ap.parse_args()

    years = [int(y) for y in args.seasons.split(",")]
    rows = fetch_seasons(args.division, years)
    print(f"{DIVISIONS[args.division]}: {len(rows)} matches, seasons {years[0]}-{years[-1]}")
    if len(rows) < 500:
        print("Not enough history for a meaningful walk-forward.")
        return 1

    runner = walk_forward_totals if args.totals else walk_forward
    res = runner(rows, min_train=380, step=10, xi=args.xi)
    if not res.get("n"):
        print(res.get("error", "no evaluable matches"))
        return 1

    print(f"\nWalk-forward over {res['n']} matches "
          f"({'Over/Under 2.5' if args.totals else '1X2'}):")
    for name in ("model", "market", "blend"):
        block = res[name]
        extra = f"  weight={block['weight']}" if name == "blend" else ""
        label = "market (Pinnacle close)" if name == "market" else name
        print(f"  {label:24s} log-loss {block['log_loss']:.5f}  brier {block['brier']:.5f}{extra}")

    if res["blend"]["weight"] == 0:
        print("\n  -> Optimal model weight is ZERO: the closing line already contains "
              "everything\n     the model knows. Do not bet this market on the model.")

    if not args.totals:
        print("\nBankroll simulation (quarter-Kelly, min EV "
              f"{args.min_ev:.0%}, capped at 2% per bet):")
        for source in ("market", "model", "blend"):
            for price_key, label in (("psc", "Pinnacle"), ("maxc", "best available")):
                sim = simulate_bankroll(res, source=source, price_key=price_key,
                                        min_ev=args.min_ev)
                if not sim.get("n_bets"):
                    print(f"  {source:6s} @ {label:15s} no qualifying bets")
                    continue
                print(f"  {source:6s} @ {label:15s} bets {sim['n_bets']:5d}  "
                      f"ROI {sim['roi'] * 100:+6.2f}%  t={sim['roi_tstat']:+5.2f}  "
                      f"hit {sim['hit_rate'] * 100:4.1f}%  maxDD {sim['max_drawdown'] * 100:4.1f}%")
        print("\n  A |t| below 2 means the ROI cannot be distinguished from luck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
