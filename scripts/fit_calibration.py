"""Refit the residual calibration exponent against closing prices.

    python -m scripts.fit_calibration --seasons 2015 2024 --market h2h

Downloads football-data.co.uk closing prices for every division in
``DIVISIONS``, fits the sharpening exponent, and — the part that matters —
validates it the way it will actually be used: refit before each season on
prior seasons only, then scored on that season. An exponent fitted and scored
on the same matches will always look like an improvement; this one has to earn
it on matches it has never seen.

Prints the constant to paste into ``src/calibration.py`` together with the
evidence for it, so the number in the source is never further from its
justification than one command.
"""

from __future__ import annotations

import argparse
import math
import sys

from src.calibration import (
    SANE_EXPONENT_RANGE,
    SHRINKAGE_K,
    bias_above,
    fit_exponent,
    reliability,
    sharpen,
)
from src.datafeeds.football_data import DIVISIONS, FootballDataError, fetch_season
from src.devig import DEFAULT_METHOD, devig

OUTCOMES = ("home", "draw", "away")


def load_rows(divisions, start, end):
    rows = []
    for div in divisions:
        got = 0
        for year in range(start, end + 1):
            try:
                got += len(season := fetch_season(div, year))
            except FootballDataError:
                continue
            rows.extend(season)
        print(f"  {div:4s} {DIVISIONS.get(div, '?'):32s} {got:6d} matches", file=sys.stderr)
    return rows


def vectors(rows, market, method):
    """(probability vector, realised outcome, season, division) per usable match."""
    out = []
    for r in rows:
        if market == "h2h":
            prices, labels = r.get("psc"), OUTCOMES
            actual = r["result"]
        else:
            prices, labels = r.get("ou25_psc"), ("Over", "Under")
            actual = "Over" if (r["home_score"] + r["away_score"]) > 2.5 else "Under"
        if not prices or not all(prices):
            continue
        fair = devig(list(prices), method=method)
        if any(f is None for f in fair):
            continue
        out.append((dict(zip(labels, fair, strict=True)), actual, r["season"], r["division"]))
    return out


def log_loss(vecs, outcomes):
    return -sum(math.log(max(p[y], 1e-12)) for p, y in zip(vecs, outcomes, strict=True)) / len(vecs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2015, 2024], metavar=("FROM", "TO"))
    ap.add_argument("--market", choices=("h2h", "totals"), default="h2h")
    ap.add_argument("--devig", default=DEFAULT_METHOD)
    ap.add_argument("--divisions", nargs="*", default=sorted(DIVISIONS))
    args = ap.parse_args()

    print(f"fetching {args.seasons[0]}-{args.seasons[1]} ...", file=sys.stderr)
    rows = load_rows(args.divisions, *args.seasons)
    data = vectors(rows, args.market, args.devig)
    if len(data) < 1000:
        print(f"only {len(data)} usable markets — refusing to fit", file=sys.stderr)
        return 1

    probs = [d[0] for d in data]
    actuals = [d[1] for d in data]
    seasons = [d[2] for d in data]
    divisions = [d[3] for d in data]
    print(f"\n{len(data):,} scored {args.market} markets, de-vig={args.devig}\n")

    exponent = fit_exponent(probs, actuals)
    print(f"exponent fitted on everything: b = {exponent:.4f}")
    lo, hi = SANE_EXPONENT_RANGE
    if not lo <= exponent <= hi:
        print(f"  !! outside the sane range {SANE_EXPONENT_RANGE} — check the data, do not ship")
        return 1

    print("\nexpanding window (fit on prior seasons, score on the next):")
    raw_tot = cal_tot = 0.0
    n_tot = wins = tested = 0
    for season in sorted(set(seasons)):
        train = [(p, y) for p, y, s, _ in data if s < season]
        test = [(p, y) for p, y, s, _ in data if s == season]
        if len(train) < 3000 or not test:
            continue
        b = fit_exponent([p for p, _ in train], [y for _, y in train])
        tp, ty = [p for p, _ in test], [y for _, y in test]
        raw, cal = log_loss(tp, ty), log_loss([sharpen(p, b) for p in tp], ty)
        n = len(test)
        raw_tot += raw * n
        cal_tot += cal * n
        n_tot += n
        tested += 1
        wins += raw > cal
        print(f"  {season}/{(season + 1) % 100:02d}  n={n:6d}  raw {raw:.5f}  cal {cal:.5f}"
              f"  {raw - cal:+.5f}  (b={b:.4f})")
    if n_tot:
        print(f"  {'POOLED':9s} n={n_tot:6d}  raw {raw_tot / n_tot:.5f}  cal {cal_tot / n_tot:.5f}"
              f"  {(raw_tot - cal_tot) / n_tot:+.6f}   [{wins}/{tested} seasons improved]")

    print("\nreliability of the RAW de-vigged close:")
    print(f"  {'band':>10} {'n':>7} {'pred':>8} {'actual':>8} {'bias':>9} {'95% CI':>8}")
    for row in reliability(probs, actuals):
        flag = " *" if row["significant"] else ""
        print(f"  {row['band']:>10} {row['n']:7d} {row['predicted']:8.4f} {row['actual']:8.4f}"
              f" {row['bias']:+9.4f} {row['ci95']:8.4f}{flag}")

    print("\nbias in the bands a card selects from:")
    cal_probs = [sharpen(p, exponent) for p in probs]
    for threshold in (0.5, 0.6, 0.7, 0.8):
        before = bias_above(probs, actuals, threshold)
        after = bias_above(cal_probs, actuals, threshold)
        if not before.get("n"):
            continue
        print(f"  p>={threshold:.1f}  n={before['n']:6d}  bias {before['bias']:+.4f}"
              f" -> {after['bias']:+.4f}  (+/-{before['ci95']:.4f})")

    const = "SOCCER_EXPONENT" if args.market == "h2h" else "SOCCER_TOTALS_EXPONENT"
    print(f"\n--> {const} = {exponent:.4f}   # {len(data):,} markets, "
          f"{args.seasons[0]}/{(args.seasons[0] + 1) % 100:02d}-"
          f"{args.seasons[1]}/{(args.seasons[1] + 1) % 100:02d}")

    if args.market != "h2h":
        return 0

    # Per-league exponents, shrunk toward the global one. Printed ready to
    # paste: a refit that does not also update this table leaves the shipped
    # per-league numbers stale while the global one moves, which is worse than
    # not refitting at all.
    print(f"\nper-league, shrunk K={SHRINKAGE_K} toward the global exponent:")
    print("LEAGUE_EXPONENTS = {")
    for division in sorted(set(divisions)):
        idx = [i for i, d in enumerate(divisions) if d == division]
        if len(idx) < 800:
            print(f'    # "{division}" skipped: only {len(idx)} markets')
            continue
        raw = fit_exponent([probs[i] for i in idx], [actuals[i] for i in idx])
        shrunk = exponent + SHRINKAGE_K * (raw - exponent)
        name = DIVISIONS.get(division, "?")
        print(f'    "{division}": {shrunk:.4f},   # unshrunk {raw:.4f}, '
              f'n={len(idx)}   {name}')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
