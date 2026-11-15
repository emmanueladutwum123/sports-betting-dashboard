"""football-data.co.uk historical results and closing odds.

Free, open CSVs going back to the 1990s for the major European leagues. This is
the only source here that carries *closing* prices alongside results, which
makes it the only source that can answer the question that matters: would this
model have beaten the number the market settled on?

Column conventions used below (the file has many more):

==========  =================================================================
Date        dd/mm/yy or dd/mm/yyyy
FTHG, FTAG  full-time goals, home and away
HS, AS      shots, home and away
HST, AST    shots on target, home and away
PSCH/D/A    Pinnacle *closing* 1X2 prices -- the sharpest public number
B365C*      Bet365 closing
AvgC*       average closing across books
MaxC*       best closing price available anywhere (the price a line-shopper got)
PC>2.5      Pinnacle *closing* Over 2.5. Note the naming break: the 1X2 closing
            columns are PSCH/PSCD/PSCA but the totals ones are PC>2.5/PC<2.5,
            and "P>2.5" without the C is the OPENING price.
==========  =================================================================

Backtests use the closing price deliberately. Testing against opening prices
flatters a model enormously, because opening lines are soft and get corrected
by exactly the information a good model encodes. If a model cannot beat the
close, it has no edge -- it is merely rediscovering what the market already knew
by kickoff.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

import requests

BASE = "https://www.football-data.co.uk/mmz4281"

# division code -> human name. The Odds API league titles are mapped onto these.
DIVISIONS = {
    "E0": "English Premier League",
    "E1": "English Championship",
    "E2": "English League One",
    "E3": "English League Two",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda Division",
    "D1": "Germany Bundesliga",
    "D2": "Germany Bundesliga 2",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
    "B1": "Belgium First Div",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
    "SC0": "Scotland Premiership",
}

# Odds API sport keys -> football-data division codes.
SPORT_KEY_TO_DIVISION = {
    "soccer_epl": "E0",
    "soccer_efl_champ": "E1",
    "soccer_england_league1": "E2",
    "soccer_england_league2": "E3",
    "soccer_spain_la_liga": "SP1",
    "soccer_spain_segunda_division": "SP2",
    "soccer_germany_bundesliga": "D1",
    "soccer_germany_bundesliga2": "D2",
    "soccer_italy_serie_a": "I1",
    "soccer_italy_serie_b": "I2",
    "soccer_france_ligue_one": "F1",
    "soccer_france_ligue_two": "F2",
    "soccer_netherlands_eredivisie": "N1",
    "soccer_portugal_primeira_liga": "P1",
    "soccer_belgium_first_div": "B1",
    "soccer_turkey_super_league": "T1",
    "soccer_greece_super_league": "G1",
    "soccer_spl": "SC0",
}


class FootballDataError(RuntimeError):
    pass


def season_code(start_year: int) -> str:
    """2023 -> '2324' (the 2023/24 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _parse_date(raw: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def _f(row: dict, key: str):
    val = (row.get(key) or "").strip()
    try:
        out = float(val)
    except ValueError:
        return None
    return out if out > 1 else None


def _i(row: dict, key: str):
    """Integer count column (shots, cards). Unlike :func:`_f` these are counts,
    so 0 and 1 are perfectly valid values and must not be filtered out."""
    val = (row.get(key) or "").strip()
    try:
        return int(float(val))
    except ValueError:
        return None


def fetch_season(division: str, start_year: int, timeout: int = 25) -> list:
    """One season of one division as a list of row dicts.

    Rows carry results plus the closing price vectors that exist in the file.
    Missing odds columns come back as ``None`` rather than being dropped, so a
    caller can decide per-analysis which price series it requires.
    """
    url = f"{BASE}/{season_code(start_year)}/{division}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FootballDataError(f"could not fetch {url}: {exc}") from exc

    text = resp.content.decode("utf-8-sig", errors="replace")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        date = _parse_date(row.get("Date", ""))
        home, away = (row.get("HomeTeam") or "").strip(), (row.get("AwayTeam") or "").strip()
        if not date or not home or not away:
            continue
        try:
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
        except (KeyError, TypeError, ValueError):
            continue

        rows.append(
            {
                "date": date,
                "division": division,
                "season": start_year,
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "result": "home" if hg > ag else ("away" if ag > hg else "draw"),
                # Pinnacle closing -- the reference price.
                "psc": [_f(row, "PSCH"), _f(row, "PSCD"), _f(row, "PSCA")],
                # Market average and best-available closing.
                "avgc": [_f(row, "AvgCH"), _f(row, "AvgCD"), _f(row, "AvgCA")],
                "maxc": [_f(row, "MaxCH"), _f(row, "MaxCD"), _f(row, "MaxCA")],
                # Shots and shots on target. Cheap to carry and a materially
                # better strength signal than goals alone -- see
                # src/models/dixon_coles.py on the goals/shots mix.
                "home_shots": _i(row, "HS"),
                "away_shots": _i(row, "AS"),
                "home_sot": _i(row, "HST"),
                "away_sot": _i(row, "AST"),
                # Closing over/under 2.5. The Pinnacle *closing* columns are
                # "PC>2.5"/"PC<2.5"; "P>2.5" is the OPENING price. An earlier
                # version looked for a "PSCO>2.5" column that does not exist in
                # these files and silently fell through to the opening price,
                # which quietly turned the totals backtest into a test of how
                # fast the model is rather than how right it is.
                "ou25_psc": [_f(row, "PC>2.5"), _f(row, "PC<2.5")],
                "ou25_pso": [_f(row, "P>2.5"), _f(row, "P<2.5")],
                "ou25_avg": [_f(row, "AvgC>2.5"), _f(row, "AvgC<2.5")],
                "ou25_max": [_f(row, "MaxC>2.5"), _f(row, "MaxC<2.5")],
            }
        )
    return rows


def fetch_seasons(division: str, start_years: list) -> list:
    """Several seasons, chronologically ordered. Missing seasons are skipped."""
    out = []
    for year in sorted(start_years):
        try:
            out.extend(fetch_season(division, year))
        except FootballDataError:
            continue
    out.sort(key=lambda r: r["date"])
    return out


def to_matches(rows: list, count: str = "goals"):
    """Rows -> :class:`src.models.dixon_coles.Match` objects.

    ``count="sot"`` substitutes shots on target for goals, so the same
    Dixon-Coles machinery can be fitted to the shot process. Rows missing shot
    data are dropped rather than zero-filled: a zero-filled row is a claim that
    a team had no shots, which would drag its attack rating down for a reason
    that is purely an artefact of the file.
    """
    from src.models.dixon_coles import Match

    if count not in ("goals", "sot"):
        raise ValueError(f"count must be 'goals' or 'sot', got {count!r}")

    home_key, away_key = ("home_score", "away_score") if count == "goals" else ("home_sot", "away_sot")
    out = []
    for r in rows:
        home, away = r.get(home_key), r.get(away_key)
        if home is None or away is None:
            continue
        out.append(
            Match(
                date=r["date"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                home_score=home,
                away_score=away,
            )
        )
    return out


def shot_conversion(rows: list, default: float = 0.30) -> float:
    """League goals per shot on target over ``rows``.

    Converts a shots-on-target rate into a goal rate. Computed on the training
    window rather than hard-coded, because it varies by league and drifts over
    time (it is about 0.32 in the Eredivisie and 0.28 in Serie B).
    """
    goals = shots = 0
    for r in rows:
        if r.get("home_sot") is None or r.get("away_sot") is None:
            continue
        goals += r["home_score"] + r["away_score"]
        shots += r["home_sot"] + r["away_sot"]
    return goals / shots if shots else default
