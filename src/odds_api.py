"""Thin client for The Odds API (https://the-odds-api.com).

Only source of live/upcoming odds in this project. We never scrape
sportsbook websites directly (bet365, SportyBet, Betway, etc.) — that
violates their terms of service and breaks constantly. This wraps a
licensed odds aggregator instead. Its bookmaker coverage is US/UK/EU
books (FanDuel, DraftKings, Pinnacle, William Hill, Unibet, ...), NOT
bet365/SportyBet/Betway — see README for why.
"""
import os

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_GROUPS = ("Soccer", "Basketball", "Baseball", "Cricket")


class OddsAPIError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key or key == "your_key_here":
        raise OddsAPIError(
            "Missing ODDS_API_KEY. Get a free key at https://the-odds-api.com "
            "and put it in your .env file (see .env.example)."
        )
    return key


def _get(path: str, params: dict) -> tuple:
    """Returns (json_body, remaining_quota_str_or_None)."""
    params = {**params, "apiKey": _api_key()}
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    if resp.status_code in (401, 429):
        # The API confusingly returns 401 (not just 429) for a quota-exhausted
        # key, so check the body instead of trusting the status code alone.
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if body.get("error_code") == "OUT_OF_USAGE_CREDITS":
            raise OddsAPIError(
                "Odds API monthly quota exhausted (free tier: 500 credits/month). "
                "Wait for your monthly reset, upgrade at https://the-odds-api.com, "
                "or narrow Sports/Regions/Leagues in the sidebar to use less quota."
            )
        raise OddsAPIError("Odds API rejected the key — check ODDS_API_KEY in .env.")
    if resp.status_code == 422:
        # Unsupported sport/market/region combo for this endpoint — treat as empty.
        return [], resp.headers.get("x-requests-remaining")
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


def check_api_key() -> None:
    """Raises OddsAPIError with a helpful message if no key is configured."""
    _api_key()


def list_sports() -> list:
    """All sports (in-season and not). No quota cost."""
    data, _ = _get("/sports", {"all": "true"})
    return data


def in_season_sports_by_group() -> dict:
    """{'Soccer': [sport_dict, ...], 'Basketball': [...], ...} filtered to our 4 groups,
    active (in-season) only. Excludes outright/futures markets (championship winner,
    etc.) — those don't have h2h/totals odds, so pulling them just wastes quota."""
    grouped = {g: [] for g in SPORT_GROUPS}
    for s in list_sports():
        if s.get("active") and s.get("group") in grouped and not s.get("has_outrights"):
            grouped[s["group"]].append(s)
    return grouped


def get_odds(sport_key: str, regions: str = "eu", markets: str = "h2h,totals") -> tuple:
    """Upcoming/live pre-match odds for one sport key. Returns (events, quota_remaining)."""
    return _get(
        f"/sports/{sport_key}/odds",
        {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
    )


def get_additional_markets(sport_key: str, regions: str = "eu") -> tuple:
    """BTTS / draw-no-bet / double-chance — mostly soccer-only, often not on every
    plan/region. Any failure here is swallowed by the caller; treat as best-effort."""
    return _get(
        f"/sports/{sport_key}/odds",
        {
            "regions": regions,
            "markets": "btts,draw_no_bet,double_chance",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
    )


def get_scores(sport_key: str, days_from: int = 1) -> tuple:
    """Live + recently completed games (for the live scoreboard)."""
    return _get(f"/sports/{sport_key}/scores", {"daysFrom": days_from, "dateFormat": "iso"})
