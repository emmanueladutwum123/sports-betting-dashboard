"""Best-effort recent-form lookup via TheSportsDB's free public test endpoint.

Team-name matching between the odds feed and TheSportsDB is approximate
(exact-string search only) — if a team can't be found, we show "N/A" rather
than guess. This is a secondary signal; the dashboard works fine without it.
"""
import requests

BASE = "https://www.thesportsdb.com/api/v1/json/3"
_TEAM_CACHE: dict = {}


def _search_team(name: str) -> dict | None:
    if name in _TEAM_CACHE:
        return _TEAM_CACHE[name]
    try:
        resp = requests.get(f"{BASE}/searchteams.php", params={"t": name}, timeout=8)
        resp.raise_for_status()
        teams = (resp.json() or {}).get("teams") or []
    except (requests.RequestException, ValueError):
        teams = []
    team = teams[0] if teams else None
    _TEAM_CACHE[name] = team
    return team


def get_form(team_name: str, n: int = 5) -> dict | None:
    """Returns {'score': 0-100, 'results': 'WWDLW', 'n': games_found} or None
    if the team can't be matched or has no recent results on file."""
    team = _search_team(team_name)
    if not team:
        return None
    team_id = team.get("idTeam")
    if not team_id:
        return None
    try:
        resp = requests.get(f"{BASE}/eventslast.php", params={"id": team_id}, timeout=8)
        resp.raise_for_status()
        events = (resp.json() or {}).get("results") or []
    except (requests.RequestException, ValueError):
        events = []
    if not events:
        return None

    points, results = 0, []
    for ev in events[:n]:
        try:
            home_score, away_score = int(ev["intHomeScore"]), int(ev["intAwayScore"])
        except (TypeError, ValueError, KeyError):
            continue
        is_home = (ev.get("strHomeTeam") or "").strip().lower() == team_name.strip().lower()
        team_score, opp_score = (home_score, away_score) if is_home else (away_score, home_score)
        if team_score > opp_score:
            points += 3
            results.append("W")
        elif team_score == opp_score:
            points += 1
            results.append("D")
        else:
            results.append("L")

    if not results:
        return None
    return {"score": round(points / (len(results) * 3) * 100), "results": "".join(results), "n": len(results)}
