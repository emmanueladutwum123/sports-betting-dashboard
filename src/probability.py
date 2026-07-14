"""Pure math: implied probability, de-vig (overround removal), confidence rating.

No fabricated numbers here — every probability is derived directly from real
quoted odds. Confidence reflects market liquidity/agreement (how many books
quote it, how much they agree), not certainty of the outcome.
"""


def implied_prob(decimal_odds: float) -> float | None:
    if not decimal_odds or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds


def devig(decimal_odds_list: list) -> list:
    """Remove the bookmaker overround so probabilities sum to 1.

    decimal_odds_list: odds for mutually-exclusive outcomes of ONE market
    (e.g. [home, draw, away] or [over, under]).
    """
    raw = [implied_prob(o) for o in decimal_odds_list]
    total = sum(p for p in raw if p is not None)
    if not total:
        return [None] * len(decimal_odds_list)
    return [(p / total) if p is not None else None for p in raw]


def confidence_stars(book_count: int) -> int:
    """1-5 stars from how many independent bookmakers quote the market.
    More books quoting = more liquid/agreed-upon price = higher confidence
    in the probability estimate itself (NOT in the bet winning)."""
    if book_count >= 6:
        return 5
    if book_count >= 4:
        return 4
    if book_count >= 2:
        return 3
    if book_count == 1:
        return 2
    return 1


def fmt_stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)
