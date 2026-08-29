"""Compatibility shim plus the market-liquidity confidence rating.

The probability mathematics that used to live here now lives in :mod:`src.devig`
(five de-vig methods instead of one) and :mod:`src.market` (aggregation across
books). This module re-exports the old names so nothing breaks, and keeps the
confidence rating, which is about market structure rather than probability.
"""

from src.devig import devig, implied_prob  # noqa: F401  (re-exported)

__all__ = ["implied_prob", "devig", "confidence_stars", "fmt_stars"]


def confidence_stars(book_count: int) -> int:
    """1-5 stars from how many independent books quote the market.

    This rates the *estimate*, not the bet. More books quoting means a more
    liquid market and a tighter fair-value estimate. It says nothing about
    whether the selection wins, and a five-star rating on a -4% EV price is
    still a bad bet.
    """
    if book_count >= 8:
        return 5
    if book_count >= 6:
        return 4
    if book_count >= 4:
        return 3
    if book_count >= 2:
        return 2
    return 1


def fmt_stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)
