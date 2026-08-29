"""Bet log and closing-line-value tracking.

This is the most valuable module in the project and the least exciting, so it is
worth being explicit about why.

Settled profit is an almost useless feedback signal over any period you will
actually live through. At a realistic 2-3% edge, the standard deviation of
results swamps the mean for hundreds of bets: you can be genuinely good and down
money for a season, or genuinely bad and up. Waiting for P&L to tell you whether
your model works means waiting years and probably misreading the answer.

Closing line value is the way out. If you take a price and the market's sharp
closing price then moves *toward* your side, you bought something the market
subsequently agreed was underpriced. CLV is observable within hours of every bet
rather than after a season, has a far lower variance than results, and is the
metric sportsbooks themselves use to identify and limit winning accounts --
which is as strong an endorsement of its predictive power as exists.

So the workflow this module enforces is: log the price you took, fetch the
closing price at kickoff, and grade yourself on CLV continuously while results
accumulate in the background. A strategy showing consistent positive CLV over a
few hundred bets is far better evidence of an edge than a positive P&L over the
same sample.

Storage is a plain SQLite file. Note that on Streamlit Community Cloud the
filesystem is ephemeral and resets on redeploy, so the ledger there is a
scratchpad -- use the CSV export to keep anything you care about.
"""

from __future__ import annotations

import csv
import io
import math
import os
import sqlite3
from datetime import UTC, datetime

DEFAULT_DB = os.environ.get("BET_LEDGER_DB", "bets.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at       TEXT NOT NULL,
    event_id        TEXT,
    commence_time   TEXT,
    league          TEXT,
    home            TEXT,
    away            TEXT,
    market          TEXT NOT NULL,
    point           REAL,
    selection       TEXT NOT NULL,
    book            TEXT,
    odds_taken      REAL NOT NULL,
    stake           REAL NOT NULL,
    fair_prob       REAL,
    model_prob      REAL,
    ev_at_bet       REAL,
    closing_odds    REAL,
    closing_prob    REAL,
    result          TEXT,
    pnl             REAL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_bets_event ON bets(event_id);
CREATE INDEX IF NOT EXISTS idx_bets_open  ON bets(result) WHERE result IS NULL;
"""

_FIELDS = (
    "placed_at", "event_id", "commence_time", "league", "home", "away", "market",
    "point", "selection", "book", "odds_taken", "stake", "fair_prob", "model_prob",
    "ev_at_bet", "closing_odds", "closing_prob", "result", "pnl", "notes",
)


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def log_bet(conn: sqlite3.Connection, **kw) -> int:
    kw.setdefault("placed_at", datetime.now(UTC).isoformat())
    cols = [f for f in _FIELDS if f in kw]
    cur = conn.execute(
        f"INSERT INTO bets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [kw[c] for c in cols],
    )
    conn.commit()
    return cur.lastrowid


def settle(conn: sqlite3.Connection, bet_id: int, result: str) -> None:
    """Grade a bet. ``result`` is 'win', 'loss', 'push', or 'void'."""
    row = conn.execute("SELECT odds_taken, stake FROM bets WHERE id = ?", (bet_id,)).fetchone()
    if row is None:
        raise KeyError(f"no bet with id {bet_id}")
    pnl = {
        "win": row["stake"] * (row["odds_taken"] - 1.0),
        "loss": -row["stake"],
        "push": 0.0,
        "void": 0.0,
    }.get(result)
    if pnl is None:
        raise ValueError(f"unknown result {result!r}")
    conn.execute("UPDATE bets SET result = ?, pnl = ? WHERE id = ?", (result, pnl, bet_id))
    conn.commit()


def record_close(conn: sqlite3.Connection, bet_id: int, closing_odds: float,
                 closing_prob: float | None = None) -> None:
    """Attach the sharp closing price so CLV can be computed for this bet."""
    conn.execute(
        "UPDATE bets SET closing_odds = ?, closing_prob = ? WHERE id = ?",
        (closing_odds, closing_prob, bet_id),
    )
    conn.commit()


def open_bets(conn: sqlite3.Connection) -> list:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM bets WHERE result IS NULL ORDER BY commence_time"
    )]


def all_bets(conn: sqlite3.Connection) -> list:
    return [dict(r) for r in conn.execute("SELECT * FROM bets ORDER BY placed_at DESC")]


def clv(odds_taken: float, closing_odds: float) -> float:
    """Closing line value as a fraction: ``odds_taken / closing_odds - 1``.

    Positive means you got a better price than the close. Expressed on the price
    scale rather than the probability scale because that is what compounds
    directly into returns: +3% CLV is +3% on every unit you staked.
    """
    if not odds_taken or not closing_odds or closing_odds <= 1:
        return 0.0
    return odds_taken / closing_odds - 1.0


def performance(conn: sqlite3.Connection) -> dict:
    """Headline report: CLV first, because it converges long before P&L does."""
    bets = all_bets(conn)
    if not bets:
        return {"n": 0}

    with_close = [b for b in bets if b.get("closing_odds")]
    clvs = [clv(b["odds_taken"], b["closing_odds"]) for b in with_close]
    settled = [b for b in bets if b.get("result") in ("win", "loss")]

    out = {
        "n": len(bets),
        "n_open": sum(1 for b in bets if not b.get("result")),
        "n_settled": len(settled),
        # Pushes and voids are deliberately excluded from ROI and hit rate: no
        # money changed hands, so counting them dilutes both metrics toward zero
        # and understates a real edge. They are reported separately instead.
        "n_push_or_void": sum(1 for b in bets if b.get("result") in ("push", "void")),
        "n_with_closing_line": len(with_close),
    }

    if clvs:
        mean_clv = sum(clvs) / len(clvs)
        out["avg_clv"] = mean_clv
        out["clv_hit_rate"] = sum(1 for c in clvs if c > 0) / len(clvs)
        if len(clvs) > 1:
            var = sum((c - mean_clv) ** 2 for c in clvs) / (len(clvs) - 1)
            se = math.sqrt(var / len(clvs))
            if se > 0:
                out["clv_tstat"] = mean_clv / se
            elif mean_clv != 0:
                # Every bet showed identical CLV. Zero sample variance, so the
                # t-statistic is unbounded; report it as decisively significant
                # rather than as NaN, which the verdict would misread as "too
                # few bets" and silently suppress a real (if suspiciously
                # uniform) signal.
                out["clv_tstat"] = math.inf if mean_clv > 0 else -math.inf
            else:
                out["clv_tstat"] = 0.0
        # Interpretation thresholds from the CLV literature: sustained CLV above
        # ~1% is a strong indicator of long-run profitability, and above ~0.5%
        # over a large sample still points to a genuine edge. Both of those are
        # claims about a *sustained* average, so the verdict is gated on sample
        # size and significance -- a +2% CLV over five bets is not evidence of
        # anything, and saying otherwise is the exact overconfidence this
        # project exists to avoid.
        tstat = out.get("clv_tstat")
        if len(clvs) < 30 or tstat is None or not (tstat == tstat):
            out["clv_verdict"] = (
                f"too few bets to judge ({len(clvs)}/30 with a closing price recorded)"
            )
        elif abs(tstat) < 2.0:
            # Significance is about the magnitude of t, not its sign. A large
            # negative t is decisive evidence of *negative* CLV, which the
            # bettor most needs to hear, so only |t| < 2 is genuinely undecided.
            out["clv_verdict"] = (
                f"{'positive' if mean_clv > 0 else 'negative'} so far but not yet "
                f"significant (t={tstat:+.1f}, need |t|>2)"
            )
        elif mean_clv > 0.01:
            out["clv_verdict"] = "strong edge indicated"
        elif mean_clv > 0.005:
            out["clv_verdict"] = "positive edge indicated, keep going"
        elif mean_clv > 0:
            out["clv_verdict"] = "marginal edge -- may not survive costs"
        else:
            out["clv_verdict"] = "no edge -- the market moves against your bets"

    if settled:
        staked = sum(b["stake"] for b in settled)
        pnl = sum(b["pnl"] or 0.0 for b in settled)
        returns = [(b["pnl"] or 0.0) / b["stake"] for b in settled if b["stake"]]
        out.update({
            "staked": staked,
            "pnl": pnl,
            "roi": pnl / staked if staked else 0.0,
            "hit_rate": sum(1 for b in settled if b["result"] == "win") / len(settled),
        })
        if len(returns) > 1:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            se_r = math.sqrt(var_r / len(returns))
            out["roi_tstat"] = mean_r / se_r if se_r > 0 else float("nan")
            # How many bets before a real edge of this size becomes visible in
            # P&L at 2 standard errors. Usually in the thousands, which is the
            # whole argument for tracking CLV instead of waiting.
            out["bets_for_significance"] = (
                int(4.0 * var_r / mean_r**2) if mean_r > 0 else None
            )
    return out


def to_csv(conn: sqlite3.Connection) -> str:
    bets = all_bets(conn)
    if not bets:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(bets[0].keys()))
    writer.writeheader()
    for b in bets:
        writer.writerow(b)
    return buf.getvalue()


def from_csv(conn: sqlite3.Connection, text: str) -> int:
    """Restore a ledger from a CSV export. Returns rows imported."""
    count = 0
    for row in csv.DictReader(io.StringIO(text)):
        payload = {k: v for k, v in row.items() if k in _FIELDS and v not in ("", None)}
        for numeric in ("point", "odds_taken", "stake", "fair_prob", "model_prob",
                        "ev_at_bet", "closing_odds", "closing_prob", "pnl"):
            if numeric in payload:
                try:
                    payload[numeric] = float(payload[numeric])
                except ValueError:
                    payload.pop(numeric)
        if "odds_taken" in payload and "stake" in payload:
            log_bet(conn, **payload)
            count += 1
    return count
