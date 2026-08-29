"""Dixon-Coles bivariate Poisson goal model, self-contained for this dashboard.

Reference
---------
Dixon, M. J., & Coles, S. G. (1997). "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market." *Journal of the Royal
Statistical Society: Series C*, 46(2), 265-280.

Model
-----
For home team *i* against away team *j*::

    log lambda = attack_i + defence_j + gamma      (expected home goals)
    log mu     = attack_j + defence_i              (expected away goals)

Goals are *almost* independent Poisson. Independence is known to understate
low-scoring draws, so the Dixon-Coles ``tau`` correction rescales the four
low-score cells::

    tau(0,0) = 1 - lambda*mu*rho    tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho           tau(1,1) = 1 - rho

Time weighting: each match contributes ``exp(-xi * age_days)`` to the
likelihood. ``xi`` is a hyperparameter and must be chosen by out-of-sample
predictive likelihood, never by in-sample fit -- a larger ``xi`` trivially
lowers the weighted in-sample objective by discarding data.

Why bother, given the market is efficient?
------------------------------------------
Not to beat the 1X2 price. That market is the most liquid on the board and a
league-level goal model will not systematically beat a sharp closing line there.
The value is that fitting a *joint distribution over scorelines* prices every
derivative market at once -- totals at any line, both-teams-to-score, Asian
handicaps, correct score -- from a single coherent object. Those markets get a
fraction of the 1X2 market's liquidity and attention, and they are where a
correctly-specified model's disagreement with the book is most likely to be
information rather than noise. The model's numbers are also *internally
consistent* by construction: its 1X2, totals and BTTS prices cannot contradict
each other, whereas a book quoting all three independently can and does drift
out of line with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

_TAU_FLOOR = 1e-10
# Strength of the soft sum-to-zero constraint pinning the unidentified level of
# {attack} vs {defence}: adding c to every attack and subtracting it from every
# defence leaves all rates unchanged, so the likelihood alone cannot fix it.
_LEVEL_PENALTY = 1e3


@dataclass(frozen=True)
class Match:
    """One finished match. The minimum the model needs."""

    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int


def _tau_correction(x, y, lam, mu, rho):
    tau = np.ones_like(lam, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
    return tau


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles fit over one league's match history."""

    def __init__(self, max_goals: int = 10) -> None:
        self.max_goals = max_goals
        self.teams: list = []
        self._index: dict = {}
        self.attack = None
        self.defence = None
        self.home_advantage = 0.0
        self.rho = 0.0
        self.n_matches = 0
        self.converged = False

    # ---------------------------------------------------------------- fit
    def fit(self, matches: list, xi: float = 0.0019, reference_date: datetime | None = None):
        """Fit by time-weighted maximum likelihood.

        ``xi = 0.0019`` is roughly a one-year half-life, a reasonable prior for
        club soccer; tune it out-of-sample per league rather than trusting it.
        ``reference_date`` must be the *prediction* date during backtests, or
        the time weights leak information from the future.
        """
        if len(matches) < 20:
            raise ValueError(f"need at least 20 matches to fit, got {len(matches)}")

        self.teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.n_matches = len(matches)

        home_idx = np.array([self._index[m.home_team] for m in matches])
        away_idx = np.array([self._index[m.away_team] for m in matches])
        hg = np.array([m.home_score for m in matches], dtype=float)
        ag = np.array([m.away_score for m in matches], dtype=float)
        weights = self._time_weights(matches, xi, reference_date)

        lg_home, lg_away = gammaln(hg + 1.0), gammaln(ag + 1.0)
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0.0]])
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), (-0.5, 0.5)]

        def nll(params):
            attack, defence = params[:n], params[n : 2 * n]
            gamma, rho = params[2 * n], params[2 * n + 1]
            log_lam = attack[home_idx] + defence[away_idx] + gamma
            log_mu = attack[away_idx] + defence[home_idx]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            log_pois = hg * log_lam - lam - lg_home + ag * log_mu - mu - lg_away
            tau = _tau_correction(hg, ag, lam, mu, rho)
            ll = weights * (np.log(np.clip(tau, _TAU_FLOOR, None)) + log_pois)
            return -ll.sum() + _LEVEL_PENALTY * attack.mean() ** 2

        result = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        self.attack = result.x[:n]
        self.defence = result.x[n : 2 * n]
        self.home_advantage = float(result.x[2 * n])
        self.rho = float(result.x[2 * n + 1])
        self.converged = bool(result.success)
        return self

    @staticmethod
    def _time_weights(matches: list, xi: float, reference_date: datetime | None):
        if xi <= 0.0:
            return np.ones(len(matches))
        ref = reference_date or max(m.date for m in matches)
        ages = np.array([(ref - m.date).days for m in matches], dtype=float)
        return np.exp(-xi * np.clip(ages, 0.0, None))

    # ------------------------------------------------------------ predict
    def knows(self, *teams: str) -> bool:
        return all(t in self._index for t in teams)

    def expected_goals(self, home_team: str, away_team: str) -> tuple:
        if self.attack is None:
            raise RuntimeError("model not fitted")
        h, a = self._index[home_team], self._index[away_team]
        lam = float(np.exp(self.attack[h] + self.defence[a] + self.home_advantage))
        mu = float(np.exp(self.attack[a] + self.defence[h]))
        return lam, mu

    def score_matrix(self, home_team: str, away_team: str):
        """``M[x, y] = P(home scores x, away scores y)``, normalised."""
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho
        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

    # --- Derived markets. All read off the one score matrix, so they are
    # --- mutually consistent by construction.
    def match_odds(self, home_team: str, away_team: str) -> dict:
        m = self.score_matrix(home_team, away_team)
        return {
            "home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum()),
        }

    def totals(self, home_team: str, away_team: str, line: float) -> dict:
        """Over/Under at any line, including whole-number lines that can push."""
        m = self.score_matrix(home_team, away_team)
        idx = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        over = float(m[idx > line].sum())
        under = float(m[idx < line].sum())
        push = float(m[idx == line].sum())
        if push > 0:  # whole line: quote the conditional (stake-back) prices
            live = over + under
            return {"Over": over / live, "Under": under / live, "push": push}
        return {"Over": over, "Under": under, "push": 0.0}

    def btts(self, home_team: str, away_team: str) -> dict:
        m = self.score_matrix(home_team, away_team)
        no = float(m[0, :].sum() + m[:, 0].sum() - m[0, 0])
        return {"Yes": 1.0 - no, "No": no}

    def asian_handicap(self, home_team: str, away_team: str, handicap: float) -> dict:
        """Home team receives ``handicap`` goals. Pushes removed pro rata."""
        m = self.score_matrix(home_team, away_team)
        diff = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1])) + handicap
        home = float(m[diff > 0].sum())
        away = float(m[diff < 0].sum())
        live = home + away
        if live <= 0:
            return {"Home": 0.5, "Away": 0.5, "push": 1.0}
        return {"Home": home / live, "Away": away / live, "push": float(m[diff == 0].sum())}

    def correct_score(self, home_team: str, away_team: str, top: int = 8) -> list:
        m = self.score_matrix(home_team, away_team)
        cells = [(f"{x}-{y}", float(m[x, y])) for x in range(m.shape[0]) for y in range(m.shape[1])]
        cells.sort(key=lambda c: -c[1])
        return cells[:top]

    def team_ratings(self) -> list:
        """Attack/defence ratings, most dangerous first. A sanity check: if the
        top of this table is not roughly the league table, the fit is wrong and
        nothing downstream should be trusted."""
        if self.attack is None:
            return []
        rows = [
            {
                "team": t,
                "attack": round(float(self.attack[i]), 4),
                "defence": round(float(self.defence[i]), 4),
                "net": round(float(self.attack[i] - self.defence[i]), 4),
            }
            for i, t in enumerate(self.teams)
        ]
        rows.sort(key=lambda r: -r["net"])
        return rows
