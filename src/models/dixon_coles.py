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

This module used to claim that *totals* were the exception -- thinner market,
less attention, and the model's Over/Under price falls out of the same fitted
score matrix for free, so that had to be where a correct model earned its keep.
That claim has now been tested properly and **it is false.** Over 10,876
Over/Under 2.5 markets in 6 leagues, walk-forward, scored against the de-vigged
Pinnacle *closing* price:

==========================  =========
market (Pinnacle close)      0.672048
model, goals only            0.687644
model, best goals/shot mix   0.681384
==========================  =========

and with the blend weight fitted on the first half and scored on the second --
the only honest way to fit it -- adding the model to the market *loses*
0.00062 +/- 0.00050 nats. That is a real loss, not a wash.

Worth being exact about what was wrong before, because the headline conclusion
did not change: the model lost on totals in the old table too. What the old
table could not support was any *weight*. The feed looked for a Pinnacle
closing column named ``PSCO>2.5``, which does not exist in these files, and
fell through to ``P>2.5`` -- the **opening** price. Every totals number in the
README was therefore scored against a soft line, which is the exact mistake
:mod:`src.backtest` warns about in its own docstring. The three small non-zero
totals weights it produced were fitted in-sample against that soft line; fit
out-of-sample against the real close, they go to zero and the blend loses.

So what is the model actually for? Description, not betting. It answers "how
strong is this team, and what does the scoreline distribution look like" with a
single coherent object whose 1X2, totals, BTTS and correct-score numbers cannot
contradict each other -- unlike a book quoting all three independently. That is
worth having. It is not worth staking on, and nothing here should be blended
into a price at a weight the data has not earned.
"""

from __future__ import annotations

import math
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


# --------------------------------------------------------------- shot-augmented

# Fraction of the team-strength estimate taken from the shots-on-target fit
# rather than the goals fit. Chosen by walk-forward log loss over 20,368
# predictions in 6 leagues (2015-2024), refitting every 20 matches:
#
#     goals only      1.003511
#     25% shots       1.000200
#     50% shots       0.999340   <- best
#     75% shots       1.001021
#     shots only      1.005441
#
# The shape is what you would expect if the two signals carry roughly equal
# amounts of independent information: a flat optimum in the middle, and a clear
# penalty at either extreme. Shots on target are a better-sampled measure of
# territorial control than goals (about 9 per match versus 2.7), so they
# estimate team strength with less noise; goals are what actually gets paid, so
# they carry finishing ability that shots throw away. Neither dominates.
SHOT_WEIGHT = 0.5


class ShotAugmentedDixonColes:
    """Dixon-Coles fitted twice -- on goals and on shots on target -- and mixed.

    The two fits are combined in *log-rate* space::

        log lambda = (1 - w) * log lambda_goals + w * (log lambda_sot + log c)

    where ``c`` is the league's goals-per-shot-on-target conversion rate over
    the training window. Log space, not linear, because these are multiplicative
    rate parameters: a team twice as dangerous should come out twice as
    dangerous under either measure, and a linear average of rates would let the
    larger-magnitude shot numbers dominate the mix regardless of ``w``.

    The low-score correction ``rho`` is taken from the goals fit alone. It
    describes the dependence between the two teams' *goal* counts, which is what
    the score matrix is over; the equivalent parameter for shots is a different
    quantity and mixing them would be a category error.

    Read-side API matches :class:`DixonColesModel` by delegation, so this is a
    drop-in wherever a fitted model is consumed.

    This is a **better model, not a profitable one.** Mixing in shots lowers
    standalone log loss on 1X2 from 1.003511 to 0.999340 and on Over/Under 2.5
    from 0.687644 to 0.681384 -- real improvements, roughly forty times the
    size of the residual calibration effect in :mod:`src.calibration`. The
    market still wins both, the fitted blend weight against a sharp close is
    still zero on 1X2, and on totals the blend is a significant loss. Better
    ratings and better scoreline distributions; no edge.
    """

    def __init__(self, max_goals: int = 10, shot_weight: float = SHOT_WEIGHT) -> None:
        if not 0.0 <= shot_weight <= 1.0:
            raise ValueError(f"shot_weight must be in [0, 1], got {shot_weight}")
        self.max_goals = max_goals
        self.shot_weight = shot_weight
        self.goals = DixonColesModel(max_goals)
        self.shots = None
        self.conversion = 0.0

    def fit(self, rows: list, xi: float = 0.0019, reference_date: datetime | None = None):
        """Fit from feed *rows* (dicts), not ``Match`` objects.

        Rows are needed rather than matches because the shot fit and the
        conversion rate both read columns that a ``Match`` does not carry. If
        too few rows have shot data the shot fit is skipped and this degrades
        silently to a plain goals model -- which is the right failure, since a
        shot fit on 30 matches is worse than no shot fit at all.
        """
        from src.datafeeds.football_data import shot_conversion, to_matches

        self.goals.fit(to_matches(rows, "goals"), xi=xi, reference_date=reference_date)

        shot_matches = to_matches(rows, "sot")
        if self.shot_weight > 0.0 and len(shot_matches) >= 20:
            self.conversion = shot_conversion(rows)
            try:
                self.shots = DixonColesModel(self.max_goals).fit(
                    shot_matches, xi=xi, reference_date=reference_date
                )
            except (ValueError, RuntimeError):
                self.shots = None
        else:
            self.shots = None
        return self

    # ------------------------------------------------------------ delegation
    @property
    def teams(self) -> list:
        return self.goals.teams

    @property
    def rho(self) -> float:
        return self.goals.rho

    @property
    def home_advantage(self) -> float:
        return self.goals.home_advantage

    @property
    def n_matches(self) -> int:
        return self.goals.n_matches

    @property
    def converged(self) -> bool:
        return self.goals.converged

    def knows(self, *teams: str) -> bool:
        return self.goals.knows(*teams)

    def _uses_shots(self, home_team: str, away_team: str) -> bool:
        return (
            self.shots is not None
            and self.conversion > 0.0
            and self.shot_weight > 0.0
            and self.shots.knows(home_team, away_team)
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple:
        lam_g, mu_g = self.goals.expected_goals(home_team, away_team)
        if not self._uses_shots(home_team, away_team):
            return lam_g, mu_g
        lam_s, mu_s = self.shots.expected_goals(home_team, away_team)
        w, c = self.shot_weight, self.conversion
        lam = math.exp((1.0 - w) * math.log(lam_g) + w * math.log(lam_s * c))
        mu = math.exp((1.0 - w) * math.log(mu_g) + w * math.log(mu_s * c))
        return lam, mu

    def score_matrix(self, home_team: str, away_team: str):
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        rho = self.rho
        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho
        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

    # The derived markets on DixonColesModel are pure functions of
    # score_matrix, so borrowing the implementations keeps the two models
    # guaranteed consistent instead of merely intended to be.
    match_odds = DixonColesModel.match_odds
    totals = DixonColesModel.totals
    btts = DixonColesModel.btts
    asian_handicap = DixonColesModel.asian_handicap
    correct_score = DixonColesModel.correct_score

    def team_ratings(self) -> list:
        """Ratings on the mixed scale, so the table matches what is predicted."""
        if self.goals.attack is None:
            return []
        if self.shots is None or self.conversion <= 0.0:
            return self.goals.team_ratings()
        w = self.shot_weight
        rows = []
        for team in self.goals.teams:
            gi = self.goals._index[team]
            attack = float(self.goals.attack[gi])
            defence = float(self.goals.defence[gi])
            if self.shots.knows(team):
                si = self.shots._index[team]
                # The conversion factor is a league-wide level shift; it cancels
                # out of any comparison between teams, so it is left out here
                # and applied only where an absolute rate is needed.
                attack = (1.0 - w) * attack + w * float(self.shots.attack[si])
                defence = (1.0 - w) * defence + w * float(self.shots.defence[si])
            rows.append({
                "team": team,
                "attack": round(attack, 4),
                "defence": round(defence, 4),
                "net": round(attack - defence, 4),
            })
        rows.sort(key=lambda r: -r["net"])
        return rows
