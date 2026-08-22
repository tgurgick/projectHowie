"""Monte Carlo season simulation of a roster.

Samples every player's weekly points (availability x matchup-adjusted
truncated-normal outcome), sets the lineup each week by EXPECTED points among
players who are actually available (what a real manager knows: injuries and
byes, not final scores), and totals realized points of the chosen lineup for
weeks 1-17.

This is where bench value becomes real: a backup RB scores zero in the value
model until a starter misses time in a sampled season — then he starts.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..config import LeagueConfig
from .distributions import SimPlayer, truncation_factor
from .lineup import FLEX_ELIGIBLE

FANTASY_WEEKS = 17
PLAYOFF_WEEKS = (15, 16, 17)


@dataclass
class SimResult:
    mean: float
    std: float
    p10: float
    p90: float


def simulate_roster(
    players: List[SimPlayer],
    league: LeagueConfig,
    n_sims: int = 300,
    seed: int = 7,
    playoff_weight: float = 1.0,
) -> SimResult:
    if not players:
        return SimResult(0.0, 0.0, 0.0, 0.0)
    week_weight = np.ones(FANTASY_WEEKS)
    for w in PLAYOFF_WEEKS:
        week_weight[w - 1] = playoff_weight
    rng = np.random.default_rng(seed)
    n = len(players)

    mu = np.array([p.weekly_mu for p in players])           # (n,)
    cv = np.array([p.cv for p in players])
    p_play = np.array([p.p_play for p in players])
    sos = np.stack([p.sos_mult[:FANTASY_WEEKS] for p in players])  # (n, W)
    week_mu = mu[:, None] * sos                              # (n, W)
    for i, p in enumerate(players):
        if p.bye_week and p.bye_week <= FANTASY_WEEKS:
            week_mu[i, p.bye_week - 1] = 0.0

    slots = {
        "QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
        "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots,
    }
    positions = [p.position for p in players]

    # Divide sampling means by the truncation factor so clip-at-zero keeps
    # E[weekly points] == week_mu exactly
    trunc = np.array([truncation_factor(p.cv) for p in players])
    sample_mu = week_mu / trunc[:, None]
    sigma = np.array([p.season_sigma for p in players])

    totals = np.empty(n_sims)
    for s in range(n_sims):
        # one projection-error shock per player per season (see SEASON_SIGMA)
        shock = np.clip(rng.normal(1.0, sigma), 0.25, 2.2)
        mu_s = sample_mu * shock[:, None]
        available = rng.random((n, FANTASY_WEEKS)) < p_play[:, None]
        available &= week_mu > 0
        raw = rng.normal(mu_s, np.maximum(cv[:, None] * mu_s, 1e-9))
        scores = np.clip(raw, 0.0, None) * available

        season_total = 0.0
        for w in range(FANTASY_WEEKS):
            exp_w = np.where(available[:, w], week_mu[:, w], -1.0)
            season_total += week_weight[w] * _week_lineup_score(
                positions, exp_w, scores[:, w], slots, league)
        totals[s] = season_total

    return SimResult(
        mean=float(totals.mean()),
        std=float(totals.std()),
        p10=float(np.percentile(totals, 10)),
        p90=float(np.percentile(totals, 90)),
    )


def simulate_player_totals(player: SimPlayer, n_sims: int = 300, seed: int = 7) -> np.ndarray:
    """Sampled season totals for one player (no lineup context)."""
    rng = np.random.default_rng(seed)
    week_mu = player.weekly_mu * player.sos_mult[:FANTASY_WEEKS]
    if player.bye_week and player.bye_week <= FANTASY_WEEKS:
        week_mu = week_mu.copy()
        week_mu[player.bye_week - 1] = 0.0
    sample_mu = week_mu / truncation_factor(player.cv)
    shock = np.clip(rng.normal(1.0, player.season_sigma, size=(n_sims, 1)), 0.25, 2.2)
    mu_s = sample_mu[None, :] * shock
    available = rng.random((n_sims, FANTASY_WEEKS)) < player.p_play
    available &= week_mu[None, :] > 0
    raw = rng.normal(mu_s, np.maximum(player.cv * mu_s, 1e-9), size=(n_sims, FANTASY_WEEKS))
    return (np.clip(raw, 0.0, None) * available).sum(axis=1)


def _week_lineup_score(
    positions: List[str],
    expected: np.ndarray,
    realized: np.ndarray,
    slots: Dict[str, int],
    league: LeagueConfig,
) -> float:
    """Fill slots by expected points, score the realized points of the chosen."""
    order = np.argsort(-expected)
    used = set()
    remaining = dict(slots)
    flex_left = league.flex_slots
    score = 0.0
    for i in order:
        if expected[i] < 0:
            break  # everyone after is unavailable
        pos = positions[i]
        if remaining.get(pos, 0) > 0:
            remaining[pos] -= 1
            used.add(i)
            score += realized[i]
    for i in order:
        if flex_left <= 0 or expected[i] < 0:
            break
        if i not in used and positions[i] in FLEX_ELIGIBLE:
            flex_left -= 1
            score += realized[i]
    return score
