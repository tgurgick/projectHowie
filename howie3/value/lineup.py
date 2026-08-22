"""Optimal-starting-lineup value: the single objective everything optimizes.

A roster is worth the points of its best legal starting lineup. Bench players
contribute nothing here (deterministic view); their value appears once
outcome distributions arrive (a bench player's worth is the probability he
becomes a starter).
"""

from typing import Dict, List, Optional

from ..config import LeagueConfig

FLEX_ELIGIBLE = ("RB", "WR", "TE")


def lineup_points(by_pos: Dict[str, List[float]], league: LeagueConfig) -> float:
    """Points of the optimal legal lineup. `by_pos` maps position -> player
    season points (any order). Greedy fill is exact here: dedicated slots take
    the best at each position, flex takes the best leftover among RB/WR/TE."""
    slots = {
        "QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
        "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots,
    }
    total = 0.0
    leftovers: List[float] = []
    for pos, n in slots.items():
        pts = sorted(by_pos.get(pos, []), reverse=True)
        total += sum(pts[:n])
        if pos in FLEX_ELIGIBLE:
            leftovers.extend(pts[n:])
    leftovers.sort(reverse=True)
    total += sum(leftovers[: league.flex_slots])
    return total


# Weekly availability by position (empirical 2018-2025 starter tiers, see
# distributions.calibrate) times the 16/17 bye factor. Position-level
# constants keep the objective cheap enough for the rollout's inner loop.
P_AVAILABLE = {"QB": 0.93 * 16 / 17, "RB": 0.88 * 16 / 17, "WR": 0.90 * 16 / 17,
               "TE": 0.90 * 16 / 17, "K": 0.96 * 16 / 17, "DST": 1.0 * 16 / 17}


def expected_lineup_points(by_pos: Dict[str, List[float]], league: LeagueConfig,
                           p_available: Optional[Dict[str, float]] = None) -> float:
    """Expected season points of the weekly-optimal lineup when every player
    is independently available each week with his position's probability.

    This is the deterministic objective WITH insurance: a starter counts in
    full (his projection already prices his own missed games); a backup
    counts his points times the probability the better players ahead of him
    leave a slot open that week. A lone QB on a 1-QB roster therefore
    scores a bye/injury hole, and a QB2 or RB4 earns real — not zero —
    value. Measured on the 2025 replay this closed the gap between the
    engine and a follow-ADP-with-need baseline (see evals tier C).

    Per position with k dedicated slots and players v1 >= v2 >= ... :
      P(i starts dedicated) = p * P(Binomial(i-1, p) < k)
    Flex: among the leftover RB/WR/TE, player i starts in flex with
      P = q_i * P(fewer than flex_slots better leftovers are present),
      q_i = p_i - P(i starts dedicated), present_j ~ Bernoulli(q_j)
    E[season] = sum_i v_i * (P_ded_i + P_flex_i) / p_i
    """
    from math import comb

    probs = p_available or P_AVAILABLE
    slots = {
        "QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
        "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots,
    }
    total = 0.0
    leftovers: List[tuple] = []  # (v, q, p) for flex competition
    for pos, k in slots.items():
        pts = sorted(by_pos.get(pos, []), reverse=True)
        if not pts:
            continue
        p = probs.get(pos, 0.9)
        tail = [1.0]  # tail[i] = P(Binomial(i, p) < k)
        for i in range(1, len(pts)):
            tail.append(sum(comb(i, j) * p ** j * (1 - p) ** (i - j) for j in range(min(k, i + 1))))
        for i, v in enumerate(pts):
            p_ded = p * tail[i]
            total += v * p_ded / p
            if pos in FLEX_ELIGIBLE and i >= k:
                leftovers.append((v, p - p_ded, p))
    if league.flex_slots and leftovers:
        leftovers.sort(key=lambda t: -t[0])
        # Poisson-binomial over better leftovers: dist[m] = P(m present)
        dist = [1.0]
        for v, q, p in leftovers:
            p_flex = q * sum(dist[: league.flex_slots])
            total += v * p_flex / p
            nxt = [0.0] * (len(dist) + 1)
            for m, pm in enumerate(dist):
                nxt[m] += pm * (1 - q)
                nxt[m + 1] += pm * q
            dist = nxt[: league.flex_slots + 1] + ([sum(nxt[league.flex_slots + 1:])] if len(nxt) > league.flex_slots + 1 else [])
    return total
