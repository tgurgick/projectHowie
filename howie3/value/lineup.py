"""Optimal-starting-lineup value: the single objective everything optimizes.

A roster is worth the points of its best legal starting lineup. Bench players
contribute nothing here (deterministic view); their value appears once
outcome distributions arrive (a bench player's worth is the probability he
becomes a starter).
"""

from typing import Dict, List

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
