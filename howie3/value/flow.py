"""Draft-flow simulation: what is actually still there at my next picks.

The analytic availability model (ADP as an independent normal curve) cannot
see the live board — a run in progress, a room that has ignored TE, the
specific players already gone. This rolls the draft forward from the current
state with the mock bots a few hundred times and records who survives to
each of the user's next `horizon` picks. The engine uses it as the
availability term for those picks (the 'what if I wait' half of its
urgency decision); the cockpit and the agent show it as a sequence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import LeagueConfig
from ..state import DraftState, snake_team_for_pick
from .board import PoolPlayer, snake_picks

RUN_WINDOW = 5


@dataclass
class FlowResult:
    picks: List[int]                                  # my upcoming picks covered
    avail: Dict[str, Dict[int, float]]                # uid -> {pick -> P(still there)}
    survivors: Dict[int, Dict[str, float]]            # pick -> {pos -> expected starter-tier players left}
    runs: Dict[str, int] = field(default_factory=dict)  # pos -> picks in the last RUN_WINDOW
    n: int = 0

    def p(self, uid: str, pick: int) -> Optional[float]:
        return self.avail.get(uid, {}).get(pick)


def draft_flow(pool: Sequence[PoolPlayer], state: DraftState, league: LeagueConfig,
               n: int = 300, horizon: int = 3, seed: Optional[int] = None,
               my_plan: Optional[Sequence[str]] = None) -> FlowResult:
    """Roll the draft forward `n` times from the live board.

    At the user's own intermediate picks the rollout assumes the planned
    position (or the best projected player) is taken, so availability at
    the SECOND and THIRD picks is conditioned on a realistic first pick."""
    from ..mock import bot_pick

    me = league.draft_position
    total = league.num_teams * league.roster_size
    start = state.next_pick_no()
    mine = [k for k in snake_picks(league) if k >= start]
    on_clock = bool(mine) and mine[0] == start
    targets = mine[1:1 + horizon] if on_clock else mine[:horizon]
    if not targets:
        return FlowResult(picks=[], avail={}, survivors={}, n=0)
    end = targets[-1]
    taken0 = set(state.taken_uids())
    tp0: Dict[int, Dict[str, int]] = {}
    for e in state.events:
        if e.team and e.position:
            tp0.setdefault(e.team, {})[e.position] = tp0.get(e.team, {}).get(e.position, 0) + 1
    recent0 = [e.position or "" for e in state.events[-RUN_WINDOW:]]
    runs = {pos: recent0.count(pos) for pos in set(recent0) if pos and recent0.count(pos) >= 2}
    draftable = [p for p in pool if p.draftable]
    by_proj = sorted(draftable, key=lambda p: -p.proj)
    counts: Dict[str, Dict[int, int]] = {}
    base = seed if seed is not None else (state.seed * 7919 + start)
    plan = list(my_plan or [])
    for r in range(n):
        rng = np.random.default_rng(base + r)
        taken = set(taken0)
        tp = {t: dict(v) for t, v in tp0.items()}
        recent = list(recent0)
        my_i = 0
        for pick_no in range(start, min(end, total) + 1):
            team = snake_team_for_pick(league, pick_no)
            if pick_no in targets:
                for p in by_proj:
                    if p.uid not in taken:
                        counts.setdefault(p.uid, {}).setdefault(pick_no, 0)
                        counts[p.uid][pick_no] += 1
                if pick_no == end:
                    break
            if team == me:
                # my own pick inside the horizon: take the planned position's best, else the best player
                want = plan[my_i] if my_i < len(plan) else None
                my_i += 1
                choice = next((p for p in by_proj if p.uid not in taken and (want is None or p.position == want)), None) \
                    or next((p for p in by_proj if p.uid not in taken), None)
            else:
                rnd = (pick_no - 1) // league.num_teams + 1
                choice = bot_pick(draftable, frozenset(taken), tp.setdefault(team, {}), rnd, league, rng,
                                  recent_positions=recent)
                if choice:
                    tp[team][choice.position] = tp[team].get(choice.position, 0) + 1
            if choice is None:
                break
            taken.add(choice.uid)
            recent = (recent + [choice.position])[-RUN_WINDOW:]
    avail = {uid: {k: c / n for k, c in per.items()} for uid, per in counts.items()}
    # starter-tier survivors per position: top (teams x slots [+ flex share]) by projection
    slots = {"QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots, "TE": league.te_slots}
    survivors: Dict[int, Dict[str, float]] = {}
    for k in targets:
        survivors[k] = {}
        for pos, n_slots in slots.items():
            tier_n = int(league.num_teams * (n_slots + (league.flex_slots / 3.0 if pos != "QB" else 0)))
            tier = [p for p in by_proj if p.position == pos][:tier_n]
            survivors[k][pos] = round(sum(avail.get(p.uid, {}).get(k, 0.0) for p in tier if p.uid not in taken0), 1)
    return FlowResult(picks=targets, avail=avail, survivors=survivors, runs=runs, n=n)


def attach(pool: Sequence[PoolPlayer], flow: FlowResult) -> None:
    """Give every pool player his conditioned availability for the horizon
    picks — PoolPlayer.p_available prefers it over the analytic model."""
    for p in pool:
        per = flow.avail.get(p.uid)
        if per:
            p.flow_avail = dict(per)
        elif flow.picks and p.draftable:
            p.flow_avail = {k: 0.0 for k in flow.picks}  # never survived a single rollout
