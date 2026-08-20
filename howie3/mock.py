"""Mock-draft opponents: ADP-noise bots for solo practice drafts.

Bots draft off market behavior, not the engine (that would be practicing
against yourself): each bot pick samples every available player's perceived
value as N(adp, stdev·1.25) and takes the best, under light roster sanity
(no early K/DST, positional caps). Deterministic per (seed, pick) so a mock
draft replays identically.
"""

from typing import Dict, List, Optional

import numpy as np

from .config import Settings
from .state import DraftState, snake_team_for_pick
from .value.board import PoolPlayer

NO_K_DST_BEFORE_ROUND = 11


def _caps(league) -> Dict[str, int]:
    """Positional stockpile caps derived from the league shape, so 2QB or
    superflex leagues get bots that match their format."""
    return {"QB": league.qb_slots + 1, "TE": league.te_slots + 1,
            "K": max(league.k_slots, 0), "DST": max(league.dst_slots, 0)}


def bot_pick(pool: List[PoolPlayer], taken: frozenset, team_positions: Dict[str, int],
             rnd: int, league, rng: np.random.Generator) -> Optional[PoolPlayer]:
    caps = _caps(league)
    roster_size = league.roster_size
    candidates = []
    n_picked = sum(team_positions.values())
    for p in pool:
        if p.uid in taken or p.adp is None:
            continue
        pos_count = team_positions.get(p.position, 0)
        if p.position in ("K", "DST"):
            if rnd < NO_K_DST_BEFORE_ROUND or pos_count >= caps[p.position]:
                continue
        if p.position == "QB":
            early_cap = league.qb_slots if rnd < 9 else caps["QB"]
            if pos_count >= early_cap:
                continue
        if p.position == "TE" and pos_count >= caps["TE"]:
            continue
        candidates.append(p)
    if not candidates:
        candidates = [p for p in pool if p.uid not in taken]
        if not candidates:
            return None

    picks_left = roster_size - n_picked
    missing = [pos for pos in ("K", "DST") if team_positions.get(pos, 0) == 0]
    if len(missing) >= picks_left and picks_left > 0:
        forced = [p for p in candidates if p.position in missing]
        if forced:
            candidates = forced

    window = sorted(candidates, key=lambda p: p.adp if p.adp else 999)[:24]
    perceived = [
        rng.normal(p.adp, max((p.stdev or 2.0), 2.0) * 1.25) for p in window
    ]
    return window[int(np.argmin(perceived))]


def advance_bots(settings: Settings, state: DraftState, pool: List[PoolPlayer],
                 seed: int = 11) -> List[dict]:
    """Let bots pick until the user is on the clock (or the draft ends).
    Returns the picks made."""
    league = settings.league
    me = league.draft_position
    total = league.num_teams * league.roster_size
    made = []
    while True:
        pick_no = state.next_pick_no()
        if pick_no > total:
            break
        team = snake_team_for_pick(league, pick_no)
        if team == me:
            break
        rnd = (pick_no - 1) // league.num_teams + 1
        team_positions: Dict[str, int] = {}
        for e in state.events:
            if e.team == team and e.position:
                team_positions[e.position] = team_positions.get(e.position, 0) + 1
        rng = np.random.default_rng(seed * 100_000 + pick_no)
        choice = bot_pick(pool, state.taken_uids(), team_positions, rnd,
                          league, rng)
        if choice is None:
            break
        event = state.add_pick(pick_no, team, choice.uid, choice.name,
                               choice.position, source="mock")
        made.append({"pick_no": event.pick_no, "team": team, "name": choice.name,
                     "position": choice.position})
    state.save(settings)
    return made
