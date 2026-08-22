"""Roster-conditional pick valuation.

Value of taking player X at your current pick = optimal-lineup points of the
roster you END the draft with, assuming you draft optimally afterward against
expected availability at each of your future picks.

The rollout is greedy over positions per future pick, pricing each position
at E[k-th best available] (k tracks how many of that position the plan has
already claimed from the market). Greedy is a lower bound on optimal play,
but it is the same bound for every candidate X, so the *ranking* is fair —
and it inherently encodes roster fit: a 4th WR only adds value through the
flex or by beating a current starter.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from ..config import LeagueConfig
from .board import POSITIONS, PoolPlayer, expected_kth_best
from .lineup import expected_lineup_points as lineup_points

if TYPE_CHECKING:  # simulate pulls in numpy; only needed here for the annotation
    from .simulate import SimResult


@dataclass
class PickPlan:
    """Result of evaluating one candidate at the current pick."""
    player: PoolPlayer
    final_value: float           # expected optimal-lineup points at draft end
    plan: List[Tuple[str, float]] = field(default_factory=list)  # (position, expected pts) per future pick
    sim: Optional["SimResult"] = None  # set once Monte Carlo reranking runs

    @property
    def plan_positions(self) -> List[str]:
        return [pos for pos, _ in self.plan]


KDST_RESERVE_PICKS = 2  # K/DST become candidates only in the last (open slots + 2) picks


def _rollout(
    roster_pts: Dict[str, List[float]],
    pools: Dict[str, List[PoolPlayer]],
    future_picks: Sequence[int],
    league: LeagueConfig,
    taken: frozenset,
) -> Tuple[float, List[Tuple[str, float]]]:
    """Greedily complete the draft; returns (final lineup value, plan)."""
    by_pos = {pos: list(pts) for pos, pts in roster_pts.items()}
    claims = {pos: 0 for pos in POSITIONS}
    plan: List[Tuple[str, float]] = []
    current = lineup_points(by_pos, league)
    for i, pick in enumerate(future_picks):
        next_pick = future_picks[i + 1] if i + 1 < len(future_picks) else None
        # Urgency greedy: value of taking the position now minus taking it at
        # the next pick instead. Flat positions (K/DST: replacement is free
        # all draft) defer to the end on their own; steep tiers get grabbed.
        best = None  # (urgency, gain_now, pos, pts)
        for pos in POSITIONS:
            pts_now = expected_kth_best(pools[pos], pick, claims[pos] + 1, taken)
            if pts_now <= 0:
                continue
            trial = dict(by_pos)
            trial[pos] = by_pos.get(pos, []) + [pts_now]
            gain_now = lineup_points(trial, league) - current
            if gain_now <= 0:
                continue
            if next_pick is not None:
                pts_later = expected_kth_best(pools[pos], next_pick, claims[pos] + 1, taken)
                trial[pos] = by_pos.get(pos, []) + [pts_later]
                gain_later = lineup_points(trial, league) - current
                urgency = gain_now - gain_later
            else:
                urgency = gain_now
            key = (urgency, gain_now)
            if best is None or key > (best[0], best[1]):
                best = (urgency, gain_now, pos, pts_now)
        if best is None:
            plan.append(("—", 0.0))
            continue
        _, gain_now, pos, pts = best
        by_pos[pos] = by_pos.get(pos, []) + [pts]
        claims[pos] += 1
        current += gain_now
        plan.append((pos, pts))
    return current, plan


def evaluate_candidates(
    pool: Sequence[PoolPlayer],
    roster: Sequence[PoolPlayer],
    current_pick: int,
    future_picks: Sequence[int],
    league: LeagueConfig,
    taken: frozenset = frozenset(),
    min_p_available: float = 0.10,
    top_n: int = 12,
) -> List[PickPlan]:
    """Rank the realistic candidates at the current pick by final roster value."""
    pools: Dict[str, List[PoolPlayer]] = {pos: [] for pos in POSITIONS}
    roster_uids = {p.uid for p in roster}
    for player in pool:
        if player.uid not in taken and player.uid not in roster_uids:
            pools[player.position].append(player)

    roster_pts: Dict[str, List[float]] = {}
    for p in roster:
        roster_pts.setdefault(p.position, []).append(p.proj)

    # Realistic candidates: available now with meaningful probability; cap the
    # field per position to keep the rollout cheap. Positional sanity caps stop
    # useless stockpiling (a 3rd QB or 2nd K can never start).
    pos_counts: Dict[str, int] = {}
    for p in roster:
        pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
    caps = {
        "QB": league.qb_slots + 1, "TE": league.te_slots + 1,
        "K": league.k_slots, "DST": league.dst_slots,
    }
    # K/DST: projections barely separate them and the engine cannot measure
    # them (the 2025 replay drops them), so they are a closing-rounds
    # decision by policy — considered only once the remaining picks are down
    # to the open K/DST slots plus a small reserve. Matches how the room
    # drafts (bots: round 11+) and keeps a round-9 kicker off the board.
    kdst_open = sum(max(caps[pos] - pos_counts.get(pos, 0), 0) for pos in ("K", "DST"))
    defer_kdst = len(future_picks) > kdst_open + KDST_RESERVE_PICKS
    candidates: List[PoolPlayer] = []
    for pos in POSITIONS:
        if pos in caps and pos_counts.get(pos, 0) >= caps[pos]:
            continue
        if defer_kdst and pos in ("K", "DST"):
            continue
        avail = [p for p in pools[pos] if p.p_available(current_pick) >= min_p_available]
        candidates.extend(avail[:8])

    results: List[PickPlan] = []
    for cand in candidates:
        take_pts = dict(roster_pts)
        take_pts[cand.position] = roster_pts.get(cand.position, []) + [cand.proj]
        pools_after = dict(pools)
        pools_after[cand.position] = [p for p in pools[cand.position] if p.uid != cand.uid]
        value, plan = _rollout(take_pts, pools_after, future_picks, league, taken)
        results.append(PickPlan(cand, value, plan))

    # Primary: expected final lineup points. Secondary (breaks the late-round
    # ties where every bench pick adds zero deterministic value): prefer the
    # bench with real insurance value — flex-eligible positions first, QBs
    # weighted up in multi-QB formats.
    bench_weight = {
        "RB": 1.0, "WR": 1.0, "TE": 0.6,
        "QB": 0.8 if league.qb_slots >= 2 else 0.35,
        "K": 0.0, "DST": 0.0,
    }
    results.sort(key=lambda r: (
        -round(r.final_value),
        -r.player.proj * bench_weight.get(r.player.position, 0.0),
    ))
    return results[:top_n]


def mc_rerank(
    conn,
    results: List[PickPlan],
    roster: Sequence[PoolPlayer],
    pool: Sequence[PoolPlayer],
    league: LeagueConfig,
    season: int,
    n_sims: int = 200,
) -> List[PickPlan]:
    """Re-rank candidate plans by Monte Carlo expected season lineup points,
    building every SimPlayer from the local db (see mc_rerank_with)."""
    from .distributions import build_sim_players, calibrate

    fmt = league.scoring_format
    buckets = calibrate(conn, fmt)
    games_by_uid = {
        r["player_uid"]: r["games"]
        for r in conn.execute(
            "SELECT player_uid, games FROM projections WHERE season = ? AND source = 'pff'",
            (season,),
        )
    }
    proj_rank = _proj_ranks(pool)

    def build(players: Sequence[PoolPlayer]):
        return build_sim_players(conn, list(players), season, fmt, proj_rank, games_by_uid)

    return mc_rerank_with(build, buckets, results, roster, pool, league, n_sims=n_sims)


def mc_rerank_with(
    build_sims,                 # Callable[[Sequence[PoolPlayer]], List[SimPlayer]]
    buckets: Dict,              # (pos, tier) -> Bucket-like with .cv/.p_play
    results: List[PickPlan],
    roster: Sequence[PoolPlayer],
    pool: Sequence[PoolPlayer],
    league: LeagueConfig,
    n_sims: int = 200,
) -> List[PickPlan]:
    """Source-agnostic Monte Carlo re-rank: the candidate and current roster
    are simulated as themselves (via build_sims — db-backed or from a
    strategy-context artifact); the rest of the plan is simulated as phantom
    players carrying the rollout's expected points, with variance and
    availability from the empirical bucket their projection would fall into.
    """
    import numpy as np

    from .distributions import SEASON_SIGMA, STATIC_BUCKETS, SimPlayer, tier_of
    from .simulate import simulate_roster

    pos_projs: Dict[str, List[float]] = {}
    for p in pool:
        pos_projs.setdefault(p.position, []).append(p.proj)

    def phantom(pos: str, pts: float) -> SimPlayer:
        rank = 1 + sum(1 for v in pos_projs.get(pos, []) if v > pts)
        if pos in STATIC_BUCKETS:
            cv, p_play = STATIC_BUCKETS[pos]
        else:
            b = buckets.get((pos, tier_of(pos, rank)))
            cv, p_play = (b.cv, b.p_play) if b else (0.5, 0.9)
        # Same calibration invariant as real players: E[season] == pts
        return SimPlayer(
            name=f"~{pos}{rank}", position=pos, proj=pts,
            weekly_mu=pts / (17.0 * p_play), cv=cv, p_play=p_play,
            bye_week=None, sos_mult=np.ones(18),
            season_sigma=SEASON_SIGMA.get(pos, 0.3),
        )

    for r in results:
        sim_players = list(build_sims(list(roster) + [r.player]))
        sim_players += [phantom(pos, pts) for pos, pts in r.plan if pos != "—" and pts > 0]
        r.sim = simulate_roster(sim_players, league, n_sims=n_sims,
                                playoff_weight=league.playoff_weight)

    results.sort(key=lambda r: -(r.sim.mean if r.sim else r.final_value))
    return results


def _proj_ranks(pool: Sequence[PoolPlayer]) -> Dict[str, int]:
    """uid -> projection rank within position (pool sorted by proj desc)."""
    proj_rank: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for p in pool:
        counts[p.position] = counts.get(p.position, 0) + 1
        proj_rank[p.uid] = counts[p.position]
    return proj_rank


def resolve_names(conn, names: Sequence[str], pool: Sequence[PoolPlayer]) -> Tuple[List[PoolPlayer], List[str]]:
    """Map user-typed names (or DST team codes) to pool players."""
    from ..data.names import name_key

    by_uid = {p.uid: p for p in pool}
    by_key: Dict[str, PoolPlayer] = {}
    for p in pool:
        by_key.setdefault(name_key(p.name), p)

    found, missing = [], []
    for raw in names:
        raw = raw.strip()
        if not raw:
            continue
        key = name_key(raw)
        hit: Optional[PoolPlayer] = by_key.get(key)
        if hit is None and len(raw) <= 3:  # bare team code -> that team's DST
            hit = by_uid.get(f"dst:{raw.upper()}")
        if hit is None and conn is not None:
            row = conn.execute(
                "SELECT player_uid FROM name_aliases WHERE name_key = ?", (key,)
            ).fetchone()
            if row:
                hit = by_uid.get(row["player_uid"])
        if hit:
            found.append(hit)
        else:
            missing.append(raw)
    return found, missing
