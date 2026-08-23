"""Draft-board math: expected best available and marginal value.

The core quantity is marginal value at a pick:

    MV(player X at pick k) = proj(X) - E[best projection at pos(X) at your next pick]

i.e. what you actually gain by taking X *now* instead of addressing the
position one pick later. A great player who will still be there next round
has low marginal value now; a mid tier-break player about to be swept has
high marginal value. This replaces static VORP as the ranking signal.
"""

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import LeagueConfig
from ..status import availability_factor, current_status
from .availability import p_available

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
EMPIRICAL_PRIOR_N = 30  # drafts at which the lab's rate and the ADP model weigh equally


@dataclass
class PoolPlayer:
    uid: str
    name: str
    position: str
    team: Optional[str]
    proj: float                 # the engine's value estimate (market-anchored)
    adp: Optional[float]
    stdev: Optional[float]
    bye: Optional[int]
    raw: Optional[float] = None  # the source projection, for display (None = proj is raw)
    # Empirical availability from the Mock Draft Lab: pick -> (rate, n_drafts).
    # Blended into the ADP model with weight n / (n + EMPIRICAL_PRIOR_N), so a
    # handful of drafts nudge and a large sample dominates.
    emp_avail: Optional[Dict[int, Tuple[float, int]]] = None
    # Current status row (howie3.status) — None when nothing is known.
    status: Optional[dict] = None

    @property
    def draftable(self) -> bool:
        return not (self.status and self.status["status"] in ("out_season", "released", "retired"))

    def p_available(self, pick: float) -> float:
        model = p_available(self.adp, self.stdev, pick)
        if self.emp_avail:
            emp = self.emp_avail.get(int(pick))
            if emp is not None and emp[1] > 0:
                rate, n = emp
                w = n / (n + EMPIRICAL_PRIOR_N)
                return w * rate + (1 - w) * model
        return model

    def availability_source(self, pick: float) -> str:
        emp = (self.emp_avail or {}).get(int(pick))
        return f"blend n={emp[1]}" if emp else "model"


def apply_market_anchor(pool: List[PoolPlayer], weight: float) -> List[PoolPlayer]:
    """Shrink projections toward market-implied value (winner's-curse control).

    For each position, the market-implied value of the k-th player by ADP is
    the k-th best projection at that position — the market's ordering priced
    on the projection scale. blended = (1-w)*proj + w*market_implied.
    Measured on the 2025 backtest: projection-only drafting systematically
    overweights proj-vs-market outliers, and the market wins those arguments
    more often than not.
    """
    if weight <= 0:
        return pool
    by_pos: dict = {}
    for p in pool:
        by_pos.setdefault(p.position, []).append(p)
    blended: List[PoolPlayer] = []
    for pos, plist in by_pos.items():
        proj_sorted = sorted((p.proj for p in plist), reverse=True)
        with_adp = sorted((p for p in plist if p.adp is not None), key=lambda p: p.adp)
        market = {p.uid: proj_sorted[min(i, len(proj_sorted) - 1)]
                  for i, p in enumerate(with_adp)}
        for p in plist:
            implied = market.get(p.uid)
            new_proj = (1 - weight) * p.proj + weight * implied if implied is not None else p.proj
            blended.append(PoolPlayer(p.uid, p.name, p.position, p.team,
                                      round(new_proj, 1), p.adp, p.stdev, p.bye,
                                      raw=p.raw if p.raw is not None else p.proj))
    blended.sort(key=lambda p: -p.proj)
    return blended


def load_pool(
    conn: sqlite3.Connection,
    season: int,
    fmt: str,
    proj_source: str = "pff",
    adp_source: str = "ffc",
    market_anchor: float = 0.75,
) -> List[PoolPlayer]:
    rows = conn.execute(
        f"""
        SELECT pr.player_uid, p.name, pr.position, pr.team,
               pr.pts_{fmt} AS proj, a.adp, a.stdev, a.bye_week
        FROM projections pr
        JOIN players p ON p.player_uid = pr.player_uid
        LEFT JOIN adp a ON a.player_uid = pr.player_uid
             AND a.season = pr.season AND a.source = ? AND a.format = ?
        WHERE pr.season = ? AND pr.source = ? AND pr.pts_{fmt} IS NOT NULL
        ORDER BY proj DESC
        """,
        (adp_source, fmt, season, proj_source),
    ).fetchall()
    pool = [
        PoolPlayer(
            r["player_uid"], r["name"], r["position"], r["team"],
            float(r["proj"]), r["adp"], r["stdev"], r["bye_week"],
        )
        for r in rows
        if r["position"] in POSITIONS
    ]
    pool = apply_market_anchor(pool, market_anchor)
    # Status is applied AFTER the anchor: a stale ADP must not pull an
    # injured or released player's value back up.
    return apply_status(pool, current_status(conn, season))


def apply_status(pool: List[PoolPlayer], statuses: Dict[str, dict]) -> List[PoolPlayer]:
    """Scale each player's engine value by his availability (games he will
    play × P(not cut)); out-for-season / released / retired go to zero and
    are excluded from candidates. `raw` keeps the source projection."""
    if not statuses:
        return pool
    for p in pool:
        row = statuses.get(p.uid)
        if row is None:
            continue
        p.status = row
        if p.raw is None:
            p.raw = p.proj
        p.proj = round(p.proj * availability_factor(row), 1)
    pool.sort(key=lambda p: -p.proj)
    return pool


def snake_picks(league: LeagueConfig, rounds: Optional[int] = None) -> List[int]:
    """Your overall pick numbers in a snake draft."""
    t, p = league.num_teams, league.draft_position
    rounds = rounds or league.roster_size
    return [
        (r - 1) * t + p if r % 2 == 1 else r * t - p + 1
        for r in range(1, rounds + 1)
    ]


def expected_kth_best(
    candidates: Sequence[PoolPlayer],
    pick: float,
    k: int = 1,
    taken: frozenset = frozenset(),
) -> float:
    """E[projection of the k-th best player still available at `pick`].

    k=1 is expected best available. Higher k matters when a draft plan takes
    the same position more than once: the second claim gets the second-best
    expected player, not the best again.

    Scans candidates in descending projection order with a DP over how many
    better players are still available. Availability is treated as
    independent across players — acceptable at this granularity because ADP
    already encodes market ordering.
    """
    # dp[j] = P(exactly j of the players scanned so far are available), j < k
    dp = [0.0] * k
    dp[0] = 1.0
    expected = 0.0
    tail = 1.0  # P(fewer than k better players available) upper-bound tracker
    for player in candidates:
        if player.uid in taken:
            continue
        p_here = player.p_available(pick)
        # player is the k-th best available iff they survive and exactly k-1
        # better players survived
        expected += player.proj * p_here * dp[k - 1]
        for j in range(k - 1, 0, -1):
            dp[j] = dp[j] * (1.0 - p_here) + dp[j - 1] * p_here
        dp[0] *= 1.0 - p_here
        tail = sum(dp)
        if tail < 1e-6:
            break
    return expected


def expected_best_available(
    candidates: Sequence[PoolPlayer], pick: float, taken: frozenset = frozenset()
) -> float:
    return expected_kth_best(candidates, pick, 1, taken)


def marginal_values(
    pool: Sequence[PoolPlayer],
    current_pick: int,
    next_pick: int,
    taken: frozenset = frozenset(),
    top_n: int = 5,
) -> Dict[str, List[dict]]:
    """Per position: top available players at current_pick with their marginal
    value over waiting until next_pick."""
    by_pos: Dict[str, List[PoolPlayer]] = {pos: [] for pos in POSITIONS}
    for player in pool:
        if player.uid not in taken:
            by_pos[player.position].append(player)

    out: Dict[str, List[dict]] = {}
    for pos, candidates in by_pos.items():
        eba_next = expected_best_available(candidates, next_pick, taken)
        rows = []
        # Only show players with a real chance of being there at this pick
        likely = [p for p in candidates if p.p_available(current_pick) >= 0.10]
        for player in likely[:top_n]:
            rows.append(
                {
                    "player": player,
                    "p_now": player.p_available(current_pick),
                    "p_next": player.p_available(next_pick),
                    "eba_next": eba_next,
                    "marginal": player.proj - eba_next,
                }
            )
        out[pos] = rows
    return out
