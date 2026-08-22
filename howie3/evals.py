"""Backtests against realized 2025 results — the measurement layer.

Three tiers, run with `howie eval run`:
  A. Inputs   — how good were 2025 preseason projections and ADP at all?
  B. Calibration — did simulated p10-p90 bands cover realized outcomes?
  C. Policy   — replay 2025 drafts: Howie's marginal-value policy vs
                pure-projection, static-VORP, ADP+need and follow-ADP
                baselines, scored with realized weekly points and weekly
                optimal lineups. Paired design (same seeded opponents per
                draft slot and rep for every policy) with bootstrap CIs on
                the mean paired difference vs follow-ADP.

2025 preseason inputs come from the legacy database (PPR-scored projections);
all policies draft from the SAME inputs, so tier C's comparison is fair even
though realized scoring is half-PPR. Buckets for tier B are calibrated on
seasons <= 2024 only (no leakage).
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import Settings
from .db import connect
from .data.names import fix_position, fix_team, name_key
from .value.board import PoolPlayer

EVAL_SEASON = 2025
FANTASY_WEEKS = 17
POOL_TOP = {"QB": 28, "RB": 55, "WR": 70, "TE": 26, "K": 20, "DST": 24}


@dataclass
class EvalPlayer:
    uid: str
    name: str
    position: str
    proj: float
    games_proj: float
    adp: Optional[float]
    actual_total: float
    actual_weeks: Dict[int, float]

    @property
    def stdev(self) -> float:
        # FantasyPros 2025 export has no per-player spread; use the empirical
        # rule that spread grows with ADP (early picks are consensus).
        return 1.0 + 0.09 * (self.adp or 200.0)


# ------------------------------------------------------------ data assembly

def load_eval_players(settings: Settings) -> List[EvalPlayer]:
    legacy = settings.data_dir / "fantasy_ppr.db"
    if not legacy.exists():
        raise FileNotFoundError("Legacy fantasy_ppr.db with 2025 preseason inputs not found")
    conn = connect(settings.db_path)
    lconn = sqlite3.connect(f"file:{legacy}?mode=ro", uri=True)
    lconn.row_factory = sqlite3.Row

    fmt = settings.league.scoring_format
    actual_rows = conn.execute(
        f"SELECT player_uid, week, pts_{fmt} AS pts FROM weekly_stats "
        "WHERE season = ? AND week <= ?", (EVAL_SEASON, FANTASY_WEEKS)).fetchall()
    weeks: Dict[str, Dict[int, float]] = {}
    for r in actual_rows:
        weeks.setdefault(r["player_uid"], {})[r["week"]] = r["pts"] or 0.0

    adp = {}
    for r in lconn.execute(
        "SELECT player_name, position, avg_adp FROM adp_data "
        "WHERE season = ? AND scoring_format = 'ppr'", (EVAL_SEASON,)):
        adp[(name_key(r["player_name"]), fix_position(r["position"]))] = r["avg_adp"]

    out: List[EvalPlayer] = []
    seen = set()
    for r in lconn.execute(
        "SELECT player_name, position, team_name, games, "
        "pass_yds, pass_td, pass_int, rush_yds, rush_td, "
        "recv_receptions, recv_yds, recv_td, fumbles_lost, two_pt "
        "FROM player_projections "
        "WHERE season = ? AND projection_type = 'preseason'", (EVAL_SEASON,)):
        pos = fix_position(r["position"])
        if pos not in POOL_TOP or pos in ("K", "DST"):
            continue  # offense only: K/DST scoring differs by provider
        from .data.names import resolve_uid
        uid = resolve_uid(conn, r["player_name"], pos, fix_team(r["team_name"]))
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        w = weeks.get(uid, {})
        g = lambda k: float(r[k] or 0.0)
        # rescore the preseason stat lines under the LEAGUE's scoring rules
        # (shared constants from data/scoring.py) so projections and realized
        # points share one scale for any format
        from .data.scoring import BASE, RECEPTION_VALUE

        proj_fmt = (
            BASE["pass_yds"] * g("pass_yds") + BASE["pass_td"] * g("pass_td")
            + BASE["pass_int"] * g("pass_int")
            + BASE["rush_yds"] * g("rush_yds") + BASE["rush_td"] * g("rush_td")
            + BASE["rec_yds"] * g("recv_yds") + BASE["rec_td"] * g("recv_td")
            + RECEPTION_VALUE[fmt] * g("recv_receptions")
            + BASE["fumble_lost"] * g("fumbles_lost") + BASE["two_pt"] * g("two_pt")
        )
        out.append(EvalPlayer(
            uid=uid, name=r["player_name"], position=pos,
            proj=round(proj_fmt, 1),
            games_proj=float(r["games"] or 16.0),
            adp=adp.get((name_key(r["player_name"]), pos)),
            actual_total=round(sum(w.values()), 1),
            actual_weeks=w,
        ))
    lconn.close()
    conn.close()
    out.sort(key=lambda p: -p.proj)
    return out


def _top_pool(players: List[EvalPlayer]) -> List[EvalPlayer]:
    by_pos: Dict[str, int] = {}
    pool = []
    for p in players:
        by_pos[p.position] = by_pos.get(p.position, 0) + 1
        if by_pos[p.position] <= POOL_TOP.get(p.position, 0):
            pool.append(p)
    return pool


# ------------------------------------------------------------ tier A: inputs

def eval_inputs_report(players: List[EvalPlayer]) -> List[dict]:
    out = []
    pool = _top_pool(players)
    for pos in ("QB", "RB", "WR", "TE"):
        rows = [p for p in pool if p.position == pos]
        if len(rows) < 8:
            continue
        proj = np.array([p.proj for p in rows])
        act = np.array([p.actual_total for p in rows])
        mae = float(np.mean(np.abs(proj - act)))
        rank_corr = _spearman(proj, act)
        out.append({"pos": pos, "n": len(rows), "proj_mae": round(mae, 1),
                    "rank_corr": round(rank_corr, 3)})
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denominator = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denominator) if denominator else 0.0


# ------------------------------------------------------------ tier B: calibration

def eval_calibration(settings: Settings, players: List[EvalPlayer],
                     n_sims: int = 400) -> dict:
    from .value.distributions import SEASON_SIGMA, SimPlayer, calibrate, tier_of
    from .value.simulate import simulate_player_totals

    conn = connect(settings.db_path)
    buckets = calibrate(conn, settings.league.scoring_format, max_season=EVAL_SEASON - 1)
    conn.close()

    pool = _top_pool(players)
    rank_in_pos: Dict[str, int] = {}
    inside = inside_healthy = total = total_healthy = 0
    for p in pool:
        rank_in_pos[p.position] = rank_in_pos.get(p.position, 0) + 1
        b = buckets.get((p.position, tier_of(p.position, rank_in_pos[p.position])))
        cv, p_play = (b.cv, b.p_play) if b else (0.5, 0.9)
        p_play = min((p.games_proj / 17.0 + p_play) / 2.0, 1.0)
        # no bye in this SimPlayer, so 17 playable weeks — matches the
        # simulator's FANTASY_WEEKS and keeps E[sim total] == proj
        sp = SimPlayer(name=p.name, position=p.position, proj=p.proj,
                       weekly_mu=p.proj / (17 * p_play), cv=cv, p_play=p_play,
                       bye_week=None, sos_mult=np.ones(18),
                       season_sigma=SEASON_SIGMA.get(p.position, 0.3))
        totals = simulate_player_totals(sp, n_sims=n_sims, seed=13)
        lo, hi = np.percentile(totals, 10), np.percentile(totals, 90)
        total += 1
        hit = lo <= p.actual_total <= hi
        inside += hit
        if len(p.actual_weeks) >= 8:
            total_healthy += 1
            inside_healthy += hit
    return {
        "coverage_all": round(inside / max(total, 1), 3),
        "coverage_8plus_games": round(inside_healthy / max(total_healthy, 1), 3),
        "target": 0.80, "n": total,
    }


# ------------------------------------------------------------ tier C: policy replay

def _to_pool_player(p: EvalPlayer) -> PoolPlayer:
    return PoolPlayer(uid=p.uid, name=p.name, position=p.position, team=None,
                      proj=p.proj, adp=p.adp, stdev=p.stdev, bye=None)


def _score_roster(players: List[EvalPlayer], league) -> float:
    """Realized weekly-optimal-lineup points: lineups chosen by expectation
    (proj/17, only among those who actually played that week), scored realized."""
    from .value.simulate import _week_lineup_score

    slots = {"QB": league.qb_slots, "RB": league.rb_slots, "WR": league.wr_slots,
             "TE": league.te_slots, "K": league.k_slots, "DST": league.dst_slots}
    positions = [p.position for p in players]
    total = 0.0
    for week in range(1, FANTASY_WEEKS + 1):
        expected = np.array([
            (p.proj / 17.0) if week in p.actual_weeks else -1.0 for p in players
        ])
        realized = np.array([p.actual_weeks.get(week, 0.0) for p in players])
        total += _week_lineup_score(positions, expected, realized, dict(slots), league)
    return total


# Policies replayed, in report order. "adp" is the baseline every other policy
# is paired against.
#   howie    — the marginal-value engine with the CONFIGURED market anchor
#   proj     — the same engine with market_anchor=0 (pure projections)
#   vorp     — static value-over-replacement
#   adp_need — follow ADP, but skip positions whose starters are filled until
#              the bench rounds
#   adp      — follow ADP
POLICIES = ("howie", "proj", "vorp", "adp_need", "adp")
BASELINE_POLICY = "adp"
EVAL_SLOTS = (2, 5, 8, 11)
EVAL_ROSTER_SIZE = 14  # K/DST are dropped from the replay; 14 rounds of offense
BOOTSTRAP_RESAMPLES = 2000
_FLEX_ELIGIBLE = ("RB", "WR", "TE")

# A policy is a name from POLICIES, or any callable with this signature
# (used by tests to inject a known policy into the paired replay).
PolicyFn = Callable[[List[EvalPlayer], List[EvalPlayer], Any, int, List[int]], EvalPlayer]


def _pick_adp(pool_avail: List[EvalPlayer]) -> EvalPlayer:
    return min(pool_avail, key=lambda p: p.adp if p.adp else 999)


def _pick_adp_need(pool_avail: List[EvalPlayer], roster: List[EvalPlayer],
                   league, current_pick: int) -> EvalPlayer:
    """Follow ADP among positions that still have an open starting slot
    (dedicated slot first, then flex for RB/WR/TE). Once every starter is
    filled — or in the bench rounds — fall back to plain ADP."""
    starters = {"QB": league.qb_slots, "RB": league.rb_slots,
                "WR": league.wr_slots, "TE": league.te_slots}
    n_starter_rounds = sum(starters.values()) + league.flex_slots
    rnd = (current_pick - 1) // league.num_teams + 1
    if rnd > n_starter_rounds:
        return _pick_adp(pool_avail)
    counts: Dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    flex_used = sum(max(counts.get(pos, 0) - starters[pos], 0) for pos in _FLEX_ELIGIBLE)
    open_positions = set()
    for pos, n in starters.items():
        if counts.get(pos, 0) < n:
            open_positions.add(pos)
        elif pos in _FLEX_ELIGIBLE and flex_used < league.flex_slots:
            open_positions.add(pos)
    eligible = [p for p in pool_avail if p.position in open_positions]
    return _pick_adp(eligible or pool_avail)


def _pick_vorp(pool_avail: List[EvalPlayer], roster: List[EvalPlayer]) -> EvalPlayer:
    repl_rank = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
    by_pos: Dict[str, List[EvalPlayer]] = {}
    for p in pool_avail:
        by_pos.setdefault(p.position, []).append(p)
    best, best_v = pool_avail[0], -1e9
    for pos, plist in by_pos.items():
        counts = sum(1 for r in roster if r.position == pos)
        cap = {"QB": 2, "TE": 2}.get(pos, 8)
        if counts >= cap:
            continue
        repl = plist[min(repl_rank.get(pos, 12), len(plist) - 1)].proj if plist else 0
        v = plist[0].proj - repl
        if v > best_v:
            best, best_v = plist[0], v
    return best


def _pick_engine(pool_avail: List[EvalPlayer], roster: List[EvalPlayer], league,
                 current_pick: int, future: List[int],
                 anchored: Optional[Dict[str, float]]) -> EvalPlayer:
    """Marginal-value engine. With `anchored` (uid -> blended proj, computed
    ONCE over the full pool exactly as load_pool does) this is the product;
    with None it is the engine on raw projections (market_anchor=0)."""
    from .value.roster import evaluate_candidates

    def as_pp(p: EvalPlayer) -> PoolPlayer:
        pp = _to_pool_player(p)
        if anchored and p.uid in anchored:
            pp.proj = anchored[p.uid]
        return pp

    results = evaluate_candidates(
        sorted((as_pp(p) for p in pool_avail), key=lambda p: -p.proj),
        [as_pp(p) for p in roster],
        current_pick, future, league, frozenset(), top_n=1,
    )
    if not results:
        return pool_avail[0]
    uid = results[0].player.uid
    return next(p for p in pool_avail if p.uid == uid)


def _policy_pick(policy: Union[str, PolicyFn], pool_avail: List[EvalPlayer],
                 roster: List[EvalPlayer], league, current_pick: int, future: List[int],
                 anchored: Optional[Dict[str, float]] = None) -> EvalPlayer:
    if callable(policy):
        return policy(pool_avail, roster, league, current_pick, future)
    if policy == "adp":
        return _pick_adp(pool_avail)
    if policy == "adp_need":
        return _pick_adp_need(pool_avail, roster, league, current_pick)
    if policy == "vorp":
        return _pick_vorp(pool_avail, roster)
    if policy == "howie":
        return _pick_engine(pool_avail, roster, league, current_pick, future, anchored)
    if policy == "proj":
        return _pick_engine(pool_avail, roster, league, current_pick, future, None)
    raise ValueError(f"unknown policy {policy!r}")


def replay_seed(slot: int, rep: int) -> int:
    """Seed for one paired replay. Every policy drafting from (slot, rep)
    faces opponents driven by this same seed (common random numbers), so
    policy comparisons are paired differences rather than independent draws."""
    return slot * 1_000_000 + rep * 10_000


def replay_draft(policy: Union[str, PolicyFn], pool: List[EvalPlayer], league,
                 slot: int, rep: int,
                 anchored: Optional[Dict[str, float]] = None,
                 ) -> Tuple[List[EvalPlayer], List[Tuple[int, str]]]:
    """Replay one snake draft from `slot` with `policy` in my seat and ADP-noise
    bots everywhere else. `league` must already carry the replay shape (K/DST
    off, roster_size = rounds). Returns (my roster, opponent picks as
    (pick_no, uid)) — the opponent list is what the pairing tests inspect.

    Bots are seeded per (slot, rep, pick_no): two policies from the same
    (slot, rep) see identical opponent behavior for as long as their boards
    coincide, and the same noise stream afterwards."""
    from .mock import bot_pick
    from .state import snake_team_for_pick
    from .value.board import snake_picks

    rounds = league.roster_size
    my_picks = set(snake_picks(league, rounds=rounds))
    pp_by_uid = {p.uid: _to_pool_player(p) for p in pool}  # bots only read adp/stdev
    seed = replay_seed(slot, rep)
    taken: set = set()
    roster: List[EvalPlayer] = []
    opponents: List[Tuple[int, str]] = []
    team_positions: Dict[int, Dict[str, int]] = {}
    for pick_no in range(1, league.num_teams * rounds + 1):
        avail = [p for p in pool if p.uid not in taken]
        if not avail:
            break
        if pick_no in my_picks:
            future = sorted(n for n in my_picks if n > pick_no)
            choice = _policy_pick(policy, avail, roster, league, pick_no, future,
                                  anchored=anchored)
            roster.append(choice)
        else:
            team = snake_team_for_pick(league, pick_no)
            tp = team_positions.setdefault(team, {})
            rng = np.random.default_rng(seed + pick_no)
            rnd = (pick_no - 1) // league.num_teams + 1
            bot = bot_pick([pp_by_uid[p.uid] for p in avail], frozenset(), tp, rnd, league, rng)
            choice = next(p for p in avail if p.uid == bot.uid) if bot else avail[0]
            tp[choice.position] = tp.get(choice.position, 0) + 1
            opponents.append((pick_no, choice.uid))
        taken.add(choice.uid)
    return roster, opponents


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`. Deterministic for a
    given seed. A single value (or constant sample) collapses to a point."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1 or np.ptp(arr) == 0:
        m = float(arr.mean())
        return (m, m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def summarize_paired(scores: Dict[str, List[float]], baseline: str = BASELINE_POLICY,
                     seed: int = 0) -> Dict[str, dict]:
    """Turn aligned per-policy score lists (index i = the same (slot, rep)
    replay for every policy) into the tier C report: for each policy, the
    mean total, the mean PAIRED difference vs the baseline, and a 95%
    bootstrap CI on that paired difference."""
    base = np.asarray(scores[baseline], dtype=float)
    out: Dict[str, dict] = {}
    for policy, vals in scores.items():
        arr = np.asarray(vals, dtype=float)
        if arr.shape != base.shape:
            raise ValueError(f"{policy}: {arr.size} replays vs baseline {base.size} — not paired")
        diffs = arr - base
        lo, hi = bootstrap_mean_ci(diffs, seed=seed)
        out[policy] = {
            "mean_total": round(float(arr.mean()), 1),
            "std_total": round(float(arr.std()), 1),
            "delta_vs_adp": round(float(diffs.mean()), 1),
            "ci_low": round(lo, 1),
            "ci_high": round(hi, 1),
            "crosses_zero": bool(lo <= 0.0 <= hi) if policy != baseline else False,
            "win_rate": round(float((diffs > 0).mean()), 2) if policy != baseline else None,
            "n": int(arr.size),
        }
    return out


def eval_policy(settings: Settings, players: List[EvalPlayer],
                slots_to_test: Optional[List[int]] = None, reps: int = 10,
                policies: Sequence[str] = POLICIES) -> Dict[str, dict]:
    """Tier C. Paired design: for each (slot, rep) every policy drafts against
    the same seeded opponents; n = len(slots) * reps paired replays.

    Returns {policy: {mean_total, std_total, delta_vs_adp, ci_low, ci_high,
    crosses_zero, win_rate, n}} in POLICIES order (baseline last)."""
    from dataclasses import replace as dc_replace

    from .value.board import apply_market_anchor

    base_league = settings.league
    pool = [p for p in _top_pool(players) if p.adp is not None]
    # anchor once over the full pool with the CONFIGURED weight, as the product does
    anchored = {p.uid: p.proj for p in apply_market_anchor(
        [_to_pool_player(p) for p in pool], base_league.market_anchor)}
    slots_to_test = list(slots_to_test or EVAL_SLOTS)
    if BASELINE_POLICY not in policies:
        policies = tuple(policies) + (BASELINE_POLICY,)
    scores: Dict[str, List[float]] = {policy: [] for policy in policies}

    for slot in slots_to_test:
        league = dc_replace(base_league, draft_position=slot, k_slots=0, dst_slots=0,
                            roster_size=EVAL_ROSTER_SIZE)
        for rep in range(reps):
            for policy in policies:
                roster, _ = replay_draft(policy, pool, league, slot, rep, anchored=anchored)
                scores[policy].append(_score_roster(roster, league))

    return summarize_paired(scores)


# ------------------------------------------------------------ tier D: does SoS predict anything?

def eval_sos(settings: Settings, players: List[EvalPlayer]) -> dict:
    """Two questions, both on 2025 actuals with PFF's 2025 PRESEASON SoS:
    (1) season level — did an easier projected schedule predict beating the
        projection? If ~0, projections already price schedule and SoS must
        stay normalized (no effect on season totals).
    (2) weekly level — within a player's season, did easier weeks score more
        relative to his own average? If > 0, reshaping weeks by SoS is valid.
    """
    from .data.sources.pff_sos import refresh_sos

    conn = connect(settings.db_path)
    if not conn.execute("SELECT 1 FROM sos WHERE season = ? LIMIT 1", (EVAL_SEASON,)).fetchone():
        try:
            refresh_sos(conn, settings.pff_dir, EVAL_SEASON)
        except FileNotFoundError:
            conn.close()
            return {"available": False}

    fmt = settings.league.scoring_format
    sos: Dict[tuple, float] = {}
    for r in conn.execute("SELECT team, position, week, value FROM sos WHERE season = ?", (EVAL_SEASON,)):
        sos[(r["team"], r["position"], r["week"])] = r["value"]
    team_of: Dict[str, str] = {}
    for r in conn.execute(
        "SELECT player_uid, team, COUNT(*) n FROM weekly_stats WHERE season = ? "
        "GROUP BY player_uid, team ORDER BY n", (EVAL_SEASON,)):
        team_of[r["player_uid"]] = r["team"]  # last row per uid = most games
    conn.close()

    season_x, season_y, by_pos = [], [], {}
    weekly_x, weekly_y = [], []
    for p in _top_pool(players):
        team = team_of.get(p.uid)
        if not team or len(p.actual_weeks) < 8 or p.proj < 50:
            continue
        weeks = [w for w in range(1, 18) if (team, p.position, w) in sos]
        if len(weeks) < 10:
            continue
        season_sos = float(np.mean([sos[(team, p.position, w)] for w in weeks]))
        ratio = (p.actual_total / len(p.actual_weeks)) / (p.proj / max(p.games_proj, 1))
        season_x.append(season_sos); season_y.append(ratio)
        by_pos.setdefault(p.position, ([], []))
        by_pos[p.position][0].append(season_sos); by_pos[p.position][1].append(ratio)
        own_mean = p.actual_total / len(p.actual_weeks)
        for w, pts in p.actual_weeks.items():
            if (team, p.position, w) in sos and own_mean > 0:
                weekly_x.append(sos[(team, p.position, w)]); weekly_y.append(pts / own_mean)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and a.std() > 0 and b.std() > 0 else 0.0

    # Decomposition on realized 2025 box scores (the ceiling of matchup value):
    #   hindsight: leave-one-out realized defense-vs-position -> player weekly scoring
    #   forecast : preseason SoS grade -> realized defense-vs-position
    conn = connect(settings.db_path)
    from collections import defaultdict
    rows = conn.execute(
        f"SELECT player_uid, opponent, position, week, pts_{fmt} AS pts FROM weekly_stats "
        "WHERE season = ? AND week <= 17 AND position IN ('QB','RB','WR','TE') "
        "AND opponent IS NOT NULL", (EVAL_SEASON,)).fetchall()
    sched = {}
    for r in conn.execute("SELECT week, home_team, away_team FROM games WHERE season = ? AND week <= 17",
                          (EVAL_SEASON,)):
        sched[(r["home_team"], r["week"])] = r["away_team"]
        sched[(r["away_team"], r["week"])] = r["home_team"]
    conn.close()
    allowed = defaultdict(float)
    for r in rows:
        allowed[(r["opponent"], r["position"], r["week"])] += r["pts"] or 0.0
    weeks_by_def = defaultdict(set)
    for (d, pos, w) in allowed:
        weeks_by_def[(d, pos)].add(w)
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_uid"]].append(r)
    hx, hy = [], []
    for uid, rs in by_player.items():
        if len(rs) < 8:
            continue
        mean = np.mean([r["pts"] or 0.0 for r in rs])
        if mean < 6:
            continue
        pos = rs[0]["position"]
        for r in rs:
            ws = [w for w in weeks_by_def[(r["opponent"], pos)] if w != r["week"]]
            if len(ws) >= 4:
                hx.append(np.mean([allowed[(r["opponent"], pos, w)] for w in ws]))
                hy.append((r["pts"] or 0.0) / mean)
    season_dvp = {k: np.mean([allowed[(k[0], k[1], w)] for w in ws])
                  for k, ws in weeks_by_def.items() if len(ws) >= 8}
    fx, fy, seen = [], [], set()
    for (team, pos, w), val in sos.items():
        opp = sched.get((team, w))
        if opp and (opp, pos) in season_dvp and (opp, pos) not in seen:
            seen.add((opp, pos)); fx.append(val); fy.append(season_dvp[(opp, pos)])
    hx_a, hy_a = np.array(hx), np.array(hy)
    q1, q3 = (np.percentile(hx_a, 25), np.percentile(hx_a, 75)) if len(hx_a) else (0, 0)

    return {
        "available": True,
        "hindsight_corr": round(corr(hx, hy), 3),
        "hindsight_hard_vs_soft": (round(float(hy_a[hx_a <= q1].mean()), 2),
                                   round(float(hy_a[hx_a >= q3].mean()), 2)) if len(hx_a) else None,
        "forecast_corr": round(corr(fx, fy), 3),
        "season_n": len(season_x),
        "season_corr": round(corr(season_x, season_y), 3),
        "season_by_pos": {pos: round(corr(x, y), 3) for pos, (x, y) in by_pos.items() if len(x) > 8},
        "weekly_n": len(weekly_x),
        "weekly_corr": round(corr(weekly_x, weekly_y), 3),
    }
