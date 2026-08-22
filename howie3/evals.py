"""Backtests against realized 2025 results — the measurement layer.

Three tiers, run with `howie eval run`:
  A. Inputs   — how good were 2025 preseason projections and ADP at all?
  B. Calibration — did simulated p10-p90 bands cover realized outcomes?
  C. Policy   — replay 2025 drafts: Howie's marginal-value policy vs
                follow-ADP and static-VORP baselines, scored with realized
                weekly points and weekly optimal lineups.

2025 preseason inputs come from the legacy database (PPR-scored projections);
all policies draft from the SAME inputs, so tier C's comparison is fair even
though realized scoring is half-PPR. Buckets for tier B are calibrated on
seasons <= 2024 only (no leakage).
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


def _policy_pick(policy: str, pool_avail: List[EvalPlayer], roster: List[EvalPlayer],
                 league, current_pick: int, future: List[int],
                 anchored: Optional[Dict[str, float]] = None) -> EvalPlayer:
    if policy == "adp":
        return min(pool_avail, key=lambda p: p.adp if p.adp else 999)
    if policy == "vorp":
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
    # howie: marginal-value engine. Values come from the anchor map computed
    # ONCE over the full pool (exactly how the product anchors in load_pool)
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


def eval_policy(settings: Settings, players: List[EvalPlayer],
                slots_to_test: Optional[List[int]] = None, reps: int = 3) -> List[dict]:
    from dataclasses import replace as dc_replace

    from .mock import bot_pick
    from .state import snake_team_for_pick
    from .value.board import apply_market_anchor, snake_picks

    base_league = settings.league
    pool = [p for p in _top_pool(players) if p.adp is not None]
    # anchor once over the full pool with the CONFIGURED weight, as the product does
    anchored = {p.uid: p.proj for p in apply_market_anchor(
        [_to_pool_player(p) for p in pool], base_league.market_anchor)}
    slots_to_test = slots_to_test or [2, 5, 8, 11]
    results: Dict[str, List[float]] = {"howie": [], "adp": [], "vorp": []}

    for policy in results:
        for slot in slots_to_test:
            for rep in range(reps):
                league = dc_replace(base_league, draft_position=slot,
                                    k_slots=0, dst_slots=0, roster_size=14)
                my_picks = set(snake_picks(league, rounds=14))
                taken: set = set()
                roster: List[EvalPlayer] = []
                team_positions: Dict[int, Dict[str, int]] = {}
                total_picks = league.num_teams * 14
                for pick_no in range(1, total_picks + 1):
                    avail = [p for p in pool if p.uid not in taken]
                    if not avail:
                        break
                    if pick_no in my_picks:
                        future = sorted(n for n in my_picks if n > pick_no)
                        choice = _policy_pick(policy, avail, roster, league,
                                              pick_no, future, anchored=anchored)
                        roster.append(choice)
                    else:
                        team = snake_team_for_pick(league, pick_no)
                        tp = team_positions.setdefault(team, {})
                        rng = np.random.default_rng(rep * 1_000_000 + pick_no)
                        rnd = (pick_no - 1) // league.num_teams + 1
                        bot = bot_pick([_to_pool_player(p) for p in avail], frozenset(),
                                       tp, rnd, league, rng)
                        choice = next(p for p in avail if p.uid == bot.uid) if bot else avail[0]
                        tp[choice.position] = tp.get(choice.position, 0) + 1
                    taken.add(choice.uid)
                results[policy].append(_score_roster(roster, league))

    out = []
    baseline = float(np.mean(results["adp"]))
    for policy in ("howie", "vorp", "adp"):
        scores = np.array(results[policy])
        out.append({
            "policy": policy, "mean": round(float(scores.mean()), 1),
            "std": round(float(scores.std()), 1),
            "vs_adp": round(float(scores.mean()) - baseline, 1),
            "n_drafts": len(scores),
        })
    return out


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

    return {
        "available": True,
        "season_n": len(season_x),
        "season_corr": round(corr(season_x, season_y), 3),
        "season_by_pos": {pos: round(corr(x, y), 3) for pos, (x, y) in by_pos.items() if len(x) > 8},
        "weekly_n": len(weekly_x),
        "weekly_corr": round(corr(weekly_x, weekly_y), 3),
    }
