"""Empirical weekly-outcome distributions, calibrated from history in howie.db.

Replaces v2's hand-picked CV tables. For each (position, tier) we measure from
2018-2025 weekly stats:
  - weekly CV (std/mean of weekly points across games played)
  - availability (fraction of possible games played)
Tiers are quantile bands of season-total rank within position, so a 2026
player maps to a bucket by projection rank the same way historical players
mapped by realized rank.

K and DST are not in weekly_stats (offense-only source), so they use static,
documented constants — their variance barely differentiates players anyway.
"""

import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TIER_FRACS = (0.125, 0.25, 0.5, 1.0)   # elite / starter / mid / depth bands
POOL_SIZE = {"QB": 32, "RB": 60, "WR": 72, "TE": 32}
STATIC_BUCKETS = {"K": (0.45, 0.97), "DST": (0.55, 1.0)}  # (cv, p_play)


@dataclass
class Bucket:
    cv: float
    p_play: float
    n: int  # player-seasons observed


_cache: Dict[Tuple[int, str], Dict[Tuple[str, int], Bucket]] = {}


def calibrate(conn: sqlite3.Connection, fmt: str) -> Dict[Tuple[str, int], Bucket]:
    """(position, tier_index) -> Bucket, measured from all completed seasons."""
    key = (id(conn), fmt)
    if key in _cache:
        return _cache[key]

    df = pd.read_sql_query(
        f"SELECT season, player_uid, position, week, pts_{fmt} AS pts "
        "FROM weekly_stats WHERE week <= 17 AND position IN ('QB','RB','WR','TE')",
        conn,
    )
    per = (
        df.groupby(["season", "position", "player_uid"])["pts"]
        .agg(total="sum", games="count", mean="mean", std="std")
        .reset_index()
    )
    per["possible"] = np.where(per["season"] >= 2021, 17, 16)
    per["p_play"] = (per["games"] / per["possible"]).clip(upper=1.0)
    per["cv"] = per["std"] / per["mean"]

    buckets: Dict[Tuple[str, int], Bucket] = {}
    frames = []
    for (season, pos), grp in per.groupby(["season", "position"]):
        pool_n = POOL_SIZE[pos]
        grp = grp.sort_values("total", ascending=False).head(pool_n).copy()
        grp["rank_frac"] = (np.arange(len(grp)) + 1) / pool_n
        grp["tier"] = np.searchsorted(TIER_FRACS, grp["rank_frac"], side="left")
        frames.append(grp)
    allp = pd.concat(frames)
    for (pos, tier), grp in allp.groupby(["position", "tier"]):
        valid = grp[(grp["games"] >= 6) & (grp["mean"] >= 4)]
        buckets[(pos, int(tier))] = Bucket(
            cv=float(valid["cv"].median()) if len(valid) else 0.5,
            p_play=float(grp["p_play"].mean()),
            n=len(grp),
        )
    _cache[key] = buckets
    return buckets


def tier_of(position: str, rank: int) -> int:
    pool_n = POOL_SIZE.get(position)
    if pool_n is None:
        return 0
    frac = min(rank / pool_n, 1.0)
    return int(np.searchsorted(TIER_FRACS, frac, side="left"))


@dataclass
class SimPlayer:
    """Everything the simulator needs to sample one player's season."""
    name: str
    position: str
    proj: float
    weekly_mu: float
    cv: float
    p_play: float
    bye_week: Optional[int]
    sos_mult: np.ndarray  # length-18 multiplier indexed by week-1 (1.0 = neutral)


# SoS scale is PFF's 0-10 (higher = easier). +/-3% per point around neutral 5:
# a full-scale swing moves a weekly mean by ~15% — deliberately modest.
def _sos_multiplier(value: float) -> float:
    return 1.0 + (value - 5.0) * 0.03


def truncation_factor(cv: float) -> float:
    """g(cv) with E[max(0, N(m, cv*m))] = m * g(cv). Samplers divide the mean
    by this so clip-at-zero sampling preserves the intended expectation."""
    if cv <= 0:
        return 1.0
    z = 1.0 / cv
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return cdf + cv * phi


def build_sim_players(
    conn: sqlite3.Connection,
    players: List,               # PoolPlayer-like: name/position/team/proj + optional games
    season: int,
    fmt: str,
    proj_rank: Dict[str, int],   # uid -> projection rank within position
    games_by_uid: Dict[str, float],
    default_games: float = 16.0,
) -> List[SimPlayer]:
    buckets = calibrate(conn, fmt)
    byes = team_bye_weeks(conn, season)
    sos = sos_multipliers(conn, season)

    out = []
    for p in players:
        if p.position in STATIC_BUCKETS:
            cv, p_play = STATIC_BUCKETS[p.position]
        else:
            b = buckets.get((p.position, tier_of(p.position, proj_rank.get(p.uid, 999))))
            cv, p_play = (b.cv, b.p_play) if b else (0.5, 0.9)
        games = games_by_uid.get(p.uid) or default_games
        games = max(min(games, 17.0), 1.0)
        # PFF's games projection already prices availability; blend with the
        # historical bucket so tail risk isn't understated
        p_play = min((games / 17.0 + p_play) / 2.0, 1.0)

        bye = byes.get(p.team) if p.team else None
        playable = 17 - (1 if bye and bye <= 17 else 0)

        # SoS redistributes points WITHIN the season (projections already price
        # schedule at season level): normalize multipliers to mean 1.0 over
        # playable weeks so the season expectation is untouched.
        mults = np.ones(18)
        if p.team and (p.team in sos.get(p.position, {})):
            for week, val in sos[p.position][p.team].items():
                if 1 <= week <= 18:
                    mults[week - 1] = _sos_multiplier(val)
        playable_idx = [w for w in range(17) if (bye is None or w != bye - 1)]
        mean_mult = float(np.mean(mults[playable_idx])) if playable_idx else 1.0
        if mean_mult > 0:
            mults = mults / mean_mult

        # Calibration invariant: E[simulated season points] == proj.
        # Weekly mean is spread over expected PLAYED weeks (playable * p_play).
        out.append(
            SimPlayer(
                name=p.name,
                position=p.position,
                proj=p.proj,
                weekly_mu=p.proj / (playable * p_play),
                cv=cv,
                p_play=p_play,
                bye_week=bye,
                sos_mult=mults,
            )
        )
    return out


def team_bye_weeks(conn: sqlite3.Connection, season: int) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT week, home_team, away_team FROM games WHERE season = ? AND week <= 18",
        (season,),
    ).fetchall()
    weeks_played: Dict[str, set] = {}
    all_weeks = set()
    for r in rows:
        all_weeks.add(r["week"])
        for t in (r["home_team"], r["away_team"]):
            weeks_played.setdefault(t, set()).add(r["week"])
    return {
        team: min(all_weeks - played)
        for team, played in weeks_played.items()
        if all_weeks - played
    }


def sos_multipliers(conn: sqlite3.Connection, season: int) -> Dict[str, Dict[str, Dict[int, float]]]:
    """position -> team -> week -> raw SoS value."""
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for r in conn.execute("SELECT team, position, week, value FROM sos WHERE season = ?", (season,)):
        out.setdefault(r["position"], {}).setdefault(r["team"], {})[r["week"]] = r["value"]
    return out
