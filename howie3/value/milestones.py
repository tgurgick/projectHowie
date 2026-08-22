"""Empirical milestone anchors — pure counting over the box scores.

Independent of the simulation by design: these are frequencies of things
that actually happened (100-yard games, multi-TD games, target volume), by
position, tier, season, and player. They answer "what does a good week from
this slot look like in this era?" and expose the league's drift (e.g. the
300-yard QB game halved between 2018 and 2025).
"""

import sqlite3
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# (label, predicate over a weekly_stats row)
MILESTONES: Dict[str, List[Tuple[str, Callable[[sqlite3.Row], bool]]]] = {
    "RB": [
        ("100+ rush yds", lambda r: (r["rush_yards"] or 0) >= 100),
        ("2+ TD", lambda r: (r["rush_tds"] or 0) + (r["rec_tds"] or 0) >= 2),
        ("100+ scrimmage", lambda r: (r["rush_yards"] or 0) + (r["rec_yards"] or 0) >= 100),
        ("5+ rec", lambda r: (r["receptions"] or 0) >= 5),
    ],
    "WR": [
        ("100+ rec yds", lambda r: (r["rec_yards"] or 0) >= 100),
        ("TD", lambda r: (r["rec_tds"] or 0) + (r["rush_tds"] or 0) >= 1),
        ("2+ TD", lambda r: (r["rec_tds"] or 0) + (r["rush_tds"] or 0) >= 2),
        ("10+ targets", lambda r: (r["targets"] or 0) >= 10),
    ],
    "TE": [
        ("75+ rec yds", lambda r: (r["rec_yards"] or 0) >= 75),
        ("TD", lambda r: (r["rec_tds"] or 0) >= 1),
        ("7+ targets", lambda r: (r["targets"] or 0) >= 7),
    ],
    "QB": [
        ("300+ pass yds", lambda r: (r["pass_yards"] or 0) >= 300),
        ("3+ pass TD", lambda r: (r["pass_tds"] or 0) >= 3),
        ("rush TD", lambda r: (r["rush_tds"] or 0) >= 1),
        ("0 TD game", lambda r: (r["pass_tds"] or 0) + (r["rush_tds"] or 0) == 0),
    ],
}
# "boom" = the headline milestone per position, used for roster anchors
BOOM = {"RB": "100+ scrimmage", "WR": "100+ rec yds", "TE": "75+ rec yds", "QB": "300+ pass yds"}
STARTER_TIER_N = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

_COLS = ("season, week, team, opponent, position, pass_yards, pass_tds, interceptions, "
         "rush_attempts, rush_yards, rush_tds, targets, receptions, rec_yards, rec_tds")


def player_games(conn: sqlite3.Connection, uid: str, fmt: str,
                 seasons: Sequence[int]) -> List[dict]:
    """Chronological game log with milestone flags — the hover distribution."""
    qmarks = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT {_COLS}, pts_{fmt} AS pts FROM weekly_stats "
        f"WHERE player_uid = ? AND season IN ({qmarks}) AND week <= 18 "
        "ORDER BY season, week",
        (uid, *seasons),
    ).fetchall()
    out = []
    for r in rows:
        pos = r["position"] if r["position"] in MILESTONES else None
        flags = {label: bool(pred(r)) for label, pred in MILESTONES.get(pos, [])}
        out.append({
            "season": r["season"], "week": r["week"], "opp": r["opponent"],
            "pts": round(r["pts"] or 0.0, 1),
            "rush_yds": r["rush_yards"] or 0, "rush_tds": r["rush_tds"] or 0,
            "rec": r["receptions"] or 0, "rec_yds": r["rec_yards"] or 0,
            "rec_tds": r["rec_tds"] or 0, "targets": r["targets"] or 0,
            "pass_yds": r["pass_yards"] or 0, "pass_tds": r["pass_tds"] or 0,
            "flags": flags,
        })
    return out


def player_rates(games: List[dict], position: str) -> Dict[str, float]:
    labels = [label for label, _ in MILESTONES.get(position, [])]
    if not games:
        return {}
    return {label: round(sum(1 for g in games if g["flags"].get(label)) / len(games), 3)
            for label in labels}


def tier_rates(conn: sqlite3.Connection, fmt: str, position: str, season: int,
               top_n: Optional[int] = None) -> Dict[str, float]:
    """Milestone rates across starter-tier player-weeks in one season."""
    if position not in MILESTONES:
        return {}
    n = top_n or STARTER_TIER_N[position]
    rows = conn.execute(
        f"""WITH starters AS (
              SELECT player_uid FROM weekly_stats WHERE season = ? AND position = ?
              GROUP BY player_uid ORDER BY SUM(pts_{fmt}) DESC LIMIT ?)
            SELECT {_COLS} FROM weekly_stats w JOIN starters USING (player_uid)
            WHERE w.season = ? AND w.week <= 17""",
        (season, position, n, season),
    ).fetchall()
    if not rows:
        return {}
    return {label: round(sum(1 for r in rows if pred(r)) / len(rows), 3)
            for label, pred in MILESTONES[position]}


def league_trend(conn: sqlite3.Connection, fmt: str,
                 seasons: Sequence[int]) -> Dict[str, Dict[str, Dict[int, float]]]:
    """position -> milestone -> {season: rate} over starter-tier weeks."""
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for pos in MILESTONES:
        out[pos] = {}
        for season in seasons:
            for label, rate in tier_rates(conn, fmt, pos, season).items():
                out[pos].setdefault(label, {})[season] = rate
    return out


def roster_anchors(conn: sqlite3.Connection, fmt: str, starters: List[dict],
                   seasons: Sequence[int], fallback_season: int) -> dict:
    """Typical-week anchors for a set of starters [{uid, name, position}].

    Uses each player's own last-two-season rates; players without history
    (rookies) fall back to the starter-tier rate of their position."""
    boom_ps, td_per_game, rows = [], 0.0, []
    tier_cache: Dict[str, Dict[str, float]] = {}
    for p in starters:
        pos = p["position"]
        if pos not in MILESTONES:
            continue
        games = player_games(conn, p["uid"], fmt, seasons)
        rates = player_rates(games, pos)
        source = f"{len(games)} games"
        if len(games) < 6:
            if pos not in tier_cache:
                tier_cache[pos] = tier_rates(conn, fmt, pos, fallback_season)
            rates = tier_cache[pos]
            source = "tier baseline"
        boom = rates.get(BOOM[pos], 0.0)
        tds = (sum(g["rush_tds"] + g["rec_tds"] + (g["pass_tds"] if pos == "QB" else 0)
                   for g in games) / len(games)) if len(games) >= 6 else None
        if tds is None:
            tds = {"QB": 1.9, "RB": 0.6, "WR": 0.45, "TE": 0.35}[pos]
        boom_ps.append(boom)
        td_per_game += tds
        rows.append({"name": p["name"], "position": pos, "boom": BOOM[pos],
                     "boom_rate": round(boom, 3), "tds_per_game": round(tds, 2),
                     "source": source})
    p_any = 1.0 - float(__import__("math").prod(1.0 - b for b in boom_ps)) if boom_ps else 0.0
    return {
        "starters": rows,
        "expected_booms_per_week": round(sum(boom_ps), 2),
        "expected_tds_per_week": round(td_per_game, 2),
        "p_any_boom": round(p_any, 3),
    }
