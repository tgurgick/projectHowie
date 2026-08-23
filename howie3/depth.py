"""Official team depth charts (nflverse, updated daily) — the spine of the
TEAM report. Only the latest snapshot per team is kept; QB/RB/WR/TE rows,
matched to players by gsis id (names kept for unmatched bodies so the chart
still reads complete)."""

import io
import sqlite3
import urllib.request
from typing import Dict, List

DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"
SKILL = ("QB", "RB", "WR", "TE")
# nflverse WR slot ids -> labels (X / Z / slot); others keep the raw id
SLOT_LABEL = {"1": "X", "2": "Z", "8": "slot"}


def fetch_depth_charts(season: int):
    import pandas as pd

    raw = urllib.request.urlopen(DEPTH_URL.format(season=season), timeout=120).read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def refresh_depth_charts(conn: sqlite3.Connection, season: int, frame=None) -> int:
    df = frame if frame is not None else fetch_depth_charts(season)
    df = df[df["pos_abb"].isin(SKILL)]
    known = {r[0] for r in conn.execute("SELECT player_uid FROM players")}
    n = 0
    for team, sub in df.groupby("team"):
        latest = sub[sub["dt"] == sub["dt"].max()]
        conn.execute("DELETE FROM depth_charts WHERE season = ? AND team = ?", (season, team))
        for _, r in latest.iterrows():
            uid = r.get("gsis_id") if isinstance(r.get("gsis_id"), str) and r.get("gsis_id") in known else None
            slot = str(r.get("pos_slot")) if r.get("pos_slot") is not None else ""
            conn.execute(
                "INSERT OR REPLACE INTO depth_charts (season, team, dt, player_uid, player_name, position, slot, rank) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (season, team, str(r["dt"])[:19], uid, str(r["player_name"]), str(r["pos_abb"]),
                 SLOT_LABEL.get(slot, slot) if r["pos_abb"] == "WR" else "", int(r["pos_rank"])),
            )
            n += 1
    conn.commit()
    return n


def team_depth(conn: sqlite3.Connection, season: int, team: str) -> Dict[str, List[dict]]:
    """position -> [{rank, slot, name, uid}] in official order."""
    out: Dict[str, List[dict]] = {pos: [] for pos in SKILL}
    for r in conn.execute(
        "SELECT player_uid, player_name, position, slot, rank, dt FROM depth_charts "
        "WHERE season = ? AND team = ? ORDER BY position, rank", (season, team.upper())):
        out[r["position"]].append({"rank": r["rank"], "slot": r["slot"] or None, "name": r["player_name"],
                                   "uid": r["player_uid"], "dt": r["dt"]})
    return out
