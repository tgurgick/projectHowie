"""Name and team-code normalization + player resolution.

The crosswalk (player_ids) is the primary identity mechanism; name resolution
is the ingest-time fallback for sources that only give us display names
(FantasyPros, PFF). Failures are recorded in unmatched_names, and manual
fixes go in name_aliases — never fuzzy-matched silently.
"""

import re
import sqlite3
import unicodedata
from typing import Optional

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.'’`\-,]")

# Everything maps onto nflverse team codes (LA=Rams, JAX, WAS, ...)
TEAM_FIX = {
    "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU", "LAR": "LA", "SL": "LA", "STL": "LA", "SD": "LAC",
    "OAK": "LV", "GBP": "GB", "KCC": "KC", "NEP": "NE", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "LVR": "LV",
}

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "FB", "DST"}


def name_key(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = n.lower()
    n = _PUNCT_RE.sub("", n)
    n = _SUFFIX_RE.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def fix_team(team: Optional[str]) -> Optional[str]:
    if not team:
        return team
    t = team.strip().upper()
    return TEAM_FIX.get(t, t)


def fix_position(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return pos
    p = re.sub(r"\d+$", "", pos.strip().upper())  # 'RB12' -> 'RB'
    if p in {"D/ST", "DEF", "DS"}:
        p = "DST"
    if p == "FB":
        p = "RB"
    return p


def resolve_uid(
    conn: sqlite3.Connection,
    name: str,
    position: Optional[str] = None,
    team: Optional[str] = None,
) -> Optional[str]:
    """Resolve a display name to a player_uid, or None if ambiguous/unknown."""
    key = name_key(name)

    alias = conn.execute(
        "SELECT player_uid FROM name_aliases WHERE name_key = ?", (key,)
    ).fetchone()
    if alias:
        return alias["player_uid"]

    rows = conn.execute(
        "SELECT player_uid, position, team, draft_year FROM players WHERE name_key = ?",
        (key,),
    ).fetchall()
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]["player_uid"]

    pos = fix_position(position) if position else None
    if pos:
        pos_rows = [r for r in rows if fix_position(r["position"]) == pos]
        if len(pos_rows) == 1:
            return pos_rows[0]["player_uid"]
        rows = pos_rows or rows

    tm = fix_team(team) if team else None
    if tm:
        team_rows = [r for r in rows if fix_team(r["team"]) == tm]
        if len(team_rows) == 1:
            return team_rows[0]["player_uid"]

    return None  # ambiguous — caller records it in unmatched_names


def record_unmatched(
    conn: sqlite3.Connection,
    source: str,
    name: str,
    season: Optional[int] = None,
    position: Optional[str] = None,
    team: Optional[str] = None,
    detail: str = "",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO unmatched_names (source, season, name, position, team, detail) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source, season, name, position, team, detail),
    )
