"""Player status: the machine-actionable layer the engine reads.

Projections assume 17 games for everyone (PFF's 2026 export projects 541 of
547 players at exactly 17.0), and ADP only prices injuries indirectly. This
table is where "is he hurt / suspended / about to be cut / what's his role"
lives as typed fields, so the engine can act on it instead of leaving a value
on the board for a torn ACL.

Two writers, one precedence rule:
  1. `refresh_roster_status` — nflverse's live roster feed (RES / PUP /
     exempt / suspended / released), automatic in `howie data refresh`.
  2. research JSON (`howie graph import`, the research-teams workflow) —
     a `players` array with one record per draft-relevant player.
The current status is the row with the latest `as_of`; on the same day a
researched row beats the roster feed.
"""

import io
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

STATUSES = ("active", "questionable", "injured", "out_season", "suspended",
            "holdout", "cut_risk", "released", "retired")
ROLES = ("starter", "committee", "backup", "depth", "unknown")
OUT = frozenset({"out_season", "released", "retired"})   # not draftable
FANTASY_GAMES = 17
ROSTER_SOURCE = "nflverse_roster"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"

# nflverse roster `status` -> (status, games_out, confidence). Games are a
# conservative preseason prior; research overrides them with specifics.
ROSTER_MAP = {
    "ACT": ("active", 0, 0.9),
    "RES": ("injured", 8, 0.6),       # reserve (IR / PUP-R / NFI): return unknown
    "PUP": ("injured", 6, 0.6),
    "SUS": ("suspended", 6, 0.7),
    "INA": ("questionable", 1, 0.5),
    "NWT": ("released", FANTASY_GAMES, 0.8),   # not with team
    "CUT": ("released", FANTASY_GAMES, 0.9),
    "RLS": ("released", FANTASY_GAMES, 0.9),
    "RET": ("retired", FANTASY_GAMES, 0.95),
    "DEV": ("active", 0, 0.7),        # practice squad
}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------- import

def validate_record(rec: dict) -> dict:
    """Normalize one status record from research output; raise on contract
    violations (no silent coercion of a bad status into 'active')."""
    status = str(rec.get("status", "")).strip().lower()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r} for {rec.get('name')}")
    games = rec.get("games_out", 0)
    games = int(games) if games is not None else 0
    if not 0 <= games <= FANTASY_GAMES:
        raise ValueError(f"games_out must be 0..{FANTASY_GAMES}, got {games} for {rec.get('name')}")
    if status in OUT:
        games = FANTASY_GAMES
    role = rec.get("role") or "unknown"
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r} for {rec.get('name')}")
    cut = float(rec.get("cut_risk") or 0.0)
    if not 0.0 <= cut <= 1.0:
        raise ValueError(f"cut_risk must be 0..1, got {cut} for {rec.get('name')}")
    conf = rec.get("confidence")
    conf = float(conf) if conf is not None else 0.7
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence must be 0..1 for {rec.get('name')}")
    return {"status": status, "games_out": games, "injury": rec.get("injury") or None,
            "role": role, "cut_risk": cut, "note": (rec.get("note") or "")[:400] or None,
            "confidence": conf}


def import_player_status(conn: sqlite3.Connection, doc: dict, season: int) -> int:
    """Import the `players` array of a research document. Each record:
    {"name": "Quinshon Judkins" | "player:<uid>", "status": ..., "games_out": n,
     "injury": "...", "role": ..., "cut_risk": 0-1, "note": "...",
     "confidence": 0-1, "source": "..."}. Unresolvable names fail loudly."""
    from .graph import _resolve_entity

    as_of = str(doc.get("as_of") or _today())[:10]
    default_source = f"research {as_of}"
    n = 0
    for rec in doc.get("players", []):
        ref = rec.get("player") or rec.get("name")
        if not ref:
            raise ValueError(f"status record without a player name: {rec}")
        ref = ref if ref.startswith("player:") else f"player:{ref}"
        uid = _resolve_entity(conn, ref).split(":", 1)[1]
        fields = validate_record(rec)
        conn.execute(
            "INSERT OR REPLACE INTO player_status (season, player_uid, as_of, status, games_out, "
            "injury, role, cut_risk, note, confidence, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (doc.get("season", season), uid, as_of, fields["status"], fields["games_out"],
             fields["injury"], fields["role"], fields["cut_risk"], fields["note"],
             fields["confidence"], str(rec.get("source") or doc.get("source") or default_source)[:200]),
        )
        n += 1
    conn.commit()
    return n


# ---------------------------------------------------------------- nflverse roster feed

def fetch_roster(season: int):
    import pandas as pd

    raw = urllib.request.urlopen(ROSTER_URL.format(season=season), timeout=60).read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def refresh_roster_status(conn: sqlite3.Connection, season: int, frame=None) -> int:
    """Record today's roster status for every player with a current
    projection. Non-active codes become injured/suspended/released rows;
    active players get an 'active' row so a cleared injury overrides an
    older reserve row."""
    df = frame if frame is not None else fetch_roster(season)
    id_col = "gsis_id" if "gsis_id" in df.columns else "player_id"
    pool = {r[0] for r in conn.execute(
        "SELECT DISTINCT player_uid FROM projections WHERE season = ?", (season,))}
    as_of = _today()
    n = 0
    seen = set()
    for _, row in df.iterrows():
        uid = row.get(id_col)
        if not isinstance(uid, str) or uid not in pool:
            continue
        code = str(row.get("status") or "ACT").upper()
        status, games, conf = ROSTER_MAP.get(code, ("questionable", 4, 0.4))
        if code.startswith("E") and code not in ROSTER_MAP:
            status, games, conf = "suspended", 4, 0.5   # exempt / commissioner lists
        note = None if code == "ACT" else f"roster status {code}"
        conn.execute(
            "INSERT OR REPLACE INTO player_status (season, player_uid, as_of, status, games_out, "
            "injury, role, cut_risk, note, confidence, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (season, uid, as_of, status, games, None, None, 0.0, note, conf, ROSTER_SOURCE),
        )
        seen.add(uid)
        n += 1
    # Projected players on NO roster (unsigned veterans in late August): a
    # real cut risk until research says otherwise. DST ids are not players.
    for uid in sorted(pool - seen):
        if not uid.startswith("00-"):
            continue
        conn.execute(
            "INSERT OR REPLACE INTO player_status (season, player_uid, as_of, status, games_out, "
            "injury, role, cut_risk, note, confidence, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (season, uid, as_of, "cut_risk", 0, None, None, 0.7, "not on any NFL roster", 0.5, ROSTER_SOURCE),
        )
        n += 1
    conn.commit()
    return n


# ---------------------------------------------------------------- read

def current_status(conn: sqlite3.Connection, season: int, include_active: bool = False) -> Dict[str, dict]:
    """uid -> current status row.

    Precedence: the latest researched row wins. The roster feed overrides it
    only with a NEWER non-active signal (reserve, suspension, release) — an
    "ACT" roster row carries no health information and must never erase a
    researched injury, role or cut risk. With no research, the latest roster
    row stands. Active rows with nothing to say are omitted unless
    include_active (the TEAM report wants researched roles too)."""
    rows = conn.execute(
        "SELECT player_uid, as_of, status, games_out, injury, role, cut_risk, note, confidence, source "
        "FROM player_status WHERE season = ? ORDER BY player_uid, as_of", (season,)).fetchall()
    research: Dict[str, dict] = {}
    roster: Dict[str, dict] = {}
    for r in rows:
        (roster if r["source"] == ROSTER_SOURCE else research)[r["player_uid"]] = dict(r)  # ascending: last wins
    latest: Dict[str, dict] = {}
    for uid in set(research) | set(roster):
        res, ros = research.get(uid), roster.get(uid)
        if res is None:
            latest[uid] = ros
        elif ros is not None and ros["status"] != "active" and ros["as_of"] > res["as_of"]:
            latest[uid] = ros
        else:
            latest[uid] = res
    if include_active:
        return latest
    return {uid: r for uid, r in latest.items()
            if not (r["status"] == "active" and r["games_out"] == 0 and r["cut_risk"] == 0)}


def availability_factor(row: Optional[dict]) -> float:
    """Multiplier on season value: games available × P(not cut)."""
    if not row:
        return 1.0
    if row["status"] in OUT:
        return 0.0
    games = max(0, FANTASY_GAMES - int(row["games_out"] or 0))
    return (games / FANTASY_GAMES) * (1.0 - float(row["cut_risk"] or 0.0))


def chip(row: Optional[dict]) -> Optional[dict]:
    """Short label for the UI: {'text': 'OUT · ACL', 'level': 'out'|'warn'|'note'}."""
    if not row:
        return None
    st, g, inj = row["status"], int(row["games_out"] or 0), row.get("injury")
    if st in OUT:
        return {"text": ("OUT" if st == "out_season" else st.upper()) + (f" · {inj}" if inj else ""), "level": "out"}
    if st == "suspended":
        return {"text": f"SUSP {g}", "level": "warn"}
    if st == "injured":
        return {"text": f"OUT {g}" + (f" · {inj}" if inj else ""), "level": "warn"}
    if st == "questionable":
        return {"text": "Q" + (f" · {inj}" if inj else ""), "level": "note"}
    if st == "holdout":
        return {"text": "HOLDOUT", "level": "warn"}
    if st == "cut_risk" or float(row.get("cut_risk") or 0) >= 0.3:
        return {"text": f"CUT? {int(round(float(row.get('cut_risk') or 0) * 100))}%", "level": "warn"}
    return None


# ---------------------------------------------------------------- research support

def research_targets(conn: sqlite3.Connection, season: int, team: str, fmt: str = "half") -> List[dict]:
    """Every draft-relevant player on a team (has a current projection),
    with what we already know — the list a research agent must cover."""
    status = current_status(conn, season)
    rows = conn.execute(
        f"""SELECT p.player_uid AS uid, p.name, pr.position, pr.pts_{fmt} AS proj, a.adp
            FROM projections pr JOIN players p ON p.player_uid = pr.player_uid
            LEFT JOIN adp a ON a.player_uid = pr.player_uid AND a.season = pr.season AND a.format = ?
            WHERE pr.season = ? AND pr.team = ? ORDER BY COALESCE(a.adp, 999), proj DESC""",
        (fmt, season, team.upper())).fetchall()
    out = []
    for r in rows:
        st = status.get(r["uid"])
        out.append({"uid": r["uid"], "name": r["name"], "position": r["position"],
                    "proj": round(r["proj"] or 0), "adp": r["adp"],
                    "known_status": f"{st['status']} ({st['source']}, {st['as_of']})" if st else "none"})
    return out


def research_coverage(conn: sqlite3.Connection, season: int) -> List[dict]:
    """Per team: draft-relevant players, how many have a researched status,
    the latest research date, and fact counts — what `stale` reads."""
    from .graph import TEAM_NAMES, ensure_graph_schema

    ensure_graph_schema(conn)
    out = []
    for team in sorted(TEAM_NAMES):
        targets = [r[0] for r in conn.execute(
            "SELECT DISTINCT player_uid FROM projections WHERE season = ? AND team = ?", (season, team))]
        researched = conn.execute(
            "SELECT COUNT(DISTINCT player_uid), MAX(as_of) FROM player_status "
            "WHERE season = ? AND source != ? AND player_uid IN (SELECT player_uid FROM projections "
            "WHERE season = ? AND team = ?)", (season, ROSTER_SOURCE, season, team)).fetchone()
        facts = conn.execute(
            "SELECT COUNT(*), MAX(created) FROM facts WHERE source != 'derived' AND "
            "(entity_id = ? OR entity_id LIKE ? OR entity_id IN (SELECT 'player:' || player_uid "
            "FROM projections WHERE season = ? AND team = ?))",
            (f"team:{team}", f"unit:{team}-%", season, team)).fetchone()
        latest = max([x for x in (researched[1], (facts[1] or "")[:10]) if x] or [None])
        out.append({"team": team, "name": TEAM_NAMES[team], "targets": len(targets),
                    "players_researched": researched[0] or 0, "facts": facts[0] or 0,
                    "latest": latest})
    return out


def stale_teams(conn: sqlite3.Connection, season: int, days: int = 7) -> List[str]:
    """Teams with no research, incomplete player coverage, or research older
    than `days` — the argument to hand the research workflow."""
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out = []
    for row in research_coverage(conn, season):
        if not row["latest"] or row["latest"] < cutoff or row["players_researched"] < row["targets"]:
            out.append(row["team"])
    return out
