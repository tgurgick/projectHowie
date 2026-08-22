"""The knowledge plane: a property graph in SQLite + FTS5 search.

Three tables (entities, edges, facts) hold everything the engine can't get
from box scores: who's in which room, how volume was shared, what changed,
what research learned. Two layers with different trust:

- DERIVED (rebuilt every `refresh --steps graph`, provenance 'derived'):
  teams, position rooms, membership edges, last-season target/carry shares,
  vacated volume, team pass-rate YoY. Pure computation from the db.
- RESEARCHED (written by `howie graph import`, provenance kept per fact):
  scheme notes, coaching changes, o-line grades — structured records with
  confidence + source, never prose blobs.

Entity ids: "player:<uid>", "team:<abbr>", "unit:<abbr>-<POS>".
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .data.names import fix_position, fix_team, name_key, resolve_uid

ROOM_POSITIONS = ("QB", "RB", "WR", "TE")
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


# ---------------------------------------------------------------- schema

def ensure_graph_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
            name_key TEXT NOT NULL, team TEXT, position TEXT, meta TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL,
            season INTEGER, value REAL, attrs TEXT, provenance TEXT,
            PRIMARY KEY (src, dst, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, kind);
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL, kind TEXT NOT NULL, season INTEGER,
            text TEXT, value REAL, confidence REAL, source TEXT,
            created TEXT, expires TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
        """
    )
    conn.commit()


# ---------------------------------------------------------------- derived layer

def rebuild_derived(conn: sqlite3.Connection, season: int) -> int:
    """Rebuild the derived graph layer from db data. Idempotent; researched
    facts (provenance != 'derived') are preserved."""
    ensure_graph_schema(conn)
    conn.execute("DELETE FROM edges WHERE provenance = 'derived'")
    conn.execute("DELETE FROM facts WHERE source = 'derived'")
    conn.execute("DELETE FROM entities WHERE kind IN ('team', 'unit', 'player')")

    last = season - 1
    n = 0
    teams = [r["team"] for r in conn.execute(
        "SELECT DISTINCT team FROM sos WHERE season = ? ORDER BY team", (season,))]
    if not teams:
        teams = [r["home_team"] for r in conn.execute(
            "SELECT DISTINCT home_team AS home_team FROM games WHERE season = ?", (season,))]

    # team + unit entities
    for t in teams:
        conn.execute(
            "INSERT OR REPLACE INTO entities (id, kind, name, name_key, team) VALUES (?,?,?,?,?)",
            (f"team:{t}", "team", TEAM_NAMES.get(t, t), name_key(f"{t} {TEAM_NAMES.get(t, t)}"), t),
        )
        n += 1
        for pos in ROOM_POSITIONS:
            conn.execute(
                "INSERT OR REPLACE INTO entities (id, kind, name, name_key, team, position) "
                "VALUES (?,?,?,?,?,?)",
                (f"unit:{t}-{pos}", "unit", f"{t} {pos} room", name_key(f"{t} {pos} room"), t, pos),
            )
            conn.execute(
                "INSERT OR REPLACE INTO edges (src, dst, kind, season, provenance) "
                "VALUES (?,?,?,?, 'derived')",
                (f"unit:{t}-{pos}", f"team:{t}", "room_of", season),
            )
            n += 2

    # player entities: anyone with a current projection or last-season stats
    players = conn.execute(
        """
        SELECT DISTINCT p.player_uid AS uid, p.name, p.position, p.team
        FROM players p
        WHERE p.player_uid IN (SELECT player_uid FROM projections WHERE season = ?)
           OR p.player_uid IN (SELECT player_uid FROM weekly_stats WHERE season = ?)
        """,
        (season, last),
    ).fetchall()

    # last-season volume: player and team totals
    vol = {}
    for r in conn.execute(
        "SELECT player_uid, team, SUM(targets) tg, SUM(rush_attempts) ca "
        "FROM weekly_stats WHERE season = ? GROUP BY player_uid, team", (last,)):
        vol[r["player_uid"]] = (r["team"], r["tg"] or 0.0, r["ca"] or 0.0)
    team_vol = {}
    for r in conn.execute(
        "SELECT team, SUM(targets) tg, SUM(rush_attempts) ca "
        "FROM weekly_stats WHERE season = ? GROUP BY team", (last,)):
        team_vol[r["team"]] = (max(r["tg"] or 1.0, 1.0), max(r["ca"] or 1.0, 1.0))

    vacated: Dict[str, float] = {}
    for p in players:
        uid, pos, team = p["uid"], fix_position(p["position"]), fix_team(p["team"])
        conn.execute(
            "INSERT OR REPLACE INTO entities (id, kind, name, name_key, team, position) "
            "VALUES (?,?,?,?,?,?)",
            (f"player:{uid}", "player", p["name"], name_key(p["name"]), team, pos),
        )
        n += 1
        last_team, tg, ca = vol.get(uid, (None, 0.0, 0.0))
        share = None
        if last_team and last_team in team_vol and pos != "QB":
            ttg, tca = team_vol[last_team]
            share = ca / tca if pos == "RB" else tg / ttg  # QB share is not a target share
        if team and pos in ROOM_POSITIONS:
            conn.execute(
                "INSERT OR REPLACE INTO edges (src, dst, kind, season, value, attrs, provenance) "
                "VALUES (?,?,?,?,?,?, 'derived')",
                (
                    f"player:{uid}", f"unit:{team}-{pos}", "in_room", season, share,
                    json.dumps({"targets_last": tg, "carries_last": ca, "last_team": last_team}),
                ),
            )
            n += 1
        # vacated volume: produced share last season for a team they left
        if last_team and share and last_team != team:
            vacated[f"unit:{last_team}-{pos}"] = vacated.get(f"unit:{last_team}-{pos}", 0.0) + share

    for unit, share in vacated.items():
        conn.execute(
            "INSERT INTO facts (entity_id, kind, season, text, value, confidence, source, created) "
            "VALUES (?,?,?,?,?,?, 'derived', ?)",
            (
                unit, "vacated_share", season,
                f"{share:.0%} of last season's volume left this room", round(share, 3),
                1.0, _now(),
            ),
        )
        n += 1

    # team pass-rate year over year
    for t in teams:
        rates = {}
        for yr in (last - 1, last):
            r = conn.execute(
                "SELECT SUM(pass_attempts) pa, SUM(rush_attempts) ra FROM weekly_stats "
                "WHERE season = ? AND team = ?", (yr, t)).fetchone()
            if r and r["pa"]:
                rates[yr] = r["pa"] / max(r["pa"] + (r["ra"] or 0), 1)
        if last in rates and (last - 1) in rates:
            delta = rates[last] - rates[last - 1]
            conn.execute(
                "INSERT INTO facts (entity_id, kind, season, text, value, confidence, source, created) "
                "VALUES (?,?,?,?,?,?, 'derived', ?)",
                (
                    f"team:{t}", "pass_rate_yoy", season,
                    f"pass rate {rates[last]:.1%} ({'+' if delta >= 0 else ''}{delta:.1%} YoY)",
                    round(delta, 4), 1.0, _now(),
                ),
            )
            n += 1

    _rebuild_fts(conn)
    conn.commit()
    return n


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS entities_fts")
    conn.execute("CREATE VIRTUAL TABLE entities_fts USING fts5(eid UNINDEXED, name, kind, team)")
    conn.execute(
        "INSERT INTO entities_fts (eid, name, kind, team) "
        "SELECT id, name || ' ' || name_key, kind, COALESCE(team, '') FROM entities"
    )


# ---------------------------------------------------------------- search & context

def search(conn: sqlite3.Connection, q: str, limit: int = 8) -> List[dict]:
    q = q.strip()
    if not q:
        return []
    tokens = [t for t in q.split() if t]
    match = " ".join(f'"{t}"*' for t in tokens)
    try:
        rows = conn.execute(
            "SELECT e.id, e.kind, e.name, e.team, e.position FROM entities_fts f "
            "JOIN entities e ON e.id = f.eid WHERE entities_fts MATCH ? "
            "ORDER BY (e.kind = 'player') DESC, rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            "SELECT id, kind, name, team, position FROM entities WHERE name_key LIKE ? LIMIT ?",
            (f"%{name_key(q)}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def entity_context(conn: sqlite3.Connection, entity_id: str) -> dict:
    """1-hop neighborhood + facts: the compact context blob for cards/agents."""
    ensure_graph_schema(conn)
    ent = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    if ent is None:
        return {}
    out: dict = {"entity": dict(ent), "facts": [], "room": None, "team_facts": []}
    out["facts"] = [dict(r) for r in conn.execute(
        "SELECT kind, season, text, value, confidence, source, created FROM facts "
        "WHERE entity_id = ? ORDER BY id DESC LIMIT 12", (entity_id,))]

    if ent["kind"] == "player":
        room_edge = conn.execute(
            "SELECT dst, value, attrs FROM edges WHERE src = ? AND kind = 'in_room'",
            (entity_id,)).fetchone()
        if room_edge:
            unit = room_edge["dst"]
            members = conn.execute(
                "SELECT e.name, ed.value, ed.attrs FROM edges ed "
                "JOIN entities e ON e.id = ed.src "
                "WHERE ed.dst = ? AND ed.kind = 'in_room' "
                "ORDER BY COALESCE(ed.value, 0) DESC LIMIT 6",
                (unit,)).fetchall()
            unit_facts = [dict(r) for r in conn.execute(
                "SELECT kind, text, value, confidence, source FROM facts "
                "WHERE entity_id = ? ORDER BY id DESC LIMIT 6", (unit,))]
            out["room"] = {
                "unit": unit,
                "members": [
                    {"name": m["name"],
                     "share": round(m["value"], 3) if m["value"] is not None else None}
                    for m in members
                ],
                "facts": unit_facts,
            }
        if ent["team"]:
            out["team_facts"] = [dict(r) for r in conn.execute(
                "SELECT kind, text, value, confidence, source FROM facts "
                "WHERE entity_id = ? ORDER BY id DESC LIMIT 6", (f"team:{ent['team']}",))]
    return out


# ---------------------------------------------------------------- researched imports

def import_facts(conn: sqlite3.Connection, path: Path, season: int) -> int:
    """Import researched facts/edges. Contract:
    {"facts": [{"entity": "team:ARI"|"unit:ARI-TE"|"player:<name or uid>",
                "kind": "...", "text": "...", "value": 0.1?, "confidence": 0.8,
                "source": "...", "expires": "2026-09-01"?}],
     "edges": [{"src": ..., "dst": ..., "kind": ..., "value": ...}]}
    Entities referenced by player NAME are resolved through the crosswalk;
    unresolvable references fail the import (no silent drops)."""
    ensure_graph_schema(conn)
    doc = json.loads(Path(path).read_text())
    unknown = set(doc) - {"facts", "edges", "season"}
    if unknown:
        raise ValueError(f"Unknown top-level keys in fact import: {sorted(unknown)}")
    count = 0
    for f in doc.get("facts", []):
        required = {"entity", "kind", "text", "confidence", "source"}
        missing = required - set(f)
        if missing:
            raise ValueError(f"Fact missing fields {sorted(missing)}: {f}")
        eid = _resolve_entity(conn, f["entity"])
        conn.execute(
            "INSERT INTO facts (entity_id, kind, season, text, value, confidence, source, created, expires) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, f["kind"], doc.get("season", season), f["text"], f.get("value"),
             float(f["confidence"]), f["source"], _now(), f.get("expires")),
        )
        count += 1
    for e in doc.get("edges", []):
        conn.execute(
            "INSERT OR REPLACE INTO edges (src, dst, kind, season, value, attrs, provenance) "
            "VALUES (?,?,?,?,?,?,?)",
            (_resolve_entity(conn, e["src"]), _resolve_entity(conn, e["dst"]),
             e["kind"], doc.get("season", season), e.get("value"),
             json.dumps(e.get("attrs", {})), e.get("source", "import")),
        )
        count += 1
    conn.commit()
    return count


def _resolve_entity(conn: sqlite3.Connection, ref: str) -> str:
    if ":" not in ref:
        raise ValueError(f"Entity ref must be kind:identifier, got {ref!r}")
    kind, ident = ref.split(":", 1)
    if kind in ("team", "unit"):
        return f"{kind}:{ident.upper()}"
    if kind == "player":
        if ident.startswith("00-") or ident.startswith("mfl:") or ident.startswith("dst:"):
            return f"player:{ident}"
        uid = resolve_uid(conn, ident)
        if uid is None:
            # ambiguous or unknown name: prefer the draft-relevant one (has a
            # current projection), then the most recently drafted
            rows = conn.execute(
                "SELECT p.player_uid, p.draft_year, "
                "(SELECT COUNT(*) FROM projections pr WHERE pr.player_uid = p.player_uid) AS projs "
                "FROM players p WHERE p.name_key = ? ORDER BY projs DESC, p.draft_year DESC LIMIT 1",
                (name_key(ident),)).fetchone()
            if rows and rows["projs"]:
                uid = rows["player_uid"]
        if uid is None:
            raise ValueError(f"Cannot resolve player {ident!r}")
        return f"player:{uid}"
    raise ValueError(f"Unknown entity kind {kind!r}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
