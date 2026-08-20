"""FantasyPros ADP — parses the JSON the page embeds for its ranking widget.

Rows carry FantasyPros' own numeric player ids, which the DynastyProcess
crosswalk already maps, so resolution is id-based with name matching only
as a fallback.
"""

import json
import re
import sqlite3
from typing import List, Optional

import requests

from ..names import fix_position, fix_team, record_unmatched, resolve_uid

FORMAT_URLS = {
    "std": "https://www.fantasypros.com/nfl/adp/overall.php",
    "half": "https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php",
    "ppr": "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def refresh_adp(conn: sqlite3.Connection, season: int, formats=("std", "half", "ppr")) -> int:
    total = 0
    for fmt in formats:
        resp = requests.get(FORMAT_URLS[fmt], headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        rows = _extract_rows(resp.text)
        if not rows:
            raise RuntimeError(f"No embedded ADP rows found for format {fmt} — page layout changed?")
        total += _load_rows(conn, rows, season, fmt)
    conn.commit()
    return total


def _extract_rows(html: str) -> List[dict]:
    """The ranking widget's config contains "rows":[{"id":..,"rank":..,...}].
    Decode just that array with a raw JSON decoder."""
    marker = re.search(r'"rows"\s*:\s*\[', html)
    if not marker:
        return []
    rows, _ = json.JSONDecoder().raw_decode(html, marker.end() - 1)
    return rows if isinstance(rows, list) else []


def _load_rows(conn: sqlite3.Connection, rows: List[dict], season: int, fmt: str) -> int:
    records = []
    for row in rows:
        player = row.get("player") or {}
        name = player.get("name")
        fp_id = player.get("id") or row.get("id")
        if not name:
            continue
        pos = fix_position(row.get("pos"))
        team = _team_from_label(player.get("team"))
        adp = row.get("avg")
        rank = row.get("rank")

        uid = _resolve(conn, fp_id, name, pos, team)
        if uid is None:
            record_unmatched(conn, "fantasypros_adp", name, season, pos, team, f"fp_id={fp_id} format={fmt}")
            continue
        records.append((season, "fantasypros", fmt, uid, adp, rank))

    conn.execute(
        "DELETE FROM adp WHERE season = ? AND source = 'fantasypros' AND format = ?",
        (season, fmt),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO adp (season, source, format, player_uid, adp, rank) VALUES (?, ?, ?, ?, ?, ?)",
        records,
    )
    return len(records)


def _resolve(
    conn: sqlite3.Connection,
    fp_id,
    name: str,
    pos: Optional[str],
    team: Optional[str],
) -> Optional[str]:
    if pos == "DST":
        return f"dst:{team}" if team else None
    if fp_id is not None:
        hit = conn.execute(
            "SELECT player_uid FROM player_ids WHERE source = 'fantasypros' AND source_id = ?",
            (str(fp_id),),
        ).fetchone()
        if hit:
            return hit["player_uid"]
    return resolve_uid(conn, name, pos, team)


def _team_from_label(label: Optional[str]) -> Optional[str]:
    # "CIN (6)" -> "CIN"; DSTs come through as e.g. "PHI (9)" too
    if not label:
        return None
    m = re.match(r"^([A-Z]{2,3})", label.strip())
    return fix_team(m.group(1)) if m else None
