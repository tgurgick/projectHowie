"""Fantasy Football Calculator ADP — public JSON API, real mock-draft data.

Primary ADP source: no scraping, no key, and it reports adp stdev/high/low,
which feeds the pick-availability model directly.
"""

import sqlite3
from typing import Optional

import requests

from ..names import fix_position, fix_team, record_unmatched, resolve_uid

_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
_FORMAT_PATH = {"std": "standard", "half": "half-ppr", "ppr": "ppr"}


def refresh_adp(
    conn: sqlite3.Connection,
    season: int,
    num_teams: int = 12,
    formats=("std", "half", "ppr"),
) -> int:
    total = 0
    for fmt in formats:
        resp = requests.get(
            _URL.format(fmt=_FORMAT_PATH[fmt]),
            params={"teams": num_teams, "year": season},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "Success":
            raise RuntimeError(f"FFC ADP request failed for {fmt}: {payload.get('status')}")

        rows = []
        for rank, p in enumerate(payload.get("players", []), start=1):
            uid = _resolve(conn, p, season, fmt)
            if uid is None:
                continue
            rows.append(
                (
                    season, "ffc", fmt, uid, p.get("adp"), rank,
                    p.get("stdev"), p.get("high"), p.get("low"),
                    p.get("times_drafted"), p.get("bye"),
                )
            )
        conn.execute("DELETE FROM adp WHERE season = ? AND source = 'ffc' AND format = ?", (season, fmt))
        conn.executemany(
            "INSERT OR REPLACE INTO adp "
            "(season, source, format, player_uid, adp, rank, stdev, high, low, drafts, bye_week) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        total += len(rows)
    conn.commit()
    return total


def _resolve(conn: sqlite3.Connection, p: dict, season: int, fmt: str) -> Optional[str]:
    name = p.get("name")
    if not name:
        return None
    pos = fix_position(p.get("position"))
    team = fix_team(p.get("team"))
    if pos == "DST":
        return f"dst:{team}" if team else None
    uid = resolve_uid(conn, name, pos, team)
    if uid is None:
        record_unmatched(conn, "ffc_adp", name, season, pos, team, f"format={fmt}")
    return uid
