"""One-time port of LLM intelligence data from the legacy fantasy_ppr.db.

The 2025 research artifacts (team/position scouting, player draft intel) are
kept queryable in howie.db, tagged by their original season. Fresh 2026
research will be written by the v3 agent into the same tables.
"""

import sqlite3
from pathlib import Path

from ..names import fix_team, resolve_uid


def port_legacy_intel(conn: sqlite3.Connection, legacy_db: Path) -> int:
    if not legacy_db.exists():
        raise FileNotFoundError(f"Legacy database not found: {legacy_db}")
    legacy = sqlite3.connect(str(legacy_db))
    legacy.row_factory = sqlite3.Row
    total = 0

    for r in legacy.execute(
        "SELECT team, position, season, last_updated, intelligence_summary, "
        "usage_notes, coaching_style, injury_updates, confidence_score "
        "FROM team_position_intelligence"
    ):
        conn.execute(
            "INSERT OR REPLACE INTO team_intel "
            "(season, team, position, summary, usage_notes, coaching_style, "
            " injury_updates, confidence, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                r["season"], fix_team(r["team"]), (r["position"] or "").upper(),
                r["intelligence_summary"], r["usage_notes"], r["coaching_style"],
                r["injury_updates"], r["confidence_score"], r["last_updated"],
            ),
        )
        total += 1

    for r in legacy.execute(
        "SELECT player_name, team, position, season, is_projected_starter, "
        "injury_risk_level, injury_details, usage_notes, confidence_score, last_updated "
        "FROM player_draft_intelligence"
    ):
        team = fix_team(r["team"])
        pos = (r["position"] or "").upper()
        uid = resolve_uid(conn, r["player_name"], pos, team)
        conn.execute(
            "INSERT OR REPLACE INTO player_intel "
            "(season, player_uid, player_name, team, position, is_projected_starter, "
            " injury_risk, injury_details, usage_notes, confidence, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["season"], uid, r["player_name"], team, pos,
                r["is_projected_starter"], r["injury_risk_level"], r["injury_details"],
                r["usage_notes"], r["confidence_score"], r["last_updated"],
            ),
        )
        total += 1
    legacy.close()
    conn.commit()
    return total
