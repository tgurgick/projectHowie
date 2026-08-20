"""Post-ingest referential-integrity checks.

SQLite FKs are not retrofitted onto bulk-loaded tables; instead the refresh
pipeline ends with this verification and reports an error if any player
reference dangles."""

import sqlite3
from typing import List, Tuple

CHECKS: List[Tuple[str, str]] = [
    (
        "weekly_stats -> players",
        "SELECT COUNT(*) FROM weekly_stats w LEFT JOIN players p USING (player_uid) "
        "WHERE p.player_uid IS NULL",
    ),
    (
        "projections -> players",
        "SELECT COUNT(*) FROM projections pr LEFT JOIN players p USING (player_uid) "
        "WHERE p.player_uid IS NULL",
    ),
    (
        "adp -> players",
        "SELECT COUNT(*) FROM adp a LEFT JOIN players p USING (player_uid) "
        "WHERE p.player_uid IS NULL",
    ),
    (
        "player_ids -> players",
        "SELECT COUNT(*) FROM player_ids i LEFT JOIN players p USING (player_uid) "
        "WHERE p.player_uid IS NULL",
    ),
    (
        "malformed uids",
        "SELECT COUNT(*) FROM players WHERE player_uid IN ('', 'mfl:nan', 'nan') "
        "OR player_uid LIKE '%nan'",
    ),
]


def verify_integrity(conn: sqlite3.Connection) -> int:
    """Returns number of checks run; raises RuntimeError listing any failures."""
    failures = []
    for label, sql in CHECKS:
        n = conn.execute(sql).fetchone()[0]
        if n:
            failures.append(f"{label}: {n} dangling/malformed rows")
    if failures:
        raise RuntimeError("; ".join(failures))
    return len(CHECKS)
