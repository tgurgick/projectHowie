"""PFF fantasy strength-of-schedule CSVs (manual exports, one per position).

Format: one row per offense team, columns "1".."17" holding matchup
favorability for that week (empty = bye), on PFF's scale where higher is
more favorable. Aggregate columns in the file are ignored — we keep the
weekly grain and aggregate at query time.
"""

import csv
import sqlite3
from pathlib import Path

from ..names import fix_team

_POSITIONS = ("qb", "rb", "wr", "te", "dst")


def sos_files(pff_dir: Path, season: int) -> dict:
    found = {}
    for pos in _POSITIONS:
        path = pff_dir / f"{pos}-fantasy-sos_{season}_preseason.csv"
        if path.exists():
            found[pos.upper()] = path
    return found


def refresh_sos(conn: sqlite3.Connection, pff_dir: Path, season: int) -> int:
    files = sos_files(pff_dir, season)
    if not files:
        raise FileNotFoundError(
            f"No PFF SoS CSVs for {season} in {pff_dir} "
            f"(expected e.g. qb-fantasy-sos_{season}_preseason.csv)"
        )
    rows = []
    for position, path in files.items():
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                # Offense-position files key rows by "Offense"; the DST file by "Defense"
                team = fix_team(rec.get("Offense") or rec.get("Defense"))
                if not team:
                    continue
                for week in range(1, 18):
                    raw = (rec.get(str(week)) or "").strip()
                    if raw:
                        rows.append((season, team, position, week, float(raw)))
    conn.execute("DELETE FROM sos WHERE season = ?", (season,))
    conn.executemany(
        "INSERT OR REPLACE INTO sos (season, team, position, week, value) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
