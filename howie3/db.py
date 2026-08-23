"""SQLite connection + schema management for howie.db."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        conn.executescript(_SCHEMA_PATH.read_text())
    if version == 1:
        # v2: ADP spread columns (FFC reports mock-draft variance we need for
        # pick-availability modeling)
        for col, typ in [
            ("stdev", "REAL"), ("high", "INTEGER"), ("low", "INTEGER"),
            ("drafts", "INTEGER"), ("bye_week", "INTEGER"),
        ]:
            conn.execute(f"ALTER TABLE adp ADD COLUMN {col} {typ}")
    if version <= 2:
        # v3: sos moves from season-grain to week-grain (table was empty)
        conn.execute("DROP TABLE IF EXISTS sos")
        conn.execute(
            "CREATE TABLE sos (season INTEGER NOT NULL, team TEXT NOT NULL, "
            "position TEXT NOT NULL, week INTEGER NOT NULL, value REAL, "
            "PRIMARY KEY (season, team, position, week))"
        )
    if version <= 3:
        # v4: intelligence tables (LLM research artifacts ported from legacy db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS team_intel ("
            "season INTEGER NOT NULL, team TEXT NOT NULL, position TEXT NOT NULL, "
            "summary TEXT, usage_notes TEXT, coaching_style TEXT, injury_updates TEXT, "
            "confidence REAL, updated_at TEXT, PRIMARY KEY (season, team, position))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS player_intel ("
            "season INTEGER NOT NULL, player_uid TEXT, player_name TEXT NOT NULL, "
            "team TEXT NOT NULL, position TEXT, is_projected_starter INTEGER, "
            "injury_risk TEXT, injury_details TEXT, usage_notes TEXT, "
            "confidence REAL, updated_at TEXT, "
            "PRIMARY KEY (season, player_name, team))"
        )
    if version <= 4:
        # v5: machine-actionable player status (injury / suspension / cut risk /
        # role) from the nflverse roster feed and the research workflow. One
        # row per (player, as_of); the latest row is the current status.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS player_status ("
            "season INTEGER NOT NULL, player_uid TEXT NOT NULL, as_of TEXT NOT NULL, "
            "status TEXT NOT NULL, games_out INTEGER NOT NULL DEFAULT 0, injury TEXT, "
            "role TEXT, cut_risk REAL NOT NULL DEFAULT 0, note TEXT, confidence REAL, "
            "source TEXT NOT NULL, PRIMARY KEY (season, player_uid, as_of, source))"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_status ON player_status(season, player_uid)")
    if version != SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
