"""nflverse ingest via nfl_data_py: players, schedules, weekly stats.

Weekly stats key directly on gsis ids, so no name matching is involved.
Players missing from the crosswalk are inserted from nflverse data so
weekly_stats never references an unknown uid.
"""

import sqlite3
from typing import Iterable, List

import pandas as pd

from ..names import fix_position, fix_team, name_key
from ..scoring import add_points_columns

_WEEKLY_RENAME = {
    "player_id": "player_uid",  # nfl_data_py weekly player_id IS the gsis id
    "recent_team": "team",
    "opponent_team": "opponent",
    "attempts": "pass_attempts",
    "completions": "pass_completions",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "carries": "rush_attempts",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    "special_teams_tds": "st_tds",
}

_STAT_COLS = [
    "pass_attempts", "pass_completions", "pass_yards", "pass_tds", "interceptions",
    "rush_attempts", "rush_yards", "rush_tds", "targets", "receptions",
    "rec_yards", "rec_tds", "fumbles_lost", "two_pt", "st_tds",
]


def _nfl():
    import nfl_data_py as nfl  # lazy: slow import, and optional at module load
    return nfl


def refresh_players(conn: sqlite3.Connection) -> int:
    """Upsert current player info (team/status) for anyone with a gsis id."""
    players = _nfl().import_players()
    players = players.dropna(subset=["gsis_id"])
    rows = []
    for row in players.itertuples(index=False):
        d = row._asdict()
        name = d.get("display_name") or d.get("player_name")
        if not name:
            continue
        rows.append(
            (
                d["gsis_id"],
                name,
                name_key(name),
                fix_position(d.get("position")),
                fix_team(d.get("latest_team") or d.get("team_abbr")),
                str(d.get("birth_date")) if pd.notna(d.get("birth_date")) else None,
                int(d["draft_year"]) if pd.notna(d.get("draft_year")) else None,
                d.get("status"),
            )
        )
    conn.executemany(
        "INSERT INTO players (player_uid, name, name_key, position, team, birthdate, draft_year, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(player_uid) DO UPDATE SET "
        "  name = excluded.name, name_key = excluded.name_key, "
        "  position = COALESCE(excluded.position, players.position), "
        "  team = COALESCE(excluded.team, players.team), "
        "  birthdate = COALESCE(excluded.birthdate, players.birthdate), "
        "  draft_year = COALESCE(excluded.draft_year, players.draft_year), "
        "  status = excluded.status",
        rows,
    )
    # gsis is also an external id in the crosswalk sense
    conn.executemany(
        "INSERT OR IGNORE INTO player_ids (source, source_id, player_uid) VALUES ('gsis', ?, ?)",
        [(r[0], r[0]) for r in rows],
    )
    conn.commit()
    return len(rows)


def refresh_games(conn: sqlite3.Connection, seasons: List[int]) -> int:
    sched = _nfl().import_schedules(seasons)
    rows = [
        (
            d["game_id"], int(d["season"]), int(d["week"]), str(d.get("gameday")),
            fix_team(d.get("home_team")), fix_team(d.get("away_team")),
            float(d["spread_line"]) if pd.notna(d.get("spread_line")) else None,
            float(d["total_line"]) if pd.notna(d.get("total_line")) else None,
        )
        for d in (r._asdict() for r in sched.itertuples(index=False))
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO games (game_id, season, week, gameday, home_team, away_team, spread, total) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


# nflverse's newer stats pipeline publishes under this release tag; nfl_data_py
# still points at the retired player_stats assets for recent seasons.
_STATS_PLAYER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)
_NEW_SCHEMA_RENAME = {
    "player_display_name": "player_name",
    "passing_interceptions": "interceptions",
}


def _fetch_weekly(season: int) -> pd.DataFrame:
    try:
        return _nfl().import_weekly_data([season])
    except Exception:
        weekly = pd.read_parquet(_STATS_PLAYER_URL.format(season=season))
        return weekly.rename(columns=_NEW_SCHEMA_RENAME)


def refresh_weekly(conn: sqlite3.Connection, seasons: Iterable[int]) -> int:
    """Weekly box-score stats with all three scoring formats. Per-season loop so
    one bad/missing season (e.g. data not yet published) doesn't sink the rest."""
    total = 0
    errors = []
    for season in seasons:
        try:
            weekly = _fetch_weekly(season)
        except Exception as e:  # per-season resilience by design
            errors.append(f"{season}: {e}")
            continue
        total += _load_weekly_frame(conn, season, weekly)
    if errors and total == 0:
        raise RuntimeError("; ".join(errors))
    if errors:
        print(f"  weekly: skipped seasons -> {'; '.join(errors)}")
    return total


def _load_weekly_frame(conn: sqlite3.Connection, season: int, weekly: pd.DataFrame) -> int:
    w = weekly.rename(columns=_WEEKLY_RENAME).copy()
    w = w[w.get("season_type", "REG") == "REG"] if "season_type" in w.columns else w
    w = w.dropna(subset=["player_uid"])

    for part in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"):
        if part not in w.columns:
            w[part] = 0.0
    w["fumbles_lost"] = (
        w["sack_fumbles_lost"].fillna(0)
        + w["rushing_fumbles_lost"].fillna(0)
        + w["receiving_fumbles_lost"].fillna(0)
    )
    for part in ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"):
        if part not in w.columns:
            w[part] = 0.0
    w["two_pt"] = (
        w["passing_2pt_conversions"].fillna(0)
        + w["rushing_2pt_conversions"].fillna(0)
        + w["receiving_2pt_conversions"].fillna(0)
    )
    for c in _STAT_COLS:
        if c not in w.columns:
            w[c] = 0.0
    w = add_points_columns(w)
    w["team"] = w["team"].map(fix_team)
    w["opponent"] = w["opponent"].map(fix_team)

    # Attach game_id via (season, week, team); some source schemas ship their
    # own game_id column, which would collide in the merge
    w = w.drop(columns=["game_id"], errors="ignore")
    games = pd.read_sql_query(
        "SELECT game_id, season, week, home_team, away_team FROM games WHERE season = ?",
        conn, params=(season,),
    )
    team_games = pd.concat(
        [
            games[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
            games[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    w = w.merge(team_games, on=["season", "week", "team"], how="left")

    # Any uid not yet in players gets a stub row so the reference is never dangling
    known = {r[0] for r in conn.execute("SELECT player_uid FROM players")}
    missing = w[~w["player_uid"].isin(known)][["player_uid", "player_name", "position"]].drop_duplicates("player_uid")
    conn.executemany(
        "INSERT OR IGNORE INTO players (player_uid, name, name_key, position) VALUES (?, ?, ?, ?)",
        [
            (d["player_uid"], d["player_name"], name_key(str(d["player_name"])), fix_position(d.get("position")))
            for d in (r._asdict() for r in missing.itertuples(index=False))
            if pd.notna(d.get("player_name"))
        ],
    )

    cols = ["season", "week", "player_uid", "game_id", "team", "opponent", "position"] + _STAT_COLS + ["pts_std", "pts_half", "pts_ppr"]
    records = w[cols].where(pd.notna(w[cols]), None).values.tolist()
    conn.executemany(
        f"INSERT OR REPLACE INTO weekly_stats ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        records,
    )
    conn.commit()
    return len(records)
