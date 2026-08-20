"""DynastyProcess player-ID crosswalk — seeds players + player_ids.

db_playerids.csv is a maintained many-system ID mapping (gsis, pff, pfr,
fantasypros, sleeper, espn, ...). It is the backbone of player identity in
howie.db: every other source attaches to these uids either directly by ID
or via name resolution.
"""

import sqlite3

import pandas as pd

from ..names import fix_position, fix_team, name_key

URL = "https://raw.githubusercontent.com/DynastyProcess/data/master/files/db_playerids.csv"

# csv column -> our player_ids.source label
ID_COLUMNS = {
    "gsis_id": "gsis",
    "pff_id": "pff",
    "pfr_id": "pfr",
    "fantasypros_id": "fantasypros",
    "sleeper_id": "sleeper",
    "espn_id": "espn",
    "yahoo_id": "yahoo",
    "nfl_id": "nfl",
    "cbs_id": "cbs",
    "rotowire_id": "rotowire",
    "mfl_id": "mfl",
    "sportradar_id": "sportradar",
}


def refresh_crosswalk(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(URL, dtype=str, low_memory=False)

    # uid = gsis id when present, else a namespaced mfl id; rows with neither
    # have no stable identity and are dropped rather than becoming "mfl:nan"
    has_gsis = df["gsis_id"].notna() & (df["gsis_id"].str.strip() != "")
    has_mfl = df["mfl_id"].notna() & (df["mfl_id"].str.strip() != "")
    df = df[has_gsis | has_mfl].copy()
    df["player_uid"] = df["gsis_id"].where(
        df["gsis_id"].notna() & (df["gsis_id"].str.strip() != ""),
        "mfl:" + df["mfl_id"].astype(str),
    )
    df = df.dropna(subset=["name"]).drop_duplicates(subset=["player_uid"])

    player_rows = []
    id_rows = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        uid = d["player_uid"]
        player_rows.append(
            (
                uid,
                d["name"],
                name_key(d["name"]),
                fix_position(d.get("position")),
                fix_team(d.get("team")),
                d.get("birthdate"),
                int(float(d["draft_year"])) if d.get("draft_year") and pd.notna(d["draft_year"]) else None,
                None,
            )
        )
        for col, source in ID_COLUMNS.items():
            val = d.get(col)
            if val is not None and pd.notna(val) and str(val).strip():
                id_rows.append((source, str(val).strip(), uid))

    conn.executemany(
        "INSERT INTO players (player_uid, name, name_key, position, team, birthdate, draft_year, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(player_uid) DO UPDATE SET "
        "  name = excluded.name, name_key = excluded.name_key, "
        "  position = COALESCE(excluded.position, players.position), "
        "  team = COALESCE(excluded.team, players.team), "
        "  birthdate = COALESCE(excluded.birthdate, players.birthdate), "
        "  draft_year = COALESCE(excluded.draft_year, players.draft_year)",
        player_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_ids (source, source_id, player_uid) VALUES (?, ?, ?)",
        id_rows,
    )
    conn.commit()
    return len(player_rows)
