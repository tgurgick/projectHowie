-- Howie v3 schema. One database; scoring formats are columns, not files.
-- All player references use player_uid (gsis id when known). Name-based joins
-- are banned outside ingest-time resolution.

CREATE TABLE IF NOT EXISTS players (
    player_uid  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    name_key    TEXT NOT NULL,          -- normalized for resolution, never joined across tables
    position    TEXT,
    team        TEXT,
    birthdate   TEXT,
    draft_year  INTEGER,
    status      TEXT
);
CREATE INDEX IF NOT EXISTS idx_players_name_key ON players(name_key);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);

-- Long-format crosswalk: one row per (external system, id)
CREATE TABLE IF NOT EXISTS player_ids (
    source     TEXT NOT NULL,           -- gsis | pff | pfr | fantasypros | sleeper | espn | ...
    source_id  TEXT NOT NULL,
    player_uid TEXT NOT NULL REFERENCES players(player_uid),
    PRIMARY KEY (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_player_ids_uid ON player_ids(player_uid);

-- Manual escape hatch for name-resolution misses (e.g. nicknames, misspellings)
CREATE TABLE IF NOT EXISTS name_aliases (
    name_key   TEXT PRIMARY KEY,
    player_uid TEXT NOT NULL REFERENCES players(player_uid)
);

CREATE TABLE IF NOT EXISTS games (
    game_id   TEXT PRIMARY KEY,
    season    INTEGER NOT NULL,
    week      INTEGER NOT NULL,
    gameday   TEXT,
    home_team TEXT,
    away_team TEXT,
    spread    REAL,
    total     REAL
);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);

CREATE TABLE IF NOT EXISTS weekly_stats (
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    player_uid       TEXT NOT NULL,
    game_id          TEXT,
    team             TEXT,
    opponent         TEXT,
    position         TEXT,
    pass_attempts    REAL DEFAULT 0,
    pass_completions REAL DEFAULT 0,
    pass_yards       REAL DEFAULT 0,
    pass_tds         REAL DEFAULT 0,
    interceptions    REAL DEFAULT 0,
    rush_attempts    REAL DEFAULT 0,
    rush_yards       REAL DEFAULT 0,
    rush_tds         REAL DEFAULT 0,
    targets          REAL DEFAULT 0,
    receptions       REAL DEFAULT 0,
    rec_yards        REAL DEFAULT 0,
    rec_tds          REAL DEFAULT 0,
    fumbles_lost     REAL DEFAULT 0,
    two_pt           REAL DEFAULT 0,
    st_tds           REAL DEFAULT 0,
    pts_std          REAL,
    pts_half         REAL,
    pts_ppr          REAL,
    PRIMARY KEY (season, week, player_uid)
);
CREATE INDEX IF NOT EXISTS idx_weekly_player ON weekly_stats(player_uid, season);

-- Stat-level projections so points can be recomputed under any scoring rules
CREATE TABLE IF NOT EXISTS projections (
    season        INTEGER NOT NULL,
    source        TEXT NOT NULL,        -- pff | fantasypros | ...
    player_uid    TEXT NOT NULL,
    position      TEXT,
    team          TEXT,
    bye_week      INTEGER,
    games         REAL,
    pass_yards    REAL, pass_tds REAL, interceptions REAL,
    rush_yards    REAL, rush_tds REAL,
    targets       REAL, receptions REAL, rec_yards REAL, rec_tds REAL,
    fumbles_lost  REAL,
    two_pt        REAL,
    pts_std       REAL,
    pts_half      REAL,
    pts_ppr       REAL,
    PRIMARY KEY (season, source, player_uid)
);

CREATE TABLE IF NOT EXISTS adp (
    season     INTEGER NOT NULL,
    source     TEXT NOT NULL,
    format     TEXT NOT NULL,           -- std | half | ppr
    player_uid TEXT NOT NULL,
    adp        REAL,
    rank       INTEGER,
    stdev      REAL,
    high       INTEGER,
    low        INTEGER,
    drafts     INTEGER,
    bye_week   INTEGER,
    PRIMARY KEY (season, source, format, player_uid)
);

-- Week-grain matchup difficulty (PFF fantasy SoS exports); higher = more
-- favorable on PFF's 1-10 scale. Season/playoff aggregates are computed at
-- query time, never stored.
CREATE TABLE IF NOT EXISTS sos (
    season   INTEGER NOT NULL,
    team     TEXT NOT NULL,
    position TEXT NOT NULL,
    week     INTEGER NOT NULL,
    value    REAL,
    PRIMARY KEY (season, team, position, week)
);

-- Every ingest-time name-resolution failure lands here for triage
CREATE TABLE IF NOT EXISTS unmatched_names (
    source   TEXT NOT NULL,
    season   INTEGER,
    name     TEXT NOT NULL,
    position TEXT,
    team     TEXT,
    detail   TEXT,
    PRIMARY KEY (source, name, season)
);

CREATE TABLE IF NOT EXISTS refresh_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        TEXT NOT NULL,
    seasons     TEXT,
    rows        INTEGER,
    status      TEXT,                   -- ok | error | skipped
    detail      TEXT,
    finished_at TEXT
);
