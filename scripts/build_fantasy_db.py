#!/usr/bin/env python3
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Date, Boolean, JSON, Text, text,
    PrimaryKeyConstraint, MetaData
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# -------- Config & DB setup --------
load_dotenv()
DEFAULT_DB_URL = os.getenv("DB_URL", "sqlite:///data/fantasy.db")

Base = declarative_base(metadata=MetaData(schema=None))

# -------- SQLAlchemy Models (Schema we agreed) --------
class Players(Base):
    __tablename__ = "players"
    player_id = Column(String, primary_key=True)    # unified id (gsis/dp mapping)
    name = Column(String)
    position = Column(String)
    team = Column(String)
    birthdate = Column(Date, nullable=True)
    height = Column(Integer, nullable=True)   # inches
    weight = Column(Integer, nullable=True)   # pounds

class Games(Base):
    __tablename__ = "games"
    game_id = Column(String, primary_key=True)   # game_id from nflverse
    season = Column(Integer)
    week = Column(Integer)
    date = Column(Date)
    home_team = Column(String)
    away_team = Column(String)
    stadium = Column(String, nullable=True)
    spread = Column(Float, nullable=True)
    over_under = Column(Float, nullable=True)
    weather_temp = Column(Float, nullable=True)
    weather_wind = Column(Float, nullable=True)
    weather_precip = Column(Boolean, nullable=True)

class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"
    game_id = Column(String)
    player_id = Column(String)
    pass_attempts = Column(Integer, default=0)
    pass_completions = Column(Integer, default=0)
    pass_yards = Column(Integer, default=0)
    pass_tds = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    rush_attempts = Column(Integer, default=0)
    rush_yards = Column(Integer, default=0)
    rush_tds = Column(Integer, default=0)
    targets = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    rec_yards = Column(Integer, default=0)
    rec_tds = Column(Integer, default=0)
    fumbles = Column(Integer, default=0)
    fantasy_points = Column(Float, default=0.0)
    __table_args__ = (PrimaryKeyConstraint('game_id', 'player_id'),)

class PlayerAdvancedStats(Base):
    __tablename__ = "player_advanced_stats"
    game_id = Column(String)
    player_id = Column(String)
    snap_share = Column(Float)      # 0..1
    route_runs = Column(Integer)    # may be NaN (not public for all years)
    target_share = Column(Float)    # 0..1 (team share)
    air_yards = Column(Float)
    aDOT = Column(Float)
    yac = Column(Float)
    broken_tackles = Column(Integer)
    epa_per_play = Column(Float)
    cpoe = Column(Float)            # QB-only, may be NaN otherwise
    ryoe = Column(Float)            # placeholder if you add NGS later
    __table_args__ = (PrimaryKeyConstraint('game_id', 'player_id'),)

class FantasyMarket(Base):
    __tablename__ = "fantasy_market"
    season = Column(Integer)
    week = Column(Integer)
    player_id = Column(String)
    adp_overall = Column(Float)
    adp_position = Column(Float)
    ecr_rank = Column(Integer)
    __table_args__ = (PrimaryKeyConstraint('season', 'week', 'player_id'),)

class ADPData(Base):
    __tablename__ = "adp_data"
    season = Column(Integer)
    scoring_format = Column(String)  # 'ppr', 'half_ppr', 'standard'
    player_name = Column(String)
    position = Column(String)
    team = Column(String)
    bye_week = Column(Integer, nullable=True)
    rank = Column(Integer)
    adp_overall = Column(Float)
    adp_position = Column(Float)
    # Individual source ADPs
    espn_adp = Column(Float, nullable=True)
    sleeper_adp = Column(Float, nullable=True)
    cbs_adp = Column(Float, nullable=True)
    nfl_adp = Column(Float, nullable=True)
    rtsports_adp = Column(Float, nullable=True)
    fantrax_adp = Column(Float, nullable=True)
    avg_adp = Column(Float, nullable=True)
    # Metadata
    scraped_at = Column(String)  # ISO timestamp
    source_url = Column(String)
    __table_args__ = (PrimaryKeyConstraint('season', 'scoring_format', 'player_name'),)

class DraftScenarios(Base):
    __tablename__ = "draft_scenarios"
    sim_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer)
    league_id = Column(String)
    num_teams = Column(Integer)
    roster_settings = Column(JSON)  # JSON blob with starters/bench/positions
    draft_order = Column(JSON)

class DraftResults(Base):
    __tablename__ = "draft_results"
    sim_id = Column(Integer)
    pick_number = Column(Integer)
    team_id = Column(String)
    player_id = Column(String)
    __table_args__ = (PrimaryKeyConstraint('sim_id', 'pick_number'),)

# -------- Scoring Profiles --------
SCORING_PRESETS = {
    "standard": dict(
        pass_yds=0.04, pass_td=4, pass_int=-2, rush_yds=0.1, rush_td=6,
        rec_yds=0.1, rec_td=6, reception=0.0, fumble=-2
    ),
    "half_ppr": dict(
        pass_yds=0.04, pass_td=4, pass_int=-2, rush_yds=0.1, rush_td=6,
        rec_yds=0.1, rec_td=6, reception=0.5, fumble=-2
    ),
    "ppr": dict(
        pass_yds=0.04, pass_td=4, pass_int=-2, rush_yds=0.1, rush_td=6,
        rec_yds=0.1, rec_td=6, reception=1.0, fumble=-2
    ),
}

@dataclass
class Args:
    start: int
    end: int
    db_url: str
    scoring: Dict[str, float]
    custom: Dict[str, float]

# -------- Data Import Helpers --------
def import_dynastyprocess_playerids() -> pd.DataFrame:
    # maps across systems; used to create stable player_id
    url = "https://raw.githubusercontent.com/DynastyProcess/data/master/files/db_playerids.csv"
    df = pd.read_csv(url)
    # Prefer gsis_id when available, else use gsis_id-like or udf:
    df["player_id"] = df["gsis_id"].fillna(df["pfr_id"]).fillna(df["sportradar_id"])
    df = df.rename(columns={"player":"name", "position":"position"})
    return df[["player_id", "name", "position", "team"]].dropna(subset=["player_id"]).drop_duplicates()

def import_dynastyprocess_ecr() -> pd.DataFrame:
    # Expert consensus ranks (weekly, preseason)
    url = "https://raw.githubusercontent.com/DynastyProcess/data/master/files/db_fpecr.csv"
    df = pd.read_parquet(url.replace(".csv", ".parquet"), engine="pyarrow")
    # Standardize
    # columns: season, week, player, pos, ecr, etc.
    df = df.rename(columns={"ecr":"ecr_rank"})
    return df

def import_dynastyprocess_values() -> pd.DataFrame:
    # optional: global player value curves (dynasty bias); can help priors
    url = "https://raw.githubusercontent.com/DynastyProcess/data/master/files/values-players.csv"
    return pd.read_csv(url)

def compute_fantasy_points(row, scoring):
    return (
        scoring["pass_yds"] * (row.get("pass_yards", 0) or 0) +
        scoring["pass_td"]  * (row.get("pass_tds", 0) or 0) +
        scoring["pass_int"] * (row.get("interceptions", 0) or 0) +
        scoring["rush_yds"] * (row.get("rush_yards", 0) or 0) +
        scoring["rush_td"]  * (row.get("rush_tds", 0) or 0) +
        scoring["rec_yds"]  * (row.get("rec_yards", 0) or 0) +
        scoring["rec_td"]   * (row.get("rec_tds", 0) or 0) +
        scoring["reception"]* (row.get("receptions", 0) or 0) +
        scoring["fumble"]   * (row.get("fumbles", 0) or 0)
    )

def ensure_dt(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

def safe_div(num, den):
    return float(num) / float(den) if den not in [0, 0.0, None, np.nan] else np.nan

# -------- Main ETL using nfl_data_py --------
def pull_and_load(args: Args):
    # DB
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()

    # Lazy import here so script can show helpful error if lib missing
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print("Please `pip install nfl_data_py`", file=sys.stderr)
        raise

    seasons = list(range(args.start, args.end + 1))

    # --- Rosters / Players ---
    print("Importing rosters…")
    rosters = nfl.import_players()
    # Merge with DynastyProcess IDs to get a stable player_id
    dp_ids = import_dynastyprocess_playerids()
    # Map on best available keys
    players = rosters.merge(dp_ids, how="left", left_on=["gsis_id"], right_on=["player_id"], suffixes=("", "_dp"))
    # Fallback to nfl gsis_id as player_id if dp missing
    players["player_id"] = players["player_id"].fillna(players["gsis_id"])
    players["name"] = players["display_name"]
    keep_cols = ["player_id", "name", "position", "team", "birth_date", "height", "weight"]
    players_out = players[keep_cols].drop_duplicates().copy()
    ensure_dt(players_out, "birth_date")
    players_out = players_out.rename(columns={"birth_date": "birthdate"})
    # Check if players table is empty
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {Players.__tablename__}"))
        player_count = result.scalar()
    
    if player_count == 0:
        print(f"  Loading {len(players_out)} players...")
        players_out.to_sql(Players.__tablename__, engine, if_exists="append", index=False)
    else:
        print(f"  Players table already has {player_count} records, skipping...")

    # --- Schedules / Games / Vegas / Weather ---
    print("Importing schedules / lines / weather…")
    try:
        schedules = nfl.import_schedules(seasons)
        print(f"  Loaded {len(schedules)} games from {seasons[0]} to {seasons[-1]}")
    except Exception as e:
        print(f"  Error loading schedules: {e}")
        schedules = pd.DataFrame()
    
    try:
        lines = nfl.import_sc_lines(seasons)  # may be partial for older years
        print(f"  Loaded {len(lines)} betting lines")
    except Exception as e:
        print(f"  Error loading betting lines: {e}")
        lines = pd.DataFrame()
    # weather is embedded in pbp by play; for game-level, take medians from pbp later
    games = schedules[["game_id","season","week","gameday","home_team","away_team","stadium"]].copy()
    games.rename(columns={"gameday":"date"}, inplace=True)
    ensure_dt(games, "date")
    # attach vegas totals if present
    if not lines.empty:
        keep = lines.groupby("game_id").agg(
            spread=pd.NamedAgg(column="spread_line", aggfunc="last"),
            over_under=pd.NamedAgg(column="total_line", aggfunc="last")
        ).reset_index()
        games = games.merge(keep, on="game_id", how="left")
    games["weather_temp"] = np.nan
    games["weather_wind"] = np.nan
    games["weather_precip"] = np.nan
    
    if not games.empty:
        # Check which seasons already exist in games table
        with engine.connect() as conn:
            existing_seasons = pd.read_sql(text("SELECT DISTINCT season FROM games"), conn)
        
        if not existing_seasons.empty:
            existing_seasons = set(existing_seasons['season'].tolist())
            # Filter to only new seasons
            new_games = games[~games['season'].isin(existing_seasons)]
            if not new_games.empty:
                print(f"  Loading {len(new_games)} new games for seasons {list(new_games['season'].unique())}...")
                new_games.to_sql(Games.__tablename__, engine, if_exists="append", index=False)
            else:
                print(f"  All seasons {seasons} already exist in games table, skipping...")
        else:
            print(f"  Loading {len(games)} games for all seasons {seasons}...")
            games.to_sql(Games.__tablename__, engine, if_exists="append", index=False)
    else:
        print("  No games data to load")

    # --- Weekly player stats (box score level) ---
    print("Importing weekly player stats…")
    try:
        weekly = nfl.import_weekly_data(seasons)
        print(f"  Loaded {len(weekly)} weekly stat records")
    except Exception as e:
        print(f"  Error loading weekly stats: {e}")
        weekly = pd.DataFrame()
    # Standardize columns to our schema
    w = weekly.rename(columns={
        "player_id": "gsis_id",           # nfl_data_py uses gsis_id
        "attempts": "pass_attempts",
        "completions":"pass_completions",
        "passing_yards":"pass_yards",
        "passing_tds":"pass_tds",
        "interceptions":"interceptions",
        "carries":"rush_attempts",
        "rushing_yards":"rush_yards",
        "rushing_tds":"rush_tds",
        "targets":"targets",
        "receptions":"receptions",
        "receiving_yards":"rec_yards",
        "receiving_tds":"rec_tds",
        "fumbles":"fumbles",
    })
    # Weekly data doesn't have game_id, so we need to create it from season/week/teams
    # First, let's get the game mapping from the games table
    with engine.connect() as conn:
        game_mapping = pd.read_sql(text("SELECT game_id, season, week, home_team, away_team FROM games"), conn)
    
    # Create a mapping for each team's games
    home_games = game_mapping[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away_games = game_mapping[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    
    # Merge weekly stats with game mapping and player mapping
    w = w.merge(all_games, left_on=["season", "week", "recent_team"], right_on=["season", "week", "team"], how="left")
    w = w.merge(players[["gsis_id","player_id"]].drop_duplicates(), on="gsis_id", how="left")
    # Compute fantasy points
    scoring = args.scoring.copy()
    if args.custom:
        scoring.update(args.custom)
    cols_needed = ["pass_yards","pass_tds","interceptions","rush_yards","rush_tds","rec_yards","rec_tds","receptions","fumbles"]
    for c in cols_needed:
        if c not in w.columns:
            w[c] = 0
    w["fantasy_points"] = w.apply(lambda r: compute_fantasy_points(r, scoring), axis=1)
    pg = w[[
        "game_id","player_id","pass_attempts","pass_completions","pass_yards","pass_tds","interceptions",
        "rush_attempts","rush_yards","rush_tds","targets","receptions","rec_yards","rec_tds","fumbles",
        "fantasy_points"
    ]].dropna(subset=["game_id","player_id"])
    
    if not pg.empty:
        # Check which seasons already exist in player_game_stats table
        with engine.connect() as conn:
            existing_seasons = pd.read_sql(text("SELECT DISTINCT g.season FROM player_game_stats pgs JOIN games g ON pgs.game_id = g.game_id"), conn)
        
        if not existing_seasons.empty:
            existing_seasons = set(existing_seasons['season'].tolist())
            # Filter to only new seasons
            new_pg = pg[pg['game_id'].isin(games[~games['season'].isin(existing_seasons)]['game_id'])]
            if not new_pg.empty:
                print(f"  Loading {len(new_pg)} new player game stats for seasons {list(new_pg['game_id'].str[:4].unique())}...")
                new_pg.to_sql(PlayerGameStats.__tablename__, engine, if_exists="append", index=False)
            else:
                print(f"  All seasons {seasons} already exist in player_game_stats table, skipping...")
        else:
            print(f"  Loading {len(pg)} player game stats for all seasons {seasons}...")
            pg.to_sql(PlayerGameStats.__tablename__, engine, if_exists="append", index=False)
    else:
        print("  No player game stats to load")

    # --- Advanced metrics per player-game from PBP ---
    print("Skipping advanced metrics (play-by-play data) for now...")
    # Note: Advanced stats require complex player ID mapping between different data sources
    # This can be added later if needed

    # --- Fantasy market: ADP/ECR ---
    print("Importing fantasy market (ADP/ECR) from DynastyProcess…")
    ecr = import_dynastyprocess_ecr()
    # Map names/ids using DP player map again
    ecr = ecr.merge(dp_ids[["player_id","name"]].drop_duplicates(), on="player_id", how="left")
    # DynastyProcess ADP is in separate repo (ADP by site fluctuates); we’ll treat ECR as market rank.
    # For ADP you can also supplement from FantasyPros web or Sleeper export if desired.
    ecr.rename(columns={"week":"week"}, inplace=True)
    # We don’t have ADP columns here; set as NaN unless you wire in a source.
    market = ecr[["season","week","player_id","ecr_rank"]].copy()
    market["adp_overall"] = np.nan
    market["adp_position"] = np.nan
    market.to_sql(FantasyMarket.__tablename__, engine, if_exists="append", index=False)

    print("Done.")
    sess.close()
    engine.dispose()

# -------- CLI --------
def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build fantasy DB (open sources).")
    p.add_argument("--start", type=int, default=2018, help="Start season (YYYY)")
    p.add_argument("--end", type=int, default=2024, help="End season (YYYY)")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--scoring", type=str, default="ppr", choices=list(SCORING_PRESETS.keys()))
    # Optional overrides
    p.add_argument("--pass-yds", type=float, help="Points per passing yard")
    p.add_argument("--pass-td", type=float)
    p.add_argument("--pass-int", type=float)
    p.add_argument("--rush-yds", type=float)
    p.add_argument("--rush-td", type=float)
    p.add_argument("--rec-yds", type=float)
    p.add_argument("--rec-td", type=float)
    p.add_argument("--reception", type=float)
    p.add_argument("--fumble", type=float)
    a = p.parse_args()

    custom = {k.replace("_", "-"): v for k, v in vars(a).items() if k in {
        "pass_yds","pass_td","pass_int","rush_yds","rush_td","rec_yds","rec_td","reception","fumble"
    } and v is not None}
    # map back to keys used in scoring dict
    keymap = {k: k for k in ["pass_yds","pass_td","pass_int","rush_yds","rush_td","rec_yds","rec_td","reception","fumble"]}
    custom2 = {keymap[k]: v for k, v in custom.items()}

    return Args(
        start=a.start,
        end=a.end,
        db_url=a.db_url,
        scoring=SCORING_PRESETS[a.scoring],
        custom=custom2
    )

if __name__ == "__main__":
    pull_and_load(parse_args())