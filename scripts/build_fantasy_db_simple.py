#!/usr/bin/env python3
"""
Simplified fantasy football database builder
Focuses on core data: players, games, and player game stats
"""

import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict

# Import the database models and helpers from the main script
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasy_db import (
    Players, Games, PlayerGameStats, Base,
    import_dynastyprocess_playerids, compute_fantasy_points, ensure_dt,
    SCORING_PRESETS, DEFAULT_DB_URL
)

@dataclass
class Args:
    start: int
    end: int
    db_url: str
    scoring: Dict[str, float]
    custom: Dict[str, float]

def pull_and_load_simple(args: Args):
    """Simplified ETL focusing on core data"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Import nfl_data_py
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print("Please `pip install nfl_data_py`", file=sys.stderr)
        raise

    seasons = list(range(args.start, args.end + 1))
    print(f"Processing seasons: {seasons}")

    # --- Players (load once) ---
    print("Importing rosters…")
    rosters = nfl.import_players()
    dp_ids = import_dynastyprocess_playerids()
    players = rosters.merge(dp_ids, how="left", left_on=["gsis_id"], right_on=["player_id"], suffixes=("", "_dp"))
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

    # --- Games (incremental) ---
    print("Importing schedules…")
    try:
        schedules = nfl.import_schedules(seasons)
        print(f"  Loaded {len(schedules)} games from {seasons[0]} to {seasons[-1]}")
    except Exception as e:
        print(f"  Error loading schedules: {e}")
        schedules = pd.DataFrame()
    
    if not schedules.empty:
        games = schedules[["game_id","season","week","gameday","home_team","away_team","stadium"]].copy()
        games.rename(columns={"gameday":"date"}, inplace=True)
        ensure_dt(games, "date")
        games["weather_temp"] = np.nan
        games["weather_wind"] = np.nan
        games["weather_precip"] = np.nan
        
        # Check which seasons already exist
        with engine.connect() as conn:
            existing_seasons = pd.read_sql(text("SELECT DISTINCT season FROM games"), conn)
        
        if not existing_seasons.empty:
            existing_seasons = set(existing_seasons['season'].tolist())
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

    # --- Player Game Stats (incremental) ---
    print("Importing weekly player stats…")
    try:
        weekly = nfl.import_weekly_data(seasons)
        print(f"  Loaded {len(weekly)} weekly stat records")
    except Exception as e:
        print(f"  Error loading weekly stats: {e}")
        weekly = pd.DataFrame()
    
    if not weekly.empty:
        # Standardize columns
        w = weekly.rename(columns={
            "player_id": "gsis_id",
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
        
        # Get game mapping and merge
        with engine.connect() as conn:
            game_mapping = pd.read_sql(text("SELECT game_id, season, week, home_team, away_team FROM games"), conn)
        
        home_games = game_mapping[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"})
        away_games = game_mapping[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})
        all_games = pd.concat([home_games, away_games], ignore_index=True)
        
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
        
        # Check which seasons already exist
        with engine.connect() as conn:
            existing_seasons = pd.read_sql(text("SELECT DISTINCT g.season FROM player_game_stats pgs JOIN games g ON pgs.game_id = g.game_id"), conn)
        
        if not existing_seasons.empty:
            existing_seasons = set(existing_seasons['season'].tolist())
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

    print("✅ Done! Core data loaded successfully.")
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build fantasy DB (core data only).")
    p.add_argument("--start", type=int, default=2018, help="Start season (YYYY)")
    p.add_argument("--end", type=int, default=2024, help="End season (YYYY)")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--scoring", type=str, default="ppr", choices=list(SCORING_PRESETS.keys()))
    
    a = p.parse_args()
    
    return Args(
        start=a.start,
        end=a.end,
        db_url=a.db_url,
        scoring=SCORING_PRESETS[a.scoring],
        custom={}
    )

if __name__ == "__main__":
    pull_and_load_simple(parse_args())
