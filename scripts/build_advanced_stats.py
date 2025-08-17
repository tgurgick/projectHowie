#!/usr/bin/env python3
"""
Build advanced stats for fantasy football database
Handles player ID mapping between different data sources
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
    PlayerAdvancedStats, Base, ensure_dt,
    DEFAULT_DB_URL
)

@dataclass
class Args:
    start: int
    end: int
    db_url: str

def safe_div(a, b):
    """Safe division that returns NaN if denominator is 0"""
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def create_player_id_mapping(engine):
    """Create comprehensive player ID mapping between different sources"""
    print("Creating player ID mapping...")
    
    # Get existing players from our database
    with engine.connect() as conn:
        our_players = pd.read_sql(text("SELECT player_id, name, position, team FROM players"), conn)
    
    # Import nfl_data_py players
    import nfl_data_py as nfl
    nfl_players = nfl.import_players()
    
    # Import DynastyProcess IDs
    from build_fantasy_db import import_dynastyprocess_playerids
    dp_ids = import_dynastyprocess_playerids()
    
    # Create mapping dictionary
    mapping = {}
    
    # Map 1: Our player_id to nfl_data_py gsis_id
    # nfl_data_py uses gsis_id as player_id
    nfl_mapping = nfl_players[['gsis_id', 'display_name', 'position', 'latest_team']].copy()
    nfl_mapping = nfl_mapping.merge(our_players, 
                                   left_on=['display_name', 'position', 'latest_team'], 
                                   right_on=['name', 'position', 'team'], 
                                   how='inner')
    for _, row in nfl_mapping.iterrows():
        mapping[row['gsis_id']] = row['player_id']
    
    # Map 2: DynastyProcess player_id to our player_id
    dp_mapping = dp_ids.merge(our_players, on='player_id', how='inner')
    for _, row in dp_mapping.iterrows():
        mapping[row['player_id']] = row['player_id']  # Already mapped
    
    # Map 3: PFR player IDs (from snap counts)
    try:
        snaps = nfl.import_snap_counts([2024])  # Use recent year for mapping
        pfr_mapping = snaps[['pfr_player_id', 'player', 'position', 'team']].copy()
        pfr_mapping = pfr_mapping.merge(our_players, 
                                       left_on=['player', 'position', 'team'], 
                                       right_on=['name', 'position', 'team'], 
                                       how='inner')
        for _, row in pfr_mapping.iterrows():
            mapping[row['pfr_player_id']] = row['player_id']
    except Exception as e:
        print(f"Warning: Could not create PFR mapping: {e}")
    
    print(f"Created mapping for {len(mapping)} player IDs")
    return mapping

def build_advanced_stats(args: Args):
    """Build advanced stats with proper player ID mapping"""
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
    print(f"Building advanced stats for seasons: {seasons}")

    # Create player ID mapping
    player_mapping = create_player_id_mapping(engine)
    
    # --- Play-by-play data ---
    print("Importing play-by-play data...")
    try:
        pbp = nfl.import_pbp_data(seasons)
        print(f"  Loaded {len(pbp)} play-by-play records")
    except Exception as e:
        print(f"  Error loading play-by-play: {e}")
        return
    
    # --- Build receiver stats (targets, air yards, YAC) ---
    print("Building receiver stats...")
    recv = pbp[pbp["pass_attempt"] == 1].copy()
    
    # Map receiver IDs
    recv['player_id'] = recv['receiver_player_id'].map(player_mapping)
    recv = recv.dropna(subset=['player_id'])
    
    if not recv.empty:
        recv_stats = recv.groupby(["game_id","player_id"], dropna=True).agg(
            targets=("receiver_player_id", "count"),
            air_yards=("air_yards", "sum"),
            yac=("yards_after_catch", "sum")
        ).reset_index()
        print(f"  Created {len(recv_stats)} receiver stat records")
    else:
        recv_stats = pd.DataFrame(columns=["game_id","player_id","targets","air_yards","yac"])
        print("  No receiver stats created")

    # --- Build QB stats (EPA, CPOE) ---
    print("Building QB stats...")
    qb = pbp[pbp["pass_attempt"] == 1].copy()
    
    # Map passer IDs
    qb['player_id'] = qb['passer_player_id'].map(player_mapping)
    qb = qb.dropna(subset=['player_id'])
    
    if not qb.empty:
        qb_stats = qb.groupby(["game_id","player_id"], dropna=True).agg(
            qb_epa_per_play=("epa", "mean"),
            cpoe=("cpoe", "mean")
        ).reset_index()
        print(f"  Created {len(qb_stats)} QB stat records")
    else:
        qb_stats = pd.DataFrame(columns=["game_id","player_id","qb_epa_per_play","cpoe"])
        print("  No QB stats created")

    # --- Build rusher stats (EPA) ---
    print("Building rusher stats...")
    rush = pbp[pbp["rush_attempt"] == 1].copy()
    
    # Map rusher IDs
    rush['player_id'] = rush['rusher_player_id'].map(player_mapping)
    rush = rush.dropna(subset=['player_id'])
    
    if not rush.empty:
        rush_stats = rush.groupby(["game_id","player_id"], dropna=True).agg(
            rush_epa_per_play=("epa", "mean")
        ).reset_index()
        print(f"  Created {len(rush_stats)} rusher stat records")
    else:
        rush_stats = pd.DataFrame(columns=["game_id","player_id","rush_epa_per_play"])
        print("  No rusher stats created")

    # --- Team totals for shares ---
    print("Building team totals...")
    team_targets = pbp[pbp["pass_attempt"] == 1].groupby(["game_id","posteam"]).size().reset_index(name="team_targets")
    
    # --- Snap counts ---
    print("Importing snap counts...")
    try:
        snaps = nfl.import_snap_counts(seasons)
        print(f"  Loaded {len(snaps)} snap count records")
        
        # Map player IDs
        snaps['player_id'] = snaps['pfr_player_id'].map(player_mapping)
        snaps = snaps.dropna(subset=['player_id'])
        
        if not snaps.empty:
            # Calculate team offense snaps and snap share
            team_snaps = snaps.groupby(['game_id', 'team'])['offense_snaps'].sum().reset_index()
            team_snaps = team_snaps.rename(columns={'offense_snaps': 'team_offense_snaps'})
            snaps = snaps.merge(team_snaps, on=['game_id', 'team'], how='left')
            snaps["snap_share"] = snaps["offense_snaps"] / snaps["team_offense_snaps"]
            
            snap_stats = snaps[["game_id","player_id","snap_share"]].copy()
            print(f"  Created {len(snap_stats)} snap stat records")
        else:
            snap_stats = pd.DataFrame(columns=["game_id","player_id","snap_share"])
            print("  No snap stats created")
            
    except Exception as e:
        print(f"  Error loading snap counts: {e}")
        snap_stats = pd.DataFrame(columns=["game_id","player_id","snap_share"])

    # --- Merge all advanced features ---
    print("Merging advanced features...")
    
    # Start with receiver stats
    adv = recv_stats.copy()
    
    # Add target share
    if not adv.empty and not team_targets.empty:
        adv = adv.merge(team_targets, left_on=["game_id"], right_on=["game_id"], how="left")
        adv["target_share"] = adv.apply(lambda r: safe_div(r["targets"], r["team_targets"]), axis=1)
        adv = adv.drop(columns=["posteam"])
    
    # Merge QB stats
    if not qb_stats.empty:
        adv = adv.merge(qb_stats, on=["game_id","player_id"], how="outer")
    
    # Merge rusher stats
    if not rush_stats.empty:
        adv = adv.merge(rush_stats, on=["game_id","player_id"], how="outer")
    
    # Combine EPA values (QB takes precedence if both exist)
    if 'qb_epa_per_play' in adv.columns and 'rush_epa_per_play' in adv.columns:
        adv['epa_per_play'] = adv['qb_epa_per_play'].fillna(adv['rush_epa_per_play'])
    elif 'qb_epa_per_play' in adv.columns:
        adv['epa_per_play'] = adv['qb_epa_per_play']
    elif 'rush_epa_per_play' in adv.columns:
        adv['epa_per_play'] = adv['rush_epa_per_play']
    else:
        adv['epa_per_play'] = np.nan
    
    # Merge snap stats
    if not snap_stats.empty:
        adv = adv.merge(snap_stats, on=["game_id","player_id"], how="left")
    
    # Calculate aDOT
    adv["aDOT"] = adv.apply(lambda r: safe_div(r.get("air_yards", np.nan), r.get("targets", np.nan)), axis=1)
    
    # Add placeholders for missing public data
    adv["route_runs"] = np.nan
    adv["broken_tackles"] = np.nan
    adv["ryoe"] = np.nan
    
    # Select final columns (only include columns that exist)
    expected_columns = [
        "game_id","player_id","snap_share","route_runs","target_share","air_yards","aDOT","yac",
        "broken_tackles","epa_per_play","cpoe","ryoe"
    ]
    
    available_columns = [col for col in expected_columns if col in adv.columns]
    missing_columns = [col for col in expected_columns if col not in adv.columns]
    
    if missing_columns:
        print(f"  Warning: Missing columns: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            adv[col] = np.nan
    
    adv_out = adv[expected_columns].dropna(subset=["game_id","player_id"], how="any")
    
    # Deduplicate by game_id and player_id, keeping the first occurrence
    adv_out = adv_out.drop_duplicates(subset=["game_id", "player_id"], keep="first")
    
    print(f"  Final advanced stats: {len(adv_out)} unique player-game records")
    
    # Check which seasons already exist
    with engine.connect() as conn:
        existing_seasons = pd.read_sql(text("SELECT DISTINCT g.season FROM player_advanced_stats pas JOIN games g ON pas.game_id = g.game_id"), conn)
    
    if not existing_seasons.empty:
        existing_seasons = set(existing_seasons['season'].tolist())
        # Filter to only new seasons
        new_adv = adv_out[adv_out['game_id'].str[:4].astype(int).isin([s for s in seasons if s not in existing_seasons])]
        if not new_adv.empty:
            print(f"  Loading {len(new_adv)} new advanced stats for seasons {list(new_adv['game_id'].str[:4].unique())}...")
            new_adv.to_sql(PlayerAdvancedStats.__tablename__, engine, if_exists="append", index=False)
        else:
            print(f"  All seasons {seasons} already exist in advanced stats table, skipping...")
    else:
        print(f"  Loading {len(adv_out)} advanced stats for all seasons {seasons}...")
        adv_out.to_sql(PlayerAdvancedStats.__tablename__, engine, if_exists="append", index=False)
    
    print("✅ Advanced stats loaded successfully!")
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build advanced stats for fantasy DB.")
    p.add_argument("--start", type=int, default=2018, help="Start season (YYYY)")
    p.add_argument("--end", type=int, default=2024, help="End season (YYYY)")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    
    a = p.parse_args()
    
    return Args(
        start=a.start,
        end=a.end,
        db_url=a.db_url
    )

if __name__ == "__main__":
    build_advanced_stats(parse_args())
