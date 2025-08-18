#!/usr/bin/env python3
"""
Build route running data from PFF 2025 receiving CSV
Custom import for the specific PFF format
"""

import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

# Import the database models and helpers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasy_db import (
    PlayerAdvancedStats, Base, ensure_dt,
    DEFAULT_DB_URL
)

@dataclass
class Args:
    csv_file: str
    db_url: str
    season: int = 2025

class PFFRouteDataBuilder:
    """Build route running data from PFF 2025 receiving CSV"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        
    def load_pff_data(self) -> pd.DataFrame:
        """Load PFF receiving data"""
        print(f"Loading PFF data from: {os.path.basename(self.csv_file)}")
        
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def process_route_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Process PFF route data into our format"""
        print("Processing PFF route data...")
        
        # Create processed DataFrame with our expected columns
        processed_df = pd.DataFrame()
        
        # Map PFF columns to our expected format
        column_mapping = {
            'player': 'player_name',
            'player_id': 'pff_player_id', 
            'position': 'position',
            'team_name': 'team',
            'player_game_count': 'games_played',
            'avg_depth_of_target': 'route_depth',
            'routes': 'routes_run',
            'route_rate': 'route_participation',
            'grades_pass_route': 'route_grade',
            'targets': 'targets',
            'receptions': 'receptions',
            'yards': 'yards',
            'yards_after_catch': 'yac',
            'yards_after_catch_per_reception': 'yac_per_reception',
            'yards_per_reception': 'yards_per_reception',
            'yprr': 'yards_per_route_run',
            'contested_catch_rate': 'contested_catch_rate',
            'contested_receptions': 'contested_receptions',
            'contested_targets': 'contested_targets',
            'targeted_qb_rating': 'targeted_qb_rating',
            'caught_percent': 'catch_rate',
            'drop_rate': 'drop_rate',
            'drops': 'drops',
            'slot_rate': 'slot_rate',
            'wide_rate': 'wide_rate',
            'inline_rate': 'inline_rate'
        }
        
        # Copy mapped columns
        for pff_col, our_col in column_mapping.items():
            if pff_col in df.columns:
                processed_df[our_col] = df[pff_col]
        
        # Add season
        processed_df['season'] = season
        
        # Calculate additional metrics
        if 'routes_run' in processed_df.columns and 'targets' in processed_df.columns:
            processed_df['target_per_route'] = processed_df['targets'] / processed_df['routes_run']
        
        if 'routes_run' in processed_df.columns and 'receptions' in processed_df.columns:
            processed_df['reception_per_route'] = processed_df['receptions'] / processed_df['routes_run']
        
        # Convert route participation to percentage
        if 'route_participation' in processed_df.columns:
            processed_df['route_participation'] = processed_df['route_participation'] * 100
        
        # Convert rates to percentages
        rate_columns = ['slot_rate', 'wide_rate', 'inline_rate']
        for col in rate_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col] * 100
        
        print(f"📊 Processed {len(processed_df)} route records")
        print(f"📋 Columns: {list(processed_df.columns)}")
        
        return processed_df
    
    def create_player_mapping(self, engine) -> Dict:
        """Create mapping from PFF player_id to our player_id"""
        print("Creating player ID mapping...")
        
        # Get our existing players
        with engine.connect() as conn:
            our_players = pd.read_sql(text("SELECT player_id, name, position, team FROM players"), conn)
        
        # Import nfl_data_py players for additional mapping
        import nfl_data_py as nfl
        nfl_players = nfl.import_players()
        
        # Create mapping dictionary
        mapping = {}
        
        # Map 1: Our player_id to nfl_data_py gsis_id
        nfl_mapping = nfl_players[['gsis_id', 'display_name', 'position', 'latest_team']].copy()
        nfl_mapping = nfl_mapping.merge(our_players, 
                                       left_on=['display_name', 'position', 'latest_team'], 
                                       right_on=['name', 'position', 'team'], 
                                       how='inner')
        for _, row in nfl_mapping.iterrows():
            mapping[row['gsis_id']] = row['player_id']
        
        print(f"Created mapping for {len(mapping)} player IDs")
        return mapping

def build_pff_route_data(args: Args):
    """Build route running data from PFF CSV"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize builder
    builder = PFFRouteDataBuilder(args.csv_file)
    
    # Load PFF data
    df = builder.load_pff_data()
    
    if df.empty:
        print("❌ Could not load PFF data")
        return
    
    # Process route data
    route_data = builder.process_route_data(df, args.season)
    
    if route_data.empty:
        print("❌ Could not process route data")
        return
    
    # Create player mapping
    player_mapping = builder.create_player_mapping(engine)
    
    # Create route stats table
    route_table_sql = """
    CREATE TABLE IF NOT EXISTS player_route_stats (
        season INTEGER,
        player_name TEXT,
        pff_player_id INTEGER,
        position TEXT,
        team TEXT,
        games_played INTEGER,
        routes_run INTEGER,
        route_participation REAL,
        route_grade REAL,
        route_depth REAL,
        targets INTEGER,
        receptions INTEGER,
        yards INTEGER,
        yac INTEGER,
        yac_per_reception REAL,
        yards_per_reception REAL,
        yards_per_route_run REAL,
        target_per_route REAL,
        reception_per_route REAL,
        contested_catch_rate REAL,
        contested_receptions INTEGER,
        contested_targets INTEGER,
        targeted_qb_rating REAL,
        catch_rate REAL,
        drop_rate REAL,
        drops INTEGER,
        slot_rate REAL,
        wide_rate REAL,
        inline_rate REAL,
        PRIMARY KEY (season, pff_player_id)
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(route_table_sql))
        conn.commit()
    
    # Save to database
    print("Saving route data to database...")
    
    # Remove existing data for this season
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM player_route_stats WHERE season = :season"), 
                    {"season": args.season})
        conn.commit()
    
    # Insert new data
    route_data.to_sql('player_route_stats', engine, if_exists='append', index=False)
    
    print("✅ PFF route data import complete!")
    print(f"📊 Imported {len(route_data)} route records for {args.season}")
    
    # Show summary
    print(f"\n📈 Route Data Summary:")
    print(f"  Total players: {len(route_data)}")
    print(f"  WRs: {len(route_data[route_data['position'] == 'WR'])}")
    print(f"  TEs: {len(route_data[route_data['position'] == 'TE'])}")
    print(f"  HBs: {len(route_data[route_data['position'] == 'HB'])}")
    print(f"  FBs: {len(route_data[route_data['position'] == 'FB'])}")
    
    if 'routes_run' in route_data.columns:
        total_routes = route_data['routes_run'].sum()
        print(f"  Total routes run: {total_routes:,}")
    
    if 'route_grade' in route_data.columns:
        avg_grade = route_data['route_grade'].mean()
        print(f"  Average route grade: {avg_grade:.1f}")
    
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build route data from PFF 2025 receiving CSV.")
    p.add_argument("--csv-file", type=str, required=True, help="PFF receiving CSV file")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--season", type=int, default=2025, help="Season year")
    
    a = p.parse_args()
    
    return Args(
        csv_file=a.csv_file,
        db_url=a.db_url,
        season=a.season
    )

if __name__ == "__main__":
    build_pff_route_data(parse_args())
