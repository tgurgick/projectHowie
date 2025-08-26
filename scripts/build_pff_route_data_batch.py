#!/usr/bin/env python3
"""
Build route running data from PFF regular season CSV files (2018-2024)
Batch processing for all seasons
"""

import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import glob

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
    csv_dir: str
    db_url: str
    seasons: Optional[List[int]] = None

class PFFRouteDataBatchBuilder:
    """Build route running data from PFF regular season CSV files"""
    
    def __init__(self, csv_directory: str):
        self.csv_directory = csv_directory
        
    def find_csv_files(self, seasons: Optional[List[int]] = None) -> Dict[int, str]:
        """Find CSV files for specified seasons"""
        print(f"Looking for PFF regular season CSV files in: {self.csv_directory}")
        
        csv_files = {}
        
        # Look for files with pattern: receiving_YYYY_reg.csv
        pattern = os.path.join(self.csv_directory, "receiving_*_reg.csv")
        found_files = glob.glob(pattern)
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            # Extract year from filename: receiving_2024_reg.csv -> 2024
            try:
                year = int(filename.split('_')[1])
                if seasons is None or year in seasons:
                    csv_files[year] = file_path
                    print(f"  ✅ Found {year}: {filename}")
            except (IndexError, ValueError):
                print(f"  ⚠️  Skipping {filename} (couldn't parse year)")
        
        return csv_files
    
    def load_pff_data(self, csv_file: str) -> pd.DataFrame:
        """Load PFF receiving data"""
        print(f"Loading PFF data from: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  ✅ Loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"  ❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def process_route_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Process PFF route data into our format"""
        print(f"Processing {season} route data...")
        
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
        
        # Note: PFF data is already in percentage format, no conversion needed
        # route_participation, slot_rate, wide_rate, inline_rate are already percentages
        
        print(f"  📊 Processed {len(processed_df)} route records")
        
        return processed_df

def build_pff_route_data_batch(args: Args):
    """Build route running data from PFF CSV files"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize builder
    builder = PFFRouteDataBatchBuilder(args.csv_dir)
    
    # Find CSV files
    csv_files = builder.find_csv_files(args.seasons)
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
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
    
    # Process each season
    total_records = 0
    season_summaries = {}
    
    for season, csv_file in sorted(csv_files.items()):
        print(f"\n{'='*60}")
        print(f"Processing {season} season...")
        print(f"{'='*60}")
        
        # Load PFF data
        df = builder.load_pff_data(csv_file)
        
        if df.empty:
            print(f"❌ Skipping {season} - could not load data")
            continue
        
        # Process route data
        route_data = builder.process_route_data(df, season)
        
        if route_data.empty:
            print(f"❌ Skipping {season} - could not process data")
            continue
        
        # Remove existing data for this season
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM player_route_stats WHERE season = :season"), 
                        {"season": season})
            conn.commit()
        
        # Insert new data
        route_data.to_sql('player_route_stats', engine, if_exists='append', index=False)
        
        # Store summary
        season_summaries[season] = {
            'total_players': len(route_data),
            'wrs': len(route_data[route_data['position'] == 'WR']),
            'tes': len(route_data[route_data['position'] == 'TE']),
            'hbs': len(route_data[route_data['position'] == 'HB']),
            'fbs': len(route_data[route_data['position'] == 'FB']),
            'total_routes': route_data['routes_run'].sum() if 'routes_run' in route_data.columns else 0,
            'avg_grade': route_data['route_grade'].mean() if 'route_grade' in route_data.columns else 0
        }
        
        total_records += len(route_data)
        
        print(f"✅ {season} complete: {len(route_data)} players")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PFF Route Data Import Complete!")
    print(f"{'='*60}")
    print(f"📊 Total records imported: {total_records:,}")
    print(f"📅 Seasons processed: {len(season_summaries)}")
    
    print(f"\n📈 Season Breakdown:")
    for season in sorted(season_summaries.keys()):
        summary = season_summaries[season]
        print(f"  {season}: {summary['total_players']} players, "
              f"{summary['wrs']} WRs, {summary['tes']} TEs, "
              f"{summary['total_routes']:,} routes, "
              f"avg grade: {summary['avg_grade']:.1f}")
    
    # Position totals
    total_wrs = sum(s['wrs'] for s in season_summaries.values())
    total_tes = sum(s['tes'] for s in season_summaries.values())
    total_hbs = sum(s['hbs'] for s in season_summaries.values())
    total_fbs = sum(s['fbs'] for s in season_summaries.values())
    
    print(f"\n🏈 Position Totals (2018-2024):")
    print(f"  WRs: {total_wrs:,}")
    print(f"  TEs: {total_tes:,}")
    print(f"  HBs: {total_hbs:,}")
    print(f"  FBs: {total_fbs:,}")
    
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build route data from PFF regular season CSV files (2018-2024).")
    p.add_argument("--csv-dir", type=str, default="data/pff_csv", help="Directory containing PFF CSV files")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--seasons", type=int, nargs="+", help="Specific seasons to process (default: all)")
    
    a = p.parse_args()
    
    return Args(
        csv_dir=a.csv_dir,
        db_url=a.db_url,
        seasons=a.seasons
    )

if __name__ == "__main__":
    build_pff_route_data_batch(parse_args())
