#!/usr/bin/env python3
"""
Build PFF scheme splits data (Man vs Zone coverage)
Analyzes route running performance against different coverage types
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

class PFFSchemeDataBuilder:
    """Build scheme splits data from PFF CSV files"""
    
    def __init__(self, csv_directory: str):
        self.csv_directory = csv_directory
        
    def find_csv_files(self, seasons: Optional[List[int]] = None) -> Dict[int, str]:
        """Find scheme CSV files for specified seasons"""
        print(f"Looking for PFF scheme CSV files in: {self.csv_directory}")
        
        csv_files = {}
        
        # Look for files with pattern: receiving_scheme_YYYY.csv
        pattern = os.path.join(self.csv_directory, "receiving_scheme_*.csv")
        found_files = glob.glob(pattern)
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            # Extract year from filename: receiving_scheme_2018.csv -> 2018
            try:
                year = int(filename.split('_')[2].split('.')[0])
                if seasons is None or year in seasons:
                    csv_files[year] = file_path
                    print(f"  ✅ Found {year}: {filename}")
            except (IndexError, ValueError):
                print(f"  ⚠️  Skipping {filename} (couldn't parse year)")
        
        return csv_files
    
    def load_scheme_data(self, csv_file: str) -> pd.DataFrame:
        """Load PFF scheme data"""
        print(f"Loading PFF scheme data from: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  ✅ Loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"  ❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def process_scheme_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Process PFF scheme data into our format"""
        print(f"Processing {season} scheme data...")
        
        # Create processed DataFrame with our expected columns
        processed_df = pd.DataFrame()
        
        # Map PFF columns to our expected format
        column_mapping = {
            'player': 'player_name',
            'player_id': 'pff_player_id', 
            'position': 'position',
            'team_name': 'team',
            'player_game_count': 'games_played',
            'base_targets': 'total_targets',
            
            # Man coverage metrics
            'man_routes': 'man_routes_run',
            'man_route_rate': 'man_route_participation',
            'man_grades_pass_route': 'man_route_grade',
            'man_targets': 'man_targets',
            'man_targets_percent': 'man_target_share',
            'man_receptions': 'man_receptions',
            'man_yards': 'man_yards',
            'man_yprr': 'man_yards_per_route_run',
            'man_avg_depth_of_target': 'man_route_depth',
            'man_contested_catch_rate': 'man_contested_catch_rate',
            'man_contested_receptions': 'man_contested_receptions',
            'man_contested_targets': 'man_contested_targets',
            'man_targeted_qb_rating': 'man_targeted_qb_rating',
            'man_caught_percent': 'man_catch_rate',
            'man_drop_rate': 'man_drop_rate',
            'man_yards_after_catch': 'man_yac',
            'man_yards_after_catch_per_reception': 'man_yac_per_reception',
            'man_yards_per_reception': 'man_yards_per_reception',
            
            # Zone coverage metrics
            'zone_routes': 'zone_routes_run',
            'zone_route_rate': 'zone_route_participation',
            'zone_grades_pass_route': 'zone_route_grade',
            'zone_targets': 'zone_targets',
            'zone_targets_percent': 'zone_target_share',
            'zone_receptions': 'zone_receptions',
            'zone_yards': 'zone_yards',
            'zone_yprr': 'zone_yards_per_route_run',
            'zone_avg_depth_of_target': 'zone_route_depth',
            'zone_contested_catch_rate': 'zone_contested_catch_rate',
            'zone_contested_receptions': 'zone_contested_receptions',
            'zone_contested_targets': 'zone_contested_targets',
            'zone_targeted_qb_rating': 'zone_targeted_qb_rating',
            'zone_caught_percent': 'zone_catch_rate',
            'zone_drop_rate': 'zone_drop_rate',
            'zone_yards_after_catch': 'zone_yac',
            'zone_yards_after_catch_per_reception': 'zone_yac_per_reception',
            'zone_yards_per_reception': 'zone_yards_per_reception'
        }
        
        # Copy mapped columns
        for pff_col, our_col in column_mapping.items():
            if pff_col in df.columns:
                processed_df[our_col] = df[pff_col]
        
        # Add season
        processed_df['season'] = season
        
        # Calculate additional metrics
        if 'man_routes_run' in processed_df.columns and 'man_targets' in processed_df.columns:
            processed_df['man_target_per_route'] = processed_df['man_targets'] / processed_df['man_routes_run']
        
        if 'zone_routes_run' in processed_df.columns and 'zone_targets' in processed_df.columns:
            processed_df['zone_target_per_route'] = processed_df['zone_targets'] / processed_df['zone_routes_run']
        
        # Calculate scheme efficiency differences
        if 'man_yards_per_route_run' in processed_df.columns and 'zone_yards_per_route_run' in processed_df.columns:
            processed_df['yprr_man_vs_zone_diff'] = processed_df['man_yards_per_route_run'] - processed_df['zone_yards_per_route_run']
        
        if 'man_route_grade' in processed_df.columns and 'zone_route_grade' in processed_df.columns:
            processed_df['route_grade_man_vs_zone_diff'] = processed_df['man_route_grade'] - processed_df['zone_route_grade']
        
        # Convert rates to percentages
        rate_columns = ['man_route_participation', 'zone_route_participation', 
                       'man_target_share', 'zone_target_share']
        for col in rate_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col] * 100
        
        print(f"  📊 Processed {len(processed_df)} scheme records")
        
        return processed_df

def build_pff_scheme_data(args: Args):
    """Build scheme splits data from PFF CSV files"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize builder
    builder = PFFSchemeDataBuilder(args.csv_dir)
    
    # Find CSV files
    csv_files = builder.find_csv_files(args.seasons)
    
    if not csv_files:
        print("❌ No scheme CSV files found")
        return
    
    # Create scheme stats table
    scheme_table_sql = """
    CREATE TABLE IF NOT EXISTS player_scheme_stats (
        season INTEGER,
        player_name TEXT,
        pff_player_id INTEGER,
        position TEXT,
        team TEXT,
        games_played INTEGER,
        total_targets INTEGER,
        
        -- Man coverage metrics
        man_routes_run INTEGER,
        man_route_participation REAL,
        man_route_grade REAL,
        man_targets INTEGER,
        man_target_share REAL,
        man_receptions INTEGER,
        man_yards INTEGER,
        man_yards_per_route_run REAL,
        man_route_depth REAL,
        man_contested_catch_rate REAL,
        man_contested_receptions INTEGER,
        man_contested_targets INTEGER,
        man_targeted_qb_rating REAL,
        man_catch_rate REAL,
        man_drop_rate REAL,
        man_yac INTEGER,
        man_yac_per_reception REAL,
        man_yards_per_reception REAL,
        man_target_per_route REAL,
        
        -- Zone coverage metrics
        zone_routes_run INTEGER,
        zone_route_participation REAL,
        zone_route_grade REAL,
        zone_targets INTEGER,
        zone_target_share REAL,
        zone_receptions INTEGER,
        zone_yards INTEGER,
        zone_yards_per_route_run REAL,
        zone_route_depth REAL,
        zone_contested_catch_rate REAL,
        zone_contested_receptions INTEGER,
        zone_contested_targets INTEGER,
        zone_targeted_qb_rating REAL,
        zone_catch_rate REAL,
        zone_drop_rate REAL,
        zone_yac INTEGER,
        zone_yac_per_reception REAL,
        zone_yards_per_reception REAL,
        zone_target_per_route REAL,
        
        -- Comparison metrics
        yprr_man_vs_zone_diff REAL,
        route_grade_man_vs_zone_diff REAL,
        
        PRIMARY KEY (season, pff_player_id)
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(scheme_table_sql))
        conn.commit()
    
    # Process each season
    total_records = 0
    season_summaries = {}
    
    for season, csv_file in sorted(csv_files.items()):
        print(f"\n{'='*60}")
        print(f"Processing {season} scheme season...")
        print(f"{'='*60}")
        
        # Load PFF data
        df = builder.load_scheme_data(csv_file)
        
        if df.empty:
            print(f"❌ Skipping {season} - could not load data")
            continue
        
        # Process scheme data
        scheme_data = builder.process_scheme_data(df, season)
        
        if scheme_data.empty:
            print(f"❌ Skipping {season} - could not process data")
            continue
        
        # Remove existing data for this season
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM player_scheme_stats WHERE season = :season"), 
                        {"season": season})
            conn.commit()
        
        # Insert new data
        scheme_data.to_sql('player_scheme_stats', engine, if_exists='append', index=False)
        
        # Store summary
        season_summaries[season] = {
            'total_players': len(scheme_data),
            'wrs': len(scheme_data[scheme_data['position'] == 'WR']),
            'tes': len(scheme_data[scheme_data['position'] == 'TE']),
            'hbs': len(scheme_data[scheme_data['position'] == 'HB']),
            'total_man_routes': scheme_data['man_routes_run'].sum() if 'man_routes_run' in scheme_data.columns else 0,
            'total_zone_routes': scheme_data['zone_routes_run'].sum() if 'zone_routes_run' in scheme_data.columns else 0,
            'avg_man_grade': scheme_data['man_route_grade'].mean() if 'man_route_grade' in scheme_data.columns else 0,
            'avg_zone_grade': scheme_data['zone_route_grade'].mean() if 'zone_route_grade' in scheme_data.columns else 0
        }
        
        total_records += len(scheme_data)
        
        print(f"✅ {season} complete: {len(scheme_data)} players")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PFF Scheme Data Import Complete!")
    print(f"{'='*60}")
    print(f"📊 Total records imported: {total_records:,}")
    print(f"📅 Seasons processed: {len(season_summaries)}")
    
    print(f"\n📈 Season Breakdown:")
    for season in sorted(season_summaries.keys()):
        summary = season_summaries[season]
        print(f"  {season}: {summary['total_players']} players, "
              f"{summary['wrs']} WRs, {summary['tes']} TEs, "
              f"{summary['total_man_routes']:,.0f} man routes, "
              f"{summary['total_zone_routes']:,.0f} zone routes, "
              f"avg man grade: {summary['avg_man_grade']:.1f}, "
              f"avg zone grade: {summary['avg_zone_grade']:.1f}")
    
    # Position totals
    total_wrs = sum(s['wrs'] for s in season_summaries.values())
    total_tes = sum(s['tes'] for s in season_summaries.values())
    total_hbs = sum(s['hbs'] for s in season_summaries.values())
    
    print(f"\n🏈 Position Totals:")
    print(f"  WRs: {total_wrs:,}")
    print(f"  TEs: {total_tes:,}")
    print(f"  HBs: {total_hbs:,}")
    
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build scheme splits data from PFF CSV files.")
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
    build_pff_scheme_data(parse_args())

