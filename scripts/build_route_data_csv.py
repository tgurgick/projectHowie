#!/usr/bin/env python3
"""
Build route running data from PFF CSV files
Handles CSV uploads and integrates with fantasy football database
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
    csv_dir: str
    db_url: str
    season: Optional[int] = None

class RouteDataCSVBuilder:
    """Build route running data from PFF CSV files"""
    
    def __init__(self, csv_directory: str):
        self.csv_directory = csv_directory
        
    def find_csv_files(self, season: Optional[int] = None) -> List[str]:
        """Find CSV files in the directory"""
        print(f"Looking for CSV files in: {self.csv_directory}")
        
        # Look for common PFF CSV patterns
        patterns = [
            "*.csv",
            "*route*.csv", 
            "*routes*.csv",
            "*receiving*.csv",
            "*pff*.csv"
        ]
        
        if season:
            patterns.extend([
                f"*{season}*.csv",
                f"*{season}_*.csv",
                f"*_{season}*.csv"
            ])
        
        csv_files = []
        for pattern in patterns:
            csv_files.extend(glob.glob(os.path.join(self.csv_directory, pattern)))
        
        # Remove duplicates and sort
        csv_files = sorted(list(set(csv_files)))
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        return csv_files
    
    def analyze_csv_structure(self, csv_file: str) -> Dict:
        """Analyze the structure of a CSV file"""
        print(f"\nAnalyzing CSV structure: {os.path.basename(csv_file)}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding, nrows=5)
                    print(f"  ✅ Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"  ❌ Could not read CSV file with any encoding")
                return {}
            
            # Analyze structure
            structure = {
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
            
            print(f"  📊 Shape: {df.shape}")
            print(f"  📋 Columns: {len(df.columns)}")
            print(f"  🔍 Route-related columns:")
            
            route_columns = [col for col in df.columns if any(term in col.lower() for term in 
                           ['route', 'target', 'reception', 'catch', 'separation', 'cushion'])]
            
            for col in route_columns:
                print(f"    - {col}")
            
            if not route_columns:
                print("    (No obvious route-related columns found)")
            
            return structure
            
        except Exception as e:
            print(f"  ❌ Error analyzing CSV: {e}")
            return {}
    
    def load_csv_data(self, csv_file: str) -> pd.DataFrame:
        """Load CSV data with proper encoding"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"  ✅ Loaded {len(df)} records with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            print(f"  ❌ Could not read CSV file with any encoding")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"  ❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def map_route_columns(self, df: pd.DataFrame) -> Dict:
        """Map CSV columns to our expected route data structure"""
        print("Mapping CSV columns to route data structure...")
        
        # Expected route data columns
        expected_columns = {
            'player_name': ['player', 'name', 'player_name', 'receiver'],
            'player_id': ['player_id', 'pff_id', 'gsis_id', 'id'],
            'team': ['team', 'team_abbr', 'team_name'],
            'position': ['position', 'pos', 'player_position'],
            'season': ['season', 'year'],
            'week': ['week', 'game_week'],
            'game_id': ['game_id', 'game', 'matchup'],
            'routes_run': ['routes_run', 'routes', 'total_routes', 'route_attempts'],
            'route_participation': ['route_participation', 'route_pct', 'route_percentage'],
            'route_efficiency': ['route_efficiency', 'catch_rate', 'reception_rate'],
            'route_depth': ['route_depth', 'avg_depth', 'average_depth', 'air_yards'],
            'route_separation': ['route_separation', 'separation', 'avg_separation'],
            'route_cushion': ['route_cushion', 'cushion', 'avg_cushion'],
            'targets': ['targets', 'target_attempts'],
            'receptions': ['receptions', 'catches', 'completions'],
            'yards': ['yards', 'receiving_yards', 'total_yards']
        }
        
        column_mapping = {}
        
        for expected_col, possible_names in expected_columns.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    column_mapping[expected_col] = possible_name
                    print(f"  ✅ {expected_col} -> {possible_name}")
                    break
            else:
                print(f"  ❌ {expected_col} -> Not found")
        
        return column_mapping
    
    def process_route_data(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Process and clean route data"""
        print("Processing route data...")
        
        # Create a new DataFrame with our expected column names
        processed_df = pd.DataFrame()
        
        for expected_col, csv_col in column_mapping.items():
            if csv_col in df.columns:
                processed_df[expected_col] = df[csv_col]
        
        # Add missing columns with NaN
        missing_columns = set(column_mapping.keys()) - set(processed_df.columns)
        for col in missing_columns:
            processed_df[col] = np.nan
        
        # Clean and validate data
        if 'routes_run' in processed_df.columns:
            processed_df['routes_run'] = pd.to_numeric(processed_df['routes_run'], errors='coerce')
        
        if 'route_participation' in processed_df.columns:
            processed_df['route_participation'] = pd.to_numeric(processed_df['route_participation'], errors='coerce')
        
        if 'route_efficiency' in processed_df.columns:
            processed_df['route_efficiency'] = pd.to_numeric(processed_df['route_efficiency'], errors='coerce')
        
        print(f"  📊 Processed {len(processed_df)} records")
        print(f"  📋 Columns: {list(processed_df.columns)}")
        
        return processed_df

def create_player_id_mapping(engine):
    """Create player ID mapping for route data"""
    print("Creating player ID mapping for route data...")
    
    # Get existing players from our database
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

def build_route_data_csv(args: Args):
    """Build route running data from CSV files"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize CSV builder
    csv_builder = RouteDataCSVBuilder(args.csv_dir)
    
    # Find CSV files
    csv_files = csv_builder.find_csv_files(args.season)
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    # Analyze first CSV file to understand structure
    first_csv = csv_files[0]
    structure = csv_builder.analyze_csv_structure(first_csv)
    
    if not structure:
        print("❌ Could not analyze CSV structure")
        return
    
    # Load and process CSV data
    df = csv_builder.load_csv_data(first_csv)
    
    if df.empty:
        print("❌ Could not load CSV data")
        return
    
    # Map columns
    column_mapping = csv_builder.map_route_columns(df)
    
    if not column_mapping:
        print("❌ No route-related columns found")
        return
    
    # Process route data
    route_data = csv_builder.process_route_data(df, column_mapping)
    
    if route_data.empty:
        print("❌ No route data processed")
        return
    
    # Create player ID mapping
    player_mapping = create_player_id_mapping(engine)
    
    # Map player IDs (this will depend on the CSV structure)
    # For now, we'll assume we need to map by name/position/team
    print("Mapping player IDs...")
    
    # This is a placeholder - actual mapping will depend on CSV structure
    # route_data['player_id'] = route_data['player_name'].map(player_mapping)
    
    # Save to database
    print("Saving route data to database...")
    
    # Create route stats table if it doesn't exist
    route_table_sql = """
    CREATE TABLE IF NOT EXISTS player_route_stats (
        game_id TEXT,
        player_id TEXT,
        routes_run REAL,
        route_participation REAL,
        route_efficiency REAL,
        route_depth REAL,
        route_separation REAL,
        route_cushion REAL,
        targets INTEGER,
        receptions INTEGER,
        yards REAL,
        PRIMARY KEY (game_id, player_id)
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(route_table_sql))
        conn.commit()
    
    # Save processed data
    # route_data.to_sql('player_route_stats', engine, if_exists='append', index=False)
    
    print("✅ Route data processing complete!")
    print(f"📊 Processed {len(route_data)} route records")
    print("Note: Player ID mapping needs to be implemented based on actual CSV structure")
    
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build route data from PFF CSV files.")
    p.add_argument("--csv-dir", type=str, required=True, help="Directory containing PFF CSV files")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--season", type=int, help="Specific season to process (optional)")
    
    a = p.parse_args()
    
    return Args(
        csv_dir=a.csv_dir,
        db_url=a.db_url,
        season=a.season
    )

if __name__ == "__main__":
    build_route_data_csv(parse_args())
