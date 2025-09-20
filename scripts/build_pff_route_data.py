#!/usr/bin/env python3
"""
Build PFF route running data for fantasy football database
Integrates with Pro Football Focus API for comprehensive route metrics
"""

import sys
import argparse
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

# Import the database models and helpers from the main script
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasy_db import (
    PlayerAdvancedStats, Base, ensure_dt,
    DEFAULT_DB_URL
)
try:
    from howie_cli.core.paths import get_db_url
    _DEFAULT_URL = os.getenv("DB_URL", get_db_url("ppr"))
except Exception:
    _DEFAULT_URL = DEFAULT_DB_URL

@dataclass
class Args:
    start: int
    end: int
    db_url: str
    pff_api_key: str
    pff_base_url: str = "https://api.profootballfocus.com/v1"

class PFFRouteDataBuilder:
    """Build route running data from PFF API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.profootballfocus.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_route_data(self, season: int) -> pd.DataFrame:
        """Get route running data for a specific season"""
        print(f"Fetching PFF route data for {season}...")
        
        try:
            # PFF API endpoint for route data
            # Note: This is a placeholder - actual endpoint will depend on PFF API structure
            url = f"{self.base_url}/stats/routes"
            params = {
                "season": season,
                "format": "json"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            print(f"  Retrieved {len(df)} route records for {season}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching PFF data for {season}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Unexpected error for {season}: {e}")
            return pd.DataFrame()
    
    def get_available_seasons(self) -> List[int]:
        """Get list of available seasons from PFF"""
        try:
            url = f"{self.base_url}/seasons"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            seasons = [int(season) for season in data.get("seasons", [])]
            return seasons
            
        except Exception as e:
            print(f"Error fetching available seasons: {e}")
            return []
    
    def get_route_metrics(self, seasons: List[int]) -> pd.DataFrame:
        """Get route metrics for multiple seasons"""
        all_data = []
        
        for season in seasons:
            season_data = self.get_route_data(season)
            if not season_data.empty:
                all_data.append(season_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Combined {len(combined_data)} total route records")
            return combined_data
        else:
            print("No route data retrieved")
            return pd.DataFrame()

def create_player_id_mapping(engine):
    """Create player ID mapping for PFF data"""
    print("Creating player ID mapping for PFF data...")
    
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

def build_pff_route_data(args: Args):
    """Build PFF route running data with proper player ID mapping"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize PFF client
    pff_client = PFFRouteDataBuilder(args.pff_api_key, args.pff_base_url)
    
    # Check available seasons
    available_seasons = pff_client.get_available_seasons()
    if available_seasons:
        print(f"Available PFF seasons: {available_seasons}")
    
    # Get route data for requested seasons
    seasons = list(range(args.start, args.end + 1))
    print(f"Building PFF route data for seasons: {seasons}")
    
    # Get route metrics
    route_data = pff_client.get_route_metrics(seasons)
    
    if route_data.empty:
        print("❌ No PFF route data available")
        return
    
    print(f"Retrieved {len(route_data)} route records")
    print(f"Columns: {route_data.columns.tolist()}")
    
    # Create player ID mapping
    player_mapping = create_player_id_mapping(engine)
    
    # Map player IDs (this will depend on PFF data structure)
    # route_data['player_id'] = route_data['pff_player_id'].map(player_mapping)
    # route_data = route_data.dropna(subset=['player_id'])
    
    # Process and clean route data
    # This will depend on the actual PFF data structure
    print("Processing route data...")
    
    # Example processing (adjust based on actual PFF data):
    # route_metrics = route_data.groupby(['game_id', 'player_id']).agg({
    #     'routes_run': 'sum',
    #     'route_participation': 'mean',
    #     'route_efficiency': 'mean',
    #     'route_depth': 'mean',
    #     'route_separation': 'mean'
    # }).reset_index()
    
    # Save to database
    # route_metrics.to_sql('player_route_stats', engine, if_exists='append', index=False)
    
    print("✅ PFF route data processing complete!")
    print("Note: This is a template - actual implementation depends on PFF API structure")
    
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build PFF route data for fantasy DB.")
    p.add_argument("--start", type=int, default=2018, help="Start season (YYYY)")
    p.add_argument("--end", type=int, default=2024, help="End season (YYYY)")
    p.add_argument("--db-url", type=str, default=_DEFAULT_URL, help="SQLAlchemy DB URL")
    p.add_argument("--pff-api-key", type=str, required=True, help="PFF API key")
    p.add_argument("--pff-base-url", type=str, default="https://api.profootballfocus.com/v1", 
                   help="PFF API base URL")
    
    a = p.parse_args()
    
    return Args(
        start=a.start,
        end=a.end,
        db_url=a.db_url,
        pff_api_key=a.pff_api_key,
        pff_base_url=a.pff_base_url
    )

if __name__ == "__main__":
    build_pff_route_data(parse_args())
