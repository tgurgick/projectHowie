#!/usr/bin/env python3
"""
Build fantasy market data (ADP/ECR) for fantasy football database
Handles player ID mapping and data validation
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
    FantasyMarket, Base, ensure_dt,
    DEFAULT_DB_URL
)

@dataclass
class Args:
    start: int
    end: int
    db_url: str

def create_ecr_data(engine, seasons):
    """Create ECR data with proper player mapping"""
    print("Creating ECR data...")
    
    try:
        # Import DynastyProcess ECR
        from build_fantasy_db import import_dynastyprocess_ecr
        ecr = import_dynastyprocess_ecr()
        
        if ecr.empty:
            print("  No ECR data available")
            return pd.DataFrame()
        
        # Get existing players from our database
        with engine.connect() as conn:
            our_players = pd.read_sql(text("SELECT player_id, name, position, team FROM players"), conn)
        
        # Import DynastyProcess IDs for mapping
        from build_fantasy_db import import_dynastyprocess_playerids
        dp_ids = import_dynastyprocess_playerids()
        
        # Map player IDs using player name
        ecr = ecr.merge(dp_ids[["player_id","name"]].drop_duplicates(), left_on="player", right_on="name", how="left")
        
        # Filter out records with NULL player_id
        ecr = ecr.dropna(subset=['player_id'])
        
        print(f"  ECR data available: {len(ecr)} records after mapping")
        
        # Create market data
        market = ecr[["player_id","ecr_rank"]].copy()
        market["season"] = 2024  # Default to current year
        market["week"] = 1  # Default to week 1
        market["adp_overall"] = np.nan
        market["adp_position"] = np.nan
        
        # Deduplicate by player_id, keeping the best (lowest) ECR rank
        market = market.sort_values('ecr_rank').drop_duplicates(subset=['player_id'], keep='first')
        
        print(f"  Created {len(market)} unique ECR records for seasons {seasons}")
        return market
        
    except Exception as e:
        print(f"  Error creating ECR data: {e}")
        return pd.DataFrame()

def create_fantasypros_adp_data(engine, seasons):
    """Create FantasyPros ADP data"""
    print("Creating FantasyPros ADP data...")
    
    # This would integrate with FantasyPros API or web scraping
    # For now, return empty DataFrame as placeholder
    print("  FantasyPros ADP integration not yet implemented")
    return pd.DataFrame()

def build_fantasy_market(args: Args):
    """Build fantasy market data with proper validation"""
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    seasons = list(range(args.start, args.end + 1))
    print(f"Building fantasy market data for seasons: {seasons}")

    # Create ECR data
    ecr_data = create_ecr_data(engine, seasons)
    
    # Create ADP data (placeholder for now)
    adp_data = create_fantasypros_adp_data(engine, seasons)
    
    # Combine market data
    if not ecr_data.empty:
        market_data = ecr_data.copy()
        
        # Check which seasons already exist
        with engine.connect() as conn:
            existing_seasons = pd.read_sql(text("SELECT DISTINCT season FROM fantasy_market"), conn)
        
        if not existing_seasons.empty:
            existing_seasons = set(existing_seasons['season'].tolist())
            # Filter to only new seasons
            new_market = market_data[~market_data['season'].isin(existing_seasons)]
            if not new_market.empty:
                print(f"  Loading {len(new_market)} new market records for seasons {list(new_market['season'].unique())}...")
                new_market.to_sql(FantasyMarket.__tablename__, engine, if_exists="append", index=False)
            else:
                print(f"  All seasons {seasons} already exist in market table, skipping...")
        else:
            print(f"  Loading {len(market_data)} market records for all seasons {seasons}...")
            market_data.to_sql(FantasyMarket.__tablename__, engine, if_exists="append", index=False)
    else:
        print("  No market data to load")
    
    print("✅ Fantasy market data loaded successfully!")
    engine.dispose()

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build fantasy market data for fantasy DB.")
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
    build_fantasy_market(parse_args())
