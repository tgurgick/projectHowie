#!/usr/bin/env python3
"""
Import historical defensive statistics (2018-2024) from nfl_data_py.
Creates both individual player defensive stats and aggregated team defense totals.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, PrimaryKeyConstraint, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Database setup
try:
    from howie_cli.core.paths import get_db_url
    DEFAULT_DB_URL = os.getenv("DB_URL", get_db_url("ppr"))
except Exception:
    # Fallback to previous behavior if package import not available
    DEFAULT_DB_URL = os.getenv("DB_URL", "sqlite:///data/fantasy_ppr.db")
Base = declarative_base(metadata=MetaData(schema=None))

@dataclass
class Args:
    """Command line arguments"""
    start_year: int = 2018
    end_year: int = 2024
    db_url: str = DEFAULT_DB_URL
    test: bool = False

# Database Models
class PlayerDefensiveStats(Base):
    __tablename__ = "player_defensive_stats"
    
    player_name = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    team = Column(String)
    position = Column(String)
    pfr_id = Column(String)
    age = Column(Integer)
    games = Column(Integer)
    games_started = Column(Integer)
    
    # Core defensive stats
    sacks = Column(Float, nullable=True)
    interceptions = Column(Float, nullable=True)
    tackles = Column(Float, nullable=True)
    missed_tackles = Column(Float, nullable=True)
    missed_tackle_pct = Column(Float, nullable=True)
    
    # Pass coverage stats
    targets = Column(Float, nullable=True)
    completions_allowed = Column(Float, nullable=True)
    completion_pct_allowed = Column(Float, nullable=True)
    yards_allowed = Column(Float, nullable=True)
    yards_per_completion = Column(Float, nullable=True)
    yards_per_target = Column(Float, nullable=True)
    tds_allowed = Column(Float, nullable=True)
    passer_rating_allowed = Column(Float, nullable=True)
    
    # Pass rush stats  
    blitzes = Column(Float, nullable=True)
    hurries = Column(Float, nullable=True)
    qb_hits = Column(Float, nullable=True)
    pressures = Column(Float, nullable=True)
    passes_defended = Column(Float, nullable=True)
    
    # Air yards and YAC
    avg_depth_of_target = Column(Float, nullable=True)
    air_yards = Column(Float, nullable=True)
    yards_after_catch = Column(Float, nullable=True)
    


class TeamDefensiveStats(Base):
    __tablename__ = "team_defensive_stats"
    
    team = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    
    # Aggregated team totals for fantasy purposes
    total_sacks = Column(Float, nullable=True)
    total_interceptions = Column(Float, nullable=True)
    total_tackles = Column(Float, nullable=True)
    total_hurries = Column(Float, nullable=True)
    total_qb_hits = Column(Float, nullable=True)
    total_pressures = Column(Float, nullable=True)
    total_passes_defended = Column(Float, nullable=True)
    total_tds_allowed = Column(Float, nullable=True)
    
    # Averages and rates
    avg_passer_rating_allowed = Column(Float, nullable=True)
    completion_pct_allowed = Column(Float, nullable=True)
    missed_tackle_pct = Column(Float, nullable=True)

def normalize_team_name(team: str) -> str:
    """Normalize team abbreviations to match our database standards"""
    team_mapping = {
        'JAX': 'JAC',  # Jacksonville Jaguars
        'LVR': 'LV',   # Las Vegas Raiders (if using LV)
        'OAK': 'LV',   # Oakland Raiders -> Las Vegas
        'LAR': 'LA',   # Los Angeles Rams (if using LA)
        # Add other mappings as needed
    }
    return team_mapping.get(team, team)

def normalize_position(pos: str) -> str:
    """Normalize position names to standard format"""
    pos_mapping = {
        'LILB': 'LB',
        'RILB': 'LB', 
        'MLB': 'LB',
        'OLB': 'LB',
        'LOLB': 'LB',
        'ROLB': 'LB',
        'LLB': 'LB',
        'RLB': 'LB',
        'S-SS': 'S',
        'FS-S-SS': 'S',
        'SS/FS': 'S',
        'DB-S': 'S',
        'FS': 'S',
        'SS': 'S',
        'LCB-RCB': 'CB',
        'CB-RCB': 'CB',
        'CB-DB': 'CB',
        'LCB': 'CB',
        'RCB': 'CB',
        'LDE-RDE': 'DE',
        'LDE': 'DE',
        'RDE': 'DE',
        'DE-RDE': 'DE',
        'DE-LB': 'DE',
        'DE-DT': 'DE',
        'NT-RDE': 'DT',
        'RDT': 'DT',
        'LDT': 'DT',
        'NT': 'DT',
        'DT-LDE/RDE': 'DT',
        'DT-LDT': 'DT',
        'DB-NT': 'DT',
        'DL': 'DT',  # Generic defensive line -> DT
    }
    return pos_mapping.get(pos, pos)

def import_defensive_stats(args: Args):
    """Import defensive statistics from nfl_data_py"""
    
    print(f"Importing defensive stats from {args.start_year} to {args.end_year}...")
    
    # Import nfl_data_py
    try:
        import nfl_data_py as nfl
    except ImportError:
        print("Error: nfl_data_py not installed. Run: pip install nfl_data_py")
        return
    
    # Setup database
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get years to import
    years = list(range(args.start_year, args.end_year + 1))
    print(f"Fetching data for years: {years}")
    
    # Import seasonal defensive data
    try:
        df = nfl.import_seasonal_pfr(years=years, s_type='def')
        print(f"Retrieved {len(df)} defensive records")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Clean and prepare data
    print("Cleaning and preparing data...")
    
    # Normalize team and position names
    df['team_norm'] = df['tm'].apply(normalize_team_name)
    df['position_norm'] = df['pos'].apply(normalize_position)
    
    # Map columns to our schema
    column_mapping = {
        'player': 'player_name',
        'season': 'season', 
        'team_norm': 'team',
        'position_norm': 'position',
        'pfr_id': 'pfr_id',
        'age': 'age',
        'g': 'games',
        'gs': 'games_started',
        'sk': 'sacks',
        'int': 'interceptions', 
        'comb': 'tackles',
        'm_tkl': 'missed_tackles',
        'm_tkl_percent': 'missed_tackle_pct',
        'tgt': 'targets',
        'cmp': 'completions_allowed',
        'cmp_percent': 'completion_pct_allowed',
        'yds': 'yards_allowed',
        'yds_cmp': 'yards_per_completion',
        'yds_tgt': 'yards_per_target',
        'td': 'tds_allowed',
        'rat': 'passer_rating_allowed',
        'bltz': 'blitzes',
        'hrry': 'hurries',
        'qbkd': 'qb_hits',
        'prss': 'pressures',
        'bats': 'passes_defended',
        'dadot': 'avg_depth_of_target',
        'air': 'air_yards',
        'yac': 'yards_after_catch'
    }
    
    # Rename columns
    df_clean = df.rename(columns=column_mapping)
    
    # Filter to only include relevant defensive positions and real teams
    defensive_positions = ['LB', 'CB', 'S', 'DE', 'DT', 'OLB']
    # Filter out multi-team designations (2TM, 3TM, etc.)
    real_teams = df_clean[~df_clean['team'].str.contains('TM', na=False)]
    df_defense = real_teams[real_teams['position'].isin(defensive_positions)].copy()
    
    # Remove duplicates - keep first occurrence of player_name + season combination
    df_defense = df_defense.drop_duplicates(subset=['player_name', 'season'], keep='first')
    
    print(f"Filtered to {len(df_defense)} defensive player records")
    
    # Insert individual player stats
    if not args.test:
        print("Inserting individual player defensive stats...")
        
        # Drop existing data for these years
        session.query(PlayerDefensiveStats).filter(
            PlayerDefensiveStats.season.in_(years)
        ).delete()
        
        # Prepare records for insertion
        records_to_insert = []
        for _, row in df_defense.iterrows():
            record_dict = {}
            for col in PlayerDefensiveStats.__table__.columns.keys():
                if col in df_defense.columns:
                    value = row[col]
                    # Convert NaN to None
                    if pd.isna(value):
                        value = None
                    record_dict[col] = value
            records_to_insert.append(record_dict)
        
        # Bulk insert
        session.bulk_insert_mappings(PlayerDefensiveStats, records_to_insert)
        print(f"Inserted {len(records_to_insert)} player defensive stat records")
    
    # Create team aggregations
    print("Creating team defensive aggregations...")
    
    # Group by team and season to create team totals
    team_stats = df_defense.groupby(['team', 'season']).agg({
        'sacks': 'sum',
        'interceptions': 'sum', 
        'tackles': 'sum',
        'hurries': 'sum',
        'qb_hits': 'sum',
        'pressures': 'sum',
        'passes_defended': 'sum',
        'tds_allowed': 'sum',
        'passer_rating_allowed': 'mean',
        'completion_pct_allowed': 'mean',
        'missed_tackle_pct': 'mean'
    }).reset_index()
    
    # Rename columns for team stats
    team_stats = team_stats.rename(columns={
        'sacks': 'total_sacks',
        'interceptions': 'total_interceptions',
        'tackles': 'total_tackles', 
        'hurries': 'total_hurries',
        'qb_hits': 'total_qb_hits',
        'pressures': 'total_pressures',
        'passes_defended': 'total_passes_defended',
        'tds_allowed': 'total_tds_allowed',
        'passer_rating_allowed': 'avg_passer_rating_allowed'
    })
    
    print(f"Created {len(team_stats)} team defensive stat records")
    
    # Insert team stats
    if not args.test:
        print("Inserting team defensive stats...")
        
        # Drop existing data for these years
        session.query(TeamDefensiveStats).filter(
            TeamDefensiveStats.season.in_(years)
        ).delete()
        
        # Prepare team records for insertion
        team_records = []
        for _, row in team_stats.iterrows():
            record_dict = {}
            for col in TeamDefensiveStats.__table__.columns.keys():
                if col in team_stats.columns:
                    value = row[col]
                    if pd.isna(value):
                        value = None
                    record_dict[col] = value
            team_records.append(record_dict)
        
        # Bulk insert team stats
        session.bulk_insert_mappings(TeamDefensiveStats, team_records)
        print(f"Inserted {len(team_records)} team defensive stat records")
        
        # Commit changes
        session.commit()
        print("✅ All defensive stats imported successfully!")
    else:
        print("TEST MODE: No data inserted")
    
    # Show sample of what was imported
    print("\nSample individual player stats:")
    sample_players = df_defense[['player_name', 'team', 'position', 'season', 'sacks', 'interceptions', 'tackles']].head(10)
    print(sample_players.to_string(index=False))
    
    print("\nSample team stats:")
    sample_teams = team_stats[['team', 'season', 'total_sacks', 'total_interceptions']].head(10)
    print(sample_teams.to_string(index=False))
    
    session.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Import historical defensive statistics")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="Database URL")
    parser.add_argument("--test", action="store_true", help="Test mode - don't insert data")
    
    args = Args(**vars(parser.parse_args()))
    
    print(f"🏈 Importing Defensive Stats (2018-2024)")
    print(f"Database: {args.db_url}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Test mode: {args.test}")
    print("-" * 50)
    
    import_defensive_stats(args)

if __name__ == "__main__":
    main()
