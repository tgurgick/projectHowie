#!/usr/bin/env python3
"""
PFF Projections and Strength of Schedule Data Import

Imports PFF projections and strength of schedule data into the database.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import the database models
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasy_db import (
    PlayerProjections, StrengthOfSchedule, Base, 
    DEFAULT_DB_URL
)
try:
    from howie_cli.core.paths import get_db_url
    _DEFAULT_URL = os.getenv("DB_URL", get_db_url("ppr"))
except Exception:
    _DEFAULT_URL = DEFAULT_DB_URL

@dataclass
class Args:
    season: int
    db_url: str
    data_dir: str

def clean_numeric_value(value):
    """Clean and convert numeric values"""
    if pd.isna(value) or value == '' or value == 'nan':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def import_projections(args: Args):
    """Import PFF projections data"""
    print(f"🏈 Importing PFF projections for {args.season}")
    
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Find projections file
    data_path = Path(args.data_dir)
    projection_files = [
        data_path / f"offensive_projections_{args.season}_preseason.csv",
        data_path / f"offensive_projections.csv"
    ]
    
    projection_file = None
    for file in projection_files:
        if file.exists():
            projection_file = file
            break
    
    if not projection_file:
        print(f"❌ No projection file found for {args.season}")
        return
    
    print(f"  📊 Reading: {projection_file}")
    df = pd.read_csv(projection_file)
    
    print(f"  📦 Processing {len(df)} player projections")
    
    # Clear existing data
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM player_projections 
            WHERE season = :season AND projection_type = 'preseason'
        """), {'season': args.season})
        conn.commit()
    
    # Process projections
    projection_data = []
    
    for _, row in df.iterrows():
        # Map column names to database fields
        data = {
            'season': args.season,
            'projection_type': 'preseason',
            'player_name': row.get('playerName', ''),
            'team_name': row.get('teamName', ''),
            'position': row.get('position', '').lower(),
            'bye_week': clean_numeric_value(row.get('byeWeek')),
            'games': clean_numeric_value(row.get('games')),
            'fantasy_points': clean_numeric_value(row.get('fantasyPoints')),
            'fantasy_points_rank': clean_numeric_value(row.get('fantasyPointsRank')),
            'auction_value': clean_numeric_value(row.get('auctionValue')),
            # Passing
            'pass_comp': clean_numeric_value(row.get('passComp')),
            'pass_att': clean_numeric_value(row.get('passAtt')),
            'pass_yds': clean_numeric_value(row.get('passYds')),
            'pass_td': clean_numeric_value(row.get('passTd')),
            'pass_int': clean_numeric_value(row.get('passInt')),
            'pass_sacked': clean_numeric_value(row.get('passSacked')),
            # Rushing
            'rush_att': clean_numeric_value(row.get('rushAtt')),
            'rush_yds': clean_numeric_value(row.get('rushYds')),
            'rush_td': clean_numeric_value(row.get('rushTd')),
            # Receiving
            'recv_targets': clean_numeric_value(row.get('recvTargets')),
            'recv_receptions': clean_numeric_value(row.get('recvReceptions')),
            'recv_yds': clean_numeric_value(row.get('recvYds')),
            'recv_td': clean_numeric_value(row.get('recvTd')),
            # Other
            'fumbles': clean_numeric_value(row.get('fumbles')),
            'fumbles_lost': clean_numeric_value(row.get('fumblesLost')),
            'two_pt': clean_numeric_value(row.get('twoPt')),
            # Kicking
            'fg_made_0_19': clean_numeric_value(row.get('fgMade019')),
            'fg_att_0_19': clean_numeric_value(row.get('fgAtt019')),
            'fg_made_20_29': clean_numeric_value(row.get('fgMade2029')),
            'fg_att_20_29': clean_numeric_value(row.get('fgAtt2029')),
            'fg_made_30_39': clean_numeric_value(row.get('fgMade3039')),
            'fg_att_30_39': clean_numeric_value(row.get('fgAtt3039')),
            'fg_made_40_49': clean_numeric_value(row.get('fgMade4049')),
            'fg_att_40_49': clean_numeric_value(row.get('fgAtt4049')),
            'fg_made_50_plus': clean_numeric_value(row.get('fgMade50plus')),
            'fg_att_50_plus': clean_numeric_value(row.get('fgAtt50plus')),
            'pat_made': clean_numeric_value(row.get('patMade')),
            'pat_att': clean_numeric_value(row.get('patAtt')),
            # Defense
            'dst_sacks': clean_numeric_value(row.get('dstSacks')),
            'dst_safeties': clean_numeric_value(row.get('dstSafeties')),
            'dst_int': clean_numeric_value(row.get('dstInt')),
            'dst_fumbles_forced': clean_numeric_value(row.get('dstFumblesForced')),
            'dst_fumbles_recovered': clean_numeric_value(row.get('dstFumblesRecovered')),
            'dst_td': clean_numeric_value(row.get('dstTd')),
            'dst_return_yds': clean_numeric_value(row.get('dstReturnYds')),
            'dst_return_td': clean_numeric_value(row.get('dstReturnTd')),
            'dst_pts_0': clean_numeric_value(row.get('dstPts0')),
            'dst_pts_1_6': clean_numeric_value(row.get('dstPts16')),
            'dst_pts_7_13': clean_numeric_value(row.get('dstPts713')),
            'dst_pts_14_20': clean_numeric_value(row.get('dstPts1420')),
            'dst_pts_21_27': clean_numeric_value(row.get('dstPts2127')),
            'dst_pts_28_34': clean_numeric_value(row.get('dstPts2834')),
            'dst_pts_35_plus': clean_numeric_value(row.get('dstPts35plus'))
        }
        projection_data.append(data)
    
    # Insert into database
    df_projections = pd.DataFrame(projection_data)
    df_projections.to_sql(PlayerProjections.__tablename__, engine, if_exists="append", index=False)
    
    print(f"  ✅ Inserted {len(projection_data)} projections")
    
    # Show sample data by position
    print("\n📊 Sample Projections by Position:")
    for pos in ['qb', 'rb', 'wr', 'te', 'k', 'dst']:
        pos_data = df_projections[df_projections['position'] == pos].head(3)
        if not pos_data.empty:
            print(f"  {pos.upper()}:")
            for _, row in pos_data.iterrows():
                fp = row['fantasy_points'] if row['fantasy_points'] else 0
                print(f"    {row['player_name']} ({row['team_name']}) - {fp:.1f} pts")

def import_strength_of_schedule(args: Args):
    """Import strength of schedule data"""
    print(f"\n🗓️ Importing Strength of Schedule data for {args.season}")
    
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Clear existing data
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM strength_of_schedule WHERE season = :season
        """), {'season': args.season})
        conn.commit()
    
    data_path = Path(args.data_dir)
    positions = ['qb', 'rb', 'wr', 'te', 'dst']
    
    sos_data = []
    
    for position in positions:
        sos_file = data_path / f"{position}-fantasy-sos_{args.season}_preseason.csv"
        if not sos_file.exists():
            print(f"  ⚠️ SoS file not found: {sos_file}")
            continue
        
        print(f"  📊 Reading {position.upper()} SoS: {sos_file}")
        df = pd.read_csv(sos_file)
        
        for _, row in df.iterrows():
            # DST files use 'Defense' column, others use 'Offense'
            team = row.get('Defense', '') if position == 'dst' else row.get('Offense', '')
            if not team:
                continue
            
            data = {
                'season': args.season,
                'position': position,
                'team': team,
                'week_1': clean_numeric_value(row.get('1')),
                'week_2': clean_numeric_value(row.get('2')),
                'week_3': clean_numeric_value(row.get('3')),
                'week_4': clean_numeric_value(row.get('4')),
                'week_5': clean_numeric_value(row.get('5')),
                'week_6': clean_numeric_value(row.get('6')),
                'week_7': clean_numeric_value(row.get('7')),
                'week_8': clean_numeric_value(row.get('8')),
                'week_9': clean_numeric_value(row.get('9')),
                'week_10': clean_numeric_value(row.get('10')),
                'week_11': clean_numeric_value(row.get('11')),
                'week_12': clean_numeric_value(row.get('12')),
                'week_13': clean_numeric_value(row.get('13')),
                'week_14': clean_numeric_value(row.get('14')),
                'week_15': clean_numeric_value(row.get('15')),
                'week_16': clean_numeric_value(row.get('16')),
                'week_17': clean_numeric_value(row.get('17')),
                'season_games': clean_numeric_value(row.get('Season #G')),
                'season_sos': clean_numeric_value(row.get('Season SOS')),
                'playoffs_games': clean_numeric_value(row.get('Playoffs #G')),
                'playoffs_sos': clean_numeric_value(row.get('Playoffs SOS')),
                'all_games': clean_numeric_value(row.get('All #G')),
                'all_sos': clean_numeric_value(row.get('All SOS'))
            }
            sos_data.append(data)
    
    if sos_data:
        df_sos = pd.DataFrame(sos_data)
        df_sos.to_sql(StrengthOfSchedule.__tablename__, engine, if_exists="append", index=False)
        print(f"  ✅ Inserted {len(sos_data)} SoS records")
        
        # Show sample data
        print("\n📊 Sample Strength of Schedule:")
        for pos in positions:
            pos_data = df_sos[df_sos['position'] == pos].head(3)
            if not pos_data.empty:
                print(f"  {pos.upper()}:")
                for _, row in pos_data.iterrows():
                    sos = row['season_sos'] if row['season_sos'] else 0
                    print(f"    {row['team']}: Season SoS = {sos:.1f}")
    else:
        print("  ❌ No SoS data processed")

def main():
    parser = argparse.ArgumentParser(description="Import PFF projections and SoS data")
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument("--db-url", default=_DEFAULT_URL, help="Database URL")
    parser.add_argument("--data-dir", default="data/pff_csv", help="Data directory")
    
    args = parser.parse_args()
    
    print(f"🏈 PFF Data Import for {args.season}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🗄️ Database: {args.db_url}")
    
    # Import projections
    import_projections(args)
    
    # Import strength of schedule
    import_strength_of_schedule(args)
    
    print(f"\n✅ PFF data import completed for {args.season}")

if __name__ == "__main__":
    main()
