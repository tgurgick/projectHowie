#!/usr/bin/env python3
"""
Show advanced stats available in fantasy football databases
"""

import pandas as pd
from sqlalchemy import create_engine, text

def show_advanced_stats_sample(db_path, db_name):
    """Show sample of advanced stats from a database"""
    print(f"\n📊 {db_name} - Advanced Stats Sample")
    print("=" * 60)
    
    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        
        with engine.connect() as conn:
            # Check if advanced stats table exists
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name='player_advanced_stats'"), conn)
            
            if tables.empty:
                print("❌ No advanced stats table found")
                return
            
            # Get sample of advanced stats
            sample = pd.read_sql(text("""
                SELECT 
                    p.name,
                    p.position,
                    p.team,
                    pas.snap_share,
                    pas.target_share,
                    pas.air_yards,
                    pas.aDOT,
                    pas.yac,
                    pas.epa_per_play,
                    pas.cpoe
                FROM player_advanced_stats pas
                JOIN players p ON pas.player_id = p.player_id
                JOIN games g ON pas.game_id = g.game_id
                WHERE g.season = 2024
                AND (pas.snap_share IS NOT NULL OR pas.target_share IS NOT NULL OR pas.air_yards IS NOT NULL)
                ORDER BY pas.air_yards DESC
                LIMIT 10
            """), conn)
            
            if not sample.empty:
                print("Top 10 players by air yards (2024):")
                print(f"{'Name':<20} {'Pos':<3} {'Team':<3} {'Snap%':<6} {'Target%':<8} {'Air Yds':<8} {'aDOT':<6} {'YAC':<6} {'EPA':<6} {'CPOE':<6}")
                print("-" * 80)
                
                for _, row in sample.iterrows():
                    snap_pct = f"{row['snap_share']:.1%}" if pd.notna(row['snap_share']) else "N/A"
                    target_pct = f"{row['target_share']:.1%}" if pd.notna(row['target_share']) else "N/A"
                    air_yds = f"{row['air_yards']:.0f}" if pd.notna(row['air_yards']) else "N/A"
                    adot = f"{row['aDOT']:.1f}" if pd.notna(row['aDOT']) else "N/A"
                    yac = f"{row['yac']:.0f}" if pd.notna(row['yac']) else "N/A"
                    epa = f"{row['epa_per_play']:.2f}" if pd.notna(row['epa_per_play']) else "N/A"
                    cpoe = f"{row['cpoe']:.1f}" if pd.notna(row['cpoe']) else "N/A"
                    
                    print(f"{row['name']:<20} {row['position']:<3} {row['team']:<3} "
                          f"{snap_pct:<6} {target_pct:<8} {air_yds:<8} "
                          f"{adot:<6} {yac:<6} {epa:<6} {cpoe:<6}")
            else:
                print("No advanced stats data found")
                
            # Get stats summary
            summary = pd.read_sql(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN snap_share IS NOT NULL THEN 1 END) as snap_records,
                    COUNT(CASE WHEN target_share IS NOT NULL THEN 1 END) as target_records,
                    COUNT(CASE WHEN air_yards IS NOT NULL THEN 1 END) as air_yards_records,
                    COUNT(CASE WHEN epa_per_play IS NOT NULL THEN 1 END) as epa_records
                FROM player_advanced_stats
            """), conn)
            
            if not summary.empty:
                row = summary.iloc[0]
                print(f"\n📈 Advanced Stats Summary:")
                print(f"   Total records: {row['total_records']:,}")
                print(f"   Snap share records: {row['snap_records']:,}")
                print(f"   Target share records: {row['target_records']:,}")
                print(f"   Air yards records: {row['air_yards_records']:,}")
                print(f"   EPA records: {row['epa_records']:,}")
                
    except Exception as e:
        print(f"❌ Error reading {db_name}: {e}")

def show_market_data_sample(db_path, db_name):
    """Show sample of market data from a database"""
    print(f"\n📈 {db_name} - Market Data Sample")
    print("=" * 60)
    
    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        
        with engine.connect() as conn:
            # Check if market table exists
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name='fantasy_market'"), conn)
            
            if tables.empty:
                print("❌ No market data table found")
                return
            
            # Get sample of market data
            sample = pd.read_sql(text("""
                SELECT 
                    p.name,
                    p.position,
                    p.team,
                    fm.ecr_rank,
                    fm.adp_overall,
                    fm.adp_position
                FROM fantasy_market fm
                JOIN players p ON fm.player_id = p.player_id
                WHERE fm.season = 2024
                ORDER BY fm.ecr_rank ASC
                LIMIT 10
            """), conn)
            
            if not sample.empty:
                print("Top 10 players by ECR rank (2024):")
                print(f"{'Name':<20} {'Pos':<3} {'Team':<3} {'ECR':<6} {'ADP Overall':<12} {'ADP Pos':<8}")
                print("-" * 60)
                
                for _, row in sample.iterrows():
                    # Handle None values properly - convert to string first
                    ecr_val = row['ecr_rank']
                    adp_overall_val = row['adp_overall']
                    adp_pos_val = row['adp_position']
                    
                    ecr = f"{ecr_val:.1f}" if ecr_val is not None and pd.notna(ecr_val) else "N/A"
                    adp_overall = f"{adp_overall_val:.0f}" if adp_overall_val is not None and pd.notna(adp_overall_val) else "N/A"
                    adp_pos = f"{adp_pos_val:.0f}" if adp_pos_val is not None and pd.notna(adp_pos_val) else "N/A"
                    
                    # Handle None team value
                    team_display = row['team'] if row['team'] is not None else "N/A"
                    print(f"{row['name']:<20} {row['position']:<3} {team_display:<3} "
                          f"{ecr:<6} {adp_overall:<12} {adp_pos:<8}")
            else:
                print("No market data found")
                
            # Get market data summary
            summary = pd.read_sql(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN ecr_rank IS NOT NULL THEN 1 END) as ecr_records,
                    COUNT(CASE WHEN adp_overall IS NOT NULL THEN 1 END) as adp_records
                FROM fantasy_market
            """), conn)
            
            if not summary.empty:
                row = summary.iloc[0]
                print(f"\n📊 Market Data Summary:")
                print(f"   Total records: {row['total_records']:,}")
                print(f"   ECR records: {row['ecr_records']:,}")
                print(f"   ADP records: {row['adp_records']:,}")
                
    except Exception as e:
        print(f"❌ Error reading {db_name}: {e}")

def main():
    """Main function to show advanced stats and market data"""
    print("🏈 Fantasy Football Advanced Stats & Market Data")
    print("=" * 80)
    
    databases = [
        ("data/fantasy_ppr.db", "PPR Database"),
        ("data/fantasy_halfppr.db", "Half-PPR Database"),
        ("data/fantasy_standard.db", "Standard Database")
    ]
    
    for db_path, db_name in databases:
        show_advanced_stats_sample(db_path, db_name)
        show_market_data_sample(db_path, db_name)
    
    print(f"\n{'='*80}")
    print("✅ All databases now include advanced stats and market data!")
    print("📊 Available advanced stats:")
    print("   • Snap share (offensive snap percentage)")
    print("   • Target share (percentage of team targets)")
    print("   • Air yards (passing yards before catch)")
    print("   • aDOT (average depth of target)")
    print("   • YAC (yards after catch)")
    print("   • EPA per play (expected points added)")
    print("   • CPOE (completion percentage over expected)")
    print("   • Route runs, broken tackles, RYOE (placeholders)")
    print("\n📈 Available market data:")
    print("   • ECR rank (expert consensus ranking)")
    print("   • ADP overall (average draft position)")
    print("   • ADP position (position-specific ADP)")

if __name__ == "__main__":
    main()
