#!/usr/bin/env python3
"""
Master script to build all missing stats for fantasy football databases
Builds advanced stats and fantasy market data for all scoring types
"""

import subprocess
import sys
import os

def build_stats_for_database(db_path, db_name, start_year=2018, end_year=2024):
    """Build advanced stats and market data for a specific database"""
    print(f"\n{'='*60}")
    print(f"Building stats for: {db_name}")
    print(f"Database: {db_path}")
    print(f"{'='*60}")
    
    # Build advanced stats
    print("\n📊 Building Advanced Stats...")
    cmd_advanced = [
        sys.executable, "scripts/build_advanced_stats.py",
        "--start", str(start_year),
        "--end", str(end_year),
        "--db-url", f"sqlite:///{db_path}"
    ]
    
    print(f"Running: {' '.join(cmd_advanced)}")
    result_advanced = subprocess.run(cmd_advanced, capture_output=True, text=True)
    
    if result_advanced.returncode == 0:
        print("✅ Advanced stats built successfully")
    else:
        print("❌ Advanced stats failed")
        print("Error output:")
        print(result_advanced.stderr)
    
    # Build fantasy market data
    print("\n📈 Building Fantasy Market Data...")
    cmd_market = [
        sys.executable, "scripts/build_fantasy_market.py",
        "--start", str(start_year),
        "--end", str(end_year),
        "--db-url", f"sqlite:///{db_path}"
    ]
    
    print(f"Running: {' '.join(cmd_market)}")
    result_market = subprocess.run(cmd_market, capture_output=True, text=True)
    
    if result_market.returncode == 0:
        print("✅ Fantasy market data built successfully")
    else:
        print("❌ Fantasy market data failed")
        print("Error output:")
        print(result_market.stderr)
    
    return result_advanced.returncode == 0 and result_market.returncode == 0

def main():
    """Main function to build stats for all databases"""
    start_year = 2018
    end_year = 2024
    
    print("🏈 Fantasy Football Stats Builder")
    print(f"Building advanced stats and market data for {start_year}-{end_year}")
    
    # Define databases to process
    databases = [
        ("data/fantasy_ppr.db", "PPR Database"),
        ("data/fantasy_halfppr.db", "Half-PPR Database"),
        ("data/fantasy_standard.db", "Standard Database")
    ]
    
    results = {}
    for db_path, db_name in databases:
        if os.path.exists(db_path):
            success = build_stats_for_database(db_path, db_name, start_year, end_year)
            results[db_name] = success
        else:
            print(f"\n❌ Database not found: {db_path}")
            results[db_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    
    for db_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{db_name:<20} {status}")
    
    # Show updated database sizes
    print(f"\nUpdated database sizes:")
    for db_path, db_name in databases:
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"{db_name}: {size_mb:.1f} MB")
    
    # Check what tables were added
    print(f"\nVerifying new tables...")
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    for db_path, db_name in databases:
        if os.path.exists(db_path):
            try:
                engine = create_engine(f"sqlite:///{db_path}", future=True)
                with engine.connect() as conn:
                    tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table'"), conn)
                    
                    # Check for advanced stats table
                    if 'player_advanced_stats' in tables['name'].values:
                        result = conn.execute(text("SELECT COUNT(*) FROM player_advanced_stats"))
                        adv_count = result.scalar()
                        print(f"  {db_name}: {adv_count:,} advanced stats records")
                    else:
                        print(f"  {db_name}: No advanced stats table")
                    
                    # Check for fantasy market table
                    if 'fantasy_market' in tables['name'].values:
                        result = conn.execute(text("SELECT COUNT(*) FROM fantasy_market"))
                        market_count = result.scalar()
                        print(f"  {db_name}: {market_count:,} market records")
                    else:
                        print(f"  {db_name}: No market table")
                        
            except Exception as e:
                print(f"  {db_name}: Error checking tables - {e}")

if __name__ == "__main__":
    main()
