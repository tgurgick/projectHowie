#!/usr/bin/env python3
"""
Build Historical FantasyPros ADP Data

Loads ADP data for all available years (2021-2024) and all scoring formats.
"""

import sys
import argparse
import subprocess
from dataclasses import dataclass

# Import the ADP scraper
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasypros_adp import build_fantasypros_adp, parse_args

@dataclass
class HistoricalADPConfig:
    years: list
    scoring_formats: list
    databases: list

def build_historical_adp(config: HistoricalADPConfig, test_mode: bool = False):
    """Build historical ADP data for all years and formats"""
    print("🏈 Building Historical FantasyPros ADP Data")
    print("=" * 60)
    
    total_records = 0
    
    for year in config.years:
        print(f"\n📅 Processing {year}...")
        
        for scoring in config.scoring_formats:
            db_name = f"fantasy_{scoring.replace('_', '')}.db"
            db_url = f"sqlite:///data/{db_name}"
            
            print(f"  📊 {scoring.upper()} scoring...")
            
            # Create args for the ADP scraper
            args = type('Args', (), {
                'season': year,
                'db_url': db_url,
                'scoring': scoring,
                'test': test_mode
            })()
            
            try:
                # Call the ADP scraper
                build_fantasypros_adp(args)
                
                # Count records if not in test mode
                if not test_mode:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    engine = create_engine(db_url, future=True)
                    with engine.connect() as conn:
                        count = pd.read_sql(
                            text(f"SELECT COUNT(*) as count FROM fantasy_market WHERE season = {year} AND adp_overall IS NOT NULL"),
                            conn
                        )
                        year_records = count.iloc[0, 0]
                        total_records += year_records
                        print(f"    ✅ Added {year_records} ADP records for {year} {scoring}")
                
            except Exception as e:
                print(f"    ❌ Error processing {year} {scoring}: {e}")
    
    print(f"\n🎉 Historical ADP build completed!")
    if not test_mode:
        print(f"📊 Total ADP records added: {total_records}")
    
    return total_records

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build historical FantasyPros ADP data")
    parser.add_argument("--test", action="store_true", help="Test mode (don't write to database)")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024], 
                       help="Years to process")
    parser.add_argument("--formats", nargs="+", default=["ppr", "half_ppr", "standard"],
                       help="Scoring formats to process")
    
    args = parser.parse_args()
    
    config = HistoricalADPConfig(
        years=args.years,
        scoring_formats=args.formats,
        databases=[]
    )
    
    print(f"📋 Configuration:")
    print(f"   Years: {config.years}")
    print(f"   Formats: {config.scoring_formats}")
    print(f"   Test mode: {args.test}")
    
    build_historical_adp(config, args.test)

if __name__ == "__main__":
    main()
