#!/usr/bin/env python3
"""
Build fantasy football databases for all scoring types
Creates separate databases: fantasy_ppr.db, fantasy_halfppr.db, fantasy_standard.db
"""

import subprocess
import sys
import os

def build_scoring_database(scoring_type, start_year=2018, end_year=2024):
    """Build database for a specific scoring type"""
    db_name = f"fantasy_{scoring_type}.db"
    
    print(f"\n{'='*60}")
    print(f"Building {scoring_type.upper()} database: {db_name}")
    print(f"{'='*60}")
    
    # Remove existing database if it exists
    db_path = f"data/{db_name}"
    if os.path.exists(db_path):
        print(f"Removing existing {db_path}...")
        os.remove(db_path)
    
    # Build the database
    cmd = [
        sys.executable, "scripts/build_fantasy_db_simple.py",
        "--start", str(start_year),
        "--end", str(end_year),
        "--scoring", scoring_type,
        "--db-url", f"sqlite:///data/{db_name}"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully built {db_name}")
        return True
    else:
        print(f"❌ Failed to build {db_name}")
        print("Error output:")
        print(result.stderr)
        return False

def main():
    scoring_types = ["ppr", "half_ppr", "standard"]
    start_year = 2018
    end_year = 2024
    
    print(f"Building fantasy football databases for {start_year}-{end_year}")
    print(f"Scoring types: {', '.join(scoring_types)}")
    
    results = {}
    for scoring_type in scoring_types:
        success = build_scoring_database(scoring_type, start_year, end_year)
        results[scoring_type] = success
    
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    
    for scoring_type, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        db_name = f"fantasy_{scoring_type}.db"
        print(f"{scoring_type.upper():12} {status} -> {db_name}")
    
    # Show database sizes
    print(f"\nDatabase sizes:")
    for scoring_type in scoring_types:
        db_name = f"fantasy_{scoring_type}.db"
        db_path = f"data/{db_name}"
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"{db_path}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
