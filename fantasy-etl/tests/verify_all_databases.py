#!/usr/bin/env python3
"""
Comprehensive verification script for all fantasy football databases
Combines testing, summary, and recommendations in one place
"""

import os
import subprocess
import sys
from datetime import datetime

def run_test_script():
    """Run the comprehensive test script"""
    print("🔍 Running comprehensive database tests...")
    result = subprocess.run([sys.executable, "tests/test_all_databases.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All database tests completed successfully")
        return True
    else:
        print("❌ Some database tests failed")
        print(result.stderr)
        return False

def run_summary_script():
    """Run the database summary script"""
    print("\n📊 Generating database summary...")
    result = subprocess.run([sys.executable, "tests/database_summary.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("❌ Summary generation failed")
        print(result.stderr)
        return False

def check_database_files():
    """Check which database files exist"""
    print("\n📁 Checking database files...")
    
    expected_files = [
        "data/fantasy_ppr.db", 
        "data/fantasy_halfppr.db",
        "data/fantasy_standard.db"
    ]
    
    existing_files = []
    missing_files = []
    
    for file in expected_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            existing_files.append((file, size_mb))
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            missing_files.append(file)
            print(f"❌ {file} (missing)")
    
    return existing_files, missing_files

def get_quick_stats():
    """Get quick statistics from each database"""
    print("\n📈 Quick Statistics...")
    
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    databases = [
        ("data/fantasy_ppr.db", "PPR"),
        ("data/fantasy_halfppr.db", "Half-PPR"),
        ("data/fantasy_standard.db", "Standard")
    ]
    
    stats = []
    for db_path, db_name in databases:
        if os.path.exists(db_path):
            try:
                engine = create_engine(f"sqlite:///{db_path}", future=True)
                with engine.connect() as conn:
                    # Get basic counts
                    players = conn.execute(text("SELECT COUNT(*) FROM players")).scalar()
                    games = conn.execute(text("SELECT COUNT(*) FROM games")).scalar()
                    
                    # Check if it's multi-scoring
                    tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_game_stats_%'"), conn)
                    if not tables.empty:
                        # Multi-scoring database
                        total_stats = 0
                        for table in tables['name']:
                            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                            total_stats += count
                        scoring_types = [table.replace('player_game_stats_', '') for table in tables['name']]
                        stats.append({
                            'name': db_name,
                            'players': players,
                            'games': games,
                            'stats': total_stats,
                            'scoring_types': scoring_types
                        })
                    else:
                        # Single scoring database
                        stats_count = conn.execute(text("SELECT COUNT(*) FROM player_game_stats")).scalar()
                        stats.append({
                            'name': db_name,
                            'players': players,
                            'games': games,
                            'stats': stats_count,
                            'scoring_types': ['single']
                        })
                        
            except Exception as e:
                print(f"❌ Error reading {db_name}: {e}")
    
    return stats

def generate_recommendations(stats):
    """Generate recommendations based on the data"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if not stats:
        print("❌ No databases found to analyze")
        return
    
    # Find most complete database
    most_complete = max(stats, key=lambda x: x['games'])
    print(f"🎯 Most complete database: {most_complete['name']} ({most_complete['games']:,} games)")
    
    # Find database with most player stats
    most_stats = max(stats, key=lambda x: x['stats'])
    print(f"📊 Most player stats: {most_stats['name']} ({most_stats['stats']:,} records)")
    
    # Check for multi-scoring database
    multi_db = next((db for db in stats if len(db['scoring_types']) > 1), None)
    if multi_db:
        print(f"🔄 Multi-scoring database: {multi_db['name']} ({', '.join(multi_db['scoring_types'])})")
    
    # Check for different scoring types
    scoring_dbs = [db for db in stats if 'single' in db['scoring_types']]
    if len(scoring_dbs) >= 3:
        print(f"📈 Multiple scoring databases available: {len(scoring_dbs)} databases")
    
    print(f"\n✅ All databases are ready for fantasy football analysis!")

def main():
    """Main verification function"""
    print("🏈 Fantasy Football Database Verification Suite")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check database files
    existing_files, missing_files = check_database_files()
    
    if not existing_files:
        print("\n❌ No database files found!")
        print("Please run the build scripts first:")
        print("  python3 build_all_scoring.py")
        print("  python3 build_fantasy_db_multi_scoring.py")
        return
    
    # Get quick stats
    stats = get_quick_stats()
    
    # Run comprehensive tests
    tests_passed = run_test_script()
    
    # Run summary
    summary_generated = run_summary_script()
    
    # Generate recommendations
    generate_recommendations(stats)
    
    # Final status
    print(f"\n{'='*80}")
    print("FINAL STATUS")
    print(f"{'='*80}")
    
    print(f"📁 Database files: {len(existing_files)}/{len(existing_files) + len(missing_files)} found")
    print(f"🔍 Tests: {'✅ PASSED' if tests_passed else '❌ FAILED'}")
    print(f"📊 Summary: {'✅ GENERATED' if summary_generated else '❌ FAILED'}")
    
    if tests_passed and summary_generated:
        print(f"\n🎉 All verification completed successfully!")
        print(f"Your fantasy football databases are ready for analysis!")
    else:
        print(f"\n⚠️  Some verification steps failed. Check the output above for details.")
    
    # Show file sizes
    print(f"\n📦 Database Sizes:")
    total_size = 0
    for file, size in existing_files:
        print(f"   {file}: {size:.1f} MB")
        total_size += size
    print(f"   Total: {total_size:.1f} MB")

if __name__ == "__main__":
    main()
