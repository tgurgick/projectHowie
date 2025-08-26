#!/usr/bin/env python3
"""
Comprehensive test script to verify all fantasy football databases are properly loaded
Tests both separate databases and multi-scoring database approaches
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

def test_database_connectivity(db_path, db_name):
    """Test if database exists and can be connected to"""
    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        print(f"✅ {db_name}: Database connectivity OK")
        return engine
    except Exception as e:
        print(f"❌ {db_name}: Database connectivity failed - {e}")
        return None

def test_table_structure(engine, db_name, expected_tables):
    """Test if all expected tables exist"""
    try:
        with engine.connect() as conn:
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table'"), conn)
            existing_tables = set(tables['name'].tolist())
            
            missing_tables = expected_tables - existing_tables
            extra_tables = existing_tables - expected_tables
            
            if missing_tables:
                print(f"❌ {db_name}: Missing tables: {missing_tables}")
                return False
            if extra_tables:
                print(f"⚠️  {db_name}: Extra tables found: {extra_tables}")
            
            print(f"✅ {db_name}: All expected tables present")
            return True
    except Exception as e:
        print(f"❌ {db_name}: Table structure test failed - {e}")
        return False

def test_data_counts(engine, db_name, table_name, min_count=0):
    """Test if table has expected minimum number of records"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            
            if count >= min_count:
                print(f"✅ {db_name}.{table_name}: {count:,} records")
                return count
            else:
                print(f"❌ {db_name}.{table_name}: Only {count:,} records (expected >= {min_count:,})")
                return count
    except Exception as e:
        print(f"❌ {db_name}.{table_name}: Count test failed - {e}")
        return 0

def test_season_coverage(engine, db_name, table_name, expected_seasons):
    """Test if table has data for expected seasons"""
    try:
        with engine.connect() as conn:
            if table_name == 'players':
                # Players table doesn't have seasons, skip this test
                return True
                
            if table_name == 'games':
                seasons = pd.read_sql(text(f"SELECT DISTINCT season FROM {table_name} ORDER BY season"), conn)
            else:
                # For player_game_stats tables, join with games to get seasons
                seasons = pd.read_sql(text(f"""
                    SELECT DISTINCT g.season 
                    FROM {table_name} pgs 
                    JOIN games g ON pgs.game_id = g.game_id 
                    ORDER BY g.season
                """), conn)
            
            existing_seasons = set(seasons['season'].tolist())
            missing_seasons = expected_seasons - existing_seasons
            extra_seasons = existing_seasons - expected_seasons
            
            if missing_seasons:
                print(f"❌ {db_name}.{table_name}: Missing seasons: {sorted(missing_seasons)}")
                return False
            if extra_seasons:
                print(f"⚠️  {db_name}.{table_name}: Extra seasons: {sorted(extra_seasons)}")
            
            print(f"✅ {db_name}.{table_name}: Seasons {sorted(existing_seasons)}")
            return True
    except Exception as e:
        print(f"❌ {db_name}.{table_name}: Season coverage test failed - {e}")
        return False

def test_data_quality(engine, db_name, table_name):
    """Test data quality (no nulls in key fields, reasonable value ranges)"""
    try:
        with engine.connect() as conn:
            if table_name == 'players':
                # Test players table
                null_check = pd.read_sql(text(f"""
                    SELECT COUNT(*) as null_count 
                    FROM {table_name} 
                    WHERE player_id IS NULL OR name IS NULL OR position IS NULL
                """), conn)
                
                if null_check.iloc[0]['null_count'] == 0:
                    print(f"✅ {db_name}.{table_name}: No nulls in key fields")
                else:
                    print(f"❌ {db_name}.{table_name}: {null_check.iloc[0]['null_count']} nulls in key fields")
                    return False
                    
            elif table_name == 'games':
                # Test games table
                null_check = pd.read_sql(text(f"""
                    SELECT COUNT(*) as null_count 
                    FROM {table_name} 
                    WHERE game_id IS NULL OR season IS NULL OR week IS NULL
                """), conn)
                
                if null_check.iloc[0]['null_count'] == 0:
                    print(f"✅ {db_name}.{table_name}: No nulls in key fields")
                else:
                    print(f"❌ {db_name}.{table_name}: {null_check.iloc[0]['null_count']} nulls in key fields")
                    return False
                    
            elif 'player_game_stats' in table_name:
                # Test player_game_stats table
                null_check = pd.read_sql(text(f"""
                    SELECT COUNT(*) as null_count 
                    FROM {table_name} 
                    WHERE game_id IS NULL OR player_id IS NULL OR fantasy_points IS NULL
                """), conn)
                
                if null_check.iloc[0]['null_count'] == 0:
                    print(f"✅ {db_name}.{table_name}: No nulls in key fields")
                else:
                    print(f"❌ {db_name}.{table_name}: {null_check.iloc[0]['null_count']} nulls in key fields")
                    return False
                
                # Test fantasy points are reasonable
                points_check = pd.read_sql(text(f"""
                    SELECT 
                        MIN(fantasy_points) as min_points,
                        MAX(fantasy_points) as max_points,
                        AVG(fantasy_points) as avg_points
                    FROM {table_name}
                """), conn)
                
                row = points_check.iloc[0]
                if row['min_points'] >= -50 and row['max_points'] <= 100 and row['avg_points'] >= 0:
                    print(f"✅ {db_name}.{table_name}: Fantasy points range reasonable ({row['min_points']:.1f} to {row['max_points']:.1f}, avg {row['avg_points']:.1f})")
                else:
                    print(f"⚠️  {db_name}.{table_name}: Fantasy points range suspicious ({row['min_points']:.1f} to {row['max_points']:.1f}, avg {row['avg_points']:.1f})")
                    return False
            
            return True
    except Exception as e:
        print(f"❌ {db_name}.{table_name}: Data quality test failed - {e}")
        return False

def test_scoring_consistency(engine, db_name):
    """Test that different scoring types produce expected differences"""
    try:
        with engine.connect() as conn:
            # Check if this is a multi-scoring database
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_game_stats_%'"), conn)
            scoring_tables = [table.replace('player_game_stats_', '') for table in tables['name'].tolist()]
            
            if len(scoring_tables) >= 2:
                # Compare scoring types
                for i in range(len(scoring_tables) - 1):
                    current = scoring_tables[i]
                    next_type = scoring_tables[i + 1]
                    
                    comparison = pd.read_sql(text(f"""
                        SELECT 
                            AVG(p1.fantasy_points - p2.fantasy_points) as avg_diff,
                            COUNT(*) as sample_size
                        FROM player_game_stats_{current} p1
                        JOIN player_game_stats_{next_type} p2 ON p1.game_id = p2.game_id AND p1.player_id = p2.player_id
                        WHERE p1.fantasy_points != p2.fantasy_points
                    """), conn)
                    
                    if not comparison.empty and comparison.iloc[0]['sample_size'] > 0:
                        avg_diff = comparison.iloc[0]['avg_diff']
                        sample_size = comparison.iloc[0]['sample_size']
                        
                        # Check if difference is reasonable (should be positive for PPR vs others)
                        if current == 'ppr' and next_type == 'half_ppr' and avg_diff > 0:
                            print(f"✅ {db_name}: PPR vs Half-PPR difference reasonable ({avg_diff:.2f} pts, {sample_size:,} samples)")
                        elif current == 'half_ppr' and next_type == 'standard' and avg_diff > 0:
                            print(f"✅ {db_name}: Half-PPR vs Standard difference reasonable ({avg_diff:.2f} pts, {sample_size:,} samples)")
                        else:
                            print(f"⚠️  {db_name}: {current} vs {next_type} difference: {avg_diff:.2f} pts")
                    else:
                        print(f"⚠️  {db_name}: No differences found between {current} and {next_type}")
            
            return True
    except Exception as e:
        print(f"❌ {db_name}: Scoring consistency test failed - {e}")
        return False

def test_single_database(db_path, db_name, expected_seasons):
    """Test a single database comprehensively"""
    print(f"\n{'='*60}")
    print(f"Testing Database: {db_name}")
    print(f"{'='*60}")
    
    # Test connectivity
    engine = test_database_connectivity(db_path, db_name)
    if not engine:
        return False
    
    # Define expected tables based on database type
    if 'multi' in db_name:
        expected_tables = {'players', 'games', 'player_game_stats_ppr', 'player_game_stats_half_ppr', 'player_game_stats_standard'}
    else:
        expected_tables = {'players', 'games', 'player_game_stats'}
    
    # Test table structure
    if not test_table_structure(engine, db_name, expected_tables):
        return False
    
    # Test data counts
    total_records = 0
    for table in expected_tables:
        min_count = 1000 if table == 'players' else (100 if table == 'games' else 1000)
        count = test_data_counts(engine, db_name, table, min_count)
        total_records += count
    
    # Test season coverage
    for table in ['games', 'player_game_stats']:
        if table in expected_tables or any(table in t for t in expected_tables):
            test_season_coverage(engine, db_name, table, expected_seasons)
    
    # Test data quality
    for table in expected_tables:
        test_data_quality(engine, db_name, table)
    
    # Test scoring consistency for multi-scoring databases
    if 'multi' in db_name:
        test_scoring_consistency(engine, db_name)
    
    print(f"✅ {db_name}: All tests completed successfully")
    return True

def main():
    """Main test function"""
    print("🏈 Fantasy Football Database Test Suite")
    print(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define expected seasons (complete 2018-2024 dataset)
    expected_seasons = {2018, 2019, 2020, 2021, 2022, 2023, 2024}
    
    # Test databases (only complete 2018-2024 databases)
    databases_to_test = [
        ("data/fantasy_ppr.db", "PPR Database"),
        ("data/fantasy_halfppr.db", "Half-PPR Database"), 
        ("data/fantasy_standard.db", "Standard Database")
    ]
    
    results = {}
    for db_path, db_name in databases_to_test:
        if os.path.exists(db_path):
            results[db_name] = test_single_database(db_path, db_name, expected_seasons)
        else:
            print(f"\n{'='*60}")
            print(f"Database not found: {db_path}")
            print(f"{'='*60}")
            results[db_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for db_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{db_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} databases passed all tests")
    
    if passed == total:
        print("🎉 All databases are properly loaded and ready for use!")
    else:
        print("⚠️  Some databases have issues. Check the output above for details.")
    
    # File size summary
    print(f"\n{'='*60}")
    print("DATABASE SIZES")
    print(f"{'='*60}")
    
    for db_path, db_name in databases_to_test:
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"{db_name:<25} {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
