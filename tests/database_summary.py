#!/usr/bin/env python3
"""
Quick summary of all fantasy football databases
Shows key differences and statistics
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

def get_database_summary(db_path, db_name):
    """Get summary statistics for a database"""
    if not os.path.exists(db_path):
        return None
    
    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        
        with engine.connect() as conn:
            # Get table counts
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table'"), conn)
            
            summary = {
                'name': db_name,
                'size_mb': os.path.getsize(db_path) / (1024 * 1024),
                'tables': len(tables)
            }
            
            # Get player count
            result = conn.execute(text("SELECT COUNT(*) FROM players"))
            summary['players'] = result.scalar()
            
            # Get game count and seasons
            result = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT season) FROM games"))
            game_count, season_count = result.fetchone()
            summary['games'] = game_count
            summary['seasons'] = season_count
            
            # Check if this is a multi-scoring database
            tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_game_stats_%'"), conn)
            scoring_tables = [table.replace('player_game_stats_', '') for table in tables['name'].tolist()]
            
            if scoring_tables:
                # Multi-scoring database
                total_stats = 0
                all_points = []
                
                for table in scoring_tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM player_game_stats_{table}"))
                    total_stats += result.scalar()
                    
                    result = conn.execute(text(f"SELECT fantasy_points FROM player_game_stats_{table}"))
                    points = [row[0] for row in result.fetchall()]
                    all_points.extend(points)
                
                summary['player_stats'] = total_stats
                summary['min_points'] = min(all_points) if all_points else 0
                summary['max_points'] = max(all_points) if all_points else 0
                summary['avg_points'] = sum(all_points) / len(all_points) if all_points else 0
                summary['scoring_types'] = scoring_tables
            else:
                # Single scoring database
                result = conn.execute(text("SELECT COUNT(*) FROM player_game_stats"))
                summary['player_stats'] = result.scalar()
                
                result = conn.execute(text("SELECT MIN(fantasy_points), MAX(fantasy_points), AVG(fantasy_points) FROM player_game_stats"))
                min_pts, max_pts, avg_pts = result.fetchone()
                summary['min_points'] = min_pts
                summary['max_points'] = max_pts
                summary['avg_points'] = avg_pts
                summary['scoring_types'] = ['single']
            
            # Get seasons
            seasons = pd.read_sql(text("SELECT DISTINCT season FROM games ORDER BY season"), conn)
            summary['season_list'] = seasons['season'].tolist()
            
            return summary
            
    except Exception as e:
        print(f"Error reading {db_name}: {e}")
        return None

def main():
    """Main summary function"""
    print("🏈 Fantasy Football Database Summary")
    print("=" * 80)
    
    databases = [
        ("data/fantasy_ppr.db", "PPR Database"),
        ("data/fantasy_halfppr.db", "Half-PPR Database"),
        ("data/fantasy_standard.db", "Standard Database")
    ]
    
    summaries = []
    for db_path, db_name in databases:
        summary = get_database_summary(db_path, db_name)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        print("No databases found!")
        return
    
    # Print summary table
    print(f"{'Database':<25} {'Size':<8} {'Players':<8} {'Games':<8} {'Seasons':<8} {'Stats':<10} {'Avg Pts':<8}")
    print("-" * 80)
    
    for summary in summaries:
        print(f"{summary['name']:<25} {summary['size_mb']:<8.1f} {summary['players']:<8,} {summary['games']:<8,} "
              f"{summary['seasons']:<8} {summary['player_stats']:<10,} {summary['avg_points']:<8.1f}")
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for summary in summaries:
        print(f"\n📊 {summary['name']}")
        print(f"   Size: {summary['size_mb']:.1f} MB")
        print(f"   Players: {summary['players']:,}")
        print(f"   Games: {summary['games']:,} across {summary['seasons']} seasons")
        print(f"   Player Stats: {summary['player_stats']:,}")
        print(f"   Fantasy Points: {summary['min_points']:.1f} to {summary['max_points']:.1f} (avg {summary['avg_points']:.1f})")
        print(f"   Seasons: {summary['season_list']}")
    
    # Compare scoring differences
    print(f"\n" + "=" * 80)
    print("SCORING COMPARISON")
    print("=" * 80)
    
    # Find databases with different scoring
    scoring_dbs = [s for s in summaries if 'PPR' in s['name'] or 'Standard' in s['name'] or 'Half-PPR' in s['name']]
    
    if len(scoring_dbs) >= 2:
        print(f"{'Database':<20} {'Min Pts':<8} {'Max Pts':<8} {'Avg Pts':<8} {'Total Stats':<12}")
        print("-" * 60)
        
        for db in scoring_dbs:
            print(f"{db['name']:<20} {db['min_points']:<8.1f} {db['max_points']:<8.1f} {db['avg_points']:<8.1f} {db['player_stats']:<12,}")
        
        # Calculate differences
        if len(scoring_dbs) >= 3:
            ppr_db = next((db for db in scoring_dbs if 'PPR Database' in db['name']), None)
            half_db = next((db for db in scoring_dbs if 'Half-PPR Database' in db['name']), None)
            std_db = next((db for db in scoring_dbs if 'Standard Database' in db['name']), None)
            
            if ppr_db and half_db and std_db:
                print(f"\n📈 Scoring Differences:")
                print(f"   PPR vs Half-PPR avg difference: {ppr_db['avg_points'] - half_db['avg_points']:.1f} points")
                print(f"   Half-PPR vs Standard avg difference: {half_db['avg_points'] - std_db['avg_points']:.1f} points")
                print(f"   PPR vs Standard avg difference: {ppr_db['avg_points'] - std_db['avg_points']:.1f} points")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find the most complete database
    most_complete = max(summaries, key=lambda x: x['seasons'])
    print(f"🎯 Most complete database: {most_complete['name']} ({most_complete['seasons']} seasons)")
    
    # Find the largest database
    largest = max(summaries, key=lambda x: x['size_mb'])
    print(f"💾 Largest database: {largest['name']} ({largest['size_mb']:.1f} MB)")
    
    # Check for multi-scoring database
    multi_db = next((db for db in summaries if 'Multi-Scoring' in db['name']), None)
    if multi_db:
        print(f"🔄 Multi-scoring database available: {multi_db['name']}")
        print(f"   Contains multiple scoring types in one database")
    
    print(f"\n✅ All databases are ready for analysis!")

if __name__ == "__main__":
    main()
