#!/usr/bin/env python3
"""
Show PFF scheme splits data (Man vs Zone coverage)
"""

import pandas as pd
import sys
import argparse
from sqlalchemy import create_engine, text

def show_scheme_data(db_url: str, season: int = None, position: str = None, limit: int = 20):
    """Show scheme splits data from database"""
    engine = create_engine(db_url, future=True)
    
    print(f"🔍 PFF Scheme Splits Data from: {db_url}")
    print("=" * 80)
    
    # Build query
    query = "SELECT * FROM player_scheme_stats"
    conditions = []
    params = {}
    
    if season:
        conditions.append("season = :season")
        params['season'] = season
    
    if position:
        conditions.append("position = :position")
        params['position'] = position
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY total_targets DESC"
    query += f" LIMIT {limit}"
    
    # Execute query
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    
    if df.empty:
        print("❌ No scheme data found")
        return
    
    # Show summary
    print(f"📊 Showing {len(df)} players")
    if season:
        print(f"📅 Season: {season}")
    if position:
        print(f"🏈 Position: {position}")
    
    # Show top players by total targets
    print(f"\n🎯 Top Players by Total Targets:")
    display_cols = ['player_name', 'team', 'position', 'total_targets', 'man_targets', 'zone_targets']
    available_cols = [col for col in display_cols if col in df.columns]
    
    print(df[available_cols].head(10).to_string(index=False))
    
    # Show man coverage specialists
    if 'man_yards_per_route_run' in df.columns:
        print(f"\n👤 Top Man Coverage Specialists (YPRR):")
        man_cols = ['player_name', 'team', 'position', 'man_yards_per_route_run', 'man_route_grade', 'man_targets']
        available_man_cols = [col for col in man_cols if col in df.columns]
        man_specialists = df[df['man_targets'] >= 20].nlargest(10, 'man_yards_per_route_run')
        if not man_specialists.empty:
            print(man_specialists[available_man_cols].to_string(index=False))
    
    # Show zone coverage specialists
    if 'zone_yards_per_route_run' in df.columns:
        print(f"\n🛡️ Top Zone Coverage Specialists (YPRR):")
        zone_cols = ['player_name', 'team', 'position', 'zone_yards_per_route_run', 'zone_route_grade', 'zone_targets']
        available_zone_cols = [col for col in zone_cols if col in df.columns]
        zone_specialists = df[df['zone_targets'] >= 20].nlargest(10, 'zone_yards_per_route_run')
        if not zone_specialists.empty:
            print(zone_specialists[available_zone_cols].to_string(index=False))
    
    # Show scheme efficiency differences
    if 'yprr_man_vs_zone_diff' in df.columns:
        print(f"\n📈 Best Man vs Zone Efficiency (YPRR Difference):")
        diff_cols = ['player_name', 'team', 'position', 'yprr_man_vs_zone_diff', 'man_yards_per_route_run', 'zone_yards_per_route_run']
        available_diff_cols = [col for col in diff_cols if col in df.columns]
        man_better = df[df['yprr_man_vs_zone_diff'] > 0].nlargest(10, 'yprr_man_vs_zone_diff')
        if not man_better.empty:
            print("  Better vs Man Coverage:")
            print(man_better[available_diff_cols].to_string(index=False))
        
        zone_better = df[df['yprr_man_vs_zone_diff'] < 0].nsmallest(10, 'yprr_man_vs_zone_diff')
        if not zone_better.empty:
            print("  Better vs Zone Coverage:")
            print(zone_better[available_diff_cols].to_string(index=False))
    
    # Show route grade differences
    if 'route_grade_man_vs_zone_diff' in df.columns:
        print(f"\n⭐ Route Grade Differences (Man vs Zone):")
        grade_cols = ['player_name', 'team', 'position', 'route_grade_man_vs_zone_diff', 'man_route_grade', 'zone_route_grade']
        available_grade_cols = [col for col in grade_cols if col in df.columns]
        man_better_grade = df[df['route_grade_man_vs_zone_diff'] > 0].nlargest(10, 'route_grade_man_vs_zone_diff')
        if not man_better_grade.empty:
            print("  Better Route Grades vs Man:")
            print(man_better_grade[available_grade_cols].to_string(index=False))
        
        zone_better_grade = df[df['route_grade_man_vs_zone_diff'] < 0].nsmallest(10, 'route_grade_man_vs_zone_diff')
        if not zone_better_grade.empty:
            print("  Better Route Grades vs Zone:")
            print(zone_better_grade[available_grade_cols].to_string(index=False))
    
    # Show contested catch specialists
    if 'man_contested_catch_rate' in df.columns and 'zone_contested_catch_rate' in df.columns:
        print(f"\n🥊 Contested Catch Specialists:")
        contested_cols = ['player_name', 'team', 'position', 'man_contested_catch_rate', 'zone_contested_catch_rate', 'man_contested_targets', 'zone_contested_targets']
        available_contested_cols = [col for col in contested_cols if col in df.columns]
        contested_data = df[(df['man_contested_targets'] >= 5) | (df['zone_contested_targets'] >= 5)].nlargest(10, 'man_contested_catch_rate')
        if not contested_data.empty:
            print(contested_data[available_contested_cols].to_string(index=False))
    
    engine.dispose()

def show_scheme_summary(db_url: str):
    """Show summary of scheme data"""
    engine = create_engine(db_url, future=True)
    
    print(f"📊 Scheme Data Summary: {db_url}")
    print("=" * 60)
    
    # Get season summary
    with engine.connect() as conn:
        season_summary = pd.read_sql(text("""
            SELECT season, 
                   COUNT(*) as total_players,
                   SUM(CASE WHEN position = 'WR' THEN 1 ELSE 0 END) as wrs,
                   SUM(CASE WHEN position = 'TE' THEN 1 ELSE 0 END) as tes,
                   SUM(CASE WHEN position = 'HB' THEN 1 ELSE 0 END) as hbs,
                   SUM(man_routes_run) as total_man_routes,
                   SUM(zone_routes_run) as total_zone_routes,
                   AVG(man_route_grade) as avg_man_grade,
                   AVG(zone_route_grade) as avg_zone_grade,
                   AVG(man_yards_per_route_run) as avg_man_yprr,
                   AVG(zone_yards_per_route_run) as avg_zone_yprr
            FROM player_scheme_stats 
            GROUP BY season 
            ORDER BY season
        """), conn)
    
    print("📈 Season Breakdown:")
    for _, row in season_summary.iterrows():
        print(f"  {row['season']}: {row['total_players']} players, "
              f"{row['wrs']} WRs, {row['tes']} TEs, "
              f"{row['total_man_routes']:,.0f} man routes, "
              f"{row['total_zone_routes']:,.0f} zone routes, "
              f"avg man grade: {row['avg_man_grade']:.1f}, "
              f"avg zone grade: {row['avg_zone_grade']:.1f}")
    
    # Get position summary
    with engine.connect() as conn:
        pos_summary = pd.read_sql(text("""
            SELECT position, 
                   COUNT(*) as total_players,
                   SUM(man_routes_run) as total_man_routes,
                   SUM(zone_routes_run) as total_zone_routes,
                   AVG(man_route_grade) as avg_man_grade,
                   AVG(zone_route_grade) as avg_zone_grade,
                   AVG(man_yards_per_route_run) as avg_man_yprr,
                   AVG(zone_yards_per_route_run) as avg_zone_yprr
            FROM player_scheme_stats 
            GROUP BY position 
            ORDER BY total_players DESC
        """), conn)
    
    print(f"\n🏈 Position Summary:")
    for _, row in pos_summary.iterrows():
        print(f"  {row['position']}: {row['total_players']} players, "
              f"{row['total_man_routes']:,.0f} man routes, "
              f"{row['total_zone_routes']:,.0f} zone routes, "
              f"avg man grade: {row['avg_man_grade']:.1f}, "
              f"avg zone grade: {row['avg_zone_grade']:.1f}")
    
    # Show scheme efficiency comparison
    with engine.connect() as conn:
        efficiency_summary = pd.read_sql(text("""
            SELECT 
                   AVG(man_yards_per_route_run) as avg_man_yprr,
                   AVG(zone_yards_per_route_run) as avg_zone_yprr,
                   AVG(man_route_grade) as avg_man_grade,
                   AVG(zone_route_grade) as avg_zone_grade,
                   AVG(yprr_man_vs_zone_diff) as avg_yprr_diff,
                   AVG(route_grade_man_vs_zone_diff) as avg_grade_diff
            FROM player_scheme_stats
        """), conn)
    
    if not efficiency_summary.empty:
        row = efficiency_summary.iloc[0]
        print(f"\n📊 Overall Scheme Efficiency:")
        print(f"  Man Coverage: {row['avg_man_yprr']:.2f} YPRR, {row['avg_man_grade']:.1f} grade")
        print(f"  Zone Coverage: {row['avg_zone_yprr']:.2f} YPRR, {row['avg_zone_grade']:.1f} grade")
        print(f"  Average Difference: {row['avg_yprr_diff']:.2f} YPRR, {row['avg_grade_diff']:.1f} grade")
    
    engine.dispose()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Show PFF scheme splits data")
    parser.add_argument("--db-url", type=str, default="sqlite:///data/fantasy_ppr.db", 
                       help="Database URL")
    parser.add_argument("--season", type=int, help="Filter by season")
    parser.add_argument("--position", type=str, help="Filter by position (WR, TE, HB, FB)")
    parser.add_argument("--limit", type=int, default=20, help="Number of results to show")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    
    args = parser.parse_args()
    
    if args.summary:
        show_scheme_summary(args.db_url)
    else:
        show_scheme_data(args.db_url, args.season, args.position, args.limit)

if __name__ == "__main__":
    main()

