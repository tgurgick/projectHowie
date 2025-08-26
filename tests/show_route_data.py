#!/usr/bin/env python3
"""
Show imported PFF route running data
"""

import pandas as pd
import sys
import argparse
from sqlalchemy import create_engine, text

def show_route_data(db_url: str, season: int = None, position: str = None, limit: int = 20):
    """Show route data from database"""
    engine = create_engine(db_url, future=True)
    
    print(f"🔍 PFF Route Data from: {db_url}")
    print("=" * 80)
    
    # Build query
    query = "SELECT * FROM player_route_stats"
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
    
    query += " ORDER BY routes_run DESC"
    query += f" LIMIT {limit}"
    
    # Execute query
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    
    if df.empty:
        print("❌ No route data found")
        return
    
    # Show summary
    print(f"📊 Showing {len(df)} players")
    if season:
        print(f"📅 Season: {season}")
    if position:
        print(f"🏈 Position: {position}")
    
    # Show top route runners
    print(f"\n🏃 Top Route Runners:")
    display_cols = ['player_name', 'team', 'position', 'routes_run', 'route_participation', 'route_grade', 'yards_per_route_run']
    available_cols = [col for col in display_cols if col in df.columns]
    
    print(df[available_cols].head(10).to_string(index=False))
    
    # Show top route grades
    if 'route_grade' in df.columns:
        print(f"\n⭐ Top Route Grades:")
        grade_cols = ['player_name', 'team', 'position', 'route_grade', 'routes_run', 'yards_per_route_run']
        available_grade_cols = [col for col in grade_cols if col in df.columns]
        print(df.nlargest(10, 'route_grade')[available_grade_cols].to_string(index=False))
    
    # Show top YPRR
    if 'yards_per_route_run' in df.columns:
        print(f"\n📈 Top Yards Per Route Run:")
        yprr_cols = ['player_name', 'team', 'position', 'yards_per_route_run', 'routes_run', 'targets']
        available_yprr_cols = [col for col in yprr_cols if col in df.columns]
        print(df.nlargest(10, 'yards_per_route_run')[available_yprr_cols].to_string(index=False))
    
    # Show route participation leaders
    if 'route_participation' in df.columns:
        print(f"\n🎯 Top Route Participation:")
        participation_cols = ['player_name', 'team', 'position', 'route_participation', 'routes_run', 'targets']
        available_participation_cols = [col for col in participation_cols if col in df.columns]
        print(df.nlargest(10, 'route_participation')[available_participation_cols].to_string(index=False))
    
    # Show contested catch leaders
    if 'contested_catch_rate' in df.columns:
        contested_data = df[df['contested_targets'] >= 5].nlargest(10, 'contested_catch_rate')
        if not contested_data.empty:
            print(f"\n🥊 Top Contested Catch Rate (min 5 targets):")
            contested_cols = ['player_name', 'team', 'position', 'contested_catch_rate', 'contested_targets', 'contested_receptions']
            available_contested_cols = [col for col in contested_cols if col in df.columns]
            print(contested_data[available_contested_cols].to_string(index=False))
    
    # Show slot vs wide breakdown
    if 'slot_rate' in df.columns and 'wide_rate' in df.columns:
        print(f"\n📍 Slot vs Wide Breakdown:")
        slot_players = df[df['slot_rate'] > 50].nlargest(5, 'slot_rate')
        wide_players = df[df['wide_rate'] > 50].nlargest(5, 'wide_rate')
        
        if not slot_players.empty:
            print("  Top Slot Players:")
            slot_cols = ['player_name', 'team', 'slot_rate', 'routes_run']
            available_slot_cols = [col for col in slot_cols if col in df.columns]
            print(slot_players[available_slot_cols].to_string(index=False))
        
        if not wide_players.empty:
            print("  Top Wide Players:")
            wide_cols = ['player_name', 'team', 'wide_rate', 'routes_run']
            available_wide_cols = [col for col in wide_cols if col in df.columns]
            print(wide_players[available_wide_cols].to_string(index=False))
    
    engine.dispose()

def show_route_summary(db_url: str):
    """Show summary of route data"""
    engine = create_engine(db_url, future=True)
    
    print(f"📊 Route Data Summary: {db_url}")
    print("=" * 60)
    
    # Get season summary
    with engine.connect() as conn:
        season_summary = pd.read_sql(text("""
            SELECT season, 
                   COUNT(*) as total_players,
                   SUM(CASE WHEN position = 'WR' THEN 1 ELSE 0 END) as wrs,
                   SUM(CASE WHEN position = 'TE' THEN 1 ELSE 0 END) as tes,
                   SUM(CASE WHEN position = 'HB' THEN 1 ELSE 0 END) as hbs,
                   SUM(CASE WHEN position = 'FB' THEN 1 ELSE 0 END) as fbs,
                   SUM(routes_run) as total_routes,
                   AVG(route_grade) as avg_grade
            FROM player_route_stats 
            GROUP BY season 
            ORDER BY season
        """), conn)
    
    print("📈 Season Breakdown:")
    for _, row in season_summary.iterrows():
        print(f"  {row['season']}: {row['total_players']} players, "
              f"{row['wrs']} WRs, {row['tes']} TEs, "
              f"{row['total_routes']:,.0f} routes, "
              f"avg grade: {row['avg_grade']:.1f}")
    
    # Get position summary
    with engine.connect() as conn:
        pos_summary = pd.read_sql(text("""
            SELECT position, 
                   COUNT(*) as total_players,
                   SUM(routes_run) as total_routes,
                   AVG(route_grade) as avg_grade,
                   AVG(yards_per_route_run) as avg_yprr
            FROM player_route_stats 
            GROUP BY position 
            ORDER BY total_players DESC
        """), conn)
    
    print(f"\n🏈 Position Summary:")
    for _, row in pos_summary.iterrows():
        print(f"  {row['position']}: {row['total_players']} players, "
              f"{row['total_routes']:,.0f} routes, "
              f"avg grade: {row['avg_grade']:.1f}, "
              f"avg YPRR: {row['avg_yprr']:.2f}")
    
    engine.dispose()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Show PFF route data")
    parser.add_argument("--db-url", type=str, default="sqlite:///data/fantasy_ppr.db", 
                       help="Database URL")
    parser.add_argument("--season", type=int, help="Filter by season")
    parser.add_argument("--position", type=str, help="Filter by position (WR, TE, HB, FB)")
    parser.add_argument("--limit", type=int, default=20, help="Number of results to show")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    
    args = parser.parse_args()
    
    if args.summary:
        show_route_summary(args.db_url)
    else:
        show_route_data(args.db_url, args.season, args.position, args.limit)

if __name__ == "__main__":
    main()
