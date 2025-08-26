#!/usr/bin/env python3
"""
Show sample of PFF route data before import
"""

import pandas as pd
import sys
import os

def show_pff_route_sample(csv_file: str):
    """Show sample of PFF route data"""
    print(f"🔍 PFF Route Data Sample: {os.path.basename(csv_file)}")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Show basic info
    print(f"📊 Total players: {len(df)}")
    print(f"📋 Position breakdown:")
    for pos in df['position'].value_counts().items():
        print(f"  {pos[0]}: {pos[1]}")
    
    # Show top route runners
    print(f"\n🏃 Top Route Runners (by routes run):")
    top_routes = df.nlargest(10, 'routes')[['player', 'team_name', 'position', 'routes', 'route_rate', 'grades_pass_route']]
    print(top_routes.to_string(index=False))
    
    # Show top route grades
    print(f"\n⭐ Top Route Grades:")
    top_grades = df.nlargest(10, 'grades_pass_route')[['player', 'team_name', 'position', 'grades_pass_route', 'routes', 'yprr']]
    print(top_grades.to_string(index=False))
    
    # Show top YPRR (yards per route run)
    print(f"\n📈 Top Yards Per Route Run:")
    top_yprr = df.nlargest(10, 'yprr')[['player', 'team_name', 'position', 'yprr', 'routes', 'targets']]
    print(top_yprr.to_string(index=False))
    
    # Show route participation leaders
    print(f"\n🎯 Top Route Participation:")
    top_participation = df.nlargest(10, 'route_rate')[['player', 'team_name', 'position', 'route_rate', 'routes', 'targets']]
    print(top_participation.to_string(index=False))
    
    # Show contested catch leaders
    print(f"\n🥊 Top Contested Catch Rate:")
    contested_data = df[df['contested_targets'] >= 5].nlargest(10, 'contested_catch_rate')
    if not contested_data.empty:
        print(contested_data[['player', 'team_name', 'position', 'contested_catch_rate', 'contested_targets', 'contested_receptions']].to_string(index=False))
    else:
        print("  No players with 5+ contested targets")
    
    # Show slot vs wide breakdown
    print(f"\n📍 Slot vs Wide Breakdown:")
    slot_players = df[df['slot_rate'] > 50].nlargest(5, 'slot_rate')
    wide_players = df[df['wide_rate'] > 50].nlargest(5, 'wide_rate')
    
    print("  Top Slot Players:")
    print(slot_players[['player', 'team_name', 'slot_rate', 'routes']].to_string(index=False))
    
    print("  Top Wide Players:")
    print(wide_players[['player', 'team_name', 'wide_rate', 'routes']].to_string(index=False))
    
    print("\n" + "=" * 80)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python show_pff_route_sample.py <csv_file>")
        print("Example: python show_pff_route_sample.py data/pff_csv/receiving_2025.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    show_pff_route_sample(csv_file)

if __name__ == "__main__":
    main()
