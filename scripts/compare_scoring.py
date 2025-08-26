#!/usr/bin/env python3
"""
Compare fantasy points across different scoring types
"""

import pandas as pd
from sqlalchemy import create_engine, text

def compare_scoring_types():
    """Compare fantasy points across different scoring types"""
    
    # Connect to the multi-scoring database
    engine = create_engine("sqlite:///fantasy_multi.db", future=True)
    
    print("=== Fantasy Points Comparison Across Scoring Types ===")
    
    # Check which scoring tables exist
    with engine.connect() as conn:
        tables = pd.read_sql(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_game_stats_%'"), conn)
    
    scoring_tables = [table.replace('player_game_stats_', '') for table in tables['name'].tolist()]
    print(f"Available scoring types: {scoring_tables}")
    
    if len(scoring_tables) < 2:
        print("Need at least 2 scoring types to compare")
        return
    
    # Build dynamic query based on available tables
    select_parts = []
    join_parts = []
    
    for i, scoring_type in enumerate(scoring_tables):
        table_alias = scoring_type.replace('_', '')  # ppr, halfppr, standard
        select_parts.append(f"SUM({table_alias}.fantasy_points) as {scoring_type}_points")
        if i == 0:
            join_parts.append(f"JOIN player_game_stats_{scoring_type} {table_alias} ON p.player_id = {table_alias}.player_id")
        else:
            join_parts.append(f"JOIN player_game_stats_{scoring_type} {table_alias} ON p.player_id = {table_alias}.player_id AND ppr.game_id = {table_alias}.game_id")
    
    query = f"""
    SELECT 
        p.name,
        p.position,
        p.team,
        {', '.join(select_parts)},
        COUNT(*) as games_played
    FROM players p
    {' '.join(join_parts)}
    JOIN games g ON ppr.game_id = g.game_id
    WHERE g.season = 2024
    GROUP BY p.player_id
    HAVING games_played >= 10
    ORDER BY ppr_points DESC
    LIMIT 20
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        print(f"Top 20 players by {scoring_tables[0].upper()} points (2024, min 10 games):")
        print("=" * 100)
        
        # Build header dynamically
        header_parts = ['Rank', 'Name', 'Pos', 'Team']
        for scoring_type in scoring_tables:
            header_parts.append(scoring_type.upper())
        header_parts.append('Games')
        
        header_format = f"{'Rank':<4} {'Name':<20} {'Pos':<3} {'Team':<3}"
        for scoring_type in scoring_tables:
            header_format += f" {{{scoring_type.upper()}:<8}}"
        header_format += f" {'Games':<6}"
        
        print(header_format.format(**{scoring_type.upper(): scoring_type.upper() for scoring_type in scoring_tables}))
        print("-" * 100)
        
        # Print rows
        for i, row in df.iterrows():
            row_data = {
                'Rank': i+1,
                'Name': row['name'],
                'Pos': row['position'],
                'Team': row['team'],
                'Games': row['games_played']
            }
            for scoring_type in scoring_tables:
                row_data[scoring_type.upper()] = f"{row[f'{scoring_type}_points']:.1f}"
            
            print(header_format.format(**row_data))
        
        # Calculate differences
        if len(scoring_tables) >= 2:
            print(f"\n=== Scoring Differences ===")
            for i in range(len(scoring_tables) - 1):
                current = scoring_tables[i]
                next_type = scoring_tables[i + 1]
                diff_col = f'{current}_{next_type}_diff'
                df[diff_col] = df[f'{current}_points'] - df[f'{next_type}_points']
                print(f"Average {current.upper()} vs {next_type.upper()} difference: {df[diff_col].mean():.1f} points")
            
            # Show biggest differences
            if 'ppr' in scoring_tables and 'half_ppr' in scoring_tables:
                print(f"\nBiggest PPR vs Half-PPR differences:")
                biggest_diff = df.nlargest(5, 'ppr_half_ppr_diff')
                for _, row in biggest_diff.iterrows():
                    print(f"  {row['name']}: {row['ppr_half_ppr_diff']:.1f} points")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run build_fantasy_db_multi_scoring.py first")

def show_scoring_rules():
    """Show the scoring rules for each format"""
    from build_fantasy_db import SCORING_PRESETS
    
    print("\n=== Scoring Rules ===")
    for scoring_type, rules in SCORING_PRESETS.items():
        print(f"\n{scoring_type.upper()}:")
        for stat, points in rules.items():
            print(f"  {stat}: {points} points")

if __name__ == "__main__":
    show_scoring_rules()
    compare_scoring_types()
