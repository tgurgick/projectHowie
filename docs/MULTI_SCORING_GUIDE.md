# Multi-Scoring Fantasy Football Database Guide

## Overview

This guide explains how to handle multiple scoring types (PPR, Half-PPR, Standard) in your fantasy football database.

## Available Scoring Types

- **PPR**: 1.0 point per reception
- **Half-PPR**: 0.5 points per reception  
- **Standard**: 0.0 points per reception

All other scoring remains the same:
- Passing: 0.04 pts/yd, 4 pts/TD, -2 pts/INT
- Rushing: 0.1 pts/yd, 6 pts/TD
- Receiving: 0.1 pts/yd, 6 pts/TD
- Fumbles: -2 pts

## Option 1: Separate Databases (Recommended)

### Pros:
- ✅ Clean separation of data
- ✅ Easy to manage and backup
- ✅ No complex queries
- ✅ Can optimize each database independently

### Cons:
- ❌ More disk space (3x the data)
- ❌ Need to maintain multiple databases

### Usage:

```bash
# Build all scoring types at once
python3 build_all_scoring.py

# Or build individually
python3 build_fantasy_db_simple.py --start 2018 --end 2024 --scoring ppr --db-url sqlite:///fantasy_ppr.db
python3 build_fantasy_db_simple.py --start 2018 --end 2024 --scoring half_ppr --db-url sqlite:///fantasy_halfppr.db
python3 build_fantasy_db_simple.py --start 2018 --end 2024 --scoring standard --db-url sqlite:///fantasy_standard.db
```

### Database Files Created:
- `fantasy_ppr.db` - PPR scoring
- `fantasy_halfppr.db` - Half-PPR scoring  
- `fantasy_standard.db` - Standard scoring

## Option 2: Single Database with Multiple Tables (Deprecated)

**Note:** This option has been deprecated in favor of separate databases for better data completeness and management.

### Why Deprecated:
- ❌ Limited to single season (2024 only)
- ❌ More complex queries
- ❌ Incomplete historical data

### Recommendation:
Use **Option 1 (Separate Databases)** for complete 2018-2024 data across all scoring types.

## Comparing Scoring Types

Use the comparison script to see differences:

```bash
python3 compare_scoring.py
```

This will show:
- Top players by each scoring type
- Average differences between scoring types
- Players most affected by scoring changes

## Example Output

```
=== Fantasy Points Comparison Across Scoring Types ===
Available scoring types: ['ppr', 'half_ppr']
Top 20 players by PPR points (2024, min 10 games):
====================================================================================================
Rank Name                 Pos Team PPR      HALF_PPR Games 
----------------------------------------------------------------------------------------------------
1    CeeDee Lamb          WR  DAL  481.5    481.5    17
2    Tyreek Hill          WR  MIA  451.7    428.7    17
3    Josh Allen           QB  BUF  438.3    438.3    17
...

=== Scoring Differences ===
Average PPR vs HALF_PPR difference: 17.9 points

Biggest PPR vs Half-PPR differences:
  Ja'Marr Chase: 63.5 points
  Amon-Ra St. Brown: 61.5 points
  Justin Jefferson: 54.0 points
```

## Recommendations

### For Most Users: Option 1 (Separate Databases)
- Easier to understand and manage
- Better performance for single-scoring queries
- Can easily delete unwanted scoring types

### For All Users: Option 1 (Separate Databases) - Recommended
- Complete 2018-2024 historical data
- Better performance for single-scoring queries
- Can easily delete unwanted scoring types
- Cleaner data management

## Incremental Loading

Both options support incremental loading:
- Players: Loaded once (all players from all time)
- Games: Only loads missing seasons
- Player Stats: Only loads missing seasons for each scoring type

## File Sizes (Approximate)

For 2018-2024 data:
- Single scoring type: ~5.4 MB
- All three scoring types (separate): ~16.1 MB
- Each database contains complete 2018-2024 historical data

## Next Steps

1. Choose your preferred approach
2. Run the appropriate build script
3. Use `compare_scoring.py` to analyze differences
4. Create your own analysis scripts using the database(s)
