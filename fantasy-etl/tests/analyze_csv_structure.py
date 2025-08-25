#!/usr/bin/env python3
"""
Analyze PFF CSV file structure
Helps understand the format of route data CSV files
"""

import pandas as pd
import sys
import os

def analyze_csv_file(csv_file: str):
    """Analyze a single CSV file"""
    print(f"🔍 Analyzing CSV file: {os.path.basename(csv_file)}")
    print("=" * 60)
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ Could not read CSV file with any encoding")
            return
        
        # Basic info
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📋 Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Column analysis
        print(f"\n📋 All Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Route-related columns
        route_keywords = ['route', 'target', 'reception', 'catch', 'separation', 'cushion', 'air', 'depth']
        route_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in route_keywords):
                route_columns.append(col)
        
        if route_columns:
            print(f"\n🎯 Route-related columns ({len(route_columns)}):")
            for col in route_columns:
                print(f"  ✅ {col}")
        else:
            print(f"\n❌ No obvious route-related columns found")
        
        # Data types
        print(f"\n📊 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Sample data
        print(f"\n📄 Sample Data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Missing data
        print(f"\n❓ Missing Data:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / len(df)) * 100
                    print(f"  {col}: {missing_count} ({percentage:.1f}%)")
        else:
            print("  No missing data found")
        
        # Unique values for key columns
        print(f"\n🔢 Unique Values (for first 5 columns):")
        for col in df.columns[:5]:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {sorted(df[col].unique())}")
        
        # Route data suggestions
        print(f"\n💡 Route Data Suggestions:")
        
        # Look for player identification
        player_cols = [col for col in df.columns if any(term in col.lower() for term in ['player', 'name', 'receiver'])]
        if player_cols:
            print(f"  Player identification: {player_cols}")
        
        # Look for game identification
        game_cols = [col for col in df.columns if any(term in col.lower() for term in ['game', 'week', 'season', 'date'])]
        if game_cols:
            print(f"  Game identification: {game_cols}")
        
        # Look for route metrics
        route_metric_cols = [col for col in df.columns if any(term in col.lower() for term in ['route', 'target', 'reception'])]
        if route_metric_cols:
            print(f"  Route metrics: {route_metric_cols}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_csv_structure.py <csv_file>")
        print("Example: python analyze_csv_structure.py pff_route_data_2024.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    analyze_csv_file(csv_file)

if __name__ == "__main__":
    main()
