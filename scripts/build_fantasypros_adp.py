#!/usr/bin/env python3
"""
FantasyPros ADP Scraper

Scrapes Average Draft Position (ADP) data from FantasyPros for multiple scoring formats.
Supports PPR, Half-PPR, and Standard scoring.
"""

import sys
import argparse
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from sqlalchemy import create_engine, text
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import the database models and helpers
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fantasy_db import (
    FantasyMarket, ADPData, Base, ensure_dt,
    DEFAULT_DB_URL
)

@dataclass
class Args:
    season: int
    db_url: str
    scoring: str
    test: bool

class FantasyProsADPScraper:
    """Scraper for FantasyPros ADP data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.fantasypros.com/nfl/adp"
        
    def get_scoring_url(self, scoring: str, year: int = None) -> str:
        """Get the correct URL for the scoring format"""
        scoring_map = {
            'ppr': 'ppr-overall.php',
            'half_ppr': 'half-point-ppr-overall.php',
            'standard': 'overall.php'
        }
        
        if scoring not in scoring_map:
            raise ValueError(f"Unsupported scoring format: {scoring}")
        
        base_url = f"{self.base_url}/{scoring_map[scoring]}"
        
        # Add year parameter for historical data
        if year and year != 2025:  # 2025 is current year
            base_url += f"?year={year}"
        
        return base_url
    
    def scrape_adp_page(self, url: str) -> pd.DataFrame:
        """Scrape ADP data from a FantasyPros page"""
        print(f"  Scraping: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the ADP table
            table = soup.find('table', {'id': 'data'})
            if not table:
                print("  ❌ Could not find ADP table")
                return pd.DataFrame()
            
            # Extract table data
            rows = table.find('tbody').find_all('tr')
            data = []
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 8:  # Need at least 8 columns for full data
                    try:
                        rank = int(cells[0].get_text(strip=True))
                        player_info = cells[1].get_text(strip=True)
                        position = cells[2].get_text(strip=True)
                        
                        # Extract ADP values from all sources
                        espn_adp = float(cells[3].get_text(strip=True)) if cells[3].get_text(strip=True) and cells[3].get_text(strip=True) != '' else np.nan
                        sleeper_adp = float(cells[4].get_text(strip=True)) if cells[4].get_text(strip=True) and cells[4].get_text(strip=True) != '' else np.nan
                        cbs_adp = float(cells[5].get_text(strip=True)) if cells[5].get_text(strip=True) and cells[5].get_text(strip=True) != '' else np.nan
                        nfl_adp = float(cells[6].get_text(strip=True)) if cells[6].get_text(strip=True) and cells[6].get_text(strip=True) != '' else np.nan
                        rtsports_adp = float(cells[7].get_text(strip=True)) if cells[7].get_text(strip=True) and cells[7].get_text(strip=True) != '' else np.nan
                        fantrax_adp = float(cells[8].get_text(strip=True)) if cells[8].get_text(strip=True) and cells[8].get_text(strip=True) != '' else np.nan
                        avg_adp = float(cells[9].get_text(strip=True)) if cells[9].get_text(strip=True) and cells[9].get_text(strip=True) != '' else np.nan
                        
                        # Extract player name and team from "PlayerNameTEAM(bye)" format
                        player_name = player_info
                        team = ''
                        bye_week = ''
                        
                        if '(' in player_info:
                            # Extract bye week
                            bye_part = player_info.split('(')[1].split(')')[0]
                            if bye_part.isdigit():
                                bye_week = bye_part
                            player_name = player_info.split('(')[0]
                        
                        # Extract team from player name (look for 2-3 letter team abbreviation at the end)
                        team_match = re.search(r'([A-Z]{2,3})$', player_name)
                        if team_match:
                            team = team_match.group(1)
                            player_name = re.sub(r'([A-Z]{2,3})$', '', player_name)
                        
                        # Clean position (remove rank numbers like WR1 -> WR)
                        clean_position = re.sub(r'(\d+)$', '', position).upper()
                        
                        # Use average ADP as the primary ADP value
                        adp_overall = avg_adp if not pd.isna(avg_adp) else np.nan
                        adp_position = np.nan  # We'll calculate this based on position
                        
                        # Include all players, even if they don't have ADP data from all sources
                        data.append({
                            'rank': rank,
                            'player_name': player_name,
                            'position': clean_position,
                            'team': team,
                            'bye_week': bye_week,
                            'adp_overall': adp_overall,
                            'adp_position': adp_position,
                            'espn_adp': espn_adp,
                            'sleeper_adp': sleeper_adp,
                            'cbs_adp': cbs_adp,
                            'nfl_adp': nfl_adp,
                            'rtsports_adp': rtsports_adp,
                            'fantrax_adp': fantrax_adp,
                            'avg_adp': avg_adp
                        })
                    except (ValueError, IndexError) as e:
                        print(f"  Warning: Could not parse row: {e}")
                        continue
            
            df = pd.DataFrame(data)
            print(f"  ✅ Scraped {len(df)} ADP records")
            return df
            
        except requests.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  ❌ Scraping failed: {e}")
            return pd.DataFrame()
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize player data"""
        if df.empty:
            return df
        
        # Clean player names (remove suffixes like "Jr.", "Sr.", etc.)
        df['player_name'] = df['player_name'].str.replace(r'\s+(Jr\.|Sr\.|III|IV|V)$', '', regex=True)
        
        # Clean position names (remove rank numbers like WR1 -> WR, RB2 -> RB)
        df['position'] = df['position'].str.replace(r'(\d+)$', '', regex=True).str.upper()
        
        # The 'team' column from FantasyPros is actually the bye week, not team
        # We'll need to get team info from our database or leave it empty
        df['team'] = ''  # Clear team column as it's actually bye week
        
        return df
    
    def map_to_player_ids(self, df: pd.DataFrame, engine) -> pd.DataFrame:
        """Map player names to our database player IDs"""
        if df.empty:
            return df
        
        # Get players from our database
        try:
            with engine.connect() as conn:
                players = pd.read_sql(text("SELECT player_id, name, position, team FROM players"), conn)
            
            print(f"  📊 Retrieved {len(players)} players from database")
            
            if players.empty:
                print("  ❌ No players found in database")
                return df
        except Exception as e:
            print(f"  ❌ Error querying players: {e}")
            return df
        
        # Create mapping dictionaries for faster lookup
        name_pos_map = {}
        name_map = {}
        
        for _, player in players.iterrows():
            key1 = f"{player['name'].lower()}_{player['position'].lower()}"
            key2 = player['name'].lower()
            name_pos_map[key1] = player['player_id']
            name_map[key2] = player['player_id']
        
        # Map players using both strategies
        mapped_players = []
        
        for _, row in df.iterrows():
            player_id = None
            
            # Strategy 1: name + position
            key1 = f"{row['player_name'].lower()}_{row['position'].lower()}"
            if key1 in name_pos_map:
                player_id = name_pos_map[key1]
            
            # Strategy 2: name only (if not found by strategy 1)
            if player_id is None:
                key2 = row['player_name'].lower()
                if key2 in name_map:
                    player_id = name_map[key2]
            
            if player_id is not None:
                mapped_row = {
                    'player_id': player_id,
                    'rank': row['rank'],
                    'player_name': row['player_name'],
                    'position': row['position'],
                    'team': row['team'],
                    'bye_week': row['bye_week'],
                    'adp_overall': row['adp_overall'],
                    'adp_position': row['adp_position'],
                    'espn_adp': row['espn_adp'],
                    'sleeper_adp': row['sleeper_adp'],
                    'cbs_adp': row['cbs_adp'],
                    'nfl_adp': row['nfl_adp'],
                    'rtsports_adp': row['rtsports_adp'],
                    'fantrax_adp': row['fantrax_adp'],
                    'avg_adp': row['avg_adp']
                }
                mapped_players.append(mapped_row)
        
        result = pd.DataFrame(mapped_players)
        
        # Check mapping success
        mapped_count = len(result)
        unmapped_count = len(df) - mapped_count
        
        print(f"  ✅ Mapped {mapped_count} players to database IDs")
        if unmapped_count > 0:
            print(f"  ⚠️  {unmapped_count} players could not be mapped")
            # Show some unmapped players for debugging
            mapped_names = set()
            for p in mapped_players:
                mapped_names.add(p['player_id'])
            
            for _, row in df.iterrows():
                key1 = f"{row['player_name'].lower()}_{row['position'].lower()}"
                key2 = row['player_name'].lower()
                if key1 not in name_pos_map and key2 not in name_map:
                    print(f"    Unmapped: {row['player_name']} ({row['position']})")
                    break
        
        return result

def build_fantasypros_adp(args: Args):
    """Build FantasyPros ADP data"""
    print(f"🏈 Building FantasyPros ADP data for {args.season} ({args.scoring})")
    
    # DB setup
    engine = create_engine(args.db_url, future=True)
    Base.metadata.create_all(engine)
    
    # Initialize scraper
    scraper = FantasyProsADPScraper()
    
    # Get URL for scoring format
    try:
        url = scraper.get_scoring_url(args.scoring, args.season)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Scrape ADP data
    adp_df = scraper.scrape_adp_page(url)
    
    if adp_df.empty:
        print("❌ No ADP data scraped")
        return
    
    # Clean data
    adp_df = scraper.clean_player_data(adp_df)
    
    # Add metadata
    from datetime import datetime
    adp_df['season'] = args.season
    adp_df['scoring_format'] = args.scoring
    adp_df['scraped_at'] = datetime.now().isoformat()
    adp_df['source_url'] = url
    
    # Remove any existing data for this season/scoring format
    if not args.test:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM adp_data 
                WHERE season = :season AND scoring_format = :scoring_format
            """), {
                'season': args.season,
                'scoring_format': args.scoring
            })
            conn.commit()
        print(f"🗑️  Cleared existing {args.scoring} ADP data for {args.season}")
    
    # Insert all ADP data into the dedicated table
    if not args.test:
        adp_df.to_sql(ADPData.__tablename__, engine, if_exists="append", index=False)
        print(f"✅ Inserted {len(adp_df)} ADP records")
    else:
        print(f"🧪 Test mode: Would insert {len(adp_df)} ADP records")
    
    # Show sample data
    print("\n📊 Sample ADP Data:")
    sample = adp_df.head(10)
    for i, row in enumerate(sample.iterrows(), 1):
        adp_val = row[1]['adp_overall']
        pos_val = row[1]['adp_position']
        adp_str = f"{adp_val:.1f}" if pd.notna(adp_val) else "N/A"
        pos_str = f"{pos_val:.0f}" if pd.notna(pos_val) else "N/A"
        print(f"  {i:3d}. {row[1]['player_name']:<20} ({row[1]['position']}) - ADP: {adp_str:>5} (Pos: {pos_str:>3})")
    
    engine.dispose()

def test_scraping():
    """Test the scraping functionality"""
    print("🧪 Testing FantasyPros ADP scraping...")
    
    scraper = FantasyProsADPScraper()
    
    # Test all scoring formats
    for scoring in ['ppr', 'half_ppr', 'standard']:
        print(f"\nTesting {scoring.upper()} scoring...")
        url = scraper.get_scoring_url(scoring)
        df = scraper.scrape_adp_page(url)
        
        if not df.empty:
            print(f"  ✅ Success: {len(df)} records")
            print(f"  Sample: {df.head(3)[['player_name', 'position', 'team', 'adp_overall']].to_string()}")
        else:
            print(f"  ❌ Failed to scrape data")
        
        # Be respectful with rate limiting
        time.sleep(2)

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Build FantasyPros ADP data for fantasy DB.")
    p.add_argument("--season", type=int, default=2024, help="Season (YYYY)")
    p.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--scoring", type=str, default="ppr", choices=["ppr", "half_ppr", "standard"], 
                   help="Scoring format")
    p.add_argument("--test", action="store_true", help="Test mode (don't write to database)")
    p.add_argument("--test-scraping", action="store_true", help="Test scraping functionality")
    
    a = p.parse_args()
    
    if a.test_scraping:
        test_scraping()
        sys.exit(0)
    
    return Args(
        season=a.season,
        db_url=a.db_url,
        scoring=a.scoring,
        test=a.test
    )

if __name__ == "__main__":
    build_fantasypros_adp(parse_args())
