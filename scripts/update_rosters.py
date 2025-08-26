#!/usr/bin/env python3
"""
NFL Roster Update Script
Scrapes current NFL roster information and updates the database
"""

import asyncio
import aiohttp
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLRosterScraper:
    """Scraper for NFL roster information"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Team abbreviations mapping
        self.team_abbrev = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GBP',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAC',
            'Kansas City Chiefs': 'KCC', 'Las Vegas Raiders': 'LVR', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NEP', 'New Orleans Saints': 'NOS', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SFO', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TBB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        
        # Position mapping
        self.position_map = {
            'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'K': 'K',
            'QB/WR': 'QB', 'RB/WR': 'RB', 'WR/RB': 'WR'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_espn_rosters(self) -> Dict[str, List[Dict]]:
        """Scrape roster information from ESPN"""
        rosters = {}
        
        # ESPN team roster URLs
        espn_team_urls = {
            'ARI': 'https://www.espn.com/nfl/team/roster/_/name/ari',
            'ATL': 'https://www.espn.com/nfl/team/roster/_/name/atl',
            'BAL': 'https://www.espn.com/nfl/team/roster/_/name/bal',
            'BUF': 'https://www.espn.com/nfl/team/roster/_/name/buf',
            'CAR': 'https://www.espn.com/nfl/team/roster/_/name/car',
            'CHI': 'https://www.espn.com/nfl/team/roster/_/name/chi',
            'CIN': 'https://www.espn.com/nfl/team/roster/_/name/cin',
            'CLE': 'https://www.espn.com/nfl/team/roster/_/name/cle',
            'DAL': 'https://www.espn.com/nfl/team/roster/_/name/dal',
            'DEN': 'https://www.espn.com/nfl/team/roster/_/name/den',
            'DET': 'https://www.espn.com/nfl/team/roster/_/name/det',
            'GBP': 'https://www.espn.com/nfl/team/roster/_/name/gb',
            'HOU': 'https://www.espn.com/nfl/team/roster/_/name/hou',
            'IND': 'https://www.espn.com/nfl/team/roster/_/name/ind',
            'JAC': 'https://www.espn.com/nfl/team/roster/_/name/jax',
            'KCC': 'https://www.espn.com/nfl/team/roster/_/name/kc',
            'LVR': 'https://www.espn.com/nfl/team/roster/_/name/lv',
            'LAC': 'https://www.espn.com/nfl/team/roster/_/name/lac',
            'LAR': 'https://www.espn.com/nfl/team/roster/_/name/lar',
            'MIA': 'https://www.espn.com/nfl/team/roster/_/name/mia',
            'MIN': 'https://www.espn.com/nfl/team/roster/_/name/min',
            'NEP': 'https://www.espn.com/nfl/team/roster/_/name/ne',
            'NOS': 'https://www.espn.com/nfl/team/roster/_/name/no',
            'NYG': 'https://www.espn.com/nfl/team/roster/_/name/nyg',
            'NYJ': 'https://www.espn.com/nfl/team/roster/_/name/nyj',
            'PHI': 'https://www.espn.com/nfl/team/roster/_/name/phi',
            'PIT': 'https://www.espn.com/nfl/team/roster/_/name/pit',
            'SFO': 'https://www.espn.com/nfl/team/roster/_/name/sf',
            'SEA': 'https://www.espn.com/nfl/team/roster/_/name/sea',
            'TBB': 'https://www.espn.com/nfl/team/roster/_/name/tb',
            'TEN': 'https://www.espn.com/nfl/team/roster/_/name/ten',
            'WAS': 'https://www.espn.com/nfl/team/roster/_/name/wsh'
        }
        
        for team_abbrev, url in espn_team_urls.items():
            try:
                logger.info(f"Scraping {team_abbrev} roster from ESPN...")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find the roster table
                        roster_table = soup.find('table', {'class': 'Table'})
                        if roster_table:
                            players = self._parse_espn_roster(roster_table, team_abbrev)
                            rosters[team_abbrev] = players
                            logger.info(f"Found {len(players)} players for {team_abbrev}")
                        else:
                            logger.warning(f"No roster table found for {team_abbrev}")
                            rosters[team_abbrev] = []
                    else:
                        logger.error(f"Failed to fetch {team_abbrev}: {response.status}")
                        rosters[team_abbrev] = []
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {team_abbrev}: {e}")
                rosters[team_abbrev] = []
        
        return rosters
    
    def _parse_espn_roster(self, table, team_abbrev: str) -> List[Dict]:
        """Parse ESPN roster table"""
        players = []
        
        try:
            # Find all rows in the table
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        # Extract player information from ESPN table
                        # ESPN format: [Number, Name, POS, Age, HT, WT, Exp, College]
                        raw_name = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                        
                        # Clean the name: remove jersey numbers from ESPN data
                        clean_name = re.sub(r'\d+$', '', raw_name).strip()  # Remove trailing numbers
                        
                        player_data = {
                            'name': clean_name,  # Use cleaned name without jersey numbers
                            'position': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'team': team_abbrev,
                            'age': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                            'height': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                            'weight': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            'experience': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                            'college': cells[7].get_text(strip=True) if len(cells) > 7 else '',
                            'updated_at': datetime.now().isoformat()
                        }
                        
                        # Clean up position
                        position = player_data['position'].upper()
                        player_data['position'] = self.position_map.get(position, position)
                        
                        # Generate player ID for new players only (existing players will be found by matching)
                        player_data['player_id'] = self._generate_player_id(player_data['name'], team_abbrev)
                        
                        # Only add if we have a valid name and position
                        if player_data['name'] and player_data['position']:
                            players.append(player_data)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing player row: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing ESPN roster table: {e}")
        
        return players
    
    async def scrape_profootballreference(self) -> Dict[str, List[Dict]]:
        """Scrape roster information from Pro Football Reference"""
        rosters = {}
        
        for team_name, abbrev in self.team_abbrev.items():
            try:
                logger.info(f"Scraping {team_name} from Pro Football Reference...")
                
                # Pro Football Reference URL structure
                url = f"https://www.pro-football-reference.com/teams/{abbrev.lower()}/2025.htm"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find roster table
                        roster_table = soup.find('table', {'id': 'roster'})
                        if roster_table:
                            players = self._parse_profootballreference_roster(roster_table, abbrev)
                            rosters[abbrev] = players
                        else:
                            logger.warning(f"No roster table found for {team_name}")
                            rosters[abbrev] = []
                    else:
                        logger.error(f"Failed to fetch {team_name}: {response.status}")
                        rosters[abbrev] = []
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error scraping {team_name}: {e}")
                rosters[abbrev] = []
        
        return rosters
    
    def _parse_profootballreference_roster(self, table, team_abbrev: str) -> List[Dict]:
        """Parse Pro Football Reference roster table"""
        players = []
        
        try:
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    try:
                        # Extract player information
                        player_data = {
                            'name': cells[0].get_text(strip=True),
                            'position': cells[1].get_text(strip=True),
                            'team': team_abbrev,
                            'height': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'weight': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                            'birthdate': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                            'college': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            'experience': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                            'updated_at': datetime.now().isoformat()
                        }
                        
                        # Clean up position
                        position = player_data['position'].upper()
                        player_data['position'] = self.position_map.get(position, position)
                        
                        # Generate player ID (you might want to use a more sophisticated method)
                        player_data['player_id'] = self._generate_player_id(player_data['name'], team_abbrev)
                        
                        players.append(player_data)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing player row: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing roster table: {e}")
        
        return players
    
    def _generate_player_id(self, name: str, team: str) -> str:
        """Generate a unique player ID for new players"""
        # Clean name and create stable ID: name_team_current
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
        return f"{clean_name[:8]}_{team}_current"
    
    async def scrape_nfl_com(self) -> Dict[str, List[Dict]]:
        """Scrape roster information from NFL.com"""
        rosters = {}
        
        # NFL.com team roster URLs
        nfl_team_urls = {
            'ARI': 'https://www.nfl.com/teams/arizona-cardinals/roster',
            'ATL': 'https://www.nfl.com/teams/atlanta-falcons/roster',
            'BAL': 'https://www.nfl.com/teams/baltimore-ravens/roster',
            'BUF': 'https://www.nfl.com/teams/buffalo-bills/roster',
            'CAR': 'https://www.nfl.com/teams/carolina-panthers/roster',
            'CHI': 'https://www.nfl.com/teams/chicago-bears/roster',
            'CIN': 'https://www.nfl.com/teams/cincinnati-bengals/roster',
            'CLE': 'https://www.nfl.com/teams/cleveland-browns/roster',
            'DAL': 'https://www.nfl.com/teams/dallas-cowboys/roster',
            'DEN': 'https://www.nfl.com/teams/denver-broncos/roster',
            'DET': 'https://www.nfl.com/teams/detroit-lions/roster',
            'GBP': 'https://www.nfl.com/teams/green-bay-packers/roster',
            'HOU': 'https://www.nfl.com/teams/houston-texans/roster',
            'IND': 'https://www.nfl.com/teams/indianapolis-colts/roster',
            'JAC': 'https://www.nfl.com/teams/jacksonville-jaguars/roster',
            'KCC': 'https://www.nfl.com/teams/kansas-city-chiefs/roster',
            'LVR': 'https://www.nfl.com/teams/las-vegas-raiders/roster',
            'LAC': 'https://www.nfl.com/teams/los-angeles-chargers/roster',
            'LAR': 'https://www.nfl.com/teams/los-angeles-rams/roster',
            'MIA': 'https://www.nfl.com/teams/miami-dolphins/roster',
            'MIN': 'https://www.nfl.com/teams/minnesota-vikings/roster',
            'NEP': 'https://www.nfl.com/teams/new-england-patriots/roster',
            'NOS': 'https://www.nfl.com/teams/new-orleans-saints/roster',
            'NYG': 'https://www.nfl.com/teams/new-york-giants/roster',
            'NYJ': 'https://www.nfl.com/teams/new-york-jets/roster',
            'PHI': 'https://www.nfl.com/teams/philadelphia-eagles/roster',
            'PIT': 'https://www.nfl.com/teams/pittsburgh-steelers/roster',
            'SFO': 'https://www.nfl.com/teams/san-francisco-49ers/roster',
            'SEA': 'https://www.nfl.com/teams/seattle-seahawks/roster',
            'TBB': 'https://www.nfl.com/teams/tampa-bay-buccaneers/roster',
            'TEN': 'https://www.nfl.com/teams/tennessee-titans/roster',
            'WAS': 'https://www.nfl.com/teams/washington-commanders/roster'
        }
        
        for team_abbrev, url in nfl_team_urls.items():
            try:
                logger.info(f"Scraping {team_abbrev} roster from NFL.com...")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find the roster table (NFL.com uses different structure)
                        roster_table = soup.find('table', {'class': 'd3-o-table'})
                        if not roster_table:
                            roster_table = soup.find('table', {'class': 'roster-table'})
                        
                        if roster_table:
                            players = self._parse_nfl_com_roster(roster_table, team_abbrev)
                            rosters[team_abbrev] = players
                            logger.info(f"Found {len(players)} players for {team_abbrev}")
                        else:
                            logger.warning(f"No roster table found for {team_abbrev}")
                            rosters[team_abbrev] = []
                    else:
                        logger.error(f"Failed to fetch {team_abbrev}: {response.status}")
                        rosters[team_abbrev] = []
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {team_abbrev}: {e}")
                rosters[team_abbrev] = []
        
        return rosters
    
    def _parse_nfl_com_roster(self, table, team_abbrev: str) -> List[Dict]:
        """Parse NFL.com roster table"""
        players = []
        
        try:
            # Find all rows in the table
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    try:
                        # Extract player information from NFL.com table
                        player_data = {
                            'name': cells[0].get_text(strip=True),
                            'position': cells[1].get_text(strip=True),
                            'team': team_abbrev,
                            'height': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'weight': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                            'birthdate': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                            'college': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            'experience': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                            'updated_at': datetime.now().isoformat()
                        }
                        
                        # Clean up position
                        position = player_data['position'].upper()
                        player_data['position'] = self.position_map.get(position, position)
                        
                        # Generate player ID
                        player_data['player_id'] = self._generate_player_id(player_data['name'], team_abbrev)
                        
                        # Only add if we have a valid name and position
                        if player_data['name'] and player_data['position']:
                            players.append(player_data)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing player row: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing NFL.com roster table: {e}")
        
        return players

class DatabaseUpdater:
    """Update database with scraped roster information"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def update_rosters(self, rosters: Dict[str, List[Dict]]) -> List[Dict]:
        """Update the database with new roster information, return team changes"""
        team_changes = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create or update players table
            self._create_players_table(conn)
            
            # Update roster data
            total_players = 0
            for team_abbrev, players in rosters.items():
                logger.info(f"Updating {team_abbrev} with {len(players)} players")
                
                for player in players:
                    change_info = self._upsert_player(conn, player)
                    if change_info['action'] == 'team_change':
                        team_changes.append(change_info)
                    total_players += 1
            
            # Clean up numbered duplicates after updating
            self._cleanup_numbered_duplicates(conn)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully updated {total_players} players across {len(rosters)} teams")
            return team_changes
            
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            raise
    
    def _create_players_table(self, conn):
        """Create or update the players table structure"""
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create new table with current schema
            create_table_sql = """
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                position TEXT,
                team TEXT,
                height TEXT,
                weight TEXT,
                age TEXT,
                experience TEXT,
                college TEXT,
                updated_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(create_table_sql)
        else:
            # Check if we need to add missing columns
            cursor.execute("PRAGMA table_info(players)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'age' not in columns:
                conn.execute("ALTER TABLE players ADD COLUMN age TEXT")
            if 'experience' not in columns:
                conn.execute("ALTER TABLE players ADD COLUMN experience TEXT")
            if 'college' not in columns:
                conn.execute("ALTER TABLE players ADD COLUMN college TEXT")
            if 'updated_at' not in columns:
                conn.execute("ALTER TABLE players ADD COLUMN updated_at TEXT")
        
        # Add indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(name)")
    
    def _find_existing_player(self, conn, name: str, team: str) -> Optional[str]:
        """Find existing player by name and team, return player_id if found"""
        cursor = conn.cursor()
        
        # Clean incoming name (remove jersey numbers, periods, etc.)
        clean_incoming = re.sub(r'\d+$', '', name).strip()  # Remove trailing numbers
        clean_incoming = re.sub(r'[^a-zA-Z\s\.]', '', clean_incoming).strip()  # Keep periods for initials
        
        # Try exact name match first
        cursor.execute(
            "SELECT player_id FROM players WHERE name = ? AND team = ? LIMIT 1",
            (clean_incoming, team)
        )
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Try fuzzy name match across all players on the team
        cursor.execute(
            "SELECT player_id, name FROM players WHERE team = ?",
            (team,)
        )
        
        for row in cursor.fetchall():
            existing_id, existing_name = row
            # Clean existing name the same way
            clean_existing = re.sub(r'\d+$', '', existing_name).strip()  # Remove trailing numbers
            clean_existing = re.sub(r'[^a-zA-Z\s\.]', '', clean_existing).strip()  # Keep periods for initials
            
            # Check if names match when cleaned
            if clean_incoming.lower() == clean_existing.lower():
                return existing_id
        
        return None
    
    def _upsert_player(self, conn, player: Dict) -> Dict[str, str]:
        """Insert or update a player record, return change info"""
        # Check if player already exists
        existing_player_id = self._find_existing_player(conn, player['name'], player['team'])
        
        change_info = {
            'action': '',
            'player_name': player['name'],
            'old_team': '',
            'new_team': player['team'],
            'position': player['position']
        }
        
        if existing_player_id:
            # Get current team before updating
            cursor = conn.cursor()
            cursor.execute("SELECT team FROM players WHERE player_id = ?", (existing_player_id,))
            result = cursor.fetchone()
            old_team = result[0] if result else ''
            
            # Check for team change
            if old_team and old_team != player['team']:
                change_info['action'] = 'team_change'
                change_info['old_team'] = old_team
                logger.info(f"🔄 Team Change: {player['name']} ({player['position']}) {old_team} → {player['team']}")
            else:
                change_info['action'] = 'updated'
            
            # Update existing player (including potential team change)
            sql = """
            UPDATE players SET 
                team = ?, position = ?, height = ?, weight = ?, age = ?, 
                experience = ?, college = ?, updated_at = ?
            WHERE player_id = ?
            """
            
            conn.execute(sql, (
                player['team'],
                player['position'],
                player.get('height', ''),
                player.get('weight', ''),
                player.get('age', ''),
                player.get('experience', ''),
                player.get('college', ''),
                player.get('updated_at', datetime.now().isoformat()),
                existing_player_id
            ))
        else:
            # Insert new player
            change_info['action'] = 'new_player'
            sql = """
            INSERT INTO players 
            (player_id, name, position, team, height, weight, age, experience, college, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn.execute(sql, (
                player['player_id'],
                player['name'],
                player['position'],
                player['team'],
                player.get('height', ''),
                player.get('weight', ''),
                player.get('age', ''),
                player.get('experience', ''),
                player.get('college', ''),
                player.get('updated_at', datetime.now().isoformat())
            ))
        
        return change_info
    
    def _cleanup_numbered_duplicates(self, conn):
        """Remove numbered duplicate players (e.g., 'A.J. Brown11') that were created from old scrapes"""
        cursor = conn.cursor()
        
        # Find all players with recent updates (from our scraping)
        cursor.execute("""
            SELECT player_id, name, team, position 
            FROM players 
            WHERE updated_at IS NOT NULL
        """)
        
        all_players = cursor.fetchall()
        deleted_count = 0
        
        for player_id, name, team, position in all_players:
            # Check if this name ends with numbers
            if re.search(r'\d+$', name):
                # Clean the name
                clean_name = re.sub(r'\d+$', '', name).strip()
                
                # Check if there's a clean version of this player
                cursor.execute("""
                    SELECT player_id FROM players 
                    WHERE name = ? AND team = ? AND position = ?
                    AND player_id != ?
                """, (clean_name, team, position, player_id))
                
                clean_version = cursor.fetchone()
                
                if clean_version:
                    # Delete the numbered duplicate
                    cursor.execute("DELETE FROM players WHERE player_id = ?", (player_id,))
                    deleted_count += 1
                    logger.info(f"🗑️  Removed duplicate: {name} ({position}, {team}) → keeping {clean_name}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} numbered duplicates")

async def main():
    """Main function to update rosters"""
    logger.info("Starting NFL roster update...")
    
    # Database paths
    db_paths = [
        "data/fantasy_ppr.db",
        "data/fantasy_halfppr.db", 
        "data/fantasy_standard.db"
    ]
    
    all_team_changes = []
    
    try:
        # Scrape roster information
        async with NFLRosterScraper() as scraper:
            logger.info("Attempting to scrape roster information from ESPN...")
            rosters = await scraper.scrape_espn_rosters()
            
            if not rosters or all(len(players) == 0 for players in rosters.values()):
                logger.warning("ESPN scraping failed, trying NFL.com...")
                rosters = await scraper.scrape_nfl_com()
            
            if not rosters or all(len(players) == 0 for players in rosters.values()):
                logger.warning("NFL.com scraping failed, trying Pro Football Reference...")
                rosters = await scraper.scrape_profootballreference()
            
            if not rosters or all(len(players) == 0 for players in rosters.values()):
                logger.error("All scraping methods failed. No roster data available.")
                return []
        
        # Update all databases
        for db_path in db_paths:
            if Path(db_path).exists():
                logger.info(f"Updating {db_path}...")
                updater = DatabaseUpdater(db_path)
                team_changes = updater.update_rosters(rosters)
                
                # Only collect changes from the first database to avoid duplicates
                if db_path == db_paths[0]:
                    all_team_changes.extend(team_changes)
            else:
                logger.warning(f"Database not found: {db_path}")
        
        logger.info("Roster update completed successfully!")
        
        # Print summary
        total_players = sum(len(players) for players in rosters.values())
        logger.info(f"Total players updated: {total_players}")
        logger.info(f"Teams updated: {len(rosters)}")
        
        # Print team changes
        if all_team_changes:
            logger.info("\n🔄 Team Changes Detected:")
            for change in all_team_changes:
                logger.info(f"  • {change['player_name']} ({change['position']}) {change['old_team']} → {change['new_team']}")
        else:
            logger.info("\n✅ No team changes detected")
        
        return all_team_changes
        
    except Exception as e:
        logger.error(f"Error during roster update: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
