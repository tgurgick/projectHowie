"""
Intelligence Parser for Draft System
Extracts starter status and injury information from team intelligence data
"""

import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from .database import DraftDatabaseConnector


class IntelligenceParser:
    """Parse team intelligence data to extract starter status and injury info"""
    
    def __init__(self, db_path: str = None):
        if db_path:
            self.db_path = db_path
        else:
            self.db_connector = DraftDatabaseConnector()
            self.db_path = self.db_connector.db_path
    
    def parse_all_intelligence(self) -> Dict[str, Dict[str, Dict]]:
        """Parse all team intelligence data and extract player information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all intelligence data
        cursor.execute("""
            SELECT team, position, intelligence_summary, injury_updates, confidence_score
            FROM team_position_intelligence 
            WHERE season = 2025 AND intelligence_summary IS NOT NULL
        """)
        
        intelligence_data = cursor.fetchall()
        parsed_data = {}
        
        for team, position, summary, injury_updates, confidence in intelligence_data:
            if team not in parsed_data:
                parsed_data[team] = {}
            
            # Parse this position group
            position_data = self._parse_position_intelligence(
                team, position, summary, injury_updates, confidence
            )
            
            parsed_data[team][position] = position_data
        
        conn.close()
        return parsed_data
    
    def _parse_position_intelligence(
        self, 
        team: str, 
        position: str, 
        summary: str, 
        injury_updates: str,
        confidence: float
    ) -> Dict:
        """Parse intelligence for a specific team/position"""
        
        # Extract players and their roles
        players = self._extract_players_and_roles(summary, position)
        
        # Extract injury information
        injury_info = self._extract_injury_information(summary, injury_updates)
        
        # Combine the information
        position_data = {
            'players': players,
            'injuries': injury_info,
            'confidence': confidence or 0.7,
            'raw_summary': summary[:200] + '...' if len(summary) > 200 else summary
        }
        
        return position_data
    
    def _extract_players_and_roles(self, summary: str, position: str) -> List[Dict]:
        """Extract player names and their starter status from summary"""
        players = []
        
        if not summary:
            return players
        
        # Common patterns for identifying starters
        starter_patterns = [
            # Direct statements
            r'(?:starter|starting):\s*\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?',
            r'\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?\s*as\s+the\s+(?:clear\s+)?starter',
            r'led\s+by\s+\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?\s+as\s+the\s+(?:clear\s+)?(?:starter|WR1|RB1|QB1|TE1)',
            r'\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?\s+is\s+the\s+(?:clear\s+)?(?:starter|WR1|RB1|QB1|TE1)',
            r'the\s+(?:clear\s+)?(?:starter|starting\s+\w+)(?:\s+is)?\s+\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?',
            
            # Position-specific patterns
            r'(?:WR1|RB1|QB1|TE1):\s*\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?',
            r'(?:lead\s+back|primary\s+back|starting\s+\w+)\s+(?:is\s+)?\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?',
            
            # Backup patterns (for context)
            r'backup[s]?(?:\s+is|\s+are)?\s*:?\s*\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?',
            r'\*?\*?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\*?\*?\s+(?:serves?\s+as|serving\s+as)\s+(?:the\s+)?(?:primary\s+)?backup'
        ]
        
        # Find all players mentioned with role context
        found_players = set()
        
        for pattern in starter_patterns:
            matches = re.finditer(pattern, summary, re.IGNORECASE)
            for match in matches:
                player_name = match.group(1).strip()
                
                # Clean up the name
                player_name = re.sub(r'\*+', '', player_name)  # Remove markdown
                player_name = ' '.join(player_name.split())  # Normalize whitespace
                
                if len(player_name.split()) >= 2:  # Must be first + last name
                    found_players.add(player_name)
        
        # Determine roles based on context
        for player_name in found_players:
            is_starter = self._determine_starter_status(player_name, summary)
            confidence = self._calculate_role_confidence(player_name, summary)
            
            player_info = {
                'name': player_name,
                'is_projected_starter': is_starter,
                'starter_confidence': confidence,
                'position': position,
                'context': self._extract_player_context(player_name, summary)
            }
            
            players.append(player_info)
        
        return players
    
    def _determine_starter_status(self, player_name: str, summary: str) -> bool:
        """Determine if a player is a starter based on context"""
        player_context = summary.lower()
        name_lower = player_name.lower()
        
        # Strong starter indicators
        starter_indicators = [
            f'{name_lower} as the clear starter',
            f'{name_lower} as the starter',
            f'starter: {name_lower}',
            f'{name_lower} is the clear starter',
            f'{name_lower} is the starter',
            f'led by {name_lower}',
            f'{name_lower} as the wr1',
            f'{name_lower} as the rb1',
            f'{name_lower} as the qb1',
            f'{name_lower} as the te1',
            f'{name_lower} is the wr1',
            f'{name_lower} is the rb1',
            f'{name_lower} is the qb1',
            f'{name_lower} is the te1',
            f'lead back {name_lower}',
            f'primary back {name_lower}',
            f'starting {name_lower}'
        ]
        
        # Backup indicators
        backup_indicators = [
            f'backup: {name_lower}',
            f'{name_lower} as backup',
            f'{name_lower} serves as backup',
            f'{name_lower} serving as backup',
            f'backed up by {name_lower}',
            f'{name_lower} as the backup'
        ]
        
        # Check for starter patterns
        for indicator in starter_indicators:
            if indicator in player_context:
                return True
        
        # Check for backup patterns
        for indicator in backup_indicators:
            if indicator in player_context:
                return False
        
        # Default to True if mentioned prominently (likely starter)
        return True
    
    def _calculate_role_confidence(self, player_name: str, summary: str) -> float:
        """Calculate confidence in the player's role"""
        name_lower = player_name.lower()
        context = summary.lower()
        
        confidence = 0.6  # Base confidence
        
        # Boost confidence for strong language
        if 'clear starter' in context and name_lower in context:
            confidence += 0.3
        elif 'starter' in context and name_lower in context:
            confidence += 0.2
        
        # Boost for position-specific language (WR1, RB1, etc.)
        if any(pos in context for pos in ['wr1', 'rb1', 'qb1', 'te1']) and name_lower in context:
            confidence += 0.2
        
        # Reduce confidence for uncertainty language
        if any(word in context for word in ['competition', 'battle', 'unclear', 'uncertain']):
            confidence -= 0.1
        
        # Boost for depth chart mentions
        if 'depth chart' in context and name_lower in context:
            confidence += 0.1
        
        return min(1.0, max(0.1, confidence))
    
    def _extract_player_context(self, player_name: str, summary: str) -> str:
        """Extract relevant context around a player mention"""
        sentences = re.split(r'[.!?]+', summary)
        
        for sentence in sentences:
            if player_name.lower() in sentence.lower():
                return sentence.strip()
        
        return ""
    
    def _extract_injury_information(self, summary: str, injury_updates: str) -> Dict[str, Dict]:
        """Extract injury information for players"""
        injury_info = {}
        
        # Combine summary and injury_updates for comprehensive parsing
        full_text = f"{summary}\n\n{injury_updates or ''}"
        
        # Patterns for injury information
        injury_patterns = [
            # Specific injury mentions
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\s+(?:is\s+)?(?:currently\s+)?(?:dealing\s+with|has|suffering\s+from|injured\s+with)\s+(?:a\s+)?([^.]+)',
            
            # Health status
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+Jr\.?|\s+Sr\.?|\s+III?)?)\s+(?:is\s+)?(?:healthy|questionable|doubtful|out|injured)',
            
            # No injury reports
            r'(?:no\s+(?:current\s+)?(?:major\s+)?injury\s+(?:concerns?|reports?))|(?:all\s+healthy)|(?:fully\s+healthy)',
        ]
        
        # Extract specific player injuries
        for pattern in injury_patterns[:2]:  # Skip the "no injury" pattern for individual players
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                player_name = match.group(1).strip()
                
                # Clean up the name
                player_name = re.sub(r'\*+', '', player_name)
                player_name = ' '.join(player_name.split())
                
                if len(player_name.split()) >= 2:
                    injury_details = match.group(2) if len(match.groups()) > 1 else "injury concern"
                    
                    # Determine injury risk level
                    risk_level = self._classify_injury_risk(injury_details, full_text)
                    
                    injury_info[player_name] = {
                        'injury_risk_level': risk_level,
                        'injury_details': injury_details.strip(),
                        'context': self._extract_injury_context(player_name, full_text)
                    }
        
        # Check for "no injuries" statements
        no_injury_patterns = [
            r'no\s+(?:current\s+)?(?:major\s+)?injury\s+(?:concerns?|reports?)',
            r'all\s+(?:players?\s+)?(?:are\s+)?healthy',
            r'no\s+injury\s+(?:concerns?|updates?)',
            r'fully\s+healthy'
        ]
        
        has_no_injury_statement = any(
            re.search(pattern, full_text, re.IGNORECASE) 
            for pattern in no_injury_patterns
        )
        
        # If no specific injuries found but has "no injury" statement, 
        # this indicates low risk for all players
        if not injury_info and has_no_injury_statement:
            injury_info['_group_status'] = {
                'injury_risk_level': 'LOW',
                'injury_details': 'No current injury concerns reported',
                'context': 'Group health status'
            }
        
        return injury_info
    
    def _classify_injury_risk(self, injury_details: str, full_context: str) -> str:
        """Classify injury risk level based on details"""
        details_lower = injury_details.lower()
        context_lower = full_context.lower()
        
        # High risk indicators
        high_risk = ['out', 'surgery', 'torn', 'fracture', 'ir', 'injured reserve', 'doubtful']
        if any(indicator in details_lower for indicator in high_risk):
            return 'HIGH'
        
        # Medium risk indicators  
        medium_risk = ['questionable', 'limited', 'managing', 'minor', 'day-to-day', 'monitoring']
        if any(indicator in details_lower for indicator in medium_risk):
            return 'MEDIUM'
        
        # Low risk indicators
        low_risk = ['healthy', 'no concern', 'cleared', 'fully participating', 'no issues']
        if any(indicator in details_lower for indicator in low_risk):
            return 'LOW'
        
        # Default to medium if injury mentioned but unclear severity
        return 'MEDIUM'
    
    def _extract_injury_context(self, player_name: str, full_text: str) -> str:
        """Extract injury context for a specific player"""
        sentences = re.split(r'[.!?]+', full_text)
        
        for sentence in sentences:
            if player_name.lower() in sentence.lower() and any(
                keyword in sentence.lower() 
                for keyword in ['injury', 'health', 'concern', 'hurt', 'pain', 'recovery']
            ):
                return sentence.strip()
        
        return ""
    
    def update_draft_intelligence_table(self):
        """Update the player_draft_intelligence table with parsed data"""
        # First, create the table if it doesn't exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_draft_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Starter Status
                is_projected_starter BOOLEAN DEFAULT NULL,
                starter_confidence REAL DEFAULT NULL,
                depth_chart_position INTEGER DEFAULT NULL,
                
                -- Injury Risk Assessment  
                injury_risk_level TEXT DEFAULT NULL,
                injury_details TEXT DEFAULT NULL,
                current_injury_status TEXT DEFAULT NULL,
                
                -- Intelligence Metadata
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT NULL,
                source_summary TEXT DEFAULT NULL,
                
                UNIQUE(player_name, team, season)
            )
        """)
        
        # Parse all intelligence
        parsed_data = self.parse_all_intelligence()
        
        # Update the table
        updates_made = 0
        
        for team, positions in parsed_data.items():
            for position, pos_data in positions.items():
                # Update each player
                for player in pos_data['players']:
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_draft_intelligence (
                            player_name, team, position, season,
                            is_projected_starter, starter_confidence,
                            injury_risk_level, injury_details,
                            confidence_score, source_summary,
                            last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        player['name'],
                        team,
                        position,
                        2025,
                        player['is_projected_starter'],
                        player['starter_confidence'],
                        pos_data['injuries'].get(player['name'], {}).get('injury_risk_level'),
                        pos_data['injuries'].get(player['name'], {}).get('injury_details'),
                        pos_data['confidence'],
                        pos_data['raw_summary']
                    ))
                    updates_made += 1
                
                # Handle group injury status if no specific player injuries
                if '_group_status' in pos_data['injuries']:
                    group_status = pos_data['injuries']['_group_status']
                    # Apply low risk to all players in this position group
                    for player in pos_data['players']:
                        cursor.execute("""
                            UPDATE player_draft_intelligence 
                            SET injury_risk_level = ?, injury_details = ?
                            WHERE player_name = ? AND team = ? AND season = 2025
                        """, (
                            group_status['injury_risk_level'],
                            group_status['injury_details'],
                            player['name'],
                            team
                        ))
        
        conn.commit()
        conn.close()
        
        return updates_made
    
    def get_player_intelligence(self, player_name: str, team: str = None) -> Optional[Dict]:
        """Get intelligence data for a specific player"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if team:
            cursor.execute("""
                SELECT is_projected_starter, starter_confidence, 
                       injury_risk_level, injury_details, confidence_score
                FROM player_draft_intelligence 
                WHERE LOWER(player_name) = LOWER(?) AND team = ? AND season = 2025
            """, (player_name, team))
        else:
            cursor.execute("""
                SELECT is_projected_starter, starter_confidence, 
                       injury_risk_level, injury_details, confidence_score
                FROM player_draft_intelligence 
                WHERE LOWER(player_name) = LOWER(?) AND season = 2025
            """, (player_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'is_projected_starter': result[0],
                'starter_confidence': result[1],
                'injury_risk_level': result[2],
                'injury_details': result[3],
                'confidence_score': result[4]
            }
        
        return None
