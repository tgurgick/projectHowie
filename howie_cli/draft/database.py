"""
Database integration for draft simulation
Connects to ProjectHowie's existing fantasy football database
"""

import sqlite3
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from .models import Player


class DraftDatabaseConnector:
    """Connect to ProjectHowie database for draft analysis"""
    
    def __init__(self):
        self.db_path = self._get_database_path()
        
    def _get_database_path(self) -> str:
        """Use ProjectHowie's path resolution"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up from howie_cli/draft/ to project root
        project_root = os.path.dirname(os.path.dirname(script_dir))
        db_path = os.path.join(project_root, "data", "fantasy_ppr.db")
        
        if os.path.exists(db_path):
            return db_path
            
        # Fallback logic
        fallback_path = "data/fantasy_ppr.db"
        if os.path.exists(fallback_path):
            return fallback_path
            
        raise FileNotFoundError(f"Fantasy database not found at {db_path} or {fallback_path}")
    
    def load_player_universe(self, season: int = 2025) -> List[Player]:
        """Load all available players with projections + ADP + intelligence data"""
        conn = sqlite3.connect(self.db_path)
        
        # Check if intelligence tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_draft_intelligence'")
        has_intelligence = cursor.fetchone() is not None
        
        if has_intelligence:
            query = """
            SELECT 
                pp.player_name,
                pp.position,
                pp.team_name,
                pp.fantasy_points,
                pp.bye_week,
                COALESCE(ad.adp_overall, 999) as adp_overall,
                COALESCE(ad.adp_position, 99) as adp_position,
                sos.season_sos as sos_rank,
                sos.playoffs_sos as sos_playoff,
                pdi.is_projected_starter,
                pdi.starter_confidence,
                pdi.injury_risk_level,
                pdi.injury_details
            FROM player_projections pp
            LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) 
                AND ad.season = pp.season
            LEFT JOIN strength_of_schedule sos ON pp.team_name = sos.team 
                AND pp.position = sos.position AND sos.season = pp.season
            LEFT JOIN player_draft_intelligence pdi ON LOWER(pp.player_name) = LOWER(pdi.player_name)
                AND pp.team_name = pdi.team AND pdi.season = pp.season
            WHERE pp.season = ? AND pp.projection_type = 'preseason'
            AND pp.position IN ('qb', 'rb', 'wr', 'te', 'k', 'dst')
            ORDER BY pp.fantasy_points DESC
            """
        else:
            # Fallback query without intelligence data
            query = """
            SELECT 
                pp.player_name,
                pp.position,
                pp.team_name,
                pp.fantasy_points,
                pp.bye_week,
                COALESCE(ad.adp_overall, 999) as adp_overall,
                COALESCE(ad.adp_position, 99) as adp_position,
                sos.season_sos as sos_rank,
                sos.playoffs_sos as sos_playoff,
                NULL as is_projected_starter,
                NULL as starter_confidence,
                NULL as injury_risk_level,
                NULL as injury_details
            FROM player_projections pp
            LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) 
                AND ad.season = pp.season
            LEFT JOIN strength_of_schedule sos ON pp.team_name = sos.team 
                AND pp.position = sos.position AND sos.season = pp.season
            WHERE pp.season = ? AND pp.projection_type = 'preseason'
            AND pp.position IN ('qb', 'rb', 'wr', 'te', 'k', 'dst')
            ORDER BY pp.fantasy_points DESC
            """
        
        cursor = conn.cursor()
        cursor.execute(query, [season])
        
        players = []
        for row in cursor.fetchall():
            player = Player(
                name=row[0],
                position=row[1], 
                team=row[2],
                projection=row[3],
                bye_week=row[4] or 0,
                adp=row[5],
                adp_position=row[6],
                sos_rank=row[7],
                sos_playoff=row[8],
                is_projected_starter=row[9],
                starter_confidence=row[10],
                injury_risk_level=row[11],
                injury_details=row[12]
            )
            players.append(player)
        
        conn.close()
        return players
    
    def create_intelligence_tables(self):
        """Create enhanced tables for SoS, starter status, and injury data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Player Draft Intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_draft_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Starter Status (New Factor #2)
                is_projected_starter BOOLEAN DEFAULT NULL,
                starter_confidence REAL DEFAULT NULL,  -- 0.0 to 1.0
                depth_chart_position INTEGER DEFAULT NULL,
                
                -- Injury Risk (New Factor #3)
                injury_risk_level TEXT DEFAULT NULL,  -- 'LOW', 'MEDIUM', 'HIGH'
                injury_details TEXT DEFAULT NULL,
                current_injury_status TEXT DEFAULT NULL,
                
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT NULL,
                
                UNIQUE(player_name, team, season)
            )
        """)
        
        # Enhanced Strength of Schedule (New Factor #1)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_strength_of_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                position TEXT NOT NULL, 
                season INTEGER NOT NULL DEFAULT 2025,
                
                season_rank INTEGER DEFAULT NULL,  -- 1=easiest, 32=hardest
                playoff_rank INTEGER DEFAULT NULL,
                avg_points_allowed REAL DEFAULT NULL,
                strength_rating REAL DEFAULT NULL,  -- 0-1 normalized score
                
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team, position, season)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database for debugging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        info = {
            "db_path": self.db_path,
            "tables": [],
            "player_count": 0,
            "sample_players": []
        }
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        info["tables"] = [row[0] for row in cursor.fetchall()]
        
        # Get player count
        if "player_projections" in info["tables"]:
            cursor.execute("SELECT COUNT(*) FROM player_projections WHERE season = 2025")
            info["player_count"] = cursor.fetchone()[0]
            
            # Get sample players
            cursor.execute("""
                SELECT player_name, position, team_name, fantasy_points 
                FROM player_projections 
                WHERE season = 2025 AND projection_type = 'preseason'
                ORDER BY fantasy_points DESC 
                LIMIT 5
            """)
            info["sample_players"] = cursor.fetchall()
        
        conn.close()
        return info
