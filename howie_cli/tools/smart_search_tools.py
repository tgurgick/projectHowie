"""
Smart Search Tools for Context-Aware Database Queries
Replaces problematic DatabaseQueryTool with intelligent, schema-aware searches
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import sqlite3
import os
import re
from pathlib import Path
from datetime import datetime

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter


class SmartPlayerSearchTool(BaseTool):
    """Intelligent player search that understands context and uses correct schema"""
    
    def __init__(self):
        super().__init__()
        self.name = "smart_player_search"
        self.category = "database"
        self.description = "Context-aware player search using 2025 projections and current data"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string", 
                description="Search query (e.g., 'reliable WRs', 'injury concerns', 'top 10 QBs')",
                required=True
            ),
            ToolParameter(
                name="position",
                type="string",
                description="Position filter (optional)",
                required=False,
                choices=["qb", "rb", "wr", "te", "k", "def"]
            ),
            ToolParameter(
                name="season",
                type="int",
                description="Season year",
                required=False,
                default=2025
            )
        ]
    
    async def execute(self, query: str, position: Optional[str] = None, season: int = 2025, **kwargs) -> ToolResult:
        """Execute intelligent player search"""
        try:
            # Get database path
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(script_dir, "data", "fantasy_ppr.db")
            
            if not os.path.exists(db_path):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Database not found: {db_path}"
                )
            
            # Analyze query intent
            query_intent = self._analyze_query_intent(query)
            
            # Build appropriate SQL based on intent and available data
            sql_query, params = self._build_smart_query(query_intent, position, season)
            
            # Execute query
            conn = sqlite3.connect(db_path)
            
            # SAFE DataFrame handling - convert to dict immediately
            try:
                df = pd.read_sql_query(sql_query, conn, params=params)
                conn.close()
                
                # Convert DataFrame to safe dict format immediately
                if df.empty:
                    result_data = {
                        "players": [],
                        "total_found": 0,
                        "message": "No players found matching criteria"
                    }
                else:
                    players = []
                    for _, row in df.iterrows():
                        player_dict = {}
                        for col in df.columns:
                            value = row[col]
                            # Handle None/NaN values
                            if pd.isna(value):
                                player_dict[col] = None
                            else:
                                player_dict[col] = value
                        players.append(player_dict)
                    
                    result_data = {
                        "players": players,
                        "total_found": len(players),
                        "query_intent": query_intent,
                        "sql_used": sql_query
                    }
            
            except Exception as sql_error:
                conn.close()
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"SQL execution failed: {str(sql_error)}"
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Smart search failed: {str(e)}"
            )
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand user intent"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "focus": [],
            "filters": {},
            "sort_by": "fantasy_points",
            "sort_order": "DESC"
        }
        
        # Identify query type
        if any(word in query_lower for word in ["injury", "injured", "hurt", "health"]):
            intent["type"] = "injury_concern"
            intent["focus"].append("injury_status")
        
        elif any(word in query_lower for word in ["reliable", "consistent", "stable"]):
            intent["type"] = "reliability"
            intent["focus"].append("consistency")
        
        elif any(word in query_lower for word in ["top", "best", "rank"]):
            intent["type"] = "rankings"
            intent["focus"].append("performance")
        
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            intent["type"] = "comparison"
            intent["focus"].append("comparison")
        
        elif any(word in query_lower for word in ["sleeper", "deep", "late round"]):
            intent["type"] = "sleepers"
            intent["focus"].append("value")
            intent["sort_by"] = "adp_overall"
            intent["sort_order"] = "DESC"
        
        # Extract number if present (e.g., "top 10")
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            intent["limit"] = int(numbers[0])
        else:
            intent["limit"] = 10
        
        return intent
    
    def _build_smart_query(self, intent: Dict[str, Any], position: Optional[str], season: int) -> tuple:
        """Build SQL query based on intent and available schema"""
        
        if season >= 2025:
            # Use projections table for 2025+
            base_query = """
            SELECT 
                pp.player_name as name,
                pp.position,
                pp.team_name as team,
                pp.fantasy_points,
                pp.games,
                pp.bye_week,
                ad.adp_overall,
                ad.adp_position,
                tp.intelligence_summary
            FROM player_projections pp
            LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) AND ad.season = pp.season
            LEFT JOIN team_position_intelligence tp ON pp.team_name = tp.team AND pp.position = tp.position AND tp.season = pp.season
            WHERE pp.season = ? AND pp.projection_type = 'preseason'
            """
            params = [season]
            
        else:
            # Use historical data for older seasons (if available)
            base_query = """
            SELECT 
                p.name,
                p.position,
                p.team,
                AVG(pgs.fantasy_points) as fantasy_points,
                COUNT(*) as games
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.season = ?
            GROUP BY p.player_id, p.name, p.position, p.team
            """
            params = [season]
        
        # Add position filter
        if position:
            base_query += f" AND LOWER(pp.position) = LOWER(?)" if season >= 2025 else f" AND LOWER(p.position) = LOWER(?)"
            params.append(position)
        
        # Add intent-specific filters and sorting
        if intent["type"] == "reliability":
            if season >= 2025:
                base_query += " AND pp.games >= 16"  # Focus on players expected to play most games
            else:
                base_query += " HAVING games >= 12"
        
        elif intent["type"] == "sleepers":
            if season >= 2025:
                base_query += " AND (ad.adp_overall > 100 OR ad.adp_overall IS NULL)"
            # No ADP filter for historical data
        
        elif intent["type"] == "injury_concern":
            if season >= 2025:
                # Look for intelligence mentions of injuries
                base_query += " AND (tp.injury_updates IS NOT NULL OR tp.intelligence_summary LIKE '%injury%')"
        
        # Add sorting
        if intent["sort_by"] == "fantasy_points":
            sort_col = "pp.fantasy_points" if season >= 2025 else "fantasy_points"
        elif intent["sort_by"] == "adp_overall":
            sort_col = "ad.adp_overall" if season >= 2025 else "1"  # Fallback for historical
        else:
            sort_col = "pp.fantasy_points" if season >= 2025 else "fantasy_points"
        
        base_query += f" ORDER BY {sort_col} {intent['sort_order']}"
        
        # Add limit
        base_query += f" LIMIT {intent['limit']}"
        
        return base_query, params


class ContextualSearchTool(BaseTool):
    """Search tool that uses conversation context and database context injection"""
    
    def __init__(self):
        super().__init__()
        self.name = "contextual_search"
        self.category = "database"
        self.description = "Search using conversation context and database intelligence"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="Search query with context awareness",
                required=True
            )
        ]
    
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute contextual search using multiple data sources"""
        try:
            results = {}
            
            # Get database path
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(script_dir, "data", "fantasy_ppr.db")
            
            if not os.path.exists(db_path):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Database not found: {db_path}"
                )
            
            conn = sqlite3.connect(db_path)
            
            # For injury queries, search team intelligence
            if any(word in query.lower() for word in ["injury", "injured", "hurt", "health"]):
                injury_query = """
                SELECT team, position, injury_updates, confidence_score, last_updated
                FROM team_position_intelligence 
                WHERE season = 2025 
                AND (injury_updates IS NOT NULL AND injury_updates != '')
                ORDER BY confidence_score DESC
                LIMIT 10
                """
                
                df_injuries = pd.read_sql_query(injury_query, conn)
                if not df_injuries.empty:
                    results["injury_intel"] = df_injuries.to_dict('records')
            
            # For reliability queries, get projections with ADP
            if any(word in query.lower() for word in ["reliable", "consistent", "safe"]):
                reliability_query = """
                SELECT pp.player_name, pp.position, pp.team_name, pp.fantasy_points, 
                       ad.adp_overall, pp.bye_week
                FROM player_projections pp
                LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) AND ad.season = pp.season
                WHERE pp.season = 2025 AND pp.projection_type = 'preseason'
                AND pp.games >= 16
                ORDER BY pp.fantasy_points DESC
                LIMIT 15
                """
                
                df_reliable = pd.read_sql_query(reliability_query, conn)
                if not df_reliable.empty:
                    results["reliable_players"] = df_reliable.to_dict('records')
            
            conn.close()
            
            if not results:
                # Fallback to general search
                smart_search = SmartPlayerSearchTool()
                return await smart_search.execute(query)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=results
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Contextual search failed: {str(e)}"
            )


class QuickStatsLookupTool(BaseTool):
    """Fast, safe lookup for quick stats without DataFrame complications"""
    
    def __init__(self):
        super().__init__()
        self.name = "quick_stats_lookup"
        self.category = "database"
        self.description = "Fast, reliable stats lookup with no DataFrame issues"
        self.parameters = [
            ToolParameter(
                name="stat_type",
                type="string",
                description="Type of stat to look up",
                required=True,
                choices=["top_players", "player_info", "team_summary", "injury_check"]
            ),
            ToolParameter(
                name="position",
                type="string",
                description="Position filter",
                required=False,
                choices=["qb", "rb", "wr", "te", "k", "def"]
            ),
            ToolParameter(
                name="limit",
                type="int",
                description="Number of results",
                required=False,
                default=10
            )
        ]
    
    async def execute(self, stat_type: str, position: Optional[str] = None, limit: int = 10, **kwargs) -> ToolResult:
        """Execute quick stats lookup"""
        try:
            # Get database path
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(script_dir, "data", "fantasy_ppr.db")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if stat_type == "top_players":
                if position:
                    cursor.execute("""
                        SELECT player_name, position, team_name, fantasy_points, bye_week
                        FROM player_projections 
                        WHERE season = 2025 AND projection_type = 'preseason'
                        AND LOWER(position) = LOWER(?)
                        ORDER BY fantasy_points DESC
                        LIMIT ?
                    """, (position, limit))
                else:
                    cursor.execute("""
                        SELECT player_name, position, team_name, fantasy_points, bye_week
                        FROM player_projections 
                        WHERE season = 2025 AND projection_type = 'preseason'
                        ORDER BY fantasy_points DESC
                        LIMIT ?
                    """, (limit,))
            
            elif stat_type == "injury_check":
                cursor.execute("""
                    SELECT team, position, injury_updates, confidence_score
                    FROM team_position_intelligence 
                    WHERE season = 2025 
                    AND injury_updates IS NOT NULL 
                    AND injury_updates != ''
                    ORDER BY confidence_score DESC
                    LIMIT ?
                """, (limit,))
            
            elif stat_type == "team_summary":
                cursor.execute("""
                    SELECT team, COUNT(*) as players, AVG(fantasy_points) as avg_points
                    FROM player_projections 
                    WHERE season = 2025 AND projection_type = 'preseason'
                    GROUP BY team
                    ORDER BY avg_points DESC
                    LIMIT ?
                """, (limit,))
            
            # Fetch results safely
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            
            # Convert to safe dict format
            results = []
            for row in rows:
                result_dict = {}
                for i, col in enumerate(columns):
                    result_dict[col] = row[i]
                results.append(result_dict)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "results": results,
                    "total_found": len(results),
                    "stat_type": stat_type,
                    "position_filter": position
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Quick stats lookup failed: {str(e)}"
            )
