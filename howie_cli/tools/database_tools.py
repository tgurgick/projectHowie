"""
Database tools for accessing existing fantasy football databases
Compatible with original projectHowie database structure
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter


class DatabaseQueryTool(BaseTool):
    """Query existing fantasy football databases"""
    
    def __init__(self):
        super().__init__()
        self.name = "database_query"
        self.category = "database"
        self.description = "Query existing fantasy football databases"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="SQL query or natural language request",
                required=True
            ),
            ToolParameter(
                name="scoring_type",
                type="string",
                description="Database to query (ppr, half_ppr, standard)",
                required=False,
                default="ppr",
                choices=["ppr", "half_ppr", "standard"]
            ),
            ToolParameter(
                name="return_type",
                type="string",
                description="Format of returned data",
                required=False,
                default="dataframe",
                choices=["dataframe", "dict", "summary"]
            )
        ]
        self.db_paths = {
            "ppr": "data/fantasy_ppr.db",
            "half_ppr": "data/fantasy_halfppr.db",
            "standard": "data/fantasy_standard.db"
        }
    
    async def execute(self, query: str, scoring_type: str = "ppr",
                     return_type: str = "dataframe", **kwargs) -> ToolResult:
        """Execute database query"""
        try:
            # Check if database exists
            db_path = self.db_paths.get(scoring_type)
            if not Path(db_path).exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Database not found: {db_path}"
                )
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            
            # If natural language, convert to SQL
            if not query.strip().upper().startswith(('SELECT', 'WITH')):
                sql_query = self._convert_to_sql(query)
            else:
                sql_query = query
            
            # Execute query
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Format return based on type
            if return_type == "dict":
                data = df.to_dict('records')
            elif return_type == "summary":
                data = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample": df.head(5).to_dict('records'),
                    "stats": df.describe().to_dict() if not df.empty else {}
                }
            else:
                data = df
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=data,
                metadata={
                    "database": scoring_type,
                    "query": sql_query,
                    "rows_returned": len(df),
                    "columns": list(df.columns)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Query failed: {str(e)}"
            )
    
    def _convert_to_sql(self, natural_language: str) -> str:
        """Convert natural language to SQL query"""
        nl_lower = natural_language.lower()
        
        # Common query patterns
        if "top" in nl_lower and "qb" in nl_lower:
            return """
            SELECT p.name, p.team, AVG(pgs.fantasy_points) as avg_points
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            WHERE p.position = 'QB'
            GROUP BY p.player_id
            ORDER BY avg_points DESC
            LIMIT 10
            """
        
        elif "compare" in nl_lower:
            # Extract player names (simplified)
            return """
            SELECT p.name, p.position, p.team,
                   AVG(pgs.fantasy_points) as avg_points,
                   MAX(pgs.fantasy_points) as max_points,
                   MIN(pgs.fantasy_points) as min_points
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            GROUP BY p.player_id
            """
        
        else:
            # Default query
            return "SELECT * FROM players LIMIT 10"


class PlayerStatsTool(BaseTool):
    """Get comprehensive player statistics from existing databases"""
    
    def __init__(self):
        super().__init__()
        self.name = "player_stats"
        self.category = "database"
        self.description = "Get player statistics from existing databases"
        self.parameters = [
            ToolParameter(
                name="player_name",
                type="string",
                description="Player name to look up",
                required=True
            ),
            ToolParameter(
                name="season",
                type="int",
                description="Season year",
                required=False,
                default=2024
            ),
            ToolParameter(
                name="weeks",
                type="list",
                description="Specific weeks to include",
                required=False
            ),
            ToolParameter(
                name="include_advanced",
                type="bool",
                description="Include advanced stats",
                required=False,
                default=True
            )
        ]
    
    async def execute(self, player_name: str, season: int = 2024,
                     weeks: Optional[List[int]] = None,
                     include_advanced: bool = True, **kwargs) -> ToolResult:
        """Get player statistics"""
        try:
            conn = sqlite3.connect("data/fantasy_ppr.db")
            
            # Base query for player stats
            query = """
            SELECT p.*, pgs.*, g.week, g.season
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE p.name LIKE ?
            AND g.season = ?
            """
            
            params = [f"%{player_name}%", season]
            
            if weeks:
                placeholders = ','.join(['?' for _ in weeks])
                query += f" AND g.week IN ({placeholders})"
                params.extend(weeks)
            
            query += " ORDER BY g.week"
            
            # Get base stats
            df_stats = pd.read_sql_query(query, conn, params=params)
            
            result_data = {
                "player_info": {},
                "game_stats": [],
                "season_summary": {},
                "advanced_stats": {}
            }
            
            if not df_stats.empty:
                # Player info
                result_data["player_info"] = {
                    "name": df_stats.iloc[0]["name"],
                    "position": df_stats.iloc[0]["position"],
                    "team": df_stats.iloc[0]["team"]
                }
                
                # Game stats
                result_data["game_stats"] = df_stats.to_dict('records')
                
                # Season summary
                result_data["season_summary"] = {
                    "games_played": len(df_stats),
                    "total_points": df_stats["fantasy_points"].sum(),
                    "avg_points": df_stats["fantasy_points"].mean(),
                    "max_points": df_stats["fantasy_points"].max(),
                    "min_points": df_stats["fantasy_points"].min(),
                    "std_dev": df_stats["fantasy_points"].std()
                }
                
                # Advanced stats if requested
                if include_advanced:
                    adv_query = """
                    SELECT pas.*, prs.*, pss.*
                    FROM players p
                    LEFT JOIN player_advanced_stats pas ON p.player_id = pas.player_id
                    LEFT JOIN player_route_stats prs ON p.name = prs.player_name
                    LEFT JOIN player_scheme_stats pss ON p.name = pss.player_name
                    WHERE p.name LIKE ?
                    LIMIT 1
                    """
                    
                    df_adv = pd.read_sql_query(adv_query, conn, params=[f"%{player_name}%"])
                    if not df_adv.empty:
                        result_data["advanced_stats"] = df_adv.iloc[0].to_dict()
            
            conn.close()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                metadata={
                    "player": player_name,
                    "season": season,
                    "weeks": weeks,
                    "games_found": len(result_data["game_stats"])
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to get player stats: {str(e)}"
            )


class TeamAnalysisTool(BaseTool):
    """Analyze team performance from existing databases"""
    
    def __init__(self):
        super().__init__()
        self.name = "team_analysis"
        self.category = "database"
        self.description = "Analyze team performance and trends"
        self.parameters = [
            ToolParameter(
                name="team",
                type="string",
                description="Team abbreviation (e.g., SF, DAL)",
                required=True
            ),
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of analysis",
                required=False,
                default="overall",
                choices=["overall", "offense", "defense", "positional"]
            )
        ]
    
    async def execute(self, team: str, analysis_type: str = "overall", **kwargs) -> ToolResult:
        """Analyze team performance"""
        try:
            conn = sqlite3.connect("data/fantasy_ppr.db")
            
            if analysis_type == "positional":
                query = """
                SELECT p.position,
                       COUNT(DISTINCT p.player_id) as player_count,
                       AVG(pgs.fantasy_points) as avg_points,
                       SUM(pgs.fantasy_points) as total_points
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.team = ?
                GROUP BY p.position
                ORDER BY avg_points DESC
                """
                df = pd.read_sql_query(query, conn, params=[team.upper()])
                
            else:
                query = """
                SELECT p.name, p.position,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                WHERE p.team = ?
                GROUP BY p.player_id
                ORDER BY avg_points DESC
                """
                df = pd.read_sql_query(query, conn, params=[team.upper()])
            
            conn.close()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=df.to_dict('records'),
                metadata={
                    "team": team.upper(),
                    "analysis_type": analysis_type,
                    "players_analyzed": len(df)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to analyze team: {str(e)}"
            )


class HistoricalTrendsTool(BaseTool):
    """Analyze historical trends from existing databases"""
    
    def __init__(self):
        super().__init__()
        self.name = "historical_trends"
        self.category = "database"
        self.description = "Analyze historical performance trends"
        self.parameters = [
            ToolParameter(
                name="entity",
                type="string",
                description="Player or team to analyze",
                required=True
            ),
            ToolParameter(
                name="metric",
                type="string",
                description="Metric to analyze",
                required=False,
                default="fantasy_points"
            ),
            ToolParameter(
                name="timeframe",
                type="string",
                description="Timeframe for analysis",
                required=False,
                default="season",
                choices=["season", "last_4", "last_8", "all"]
            )
        ]
    
    async def execute(self, entity: str, metric: str = "fantasy_points",
                     timeframe: str = "season", **kwargs) -> ToolResult:
        """Analyze historical trends"""
        try:
            conn = sqlite3.connect("data/fantasy_ppr.db")
            
            # Determine if entity is player or team
            check_query = "SELECT * FROM players WHERE name LIKE ? LIMIT 1"
            df_check = pd.read_sql_query(check_query, conn, params=[f"%{entity}%"])
            
            if not df_check.empty:
                # It's a player
                query = f"""
                SELECT g.week, g.season, pgs.{metric} as value
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.name LIKE ?
                ORDER BY g.season DESC, g.week DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[f"%{entity}%"])
                
                # Apply timeframe filter
                if timeframe == "last_4":
                    df = df.head(4)
                elif timeframe == "last_8":
                    df = df.head(8)
                elif timeframe == "season":
                    if not df.empty:
                        latest_season = df.iloc[0]["season"]
                        df = df[df["season"] == latest_season]
                
                # Calculate trend
                if len(df) > 1:
                    trend = "increasing" if df["value"].iloc[0] > df["value"].iloc[-1] else "decreasing"
                    volatility = df["value"].std()
                else:
                    trend = "insufficient data"
                    volatility = 0
                
                result_data = {
                    "entity": entity,
                    "entity_type": "player",
                    "metric": metric,
                    "trend": trend,
                    "volatility": volatility,
                    "data_points": df.to_dict('records'),
                    "summary": {
                        "mean": df["value"].mean() if not df.empty else 0,
                        "median": df["value"].median() if not df.empty else 0,
                        "max": df["value"].max() if not df.empty else 0,
                        "min": df["value"].min() if not df.empty else 0
                    }
                }
                
            else:
                result_data = {"error": "Entity not found"}
            
            conn.close()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                metadata={
                    "entity": entity,
                    "metric": metric,
                    "timeframe": timeframe
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to analyze trends: {str(e)}"
            )


class DatabaseInfoTool(BaseTool):
    """Get information about existing databases"""
    
    def __init__(self):
        super().__init__()
        self.name = "database_info"
        self.category = "database"
        self.description = "Get information about existing fantasy databases"
        self.parameters = [
            ToolParameter(
                name="info_type",
                type="string",
                description="Type of information to retrieve",
                required=False,
                default="summary",
                choices=["summary", "tables", "schema", "stats"]
            )
        ]
    
    async def execute(self, info_type: str = "summary", **kwargs) -> ToolResult:
        """Get database information"""
        try:
            databases = ["fantasy_ppr.db", "fantasy_half_ppr.db", "fantasy_standard.db"]
            info = {}
            
            for db_name in databases:
                db_path = f"data/{db_name}"
                if not Path(db_path).exists():
                    info[db_name] = {"status": "not found"}
                    continue
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                if info_type == "summary":
                    # Get basic info
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    db_info = {
                        "status": "available",
                        "tables": [t[0] for t in tables],
                        "size_mb": Path(db_path).stat().st_size / (1024 * 1024)
                    }
                    
                    # Get row counts for main tables
                    for table in ["players", "player_game_stats", "games"]:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            db_info[f"{table}_count"] = count
                        except:
                            pass
                    
                    info[db_name] = db_info
                    
                elif info_type == "tables":
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    info[db_name] = [t[0] for t in cursor.fetchall()]
                    
                elif info_type == "schema":
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    schema_info = {}
                    
                    for table in tables:
                        cursor.execute(f"PRAGMA table_info({table[0]})")
                        schema_info[table[0]] = cursor.fetchall()
                    
                    info[db_name] = schema_info
                
                conn.close()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=info,
                metadata={"info_type": info_type}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to get database info: {str(e)}"
            )