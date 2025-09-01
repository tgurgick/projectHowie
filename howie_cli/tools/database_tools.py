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
            
            # Format return based on type - ALWAYS return safe dict format
            if df.empty:
                data = {
                    "rows": 0,
                    "columns": list(df.columns),
                    "records": [],
                    "message": "No data found"
                }
            elif return_type == "dict":
                data = df.to_dict('records')
            elif return_type == "summary":
                data = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample": df.head(5).to_dict('records'),
                    "stats": df.describe().to_dict() if not df.empty else {}
                }
            else:
                # Default: return dict format instead of DataFrame to avoid ambiguity
                data = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "records": df.to_dict('records')
                }
            
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


class TopPlayersTool(BaseTool):
    """Get top players by position and season"""
    
    def __init__(self):
        super().__init__()
        self.name = "top_players"
        self.category = "database"
        self.description = "Get top players by position, season, and scoring type"
        self.parameters = [
            ToolParameter(
                name="position",
                type="string",
                description="Position to rank (QB, RB, WR, TE, K, DEF)",
                required=True,
                choices=["QB", "RB", "WR", "TE", "K", "DEF"]
            ),
            ToolParameter(
                name="season",
                type="int",
                description="Season year",
                required=False,
                default=2025
            ),
            ToolParameter(
                name="limit",
                type="int",
                description="Number of top players to return",
                required=False,
                default=10
            ),
            ToolParameter(
                name="scoring_type",
                type="string",
                description="Scoring system to use",
                required=False,
                default="ppr",
                choices=["ppr", "half_ppr", "standard"]
            ),
            ToolParameter(
                name="metric",
                type="string",
                description="Metric to rank by",
                required=False,
                default="fantasy_points",
                choices=["fantasy_points", "total_yards", "touchdowns", "receptions"]
            )
        ]
    
    async def execute(self, position: str, season: int = 2025, limit: int = 10,
                     scoring_type: str = "ppr", metric: str = "fantasy_points", **kwargs) -> ToolResult:
        """Get top players by position"""
        try:
            # Map scoring type to database
            db_mapping = {
                "ppr": "fantasy_ppr.db",
                "half_ppr": "fantasy_halfppr.db", 
                "standard": "fantasy_standard.db"
            }
            
            db_name = db_mapping.get(scoring_type, "fantasy_ppr.db")
            
            # Find the database file relative to the project root
            import os
            current_dir = Path(__file__).parent  # howie_cli/tools/
            project_root = current_dir.parent.parent  # project root
            db_path = project_root / "data" / db_name
            
            if not db_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Database not found: {db_path}"
                )
            
            conn = sqlite3.connect(db_path)
            
            # For 2025+ seasons, use projections data instead of historical game stats
            if season >= 2025:
                query = """
                SELECT player_name as name, team_name as team, position,
                       fantasy_points as total_points,
                       fantasy_points as avg_points,
                       games as games_played,
                       bye_week
                FROM player_projections 
                WHERE season = ? AND projection_type = 'preseason'
                AND position = ?
                ORDER BY fantasy_points DESC
                LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=[season, position.lower(), limit])
                
                # Format results for 2025 projections
                players = []
                for _, row in df.iterrows():
                    player_info = {
                        "name": row["name"],
                        "team": row["team"],
                        "position": row["position"],
                        "total_points": row["total_points"],
                        "avg_points": row["avg_points"],
                        "games_played": row["games_played"],
                        "bye_week": row.get("bye_week", "N/A")
                    }
                    players.append(player_info)
                
                conn.close()
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "players": players,
                        "season": season,
                        "position": position,
                        "scoring_type": scoring_type,
                        "total_found": len(players)
                    }
                )
            
            # Build query based on position and metric (historical data)
            if position == "QB":
                query = """
                SELECT p.name, p.team, p.position,
                       SUM(pgs.fantasy_points) as total_points,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played,
                       SUM(pgs.pass_yards) as total_passing_yards,
                       SUM(pgs.pass_tds) as total_passing_tds,
                       SUM(pgs.rush_yards) as total_rushing_yards,
                       SUM(pgs.rush_tds) as total_rushing_tds
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.position = 'QB' AND g.season = ?
                GROUP BY p.player_id, p.name, p.team, p.position
                ORDER BY total_points DESC
                LIMIT ?
                """
            elif position == "RB":
                query = """
                SELECT p.name, p.team, p.position,
                       SUM(pgs.fantasy_points) as total_points,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played,
                       SUM(pgs.rush_yards) as total_rushing_yards,
                       SUM(pgs.rush_tds) as total_rushing_tds,
                       SUM(pgs.rec_yards) as total_receiving_yards,
                       SUM(pgs.rec_tds) as total_receiving_tds,
                       SUM(pgs.receptions) as total_receptions
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.position = 'RB' AND g.season = ?
                GROUP BY p.player_id, p.name, p.team, p.position
                ORDER BY total_points DESC
                LIMIT ?
                """
            elif position == "WR":
                query = """
                SELECT p.name, p.team, p.position,
                       SUM(pgs.fantasy_points) as total_points,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played,
                       SUM(pgs.rec_yards) as total_receiving_yards,
                       SUM(pgs.rec_tds) as total_receiving_tds,
                       SUM(pgs.receptions) as total_receptions,
                       SUM(pgs.rush_yards) as total_rushing_yards,
                       SUM(pgs.rush_tds) as total_rushing_tds
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.position = 'WR' AND g.season = ?
                GROUP BY p.player_id, p.name, p.team, p.position
                ORDER BY total_points DESC
                LIMIT ?
                """
            elif position == "TE":
                query = """
                SELECT p.name, p.team, p.position,
                       SUM(pgs.fantasy_points) as total_points,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played,
                       SUM(pgs.rec_yards) as total_receiving_yards,
                       SUM(pgs.rec_tds) as total_receiving_tds,
                       SUM(pgs.receptions) as total_receptions
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.position = 'TE' AND g.season = ?
                GROUP BY p.player_id, p.name, p.team, p.position
                ORDER BY total_points DESC
                LIMIT ?
                """
            else:
                # For K and DEF, use simpler query
                query = """
                SELECT p.name, p.team, p.position,
                       SUM(pgs.fantasy_points) as total_points,
                       AVG(pgs.fantasy_points) as avg_points,
                       COUNT(*) as games_played
                FROM players p
                JOIN player_game_stats pgs ON p.player_id = pgs.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE p.position = ? AND g.season = ?
                GROUP BY p.player_id, p.name, p.team, p.position
                ORDER BY total_points DESC
                LIMIT ?
                """
            
            # Execute query
            if position in ["QB", "RB", "WR", "TE"]:
                df = pd.read_sql_query(query, conn, params=[season, limit])
            else:
                df = pd.read_sql_query(query, conn, params=[position, season, limit])
            
            conn.close()
            
            # Format results
            players = []
            for _, row in df.iterrows():
                player_data = {
                    "name": row["name"],
                    "team": row["team"],
                    "position": row["position"],
                    "total_points": round(row["total_points"], 2),
                    "avg_points": round(row["avg_points"], 2),
                    "games_played": int(row["games_played"])
                }
                
                # Add position-specific stats
                if position == "QB":
                    player_data.update({
                        "passing_yards": int(row["total_passing_yards"]),
                        "passing_tds": int(row["total_passing_tds"]),
                        "rushing_yards": int(row["total_rushing_yards"]),
                        "rushing_tds": int(row["total_rushing_tds"])
                    })
                elif position == "RB":
                    player_data.update({
                        "rushing_yards": int(row["total_rushing_yards"]),
                        "rushing_tds": int(row["total_rushing_tds"]),
                        "receiving_yards": int(row["total_receiving_yards"]),
                        "receiving_tds": int(row["total_receiving_tds"]),
                        "receptions": int(row["total_receptions"])
                    })
                elif position == "WR":
                    player_data.update({
                        "receiving_yards": int(row["total_receiving_yards"]),
                        "receiving_tds": int(row["total_receiving_tds"]),
                        "receptions": int(row["total_receptions"]),
                        "rushing_yards": int(row["total_rushing_yards"]),
                        "rushing_tds": int(row["total_rushing_tds"])
                    })
                elif position == "TE":
                    player_data.update({
                        "receiving_yards": int(row["total_receiving_yards"]),
                        "receiving_tds": int(row["total_receiving_tds"]),
                        "receptions": int(row["total_receptions"])
                    })
                
                players.append(player_data)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "position": position,
                    "season": season,
                    "scoring_type": scoring_type,
                    "players": players,
                    "total_players": len(players)
                },
                metadata={
                    "position": position,
                    "season": season,
                    "scoring_type": scoring_type,
                    "metric": metric,
                    "limit": limit
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to get top players: {str(e)}"
            )