"""
Code generation tools for creating analysis scripts and queries
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import textwrap

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.workspace import WorkspaceManager


class GenerateAnalysisScriptTool(BaseTool):
    """Generate Python analysis scripts based on requirements"""
    
    def __init__(self):
        super().__init__()
        self.name = "generate_analysis_script"
        self.category = "code_generation"
        self.description = "Generate Python scripts for fantasy football analysis"
        self.parameters = [
            ToolParameter(
                name="requirements",
                type="string",
                description="Natural language description of analysis requirements",
                required=True
            ),
            ToolParameter(
                name="script_name",
                type="string",
                description="Name for the generated script",
                required=False,
                default="analysis_script.py"
            ),
            ToolParameter(
                name="include_imports",
                type="bool",
                description="Include common import statements",
                required=False,
                default=True
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, requirements: str, script_name: str = "analysis_script.py",
                     include_imports: bool = True, **kwargs) -> ToolResult:
        """Generate analysis script"""
        try:
            # Parse requirements to determine needed components
            requirements_lower = requirements.lower()
            
            # Start building script
            script_parts = []
            
            # Add imports if requested
            if include_imports:
                imports = self._generate_imports(requirements_lower)
                script_parts.append(imports)
            
            # Generate main analysis function
            main_function = self._generate_main_function(requirements)
            script_parts.append(main_function)
            
            # Generate helper functions based on requirements
            if "compare" in requirements_lower or "vs" in requirements_lower:
                script_parts.append(self._generate_comparison_function())
            
            if "trend" in requirements_lower or "over time" in requirements_lower:
                script_parts.append(self._generate_trend_analysis_function())
            
            if "projection" in requirements_lower or "predict" in requirements_lower:
                script_parts.append(self._generate_projection_function())
            
            if "optimize" in requirements_lower or "lineup" in requirements_lower:
                script_parts.append(self._generate_lineup_optimization_function())
            
            # Add main block
            script_parts.append(self._generate_main_block())
            
            # Combine all parts
            script = "\n\n".join(script_parts)
            
            # Save script
            script_path = self.workspace.write_file(
                data=script,
                file_name=script_name,
                subfolder="scripts"
            )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "script": script,
                    "path": str(script_path)
                },
                metadata={
                    "script_name": script_name,
                    "lines": len(script.split('\n')),
                    "functions_generated": self._count_functions(script)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to generate script: {str(e)}"
            )
    
    def _generate_imports(self, requirements: str) -> str:
        """Generate import statements based on requirements"""
        imports = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated Fantasy Football Analysis Script",
            '"""',
            "",
            "import pandas as pd",
            "import numpy as np",
            "from datetime import datetime, timedelta",
            "from pathlib import Path",
            "import sqlite3",
            "from typing import Dict, List, Optional, Tuple"
        ]
        
        if "plot" in requirements or "chart" in requirements or "visual" in requirements:
            imports.extend([
                "import matplotlib.pyplot as plt",
                "import seaborn as sns"
            ])
        
        if "api" in requirements or "fetch" in requirements:
            imports.append("import requests")
        
        if "json" in requirements:
            imports.append("import json")
        
        return "\n".join(imports)
    
    def _generate_main_function(self, requirements: str) -> str:
        """Generate main analysis function"""
        return textwrap.dedent('''
        def analyze_fantasy_data(
            players: List[str] = None,
            weeks: List[int] = None,
            season: int = 2024,
            scoring_type: str = "ppr"
        ) -> pd.DataFrame:
            """
            Main analysis function for fantasy football data
            
            Args:
                players: List of player names to analyze
                weeks: List of weeks to include
                season: Season year
                scoring_type: Scoring system (ppr, half_ppr, standard)
            
            Returns:
                DataFrame with analysis results
            """
            # Connect to database
            db_path = f"data/fantasy_{scoring_type}.db"
            conn = sqlite3.connect(db_path)
            
            # Build query
            query = """
            SELECT p.name, p.position, p.team,
                   pgs.week, pgs.fantasy_points,
                   pgs.pass_yards, pgs.rush_yards, pgs.rec_yards,
                   pgs.total_tds
            FROM players p
            JOIN player_game_stats pgs ON p.player_id = pgs.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.season = ?
            """
            
            params = [season]
            
            if players:
                placeholders = ','.join(['?' for _ in players])
                query += f" AND p.name IN ({placeholders})"
                params.extend(players)
            
            if weeks:
                placeholders = ','.join(['?' for _ in weeks])
                query += f" AND pgs.week IN ({placeholders})"
                params.extend(weeks)
            
            query += " ORDER BY p.name, pgs.week"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
        ''')
    
    def _generate_comparison_function(self) -> str:
        """Generate player comparison function"""
        return textwrap.dedent('''
        def compare_players(
            player1: str,
            player2: str,
            metrics: List[str] = None,
            season: int = 2024
        ) -> Dict:
            """
            Compare two players across specified metrics
            
            Args:
                player1: First player name
                player2: Second player name
                metrics: List of metrics to compare
                season: Season year
            
            Returns:
                Dictionary with comparison results
            """
            if not metrics:
                metrics = ["fantasy_points", "total_tds", "total_yards"]
            
            # Get data for both players
            df = analyze_fantasy_data(
                players=[player1, player2],
                season=season
            )
            
            comparison = {}
            
            for metric in metrics:
                if metric in df.columns:
                    p1_avg = df[df['name'] == player1][metric].mean()
                    p2_avg = df[df['name'] == player2][metric].mean()
                    
                    comparison[metric] = {
                        player1: round(p1_avg, 2),
                        player2: round(p2_avg, 2),
                        "winner": player1 if p1_avg > p2_avg else player2,
                        "difference": round(abs(p1_avg - p2_avg), 2)
                    }
            
            return comparison
        ''')
    
    def _generate_trend_analysis_function(self) -> str:
        """Generate trend analysis function"""
        return textwrap.dedent('''
        def analyze_trends(
            player: str,
            metric: str = "fantasy_points",
            window: int = 3
        ) -> pd.DataFrame:
            """
            Analyze trends for a player over time
            
            Args:
                player: Player name
                metric: Metric to analyze
                window: Rolling window size
            
            Returns:
                DataFrame with trend analysis
            """
            df = analyze_fantasy_data(players=[player])
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate rolling average
            df[f'{metric}_rolling_avg'] = df[metric].rolling(window=window).mean()
            
            # Calculate trend (slope of linear regression)
            from scipy import stats
            weeks = df['week'].values
            values = df[metric].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, values)
            
            df['trend_slope'] = slope
            df['trend_direction'] = 'up' if slope > 0 else 'down'
            df['trend_strength'] = abs(r_value)
            
            return df
        ''')
    
    def _generate_projection_function(self) -> str:
        """Generate projection function"""
        return textwrap.dedent('''
        def project_performance(
            player: str,
            weeks_ahead: int = 1,
            method: str = "average"
        ) -> Dict:
            """
            Project future performance for a player
            
            Args:
                player: Player name
                weeks_ahead: Number of weeks to project
                method: Projection method (average, weighted, trend)
            
            Returns:
                Dictionary with projections
            """
            df = analyze_fantasy_data(players=[player])
            
            if df.empty:
                return {}
            
            projections = {}
            
            if method == "average":
                # Simple average of recent performances
                recent_avg = df['fantasy_points'].tail(4).mean()
                projections['projected_points'] = round(recent_avg, 2)
            
            elif method == "weighted":
                # Weighted average (more recent games weighted higher)
                weights = np.array([0.1, 0.2, 0.3, 0.4])
                recent_points = df['fantasy_points'].tail(4).values
                
                if len(recent_points) == 4:
                    weighted_avg = np.average(recent_points, weights=weights)
                    projections['projected_points'] = round(weighted_avg, 2)
            
            elif method == "trend":
                # Use trend analysis
                trend_df = analyze_trends(player)
                if not trend_df.empty:
                    slope = trend_df['trend_slope'].iloc[0]
                    last_value = df['fantasy_points'].iloc[-1]
                    projections['projected_points'] = round(
                        last_value + (slope * weeks_ahead), 2
                    )
            
            projections['confidence'] = self._calculate_confidence(df)
            projections['risk_level'] = self._assess_risk(df)
            
            return projections
        
        def _calculate_confidence(df: pd.DataFrame) -> str:
            """Calculate confidence level based on consistency"""
            std_dev = df['fantasy_points'].std()
            mean = df['fantasy_points'].mean()
            cv = std_dev / mean if mean > 0 else 1
            
            if cv < 0.2:
                return "high"
            elif cv < 0.4:
                return "medium"
            else:
                return "low"
        
        def _assess_risk(df: pd.DataFrame) -> str:
            """Assess risk level"""
            recent_games = df.tail(3)
            if recent_games['fantasy_points'].min() < 5:
                return "high"
            elif recent_games['fantasy_points'].std() > 10:
                return "medium"
            else:
                return "low"
        ''')
    
    def _generate_lineup_optimization_function(self) -> str:
        """Generate lineup optimization function"""
        return textwrap.dedent('''
        def optimize_lineup(
            available_players: List[str],
            constraints: Dict = None,
            scoring_type: str = "ppr"
        ) -> Dict:
            """
            Optimize fantasy lineup based on projections
            
            Args:
                available_players: List of available player names
                constraints: Position constraints (e.g., {'QB': 1, 'RB': 2})
                scoring_type: Scoring system
            
            Returns:
                Dictionary with optimal lineup
            """
            if not constraints:
                constraints = {
                    'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1
                }
            
            # Get player data
            df = analyze_fantasy_data(
                players=available_players,
                scoring_type=scoring_type
            )
            
            if df.empty:
                return {}
            
            # Calculate average points per player
            player_avg = df.groupby(['name', 'position'])['fantasy_points'].mean().reset_index()
            player_avg = player_avg.sort_values('fantasy_points', ascending=False)
            
            lineup = {}
            used_players = set()
            
            # Fill required positions
            for position, count in constraints.items():
                if position == 'FLEX':
                    continue  # Handle FLEX last
                
                position_players = player_avg[
                    (player_avg['position'] == position) &
                    (~player_avg['name'].isin(used_players))
                ].head(count)
                
                for _, player in position_players.iterrows():
                    lineup[f"{position}_{len([k for k in lineup if k.startswith(position)]) + 1}"] = {
                        'name': player['name'],
                        'projected_points': round(player['fantasy_points'], 2)
                    }
                    used_players.add(player['name'])
            
            # Fill FLEX position
            if 'FLEX' in constraints:
                flex_eligible = player_avg[
                    (player_avg['position'].isin(['RB', 'WR', 'TE'])) &
                    (~player_avg['name'].isin(used_players))
                ].head(constraints['FLEX'])
                
                for _, player in flex_eligible.iterrows():
                    lineup['FLEX'] = {
                        'name': player['name'],
                        'position': player['position'],
                        'projected_points': round(player['fantasy_points'], 2)
                    }
                    used_players.add(player['name'])
            
            # Calculate total projected points
            total_points = sum(
                pos['projected_points'] for pos in lineup.values()
            )
            
            return {
                'lineup': lineup,
                'total_projected_points': round(total_points, 2),
                'players_used': list(used_players)
            }
        ''')
    
    def _generate_main_block(self) -> str:
        """Generate main execution block"""
        return textwrap.dedent('''
        if __name__ == "__main__":
            # Example usage
            print("Fantasy Football Analysis Script")
            print("-" * 40)
            
            # Run analysis
            results = analyze_fantasy_data(
                season=2024,
                scoring_type="ppr"
            )
            
            if not results.empty:
                print(f"\\nAnalyzed {len(results)} records")
                print(f"Players: {results['name'].nunique()}")
                print(f"Average points: {results['fantasy_points'].mean():.2f}")
            else:
                print("No data found")
        ''')
    
    def _count_functions(self, script: str) -> int:
        """Count number of functions in script"""
        return script.count("def ")


class GenerateSQLQueryTool(BaseTool):
    """Generate SQL queries from natural language"""
    
    def __init__(self):
        super().__init__()
        self.name = "generate_sql_query"
        self.category = "code_generation"
        self.description = "Generate SQL queries for fantasy football database"
        self.parameters = [
            ToolParameter(
                name="request",
                type="string",
                description="Natural language description of query",
                required=True
            ),
            ToolParameter(
                name="table_context",
                type="dict",
                description="Information about available tables",
                required=False
            )
        ]
    
    async def execute(self, request: str, table_context: Optional[Dict] = None, **kwargs) -> ToolResult:
        """Generate SQL query"""
        try:
            request_lower = request.lower()
            
            # Default table context if not provided
            if not table_context:
                table_context = {
                    "players": ["player_id", "name", "position", "team"],
                    "player_game_stats": ["player_id", "game_id", "week", "fantasy_points", "pass_yards", "rush_yards", "rec_yards", "total_tds"],
                    "games": ["game_id", "season", "week", "home_team", "away_team"],
                    "player_advanced_stats": ["player_id", "game_id", "snap_share", "target_share", "epa", "cpoe"],
                    "player_route_stats": ["player_name", "season", "routes_run", "yards_per_route_run"],
                    "fantasy_market": ["player_name", "season", "week", "adp", "ecr"]
                }
            
            # Generate query based on request
            query = self._build_query(request_lower, table_context)
            
            # Format query
            formatted_query = self._format_sql(query)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "query": formatted_query,
                    "explanation": self._explain_query(query)
                },
                metadata={
                    "tables_used": self._extract_tables(query),
                    "query_type": self._determine_query_type(query)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to generate SQL query: {str(e)}"
            )
    
    def _build_query(self, request: str, table_context: Dict) -> str:
        """Build SQL query from request"""
        # Determine query components
        select_clause = "SELECT "
        from_clause = "FROM "
        where_clause = ""
        join_clause = ""
        group_clause = ""
        order_clause = ""
        
        # Parse request for key information
        if "top" in request or "best" in request:
            limit_match = self._extract_number(request)
            limit = f" LIMIT {limit_match}" if limit_match else " LIMIT 10"
        else:
            limit = ""
        
        # Determine what to select
        if "everything" in request or "all" in request:
            select_clause += "*"
        elif "average" in request or "avg" in request:
            select_clause += "AVG(fantasy_points) as avg_points"
            group_clause = " GROUP BY p.name, p.position"
        elif "total" in request or "sum" in request:
            select_clause += "SUM(fantasy_points) as total_points"
            group_clause = " GROUP BY p.name, p.position"
        else:
            select_clause += "p.name, p.position, p.team, pgs.fantasy_points"
        
        # Determine tables
        if "player" in request:
            from_clause += "players p"
            
            if "stats" in request or "points" in request or "performance" in request:
                join_clause += " JOIN player_game_stats pgs ON p.player_id = pgs.player_id"
                join_clause += " JOIN games g ON pgs.game_id = g.game_id"
        
        # Add conditions
        conditions = []
        
        if "2024" in request or "current" in request or "this season" in request:
            conditions.append("g.season = 2024")
        elif "2023" in request or "last season" in request:
            conditions.append("g.season = 2023")
        
        if "qb" in request:
            conditions.append("p.position = 'QB'")
        elif "rb" in request or "running back" in request:
            conditions.append("p.position = 'RB'")
        elif "wr" in request or "wide receiver" in request:
            conditions.append("p.position = 'WR'")
        elif "te" in request or "tight end" in request:
            conditions.append("p.position = 'TE'")
        
        if "week" in request:
            week_num = self._extract_number(request)
            if week_num:
                conditions.append(f"g.week = {week_num}")
        
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        
        # Add ordering
        if "top" in request or "best" in request:
            order_clause = " ORDER BY fantasy_points DESC"
        elif "worst" in request or "bottom" in request:
            order_clause = " ORDER BY fantasy_points ASC"
        
        # Combine all parts
        query = select_clause + " " + from_clause + join_clause + where_clause + group_clause + order_clause + limit
        
        return query
    
    def _format_sql(self, query: str) -> str:
        """Format SQL query for readability"""
        keywords = ['SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']
        
        formatted = query
        for keyword in keywords:
            formatted = formatted.replace(f" {keyword} ", f"\n{keyword} ")
        
        return formatted.strip()
    
    def _explain_query(self, query: str) -> str:
        """Generate explanation for query"""
        explanation = "This query "
        
        if "SELECT AVG" in query:
            explanation += "calculates the average "
        elif "SELECT SUM" in query:
            explanation += "calculates the total "
        else:
            explanation += "retrieves "
        
        if "fantasy_points" in query:
            explanation += "fantasy points "
        
        if "JOIN" in query:
            explanation += "by joining player and statistics tables "
        
        if "WHERE" in query:
            explanation += "with specific conditions "
        
        if "ORDER BY" in query:
            if "DESC" in query:
                explanation += "sorted in descending order "
            else:
                explanation += "sorted in ascending order "
        
        if "LIMIT" in query:
            explanation += "limited to a specific number of results"
        
        return explanation
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        tables = []
        if "players" in query:
            tables.append("players")
        if "player_game_stats" in query:
            tables.append("player_game_stats")
        if "games" in query:
            tables.append("games")
        return tables
    
    def _determine_query_type(self, query: str) -> str:
        """Determine type of query"""
        if query.strip().upper().startswith("SELECT"):
            return "SELECT"
        elif query.strip().upper().startswith("INSERT"):
            return "INSERT"
        elif query.strip().upper().startswith("UPDATE"):
            return "UPDATE"
        elif query.strip().upper().startswith("DELETE"):
            return "DELETE"
        else:
            return "UNKNOWN"
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text"""
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        return None