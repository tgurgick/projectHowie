#!/usr/bin/env python3
"""
Strength of Schedule Tools for Howie CLI

Tools for analyzing team and position-specific strength of schedule data.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..core.base_tool import BaseTool, ToolParameter, ToolResult, ToolStatus

class StrengthOfScheduleTool(BaseTool):
    """Tool to analyze strength of schedule for teams and positions"""
    
    def __init__(self):
        self.name = "strength_of_schedule"
        self.description = "Analyze strength of schedule for teams by position and time period"
        self.parameters = [
            ToolParameter(
                name="team", 
                type="string", 
                description="Team abbreviation (e.g., 'KC', 'BUF')", 
                required=False
            ),
            ToolParameter(
                name="position", 
                type="string", 
                description="Position type: qb, rb, wr, te, dst", 
                required=False,
                default="qb"
            ),
            ToolParameter(
                name="season", 
                type="integer", 
                description="Season year", 
                required=False,
                default=2025
            ),
            ToolParameter(
                name="weeks", 
                type="string", 
                description="Week range: 'season', 'playoffs', 'all', or specific weeks like '1-4'", 
                required=False,
                default="season"
            ),
            ToolParameter(
                name="limit", 
                type="integer", 
                description="Number of teams to return (for rankings)", 
                required=False,
                default=32
            )
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the strength of schedule analysis"""
        try:
            # Get parameters
            team = kwargs.get('team', '').upper()
            position = kwargs.get('position', 'qb').lower()
            season = int(kwargs.get('season', 2025))
            weeks = kwargs.get('weeks', 'season').lower()
            limit = int(kwargs.get('limit', 32))
            
            # Validate position
            valid_positions = ['qb', 'rb', 'wr', 'te', 'dst']
            if position not in valid_positions:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Invalid position '{position}'. Must be one of: {', '.join(valid_positions)}"
                )
            
            # Database connection
            db_path = Path(__file__).parent.parent.parent / "data" / "fantasy_ppr.db"
            if not db_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Database not found: {db_path}"
                )
            
            conn = sqlite3.connect(db_path)
            
            # Build query based on weeks parameter
            if weeks == 'season':
                order_col = 'season_sos'
                select_cols = 'team, season_sos as sos_value, season_games as games'
                desc = 'Season'
            elif weeks == 'playoffs':
                order_col = 'playoffs_sos'
                select_cols = 'team, playoffs_sos as sos_value, playoffs_games as games'
                desc = 'Playoffs'
            elif weeks == 'all':
                order_col = 'all_sos'
                select_cols = 'team, all_sos as sos_value, all_games as games'
                desc = 'All'
            elif '-' in weeks:
                # Specific week range like "1-4"
                try:
                    start_week, end_week = map(int, weeks.split('-'))
                    if start_week < 1 or end_week > 17 or start_week > end_week:
                        raise ValueError("Invalid week range")
                    
                    # Calculate average SoS for the week range
                    week_cols = [f'week_{i}' for i in range(start_week, end_week + 1)]
                    avg_expr = f"({' + '.join([f'COALESCE({col}, 0)' for col in week_cols])}) / {len(week_cols)}"
                    select_cols = f'team, {avg_expr} as sos_value, {end_week - start_week + 1} as games'
                    order_col = 'sos_value'
                    desc = f'Weeks {start_week}-{end_week}'
                except (ValueError, IndexError):
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"Invalid week range '{weeks}'. Use format like '1-4' or 'season', 'playoffs', 'all'"
                    )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Invalid weeks parameter '{weeks}'. Use 'season', 'playoffs', 'all', or week range like '1-4'"
                )
            
            # Execute query
            if team:
                # Specific team query
                query = f"""
                    SELECT {select_cols}
                    FROM strength_of_schedule 
                    WHERE season = ? AND position = ? AND team = ?
                """
                params = [season, position, team]
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"No SoS data found for {team} {position.upper()} in {season}"
                    )
                
                result = {
                    'team': rows[0][0],
                    'position': position.upper(),
                    'season': season,
                    'period': desc,
                    'sos_value': round(rows[0][1], 2) if rows[0][1] else None,
                    'games': rows[0][2]
                }
                
                # Get weekly breakdown if available
                if weeks in ['season', 'playoffs', 'all']:
                    weekly_query = """
                        SELECT week_1, week_2, week_3, week_4, week_5, week_6, week_7, week_8, week_9,
                               week_10, week_11, week_12, week_13, week_14, week_15, week_16, week_17
                        FROM strength_of_schedule 
                        WHERE season = ? AND position = ? AND team = ?
                    """
                    cursor = conn.execute(weekly_query, [season, position, team])
                    weekly_row = cursor.fetchone()
                    if weekly_row:
                        result['weekly_sos'] = {
                            f'week_{i+1}': round(val, 2) if val else None 
                            for i, val in enumerate(weekly_row) if val
                        }
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message=f"SoS data for {team} {position.upper()}: {result['sos_value']} ({desc})"
                )
                
            else:
                # Rankings query
                query = f"""
                    SELECT {select_cols}
                    FROM strength_of_schedule 
                    WHERE season = ? AND position = ? AND {order_col} IS NOT NULL
                    ORDER BY {order_col} ASC
                    LIMIT ?
                """
                params = [season, position, limit]
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"No SoS data found for {position.upper()} in {season}"
                    )
                
                # Add rankings
                df['rank'] = range(1, len(df) + 1)
                df['sos_value'] = df['sos_value'].round(2)
                
                result = {
                    'position': position.upper(),
                    'season': season,
                    'period': desc,
                    'total_teams': len(df),
                    'rankings': df.to_dict('records')
                }
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message=f"SoS rankings for {position.upper()} ({desc}): {len(df)} teams"
                )
                
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Error analyzing strength of schedule: {str(e)}"
            )
        finally:
            if 'conn' in locals():
                conn.close()

class TeamScheduleAnalysisTool(BaseTool):
    """Tool to analyze a team's schedule strength across all positions"""
    
    def __init__(self):
        self.name = "team_schedule_analysis"
        self.description = "Comprehensive schedule analysis for a team across all positions"
        self.parameters = [
            ToolParameter(
                name="team", 
                type="string", 
                description="Team abbreviation (e.g., 'KC', 'BUF')", 
                required=True
            ),
            ToolParameter(
                name="season", 
                type="integer", 
                description="Season year", 
                required=False,
                default=2025
            )
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute comprehensive team schedule analysis"""
        try:
            team = kwargs.get('team', '').upper()
            season = int(kwargs.get('season', 2025))
            
            if not team:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message="Team parameter is required"
                )
            
            # Database connection
            db_path = Path(__file__).parent.parent.parent / "data" / "fantasy_ppr.db"
            if not db_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Database not found: {db_path}"
                )
            
            conn = sqlite3.connect(db_path)
            
            # Get SoS data for all positions
            query = """
                SELECT position, season_sos, playoffs_sos, all_sos,
                       season_games, playoffs_games, all_games
                FROM strength_of_schedule 
                WHERE season = ? AND team = ?
                ORDER BY position
            """
            
            df = pd.read_sql_query(query, conn, params=[season, team])
            
            if df.empty:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"No SoS data found for {team} in {season}"
                )
            
            # Calculate average SoS across positions
            avg_season_sos = df['season_sos'].mean()
            avg_playoffs_sos = df['playoffs_sos'].mean()
            avg_all_sos = df['all_sos'].mean()
            
            # Format results
            position_breakdown = []
            for _, row in df.iterrows():
                position_breakdown.append({
                    'position': row['position'].upper(),
                    'season_sos': round(row['season_sos'], 2) if row['season_sos'] else None,
                    'playoffs_sos': round(row['playoffs_sos'], 2) if row['playoffs_sos'] else None,
                    'all_sos': round(row['all_sos'], 2) if row['all_sos'] else None
                })
            
            result = {
                'team': team,
                'season': season,
                'average_sos': {
                    'season': round(avg_season_sos, 2),
                    'playoffs': round(avg_playoffs_sos, 2),
                    'all': round(avg_all_sos, 2)
                },
                'position_breakdown': position_breakdown,
                'total_positions': len(df)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Schedule analysis for {team}: Avg Season SoS = {result['average_sos']['season']}"
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Error analyzing team schedule: {str(e)}"
            )
        finally:
            if 'conn' in locals():
                conn.close()
