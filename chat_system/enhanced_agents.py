#!/usr/bin/env python3
"""
Fantasy Football Database - Enhanced Multi-Agent Chat System
Powered by GPT-4o with Pydantic structured reasoning
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load .env file
load_env()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Rich console
console = Console()

# Initialize OpenAI client (will be set properly when needed)
client = None

def get_openai_client():
    """Get or initialize OpenAI client"""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = AsyncOpenAI(api_key=api_key)
    return client

class DatabaseManager:
    """Enhanced database manager with connection pooling and error handling"""
    
    def __init__(self, db_url: str = "sqlite:///data/fantasy_ppr.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url, future=True, pool_pre_ping=True)
    
    async def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            console.print(f"[red]Query error: {e}[/red]")
            return pd.DataFrame()
    
    async def get_player_stats(self, player_name: str, season: int = 2024) -> Dict:
        """Get comprehensive player statistics"""
        query = """
        SELECT p.name, p.position, p.team,
               pgs.fantasy_points, pgs.pass_yards, pgs.rush_yards, pgs.rec_yards,
               pas.snap_share, pas.target_share, pas.air_yards, pas.aDOT, pas.yac,
               pas.epa_per_play, pas.cpoe
        FROM players p
        LEFT JOIN player_game_stats pgs ON p.player_id = pgs.player_id
        LEFT JOIN games g ON pgs.game_id = g.game_id
        LEFT JOIN player_advanced_stats pas ON pgs.game_id = pas.game_id AND pgs.player_id = pas.player_id
        WHERE p.name LIKE :player_name AND g.season = :season
        ORDER BY g.week
        """
        
        df = await self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records') if not df.empty else {}
    
    async def get_route_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get route running data for player"""
        query = """
        SELECT player_name, position, team, routes_run, route_participation,
               route_grade, yards_per_route_run, contested_catch_rate,
               slot_rate, wide_rate
        FROM player_route_stats
        WHERE player_name LIKE :player_name AND season = :season
        """
        
        df = await self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}
    
    async def get_scheme_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get scheme splits data for player"""
        query = """
        SELECT player_name, position, team,
               man_routes_run, man_route_grade, man_yards_per_route_run,
               zone_routes_run, zone_route_grade, zone_yards_per_route_run,
               yprr_man_vs_zone_diff
        FROM player_scheme_stats
        WHERE player_name LIKE :player_name AND season = :season
        """
        
        df = await self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}
    
    async def get_market_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get market data for player"""
        query = """
        SELECT p.name, p.position, p.team,
               fm.ecr_rank, fm.adp_overall, fm.adp_position
        FROM players p
        LEFT JOIN fantasy_market fm ON p.player_id = fm.player_id
        LEFT JOIN games g ON fm.game_id = g.game_id
        WHERE p.name LIKE :player_name AND g.season = :season
        """
        
        df = await self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}

# Pydantic Models for Structured Reasoning

class QueryAnalysis(BaseModel):
    """Analysis of user query intent and required data"""
    query_type: str = Field(..., description="Type of query: data, route, market, strategy, analytics, comparison")
    primary_players: List[str] = Field(default_factory=list, description="Primary players mentioned in query")
    secondary_players: List[str] = Field(default_factory=list, description="Secondary players for comparison")
    metrics_requested: List[str] = Field(default_factory=list, description="Specific metrics requested")
    season: Optional[int] = Field(default=2024, description="Season requested")
    reasoning: str = Field(..., description="Step-by-step reasoning for query classification")
    
    @field_validator('query_type')
    @classmethod
    def validate_query_type(cls, v):
        valid_types = ['data', 'route', 'market', 'strategy', 'analytics', 'comparison']
        if v not in valid_types:
            raise ValueError(f'Query type must be one of {valid_types}')
        return v

class PlayerAnalysis(BaseModel):
    """Structured player analysis with reasoning"""
    player_name: str
    position: Optional[str] = None
    team: Optional[str] = None
    
    # Performance metrics
    fantasy_points_avg: Optional[float] = None
    route_participation: Optional[float] = None
    yards_per_route_run: Optional[float] = None
    route_grade: Optional[float] = None
    
    # Scheme analysis
    man_yprr: Optional[float] = None
    zone_yprr: Optional[float] = None
    scheme_preference: Optional[str] = None
    
    # Market data
    adp_overall: Optional[int] = None
    ecr_rank: Optional[int] = None
    
    # Analysis
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Step-by-step reasoning for analysis")

class ComparisonAnalysis(BaseModel):
    """Structured comparison between players"""
    player1: PlayerAnalysis
    player2: PlayerAnalysis
    comparison_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    winner_by_metric: Dict[str, str] = Field(default_factory=dict)
    overall_recommendation: str = Field(..., description="Overall recommendation with reasoning")
    reasoning: str = Field(..., description="Step-by-step reasoning for comparison")

class RouteAnalysis(BaseModel):
    """Structured route running analysis"""
    top_performers: List[Dict[str, Any]] = Field(default_factory=list)
    average_metrics: Dict[str, float] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Step-by-step reasoning for route analysis")

class MarketAnalysis(BaseModel):
    """Structured market analysis"""
    adp_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    value_picks: List[Dict[str, Any]] = Field(default_factory=list)
    market_trends: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Step-by-step reasoning for market analysis")

class StrategyRecommendation(BaseModel):
    """Structured strategy recommendation"""
    strategy_type: str = Field(..., description="Type of strategy: draft, trade, start_sit")
    recommendations: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Step-by-step reasoning for strategy")
    risk_assessment: str = Field(..., description="Risk assessment of recommendations")
    confidence_level: str = Field(..., description="Confidence level in recommendations")

class AgentResponse(BaseModel):
    """Structured agent response with reasoning"""
    query_analysis: QueryAnalysis
    data_retrieved: Dict[str, Any] = Field(default_factory=dict)
    analysis_results: Union[PlayerAnalysis, ComparisonAnalysis, RouteAnalysis, MarketAnalysis, StrategyRecommendation]
    response_text: str = Field(..., description="Formatted response for user")
    reasoning_summary: str = Field(..., description="Summary of reasoning process")

class EnhancedQueryRouter:
    """Enhanced query router using GPT-4o for intelligent classification"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.agents = {
            'data': DataAgent(db_manager),
            'route': RouteAnalysisAgent(db_manager),
            'market': MarketAgent(db_manager),
            'strategy': StrategyAgent(db_manager),
            'analytics': AnalyticsAgent(db_manager),
            'comparison': ComparisonAgent(db_manager)
        }
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Use GPT-4o to analyze query intent and structure"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        system_prompt = """
        You are an expert fantasy football analyst. Analyze the user query and classify it into the appropriate category.
        
        Query types:
        - data: Basic player statistics and information
        - route: Route running analysis, YPRR, route grades
        - market: ADP, ECR, draft value, market trends
        - strategy: Draft strategy, trade advice, start/sit decisions
        - analytics: Trend analysis, correlations, statistical insights
        - comparison: Direct player comparisons
        
        Provide step-by-step reasoning for your classification.
        
        Return your response as a JSON object with the following structure:
        {
            "query_type": "data|route|market|strategy|analytics|comparison",
            "primary_players": ["player1", "player2"],
            "secondary_players": [],
            "metrics_requested": [],
            "season": 2024,
            "reasoning": "Step-by-step reasoning for classification"
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this query and return JSON: {query}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        analysis_data = json.loads(response.choices[0].message.content)
        return QueryAnalysis(**analysis_data)
    
    async def process_query(self, query: str) -> AgentResponse:
        """Process user query with full reasoning"""
        
        with console.status("[bold green]Analyzing query...", spinner="dots"):
            query_analysis = await self.analyze_query(query)
        
        console.print(f"[blue]Query Analysis:[/blue] {query_analysis.query_type}")
        console.print(f"[blue]Reasoning:[/blue] {query_analysis.reasoning}")
        
        # Route to appropriate agent
        agent = self.agents.get(query_analysis.query_type, self.agents['data'])
        
        with console.status("[bold green]Processing with specialized agent...", spinner="dots"):
            response = await agent.process(query, query_analysis)
        
        return response

class DataAgent:
    """Enhanced data agent with structured reasoning"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process data-related queries with reasoning"""
        
        reasoning_steps = []
        
        # Step 1: Extract player information
        reasoning_steps.append("Step 1: Extracting player information from query analysis")
        primary_player = query_analysis.primary_players[0] if query_analysis.primary_players else None
        
        if not primary_player:
            return AgentResponse(
                query_analysis=query_analysis,
                response_text="I couldn't identify a specific player in your query. Please try asking about a specific player.",
                reasoning_summary="No player identified in query",
                analysis_results=PlayerAnalysis(player_name="Unknown", reasoning="No player specified")
            )
        
        # Step 2: Retrieve comprehensive data
        reasoning_steps.append(f"Step 2: Retrieving comprehensive data for {primary_player}")
        
        stats = await self.db_manager.get_player_stats(primary_player, query_analysis.season)
        route_data = await self.db_manager.get_route_data(primary_player, query_analysis.season)
        scheme_data = await self.db_manager.get_scheme_data(primary_player, query_analysis.season)
        market_data = await self.db_manager.get_market_data(primary_player, query_analysis.season)
        
        # Step 3: Analyze data and generate insights
        reasoning_steps.append("Step 3: Analyzing data and generating insights")
        
        analysis = await self._analyze_player_data(
            primary_player, stats, route_data, scheme_data, market_data
        )
        
        # Step 4: Format response
        reasoning_steps.append("Step 4: Formatting comprehensive response")
        
        response_text = self._format_player_response(analysis)
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={
                "stats": stats,
                "route_data": route_data,
                "scheme_data": scheme_data,
                "market_data": market_data
            },
            analysis_results=analysis,
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )
    
    async def _analyze_player_data(self, player_name: str, stats: List[Dict], 
                                 route_data: Dict, scheme_data: Dict, market_data: Dict) -> PlayerAnalysis:
        """Analyze player data using GPT-4o for insights"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Prepare data summary for AI analysis
        data_summary = {
            "player_name": player_name,
            "stats_count": len(stats),
            "route_data": route_data,
            "scheme_data": scheme_data,
            "market_data": market_data
        }
        
        if stats:
            avg_fantasy_points = sum(s.get('fantasy_points', 0) for s in stats if s.get('fantasy_points')) / len(stats)
            data_summary["avg_fantasy_points"] = avg_fantasy_points
        
        system_prompt = """
        You are an expert fantasy football analyst. Analyze the provided player data and provide structured insights.
        Focus on identifying strengths, weaknesses, and actionable insights.
        Provide step-by-step reasoning for your analysis.
        
        Return your response as a JSON object with the following structure:
        {
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "insights": ["insight1", "insight2"],
            "reasoning": "Step-by-step reasoning for analysis"
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this player data and return JSON: {json.dumps(data_summary, indent=2)}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        analysis_data = json.loads(response.choices[0].message.content)
        
        return PlayerAnalysis(
            player_name=player_name,
            position=route_data.get('position') if route_data else None,
            team=route_data.get('team') if route_data else None,
            fantasy_points_avg=data_summary.get("avg_fantasy_points"),
            route_participation=route_data.get('route_participation') if route_data else None,
            yards_per_route_run=route_data.get('yards_per_route_run') if route_data else None,
            route_grade=route_data.get('route_grade') if route_data else None,
            man_yprr=scheme_data.get('man_yards_per_route_run') if scheme_data else None,
            zone_yprr=scheme_data.get('zone_yards_per_route_run') if scheme_data else None,
            adp_overall=market_data.get('adp_overall') if market_data else None,
            ecr_rank=market_data.get('ecr_rank') if market_data else None,
            strengths=analysis_data.get('strengths', []),
            weaknesses=analysis_data.get('weaknesses', []),
            insights=analysis_data.get('insights', []),
            reasoning=analysis_data.get('reasoning', '')
        )
    
    def _format_player_response(self, analysis: PlayerAnalysis) -> str:
        """Format player analysis into readable response"""
        
        response = f"📊 **{analysis.player_name} Analysis**\n\n"
        
        if analysis.position and analysis.team:
            response += f"**Player Info:** {analysis.position} - {analysis.team}\n\n"
        
        if analysis.fantasy_points_avg:
            response += f"**Performance:** Average {analysis.fantasy_points_avg:.1f} fantasy points\n\n"
        
        if analysis.yards_per_route_run:
            response += f"**Route Running:**\n"
            response += f"• YPRR: {analysis.yards_per_route_run:.2f}\n"
            if analysis.route_participation:
                response += f"• Route Participation: {analysis.route_participation:.1f}%\n"
            if analysis.route_grade:
                response += f"• Route Grade: {analysis.route_grade:.1f}\n"
            response += "\n"
        
        if analysis.man_yprr or analysis.zone_yprr:
            response += f"**Scheme Analysis:**\n"
            if analysis.man_yprr:
                response += f"• Man Coverage YPRR: {analysis.man_yprr:.2f}\n"
            if analysis.zone_yprr:
                response += f"• Zone Coverage YPRR: {analysis.zone_yprr:.2f}\n"
            response += "\n"
        
        if analysis.adp_overall or analysis.ecr_rank:
            response += f"**Market Data:**\n"
            if analysis.adp_overall:
                response += f"• ADP: {analysis.adp_overall}\n"
            if analysis.ecr_rank:
                response += f"• ECR Rank: {analysis.ecr_rank}\n"
            response += "\n"
        
        if analysis.strengths:
            response += f"**Strengths:**\n"
            for strength in analysis.strengths:
                response += f"• {strength}\n"
            response += "\n"
        
        if analysis.insights:
            response += f"**Key Insights:**\n"
            for insight in analysis.insights:
                response += f"• {insight}\n"
        
        return response

class RouteAnalysisAgent:
    """Enhanced route analysis agent"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process route analysis queries"""
        
        reasoning_steps = []
        reasoning_steps.append("Step 1: Analyzing route running query intent")
        
        # Step 2: Retrieve route data
        reasoning_steps.append("Step 2: Retrieving comprehensive route running data")
        
        query = """
        SELECT player_name, team, position, route_grade, yards_per_route_run, 
               route_participation, contested_catch_rate
        FROM player_route_stats
        WHERE season = 2024 AND position = 'WR'
        ORDER BY yards_per_route_run DESC
        LIMIT 20
        """
        
        df = await self.db_manager.execute_query(query)
        
        # Step 3: Analyze with AI
        reasoning_steps.append("Step 3: Analyzing route running patterns with AI")
        
        analysis = await self._analyze_route_data(df)
        
        # Step 4: Format response
        reasoning_steps.append("Step 4: Formatting route analysis response")
        
        response_text = self._format_route_response(analysis)
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={"route_data": df.to_dict('records')},
            analysis_results=analysis,
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )
    
    async def _analyze_route_data(self, df: pd.DataFrame) -> RouteAnalysis:
        """Analyze route data using GPT-4o"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        data_summary = {
            "top_performers": df.head(10).to_dict('records'),
            "average_metrics": {
                "avg_yprr": df['yards_per_route_run'].mean(),
                "avg_route_grade": df['route_grade'].mean(),
                "avg_participation": df['route_participation'].mean()
            }
        }
        
        system_prompt = """
        You are an expert route running analyst. Analyze the provided route running data and provide insights.
        Focus on identifying patterns, trends, and actionable recommendations.
        Provide step-by-step reasoning for your analysis.
        
        Return your response as a JSON object with the following structure:
        {
            "insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "reasoning": "Step-by-step reasoning for analysis"
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this route running data and return JSON: {json.dumps(data_summary, indent=2)}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        analysis_data = json.loads(response.choices[0].message.content)
        
        return RouteAnalysis(
            top_performers=data_summary["top_performers"],
            average_metrics=data_summary["average_metrics"],
            insights=analysis_data.get('insights', []),
            recommendations=analysis_data.get('recommendations', []),
            reasoning=analysis_data.get('reasoning', '')
        )
    
    def _format_route_response(self, analysis: RouteAnalysis) -> str:
        """Format route analysis into readable response"""
        
        response = "🏃 **Top Route Runners Analysis**\n\n"
        
        response += "**Top Performers:**\n"
        for i, player in enumerate(analysis.top_performers[:10], 1):
            response += f"{i}. **{player['player_name']}** ({player['team']}): "
            response += f"{player['yards_per_route_run']:.2f} YPRR, {player['route_grade']:.1f} grade\n"
        
        response += "\n**Average Metrics:**\n"
        for metric, value in analysis.average_metrics.items():
            response += f"• {metric.replace('_', ' ').title()}: {value:.2f}\n"
        
        if analysis.insights:
            response += "\n**Key Insights:**\n"
            for insight in analysis.insights:
                response += f"• {insight}\n"
        
        if analysis.recommendations:
            response += "\n**Recommendations:**\n"
            for rec in analysis.recommendations:
                response += f"• {rec}\n"
        
        return response

class MarketAgent:
    """Enhanced market analysis agent"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process market analysis queries"""
        
        reasoning_steps = []
        reasoning_steps.append("Step 1: Analyzing market query intent")
        
        # Step 2: Retrieve market data
        reasoning_steps.append("Step 2: Retrieving ADP and market data")
        
        query = """
        SELECT p.name, p.position, p.team, fm.adp_overall, fm.adp_position, fm.ecr_rank
        FROM players p
        JOIN fantasy_market fm ON p.player_id = fm.player_id
        JOIN games g ON fm.game_id = g.game_id
        WHERE g.season = 2024 AND fm.adp_overall IS NOT NULL
        ORDER BY fm.adp_overall
        LIMIT 20
        """
        
        df = await self.db_manager.execute_query(query)
        
        # Step 3: Analyze with AI
        reasoning_steps.append("Step 3: Analyzing market trends with AI")
        
        analysis = await self._analyze_market_data(df)
        
        # Step 4: Format response
        reasoning_steps.append("Step 4: Formatting market analysis response")
        
        response_text = self._format_market_response(analysis)
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={"market_data": df.to_dict('records')},
            analysis_results=analysis,
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )
    
    async def _analyze_market_data(self, df: pd.DataFrame) -> MarketAnalysis:
        """Analyze market data using GPT-4o"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        data_summary = {
            "adp_rankings": df.head(15).to_dict('records'),
            "positional_breakdown": df.groupby('position').size().to_dict()
        }
        
        system_prompt = """
        You are an expert fantasy football market analyst. Analyze the provided ADP and market data.
        Focus on identifying value picks, market trends, and strategic recommendations.
        Provide step-by-step reasoning for your analysis.
        
        Return your response as a JSON object with the following structure:
        {
            "value_picks": ["pick1", "pick2"],
            "market_trends": ["trend1", "trend2"],
            "recommendations": ["rec1", "rec2"],
            "reasoning": "Step-by-step reasoning for analysis"
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this market data and return JSON: {json.dumps(data_summary, indent=2)}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        analysis_data = json.loads(response.choices[0].message.content)
        
        return MarketAnalysis(
            adp_rankings=data_summary["adp_rankings"],
            value_picks=analysis_data.get('value_picks', []),
            market_trends=analysis_data.get('market_trends', []),
            recommendations=analysis_data.get('recommendations', []),
            reasoning=analysis_data.get('reasoning', '')
        )
    
    def _format_market_response(self, analysis: MarketAnalysis) -> str:
        """Format market analysis into readable response"""
        
        response = "💰 **Market Analysis**\n\n"
        
        response += "**Top ADP Rankings:**\n"
        for i, player in enumerate(analysis.adp_rankings[:10], 1):
            response += f"{i}. **{player['name']}** ({player['team']}): "
            response += f"ADP {player['adp_overall']}, Position {player['adp_position']}\n"
        
        if analysis.value_picks:
            response += "\n**Value Picks:**\n"
            for pick in analysis.value_picks:
                response += f"• {pick}\n"
        
        if analysis.market_trends:
            response += "\n**Market Trends:**\n"
            for trend in analysis.market_trends:
                response += f"• {trend}\n"
        
        if analysis.recommendations:
            response += "\n**Recommendations:**\n"
            for rec in analysis.recommendations:
                response += f"• {rec}\n"
        
        return response

class StrategyAgent:
    """Enhanced strategy agent"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process strategy queries"""
        
        reasoning_steps = []
        reasoning_steps.append("Step 1: Analyzing strategy query intent")
        
        # Step 2: Determine strategy type
        strategy_type = "draft"  # Default, could be enhanced with NLP
        if "trade" in query.lower():
            strategy_type = "trade"
        elif "start" in query.lower() or "sit" in query.lower():
            strategy_type = "start_sit"
        
        reasoning_steps.append(f"Step 2: Identified strategy type: {strategy_type}")
        
        # Step 3: Generate strategy with AI
        reasoning_steps.append("Step 3: Generating strategic recommendations with AI")
        
        strategy = await self._generate_strategy(query, strategy_type)
        
        # Step 4: Format response
        reasoning_steps.append("Step 4: Formatting strategy response")
        
        response_text = self._format_strategy_response(strategy)
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={"strategy_type": strategy_type},
            analysis_results=strategy,
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )
    
    async def _generate_strategy(self, query: str, strategy_type: str) -> StrategyRecommendation:
        """Generate strategy using GPT-4o"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        system_prompt = f"""
        You are an expert fantasy football strategist. Generate {strategy_type} strategy recommendations.
        Focus on actionable advice with clear reasoning and risk assessment.
        Provide step-by-step reasoning for your recommendations.
        
        Return your response as a JSON object with the following structure:
        {{
            "recommendations": ["rec1", "rec2"],
            "reasoning": "Step-by-step reasoning for recommendations",
            "risk_assessment": "Risk assessment of recommendations",
            "confidence_level": "High|Medium|Low"
        }}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {strategy_type} strategy and return JSON for: {query}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.4
        )
        
        strategy_data = json.loads(response.choices[0].message.content)
        
        return StrategyRecommendation(
            strategy_type=strategy_type,
            recommendations=strategy_data.get('recommendations', []),
            reasoning=strategy_data.get('reasoning', ''),
            risk_assessment=strategy_data.get('risk_assessment', ''),
            confidence_level=strategy_data.get('confidence_level', '')
        )
    
    def _format_strategy_response(self, strategy: StrategyRecommendation) -> str:
        """Format strategy into readable response"""
        
        response = f"🎯 **{strategy.strategy_type.replace('_', ' ').title()} Strategy**\n\n"
        
        response += "**Recommendations:**\n"
        for i, rec in enumerate(strategy.recommendations, 1):
            response += f"{i}. {rec}\n"
        
        response += f"\n**Reasoning:**\n{strategy.reasoning}\n"
        
        response += f"\n**Risk Assessment:**\n{strategy.risk_assessment}\n"
        
        response += f"\n**Confidence Level:** {strategy.confidence_level}\n"
        
        return response

class ComparisonAgent:
    """Enhanced comparison agent"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process comparison queries"""
        
        reasoning_steps = []
        reasoning_steps.append("Step 1: Analyzing comparison query intent")
        
        if len(query_analysis.primary_players) < 2:
            return AgentResponse(
                query_analysis=query_analysis,
                response_text="I need at least two players to compare. Please specify both players.",
                reasoning_summary="Insufficient players for comparison",
                analysis_results=ComparisonAnalysis(
                    player1=PlayerAnalysis(player_name="Unknown", reasoning="No player 1"),
                    player2=PlayerAnalysis(player_name="Unknown", reasoning="No player 2"),
                    overall_recommendation="Cannot compare without two players"
                )
            )
        
        player1, player2 = query_analysis.primary_players[0], query_analysis.primary_players[1]
        
        # Step 2: Retrieve data for both players
        reasoning_steps.append(f"Step 2: Retrieving data for {player1} and {player2}")
        
        player1_data = await self._get_comprehensive_data(player1, query_analysis.season)
        player2_data = await self._get_comprehensive_data(player2, query_analysis.season)
        
        # Step 3: Generate comparison with AI
        reasoning_steps.append("Step 3: Generating comprehensive comparison with AI")
        
        comparison = await self._generate_comparison(player1, player2, player1_data, player2_data)
        
        # Step 4: Format response
        reasoning_steps.append("Step 4: Formatting comparison response")
        
        response_text = self._format_comparison_response(comparison)
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={"player1_data": player1_data, "player2_data": player2_data},
            analysis_results=comparison,
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )
    
    async def _get_comprehensive_data(self, player_name: str, season: int) -> Dict:
        """Get comprehensive data for a player"""
        stats = await self.db_manager.get_player_stats(player_name, season)
        route_data = await self.db_manager.get_route_data(player_name, season)
        scheme_data = await self.db_manager.get_scheme_data(player_name, season)
        market_data = await self.db_manager.get_market_data(player_name, season)
        
        return {
            "stats": stats,
            "route_data": route_data,
            "scheme_data": scheme_data,
            "market_data": market_data
        }
    
    async def _generate_comparison(self, player1: str, player2: str, 
                                 player1_data: Dict, player2_data: Dict) -> ComparisonAnalysis:
        """Generate comparison using GPT-4o"""
        
        # Get OpenAI client
        client = get_openai_client()
        
        comparison_data = {
            "player1": {"name": player1, "data": player1_data},
            "player2": {"name": player2, "data": player2_data}
        }
        
        system_prompt = """
        You are an expert fantasy football analyst. Compare two players comprehensively.
        Focus on key metrics, strengths/weaknesses, and provide a clear recommendation.
        Provide step-by-step reasoning for your comparison.
        
        Return your response as a JSON object with the following structure:
        {
            "comparison_metrics": {"metric1": "comparison", "metric2": "comparison"},
            "winner_by_metric": {"category1": "player1", "category2": "player2"},
            "overall_recommendation": "Overall recommendation with reasoning",
            "reasoning": "Step-by-step reasoning for comparison"
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compare these players and return JSON: {json.dumps(comparison_data, indent=2)}"}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        comparison_data = json.loads(response.choices[0].message.content)
        
        return ComparisonAnalysis(
            player1=PlayerAnalysis(player_name=player1, reasoning="Comparison analysis"),
            player2=PlayerAnalysis(player_name=player2, reasoning="Comparison analysis"),
            comparison_metrics=comparison_data.get('comparison_metrics', {}),
            winner_by_metric=comparison_data.get('winner_by_metric', {}),
            overall_recommendation=comparison_data.get('overall_recommendation', ''),
            reasoning=comparison_data.get('reasoning', '')
        )
    
    def _format_comparison_response(self, comparison: ComparisonAnalysis) -> str:
        """Format comparison into readable response"""
        
        response = f"📊 **{comparison.player1.player_name} vs {comparison.player2.player_name}**\n\n"
        
        response += "**Comparison by Metric:**\n"
        for metric, data in comparison.comparison_metrics.items():
            response += f"• {metric}: {data}\n"
        
        response += "\n**Winner by Category:**\n"
        for category, winner in comparison.winner_by_metric.items():
            response += f"• {category}: {winner}\n"
        
        response += f"\n**Overall Recommendation:**\n{comparison.overall_recommendation}\n"
        
        response += f"\n**Reasoning:**\n{comparison.reasoning}\n"
        
        return response

class AnalyticsAgent:
    """Enhanced analytics agent for trend analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def process(self, query: str, query_analysis: QueryAnalysis) -> AgentResponse:
        """Process analytics queries"""
        
        reasoning_steps = []
        reasoning_steps.append("Step 1: Analyzing analytics query intent")
        
        # This would be enhanced with specific analytics capabilities
        response_text = "📈 **Analytics Analysis**\n\nAdvanced analytics capabilities coming soon!"
        
        return AgentResponse(
            query_analysis=query_analysis,
            data_retrieved={},
            analysis_results=RouteAnalysis(
                top_performers=[],
                average_metrics={},
                insights=["Analytics agent under development"],
                recommendations=["Enhanced analytics coming in next version"],
                reasoning="Analytics agent placeholder"
            ),
            response_text=response_text,
            reasoning_summary="\n".join(reasoning_steps)
        )

# CLI Interface

@click.group()
def cli():
    """Fantasy Football Database - Enhanced Multi-Agent Chat System"""
    pass

@cli.command()
@click.option('--db', default='ppr', help='Database type (ppr, halfppr, standard)')
@click.option('--verbose', is_flag=True, help='Show detailed reasoning')
def chat(db, verbose):
    """Start interactive chat session"""
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    db_url = f"sqlite:///../fantasy-etl/data/fantasy_{db}.db"
    
    if not Path(f"../fantasy-etl/data/fantasy_{db}.db").exists():
        console.print(f"[red]Error: Database {db_url} not found[/red]")
        return
    
    console.print(Panel.fit(
        "[bold blue]🏈 Fantasy Football Database - Enhanced Multi-Agent Chat System[/bold blue]\n"
        f"[green]Using database: {db}[/green]\n"
        "[yellow]Powered by GPT-4o with structured reasoning[/yellow]",
        border_style="blue"
    ))
    
    db_manager = DatabaseManager(db_url)
    router = EnhancedQueryRouter(db_manager)
    
    console.print("\n[bold]Example queries:[/bold]")
    console.print("• Tell me about Justin Jefferson")
    console.print("• Compare CeeDee Lamb vs Tyreek Hill")
    console.print("• Who are the best route runners?")
    console.print("• What's the ADP analysis?")
    console.print("• Give me draft strategy advice")
    console.print("\nType 'quit' to exit\n")
    
    while True:
        try:
            query = Prompt.ask("[bold green]You[/bold green]")
            
            if query.lower() in ['quit', 'exit', 'bye']:
                console.print("[yellow]Thanks for using the Enhanced Fantasy Football Chat System![/yellow]")
                break
            
            if not query.strip():
                continue
            
            # Process query
            response = asyncio.run(router.process_query(query))
            
            # Display response
            console.print("\n[bold blue]🤖 Assistant[/bold blue]")
            
            if verbose:
                console.print(f"[dim]Query Analysis: {response.query_analysis.query_type}[/dim]")
                console.print(f"[dim]Reasoning Summary:[/dim]\n[dim]{response.reasoning_summary}[/dim]\n")
            
            console.print(Markdown(response.response_text))
            console.print("\n" + "─" * 80 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Thanks for using the Enhanced Fantasy Football Chat System![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("Please try again.\n")

@cli.command()
@click.argument('query')
@click.option('--db', default='ppr', help='Database type (ppr, halfppr, standard)')
@click.option('--verbose', is_flag=True, help='Show detailed reasoning')
def ask(query, db, verbose):
    """Ask a single question"""
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        return
    
    db_url = f"sqlite:///../fantasy-etl/data/fantasy_{db}.db"
    
    if not Path(f"../fantasy-etl/data/fantasy_{db}.db").exists():
        console.print(f"[red]Error: Database {db_url} not found[/red]")
        return
    
    db_manager = DatabaseManager(db_url)
    router = EnhancedQueryRouter(db_manager)
    
    console.print(f"[bold]Query:[/bold] {query}\n")
    
    try:
        response = asyncio.run(router.process_query(query))
        
        if verbose:
            console.print(f"[dim]Query Analysis: {response.query_analysis.query_type}[/dim]")
            console.print(f"[dim]Reasoning Summary:[/dim]\n[dim]{response.reasoning_summary}[/dim]\n")
        
        console.print(Markdown(response.response_text))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    cli()
