#!/usr/bin/env python3
"""
Fantasy Football Database - Chat System Prototype
Simple demonstration of the multi-agent chat system concept
"""

import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import re
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DatabaseManager:
    """Manages database connections and queries"""
    
    def __init__(self, db_url: str = "sqlite:///data/fantasy_ppr.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url, future=True)
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            print(f"Query error: {e}")
            return pd.DataFrame()
    
    def get_player_stats(self, player_name: str, season: int = 2024) -> Dict:
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
        
        df = self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records') if not df.empty else {}
    
    def get_route_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get route running data for player"""
        query = """
        SELECT player_name, position, team, routes_run, route_participation,
               route_grade, yards_per_route_run, contested_catch_rate,
               slot_rate, wide_rate
        FROM player_route_stats
        WHERE player_name LIKE :player_name AND season = :season
        """
        
        df = self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}
    
    def get_scheme_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get scheme splits data for player"""
        query = """
        SELECT player_name, position, team,
               man_routes_run, man_route_grade, man_yards_per_route_run,
               zone_routes_run, zone_route_grade, zone_yards_per_route_run,
               yprr_man_vs_zone_diff
        FROM player_scheme_stats
        WHERE player_name LIKE :player_name AND season = :season
        """
        
        df = self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}
    
    def get_market_data(self, player_name: str, season: int = 2024) -> Dict:
        """Get market data for player"""
        query = """
        SELECT p.name, p.position, p.team,
               fm.ecr_rank, fm.adp_overall, fm.adp_position
        FROM players p
        LEFT JOIN fantasy_market fm ON p.player_id = fm.player_id
        LEFT JOIN games g ON fm.game_id = g.game_id
        WHERE p.name LIKE :player_name AND g.season = :season
        """
        
        df = self.execute_query(query, {"player_name": f"%{player_name}%", "season": season})
        return df.to_dict('records')[0] if not df.empty else {}

class QueryRouter:
    """Routes user queries to appropriate agents"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.agents = {
            'data': DataAgent(db_manager),
            'route': RouteAnalysisAgent(db_manager),
            'market': MarketAgent(db_manager),
            'strategy': StrategyAgent(db_manager)
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type based on keywords"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['route', 'yprr', 'route grade', 'route participation']):
            return 'route'
        elif any(word in query_lower for word in ['adp', 'ecr', 'draft', 'market', 'value']):
            return 'market'
        elif any(word in query_lower for word in ['strategy', 'should i', 'recommend', 'trade', 'start']):
            return 'strategy'
        else:
            return 'data'
    
    def process_query(self, query: str) -> str:
        """Process user query and return response"""
        query_type = self.classify_query(query)
        agent = self.agents[query_type]
        return agent.process(query)

class DataAgent:
    """Handles basic data queries and player statistics"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def process(self, query: str) -> str:
        """Process data-related queries"""
        # Extract player names from query
        player_names = self.extract_player_names(query)
        
        if len(player_names) == 1:
            return self.get_player_analysis(player_names[0])
        elif len(player_names) == 2:
            return self.compare_players(player_names[0], player_names[1])
        else:
            return "I can help you analyze player data. Try asking about a specific player or comparing two players."
    
    def extract_player_names(self, query: str) -> List[str]:
        """Extract player names from query"""
        # Simple extraction - in real system, use NLP
        common_names = [
            'Justin Jefferson', 'CeeDee Lamb', 'Tyreek Hill', 'Davante Adams',
            'Cooper Kupp', 'Ja\'Marr Chase', 'Rome Odunze', 'Brian Thomas Jr',
            'Adam Thielen', 'Jerry Jeudy', 'JuJu Smith-Schuster'
        ]
        
        found_names = []
        for name in common_names:
            if name.lower() in query.lower():
                found_names.append(name)
        
        return found_names
    
    def get_player_analysis(self, player_name: str) -> str:
        """Get comprehensive player analysis"""
        stats = self.db_manager.get_player_stats(player_name)
        route_data = self.db_manager.get_route_data(player_name)
        scheme_data = self.db_manager.get_scheme_data(player_name)
        market_data = self.db_manager.get_market_data(player_name)
        
        if not stats:
            return f"Sorry, I couldn't find data for {player_name}."
        
        # Calculate averages
        avg_fantasy_points = sum(s['fantasy_points'] for s in stats if s['fantasy_points']) / len(stats)
        
        response = f"📊 **{player_name} Analysis**\n\n"
        
        if route_data:
            response += f"**Route Running:**\n"
            response += f"• Routes Run: {route_data.get('routes_run', 'N/A')}\n"
            response += f"• Route Participation: {route_data.get('route_participation', 'N/A'):.1f}%\n"
            response += f"• Route Grade: {route_data.get('route_grade', 'N/A'):.1f}\n"
            response += f"• YPRR: {route_data.get('yards_per_route_run', 'N/A'):.2f}\n"
            response += f"• Slot Rate: {route_data.get('slot_rate', 'N/A'):.1f}%\n"
            response += f"• Wide Rate: {route_data.get('wide_rate', 'N/A'):.1f}%\n\n"
        
        if scheme_data:
            response += f"**Scheme Analysis:**\n"
            response += f"• Man Coverage YPRR: {scheme_data.get('man_yards_per_route_run', 'N/A'):.2f}\n"
            response += f"• Zone Coverage YPRR: {scheme_data.get('zone_yards_per_route_run', 'N/A'):.2f}\n"
            response += f"• Man vs Zone Difference: {scheme_data.get('yprr_man_vs_zone_diff', 'N/A'):.2f}\n\n"
        
        if market_data:
            response += f"**Market Data:**\n"
            response += f"• ECR Rank: {market_data.get('ecr_rank', 'N/A')}\n"
            response += f"• ADP Overall: {market_data.get('adp_overall', 'N/A')}\n"
            response += f"• ADP Position: {market_data.get('adp_position', 'N/A')}\n\n"
        
        response += f"**Performance:**\n"
        response += f"• Average Fantasy Points: {avg_fantasy_points:.1f}\n"
        
        return response
    
    def compare_players(self, player1: str, player2: str) -> str:
        """Compare two players"""
        data1 = self.db_manager.get_route_data(player1)
        data2 = self.db_manager.get_route_data(player2)
        
        if not data1 or not data2:
            return f"Sorry, I couldn't find complete data for both players."
        
        response = f"📊 **{player1} vs {player2} Comparison**\n\n"
        
        # Route running comparison
        response += f"**Route Running:**\n"
        response += f"• {player1}: {data1.get('yards_per_route_run', 'N/A'):.2f} YPRR, {data1.get('route_grade', 'N/A'):.1f} grade\n"
        response += f"• {player2}: {data2.get('yards_per_route_run', 'N/A'):.2f} YPRR, {data2.get('route_grade', 'N/A'):.1f} grade\n\n"
        
        # Scheme comparison
        scheme1 = self.db_manager.get_scheme_data(player1)
        scheme2 = self.db_manager.get_scheme_data(player2)
        
        if scheme1 and scheme2:
            response += f"**Scheme Analysis:**\n"
            response += f"• {player1}: {scheme1.get('yprr_man_vs_zone_diff', 'N/A'):.2f} man vs zone diff\n"
            response += f"• {player2}: {scheme2.get('yprr_man_vs_zone_diff', 'N/A'):.2f} man vs zone diff\n\n"
        
        return response

class RouteAnalysisAgent:
    """Specialized agent for route running analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def process(self, query: str) -> str:
        """Process route analysis queries"""
        if 'best route' in query.lower():
            return self.get_best_route_runners()
        elif 'yprr' in query.lower():
            return self.get_yprr_analysis()
        else:
            return "I can analyze route running data. Try asking about the best route runners or YPRR analysis."
    
    def get_best_route_runners(self) -> str:
        """Get top route runners"""
        query = """
        SELECT player_name, team, position, route_grade, yards_per_route_run, route_participation
        FROM player_route_stats
        WHERE season = 2024 AND position = 'WR'
        ORDER BY yards_per_route_run DESC
        LIMIT 10
        """
        
        df = self.db_manager.execute_query(query)
        
        response = "🏃 **Top Route Runners (2024)**\n\n"
        for _, row in df.iterrows():
            response += f"• **{row['player_name']}** ({row['team']}): {row['yards_per_route_run']:.2f} YPRR, {row['route_grade']:.1f} grade\n"
        
        return response
    
    def get_yprr_analysis(self) -> str:
        """Get YPRR analysis"""
        query = """
        SELECT 
            AVG(yards_per_route_run) as avg_yprr,
            MAX(yards_per_route_run) as max_yprr,
            COUNT(*) as total_players
        FROM player_route_stats
        WHERE season = 2024 AND position = 'WR'
        """
        
        df = self.db_manager.execute_query(query)
        
        if not df.empty:
            row = df.iloc[0]
            response = "📈 **YPRR Analysis (2024 WRs)**\n\n"
            response += f"• Average YPRR: {row['avg_yprr']:.2f}\n"
            response += f"• Maximum YPRR: {row['max_yprr']:.2f}\n"
            response += f"• Total Players: {row['total_players']}\n"
            return response
        
        return "Sorry, I couldn't retrieve YPRR analysis."

class MarketAgent:
    """Specialized agent for market analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def process(self, query: str) -> str:
        """Process market analysis queries"""
        if 'adp' in query.lower():
            return self.get_adp_analysis()
        elif 'value' in query.lower():
            return self.get_value_picks()
        else:
            return "I can analyze market data. Try asking about ADP analysis or value picks."
    
    def get_adp_analysis(self) -> str:
        """Get ADP analysis"""
        query = """
        SELECT p.name, p.position, p.team, fm.adp_overall, fm.adp_position
        FROM players p
        JOIN fantasy_market fm ON p.player_id = fm.player_id
        JOIN games g ON fm.game_id = g.game_id
        WHERE g.season = 2024 AND fm.adp_overall IS NOT NULL
        ORDER BY fm.adp_overall
        LIMIT 10
        """
        
        df = self.db_manager.execute_query(query)
        
        response = "💰 **Top ADP Rankings (2024)**\n\n"
        for _, row in df.iterrows():
            response += f"• **{row['name']}** ({row['team']}): ADP {row['adp_overall']}, Position {row['adp_position']}\n"
        
        return response
    
    def get_value_picks(self) -> str:
        """Get value picks analysis"""
        return "🔍 **Value Picks Analysis**\n\nBased on route running efficiency vs ADP, here are some potential value picks:\n\n• Players with high YPRR but lower ADP\n• Route specialists being undervalued\n• Scheme-specific performers\n\nTry asking about specific players for detailed value analysis."

class StrategyAgent:
    """Specialized agent for strategic recommendations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def process(self, query: str) -> str:
        """Process strategy queries"""
        if 'draft' in query.lower():
            return self.get_draft_strategy()
        elif 'trade' in query.lower():
            return self.get_trade_advice()
        else:
            return "I can provide strategic advice. Try asking about draft strategy or trade recommendations."
    
    def get_draft_strategy(self) -> str:
        """Get draft strategy recommendations"""
        response = "🎯 **Draft Strategy Recommendations**\n\n"
        response += "**Route Running Strategy:**\n"
        response += "• Target players with >2.0 YPRR in middle rounds\n"
        response += "• Prioritize route participation >90%\n"
        response += "• Look for scheme specialists (man vs zone)\n\n"
        response += "**Value Approach:**\n"
        response += "• Focus on efficient route runners\n"
        response += "• Consider contested catch specialists\n"
        response += "• Balance volume vs efficiency\n"
        
        return response
    
    def get_trade_advice(self) -> str:
        """Get trade advice"""
        return "🤝 **Trade Strategy**\n\n• Target players with improving route efficiency\n• Consider scheme-specific matchups\n• Look for undervalued route specialists\n• Balance short-term vs long-term value"

def main():
    """Main chat interface"""
    print("🏈 Fantasy Football Database Chat System")
    print("=" * 50)
    print("Ask me about players, route running, market data, or strategy!")
    print("Type 'quit' to exit\n")
    
    db_manager = DatabaseManager()
    router = QueryRouter(db_manager)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thanks for using the Fantasy Football Chat System!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Assistant:")
            response = router.process_query(user_input)
            print(response)
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nThanks for using the Fantasy Football Chat System!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
