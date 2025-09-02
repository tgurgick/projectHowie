"""
Draft analysis generator
Creates comprehensive pre-draft analysis with round-by-round recommendations
"""

from typing import List, Optional
from .models import LeagueConfig, KeeperPlayer, Roster
from .database import DraftDatabaseConnector
from .recommendation_engine import PickRecommendationEngine


class DraftAnalysisGenerator:
    """Generate comprehensive draft analysis output"""
    
    def __init__(self):
        self.db = DraftDatabaseConnector()
    
    def generate_pre_draft_analysis(
        self, 
        league_config: LeagueConfig,
        keepers: List[KeeperPlayer] = None,
        rounds_to_analyze: int = 16
    ) -> str:
        """Generate complete pre-draft analysis"""
        
        try:
            # Load data
            players = self.db.load_player_universe()
            
            if not players:
                return "❌ No player data found. Please check database connection."
            
            # Remove keepers from available pool
            if keepers:
                keeper_names = set(k.player_name.lower() for k in keepers)
                players = [p for p in players if p.name.lower() not in keeper_names]
            
            # Initialize recommendation engine
            rec_engine = PickRecommendationEngine(league_config, players)
            
            # Generate analysis for each round
            output = []
            current_roster = Roster(league_config)
            drafted_players = []  # Track players drafted in simulation
            
            # Header
            output.append("=" * 80)
            output.append(f"🏈 DRAFT ANALYSIS - {league_config.num_teams} Team League")
            output.append(f"📍 Draft Position: {league_config.draft_position}")
            output.append(f"🎯 Scoring: {league_config.scoring_type.upper()}")
            if keepers:
                output.append(f"🔒 Keepers: {len(keepers)} players")
            output.append("=" * 80)
            
            # Process rounds
            for round_num in range(1, rounds_to_analyze + 1):
                output.append(f"\n📍 ROUND {round_num} ANALYSIS")
                
                pick_number = rec_engine._calculate_pick_number(round_num)
                output.append(f"Your Pick: #{pick_number}")
                output.append("-" * 60)
                
                # Get recommendations
                try:
                    recommendations = rec_engine.generate_round_recommendations(
                        round_num, current_roster, drafted_players
                    )
                    
                    if not recommendations:
                        output.append("⚠️  No recommendations available for this round")
                        continue
                    
                    # Show different detail levels for early vs late rounds
                    if round_num <= 8:
                        output.append("🎯 TOP RECOMMENDATIONS:")
                        num_to_show = min(6, len(recommendations))  # Fewer recs to reduce clutter
                    else:
                        num_to_show = min(3, len(recommendations))  # Even fewer for late rounds
                    
                    for i, rec in enumerate(recommendations[:num_to_show], 1):
                        # Player line with ADP info
                        adp_text = f"ADP {rec.player.adp:.0f}" if rec.player.adp < 999 else "No ADP"
                        output.append(
                            f"{i:2d}. {rec.player.name:<18} {rec.player.position:2s} "
                            f"({rec.player.projection:.0f} pts, {adp_text}) - Score: {rec.overall_score:.1f}"
                        )
                        
                        # Primary reason
                        output.append(f"    💡 {rec.primary_reason}")
                        
                        # Show details only for early rounds and top picks
                        if round_num <= 8:
                            # Enhanced factors for top 3 picks only
                            if i <= 3:
                                factors = rec.enhanced_factors
                                output.append(f"    📅 {factors.get('sos', 'SoS Unknown')}")
                                output.append(f"    🏈 {factors.get('starter', 'Role Unknown')}")
                                output.append(f"    🏥 {factors.get('injury', 'Health Unknown')}")
                            
                            # Risk factors
                            if rec.risk_factors and i <= 3:
                                output.append(f"    ⚠️  Risks: {', '.join(rec.risk_factors)}")
                            
                            # Detailed metrics for top 2 only
                            if i <= 2:
                                output.append(
                                    f"    📊 VORP: {rec.vorp:.1f} | "
                                    f"SoS: {rec.sos_advantage:.2f} | "
                                    f"Starter: {rec.starter_status_score:.2f} | "
                                    f"Health: {rec.injury_risk_score:.2f}"
                                )
                        
                        output.append("")  # Blank line between recommendations
                    
                    # Simulate taking the top recommendation for next round
                    if recommendations:
                        top_pick = recommendations[0].player
                        current_roster = current_roster.add_player(top_pick)
                        drafted_players.append(top_pick)  # Add to drafted list
                        output.append(f"📝 Simulating pick: {top_pick.name} ({top_pick.position})")
                        
                        # Show updated roster
                        roster_summary = current_roster.get_summary()
                        output.append(f"📋 Updated roster: {roster_summary}")
                
                except Exception as e:
                    output.append(f"❌ Error generating recommendations: {str(e)}")
            
            # Summary insights
            output.append(f"\n💡 KEY INSIGHTS")
            output.append("-" * 50)
            insights = self._generate_key_insights(players, league_config, current_roster)
            for insight in insights:
                output.append(f"• {insight}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error generating analysis: {str(e)}\n\nPlease check your database connection and try again."
    
    def _generate_key_insights(self, players: List, league_config: LeagueConfig, final_roster: Roster) -> List[str]:
        """Generate key strategic insights"""
        insights = []
        
        # Position depth analysis
        position_counts = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = [p for p in players if p.position == position and p.projection > 100]
            position_counts[position] = len(pos_players)
        
        # Identify scarce positions
        total_teams = league_config.num_teams
        for pos, count in position_counts.items():
            if pos == 'QB' and count < total_teams * 1.5:
                insights.append(f"QB depth is limited ({count} viable options)")
            elif pos == 'TE' and count < total_teams * 1.5:
                insights.append(f"TE is very scarce ({count} viable options)")
            elif pos in ['RB', 'WR'] and count < total_teams * 2.5:
                insights.append(f"{pos} depth is concerning ({count} viable options)")
        
        # Draft position strategy
        if league_config.draft_position <= 3:
            insights.append("Early draft position: Target elite RB or WR in Round 1")
        elif league_config.draft_position >= league_config.num_teams - 2:
            insights.append("Late draft position: Consider elite QB or double up on position")
        else:
            insights.append("Middle draft position: Take best available, be flexible")
        
        # Roster construction insight
        final_needs = final_roster.get_needs()
        high_needs = [pos for pos, need in final_needs.items() if need > 0.5]
        if high_needs:
            insights.append(f"Prioritize these positions in later rounds: {', '.join(high_needs)}")
        
        # Add some general strategy insights
        insights.append("Consider strength of schedule when deciding between similar players")
        insights.append("Confirmed starters are more valuable than committee players")
        insights.append("Monitor injury reports leading up to your draft")
        
        return insights[:6]  # Limit to 6 insights
    
    def generate_simple_summary(self, league_config: LeagueConfig) -> str:
        """Generate a simple summary for testing"""
        try:
            players = self.db.load_player_universe()
            
            output = []
            output.append("🏈 DRAFT SIMULATION READY")
            output.append(f"Players loaded: {len(players)}")
            output.append(f"League: {league_config.num_teams} teams")
            output.append(f"Your position: {league_config.draft_position}")
            
            if players:
                # Show top 5 players by position
                for position in ['QB', 'RB', 'WR', 'TE']:
                    pos_players = [p for p in players if p.position == position]
                    pos_players.sort(key=lambda x: x.projection, reverse=True)
                    
                    output.append(f"\nTop {position}s:")
                    for i, player in enumerate(pos_players[:3], 1):
                        output.append(f"  {i}. {player.name} ({player.projection:.1f} pts)")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
