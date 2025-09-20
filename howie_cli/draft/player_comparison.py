"""
Player Comparison Tool

Provides detailed side-by-side comparison of two players including projections,
value metrics, ADP, and recommendation scores.
"""

from typing import List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .models import Player
from .database import DraftDatabaseConnector
from .recommendation_engine import PickRecommendationEngine
from .variance_adjusted_value import VarianceAdjustedValueCalculator

console = Console()

class PlayerComparison:
    """Tool for comparing two players side-by-side"""
    
    def __init__(self, league_config, player_universe: List[Player]):
        self.config = league_config
        self.players = player_universe
        self.db = DraftDatabaseConnector()
        self.rec_engine = PickRecommendationEngine(league_config, player_universe)
        self.value_calc = VarianceAdjustedValueCalculator(player_universe)
    
    def search_player(self, search_term: str) -> List[Player]:
        """Search for players by name (fuzzy matching)"""
        search_term = search_term.lower().strip()
        
        # Exact matches first
        exact_matches = [p for p in self.players if search_term in p.name.lower()]
        
        # If no exact matches, try partial matches
        if not exact_matches:
            words = search_term.split()
            partial_matches = []
            for player in self.players:
                player_name_lower = player.name.lower()
                if all(word in player_name_lower for word in words):
                    partial_matches.append(player)
            return partial_matches
        
        return exact_matches
    
    def compare_players(self, player1_name: str, player2_name: str) -> str:
        """Compare two players and return formatted comparison"""
        
        # Search for players
        player1_matches = self.search_player(player1_name)
        player2_matches = self.search_player(player2_name)
        
        if not player1_matches:
            return f"❌ No players found matching '{player1_name}'"
        
        if not player2_matches:
            return f"❌ No players found matching '{player2_name}'"
        
        # If multiple matches, show options
        if len(player1_matches) > 1:
            return self._show_multiple_matches(player1_name, player1_matches)
        
        if len(player2_matches) > 1:
            return self._show_multiple_matches(player2_name, player2_matches)
        
        player1 = player1_matches[0]
        player2 = player2_matches[0]
        
        return self._generate_comparison(player1, player2)
    
    def _show_multiple_matches(self, search_term: str, matches: List[Player]) -> str:
        """Show multiple player matches for disambiguation"""
        output = [f"🔍 Multiple players found for '{search_term}':"]
        output.append("")
        
        for i, player in enumerate(matches[:10], 1):  # Show max 10
            output.append(f"   {i}. {player.name} ({player.position}) - {player.projection:.0f} pts")
        
        output.append("")
        output.append("💡 Be more specific with the player name to compare")
        
        return "\n".join(output)
    
    def _generate_comparison(self, player1: Player, player2: Player) -> str:
        """Generate detailed comparison between two players"""
        
        # Get value metrics
        vorp1 = self.value_calc.calculate_vorp(player1)
        vorp2 = self.value_calc.calculate_vorp(player2)
        
        # Get player distributions for variance data (may be None)
        try:
            dist1 = self.value_calc.distribution_factory.get_distribution(player1)
            dist2 = self.value_calc.distribution_factory.get_distribution(player2)
        except:
            dist1 = None
            dist2 = None
        
        # Create comparison table
        table = Table(title=f"🆚 Player Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="bold white", width=20)
        table.add_column(f"{player1.name}", style="cyan", width=25)
        table.add_column(f"{player2.name}", style="yellow", width=25)
        table.add_column("Advantage", style="green", width=15)
        
        # Basic info
        table.add_row("Position", player1.position, player2.position, 
                     self._compare_values(player1.position, player2.position, str, False, player1.name, player2.name))
        
        # Projections
        table.add_row("Projection", f"{player1.projection:.1f} pts", f"{player2.projection:.1f} pts",
                     self._compare_values(player1.projection, player2.projection, float, False, player1.name, player2.name))
        
        # ADP
        adp1_str = f"{player1.adp:.1f}" if player1.adp < 999 else "Undrafted"
        adp2_str = f"{player2.adp:.1f}" if player2.adp < 999 else "Undrafted"
        adp_advantage = self._compare_adp(player1.adp, player2.adp)
        table.add_row("ADP", adp1_str, adp2_str, adp_advantage)
        
        # VORP
        table.add_row("VORP", f"{vorp1:.1f}", f"{vorp2:.1f}",
                     self._compare_values(vorp1, vorp2, float, False, player1.name, player2.name))
        
        # Variance data if available
        if dist1 and dist2:
            # Calculate upside/floor from distribution parameters
            std1 = dist1.coefficient_of_variation * dist1.mean_projection
            std2 = dist2.coefficient_of_variation * dist2.mean_projection
            
            upside1 = dist1.ceiling_cap
            upside2 = dist2.ceiling_cap
            floor1 = dist1.floor_cap  
            floor2 = dist2.floor_cap
            
            table.add_row("Ceiling", f"{upside1:.0f} pts", f"{upside2:.0f} pts",
                         self._compare_values(upside1, upside2, float, False, player1.name, player2.name))
            
            table.add_row("Floor", f"{floor1:.0f} pts", f"{floor2:.0f} pts",
                         self._compare_values(floor1, floor2, float, False, player1.name, player2.name))
            
            table.add_row("Volatility", f"{dist1.coefficient_of_variation:.2f}", f"{dist2.coefficient_of_variation:.2f}",
                         self._compare_values(dist1.coefficient_of_variation, dist2.coefficient_of_variation, float, True, player1.name, player2.name))
        
        # Bye week
        table.add_row("Bye Week", str(player1.bye_week), str(player2.bye_week),
                     self._compare_bye_weeks(player1.bye_week, player2.bye_week))
        
        # Create output
        output = []
        output.append("")
        
        # Add recommendation context
        if player1.position == player2.position:
            output.append(f"📊 Same Position Comparison ({player1.position})")
        else:
            output.append(f"📊 Cross-Position Comparison ({player1.position} vs {player2.position})")
        
        output.append("")
        
        # Display table
        console.print(table)
        
        # Add summary and recommendation
        output.append(self._generate_summary(player1, player2, vorp1, vorp2, dist1, dist2))
        
        return "\n".join(output)
    
    def _compare_values(self, val1, val2, value_type, lower_better=False, player1_name="Player 1", player2_name="Player 2"):
        """Compare two values and return advantage indicator"""
        if value_type == str:
            return "—"
        
        if val1 == val2:
            return "Tied"
        
        p1_short = player1_name.split()[0]
        p2_short = player2_name.split()[0]
        
        if lower_better:
            return p1_short if val1 < val2 else p2_short
        else:
            return p1_short if val1 > val2 else p2_short
    
    def _compare_adp(self, adp1, adp2):
        """Compare ADP values (lower is better for draft position)"""
        if adp1 >= 999 and adp2 >= 999:
            return "Both undrafted"
        elif adp1 >= 999:
            return "Player 2 (drafted)"
        elif adp2 >= 999:
            return "Player 1 (drafted)"
        else:
            return "Player 1" if adp1 < adp2 else "Player 2" if adp2 < adp1 else "Tied"
    
    def _compare_bye_weeks(self, bye1, bye2):
        """Compare bye weeks"""
        if bye1 == bye2:
            return "Same bye"
        else:
            return "Different"
    
    def _generate_summary(self, player1: Player, player2: Player, vorp1: float, vorp2: float, 
                         dist1, dist2) -> str:
        """Generate summary and recommendation"""
        output = []
        
        output.append("🎯 SUMMARY & RECOMMENDATION:")
        output.append("")
        
        # Overall value winner
        if vorp1 > vorp2:
            advantage = vorp1 - vorp2
            output.append(f"🏆 {player1.name} has higher overall value (+{advantage:.1f} VORP)")
        elif vorp2 > vorp1:
            advantage = vorp2 - vorp1
            output.append(f"🏆 {player2.name} has higher overall value (+{advantage:.1f} VORP)")
        else:
            output.append("⚖️  Very similar overall value")
        
        # Risk/upside analysis
        if dist1 and dist2:
            output.append("")
            
            upside1 = dist1.ceiling_cap
            upside2 = dist2.ceiling_cap
            floor1 = dist1.floor_cap
            floor2 = dist2.floor_cap
            
            if upside1 > upside2:
                output.append(f"📈 {player1.name} has higher ceiling ({upside1:.0f} vs {upside2:.0f})")
            elif upside2 > upside1:
                output.append(f"📈 {player2.name} has higher ceiling ({upside2:.0f} vs {upside1:.0f})")
            
            if floor1 > floor2:
                output.append(f"🛡️  {player1.name} has higher floor ({floor1:.0f} vs {floor2:.0f})")
            elif floor2 > floor1:
                output.append(f"🛡️  {player2.name} has higher floor ({floor2:.0f} vs {floor1:.0f})")
            
            # Volatility comparison
            if dist1.coefficient_of_variation < dist2.coefficient_of_variation:
                output.append(f"⚡ {player1.name} is more consistent (lower volatility)")
            elif dist2.coefficient_of_variation < dist1.coefficient_of_variation:
                output.append(f"⚡ {player2.name} is more consistent (lower volatility)")
        
        # Draft strategy insight
        output.append("")
        proj_diff = abs(player1.projection - player2.projection)
        adp_diff = abs(player1.adp - player2.adp) if player1.adp < 999 and player2.adp < 999 else 0
        
        if proj_diff < 10 and adp_diff > 20:
            later_player = player1 if player1.adp > player2.adp else player2
            output.append(f"💡 Similar projections but {later_player.name} available later - good value!")
        elif proj_diff > 20:
            better_player = player1 if player1.projection > player2.projection else player2
            output.append(f"💪 {better_player.name} significantly better projection - worth reaching for")
        
        return "\n".join(output)

def compare_players_command(league_config, player_universe: List[Player], 
                          player1_name: str, player2_name: str) -> str:
    """Command interface for player comparison"""
    comparison = PlayerComparison(league_config, player_universe)
    return comparison.compare_players(player1_name, player2_name)
