"""
Keeper Integration with Monte Carlo Simulator

This module integrates the keeper system with the Monte Carlo draft simulator
to provide keeper-aware draft simulations.
"""

from typing import Dict, List, Optional, Set
import copy

from .keeper_system import KeeperConfiguration, Keeper
from .models import LeagueConfig, Player
from .enhanced_monte_carlo import EnhancedMonteCarloSimulator, EnhancedPlayer
from .draft_state import DraftState


class KeeperAwareDraftSimulator(EnhancedMonteCarloSimulator):
    """Monte Carlo simulator with keeper support"""
    
    def __init__(self, league_config: LeagueConfig, keeper_config: Optional[KeeperConfiguration] = None):
        super().__init__(league_config)
        self.keeper_config = keeper_config
        self.kept_players: Set[str] = set()
        self.keeper_picks: Dict[int, str] = {}  # overall_pick -> player_name
        
        if keeper_config:
            self._process_keeper_configuration()
    
    def _process_keeper_configuration(self):
        """Process keeper configuration and determine draft impacts"""
        if not self.keeper_config:
            return
        
        print(f"🏆 Processing {len(self.keeper_config.keepers)} keepers...")
        
        # Track kept players (remove from available pool)
        self.kept_players = set(self.keeper_config.get_kept_players())
        
        # Calculate which draft picks are used by keepers
        self.keeper_picks = self._calculate_keeper_picks()
        
        print(f"   • {len(self.kept_players)} players kept")
        print(f"   • {len(self.keeper_picks)} draft picks used")
        
        # Show keeper summary
        self._print_keeper_summary()
    
    def _calculate_keeper_picks(self) -> Dict[int, str]:
        """Calculate which overall pick numbers are used by keepers"""
        picks_used = {}
        
        for keeper in self.keeper_config.keepers:
            # Calculate overall pick number based on snake draft
            round_num = keeper.keeper_round
            draft_pos = keeper.draft_position
            
            if round_num % 2 == 1:  # Odd rounds: normal order
                pick_in_round = draft_pos
            else:  # Even rounds: reverse order (snake)
                pick_in_round = self.config.num_teams + 1 - draft_pos
            
            overall_pick = (round_num - 1) * self.config.num_teams + pick_in_round
            picks_used[overall_pick] = keeper.player_name
        
        return picks_used
    
    def _print_keeper_summary(self):
        """Print keeper configuration summary"""
        print("\n🏆 KEEPER CONFIGURATION SUMMARY:")
        
        # Group by round
        round_keepers = {}
        for keeper in self.keeper_config.keepers:
            if keeper.keeper_round not in round_keepers:
                round_keepers[keeper.keeper_round] = []
            round_keepers[keeper.keeper_round].append(keeper)
        
        for round_num in sorted(round_keepers.keys()):
            keepers_in_round = round_keepers[round_num]
            print(f"\n   Round {round_num}:")
            for keeper in keepers_in_round:
                overall_pick = next(pick for pick, player in self.keeper_picks.items() 
                                  if player == keeper.player_name)
                print(f"     Pick #{overall_pick:2d}: {keeper.player_name:20s} (kept by {keeper.team_name})")
    
    def get_available_players_for_draft(self) -> List[EnhancedPlayer]:
        """Get list of players available for drafting (excluding keepers)"""
        available = []
        
        for player_name, enhanced_player in self.enhanced_players.items():
            if player_name not in self.kept_players:
                available.append(enhanced_player)
        
        # Sort by projection (best first)
        available.sort(key=lambda p: p.projection, reverse=True)
        return available
    
    def simulate_keeper_aware_draft(
        self, 
        strategy,
        num_simulations: int = 25,
        rounds_to_simulate: int = 15
    ) -> Dict[str, any]:
        """Run Monte Carlo simulation accounting for keepers"""
        
        print(f"🎯 Running Keeper-Aware Monte Carlo Simulation:")
        print(f"   • {num_simulations} draft simulations")
        print(f"   • {rounds_to_simulate} rounds each")
        print(f"   • {len(self.kept_players)} players already kept")
        print(f"   • {len(self.get_available_players_for_draft())} players available")
        
        results = []
        
        for sim in range(num_simulations):
            if sim % 5 == 0:
                print(f"   Simulation {sim+1}/{num_simulations}...")
            
            # Run single draft simulation with keepers
            draft_result = self._simulate_single_keeper_draft(
                strategy, 
                rounds_to_simulate,
                outcome_simulation_index=sim % self.outcome_matrix.num_simulations
            )
            
            results.append(draft_result)
        
        # Aggregate results
        aggregated = self._aggregate_simulation_results(results)
        
        # Add keeper-specific analysis
        aggregated['keeper_impact'] = self._analyze_keeper_impact(results)
        
        return aggregated
    
    def _simulate_single_keeper_draft(
        self, 
        strategy,
        rounds_to_simulate: int,
        outcome_simulation_index: int
    ) -> Dict[str, any]:
        """Simulate a single draft accounting for keeper picks"""
        
        # Start with available players (keepers removed)
        available_players = self.get_available_players_for_draft()
        
        # Initialize rosters with keepers
        rosters = {}
        for i in range(self.config.num_teams):
            rosters[i] = []
        
        # Add keepers to appropriate team rosters
        for keeper in self.keeper_config.keepers:
            team_index = keeper.draft_position - 1  # Convert to 0-based
            if keeper.player_name in self.enhanced_players:
                keeper_player = self.enhanced_players[keeper.player_name]
                rosters[team_index].append(keeper_player)
        
        # Create draft state
        draft_state = DraftState(
            current_round=1,
            current_pick=1,
            rosters=rosters,
            available_players=available_players,
            player_universe=available_players
        )
        
        user_roster = list(rosters[self.config.draft_position - 1])  # Start with your keepers
        user_team_index = self.config.draft_position - 1
        
        # Simulate draft with keeper picks already made
        current_overall_pick = 1
        
        for round_num in range(1, rounds_to_simulate + 1):
            # Determine pick order for this round
            if round_num % 2 == 1:  # Odd rounds
                pick_order = list(range(self.config.num_teams))
            else:  # Even rounds (snake)
                pick_order = list(range(self.config.num_teams - 1, -1, -1))
            
            for pick_pos, team_index in enumerate(pick_order):
                if not draft_state.available_players:
                    break
                
                # Check if this pick is already used by a keeper
                if current_overall_pick in self.keeper_picks:
                    # Skip this pick - keeper already selected
                    print(f"     Pick #{current_overall_pick}: {self.keeper_picks[current_overall_pick]} (keeper)")
                    current_overall_pick += 1
                    continue
                
                if team_index == user_team_index:
                    # User's pick - use strategy
                    pick = strategy.select_pick(draft_state, self.config)
                    user_roster.append(pick)
                else:
                    # AI opponent pick
                    pick = self._ai_opponent_pick(draft_state)
                
                # Remove picked player and update state
                if pick in draft_state.available_players:
                    draft_state.available_players.remove(pick)
                    draft_state.rosters[team_index].append(pick)
                
                current_overall_pick += 1
            
            draft_state.current_round += 1
        
        # Calculate roster score using specific outcome simulation
        roster_score = self._calculate_roster_outcome_score(user_roster, outcome_simulation_index)
        
        return {
            'roster': user_roster,
            'roster_score': roster_score,
            'outcome_simulation': outcome_simulation_index,
            'rounds_completed': rounds_to_simulate,
            'keepers_count': len([p for p in user_roster if p.name in self.kept_players])
        }
    
    def _analyze_keeper_impact(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Analyze the impact of keepers on draft results"""
        
        # Calculate keeper value vs ADP
        keeper_value_analysis = {}
        
        for keeper in self.keeper_config.keepers:
            if keeper.player_name in self.enhanced_players:
                player = self.enhanced_players[keeper.player_name]
                
                # Estimate where this player would normally be drafted
                # This is simplified - could be enhanced with ADP data
                round_value = keeper.keeper_round
                estimated_normal_round = max(1, int(player.projection / 20))  # Rough estimate
                
                keeper_value_analysis[keeper.player_name] = {
                    'keeper_round': keeper.keeper_round,
                    'estimated_natural_round': estimated_normal_round,
                    'value_rounds_gained': estimated_normal_round - keeper.keeper_round,
                    'projection': player.projection,
                    'team': keeper.team_name
                }
        
        # Analyze how keepers affected available player pool
        total_kept_projection = sum(
            self.enhanced_players[name].projection 
            for name in self.kept_players 
            if name in self.enhanced_players
        )
        
        available_players = self.get_available_players_for_draft()
        avg_available_projection = sum(p.projection for p in available_players) / len(available_players)
        
        return {
            'keeper_values': keeper_value_analysis,
            'total_kept_projection': total_kept_projection,
            'average_available_projection': avg_available_projection,
            'players_removed_from_pool': len(self.kept_players),
            'draft_picks_used': len(self.keeper_picks)
        }
    
    def get_user_keeper_advantage(self) -> Dict[str, any]:
        """Analyze the user's keeper advantage/disadvantage"""
        user_keepers = [k for k in self.keeper_config.keepers 
                       if k.draft_position == self.config.draft_position]
        
        if not user_keepers:
            return {'has_keepers': False}
        
        analysis = {'has_keepers': True, 'keepers': []}
        
        for keeper in user_keepers:
            if keeper.player_name in self.enhanced_players:
                player = self.enhanced_players[keeper.player_name]
                
                keeper_analysis = {
                    'player_name': keeper.player_name,
                    'position': player.position,
                    'projection': player.projection,
                    'keeper_round': keeper.keeper_round,
                    'coefficient_of_variation': player.coefficient_of_variation,
                    'consistency_tier': 'High' if player.coefficient_of_variation <= 0.15 
                                      else 'Medium' if player.coefficient_of_variation <= 0.35 
                                      else 'Low'
                }
                
                analysis['keepers'].append(keeper_analysis)
        
        return analysis


def create_example_keeper_config() -> KeeperConfiguration:
    """Create an example keeper configuration for testing"""
    keepers = [
        Keeper("Your Team", 6, "Brian Thomas Jr.", 8),  # Your 8th round keeper
        Keeper("Team Alpha", 1, "Josh Allen", 3),       # 3rd round keeper
        Keeper("Team Beta", 12, "Puka Nacua", 5),       # 5th round keeper
        Keeper("Team Gamma", 4, "Travis Kelce", 1),     # 1st round keeper
    ]
    
    return KeeperConfiguration(keepers=keepers, keeper_rules="round_based")


if __name__ == "__main__":
    # Test keeper-aware simulation
    print("🏆 Testing Keeper-Aware Monte Carlo Simulation")
    print("=" * 60)
    
    from .models import LeagueConfig
    from .enhanced_monte_carlo import OutcomeAwareStrategy
    
    # Create test configuration
    league_config = LeagueConfig(
        num_teams=12,
        draft_position=6,  # User drafts 6th
        roster_size=16
    )
    
    # Create example keeper configuration
    keeper_config = create_example_keeper_config()
    
    # Create keeper-aware simulator
    simulator = KeeperAwareDraftSimulator(league_config, keeper_config)
    
    # Analyze user's keeper advantage
    user_advantage = simulator.get_user_keeper_advantage()
    print(f"\n🏆 Your Keeper Analysis:")
    if user_advantage['has_keepers']:
        for keeper in user_advantage['keepers']:
            print(f"   {keeper['player_name']} ({keeper['position']}): {keeper['projection']:.1f} pts")
            print(f"     Round: {keeper['keeper_round']}, Consistency: {keeper['consistency_tier']}")
    else:
        print("   No keepers configured for your team")
    
    print("\n✅ Keeper integration ready for full simulation!")
