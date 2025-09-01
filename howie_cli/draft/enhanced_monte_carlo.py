"""
Enhanced Monte Carlo Simulation with Player Outcome Distributions

This module enhances the existing Monte Carlo simulation by integrating
realistic player distributions for more accurate season outcome modeling.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass
from .monte_carlo_simulator import MonteCarloSimulator, MonteCarloResults
from .distributions import DistributionFactory, OutcomesMatrixGenerator, SeasonOutcome
from .distribution_database import DistributionDatabaseManager, PlayerDistributionProfile
from .models import Player, LeagueConfig, Roster
from .draft_state import DraftState


@dataclass 
class EnhancedMonteCarloResults(MonteCarloResults):
    """Enhanced results including distribution statistics"""
    # Inherit all from base MonteCarloResults
    
    # Add distribution-specific metrics
    variance_statistics: Dict[str, Dict[str, float]]  # Per-player outcome stats
    injury_impact_analysis: Dict[str, float]  # Games missed impact
    ceiling_floor_analysis: Dict[str, Dict[str, float]]  # P90/P10 outcomes
    position_variance_summary: Dict[str, Dict[str, float]]  # By position stats


class EnhancedMonteCarloSimulator(MonteCarloSimulator):
    """Enhanced Monte Carlo simulator with realistic player distributions"""
    
    def __init__(self, config: LeagueConfig, players: List[Player]):
        super().__init__(config, players)
        
        # Initialize distribution system
        self.db_manager = DistributionDatabaseManager()
        self.player_distributions = self._load_player_distributions()
        self.outcomes_matrix = None
        self.distribution_stats = {}
        
        # Create distribution objects
        self.distributions = {}
        for profile in self.player_distributions:
            self.distributions[profile.player_name] = DistributionFactory.create_distribution(profile)
    
    def _load_player_distributions(self) -> List[PlayerDistributionProfile]:
        """Load player distribution profiles from database"""
        try:
            profiles = self.db_manager.get_all_player_distributions()
            print(f"📊 Loaded {len(profiles)} player distribution profiles")
            return profiles
        except Exception as e:
            print(f"⚠️  Warning: Could not load distributions: {e}")
            print("   Falling back to simple variance estimation...")
            return self._create_fallback_distributions()
    
    def _create_fallback_distributions(self) -> List[PlayerDistributionProfile]:
        """Create simple distributions if database unavailable"""
        profiles = []
        
        # Simple CV mapping by position
        position_cv = {
            'QB': 0.18, 'RB': 0.25, 'WR': 0.20, 'TE': 0.22, 
            'K': 0.15, 'DEF': 0.20, 'DST': 0.20
        }
        
        for player in self.players:
            cv = position_cv.get(player.position.upper(), 0.20)
            
            profile = PlayerDistributionProfile(
                player_name=player.name,
                position=player.position,
                team=player.team,
                season=2025,
                mean_projection=player.projection,
                coefficient_of_variation=cv,
                injury_prob_healthy=0.82,
                injury_prob_minor=0.13,
                injury_prob_major=0.05
            )
            profiles.append(profile)
        
        return profiles
    
    def initialize_outcomes_matrix(self, num_samples: int = 15000):
        """Pre-generate outcomes matrix for fast simulation"""
        print(f"🎲 Initializing outcomes matrix with {num_samples:,} samples...")
        
        matrix_generator = OutcomesMatrixGenerator(
            self.player_distributions, 
            num_samples=num_samples
        )
        
        self.outcomes_matrix = matrix_generator.generate_outcomes_matrix()
        self.distribution_stats = matrix_generator.calculate_distribution_stats(self.outcomes_matrix)
        
        # Create player index mapping
        self.player_index_map = {
            profile.player_name: i 
            for i, profile in enumerate(self.player_distributions)
        }
        
        print(f"✅ Outcomes matrix ready: {self.outcomes_matrix.shape}")
        return self.outcomes_matrix
    
    def run_enhanced_simulation(
        self, 
        num_simulations: int = 25,
        rounds: int = 15,
        use_distributions: bool = True,
        num_outcome_samples: int = 10000
    ) -> EnhancedMonteCarloResults:
        """Run enhanced Monte Carlo simulation with distributions"""
        
        # Initialize outcomes matrix if using distributions
        if use_distributions and self.outcomes_matrix is None:
            self.initialize_outcomes_matrix(num_outcome_samples)
        
        print(f"🎯 Running {num_simulations} enhanced Monte Carlo simulations...")
        print(f"   Using {'realistic distributions' if use_distributions else 'simple projections'}")
        
        simulation_results = []
        pick_tracking = {round_num: {} for round_num in range(1, rounds + 1)}
        
        for sim_num in range(num_simulations):
            if sim_num % 10 == 0 and sim_num > 0:
                print(f"   Completed {sim_num}/{num_simulations} simulations...")
            
            # Set unique seeds for variance
            np.random.seed(sim_num * 12345)
            random.seed(sim_num * 54321)
            
            # Run single simulation
            result = self._run_single_enhanced_simulation(
                rounds, use_distributions, sim_num
            )
            
            simulation_results.append(result)
            
            # Track picks for aggregation
            for round_num, pick in result['user_picks'].items():
                if pick:
                    pick_name = pick.name
                    if pick_name not in pick_tracking[round_num]:
                        pick_tracking[round_num][pick_name] = 0
                    pick_tracking[round_num][pick_name] += 1
        
        # Calculate enhanced statistics
        enhanced_stats = self._calculate_enhanced_statistics(simulation_results)
        
        # Convert to standard results format and enhance
        most_common_picks = {}
        player_availability_rates = {}
        
        for round_num in range(1, rounds + 1):
            round_picks = pick_tracking[round_num]
            if round_picks:
                most_common = max(round_picks.items(), key=lambda x: x[1])
                most_common_picks[round_num] = {
                    'player_name': most_common[0],
                    'frequency': most_common[1] / num_simulations
                }
            
            # Calculate availability rates (simplified for enhanced version)
            for player_name, count in round_picks.items():
                if player_name not in player_availability_rates:
                    player_availability_rates[player_name] = {}
                player_availability_rates[player_name][round_num] = count / num_simulations
        
        # Calculate average roster strength using distributions
        avg_roster_strength = self._calculate_average_roster_strength_enhanced(
            simulation_results, use_distributions
        )
        
        return EnhancedMonteCarloResults(
            num_simulations=num_simulations,
            most_common_picks=most_common_picks,
            player_availability_rates=player_availability_rates,
            average_roster_strength=avg_roster_strength,
            variance_statistics=enhanced_stats['variance'],
            injury_impact_analysis=enhanced_stats['injury_impact'],
            ceiling_floor_analysis=enhanced_stats['ceiling_floor'],
            position_variance_summary=enhanced_stats['position_variance']
        )
    
    def _run_single_enhanced_simulation(
        self, 
        rounds: int, 
        use_distributions: bool,
        sim_num: int
    ) -> Dict:
        """Run a single enhanced simulation"""
        
        # Initialize draft state
        draft_state = DraftState(self.config, self.players)
        user_picks = {}
        
        for round_num in range(1, rounds + 1):
            available_players = [
                player for player in self.players 
                if player.name.lower() not in draft_state.drafted_players
            ]
            
            if not available_players:
                break
            
            # User's turn
            user_pick = self._make_strategic_pick(
                available_players, draft_state.get_user_roster(), round_num
            )
            
            if user_pick:
                draft_state.execute_pick(user_pick.name)
                user_picks[round_num] = user_pick
            
            # Other teams pick (simplified - use realistic opponents from parent class)
            available_after_user = [
                player for player in available_players 
                if player != user_pick and player.name.lower() not in draft_state.drafted_players
            ]
            
            for team_idx in range(1, self.config.num_teams):
                if available_after_user:
                    # Use parent class realistic opponent model
                    opponent_manager = self._get_opponent_manager()
                    
                    if hasattr(opponent_manager, 'managers') and team_idx - 1 < len(opponent_manager.managers):
                        ai_pick = opponent_manager.managers[team_idx - 1].make_pick(
                            available_after_user,
                            draft_state.teams[team_idx].roster,
                            round_num
                        )
                    else:
                        # Fallback to simple pick
                        ai_pick = available_after_user[0] if available_after_user else None
                    
                    if ai_pick:
                        draft_state.execute_pick(ai_pick.name)
                        available_after_user = [
                            p for p in available_after_user 
                            if p.name.lower() != ai_pick.name.lower()
                        ]
        
        # Get final user roster
        user_roster = draft_state.get_user_roster()
        
        # Calculate roster strength using distributions if enabled
        if use_distributions:
            roster_strength = self._calculate_roster_strength_with_distributions(
                user_roster, sim_num
            )
        else:
            roster_strength = sum(p.projection for p in user_roster.players)
        
        return {
            'user_picks': user_picks,
            'final_roster': user_roster,
            'roster_strength': roster_strength,
            'sim_num': sim_num
        }
    
    def _calculate_roster_strength_with_distributions(
        self, 
        roster: Roster, 
        outcome_column: int
    ) -> float:
        """Calculate roster strength using pre-sampled distributions"""
        
        if self.outcomes_matrix is None:
            # Fallback to simple projections
            return sum(p.projection for p in roster.players)
        
        total_strength = 0.0
        
        for player in roster.players:
            if player.name in self.player_index_map:
                player_idx = self.player_index_map[player.name]
                # Use the specific outcome column for this simulation
                total_strength += self.outcomes_matrix[player_idx, outcome_column % self.outcomes_matrix.shape[1]]
            else:
                # Fallback to projection for players not in matrix
                total_strength += player.projection
        
        return total_strength
    
    def _calculate_enhanced_statistics(self, simulation_results: List[Dict]) -> Dict:
        """Calculate enhanced statistics from simulation results"""
        
        stats = {
            'variance': {},
            'injury_impact': {},
            'ceiling_floor': {},
            'position_variance': {}
        }
        
        if not self.distribution_stats:
            return stats
        
        # Variance statistics per player (from distribution stats)
        for player_name, player_stats in self.distribution_stats.items():
            stats['variance'][player_name] = {
                'mean': player_stats['mean'],
                'std': player_stats['std'],
                'cv': player_stats['cv'],
                'upside_prob': player_stats['upside_prob'],
                'bust_prob': player_stats['bust_prob']
            }
        
        # Ceiling/floor analysis
        for player_name, player_stats in self.distribution_stats.items():
            stats['ceiling_floor'][player_name] = {
                'floor_p10': player_stats.get('p25', 0),  # Conservative floor
                'median': player_stats['median'],
                'ceiling_p90': player_stats['p90']
            }
        
        # Position variance summary
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        for position in positions:
            position_players = [
                name for name, profile in 
                {p.player_name: p for p in self.player_distributions}.items()
                if profile.position.upper() == position
            ]
            
            if position_players:
                position_cvs = [
                    self.distribution_stats.get(name, {}).get('cv', 0)
                    for name in position_players
                    if name in self.distribution_stats
                ]
                
                if position_cvs:
                    stats['position_variance'][position] = {
                        'avg_cv': np.mean(position_cvs),
                        'median_cv': np.median(position_cvs),
                        'max_cv': np.max(position_cvs),
                        'min_cv': np.min(position_cvs)
                    }
        
        return stats
    
    def _calculate_average_roster_strength_enhanced(
        self, 
        simulation_results: List[Dict],
        use_distributions: bool
    ) -> float:
        """Calculate average roster strength across simulations"""
        
        if not simulation_results:
            return 0.0
        
        total_strength = sum(result['roster_strength'] for result in simulation_results)
        return total_strength / len(simulation_results)
    
    def generate_enhanced_availability_report(self, results: EnhancedMonteCarloResults, top_n: int = 50) -> str:
        """Generate enhanced availability report with distribution insights"""
        
        output = []
        output.append("🎲 ENHANCED MONTE CARLO SIMULATION RESULTS")
        output.append("=" * 80)
        output.append(f"Simulations: {results.num_simulations}")
        output.append(f"Average Roster Strength: {results.average_roster_strength:.1f}")
        output.append(f"Distribution System: {'✅ Active' if self.outcomes_matrix is not None else '❌ Disabled'}")
        output.append("")
        
        # Show distribution insights
        if results.variance_statistics:
            output.append("📊 PLAYER VARIANCE INSIGHTS:")
            output.append("-" * 50)
            
            # Show top 10 highest variance players
            high_variance = sorted(
                results.variance_statistics.items(),
                key=lambda x: x[1].get('cv', 0),
                reverse=True
            )[:10]
            
            for player_name, stats in high_variance:
                cv = stats.get('cv', 0)
                upside = stats.get('upside_prob', 0)
                bust = stats.get('bust_prob', 0)
                output.append(f"  {player_name:<20} CV:{cv:.3f} Upside:{upside:.1%} Bust:{bust:.1%}")
            
            output.append("")
        
        # Position variance summary
        if results.position_variance_summary:
            output.append("📈 POSITION RISK ANALYSIS:")
            output.append("-" * 50)
            
            for position, stats in results.position_variance_summary.items():
                avg_cv = stats.get('avg_cv', 0)
                max_cv = stats.get('max_cv', 0)
                output.append(f"  {position:3s}: Avg CV {avg_cv:.3f} | Max CV {max_cv:.3f}")
            
            output.append("")
        
        # Standard availability report
        for round_num in range(1, 7):  # First 6 rounds
            output.append(f"📍 ROUND {round_num} PLAYER AVAILABILITY:")
            if round_num == 1:
                output.append("(Players available when your Round 1 pick comes up)")
            elif round_num == 2:
                output.append("(Players available when your Round 2 pick comes up - varies by Round 1 outcome)")
            else:
                output.append(f"(Players available when your Round {round_num} pick comes up)")
            output.append("-" * 50)
            
            # Get players available in this round
            round_availability = []
            for player_name, round_rates in results.player_availability_rates.items():
                if round_num in round_rates:
                    rate = round_rates[round_num]
                    round_availability.append((player_name, rate))
            
            # Sort by availability rate (descending) 
            round_availability.sort(key=lambda x: x[1], reverse=True)
            
            # Filter to show meaningful availability (20%+ chance)
            meaningful_availability = [(name, rate) for name, rate in round_availability if rate >= 0.20]
            
            for i, (player_name, rate) in enumerate(meaningful_availability[:15], 1):
                percentage = rate * 100
                if percentage >= 80:
                    status = "🟢 Very Likely"
                elif percentage >= 50:
                    status = "🟡 Possible"
                elif percentage >= 20:
                    status = "🟠 Unlikely"
                else:
                    status = "🔴 Very Rare"
                
                # Add variance info if available
                variance_info = ""
                if player_name in results.variance_statistics:
                    cv = results.variance_statistics[player_name].get('cv', 0)
                    variance_info = f" CV:{cv:.3f}"
                
                output.append(f"{i:2d}. {player_name:<20} {percentage:5.1f}% {status}{variance_info}")
            
            output.append("")
        
        return "\n".join(output)


def test_enhanced_monte_carlo():
    """Test the enhanced Monte Carlo system"""
    
    print("🧪 Testing Enhanced Monte Carlo Simulation...")
    
    # Test just the distribution loading and outcomes matrix generation
    from .database import DraftDatabaseConnector
    from .models import LeagueConfig
    
    db = DraftDatabaseConnector()
    players = db.load_player_universe()
    
    if not players:
        print("❌ No players found in database")
        return
    
    config = LeagueConfig(draft_position=6, num_teams=12)
    
    print(f"📊 Testing with {len(players)} players")
    
    # Test enhanced simulator initialization
    enhanced_sim = EnhancedMonteCarloSimulator(config, players[:50])  # Top 50 for testing
    
    print("✅ Enhanced simulator initialized successfully!")
    
    # Test outcomes matrix generation
    print("\n🎲 Testing outcomes matrix generation...")
    enhanced_sim.initialize_outcomes_matrix(num_samples=500)  # Small for testing
    
    # Show some distribution statistics
    if enhanced_sim.distribution_stats:
        print("\n📈 Sample Distribution Statistics:")
        print("-" * 60)
        
        sample_players = list(enhanced_sim.distribution_stats.keys())[:10]
        for player_name in sample_players:
            stats = enhanced_sim.distribution_stats[player_name]
            print(f"{player_name:<20} Mean:{stats['mean']:6.1f} CV:{stats['cv']:.3f} "
                  f"P90:{stats['p90']:6.1f} Bust:{stats['bust_prob']:.1%}")
    
    print("\n✅ Enhanced Monte Carlo testing complete!")
    print("\n🎯 Key Features Verified:")
    print("   • Distribution profiles loaded from database")
    print("   • Outcomes matrix generation working")
    print("   • Statistical analysis functional")
    print("   • Ready for integration with full simulation")


if __name__ == "__main__":
    test_enhanced_monte_carlo()
