"""
Enhanced Monte Carlo Simulator with Pre-sampled Outcomes

This module enhances the existing Monte Carlo simulator to use pre-sampled
player outcomes for faster and more realistic draft simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
import time

from .outcome_matrix import OutcomeMatrix, get_cached_outcome_matrix
from .models import LeagueConfig, Player as BasePlayer
from .draft_state import DraftState


class MonteCarloStrategy:
    """Base strategy interface for Monte Carlo simulation"""
    
    def select_pick(self, draft_state: DraftState, config: LeagueConfig) -> 'EnhancedPlayer':
        """Select the best available player given current draft state"""
        raise NotImplementedError


@dataclass 
class EnhancedPlayer(BasePlayer):
    """Enhanced player with variance profile"""
    coefficient_of_variation: float = 0.25
    floor_outcome: float = 0.0
    ceiling_outcome: float = 0.0
    outcome_samples: Optional[np.ndarray] = None


class EnhancedMonteCarloSimulator:
    """Monte Carlo simulator using pre-sampled outcome distributions"""
    
    def __init__(self, league_config: LeagueConfig):
        self.config = league_config
        self.outcome_matrix: Optional[OutcomeMatrix] = None
        self.enhanced_players: Dict[str, EnhancedPlayer] = {}
        
        # Initialize outcome matrix
        self._initialize_outcome_matrix()
    
    def _initialize_outcome_matrix(self):
        """Initialize the pre-sampled outcome matrix"""
        print("🎲 Initializing Enhanced Monte Carlo Simulator...")
        
        # Load or generate outcome matrix
        self.outcome_matrix = get_cached_outcome_matrix(
            cache_file="data/outcome_matrix_15k.pkl",
            regenerate=False  # Set to True to regenerate
        )
        
        # Create enhanced player objects
        print("📊 Creating enhanced player profiles...")
        self._create_enhanced_players()
    
    def _create_enhanced_players(self):
        """Create enhanced player objects with outcome data"""
        if not self.outcome_matrix:
            raise ValueError("Outcome matrix not initialized")
        
        for player_name in self.outcome_matrix.player_names:
            distribution = self.outcome_matrix.distributions.get(player_name)
            if not distribution:
                continue
            
            # Get outcome statistics
            stats = self.outcome_matrix.get_outcome_statistics(player_name)
            outcomes = self.outcome_matrix.get_player_outcomes(player_name)
            
            enhanced_player = EnhancedPlayer(
                name=player_name,
                position=distribution.position,
                team=getattr(distribution, 'team', 'UNK'),
                projection=distribution.mean_projection,
                adp=999,  # Will be loaded from database if needed
                adp_position=99,  # Will be loaded from database if needed
                bye_week=0,  # Will be loaded from database if needed
                coefficient_of_variation=distribution.coefficient_of_variation,
                floor_outcome=stats.get('p25', 0),
                ceiling_outcome=stats.get('p75', distribution.mean_projection),
                outcome_samples=outcomes
            )
            
            self.enhanced_players[player_name] = enhanced_player
        
        print(f"✅ Created {len(self.enhanced_players)} enhanced player profiles")
    
    def simulate_draft_with_outcomes(
        self, 
        strategy: MonteCarloStrategy,
        num_simulations: int = 25,
        rounds_to_simulate: int = 16
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using pre-sampled outcomes"""
        
        print(f"🎯 Running Enhanced Monte Carlo Simulation:")
        print(f"   • {num_simulations} draft simulations")
        print(f"   • {rounds_to_simulate} rounds each")
        print(f"   • {len(self.enhanced_players)} players with individual variance")
        
        results = []
        start_time = time.time()
        
        for sim in range(num_simulations):
            if sim % 5 == 0:
                print(f"   Simulation {sim+1}/{num_simulations}...")
            
            # Run single draft simulation
            draft_result = self._simulate_single_draft(
                strategy, 
                rounds_to_simulate,
                outcome_simulation_index=sim % self.outcome_matrix.num_simulations
            )
            
            results.append(draft_result)
        
        elapsed = time.time() - start_time
        print(f"✅ Completed {num_simulations} simulations in {elapsed:.1f} seconds")
        
        # Aggregate results
        return self._aggregate_simulation_results(results)
    
    def _simulate_single_draft(
        self, 
        strategy: MonteCarloStrategy,
        rounds_to_simulate: int,
        outcome_simulation_index: int
    ) -> Dict[str, Any]:
        """Simulate a single draft using specific outcome column"""
        
        # Initialize draft state with enhanced players
        available_players = list(self.enhanced_players.values())
        random.shuffle(available_players)  # Random draft order variation
        
        # Create draft state with correct constructor
        draft_state = DraftState(self.config, available_players)
        
        user_roster = []
        user_team_index = self.config.draft_position - 1
        
        # Simulate draft
        for round_num in range(1, rounds_to_simulate + 1):
            # Determine pick order for this round
            if round_num % 2 == 1:  # Odd rounds
                pick_order = list(range(self.config.num_teams))
            else:  # Even rounds (snake)
                pick_order = list(range(self.config.num_teams - 1, -1, -1))
            
            for pick_pos, team_index in enumerate(pick_order):
                available_players = draft_state.get_available_players()
                if not available_players:
                    break
                
                if team_index == user_team_index:
                    # User's pick - use strategy
                    pick = strategy.select_pick(draft_state, self.config)
                    user_roster.append(pick)
                else:
                    # AI opponent pick
                    pick = self._ai_opponent_pick(draft_state)
                
                # Remove picked player using draft_player method
                if pick:
                    draft_state.draft_player(pick)
        
        # Calculate roster score using specific outcome simulation
        roster_score = self._calculate_roster_outcome_score(user_roster, outcome_simulation_index)
        
        return {
            'roster': user_roster,
            'roster_score': roster_score,
            'outcome_simulation': outcome_simulation_index,
            'rounds_completed': rounds_to_simulate
        }
    
    def _ai_opponent_pick(self, draft_state: DraftState) -> EnhancedPlayer:
        """AI opponent pick with ADP + variance considerations"""
        available = draft_state.get_available_players()
        
        if not available:
            return available[0]  # Fallback
        
        # Weight by projection with some ADP noise
        candidates = available[:min(20, len(available))]  # Top 20 available
        
        # Add some randomness to ADP-based selection
        weights = []
        for player in candidates:
            base_weight = player.projection
            
            # Add variance consideration (slightly favor consistent players)
            consistency_bonus = (1 - player.coefficient_of_variation) * 10
            
            # Add some random noise
            noise = random.uniform(0.8, 1.2)
            
            final_weight = base_weight + consistency_bonus * noise
            weights.append(final_weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            selected_player = np.random.choice(candidates, p=weights)
        else:
            selected_player = random.choice(candidates)
        
        return selected_player
    
    def _calculate_roster_outcome_score(self, roster: List[EnhancedPlayer], outcome_index: int) -> float:
        """Calculate roster score using STARTING LINEUP ONLY (not bench players)"""
        if not roster:
            return 0.0
        
        # Create optimal starting lineup from roster
        starting_lineup = self._get_optimal_starting_lineup(roster)
        
        if not self.outcome_matrix:
            # Fallback to projections for STARTERS ONLY
            return sum(player.projection for player in starting_lineup)
        
        total_score = 0.0
        
        # Score STARTERS ONLY using outcome simulation
        for player in starting_lineup:
            if player.name in self.outcome_matrix.player_index_map:
                player_idx = self.outcome_matrix.player_index_map[player.name]
                outcome = self.outcome_matrix.outcomes[player_idx, outcome_index]
                total_score += outcome
            else:
                # Fallback to projection
                total_score += player.projection
        
        return total_score
    
    def _get_optimal_starting_lineup(self, roster: List[EnhancedPlayer]) -> List[EnhancedPlayer]:
        """Select optimal starting lineup from roster (QB:1, RB:2, WR:2, TE:1, FLEX:1, K:1, DEF:1)"""
        if not roster:
            return []
        
        # Sort players by position and projection
        by_position = {}
        for player in roster:
            pos = player.position.upper()
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Sort each position by projection (highest first)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.projection, reverse=True)
        
        starting_lineup = []
        
        # Fill required starting positions
        # QB: 1
        if 'QB' in by_position and by_position['QB']:
            starting_lineup.append(by_position['QB'][0])
        
        # RB: 2 
        if 'RB' in by_position:
            starting_lineup.extend(by_position['RB'][:2])
        
        # WR: 2
        if 'WR' in by_position:
            starting_lineup.extend(by_position['WR'][:2])
        
        # TE: 1
        if 'TE' in by_position and by_position['TE']:
            starting_lineup.append(by_position['TE'][0])
        
        # FLEX: 1 (best remaining RB/WR/TE)
        flex_candidates = []
        if 'RB' in by_position and len(by_position['RB']) > 2:
            flex_candidates.extend(by_position['RB'][2:])  # RB3+
        if 'WR' in by_position and len(by_position['WR']) > 2:
            flex_candidates.extend(by_position['WR'][2:])  # WR3+
        if 'TE' in by_position and len(by_position['TE']) > 1:
            flex_candidates.extend(by_position['TE'][1:])  # TE2+
        
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: p.projection)
            starting_lineup.append(best_flex)
        
        # K: 1
        if 'K' in by_position and by_position['K']:
            starting_lineup.append(by_position['K'][0])
        
        # DEF: 1
        if 'DEF' in by_position and by_position['DEF']:
            starting_lineup.append(by_position['DEF'][0])
        
        return starting_lineup
    
    def _aggregate_simulation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple simulations"""
        
        # Extract roster scores
        scores = [result['roster_score'] for result in results]
        
        # Player frequency analysis
        player_frequency = {}
        for result in results:
            for player in result['roster']:
                player_key = f"{player.name} ({player.position})"
                player_frequency[player_key] = player_frequency.get(player_key, 0) + 1
        
        # Calculate statistics
        aggregated = {
            'num_simulations': len(results),
            'roster_scores': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'p25': float(np.percentile(scores, 25)),
                'p50': float(np.percentile(scores, 50)),
                'p75': float(np.percentile(scores, 75))
            },
            'player_frequency': dict(sorted(player_frequency.items(), key=lambda x: x[1], reverse=True)),
            'sample_rosters': [
                {
                    'roster': [f"{p.name} ({p.position})" for p in result['roster']],
                    'score': result['roster_score']
                }
                for result in results[:3]  # Show first 3 as examples
            ]
        }
        
        return aggregated


# Integration with existing strategy classes
class OutcomeAwareStrategy(MonteCarloStrategy):
    """Strategy that considers pre-sampled outcome variance"""
    
    def __init__(self, risk_tolerance: float = 0.5):
        self.risk_tolerance = risk_tolerance  # 0=risk-averse, 1=risk-seeking
    
    def select_pick(self, draft_state: DraftState, config: LeagueConfig) -> EnhancedPlayer:
        """Select pick considering outcome variance"""
        available = draft_state.get_available_players()
        
        if not available:
            return available[0]
        
        # Evaluate candidates
        candidates = available[:min(10, len(available))]
        best_player = None
        best_score = float('-inf')
        
        for player in candidates:
            if not isinstance(player, EnhancedPlayer):
                continue
            
            # Base value score
            value_score = player.projection
            
            # Adjust for variance preference
            if hasattr(player, 'coefficient_of_variation'):
                cv = player.coefficient_of_variation
                
                if self.risk_tolerance < 0.5:
                    # Risk-averse: penalty for high variance
                    variance_adjustment = -(cv - 0.20) * 50
                else:
                    # Risk-seeking: bonus for high variance (upside)
                    variance_adjustment = (cv - 0.20) * 20
                
                value_score += variance_adjustment
            
            # Positional scarcity (simplified)
            position_count = sum(1 for p in available if p.position == player.position)
            scarcity_bonus = max(0, (20 - position_count) * 2)
            value_score += scarcity_bonus
            
            if value_score > best_score:
                best_score = value_score
                best_player = player
        
        return best_player or available[0]


if __name__ == "__main__":
    # Test enhanced Monte Carlo system
    print("🎲 Testing Enhanced Monte Carlo System")
    print("=" * 50)
    
    # Create test configuration
    config = LeagueConfig(
        num_teams=12,
        draft_position=6,
        roster_size=16
    )
    
    # Create simulator
    simulator = EnhancedMonteCarloSimulator(config)
    
    # Test strategy
    strategy = OutcomeAwareStrategy(risk_tolerance=0.3)  # Risk-averse
    
    # Run small test simulation
    results = simulator.simulate_draft_with_outcomes(
        strategy=strategy,
        num_simulations=5,
        rounds_to_simulate=3
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Mean Score: {results['roster_scores']['mean']:.1f}")
    print(f"   Score Range: {results['roster_scores']['min']:.1f} - {results['roster_scores']['max']:.1f}")
    print(f"   Most Drafted: {list(results['player_frequency'].keys())[:3]}")
    
    print("\n✅ Enhanced Monte Carlo system working!")