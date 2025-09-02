"""
Pre-sampled Outcomes Matrix

This module generates and manages a matrix of pre-sampled player outcomes
for fast Monte Carlo simulation rollouts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import time
from pathlib import Path
from .distributions import DistributionFactory, PlayerDistribution


class OutcomeMatrix:
    """Pre-sampled outcomes matrix for fast simulation"""
    
    def __init__(self, num_simulations: int = 15000):
        self.num_simulations = num_simulations
        self.player_names: List[str] = []
        self.player_index_map: Dict[str, int] = {}
        self.outcomes: Optional[np.ndarray] = None  # Shape: (num_players, num_simulations)
        self.distributions: Dict[str, PlayerDistribution] = {}
        
    def generate_outcomes_matrix(self, season: int = 2025, seed: int = 42) -> None:
        """Generate the complete outcomes matrix"""
        print(f"🎲 Generating {self.num_simulations:,} outcome simulations...")
        
        # Load all player distributions
        factory = DistributionFactory()
        self.distributions = factory.load_all_player_distributions(season)
        
        # Set up player indexing
        self.player_names = list(self.distributions.keys())
        self.player_index_map = {name: i for i, name in enumerate(self.player_names)}
        
        num_players = len(self.player_names)
        print(f"📊 Generating outcomes for {num_players:,} players...")
        
        # Initialize outcomes matrix
        self.outcomes = np.zeros((num_players, self.num_simulations), dtype=np.float32)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate outcomes for each player
        start_time = time.time()
        
        for i, player_name in enumerate(self.player_names):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    eta = elapsed * (num_players / i - 1)
                    print(f"   Progress: {i:,}/{num_players:,} players ({100*i/num_players:.1f}%) - ETA: {eta:.1f}s")
            
            distribution = self.distributions[player_name]
            
            # Generate all outcomes for this player
            random_state = np.random.RandomState(seed + i)  # Unique seed per player
            
            for sim in range(self.num_simulations):
                outcome = distribution.sample_season_outcome(random_state)
                self.outcomes[i, sim] = outcome
        
        elapsed = time.time() - start_time
        print(f"✅ Generated {num_players * self.num_simulations:,} outcomes in {elapsed:.1f} seconds")
        print(f"   Matrix size: {self.outcomes.shape}")
        print(f"   Memory usage: {self.outcomes.nbytes / 1024 / 1024:.1f} MB")
    
    def get_player_outcomes(self, player_name: str) -> Optional[np.ndarray]:
        """Get all outcomes for a specific player"""
        if player_name not in self.player_index_map:
            return None
        
        player_idx = self.player_index_map[player_name]
        return self.outcomes[player_idx, :]
    
    def get_simulation_column(self, simulation_index: int) -> np.ndarray:
        """Get outcomes for all players in a specific simulation"""
        if self.outcomes is None:
            raise ValueError("Outcomes matrix not generated")
        
        return self.outcomes[:, simulation_index]
    
    def calculate_roster_score(self, player_names: List[str], simulation_index: int) -> float:
        """Calculate total score for a roster in a specific simulation"""
        total_score = 0.0
        
        for player_name in player_names:
            if player_name in self.player_index_map:
                player_idx = self.player_index_map[player_name]
                total_score += self.outcomes[player_idx, simulation_index]
        
        return total_score
    
    def get_outcome_statistics(self, player_name: str) -> Dict[str, float]:
        """Get statistical summary of a player's outcomes"""
        outcomes = self.get_player_outcomes(player_name)
        
        if outcomes is None:
            return {}
        
        return {
            'mean': float(np.mean(outcomes)),
            'std': float(np.std(outcomes)),
            'min': float(np.min(outcomes)),
            'max': float(np.max(outcomes)),
            'p25': float(np.percentile(outcomes, 25)),
            'p50': float(np.percentile(outcomes, 50)),
            'p75': float(np.percentile(outcomes, 75)),
            'cv': float(np.std(outcomes) / np.mean(outcomes)) if np.mean(outcomes) > 0 else 0
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the outcomes matrix to disk"""
        data = {
            'player_names': self.player_names,
            'player_index_map': self.player_index_map,
            'outcomes': self.outcomes,
            'num_simulations': self.num_simulations,
            'distributions': {name: dist for name, dist in self.distributions.items()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved outcomes matrix to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load the outcomes matrix from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.player_names = data['player_names']
        self.player_index_map = data['player_index_map']
        self.outcomes = data['outcomes']
        self.num_simulations = data['num_simulations']
        self.distributions = data.get('distributions', {})
        
        print(f"📂 Loaded outcomes matrix from {filepath}")
        print(f"   {len(self.player_names):,} players, {self.num_simulations:,} simulations")
    
    def validate_outcomes(self) -> None:
        """Validate that the outcomes match expected distributions"""
        print("🔍 Validating outcomes matrix...")
        
        sample_players = self.player_names[:5]  # Check first 5 players
        
        for player_name in sample_players:
            if player_name not in self.distributions:
                continue
                
            distribution = self.distributions[player_name]
            outcomes = self.get_player_outcomes(player_name)
            
            if outcomes is None:
                continue
            
            stats = self.get_outcome_statistics(player_name)
            
            print(f"\n📊 {player_name} ({distribution.position}):")
            print(f"   Expected Mean: {distribution.mean_projection:.1f}")
            print(f"   Actual Mean: {stats['mean']:.1f}")
            print(f"   Expected CV: {distribution.coefficient_of_variation:.1%}")
            print(f"   Actual CV: {stats['cv']:.1%}")
            print(f"   Range: {stats['min']:.1f} - {stats['max']:.1f}")
            
            # Check if mean is close (within 5%)
            mean_error = abs(stats['mean'] - distribution.mean_projection) / distribution.mean_projection
            if mean_error > 0.05:
                print(f"   ⚠️  Mean error: {mean_error:.1%}")
            else:
                print(f"   ✅ Mean error: {mean_error:.1%}")


def create_outcome_matrix(num_simulations: int = 15000, season: int = 2025) -> OutcomeMatrix:
    """Create and generate a new outcome matrix"""
    matrix = OutcomeMatrix(num_simulations)
    matrix.generate_outcomes_matrix(season)
    return matrix


def get_cached_outcome_matrix(cache_file: str = "outcome_matrix.pkl", regenerate: bool = False) -> OutcomeMatrix:
    """Get cached outcome matrix or generate if not exists"""
    cache_path = Path(cache_file)
    
    if cache_path.exists() and not regenerate:
        print(f"📂 Loading cached outcome matrix from {cache_file}")
        matrix = OutcomeMatrix()
        matrix.load_from_file(str(cache_path))
        return matrix
    else:
        print(f"🎲 Generating new outcome matrix...")
        matrix = create_outcome_matrix()
        matrix.save_to_file(str(cache_path))
        return matrix


if __name__ == "__main__":
    # Test the outcome matrix system
    print("🎲 Testing Outcome Matrix System")
    print("=" * 50)
    
    # Generate small matrix for testing
    matrix = OutcomeMatrix(num_simulations=1000)
    matrix.generate_outcomes_matrix()
    
    # Validate outcomes
    matrix.validate_outcomes()
    
    # Test roster scoring
    test_roster = matrix.player_names[:10]  # First 10 players
    test_score = matrix.calculate_roster_score(test_roster, 0)
    print(f"\n🏈 Test roster score (sim 0): {test_score:.1f}")
    
    print("\n✅ Outcome matrix system working!")
