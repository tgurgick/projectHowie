"""
Player Outcome Distribution Models

This module implements various distribution models for simulating player season outcomes,
including truncated normal, lognormal, and injury overlay models.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
from .distribution_database import PlayerDistributionProfile


@dataclass
class SeasonOutcome:
    """Single season outcome for a player"""
    total_points: float
    games_played: int
    games_missed: int
    injury_type: str  # 'none', 'minor', 'major'
    weekly_points: List[float]  # 17-week breakdown


class BaseDistribution(ABC):
    """Abstract base class for player distributions"""
    
    def __init__(self, profile: PlayerDistributionProfile):
        self.profile = profile
        self.mean = max(0.1, profile.mean_projection)  # Ensure positive mean
        self.cv = max(0.01, profile.coefficient_of_variation)  # Ensure positive CV
        
        # Ensure injury probabilities sum to 1 and are valid
        injury_probs = [
            profile.injury_prob_healthy,
            profile.injury_prob_minor, 
            profile.injury_prob_major
        ]
        
        # Normalize probabilities if they don't sum to 1
        prob_sum = sum(injury_probs)
        if prob_sum <= 0:
            self.injury_probs = [0.8, 0.15, 0.05]  # Default
        elif abs(prob_sum - 1.0) > 0.01:
            self.injury_probs = [p / prob_sum for p in injury_probs]
        else:
            self.injury_probs = injury_probs
    
    @abstractmethod
    def sample_base_points(self, size: int = 1) -> np.ndarray:
        """Sample base points (before injury adjustment)"""
        pass
    
    @abstractmethod
    def sample_season_outcome(self, include_weekly: bool = False) -> SeasonOutcome:
        """Sample complete season outcome"""
        pass
    
    def sample_injury_pattern(self) -> Tuple[int, str]:
        """Sample injury pattern for the season"""
        # Sample injury type
        injury_choice = np.random.choice(3, p=self.injury_probs)
        
        if injury_choice == 0:  # Healthy
            games_missed = 0
            injury_type = 'none'
        elif injury_choice == 1:  # Minor injury
            games_missed = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            injury_type = 'minor'
        else:  # Major injury
            games_missed = np.random.choice([4, 5, 6, 8, 10, 17], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
            injury_type = 'major'
        
        return games_missed, injury_type
    
    def apply_injury_adjustment(self, base_points: float, games_missed: int) -> float:
        """Apply injury adjustment to base points"""
        games_played = max(0, 17 - games_missed)
        return base_points * (games_played / 17.0)


class TruncatedNormalDistribution(BaseDistribution):
    """Truncated normal distribution with injury overlay"""
    
    def __init__(self, profile: PlayerDistributionProfile):
        super().__init__(profile)
        self.std = self.cv * self.mean
        
        # Handle edge case where std is 0 or very small
        if self.std <= 0.001:
            self.std = max(0.1, self.mean * 0.01)  # Minimum 1% variance
        
        # Set truncation bounds (non-negative, reasonable upper bound)
        self.lower_bound = 0
        self.upper_bound = self.mean + 3 * self.std  # 99.7% within 3 sigma
        
        # Create truncated normal distribution
        self.truncated_normal = stats.truncnorm(
            (self.lower_bound - self.mean) / self.std,
            (self.upper_bound - self.mean) / self.std,
            loc=self.mean,
            scale=self.std
        )
    
    def sample_base_points(self, size: int = 1) -> np.ndarray:
        """Sample base points from truncated normal distribution"""
        return self.truncated_normal.rvs(size=size)
    
    def sample_season_outcome(self, include_weekly: bool = False) -> SeasonOutcome:
        """Sample complete season outcome with injury overlay"""
        # Sample base season total
        base_points = self.sample_base_points(1)[0]
        
        # Sample injury pattern
        games_missed, injury_type = self.sample_injury_pattern()
        
        # Apply injury adjustment
        final_points = self.apply_injury_adjustment(base_points, games_missed)
        games_played = 17 - games_missed
        
        # Generate weekly breakdown if requested
        weekly_points = []
        if include_weekly:
            weekly_points = self._generate_weekly_breakdown(final_points, games_played)
        
        return SeasonOutcome(
            total_points=final_points,
            games_played=games_played,
            games_missed=games_missed,
            injury_type=injury_type,
            weekly_points=weekly_points
        )
    
    def _generate_weekly_breakdown(self, total_points: float, games_played: int) -> List[float]:
        """Generate realistic weekly point distribution"""
        if games_played == 0:
            return [0.0] * 17
        
        avg_weekly = total_points / games_played
        weekly_cv = 0.4  # Weekly variance is higher than seasonal
        weekly_std = weekly_cv * avg_weekly
        
        # Sample weekly points for games played
        weekly_played = np.maximum(0, np.random.normal(avg_weekly, weekly_std, games_played))
        
        # Scale to match total
        scale_factor = total_points / np.sum(weekly_played) if np.sum(weekly_played) > 0 else 1
        weekly_played *= scale_factor
        
        # Randomly place zeros for missed games
        weekly_points = [0.0] * 17
        played_weeks = np.random.choice(17, games_played, replace=False)
        
        for i, week in enumerate(played_weeks):
            weekly_points[week] = weekly_played[i]
        
        return weekly_points


class LognormalDistribution(BaseDistribution):
    """Lognormal distribution with injury overlay"""
    
    def __init__(self, profile: PlayerDistributionProfile):
        super().__init__(profile)
        
        # Handle edge case where CV is 0 or very small
        cv_safe = max(0.01, self.cv)  # Minimum 1% CV
        
        # Convert CV and mean to lognormal parameters
        # For lognormal: var = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        # CV^2 = var/mean^2 = exp(sigma^2) - 1
        self.sigma_ln = np.sqrt(np.log(1 + cv_safe**2))
        self.mu_ln = np.log(max(0.1, self.mean)) - self.sigma_ln**2 / 2  # Ensure positive mean
    
    def sample_base_points(self, size: int = 1) -> np.ndarray:
        """Sample base points from lognormal distribution"""
        return np.random.lognormal(self.mu_ln, self.sigma_ln, size)
    
    def sample_season_outcome(self, include_weekly: bool = False) -> SeasonOutcome:
        """Sample complete season outcome with injury overlay"""
        # Sample base season total
        base_points = self.sample_base_points(1)[0]
        
        # Sample injury pattern
        games_missed, injury_type = self.sample_injury_pattern()
        
        # Apply injury adjustment
        final_points = self.apply_injury_adjustment(base_points, games_missed)
        games_played = 17 - games_missed
        
        # Generate weekly breakdown if requested
        weekly_points = []
        if include_weekly:
            weekly_points = self._generate_weekly_breakdown(final_points, games_played)
        
        return SeasonOutcome(
            total_points=final_points,
            games_played=games_played,
            games_missed=games_missed,
            injury_type=injury_type,
            weekly_points=weekly_points
        )
    
    def _generate_weekly_breakdown(self, total_points: float, games_played: int) -> List[float]:
        """Generate realistic weekly point distribution using lognormal"""
        if games_played == 0:
            return [0.0] * 17
        
        avg_weekly = total_points / games_played
        weekly_cv = 0.45  # Higher variance for lognormal
        
        # Lognormal parameters for weekly distribution
        sigma_weekly = np.sqrt(np.log(1 + weekly_cv**2))
        mu_weekly = np.log(avg_weekly) - sigma_weekly**2 / 2
        
        # Sample weekly points for games played
        weekly_played = np.random.lognormal(mu_weekly, sigma_weekly, games_played)
        
        # Scale to match total
        scale_factor = total_points / np.sum(weekly_played) if np.sum(weekly_played) > 0 else 1
        weekly_played *= scale_factor
        
        # Randomly place zeros for missed games
        weekly_points = [0.0] * 17
        played_weeks = np.random.choice(17, games_played, replace=False)
        
        for i, week in enumerate(played_weeks):
            weekly_points[week] = weekly_played[i]
        
        return weekly_points


class DistributionFactory:
    """Factory for creating appropriate distribution models"""
    
    @staticmethod
    def create_distribution(profile: PlayerDistributionProfile) -> BaseDistribution:
        """Create appropriate distribution based on profile settings"""
        
        if profile.distribution_type == 'lognormal':
            return LognormalDistribution(profile)
        elif profile.distribution_type == 'truncated_normal':
            return TruncatedNormalDistribution(profile)
        else:
            # Default to truncated normal
            return TruncatedNormalDistribution(profile)
    
    @staticmethod
    def get_recommended_distribution_type(position: str, cv: float) -> str:
        """Recommend distribution type based on position and variance"""
        
        # High-variance positions benefit from lognormal (captures skewness)
        if position in ['RB', 'WR'] and cv > 0.25:
            return 'lognormal'
        
        # Lower variance positions work well with truncated normal
        elif position in ['QB', 'TE', 'K', 'DEF']:
            return 'truncated_normal'
        
        # Default
        return 'truncated_normal'


class OutcomesMatrixGenerator:
    """Generate pre-sampled outcomes matrix for fast simulation"""
    
    def __init__(self, player_profiles: List[PlayerDistributionProfile], num_samples: int = 15000):
        self.player_profiles = player_profiles
        self.num_samples = num_samples
        self.distributions = {
            profile.player_name: DistributionFactory.create_distribution(profile)
            for profile in player_profiles
        }
        self.player_index_map = {
            profile.player_name: i for i, profile in enumerate(player_profiles)
        }
    
    def generate_outcomes_matrix(self, include_weekly: bool = False) -> np.ndarray:
        """Generate pre-sampled outcomes matrix [Players × Simulations]"""
        
        num_players = len(self.player_profiles)
        outcomes_matrix = np.zeros((num_players, self.num_samples))
        
        print(f"🎲 Generating {self.num_samples:,} outcomes for {num_players} players...")
        
        for i, profile in enumerate(self.player_profiles):
            if i % 50 == 0:  # Progress indicator
                print(f"   Processing player {i+1}/{num_players}: {profile.player_name}")
            
            distribution = self.distributions[profile.player_name]
            
            # Generate all samples for this player
            for j in range(self.num_samples):
                outcome = distribution.sample_season_outcome(include_weekly=include_weekly)
                outcomes_matrix[i, j] = outcome.total_points
        
        print(f"✅ Generated outcomes matrix: {outcomes_matrix.shape}")
        return outcomes_matrix
    
    def save_outcomes_to_database(self, outcomes_matrix: np.ndarray, db_manager):
        """Save pre-sampled outcomes to database for caching"""
        
        import hashlib
        
        for i, profile in enumerate(self.player_profiles):
            # Create cache key based on distribution parameters
            cache_data = {
                'mean': profile.mean_projection,
                'cv': profile.coefficient_of_variation,
                'injury_probs': [profile.injury_prob_healthy, profile.injury_prob_minor, profile.injury_prob_major],
                'distribution_type': profile.distribution_type,
                'num_samples': self.num_samples
            }
            cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
            
            # Convert outcomes to JSON
            player_outcomes = outcomes_matrix[i, :].tolist()
            outcomes_json = json.dumps(player_outcomes)
            
            # Save to database
            import sqlite3
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO player_outcomes_cache (
                    player_name, team, season, outcomes_data, num_samples,
                    sample_method, cache_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.player_name, profile.team, profile.season,
                outcomes_json, self.num_samples, 'monte_carlo', cache_key
            ))
            
            conn.commit()
            conn.close()
    
    def calculate_distribution_stats(self, outcomes_matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each player's distribution"""
        
        stats_summary = {}
        
        for i, profile in enumerate(self.player_profiles):
            player_outcomes = outcomes_matrix[i, :]
            
            stats_summary[profile.player_name] = {
                'mean': np.mean(player_outcomes),
                'median': np.median(player_outcomes),
                'std': np.std(player_outcomes),
                'cv': np.std(player_outcomes) / np.mean(player_outcomes),
                'min': np.min(player_outcomes),
                'max': np.max(player_outcomes),
                'p25': np.percentile(player_outcomes, 25),
                'p75': np.percentile(player_outcomes, 75),
                'p90': np.percentile(player_outcomes, 90),
                'p95': np.percentile(player_outcomes, 95),
                'zero_weeks_prob': np.mean(player_outcomes == 0),  # Season-ending injury prob
                'upside_prob': np.mean(player_outcomes > profile.mean_projection * 1.2),  # 20% upside
                'bust_prob': np.mean(player_outcomes < profile.mean_projection * 0.7)   # 30% downside
            }
        
        return stats_summary


def test_distributions():
    """Test the distribution system"""
    
    from .distribution_database import DistributionDatabaseManager, PlayerDistributionProfile
    
    print("🧪 Testing Player Distribution System...")
    
    # Create sample profile
    test_profile = PlayerDistributionProfile(
        player_name="Josh Allen",
        position="QB",
        team="BUF",
        season=2025,
        mean_projection=350.0,
        coefficient_of_variation=0.18,
        distribution_type='truncated_normal',
        injury_prob_healthy=0.85,
        injury_prob_minor=0.12,
        injury_prob_major=0.03
    )
    
    # Test truncated normal distribution
    print("\n📊 Testing Truncated Normal Distribution:")
    normal_dist = TruncatedNormalDistribution(test_profile)
    
    samples = []
    for _ in range(1000):
        outcome = normal_dist.sample_season_outcome()
        samples.append(outcome.total_points)
    
    print(f"   Mean: {np.mean(samples):.1f} (target: {test_profile.mean_projection:.1f})")
    print(f"   Std:  {np.std(samples):.1f}")
    print(f"   CV:   {np.std(samples)/np.mean(samples):.3f} (target: {test_profile.coefficient_of_variation:.3f})")
    
    # Test lognormal distribution  
    print("\n📊 Testing Lognormal Distribution:")
    test_profile.distribution_type = 'lognormal'
    log_dist = LognormalDistribution(test_profile)
    
    samples = []
    for _ in range(1000):
        outcome = log_dist.sample_season_outcome()
        samples.append(outcome.total_points)
    
    print(f"   Mean: {np.mean(samples):.1f} (target: {test_profile.mean_projection:.1f})")
    print(f"   Std:  {np.std(samples):.1f}")
    print(f"   CV:   {np.std(samples)/np.mean(samples):.3f} (target: {test_profile.coefficient_of_variation:.3f})")
    
    # Test injury patterns
    print("\n🏥 Testing Injury Patterns:")
    injury_outcomes = {'none': 0, 'minor': 0, 'major': 0}
    games_missed_total = []
    
    for _ in range(1000):
        outcome = normal_dist.sample_season_outcome()
        injury_outcomes[outcome.injury_type] += 1
        games_missed_total.append(outcome.games_missed)
    
    print(f"   Healthy: {injury_outcomes['none']/1000:.1%} (target: {test_profile.injury_prob_healthy:.1%})")
    print(f"   Minor:   {injury_outcomes['minor']/1000:.1%} (target: {test_profile.injury_prob_minor:.1%})")
    print(f"   Major:   {injury_outcomes['major']/1000:.1%} (target: {test_profile.injury_prob_major:.1%})")
    print(f"   Avg games missed: {np.mean(games_missed_total):.1f}")
    
    print("\n✅ Distribution testing complete!")


if __name__ == "__main__":
    test_distributions()
