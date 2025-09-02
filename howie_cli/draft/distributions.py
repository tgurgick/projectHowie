"""
Player Outcome Distribution Models
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path


class DistributionType(Enum):
    """Types of outcome distributions"""
    TRUNCATED_NORMAL = "truncated_normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"


@dataclass
class PlayerDistribution:
    """Complete distribution profile for a player"""
    player_name: str
    position: str
    mean_projection: float
    coefficient_of_variation: float
    
    # Distribution parameters
    distribution_type: DistributionType = DistributionType.TRUNCATED_NORMAL
    
    # Injury/availability parameters
    injury_prob_healthy: float = 0.85  # P(0 games missed)
    injury_prob_minor: float = 0.12    # P(1-3 games missed)
    injury_prob_major: float = 0.03    # P(4+ games missed)
    
    # Position-specific bounds
    floor_cap: float = 0.0
    ceiling_cap: float = 1000.0
    
    def __post_init__(self):
        """Set position-specific caps after initialization"""
        self._set_position_caps()
    
    def _set_position_caps(self):
        """Set realistic floors and ceilings by position"""
        pos = self.position.lower()
        
        # Use more generous caps to preserve individual variance
        # Apply caps only for extreme outliers
        if pos == 'qb':
            # QBs: Very generous caps to preserve variance
            self.floor_cap = max(0, self.mean_projection * 0.10)
            self.ceiling_cap = min(600, self.mean_projection * 2.50)
        elif pos == 'rb':
            # RBs: Generous caps for high variance
            self.floor_cap = max(0, self.mean_projection * 0.05)
            self.ceiling_cap = min(500, self.mean_projection * 2.50)
        elif pos == 'wr':
            # WRs: Generous caps for high variance
            self.floor_cap = max(0, self.mean_projection * 0.10)
            self.ceiling_cap = min(450, self.mean_projection * 2.50)
        elif pos == 'te':
            # TEs: Generous caps for high variance
            self.floor_cap = max(0, self.mean_projection * 0.05)
            self.ceiling_cap = min(350, self.mean_projection * 2.50)
        elif pos in ['k', 'kicker']:
            # Kickers: More constrained but still allow variance
            self.floor_cap = max(50, self.mean_projection * 0.50)
            self.ceiling_cap = min(200, self.mean_projection * 1.60)
        elif pos in ['dst', 'def']:
            # Defense: More constrained but allow variance
            self.floor_cap = max(20, self.mean_projection * 0.30)
            self.ceiling_cap = min(250, self.mean_projection * 2.00)
        else:
            # Default caps
            self.floor_cap = max(0, self.mean_projection * 0.10)
            self.ceiling_cap = self.mean_projection * 2.50
    
    def sample_season_outcome(self, random_state: Optional[np.random.RandomState] = None) -> float:
        """Sample a single season outcome"""
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Sample base fantasy points
        base_points = self._sample_base_points(random_state)
        
        # Apply injury/availability overlay
        final_points = self._apply_injury_overlay(base_points, random_state)
        
        # Apply position caps
        return max(self.floor_cap, min(self.ceiling_cap, final_points))
    
    def _sample_base_points(self, random_state: np.random.RandomState) -> float:
        """Sample base fantasy points from the distribution"""
        std_dev = self.coefficient_of_variation * self.mean_projection
        
        if self.distribution_type == DistributionType.TRUNCATED_NORMAL:
            # Truncated normal (most common)
            sample = random_state.normal(self.mean_projection, std_dev)
            return max(0, sample)
        else:
            # Fallback to normal
            return max(0, random_state.normal(self.mean_projection, std_dev))
    
    def _apply_injury_overlay(self, base_points: float, random_state: np.random.RandomState) -> float:
        """Apply injury/availability effects"""
        injury_roll = random_state.random()
        
        if injury_roll < self.injury_prob_healthy:
            # Healthy season - no games missed
            return base_points
        elif injury_roll < self.injury_prob_healthy + self.injury_prob_minor:
            # Minor injury - miss 1-3 games
            games_missed = random_state.randint(1, 4)
            return base_points * (17 - games_missed) / 17
        else:
            # Major injury - miss 4+ games
            games_missed = random_state.randint(4, 9)
            return base_points * (17 - games_missed) / 17


class DistributionFactory:
    """Factory to create player distributions from database"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_database_path()
    
    def _get_database_path(self) -> str:
        """Get database path using ProjectHowie conventions"""
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        db_path = project_root / "data" / "fantasy_ppr.db"
        
        if db_path.exists():
            return str(db_path)
        
        # Fallback
        fallback = Path("data/fantasy_ppr.db")
        if fallback.exists():
            return str(fallback)
        
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    def load_all_player_distributions(self, season: int = 2025) -> Dict[str, PlayerDistribution]:
        """Load all player distributions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                pd.player_name,
                pd.position,
                pd.mean_projection,
                pd.coefficient_of_variation,
                pd.injury_prob_healthy,
                pd.injury_prob_minor,
                pd.injury_prob_major,
                pd.distribution_type
            FROM player_distributions pd
            WHERE pd.season = ?
            ORDER BY pd.mean_projection DESC
        """
        
        cursor.execute(query, (season,))
        distributions = {}
        
        for row in cursor.fetchall():
            name, pos, mean_proj, cv, inj_healthy, inj_minor, inj_major, dist_type = row
            
            # Convert distribution type string to enum
            try:
                distribution_type = DistributionType(dist_type or "truncated_normal")
            except ValueError:
                distribution_type = DistributionType.TRUNCATED_NORMAL
            
            distribution = PlayerDistribution(
                player_name=name,
                position=pos,
                mean_projection=mean_proj,
                coefficient_of_variation=cv,
                distribution_type=distribution_type,
                injury_prob_healthy=inj_healthy or 0.85,
                injury_prob_minor=inj_minor or 0.12,
                injury_prob_major=inj_major or 0.03
            )
            
            distributions[name] = distribution
        
        conn.close()
        print(f"✅ Loaded {len(distributions)} player distributions")
        return distributions