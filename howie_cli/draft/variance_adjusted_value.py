"""
Variance-Adjusted Value Calculator
Incorporates upside/downside and simulation variance into player valuations
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

from .models import Player
from .value_calculator import ValueCalculator
from .distributions import PlayerDistribution, DistributionFactory


@dataclass
class VarianceAdjustedMetrics:
    """Enhanced metrics that include variance considerations"""
    base_vorp: float
    variance_adjusted_vorp: float
    upside_premium: float
    floor_penalty: float
    ceiling_value: float
    confidence_interval_range: float
    risk_adjusted_score: float


class VarianceAdjustedValueCalculator(ValueCalculator):
    """Enhanced value calculator that considers variance and upside"""
    
    def __init__(self, player_universe: List[Player], distribution_factory: Optional[DistributionFactory] = None):
        super().__init__(player_universe)
        self.distribution_factory = distribution_factory or DistributionFactory()
        
        # Cache player distributions for performance
        self._distribution_cache = {}
        self._replacement_distributions = self._calculate_replacement_distributions()
    
    def _calculate_replacement_distributions(self) -> Dict[str, PlayerDistribution]:
        """Calculate distribution profiles for replacement level players"""
        replacement_distributions = {}
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            replacement_level = self.replacement_levels.get(position, 0)
            
            # Find players near replacement level
            if position == 'DEF':
                pos_players = [p for p in self.players if p.position.upper() in ['DEF', 'DST']]
            else:
                pos_players = [p for p in self.players if p.position.upper() == position]
            
            pos_players.sort(key=lambda x: x.projection, reverse=True)
            
            # Get replacement level index
            replacement_index = {
                'QB': 12, 'RB': 36, 'WR': 36, 'TE': 12, 'K': 12, 'DEF': 12
            }.get(position, 24)
            
            if len(pos_players) > replacement_index:
                replacement_player = pos_players[replacement_index]
                replacement_distributions[position] = self.distribution_factory.get_distribution(replacement_player)
        
        return replacement_distributions
    
    def calculate_variance_adjusted_vorp(
        self, 
        player: Player, 
        risk_tolerance: float = 0.5,
        draft_context: Optional[Dict[str, Any]] = None
    ) -> VarianceAdjustedMetrics:
        """
        Calculate VORP adjusted for variance and upside considerations
        
        Args:
            player: Player to evaluate
            risk_tolerance: 0=conservative, 0.5=neutral, 1=aggressive upside-seeking
            draft_context: Additional context like round number, roster state
        """
        
        # Get base VORP
        base_vorp = self.calculate_vorp(player)
        
        # Get player distribution
        player_dist = self._get_player_distribution(player)
        position_key = player.position.upper()
        if position_key == 'DST':
            position_key = 'DEF'
        
        # Get replacement distribution
        replacement_dist = self._replacement_distributions.get(position_key)
        
        if not player_dist or not replacement_dist:
            # Fallback to basic VORP if distributions unavailable
            return VarianceAdjustedMetrics(
                base_vorp=base_vorp,
                variance_adjusted_vorp=base_vorp,
                upside_premium=0,
                floor_penalty=0,
                ceiling_value=player.projection,
                confidence_interval_range=0,
                risk_adjusted_score=base_vorp
            )
        
        # Sample outcomes to calculate variance metrics
        num_samples = 1000
        player_samples = [player_dist.sample_season_outcome() for _ in range(num_samples)]
        replacement_samples = [replacement_dist.sample_season_outcome() for _ in range(num_samples)]
        
        # Calculate percentiles
        player_p10 = np.percentile(player_samples, 10)  # Floor
        player_p50 = np.percentile(player_samples, 50)  # Median
        player_p90 = np.percentile(player_samples, 90)  # Ceiling
        
        replacement_p50 = np.percentile(replacement_samples, 50)
        
        # Calculate variance-adjusted metrics
        ceiling_value = player_p90
        floor_value = player_p10
        confidence_range = player_p90 - player_p10
        
        # Upside premium: value of 90th percentile outcome vs median
        upside_premium = max(0, player_p90 - player_p50)
        
        # Floor penalty: risk of underperforming median
        floor_penalty = max(0, player_p50 - player_p10)
        
        # Risk-adjusted VORP based on risk tolerance
        if risk_tolerance <= 0.3:
            # Conservative: weight floor heavily
            risk_percentile = np.percentile(player_samples, 25)
            replacement_percentile = np.percentile(replacement_samples, 25)
        elif risk_tolerance >= 0.7:
            # Aggressive: weight ceiling heavily
            risk_percentile = np.percentile(player_samples, 75)
            replacement_percentile = np.percentile(replacement_samples, 75)
        else:
            # Neutral: use median
            risk_percentile = player_p50
            replacement_percentile = replacement_p50
        
        variance_adjusted_vorp = max(0, risk_percentile - replacement_percentile)
        
        # Calculate comprehensive risk-adjusted score
        risk_adjusted_score = self._calculate_risk_adjusted_score(
            player_samples, replacement_samples, risk_tolerance, draft_context
        )
        
        return VarianceAdjustedMetrics(
            base_vorp=base_vorp,
            variance_adjusted_vorp=variance_adjusted_vorp,
            upside_premium=upside_premium,
            floor_penalty=floor_penalty,
            ceiling_value=ceiling_value,
            confidence_interval_range=confidence_range,
            risk_adjusted_score=risk_adjusted_score
        )
    
    def _calculate_risk_adjusted_score(
        self, 
        player_samples: List[float], 
        replacement_samples: List[float],
        risk_tolerance: float,
        draft_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate comprehensive risk-adjusted value score"""
        
        # Calculate advantage at different percentiles
        percentiles = [10, 25, 50, 75, 90]
        advantages = []
        
        for p in percentiles:
            player_p = np.percentile(player_samples, p)
            replacement_p = np.percentile(replacement_samples, p)
            advantages.append(max(0, player_p - replacement_p))
        
        # Weight percentiles based on risk tolerance
        if risk_tolerance <= 0.3:
            # Conservative: heavily weight floor outcomes
            weights = [0.4, 0.3, 0.2, 0.075, 0.025]
        elif risk_tolerance >= 0.7:
            # Aggressive: heavily weight ceiling outcomes
            weights = [0.025, 0.075, 0.2, 0.3, 0.4]
        else:
            # Balanced: even weighting with slight median bias
            weights = [0.15, 0.2, 0.3, 0.2, 0.15]
        
        # Adjust weights based on draft context
        if draft_context:
            weights = self._adjust_weights_for_context(weights, draft_context)
        
        # Calculate weighted score
        risk_adjusted_score = sum(adv * weight for adv, weight in zip(advantages, weights))
        
        return risk_adjusted_score
    
    def _adjust_weights_for_context(self, weights: List[float], context: Dict[str, Any]) -> List[float]:
        """Adjust percentile weights based on draft context"""
        
        round_number = context.get('round_number', 5)
        roster_strength = context.get('roster_strength', 0.5)  # 0-1 scale
        
        # Early rounds: be more conservative (weight floor higher)
        if round_number <= 3:
            # Shift weight from ceiling to floor
            ceiling_weight = weights[4] * 0.3  # Take 30% from ceiling
            weights[4] -= ceiling_weight
            weights[0] += ceiling_weight * 0.6  # Give 60% to floor
            weights[1] += ceiling_weight * 0.4  # Give 40% to 25th percentile
        
        # Late rounds with weak roster: take more risks
        elif round_number >= 8 and roster_strength < 0.4:
            # Shift weight from floor to ceiling
            floor_weight = weights[0] * 0.4
            weights[0] -= floor_weight
            weights[4] += floor_weight * 0.7  # Give most to ceiling
            weights[3] += floor_weight * 0.3  # Some to 75th percentile
        
        return weights
    
    def _get_player_distribution(self, player: Player) -> Optional[PlayerDistribution]:
        """Get cached player distribution"""
        if player.name not in self._distribution_cache:
            self._distribution_cache[player.name] = self.distribution_factory.get_distribution(player)
        
        return self._distribution_cache[player.name]


class EnhancedPickScorer:
    """Enhanced scoring system that incorporates variance-adjusted values"""
    
    def __init__(self, variance_calculator: VarianceAdjustedValueCalculator):
        self.variance_calc = variance_calculator
    
    def calculate_enhanced_score(
        self,
        player: Player,
        base_metrics: Dict[str, float],
        risk_tolerance: float = 0.5,
        draft_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate enhanced score incorporating variance considerations
        
        Returns:
            Dict with enhanced scoring components
        """
        
        # Get variance-adjusted metrics
        variance_metrics = self.variance_calc.calculate_variance_adjusted_vorp(
            player, risk_tolerance, draft_context
        )
        
        # Base scoring components (from existing system)
        scarcity = base_metrics.get('scarcity', 0)
        roster_fit = base_metrics.get('roster_fit', 0)
        sos_advantage = base_metrics.get('sos_advantage', 0)
        starter_status = base_metrics.get('starter_status', 0)
        injury_risk = base_metrics.get('injury_risk', 0)
        
        # Enhanced scoring weights
        enhanced_score = (
            # Core value (40% - increased from 30%)
            variance_metrics.risk_adjusted_score * 0.40 +
            
            # Positional factors (30%)
            scarcity * 0.15 +
            roster_fit * 0.15 +
            
            # Player factors (30%)
            sos_advantage * 0.10 +
            starter_status * 0.15 +
            injury_risk * 0.05
        )
        
        return {
            'enhanced_score': enhanced_score,
            'base_vorp': variance_metrics.base_vorp,
            'variance_adjusted_vorp': variance_metrics.variance_adjusted_vorp,
            'upside_premium': variance_metrics.upside_premium,
            'floor_penalty': variance_metrics.floor_penalty,
            'ceiling_value': variance_metrics.ceiling_value,
            'confidence_range': variance_metrics.confidence_interval_range,
            'risk_adjusted_score': variance_metrics.risk_adjusted_score
        }


# Integration helper functions
def get_risk_tolerance_for_context(round_number: int, roster_state: Dict[str, Any]) -> float:
    """
    Determine appropriate risk tolerance based on draft context
    
    Returns:
        Risk tolerance between 0 (conservative) and 1 (aggressive)
    """
    
    base_tolerance = 0.5  # Neutral
    
    # Early rounds: be more conservative
    if round_number <= 2:
        base_tolerance = 0.3
    elif round_number <= 4:
        base_tolerance = 0.4
    
    # Late rounds: take more risks
    elif round_number >= 10:
        base_tolerance = 0.7
    elif round_number >= 8:
        base_tolerance = 0.6
    
    # Adjust based on roster strength
    roster_strength = roster_state.get('strength_percentile', 0.5)
    
    if roster_strength < 0.3:
        # Weak roster: take more risks
        base_tolerance = min(1.0, base_tolerance + 0.2)
    elif roster_strength > 0.7:
        # Strong roster: be more conservative
        base_tolerance = max(0.0, base_tolerance - 0.15)
    
    return base_tolerance
