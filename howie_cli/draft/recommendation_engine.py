"""
Pick recommendation engine for draft analysis
Generates top 10 picks per round with enhanced evaluation factors
"""

from typing import List, Dict, Any
from .models import Player, Roster, PickRecommendation, LeagueConfig
from .value_calculator import ValueCalculator, ScarcityAnalyzer
from .variance_adjusted_value import VarianceAdjustedValueCalculator, EnhancedPickScorer, get_risk_tolerance_for_context
from .distributions import DistributionFactory


class PickRecommendationEngine:
    """Generate optimal pick recommendations for each round"""
    
    def __init__(self, league_config: LeagueConfig, player_universe: List[Player], use_variance_adjustment: bool = True):
        self.config = league_config
        self.players = player_universe
        self.use_variance_adjustment = use_variance_adjustment
        
        # Initialize calculators
        self.value_calc = ValueCalculator(player_universe)
        self.scarcity_analyzer = ScarcityAnalyzer(player_universe)
        
        # Initialize variance-adjusted components if enabled
        if use_variance_adjustment:
            try:
                self.distribution_factory = DistributionFactory()
                self.variance_calc = VarianceAdjustedValueCalculator(player_universe, self.distribution_factory)
                self.enhanced_scorer = EnhancedPickScorer(self.variance_calc)
            except Exception as e:
                print(f"⚠️  Variance adjustment disabled due to error: {e}")
                self.use_variance_adjustment = False
                self.variance_calc = None
                self.enhanced_scorer = None
        else:
            self.variance_calc = None
            self.enhanced_scorer = None
        
    def generate_round_recommendations(
        self, 
        round_number: int,
        current_roster: Roster,
        drafted_players: List[Player] = None
    ) -> List[PickRecommendation]:
        """Generate top 10 recommendations for a specific round"""
        
        # Calculate available players
        drafted_names = set(p.name.lower() for p in (drafted_players or []))
        available = [p for p in self.players if p.name.lower() not in drafted_names]
        
        # Get realistic candidates for this round (based on ADP)
        pick_number = self._calculate_pick_number(round_number)
        candidates = self._filter_realistic_candidates(available, pick_number)
        
        recommendations = []
        
        # Determine risk tolerance for this round/context
        risk_tolerance = get_risk_tolerance_for_context(
            round_number, 
            {'strength_percentile': self._estimate_roster_strength(current_roster)}
        )
        
        for player in candidates:
            # Calculate basic metrics
            vorp = self.value_calc.calculate_vorp(player)
            vona = self.value_calc.calculate_vona(player, available)
            scarcity = self.value_calc.calculate_positional_scarcity(player.position, available)
            tier_info = self.scarcity_analyzer.get_tier_info(player)
            roster_fit = self._calculate_roster_fit(player, current_roster)
            
            # Calculate enhanced evaluation factors
            sos_advantage = self._calculate_sos_advantage(player)
            starter_status_score = self._calculate_starter_status_score(player)
            injury_risk_score = self._calculate_injury_risk_score(player)
            
            # Calculate opportunity cost
            opportunity_cost = self._calculate_opportunity_cost(player, available, current_roster)
            
            # Generate enhanced factors description
            enhanced_factors = self._generate_enhanced_factors_description(
                player, sos_advantage, starter_status_score, injury_risk_score
            )
            
            # Calculate overall score (with variance adjustment if available)
            if self.use_variance_adjustment and self.enhanced_scorer:
                # Use variance-adjusted scoring
                base_metrics = {
                    'scarcity': scarcity,
                    'roster_fit': roster_fit,
                    'sos_advantage': sos_advantage,
                    'starter_status': starter_status_score,
                    'injury_risk': injury_risk_score
                }
                
                draft_context = {
                    'round_number': round_number,
                    'roster_strength': self._estimate_roster_strength(current_roster)
                }
                
                enhanced_scores = self.enhanced_scorer.calculate_enhanced_score(
                    player, base_metrics, risk_tolerance, draft_context
                )
                
                overall_score = enhanced_scores['enhanced_score']
                
                # Add variance metrics to enhanced factors
                enhanced_factors.update({
                    'upside': f"📈 Upside: {enhanced_scores['upside_premium']:.1f} pts",
                    'floor_risk': f"📉 Floor Risk: {enhanced_scores['floor_penalty']:.1f} pts",
                    'ceiling': f"🚀 Ceiling: {enhanced_scores['ceiling_value']:.1f} pts",
                    'variance_adj_vorp': f"⚖️  Variance VORP: {enhanced_scores['variance_adjusted_vorp']:.1f}"
                })
                
            else:
                # Use traditional scoring
                overall_score = self._calculate_overall_score(
                    vorp, scarcity, roster_fit, sos_advantage, 
                    starter_status_score, injury_risk_score
                )
            
            # Generate reasoning and risk assessment
            primary_reason = self._generate_primary_reason(
                player, vorp, scarcity, tier_info, roster_fit, enhanced_factors
            )
            
            risk_factors = self._identify_risk_factors(player, enhanced_factors)
            confidence = self._calculate_confidence(player, enhanced_factors)
            
            recommendation = PickRecommendation(
                player=player,
                overall_score=overall_score,
                vorp=vorp,
                vona=vona,
                scarcity_score=scarcity,
                tier_info=tier_info,
                roster_fit=roster_fit,
                opportunity_cost=opportunity_cost,
                sos_advantage=sos_advantage,
                starter_status_score=starter_status_score,
                injury_risk_score=injury_risk_score,
                primary_reason=primary_reason,
                risk_factors=risk_factors,
                confidence=confidence,
                enhanced_factors=enhanced_factors
            )
            
            recommendations.append(recommendation)
        
        # Sort by overall score and return top 10
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        return recommendations[:10]
    
    def _calculate_pick_number(self, round_number: int) -> int:
        """Calculate your pick number in this round"""
        if round_number % 2 == 1:  # Odd rounds
            return (round_number - 1) * self.config.num_teams + self.config.draft_position
        else:  # Even rounds (snake)
            return round_number * self.config.num_teams - (self.config.draft_position - 1)
    
    def _filter_realistic_candidates(self, available: List[Player], pick_number: int) -> List[Player]:
        """Filter to players who might realistically be available and avoid overdrafts"""
        # Dynamic ADP buffer based on round - tighter in early rounds
        round_number = ((pick_number - 1) // 12) + 1
        
        if round_number <= 3:
            adp_buffer = 8  # Tighter in early rounds (avoid major overdrafts)
        elif round_number <= 6:
            adp_buffer = 12  # Moderate buffer in middle rounds
        elif round_number <= 10:
            adp_buffer = 18  # More flexible in later rounds
        else:
            adp_buffer = 30  # Very flexible in deep rounds
        
        realistic = []
        undrafted_threshold = 15  # Don't recommend players going undrafted
        
        for player in available:
            if player.adp >= 999:  # No ADP data
                # Only include high projections in early rounds, be more lenient later
                min_projection = max(50, 150 - (round_number * 8))
                if player.projection >= min_projection:
                    realistic.append(player)
            elif player.adp <= (16 * 12) + undrafted_threshold:  # Within draftable range
                adp_diff = player.adp - pick_number
                
                # Allow some early picks (up to 6 picks early) but avoid major overdrafts
                if adp_diff >= -6 and adp_diff <= adp_buffer:
                    realistic.append(player)
        
        # Sort by a combination of projection and ADP appropriateness
        def sort_key(player):
            adp_penalty = 0
            if player.adp < 999:
                adp_diff = abs(player.adp - pick_number)
                # Penalize picks that are too early or too late
                if player.adp < pick_number - 6:  # Overdraft penalty
                    adp_penalty = (pick_number - player.adp) * 2
                elif adp_diff > adp_buffer:  # Too late penalty
                    adp_penalty = adp_diff
            
            return player.projection - adp_penalty
        
        realistic.sort(key=sort_key, reverse=True)
        return realistic[:20]  # Top 20 realistic, well-timed candidates
    
    def _calculate_roster_fit(self, player: Player, roster: Roster) -> float:
        """How well does this player fit current roster needs"""
        needs = roster.get_needs()
        position_need = needs.get(player.position, 0)
        
        # Scale based on roster construction
        current_count = len([p for p in roster.players if p.position == player.position])
        
        if player.position == 'QB':
            # Don't need more than 2 QBs usually
            if current_count >= 2:
                return 0.1
            elif current_count == 1:
                return 0.3
            else:
                return 1.0
                
        elif player.position in ['RB', 'WR']:
            # High value positions, always useful
            if current_count == 0:
                return 1.0
            elif current_count == 1:
                return 0.9
            elif current_count == 2:
                return 0.7
            elif current_count == 3:
                return 0.5
            else:
                return 0.3
                
        elif player.position == 'TE':
            if current_count >= 2:
                return 0.2
            elif current_count == 1:
                return 0.4
            else:
                return 0.8
                
        else:  # K, DEF
            if current_count >= 1:
                return 0.1
            else:
                return 0.6  # Lower priority than skill positions
    
    def _calculate_sos_advantage(self, player: Player) -> float:
        """Calculate Strength of Schedule advantage (higher = easier SoS)"""
        if not player.sos_rank:
            return 0.5  # Neutral if no data
        
        # Convert rank to advantage score (1=easiest becomes highest score)
        # Rank 1-32, convert to 0.9-0.1 scale
        max_rank = 32
        sos_advantage = 1.0 - ((player.sos_rank - 1) / (max_rank - 1))
        
        # Ensure bounds
        return max(0.1, min(0.9, sos_advantage))
    
    def _calculate_starter_status_score(self, player: Player) -> float:
        """Calculate starter status confidence score"""
        if player.is_projected_starter is None:
            return 0.6  # Neutral/unknown
        
        if not player.is_projected_starter:
            return 0.2  # Backup/uncertain role
        
        # Use confidence if available, otherwise high score for confirmed starters
        if player.starter_confidence is not None:
            return player.starter_confidence
        
        return 0.8  # Default high confidence for projected starters
    
    def _calculate_injury_risk_score(self, player: Player) -> float:
        """Calculate injury risk score (higher = lower risk)"""
        if not player.injury_risk_level:
            return 0.7  # Neutral if no data
        
        risk_mapping = {
            'LOW': 0.9,
            'MEDIUM': 0.6,
            'HIGH': 0.3
        }
        
        return risk_mapping.get(player.injury_risk_level.upper(), 0.7)
    
    def _calculate_opportunity_cost(self, player: Player, available: List[Player], roster: Roster) -> float:
        """Calculate opportunity cost of taking this player"""
        # Simple opportunity cost: what's the best alternative position?
        needs = roster.get_needs()
        
        # Find best available at other positions with high need
        alternatives = []
        for pos, need in needs.items():
            if pos != player.position and need > 0.3:
                pos_players = [p for p in available if p.position == pos]
                if pos_players:
                    best_alternative = max(pos_players, key=lambda x: x.projection)
                    alternatives.append(best_alternative.projection * need)
        
        if alternatives:
            max_alternative = max(alternatives)
            player_value = player.projection * needs.get(player.position, 0.5)
            return max(0, max_alternative - player_value)
        
        return 0
    
    def _calculate_overall_score(
        self, 
        vorp: float,
        scarcity: float, 
        roster_fit: float,
        sos_advantage: float,
        starter_status_score: float,
        injury_risk_score: float
    ) -> float:
        """Calculate weighted overall score including enhanced factors"""
        
        # Normalize VORP (scale to 0-1)
        vorp_normalized = min(1.0, vorp / 100.0)
        
        # Base scoring weights (60% of score)
        base_score = (
            vorp_normalized * 0.30 +      # 30% - Player value
            scarcity * 0.15 +             # 15% - Positional scarcity  
            roster_fit * 0.15             # 15% - Roster fit
        )
        
        # Enhanced factor weights (40% of score)
        enhanced_score = (
            sos_advantage * 0.15 +        # 15% - Schedule matters significantly
            starter_status_score * 0.20 +  # 20% - Starting role is crucial
            injury_risk_score * 0.05       # 5% - Injury risk is important but not dominant
        )
        
        return base_score + enhanced_score
    
    def _generate_enhanced_factors_description(
        self,
        player: Player,
        sos_advantage: float,
        starter_status_score: float, 
        injury_risk_score: float
    ) -> Dict[str, str]:
        """Generate human-readable descriptions of enhanced factors"""
        
        factors = {}
        
        # Strength of Schedule
        if player.sos_rank:
            if player.sos_rank <= 10:
                factors['sos'] = f"🟢 Favorable SoS (Rank {player.sos_rank})"
            elif player.sos_rank <= 22:
                factors['sos'] = f"🟡 Average SoS (Rank {player.sos_rank})" 
            else:
                factors['sos'] = f"🔴 Tough SoS (Rank {player.sos_rank})"
        else:
            factors['sos'] = "❓ SoS Unknown"
        
        # Starter Status
        if player.is_projected_starter:
            confidence_pct = int((player.starter_confidence or 0.8) * 100)
            factors['starter'] = f"✅ Projected Starter ({confidence_pct}% confidence)"
        elif player.is_projected_starter is False:
            factors['starter'] = "⚠️  Backup/Committee Role"
        else:
            factors['starter'] = "❓ Role Uncertain"
        
        # Injury Risk
        if player.injury_risk_level:
            risk_emojis = {'LOW': '💪', 'MEDIUM': '⚠️ ', 'HIGH': '🚑'}
            emoji = risk_emojis.get(player.injury_risk_level.upper(), '❓')
            factors['injury'] = f"{emoji} {player.injury_risk_level.title()} Injury Risk"
            
            if player.injury_details:
                factors['injury'] += f" ({player.injury_details[:50]}...)"
        else:
            factors['injury'] = "❓ Injury Status Unknown"
        
        return factors
    
    def _generate_primary_reason(
        self, 
        player: Player, 
        vorp: float, 
        scarcity: float, 
        tier_info: Dict[str, Any],
        roster_fit: float,
        enhanced_factors: Dict[str, str]
    ) -> str:
        """Generate primary reasoning for this recommendation"""
        
        reasons = []
        
        # High VORP
        if vorp > 50:
            reasons.append(f"Elite value (VORP: {vorp:.1f})")
        elif vorp > 25:
            reasons.append(f"Strong value (VORP: {vorp:.1f})")
        
        # High scarcity
        if scarcity > 0.7:
            reasons.append("High positional scarcity")
        elif scarcity > 0.5:
            reasons.append("Growing positional scarcity")
        
        # Tier information
        if tier_info.get('tier_number', 99) <= 2:
            tier_num = tier_info['tier_number']
            reasons.append(f"Top tier player (Tier {tier_num})")
        
        # Roster fit
        if roster_fit > 0.8:
            reasons.append("Fills major roster need")
        elif roster_fit > 0.6:
            reasons.append("Good roster fit")
        
        # Enhanced factors
        if player.is_projected_starter and player.starter_confidence and player.starter_confidence > 0.85:
            reasons.append("Confirmed starter")
        
        if player.sos_rank and player.sos_rank <= 8:
            reasons.append("Favorable schedule")
        
        # Default reason
        if not reasons:
            reasons.append(f"Strong projection ({player.projection:.1f} points)")
        
        return ", ".join(reasons[:2])  # Max 2 reasons for brevity
    
    def _identify_risk_factors(self, player: Player, enhanced_factors: Dict[str, str]) -> List[str]:
        """Identify risk factors for this player"""
        risks = []
        
        # Injury risks
        if player.injury_risk_level == 'HIGH':
            risks.append("High injury risk")
        elif player.injury_risk_level == 'MEDIUM':
            risks.append("Some injury concerns")
        
        # Role uncertainty
        if player.is_projected_starter is False:
            risks.append("Backup/uncertain role")
        elif player.starter_confidence and player.starter_confidence < 0.6:
            risks.append("Role uncertainty")
        
        # Schedule difficulty
        if player.sos_rank and player.sos_rank > 28:
            risks.append("Difficult schedule")
        
        # ADP vs projection mismatch
        if player.adp < 999 and player.projection < 150:
            risks.append("Limited upside")
        
        return risks
    
    def _calculate_confidence(self, player: Player, enhanced_factors: Dict[str, str]) -> float:
        """Calculate confidence in this recommendation"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence for confirmed starters
        if player.is_projected_starter and player.starter_confidence:
            confidence += player.starter_confidence * 0.2
        
        # Reduce confidence for injury risks
        if player.injury_risk_level == 'HIGH':
            confidence -= 0.2
        elif player.injury_risk_level == 'MEDIUM':
            confidence -= 0.1
        
        # Boost confidence for favorable schedule
        if player.sos_rank and player.sos_rank <= 10:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _estimate_roster_strength(self, roster: Roster) -> float:
        """Estimate roster strength as percentile (0-1)"""
        if not roster.players:
            return 0.5  # Neutral for empty roster
        
        # Calculate total value of current roster
        total_value = sum(self.value_calc.calculate_vorp(player) for player in roster.players)
        
        # Estimate based on number of picks and average value
        num_picks = len(roster.players)
        if num_picks == 0:
            return 0.5
        
        avg_value_per_pick = total_value / num_picks
        
        # Rough percentile mapping (can be refined)
        if avg_value_per_pick >= 60:
            return 0.9  # Elite roster
        elif avg_value_per_pick >= 40:
            return 0.7  # Strong roster
        elif avg_value_per_pick >= 25:
            return 0.5  # Average roster
        elif avg_value_per_pick >= 15:
            return 0.3  # Weak roster
        else:
            return 0.1  # Very weak roster
