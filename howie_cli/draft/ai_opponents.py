"""
AI Opponent Drafting Personalities
Simulates realistic drafting behavior for Monte Carlo simulation
"""

import random
import numpy as np
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

from .models import Player, Roster, LeagueConfig


class DrafterType(Enum):
    """Different AI drafter personality types"""
    VALUE_DRAFTER = "value_drafter"
    NEED_BASED = "need_based"
    SCARCE_HUNTER = "scarce_hunter"
    TIER_BASED = "tier_based"
    ZERO_RB = "zero_rb"
    ROBUST_RB = "robust_rb"
    HERO_RB = "hero_rb"
    QB_EARLY = "qb_early"
    QB_LATE = "qb_late"
    TE_PREMIUM = "te_premium"
    BEST_AVAILABLE = "best_available"


@dataclass
class DraftingWeights:
    """Weights for different drafting factors"""
    value: float = 0.5
    need: float = 0.3
    scarcity: float = 0.2
    tier: float = 0.0
    position_multipliers: Dict[str, float] = None


class AIDrafterPersonality:
    """AI drafter with specific personality and behavior patterns"""
    
    def __init__(self, drafter_type: DrafterType, team_number: int):
        self.drafter_type = drafter_type
        self.team_number = team_number
        self.weights = self._get_personality_weights()
        self.adp_variance = self._get_adp_variance()
        self.pick_history = []
        
    def _get_personality_weights(self) -> DraftingWeights:
        """Get drafting weights based on personality type"""
        
        weights_map = {
            DrafterType.VALUE_DRAFTER: DraftingWeights(
                value=0.8, need=0.1, scarcity=0.1,
                position_multipliers={'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0}
            ),
            
            DrafterType.NEED_BASED: DraftingWeights(
                value=0.3, need=0.6, scarcity=0.1,
                position_multipliers={'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0}
            ),
            
            DrafterType.SCARCE_HUNTER: DraftingWeights(
                value=0.4, need=0.2, scarcity=0.4,
                position_multipliers={'QB': 1.2, 'RB': 1.0, 'WR': 1.0, 'TE': 1.3}
            ),
            
            DrafterType.TIER_BASED: DraftingWeights(
                value=0.5, need=0.2, scarcity=0.0, tier=0.3,
                position_multipliers={'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0}
            ),
            
            DrafterType.ZERO_RB: DraftingWeights(
                value=0.6, need=0.3, scarcity=0.1,
                position_multipliers={'QB': 1.2, 'RB': 0.3, 'WR': 1.5, 'TE': 1.2}
            ),
            
            DrafterType.ROBUST_RB: DraftingWeights(
                value=0.5, need=0.4, scarcity=0.1,
                position_multipliers={'QB': 0.8, 'RB': 1.6, 'WR': 0.7, 'TE': 0.8}
            ),
            
            DrafterType.HERO_RB: DraftingWeights(
                value=0.7, need=0.2, scarcity=0.1,
                position_multipliers={'QB': 1.0, 'RB': 1.8, 'WR': 1.3, 'TE': 1.0}
            ),
            
            DrafterType.QB_EARLY: DraftingWeights(
                value=0.4, need=0.4, scarcity=0.2,
                position_multipliers={'QB': 2.0, 'RB': 0.8, 'WR': 0.8, 'TE': 0.8}
            ),
            
            DrafterType.QB_LATE: DraftingWeights(
                value=0.6, need=0.3, scarcity=0.1,
                position_multipliers={'QB': 0.3, 'RB': 1.2, 'WR': 1.2, 'TE': 1.0}
            ),
            
            DrafterType.TE_PREMIUM: DraftingWeights(
                value=0.5, need=0.3, scarcity=0.2,
                position_multipliers={'QB': 1.0, 'RB': 0.9, 'WR': 0.9, 'TE': 1.8}
            ),
            
            DrafterType.BEST_AVAILABLE: DraftingWeights(
                value=0.9, need=0.1, scarcity=0.0,
                position_multipliers={'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0}
            )
        }
        
        return weights_map.get(self.drafter_type, weights_map[DrafterType.VALUE_DRAFTER])
    
    def _get_adp_variance(self) -> float:
        """Get ADP variance based on personality (how much they deviate from consensus)"""
        
        variance_map = {
            DrafterType.VALUE_DRAFTER: 0.8,    # Low variance, follows value
            DrafterType.NEED_BASED: 1.5,       # High variance, reaches for needs
            DrafterType.SCARCE_HUNTER: 1.2,    # Medium-high variance
            DrafterType.TIER_BASED: 1.0,       # Medium variance
            DrafterType.ZERO_RB: 1.3,          # High variance for strategy
            DrafterType.ROBUST_RB: 1.3,        # High variance for strategy
            DrafterType.HERO_RB: 1.4,          # Very high variance
            DrafterType.QB_EARLY: 2.0,         # Extreme variance for QB
            DrafterType.QB_LATE: 1.1,          # Low variance except QB
            DrafterType.TE_PREMIUM: 1.6,       # High variance for TE
            DrafterType.BEST_AVAILABLE: 0.7    # Very low variance
        }
        
        return variance_map.get(self.drafter_type, 1.0)
    
    def make_pick(
        self, 
        available_players: List[Player], 
        current_roster: Roster, 
        round_number: int
    ) -> Player:
        """Make a pick based on this AI's personality"""
        
        if not available_players:
            return None
        
        # Filter to realistic candidates based on ADP with personality variance
        candidates = self._filter_realistic_candidates(available_players, round_number)
        
        if not candidates:
            candidates = available_players[:20]  # Fallback to top 20
        
        # Score each candidate
        scored_candidates = []
        for player in candidates:
            score = self._calculate_player_score(player, current_roster, round_number)
            scored_candidates.append((player, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness - select from top candidates with weighted probability
        top_candidates = scored_candidates[:8]  # Consider top 8
        
        # Weighted selection (higher scores more likely)
        weights = [score for _, score in top_candidates]
        if sum(weights) > 0:
            selected_player = random.choices(
                [player for player, _ in top_candidates],
                weights=weights
            )[0]
        else:
            selected_player = top_candidates[0][0]
        
        self.pick_history.append(selected_player)
        return selected_player
    
    def _filter_realistic_candidates(self, available_players: List[Player], round_number: int) -> List[Player]:
        """Filter to players this AI might realistically draft"""
        
        # Calculate rough pick number
        pick_number = round_number * 6  # Rough estimate for middle of round
        
        realistic = []
        
        # When ADP data is missing (999), use position-based realistic drafting
        for player in available_players:
            if player.adp >= 999:  # No ADP data - use realistic position logic
                is_realistic = self._is_realistic_pick_by_position(player, round_number)
                if is_realistic:
                    realistic.append(player)
            else:
                # Check if within ADP range
                adp_buffer = 12 * self.adp_variance
                adp_diff = abs(player.adp - pick_number)
                if adp_diff <= adp_buffer:
                    realistic.append(player)
        
        # Ensure we have at least some candidates
        if len(realistic) < 5:
            # Add top available by projection as fallback
            high_value = sorted(available_players, key=lambda p: p.projection, reverse=True)[:15]
            for player in high_value:
                if player not in realistic:
                    realistic.append(player)
                if len(realistic) >= 15:
                    break
        
        return realistic
    
    def _is_realistic_pick_by_position(self, player: Player, round_number: int) -> bool:
        """Determine if a position pick is realistic for this round (when ADP missing)"""
        
        position = player.position.upper()
        
        # Realistic QB drafting patterns
        if position == 'QB':
            if round_number == 1:
                # Only QB-early personalities draft QB in Round 1
                return self.drafter_type.name in ['QB_EARLY'] and player.projection >= 320
            elif round_number == 2:
                # Some personalities take elite QBs in Round 2
                return (self.drafter_type.name in ['QB_EARLY', 'VALUE_DRAFTER'] and 
                        player.projection >= 310)
            elif round_number <= 4:
                # Most QBs go in rounds 3-4
                return player.projection >= 280
            else:
                # Late round QBs
                return True
        
        # RB/WR are more flexible but still position-dependent
        elif position in ['RB', 'WR']:
            if round_number <= 3:
                return player.projection >= 250  # Elite players only
            elif round_number <= 6:
                return player.projection >= 180  # Solid starters
            else:
                return True  # Any RB/WR later
        
        # TE typically goes later unless elite
        elif position == 'TE':
            if round_number <= 2:
                return player.projection >= 200  # Only elite TEs early
            elif round_number <= 5:
                return player.projection >= 150  # Solid TEs mid-rounds
            else:
                return True  # Any TE later
        
        # K/DEF go very late
        elif position in ['K', 'DEF', 'DST']:
            return round_number >= 10  # Never draft K/DEF early
        
        return True  # Default: allow the pick
    
    def _calculate_player_score(self, player: Player, roster: Roster, round_number: int) -> float:
        """Calculate score for a player based on AI personality"""
        
        # Base value score (normalized projection)
        value_score = min(1.0, player.projection / 300.0)
        
        # Need score
        need_score = self._calculate_need_score(player, roster)
        
        # Scarcity score (simplified)
        scarcity_score = self._calculate_scarcity_score(player, round_number)
        
        # Tier score (simplified - based on projection gaps)
        tier_score = self._calculate_tier_score(player)
        
        # Apply personality weights
        weights = self.weights
        base_score = (
            value_score * weights.value +
            need_score * weights.need +
            scarcity_score * weights.scarcity +
            tier_score * weights.tier
        )
        
        # Apply position multipliers
        if weights.position_multipliers:
            position_multiplier = weights.position_multipliers.get(player.position.upper(), 1.0)
            base_score *= position_multiplier
        
        # Round-specific adjustments
        base_score *= self._get_round_multiplier(player.position, round_number)
        
        # Add small random factor for realism
        random_factor = random.uniform(0.95, 1.05)
        
        return base_score * random_factor
    
    def _calculate_need_score(self, player: Player, roster: Roster) -> float:
        """Calculate how much this position is needed"""
        
        position_counts = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            position_counts[pos] = len([p for p in roster.players if p.position.upper() == pos])
        
        player_pos = player.position.upper()
        current_count = position_counts.get(player_pos, 0)
        
        # Need scoring based on position requirements
        if player_pos == 'QB':
            if current_count == 0:
                return 1.0
            elif current_count == 1:
                return 0.3
            else:
                return 0.1
        
        elif player_pos in ['RB', 'WR']:
            if current_count == 0:
                return 1.0
            elif current_count == 1:
                return 0.8
            elif current_count == 2:
                return 0.6
            elif current_count == 3:
                return 0.4
            else:
                return 0.2
        
        elif player_pos == 'TE':
            if current_count == 0:
                return 0.8
            elif current_count == 1:
                return 0.3
            else:
                return 0.1
        
        else:  # K, DEF
            if current_count == 0:
                return 0.4
            else:
                return 0.1
    
    def _calculate_scarcity_score(self, player: Player, round_number: int) -> float:
        """Calculate positional scarcity score"""
        
        # Simplified scarcity - positions get scarcer in later rounds
        scarcity_by_position = {
            'QB': max(0, (round_number - 3) / 10),  # QB scarcity starts round 4
            'RB': max(0, (round_number - 1) / 8),   # RB scarcity starts immediately
            'WR': max(0, (round_number - 2) / 8),   # WR scarcity starts round 3
            'TE': max(0, (round_number - 4) / 6),   # TE scarcity starts round 5
            'K': max(0, (round_number - 12) / 4),   # K scarcity very late
            'DEF': max(0, (round_number - 10) / 4)  # DEF scarcity late
        }
        
        return min(1.0, scarcity_by_position.get(player.position.upper(), 0))
    
    def _calculate_tier_score(self, player: Player) -> float:
        """Calculate tier-based score (simplified)"""
        
        # Simplified tier scoring based on projection ranges
        if player.projection >= 280:
            return 1.0  # Tier 1
        elif player.projection >= 240:
            return 0.8  # Tier 2
        elif player.projection >= 200:
            return 0.6  # Tier 3
        elif player.projection >= 160:
            return 0.4  # Tier 4
        else:
            return 0.2  # Lower tiers
    
    def _get_round_multiplier(self, position: str, round_number: int) -> float:
        """Get round-specific position multipliers"""
        
        # Adjust position preference by round
        multipliers = {
            'QB': {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.7},
            'RB': {1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.8},
            'WR': {1: 1.2, 2: 1.3, 3: 1.2, 4: 1.1, 5: 1.0, 6: 0.9},
            'TE': {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.0},
            'K': {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5, 6: 0.8},
            'DEF': {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.9}
        }
        
        position_upper = position.upper()
        if position_upper in multipliers and round_number in multipliers[position_upper]:
            return multipliers[position_upper][round_number]
        
        return 1.0  # Default multiplier
