"""
Value calculation engine for draft analysis
Calculates VORP, VONA, and positional scarcity
"""

from typing import List, Dict, Any
import numpy as np
from .models import Player


class ValueCalculator:
    """Calculate comprehensive player values for draft decisions"""
    
    def __init__(self, player_universe: List[Player]):
        self.players = player_universe
        self.replacement_levels = self._calculate_replacement_levels()
        
    def _calculate_replacement_levels(self) -> Dict[str, float]:
        """Calculate replacement level for each position"""
        replacement = {}
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            # Match positions case-insensitively and handle DST/DEF
            if position == 'DEF':
                pos_players = [p for p in self.players if p.position.upper() in ['DEF', 'DST']]
            else:
                pos_players = [p for p in self.players if p.position.upper() == position]
            pos_players.sort(key=lambda x: x.projection, reverse=True)
            
            # Replacement level varies by position
            if position == 'QB':
                replacement_index = 12  # 1 per team in 12-team league
            elif position in ['RB', 'WR']:
                replacement_index = 36  # ~3 per team (including flex)
            elif position == 'TE':
                replacement_index = 12  # 1 per team
            elif position in ['K', 'DEF']:
                replacement_index = 12  # 1 per team
            else:
                replacement_index = 24  # Default
                
            if len(pos_players) > replacement_index:
                replacement[position] = pos_players[replacement_index].projection
            else:
                replacement[position] = 0
                
        return replacement
    
    def calculate_vorp(self, player: Player) -> float:
        """Value Over Replacement Player"""
        # Handle position case-insensitivity and DST/DEF
        position_key = player.position.upper()
        if position_key in ['DST']:
            position_key = 'DEF'
        
        replacement = self.replacement_levels.get(position_key, 0)
        return max(0, player.projection - replacement)
    
    def calculate_vona(self, player: Player, available_players: List[Player]) -> float:
        """Value Over Next Available at position"""
        same_position = [p for p in available_players 
                        if p.position == player.position and p != player]
        
        if not same_position:
            return player.projection
            
        # Sort by projection and get next best
        same_position.sort(key=lambda x: x.projection, reverse=True)
        next_best = same_position[0] if same_position else None
        
        if next_best:
            return max(0, player.projection - next_best.projection)
        else:
            return player.projection
    
    def calculate_positional_scarcity(
        self, 
        position: str, 
        available_players: List[Player]
    ) -> float:
        """Calculate how scarce this position is becoming"""
        
        pos_players = [p for p in available_players if p.position == position]
        total_pos_players = len([p for p in self.players if p.position == position])
        
        if total_pos_players == 0:
            return 0
            
        # Scarcity = 1 - (remaining / total)
        scarcity = 1 - (len(pos_players) / total_pos_players)
        return min(1.0, max(0.0, scarcity))


class ScarcityAnalyzer:
    """Analyze positional scarcity and tier breaks"""
    
    def __init__(self, player_universe: List[Player]):
        self.players = player_universe
        self.tiers = self._identify_tiers()
    
    def _identify_tiers(self) -> Dict[str, List[List[Player]]]:
        """Identify natural tiers for each position"""
        tiers = {}
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            pos_players = [p for p in self.players if p.position == position]
            pos_players.sort(key=lambda x: x.projection, reverse=True)
            
            if not pos_players:
                tiers[position] = []
                continue
            
            # Simple tier breaks based on projection gaps
            position_tiers = []
            current_tier = []
            
            for i, player in enumerate(pos_players):
                if not current_tier:
                    current_tier.append(player)
                    continue
                
                # Check if significant drop (>10% or >15 points)
                last_projection = current_tier[-1].projection
                if last_projection > 0:
                    drop_percent = (last_projection - player.projection) / last_projection
                    drop_absolute = last_projection - player.projection
                    
                    if drop_percent > 0.1 or drop_absolute > 15:
                        # Start new tier
                        position_tiers.append(current_tier)
                        current_tier = [player]
                    else:
                        current_tier.append(player)
                else:
                    current_tier.append(player)
                    
                # Limit tier size (max 6 players per tier)
                if len(current_tier) >= 6:
                    position_tiers.append(current_tier)
                    current_tier = []
            
            if current_tier:
                position_tiers.append(current_tier)
                
            tiers[position] = position_tiers
            
        return tiers
    
    def get_tier_info(self, player: Player) -> Dict[str, Any]:
        """Get tier information for a player"""
        position_tiers = self.tiers.get(player.position, [])
        
        for tier_num, tier_players in enumerate(position_tiers):
            if player in tier_players:
                return {
                    'tier_number': tier_num + 1,
                    'tier_size': len(tier_players),
                    'players_left_in_tier': len(tier_players),
                    'next_tier_drop': self._calculate_tier_drop(tier_num, position_tiers)
                }
        
        return {
            'tier_number': 99, 
            'tier_size': 1, 
            'players_left_in_tier': 1, 
            'next_tier_drop': 0
        }
    
    def _calculate_tier_drop(self, current_tier: int, position_tiers: List[List[Player]]) -> float:
        """Calculate points drop to next tier"""
        if current_tier >= len(position_tiers) - 1:
            return 0
            
        current_avg = np.mean([p.projection for p in position_tiers[current_tier]])
        next_avg = np.mean([p.projection for p in position_tiers[current_tier + 1]])
        
        return max(0, current_avg - next_avg)
