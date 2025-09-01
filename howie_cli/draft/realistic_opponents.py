"""
Realistic Opponent Draft Model
Based on ADP + Gaussian noise + roster needs bias (following MCTS guide principles)
"""

import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from .models import Player, Roster, LeagueConfig


@dataclass
class RealisticDrafter:
    """Realistic drafter using ADP + noise + roster needs + strategic bias"""
    
    team_number: int
    adp_noise_std: float = 8.0  # Standard deviation for ADP noise
    needs_bias_strength: float = 0.3  # How much to bias toward needs (0-1)
    strategy_type: str = "balanced"  # Strategic preference type
    strategy_strength: float = 0.4  # How much to apply strategy bias (0-1)
    
    def make_pick(
        self, 
        available_players: List[Player], 
        current_roster: Roster, 
        round_number: int
    ) -> Player:
        """Make pick using ADP + noise + roster needs bias"""
        
        if not available_players:
            return None
        
        # Calculate scores for all available players
        player_scores = []
        roster_needs = current_roster.get_needs()
        
        for player in available_players:
            # Base score from ADP (lower ADP = higher score)
            adp = player.adp if player.adp < 999 else self._estimate_realistic_adp(player)
            
            # Add Gaussian noise to ADP for realistic variance
            noisy_adp = adp + np.random.normal(0, self.adp_noise_std)
            
            # Convert to score (lower noisy ADP = higher score)
            adp_score = max(1, 300 - noisy_adp)
            
            # Small roster needs bias (subtle, not dominant)
            position = player.position.upper()
            needs_multiplier = 1.0 + (roster_needs.get(position, 0) * self.needs_bias_strength * 0.3)  # Reduced impact
            
            # Very small strategic bias (just slight preferences)
            strategy_multiplier = self._get_strategy_multiplier(player, round_number)
            strategy_multiplier = 1.0 + (strategy_multiplier - 1.0) * 0.3  # Dampened strategy impact
            
            # Final score: ADP dominates, small biases for tiebreaking
            final_score = adp_score * needs_multiplier * strategy_multiplier
            
            # Small random factor for final variance
            final_score += np.random.normal(0, 3.0)
            
            player_scores.append((player, final_score))
        
        # Sort by score and select best
        player_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add some variance - select from top 3-8 candidates
        top_candidates = player_scores[:min(8, len(player_scores))]
        
        # Weighted selection (higher scores more likely)
        weights = [max(0.1, score) for _, score in top_candidates]
        selected_player = np.random.choice(
            [player for player, _ in top_candidates],
            p=np.array(weights) / sum(weights)
        )
        
        return selected_player
    
    def _get_strategy_multiplier(self, player: Player, round_number: int) -> float:
        """Get strategic bias multiplier based on strategy type and round"""
        
        position = player.position.upper()
        base_multiplier = 1.0
        
        # Define strategic preferences (soft biases, not hard rules)
        if self.strategy_type == "zero_rb":
            # Zero RB: Avoid RBs early, prefer WRs/TEs
            if position == 'RB':
                if round_number <= 3:
                    base_multiplier = 0.6  # Discourage early RBs
                elif round_number <= 6:
                    base_multiplier = 1.3  # Prefer mid-round RBs
            elif position == 'WR':
                if round_number <= 4:
                    base_multiplier = 1.4  # Prefer early WRs
            elif position == 'TE':
                if round_number <= 5:
                    base_multiplier = 1.2  # Slight TE preference
        
        elif self.strategy_type == "robust_rb":
            # Robust RB: Load up on RBs early
            if position == 'RB':
                if round_number <= 4:
                    base_multiplier = 1.5  # Strong early RB preference
                elif round_number <= 7:
                    base_multiplier = 1.2  # Continue RB preference
            elif position == 'WR':
                if round_number <= 3:
                    base_multiplier = 0.8  # Slight WR discount early
        
        elif self.strategy_type == "hero_rb":
            # Hero RB: One elite RB then pivot
            rb_count = len([p for p in [] if p.position.upper() == 'RB'])  # Would need roster
            if position == 'RB':
                if round_number == 1 and rb_count == 0:
                    base_multiplier = 1.8  # Strong preference for first RB
                elif round_number <= 4 and rb_count >= 1:
                    base_multiplier = 0.4  # Avoid RBs after first one
                elif round_number >= 5:
                    base_multiplier = 1.3  # Come back to RBs later
            elif position == 'WR':
                if round_number >= 2 and round_number <= 5:
                    base_multiplier = 1.4  # Prefer WRs after first RB
        
        elif self.strategy_type == "qb_early":
            # QB Early: Target QB in first 3 rounds
            if position == 'QB':
                if round_number <= 3:
                    base_multiplier = 2.0  # Strong early QB preference
                else:
                    base_multiplier = 0.5  # Don't need QB later
            else:
                if round_number <= 2:
                    base_multiplier = 0.9  # Slight discount on other positions early
        
        elif self.strategy_type == "qb_late":
            # QB Late: Wait on QB until later rounds
            if position == 'QB':
                if round_number <= 5:
                    base_multiplier = 0.3  # Strongly avoid early QBs
                else:
                    base_multiplier = 1.5  # Prefer QBs later
            else:
                if round_number <= 5:
                    base_multiplier = 1.1  # Slight preference for other positions
        
        elif self.strategy_type == "te_premium":
            # TE Premium: Target elite TE early
            if position == 'TE':
                if round_number <= 4 and player.projection >= 180:
                    base_multiplier = 1.8  # Strong preference for elite TEs
                elif round_number <= 6:
                    base_multiplier = 1.3  # Moderate TE preference
            else:
                if round_number <= 3:
                    base_multiplier = 0.9  # Slight discount early for TE focus
        
        elif self.strategy_type == "best_available":
            # Best Available: Minimal strategic bias
            base_multiplier = 1.0  # No strategic preferences
        
        elif self.strategy_type == "wr_heavy":
            # WR Heavy: Load up on WRs
            if position == 'WR':
                if round_number <= 6:
                    base_multiplier = 1.3  # Prefer WRs throughout
            elif position == 'RB':
                if round_number <= 4:
                    base_multiplier = 0.8  # Slight RB discount
        
        # Apply strategy strength (how much to deviate from neutral)
        final_multiplier = 1.0 + (base_multiplier - 1.0) * self.strategy_strength
        
        return max(0.2, final_multiplier)  # Don't go below 0.2 to avoid complete elimination
    
    def _estimate_realistic_adp(self, player: Player) -> float:
        """Estimate realistic ADP when missing, based on projection + position"""
        
        # Get all players for context
        # This is a simplified version - in practice you'd want the full player universe
        
        # Base ADP from projection rank
        base_adp = 50  # Default middle value
        
        # Position adjustments (QBs typically go later than projections suggest)
        position_penalty = {
            'QB': 15,   # QBs go ~15 picks later than projection rank
            'RB': -5,   # RBs go ~5 picks earlier
            'WR': 0,    # WRs go about where projected
            'TE': 10,   # TEs go ~10 picks later
            'K': 100,   # Kickers go very late
            'DEF': 80,  # Defense goes late
            'DST': 80   # Defense goes late
        }
        
        position = player.position.upper()
        penalty = position_penalty.get(position, 0)
        
        # Projection-based estimate (higher projection = earlier pick)
        if player.projection >= 320:
            base_adp = 8
        elif player.projection >= 280:
            base_adp = 20
        elif player.projection >= 240:
            base_adp = 40
        elif player.projection >= 200:
            base_adp = 70
        elif player.projection >= 160:
            base_adp = 100
        else:
            base_adp = 150
        
        # Apply position penalty
        estimated_adp = base_adp + penalty
        
        # Add some randomness
        estimated_adp += np.random.normal(0, 5)
        
        return max(1, estimated_adp)


class RealisticOpponentManager:
    """Manages realistic opponents for Monte Carlo simulation"""
    
    def __init__(self, league_config: LeagueConfig):
        self.config = league_config
        self.opponents = self._create_realistic_opponents()
    
    def _create_realistic_opponents(self) -> List[RealisticDrafter]:
        """Create realistic opponents with varied noise levels and strategies"""
        
        opponents = []
        
        # Create opponents with different characteristics
        noise_levels = [6, 7, 8, 9, 10, 11, 12, 8, 9, 10, 11]  # Variety in ADP noise
        needs_bias_levels = [0.2, 0.25, 0.3, 0.35, 0.4, 0.3, 0.25, 0.3, 0.35, 0.4, 0.2]
        
        # Mix of strategic approaches (soft biases, not rigid rules)
        strategy_types = [
            "balanced", "zero_rb", "robust_rb", "hero_rb", "qb_early", 
            "qb_late", "te_premium", "best_available", "wr_heavy", "balanced", "zero_rb"
        ]
        strategy_strengths = [0.3, 0.5, 0.4, 0.6, 0.7, 0.5, 0.4, 0.2, 0.4, 0.3, 0.4]
        
        for i in range(self.config.num_teams - 1):  # -1 because user is one team
            noise_std = noise_levels[i % len(noise_levels)]
            needs_bias = needs_bias_levels[i % len(needs_bias_levels)]
            strategy_type = strategy_types[i % len(strategy_types)]
            strategy_strength = strategy_strengths[i % len(strategy_strengths)]
            
            opponents.append(RealisticDrafter(
                team_number=i + 1,
                adp_noise_std=noise_std,
                needs_bias_strength=needs_bias,
                strategy_type=strategy_type,
                strategy_strength=strategy_strength
            ))
        
        return opponents
    
    def get_opponent_for_team(self, team_number: int) -> Optional[RealisticDrafter]:
        """Get the opponent drafter for a specific team"""
        
        # Adjust for user's position
        if team_number == self.config.draft_position:
            return None  # This is the user
        
        # Map team number to opponent index
        opponent_index = (team_number - 1) if team_number < self.config.draft_position else (team_number - 2)
        
        if 0 <= opponent_index < len(self.opponents):
            return self.opponents[opponent_index]
        
        return None
    
    def get_all_opponents(self) -> List[RealisticDrafter]:
        """Get all opponent drafters"""
        return self.opponents.copy()


def generate_realistic_adp_for_players(players: List[Player]) -> Dict[str, float]:
    """Generate realistic ADP values based on overall ranking + position penalties"""
    
    # Sort ALL players by projection (overall ranking)
    all_players_sorted = sorted(players, key=lambda p: p.projection, reverse=True)
    
    realistic_adp = {}
    
    # Position penalties (how much later each position typically goes vs their projection rank)
    position_penalties = {
        'QB': 20,   # QBs go ~20 picks later than projection rank
        'RB': -5,   # RBs go ~5 picks earlier than projection rank  
        'WR': 0,    # WRs go about where projected
        'TE': 15,   # TEs go ~15 picks later than projection rank
        'K': 100,   # Kickers go much later
        'DEF': 80,  # Defense goes much later
        'DST': 80   # Defense goes much later
    }
    
    for overall_rank, player in enumerate(all_players_sorted, 1):
        # Base ADP from overall projection rank
        base_adp = overall_rank
        
        # Apply position penalty
        position = player.position.upper()
        penalty = position_penalties.get(position, 0)
        adjusted_adp = base_adp + penalty
        
        # Add some realistic randomness (±10 picks)
        final_adp = adjusted_adp + np.random.normal(0, 5)
        
        # Ensure reasonable bounds
        final_adp = max(1, min(300, final_adp))
        
        realistic_adp[player.name] = final_adp
    
    return realistic_adp
