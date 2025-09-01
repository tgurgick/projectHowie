"""
Draft State Management
Tracks the state of a draft simulation including all picks and team rosters
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy

from .models import Player, Roster, LeagueConfig


@dataclass
class TeamState:
    """State of a single team in the draft"""
    team_number: int
    roster: Roster
    picks_made: List[Player] = field(default_factory=list)
    needs: Dict[str, float] = field(default_factory=dict)


class DraftState:
    """Manages the complete state of a draft simulation"""
    
    def __init__(self, league_config: LeagueConfig, player_universe: List[Player]):
        self.config = league_config
        self.all_players = {player.name.lower(): player for player in player_universe}
        self.available_players = set(player.name.lower() for player in player_universe)
        self.drafted_players: Set[str] = set()
        
        # Initialize team states
        self.teams: Dict[int, TeamState] = {}
        for team_num in range(1, league_config.num_teams + 1):
            self.teams[team_num] = TeamState(
                team_number=team_num,
                roster=Roster(league_config)
            )
        
        # Draft tracking
        self.current_round = 1
        self.current_pick_in_round = 1
        self.total_picks_made = 0
        self.pick_history: List[tuple] = []  # (pick_number, team_number, player)
    
    def get_available_players(self) -> List[Player]:
        """Get list of currently available players"""
        available = []
        for player_name_lower in self.available_players:
            if player_name_lower in self.all_players:
                available.append(self.all_players[player_name_lower])
        
        # Sort by projection (highest first)
        available.sort(key=lambda p: p.projection, reverse=True)
        return available
    
    def draft_player(self, player: Player) -> bool:
        """Draft a player and update state"""
        
        player_name_lower = player.name.lower()
        
        if player_name_lower not in self.available_players:
            return False  # Player already drafted
        
        # Remove from available players
        self.available_players.remove(player_name_lower)
        self.drafted_players.add(player_name_lower)
        
        # Calculate which team is picking
        current_team = self.get_current_team_number()
        
        # Add to team roster
        if current_team in self.teams:
            team_state = self.teams[current_team]
            team_state.roster = team_state.roster.add_player(player)
            team_state.picks_made.append(player)
            team_state.needs = team_state.roster.get_needs()
        
        # Record pick
        absolute_pick_number = self.total_picks_made + 1
        self.pick_history.append((absolute_pick_number, current_team, player))
        
        # Advance draft state
        self._advance_pick()
        
        return True
    
    def get_current_team_number(self) -> int:
        """Get the team number that should pick next"""
        
        if self.current_round % 2 == 1:  # Odd rounds (1, 3, 5...)
            return self.current_pick_in_round
        else:  # Even rounds (2, 4, 6...) - snake draft
            return self.config.num_teams - self.current_pick_in_round + 1
    
    def is_your_turn(self) -> bool:
        """Check if it's your turn to pick"""
        return self.get_current_team_number() == self.config.draft_position
    
    def get_team_roster(self, team_number: int) -> Roster:
        """Get current roster for a specific team"""
        if team_number in self.teams:
            return self.teams[team_number].roster
        return Roster(self.config)
    
    def get_your_roster(self) -> Roster:
        """Get your current roster"""
        return self.get_team_roster(self.config.draft_position)
    
    def get_picks_until_your_turn(self) -> int:
        """Calculate how many picks until your next turn"""
        
        if self.is_your_turn():
            return 0
        
        current_team = self.get_current_team_number()
        your_position = self.config.draft_position
        
        if self.current_round % 2 == 1:  # Odd round
            if current_team < your_position:
                return your_position - current_team
            else:
                # Need to go to next round
                picks_to_end_round = self.config.num_teams - current_team + 1
                picks_in_next_round = self.config.num_teams - your_position + 1
                return picks_to_end_round + picks_in_next_round
        else:  # Even round (snake)
            if current_team > your_position:
                return current_team - your_position
            else:
                # Need to go to next round
                picks_to_end_round = current_team
                picks_in_next_round = your_position - 1
                return picks_to_end_round + picks_in_next_round
    
    def simulate_picks_until_your_turn(self, ai_opponents: List) -> List[Player]:
        """Simulate what other teams will pick before your turn"""
        
        simulated_picks = []
        picks_to_simulate = self.get_picks_until_your_turn()
        
        # Create a copy of current state for simulation
        temp_state = deepcopy(self)
        
        for _ in range(picks_to_simulate):
            if temp_state.is_your_turn():
                break
            
            current_team = temp_state.get_current_team_number()
            available = temp_state.get_available_players()
            
            if not available:
                break
            
            # Get AI opponent for this team
            opponent_index = (current_team - 1) if current_team < self.config.draft_position else (current_team - 2)
            if opponent_index < len(ai_opponents) and opponent_index >= 0:
                ai_opponent = ai_opponents[opponent_index]
                team_roster = temp_state.get_team_roster(current_team)
                
                # AI makes pick
                selected_player = ai_opponent.make_pick(available, team_roster, temp_state.current_round)
                if selected_player:
                    temp_state.draft_player(selected_player)
                    simulated_picks.append(selected_player)
            else:
                # Fallback: pick highest ADP player
                adp_sorted = sorted(available, key=lambda p: p.adp if p.adp < 999 else 999)
                if adp_sorted:
                    temp_state.draft_player(adp_sorted[0])
                    simulated_picks.append(adp_sorted[0])
        
        return simulated_picks
    
    def _advance_pick(self):
        """Advance to the next pick"""
        
        self.total_picks_made += 1
        self.current_pick_in_round += 1
        
        # Check if round is complete
        if self.current_pick_in_round > self.config.num_teams:
            self.current_round += 1
            self.current_pick_in_round = 1
    
    def is_draft_complete(self, max_rounds: int = 16) -> bool:
        """Check if draft is complete"""
        return self.current_round > max_rounds
    
    def get_draft_summary(self) -> Dict:
        """Get summary of draft state"""
        
        summary = {
            'current_round': self.current_round,
            'current_pick_in_round': self.current_pick_in_round,
            'total_picks_made': self.total_picks_made,
            'players_remaining': len(self.available_players),
            'your_roster_summary': self.get_your_roster().get_summary(),
            'next_team_to_pick': self.get_current_team_number(),
            'is_your_turn': self.is_your_turn(),
            'picks_until_your_turn': self.get_picks_until_your_turn()
        }
        
        return summary
    
    def get_recent_picks(self, num_picks: int = 10) -> List[tuple]:
        """Get the most recent picks made"""
        return self.pick_history[-num_picks:] if self.pick_history else []
    
    def get_team_picks(self, team_number: int) -> List[Player]:
        """Get all picks made by a specific team"""
        if team_number in self.teams:
            return self.teams[team_number].picks_made.copy()
        return []
    
    def get_position_scarcity(self) -> Dict[str, int]:
        """Get count of remaining players by position"""
        
        position_counts = {}
        available = self.get_available_players()
        
        for player in available:
            pos = player.position.upper()
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        return position_counts
    
    def export_state(self) -> Dict:
        """Export complete draft state for analysis"""
        
        return {
            'config': {
                'num_teams': self.config.num_teams,
                'draft_position': self.config.draft_position,
                'scoring_type': self.config.scoring_type
            },
            'draft_progress': {
                'current_round': self.current_round,
                'total_picks': self.total_picks_made,
                'picks_remaining': len(self.available_players)
            },
            'team_rosters': {
                team_num: {
                    'players': [p.name for p in team_state.picks_made],
                    'roster_summary': team_state.roster.get_summary()
                }
                for team_num, team_state in self.teams.items()
            },
            'pick_history': [
                {
                    'pick_number': pick_num,
                    'team': team_num,
                    'player': player.name,
                    'position': player.position,
                    'projection': player.projection
                }
                for pick_num, team_num, player in self.pick_history
            ],
            'available_players': [
                {
                    'name': player.name,
                    'position': player.position,
                    'projection': player.projection,
                    'adp': player.adp
                }
                for player in self.get_available_players()[:50]  # Top 50 available
            ]
        }
