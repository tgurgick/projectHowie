"""
Core data models for draft simulation system
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import copy


@dataclass
class LeagueConfig:
    """Complete league configuration"""
    
    # Basic Settings
    num_teams: int = 12
    roster_size: int = 16
    scoring_type: str = "ppr"  # ppr, half_ppr, standard
    
    # Roster Requirements
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 1  # RB/WR/TE
    superflex_slots: int = 0  # QB/RB/WR/TE
    k_slots: int = 1
    def_slots: int = 1
    bench_slots: int = 6
    
    # Draft Settings
    draft_type: str = "snake"  # snake, linear, auction
    draft_position: int = 6  # Your pick position (1-based)
    
    # Keeper Configuration
    keepers_enabled: bool = False
    keeper_slots: int = 0
    keeper_rules: str = "round_based"  # "first_round", "round_based", "auction_value"


@dataclass
class KeeperPlayer:
    """Individual keeper configuration"""
    player_name: str
    team_name: str  # Team that owns the keeper
    keeper_round: int  # Round where keeper is "drafted"
    original_round: int = None  # Where they went last year (for round_based)
    keeper_cost: int = None  # For auction leagues


@dataclass
class Player:
    """Enhanced player model with intelligence data"""
    name: str
    position: str
    team: str
    projection: float
    adp: float
    adp_position: int
    bye_week: int
    sos_rank: float = None
    sos_playoff: float = None
    
    # Enhanced Intelligence Fields
    is_projected_starter: bool = None
    starter_confidence: float = None  # 0.0 to 1.0
    injury_risk_level: str = None  # 'LOW', 'MEDIUM', 'HIGH'
    injury_details: str = None


class Roster:
    """Track drafted players and needs"""
    
    def __init__(self, config: LeagueConfig):
        self.config = config
        self.players: List[Player] = []
        self.starters = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 
            'FLEX': [], 'K': [], 'DEF': []
        }
    
    def add_player(self, player: Player) -> 'Roster':
        """Return new roster with player added"""
        new_roster = copy.deepcopy(self)
        new_roster.players.append(player)
        new_roster._update_starters(player)
        return new_roster
    
    def _update_starters(self, player: Player):
        """Update starting lineup with new player"""
        # Simple logic - just add to position list
        # More sophisticated logic could optimize starting lineup
        position = player.position
        if position in ['K', 'DEF']:
            position_key = position if position == 'DEF' else 'K'
        else:
            position_key = position
            
        if position_key in self.starters:
            self.starters[position_key].append(player)
    
    def get_needs(self) -> Dict[str, float]:
        """Calculate positional needs (0-1 scale)"""
        needs = {}
        
        # Count filled positions (handle both upper and lowercase)
        qb_count = len([p for p in self.players if p.position.upper() == 'QB'])
        rb_count = len([p for p in self.players if p.position.upper() == 'RB'])
        wr_count = len([p for p in self.players if p.position.upper() == 'WR'])
        te_count = len([p for p in self.players if p.position.upper() == 'TE'])
        k_count = len([p for p in self.players if p.position.upper() == 'K'])
        def_count = len([p for p in self.players if p.position.upper() == 'DEF' or p.position.upper() == 'DST'])
        
        # Calculate need scores (higher = more needed)
        needs['QB'] = max(0, (self.config.qb_slots - qb_count) / max(1, self.config.qb_slots))
        needs['RB'] = max(0, (self.config.rb_slots + 0.5 - rb_count) / (self.config.rb_slots + 0.5))
        needs['WR'] = max(0, (self.config.wr_slots + 0.5 - wr_count) / (self.config.wr_slots + 0.5))
        needs['TE'] = max(0, (self.config.te_slots - te_count) / max(1, self.config.te_slots))
        needs['K'] = max(0, (self.config.k_slots - k_count) / max(1, self.config.k_slots))
        needs['DEF'] = max(0, (self.config.def_slots - def_count) / max(1, self.config.def_slots))
        
        return needs
    
    def get_summary(self) -> str:
        """Get quick roster summary"""
        counts = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            # Check both uppercase and lowercase positions for compatibility
            if pos == 'DEF':
                # Handle both DEF and DST positions
                counts[pos] = len([p for p in self.players if p.position.upper() in ['DEF', 'DST']])
            else:
                counts[pos] = len([p for p in self.players if p.position.upper() == pos])
        
        return " | ".join([f"{pos}: {count}" for pos, count in counts.items()])


@dataclass
class PickRecommendation:
    """Complete pick recommendation with enhanced factors"""
    player: Player
    overall_score: float
    vorp: float
    vona: float
    scarcity_score: float
    tier_info: Dict[str, Any]
    roster_fit: float
    opportunity_cost: float
    
    # Enhanced Evaluation Factors
    sos_advantage: float  # Strength of Schedule benefit (higher = easier SoS)
    starter_status_score: float  # Projected starter confidence
    injury_risk_score: float  # Injury risk assessment (higher = lower risk)
    
    primary_reason: str
    risk_factors: List[str]
    confidence: float
    enhanced_factors: Dict[str, str]  # Human-readable factor descriptions


@dataclass
class DraftState:
    """State of draft at any point"""
    round: int
    roster: Roster
    available_players: List[Player]
    drafted_players: List[Player] = None
    
    def __post_init__(self):
        if self.drafted_players is None:
            self.drafted_players = []
