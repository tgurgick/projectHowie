"""
Unified Configuration System
Combines league settings and keeper configuration in one organized structure
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

from .models import LeagueConfig
from .keeper_system import Keeper, KeeperConfiguration


@dataclass
class TeamInfo:
    """Information about a specific team in the league"""
    team_name: str
    draft_position: int  # 1-based position
    owner_name: Optional[str] = None
    keepers: List[Keeper] = None
    
    def __post_init__(self):
        if self.keepers is None:
            self.keepers = []


@dataclass
class UnifiedDraftConfig:
    """Complete draft configuration with league settings and keeper data"""
    
    # League Settings
    league_name: str = "My Fantasy League"
    season: int = 2025
    num_teams: int = 12
    roster_size: int = 16
    scoring_type: str = "half_ppr"  # ppr, half_ppr, standard
    
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
    
    # Your Team Info
    your_team_name: str = "Your Team"
    your_draft_position: int = 8
    
    # Keeper Settings
    keepers_enabled: bool = True
    keeper_rules: str = "round_based"  # "first_round", "round_based", "auction_value"
    max_keepers_per_team: int = 2
    
    # All Teams and Keepers
    teams: List[TeamInfo] = None
    
    # Metadata
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        if self.teams is None:
            self.teams = []
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def add_team(self, team_name: str, draft_position: int, owner_name: str = None) -> TeamInfo:
        """Add a team to the league"""
        team = TeamInfo(
            team_name=team_name,
            draft_position=draft_position,
            owner_name=owner_name,
            keepers=[]
        )
        self.teams.append(team)
        return team
    
    def get_team_by_name(self, team_name: str) -> Optional[TeamInfo]:
        """Get team by name"""
        return next((team for team in self.teams if team.team_name == team_name), None)
    
    def get_team_by_position(self, position: int) -> Optional[TeamInfo]:
        """Get team by draft position"""
        return next((team for team in self.teams if team.draft_position == position), None)
    
    def get_your_team(self) -> Optional[TeamInfo]:
        """Get your team info"""
        return self.get_team_by_position(self.your_draft_position)
    
    def add_keeper(self, team_name: str, player_name: str, keeper_round: int, original_round: int = None) -> bool:
        """Add a keeper to a team"""
        team = self.get_team_by_name(team_name)
        if not team:
            return False
        
        keeper = Keeper(
            team_name=team_name,
            draft_position=team.draft_position,
            player_name=player_name,
            keeper_round=keeper_round,
            original_round=original_round
        )
        team.keepers.append(keeper)
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        return True
    
    def get_all_keepers(self) -> List[Keeper]:
        """Get all keepers from all teams"""
        all_keepers = []
        for team in self.teams:
            all_keepers.extend(team.keepers)
        return all_keepers
    
    def get_keeper_summary(self) -> Dict[str, Any]:
        """Get summary of keeper situation"""
        all_keepers = self.get_all_keepers()
        
        return {
            'total_keepers': len(all_keepers),
            'teams_with_keepers': len([team for team in self.teams if team.keepers]),
            'keepers_by_round': self._group_keepers_by_round(all_keepers),
            'your_keepers': [k for k in all_keepers if k.draft_position == self.your_draft_position]
        }
    
    def _group_keepers_by_round(self, keepers: List[Keeper]) -> Dict[int, List[str]]:
        """Group keepers by round"""
        by_round = {}
        for keeper in keepers:
            round_num = keeper.keeper_round
            if round_num not in by_round:
                by_round[round_num] = []
            by_round[round_num].append(f"{keeper.player_name} ({keeper.team_name})")
        return by_round
    
    def to_league_config(self) -> LeagueConfig:
        """Convert to legacy LeagueConfig format"""
        return LeagueConfig(
            num_teams=self.num_teams,
            roster_size=self.roster_size,
            scoring_type=self.scoring_type,
            qb_slots=self.qb_slots,
            rb_slots=self.rb_slots,
            wr_slots=self.wr_slots,
            te_slots=self.te_slots,
            flex_slots=self.flex_slots,
            superflex_slots=self.superflex_slots,
            k_slots=self.k_slots,
            def_slots=self.def_slots,
            bench_slots=self.bench_slots,
            draft_type=self.draft_type,
            draft_position=self.your_draft_position,
            keepers_enabled=self.keepers_enabled,
            keeper_rules=self.keeper_rules
        )
    
    def to_keeper_configuration(self) -> KeeperConfiguration:
        """Convert to legacy KeeperConfiguration format"""
        all_keepers = self.get_all_keepers()
        return KeeperConfiguration(
            keepers=all_keepers,
            keeper_rules=self.keeper_rules
        )
    
    def save_to_file(self, filename: str = "data/draft_config.json") -> None:
        """Save unified configuration to file"""
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Update timestamp
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to serializable format
        config_data = {
            # League Settings
            'league_name': self.league_name,
            'season': self.season,
            'num_teams': self.num_teams,
            'roster_size': self.roster_size,
            'scoring_type': self.scoring_type,
            
            # Roster Requirements
            'qb_slots': self.qb_slots,
            'rb_slots': self.rb_slots,
            'wr_slots': self.wr_slots,
            'te_slots': self.te_slots,
            'flex_slots': self.flex_slots,
            'superflex_slots': self.superflex_slots,
            'k_slots': self.k_slots,
            'def_slots': self.def_slots,
            'bench_slots': self.bench_slots,
            
            # Draft Settings
            'draft_type': self.draft_type,
            'your_team_name': self.your_team_name,
            'your_draft_position': self.your_draft_position,
            
            # Keeper Settings
            'keepers_enabled': self.keepers_enabled,
            'keeper_rules': self.keeper_rules,
            'max_keepers_per_team': self.max_keepers_per_team,
            
            # Teams and Keepers
            'teams': [
                {
                    'team_name': team.team_name,
                    'draft_position': team.draft_position,
                    'owner_name': team.owner_name,
                    'keepers': [
                        {
                            'player_name': keeper.player_name,
                            'keeper_round': keeper.keeper_round,
                            'original_round': keeper.original_round
                        }
                        for keeper in team.keepers
                    ]
                }
                for team in self.teams
            ],
            
            # Metadata
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Saved unified draft configuration to {filename}")
    
    @classmethod
    def load_from_file(cls, filename: str = "data/draft_config.json") -> Optional['UnifiedDraftConfig']:
        """Load unified configuration from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Create config object
            config = cls(
                league_name=data.get('league_name', 'My Fantasy League'),
                season=data.get('season', 2025),
                num_teams=data.get('num_teams', 12),
                roster_size=data.get('roster_size', 16),
                scoring_type=data.get('scoring_type', 'half_ppr'),
                qb_slots=data.get('qb_slots', 1),
                rb_slots=data.get('rb_slots', 2),
                wr_slots=data.get('wr_slots', 2),
                te_slots=data.get('te_slots', 1),
                flex_slots=data.get('flex_slots', 1),
                superflex_slots=data.get('superflex_slots', 0),
                k_slots=data.get('k_slots', 1),
                def_slots=data.get('def_slots', 1),
                bench_slots=data.get('bench_slots', 6),
                draft_type=data.get('draft_type', 'snake'),
                your_team_name=data.get('your_team_name', 'Your Team'),
                your_draft_position=data.get('your_draft_position', 8),
                keepers_enabled=data.get('keepers_enabled', True),
                keeper_rules=data.get('keeper_rules', 'round_based'),
                max_keepers_per_team=data.get('max_keepers_per_team', 2),
                created_at=data.get('created_at', ''),
                last_updated=data.get('last_updated', ''),
                teams=[]
            )
            
            # Load teams and keepers
            for team_data in data.get('teams', []):
                team = TeamInfo(
                    team_name=team_data['team_name'],
                    draft_position=team_data['draft_position'],
                    owner_name=team_data.get('owner_name'),
                    keepers=[]
                )
                
                # Load keepers for this team
                for keeper_data in team_data.get('keepers', []):
                    keeper = Keeper(
                        team_name=team.team_name,
                        draft_position=team.draft_position,
                        player_name=keeper_data['player_name'],
                        keeper_round=keeper_data['keeper_round'],
                        original_round=keeper_data.get('original_round')
                    )
                    team.keepers.append(keeper)
                
                config.teams.append(team)
            
            print(f"📂 Loaded unified draft configuration from {filename}")
            return config
            
        except FileNotFoundError:
            print(f"📂 No unified configuration file found at {filename}")
            return None
        except Exception as e:
            print(f"❌ Error loading unified configuration: {e}")
            return None
    
    @classmethod
    def migrate_from_separate_configs(
        cls, 
        league_config_file: str = "data/league_config.json",
        keeper_config_file: str = "data/keeper_config.json"
    ) -> Optional['UnifiedDraftConfig']:
        """Migrate from separate league and keeper config files"""
        try:
            # Load legacy configs
            league_config = LeagueConfig.load_from_file(league_config_file)
            if not league_config:
                print("❌ Could not load league config for migration")
                return None
            
            from .keeper_system import KeeperManager
            keeper_manager = KeeperManager()
            keeper_config = None
            
            try:
                keeper_config = keeper_manager.load_keeper_configuration(keeper_config_file)
            except:
                print("⚠️  No keeper config found, creating without keepers")
            
            # Create unified config
            unified = cls(
                num_teams=league_config.num_teams,
                roster_size=league_config.roster_size,
                scoring_type=league_config.scoring_type,
                qb_slots=league_config.qb_slots,
                rb_slots=league_config.rb_slots,
                wr_slots=league_config.wr_slots,
                te_slots=league_config.te_slots,
                flex_slots=league_config.flex_slots,
                superflex_slots=league_config.superflex_slots,
                k_slots=league_config.k_slots,
                def_slots=league_config.def_slots,
                bench_slots=league_config.bench_slots,
                draft_type=league_config.draft_type,
                your_draft_position=league_config.draft_position,
                keepers_enabled=league_config.keepers_enabled,
                keeper_rules=league_config.keeper_rules,
                teams=[]
            )
            
            # Add teams and keepers from keeper config
            if keeper_config:
                # Group keepers by team
                teams_data = {}
                for keeper in keeper_config.keepers:
                    if keeper.team_name not in teams_data:
                        teams_data[keeper.team_name] = {
                            'position': keeper.draft_position,
                            'keepers': []
                        }
                    teams_data[keeper.team_name]['keepers'].append(keeper)
                
                # Create team objects
                for team_name, team_info in teams_data.items():
                    team = TeamInfo(
                        team_name=team_name,
                        draft_position=team_info['position'],
                        keepers=team_info['keepers']
                    )
                    unified.teams.append(team)
                
                # Set your team name based on position
                your_team = unified.get_your_team()
                if your_team:
                    unified.your_team_name = your_team.team_name
            
            print(f"✅ Migrated configurations successfully")
            print(f"   League: {unified.num_teams}T {unified.scoring_type.upper()}, position #{unified.your_draft_position}")
            if keeper_config:
                print(f"   Keepers: {len(keeper_config.keepers)} total across {len(unified.teams)} teams")
            
            return unified
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            return None


class UnifiedConfigManager:
    """Manager for unified draft configuration"""
    
    def __init__(self):
        self.config = None
    
    def get_current_config(self) -> Optional[UnifiedDraftConfig]:
        """Get current unified config, loading if needed"""
        if self.config is None:
            self.config = UnifiedDraftConfig.load_from_file()
        return self.config
    
    def migrate_and_save(self) -> bool:
        """Migrate from separate configs and save unified version"""
        unified = UnifiedDraftConfig.migrate_from_separate_configs()
        if unified:
            unified.save_to_file()
            self.config = unified
            return True
        return False
    
    def get_league_config(self) -> Optional[LeagueConfig]:
        """Get legacy LeagueConfig for compatibility"""
        config = self.get_current_config()
        return config.to_league_config() if config else None
    
    def get_keeper_configuration(self) -> Optional[KeeperConfiguration]:
        """Get legacy KeeperConfiguration for compatibility"""
        config = self.get_current_config()
        return config.to_keeper_configuration() if config else None
