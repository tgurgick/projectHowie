"""
Keeper System for Draft Simulation

This module manages keeper configurations, player validation, and keeper
impacts on draft simulations.
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib


@dataclass
class Keeper:
    """Individual keeper configuration"""
    team_name: str  # Team that owns the keeper
    draft_position: int  # Team's draft position (1-based)
    player_name: str  # Player being kept
    keeper_round: int  # Round where keeper counts as drafted
    original_round: Optional[int] = None  # Where they went last year (for reference)
    
    def __post_init__(self):
        """Validate keeper data"""
        if self.draft_position < 1 or self.draft_position > 32:
            raise ValueError(f"Draft position must be 1-32, got {self.draft_position}")
        if self.keeper_round < 1 or self.keeper_round > 20:
            raise ValueError(f"Keeper round must be 1-20, got {self.keeper_round}")


@dataclass
class KeeperConfiguration:
    """Complete keeper configuration for a league"""
    keepers: List[Keeper]
    keeper_rules: str = "round_based"  # "first_round" or "round_based"
    
    def get_keepers_by_team(self) -> Dict[str, List[Keeper]]:
        """Group keepers by team name"""
        team_keepers = {}
        for keeper in self.keepers:
            if keeper.team_name not in team_keepers:
                team_keepers[keeper.team_name] = []
            team_keepers[keeper.team_name].append(keeper)
        return team_keepers
    
    def get_kept_players(self) -> List[str]:
        """Get list of all kept player names"""
        return [keeper.player_name for keeper in self.keepers]
    
    def get_draft_picks_used(self) -> Dict[int, str]:
        """Get mapping of overall pick numbers to kept players"""
        picks_used = {}
        
        for keeper in self.keepers:
            # Calculate overall pick number
            if keeper.keeper_round % 2 == 1:  # Odd rounds
                pick_in_round = keeper.draft_position
            else:  # Even rounds (snake draft)
                pick_in_round = 13 - keeper.draft_position  # Assuming 12 teams
            
            overall_pick = (keeper.keeper_round - 1) * 12 + pick_in_round
            picks_used[overall_pick] = keeper.player_name
        
        return picks_used


class KeeperValidator:
    """Validates keeper configurations against the database"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_database_path()
    
    def _get_database_path(self) -> str:
        """Get database path using ProjectHowie conventions"""
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        db_path = project_root / "data" / "fantasy_ppr.db"
        
        if db_path.exists():
            return str(db_path)
        
        fallback = Path("data/fantasy_ppr.db")
        if fallback.exists():
            return str(fallback)
        
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_all_player_names(self, season: int = 2025) -> List[str]:
        """Get all player names from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT player_name 
            FROM player_projections 
            WHERE season = ?
            ORDER BY player_name
        """, (season,))
        
        player_names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return player_names
    
    def find_player_matches(self, search_name: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Find player name matches with similarity scores"""
        all_players = self.get_all_player_names()
        
        # Exact match first
        if search_name in all_players:
            return [(search_name, 1.0)]
        
        # Case insensitive exact match
        for player in all_players:
            if player.lower() == search_name.lower():
                return [(player, 1.0)]
        
        # Fuzzy matching
        matches = difflib.get_close_matches(
            search_name, 
            all_players, 
            n=max_suggestions,
            cutoff=0.6
        )
        
        # Calculate similarity scores
        scored_matches = []
        for match in matches:
            similarity = difflib.SequenceMatcher(None, search_name.lower(), match.lower()).ratio()
            scored_matches.append((match, similarity))
        
        return sorted(scored_matches, key=lambda x: x[1], reverse=True)
    
    def validate_keeper(self, keeper: Keeper) -> Dict[str, any]:
        """Validate a single keeper and return validation results"""
        result = {
            'valid': True,
            'issues': [],
            'suggestions': [],
            'player_info': None
        }
        
        # Check if player exists
        matches = self.find_player_matches(keeper.player_name)
        
        if not matches:
            result['valid'] = False
            result['issues'].append(f"Player '{keeper.player_name}' not found in database")
            
            # Try partial matches with lower threshold
            partial_matches = difflib.get_close_matches(
                keeper.player_name, 
                self.get_all_player_names(), 
                n=5, 
                cutoff=0.3
            )
            if partial_matches:
                result['suggestions'] = partial_matches
        
        elif matches[0][1] < 1.0:  # Not exact match
            result['issues'].append(f"Inexact match for '{keeper.player_name}'")
            result['suggestions'] = [match[0] for match in matches]
        
        else:
            # Get player info for exact match
            player_name = matches[0][0]
            result['player_info'] = self._get_player_info(player_name)
        
        return result
    
    def _get_player_info(self, player_name: str) -> Optional[Dict[str, any]]:
        """Get detailed player information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT player_name, position, team_name, fantasy_points, bye_week
            FROM player_projections 
            WHERE player_name = ? AND season = 2025
        """, (player_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'name': row[0],
                'position': row[1], 
                'team': row[2],
                'projection': row[3],
                'bye_week': row[4]
            }
        
        return None
    
    def validate_keeper_configuration(self, config: KeeperConfiguration) -> Dict[str, any]:
        """Validate entire keeper configuration"""
        results = {
            'valid': True,
            'total_keepers': len(config.keepers),
            'valid_keepers': 0,
            'keeper_results': [],
            'conflicts': [],
            'summary': {}
        }
        
        # Validate each keeper
        for i, keeper in enumerate(config.keepers):
            keeper_result = self.validate_keeper(keeper)
            keeper_result['keeper_index'] = i
            keeper_result['keeper'] = keeper
            results['keeper_results'].append(keeper_result)
            
            if keeper_result['valid']:
                results['valid_keepers'] += 1
            else:
                results['valid'] = False
        
        # Check for conflicts (same player kept by multiple teams)
        player_counts = {}
        for keeper in config.keepers:
            player_counts[keeper.player_name] = player_counts.get(keeper.player_name, 0) + 1
        
        for player, count in player_counts.items():
            if count > 1:
                results['conflicts'].append(f"Player '{player}' kept by {count} teams")
                results['valid'] = False
        
        # Check for draft position conflicts
        position_rounds = {}
        for keeper in config.keepers:
            key = (keeper.draft_position, keeper.keeper_round)
            if key in position_rounds:
                results['conflicts'].append(
                    f"Draft conflict: Position {keeper.draft_position}, Round {keeper.keeper_round} used multiple times"
                )
                results['valid'] = False
            position_rounds[key] = keeper.player_name
        
        # Generate summary
        positions = {}
        teams = set()
        for keeper in config.keepers:
            if keeper.player_name in [kr['keeper'].player_name for kr in results['keeper_results'] if kr['valid']]:
                teams.add(keeper.team_name)
                player_info = next((kr['player_info'] for kr in results['keeper_results'] 
                                  if kr['keeper'].player_name == keeper.player_name and kr['player_info']), None)
                if player_info:
                    pos = player_info['position']
                    positions[pos] = positions.get(pos, 0) + 1
        
        results['summary'] = {
            'teams_with_keepers': len(teams),
            'positions': positions,
            'earliest_round': min([k.keeper_round for k in config.keepers]) if config.keepers else None,
            'latest_round': max([k.keeper_round for k in config.keepers]) if config.keepers else None
        }
        
        return results


class KeeperManager:
    """Manages keeper operations for draft simulation"""
    
    def __init__(self):
        self.validator = KeeperValidator()
    
    def create_keeper_configuration_interactive(self) -> KeeperConfiguration:
        """Interactive keeper configuration setup"""
        print("🏆 KEEPER CONFIGURATION SETUP")
        print("=" * 50)
        
        # Get keeper rules
        print("\nKeeper Rules:")
        print("1. First Round - All keepers count as 1st round picks")
        print("2. Round Based - Keepers count as the round they were drafted")
        
        while True:
            rule_choice = input("Select keeper rules (1 or 2): ").strip()
            if rule_choice == "1":
                keeper_rules = "first_round"
                break
            elif rule_choice == "2":
                keeper_rules = "round_based"
                break
            else:
                print("Please enter 1 or 2")
        
        keepers = []
        
        # Get number of teams with keepers
        while True:
            try:
                num_teams = int(input("\nHow many teams have keepers? "))
                if 0 <= num_teams <= 12:
                    break
                else:
                    print("Please enter a number between 0 and 12")
            except ValueError:
                print("Please enter a valid number")
        
        # Configure each team's keepers
        for team_num in range(num_teams):
            print(f"\n--- Team {team_num + 1} Keepers ---")
            
            # Get team info
            while True:
                team_name = input(f"Team {team_num + 1} name (or 'Team{team_num + 1}'): ").strip()
                if not team_name:
                    team_name = f"Team{team_num + 1}"
                break
            
            while True:
                try:
                    draft_position = int(input(f"Team {team_name} draft position (1-12): "))
                    if 1 <= draft_position <= 12:
                        break
                    else:
                        print("Please enter a position between 1 and 12")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get keepers for this team
            while True:
                try:
                    num_keepers = int(input(f"How many keepers does {team_name} have? "))
                    if 0 <= num_keepers <= 3:
                        break
                    else:
                        print("Please enter 0-3 keepers per team")
                except ValueError:
                    print("Please enter a valid number")
            
            for keeper_num in range(num_keepers):
                print(f"\n  Keeper {keeper_num + 1} for {team_name}:")
                
                # Get player name with validation
                while True:
                    player_name = input("    Player name: ").strip()
                    if not player_name:
                        continue
                    
                    # Validate player
                    matches = self.validator.find_player_matches(player_name)
                    
                    if not matches:
                        print(f"    ❌ Player '{player_name}' not found")
                        suggestions = self.validator.find_player_matches(player_name, max_suggestions=3)
                        if suggestions:
                            print("    Suggestions:")
                            for i, (suggestion, score) in enumerate(suggestions):
                                print(f"      {i+1}. {suggestion}")
                        continue
                    
                    elif matches[0][1] < 1.0:
                        print(f"    Did you mean:")
                        for i, (suggestion, score) in enumerate(matches[:3]):
                            print(f"      {i+1}. {suggestion} ({score:.1%} match)")
                        
                        choice = input("    Choose number or type new name: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(matches):
                            player_name = matches[int(choice) - 1][0]
                        else:
                            continue
                    else:
                        player_name = matches[0][0]
                    
                    # Show player info
                    player_info = self.validator._get_player_info(player_name)
                    if player_info:
                        print(f"    ✅ {player_info['name']} ({player_info['position']}) - {player_info['team']} - {player_info['projection']:.1f} pts")
                    break
                
                # Get keeper round
                if keeper_rules == "first_round":
                    keeper_round = 1
                    print(f"    Keeper round: 1 (first round rule)")
                else:
                    while True:
                        try:
                            keeper_round = int(input("    What round does this keeper cost? "))
                            if 1 <= keeper_round <= 16:
                                break
                            else:
                                print("    Please enter a round between 1 and 16")
                        except ValueError:
                            print("    Please enter a valid number")
                
                # Create keeper
                keeper = Keeper(
                    team_name=team_name,
                    draft_position=draft_position,
                    player_name=player_name,
                    keeper_round=keeper_round
                )
                
                keepers.append(keeper)
                print(f"    ✅ Added {player_name} as Round {keeper_round} keeper for {team_name}")
        
        return KeeperConfiguration(keepers=keepers, keeper_rules=keeper_rules)
    
    def save_keeper_configuration(self, config: KeeperConfiguration, filename: str = "keeper_config.json") -> None:
        """Save keeper configuration to file"""
        import json
        
        data = {
            'keeper_rules': config.keeper_rules,
            'keepers': [
                {
                    'team_name': k.team_name,
                    'draft_position': k.draft_position,
                    'player_name': k.player_name,
                    'keeper_round': k.keeper_round,
                    'original_round': k.original_round
                }
                for k in config.keepers
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved keeper configuration to {filename}")
    
    def load_keeper_configuration(self, filename: str = "keeper_config.json") -> KeeperConfiguration:
        """Load keeper configuration from file"""
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        keepers = [
            Keeper(
                team_name=k['team_name'],
                draft_position=k['draft_position'],
                player_name=k['player_name'],
                keeper_round=k['keeper_round'],
                original_round=k.get('original_round')
            )
            for k in data['keepers']
        ]
        
        config = KeeperConfiguration(
            keepers=keepers,
            keeper_rules=data['keeper_rules']
        )
        
        print(f"📂 Loaded keeper configuration from {filename}")
        return config


if __name__ == "__main__":
    # Test the keeper system
    print("🏆 Testing Keeper System")
    print("=" * 30)
    
    # Test player validation
    validator = KeeperValidator()
    
    test_names = ["Brian Thomas Jr", "brian thomas", "Thomas Jr", "Josh Allen"]
    
    for name in test_names:
        print(f"\n🔍 Testing: '{name}'")
        matches = validator.find_player_matches(name)
        for match, score in matches[:3]:
            print(f"   {match} ({score:.1%})")
    
    print("\n✅ Keeper system ready for configuration!")
