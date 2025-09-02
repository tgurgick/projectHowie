"""
CLI Interface for Keeper Configuration

This module provides command-line interfaces for setting up and managing
keeper configurations for draft simulations.
"""

from typing import Optional
import json
from pathlib import Path

from .keeper_system import KeeperManager, KeeperConfiguration, Keeper, KeeperValidator
from .keeper_integration import KeeperAwareDraftSimulator
from .models import LeagueConfig
from .enhanced_monte_carlo import OutcomeAwareStrategy


class KeeperCLI:
    """Command-line interface for keeper configuration"""
    
    def __init__(self):
        self.manager = KeeperManager()
        self.validator = KeeperValidator()
        self.config_file = "data/keeper_config.json"
    
    def handle_keeper_command(self, args: list) -> None:
        """Handle keeper-related commands"""
        if not args:
            self.show_keeper_help()
            return
        
        command = args[0].lower()
        
        if command == "setup":
            self.setup_keepers()
        elif command == "validate":
            self.validate_keepers()
        elif command == "show":
            self.show_keepers()
        elif command == "test":
            self.test_keeper_simulation()
        elif command == "clear":
            self.clear_keepers()
        else:
            print(f"Unknown keeper command: {command}")
            self.show_keeper_help()
    
    def show_keeper_help(self):
        """Show keeper command help"""
        print("🏆 KEEPER SYSTEM COMMANDS")
        print("=" * 40)
        print("Available commands:")
        print("  setup     - Interactive keeper configuration")
        print("  validate  - Validate current keeper configuration")
        print("  show      - Display current keepers")
        print("  test      - Run test simulation with keepers")
        print("  clear     - Clear all keeper configurations")
        print()
        print("Examples:")
        print("  /draft keeper setup")
        print("  /draft keeper show")
        print("  /draft keeper test")
    
    def setup_keepers(self):
        """Interactive keeper setup"""
        print("🏆 KEEPER CONFIGURATION WIZARD")
        print("=" * 50)
        
        # Check if configuration already exists
        if Path(self.config_file).exists():
            print(f"⚠️  Existing keeper configuration found.")
            response = input("Overwrite existing configuration? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Setup cancelled.")
                return
        
        try:
            # Run interactive configuration
            config = self.manager.create_keeper_configuration_interactive()
            
            # Validate configuration
            print("\n🔍 Validating keeper configuration...")
            validation = self.validator.validate_keeper_configuration(config)
            
            if not validation['valid']:
                print("❌ Configuration has issues:")
                for issue in validation.get('conflicts', []):
                    print(f"   • {issue}")
                
                for keeper_result in validation['keeper_results']:
                    if not keeper_result['valid']:
                        keeper = keeper_result['keeper']
                        print(f"   • {keeper.player_name} (Team {keeper.team_name}): {keeper_result['issues']}")
                
                response = input("\nSave anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Setup cancelled.")
                    return
            
            # Save configuration
            self.save_keeper_config(config)
            
            # Show summary
            self.display_keeper_summary(config, validation)
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled.")
        except Exception as e:
            print(f"❌ Error during setup: {e}")
    
    def save_keeper_config(self, config: KeeperConfiguration):
        """Save keeper configuration to file"""
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Save configuration
        self.manager.save_keeper_configuration(config, self.config_file)
    
    def load_keeper_config(self) -> Optional[KeeperConfiguration]:
        """Load keeper configuration from file"""
        if not Path(self.config_file).exists():
            return None
        
        try:
            return self.manager.load_keeper_configuration(self.config_file)
        except Exception as e:
            print(f"❌ Error loading keeper configuration: {e}")
            return None
    
    def validate_keepers(self):
        """Validate current keeper configuration"""
        config = self.load_keeper_config()
        if not config:
            print("❌ No keeper configuration found. Run '/draft keeper setup' first.")
            return
        
        print("🔍 VALIDATING KEEPER CONFIGURATION")
        print("=" * 50)
        
        validation = self.validator.validate_keeper_configuration(config)
        
        if validation['valid']:
            print("✅ Keeper configuration is valid!")
        else:
            print("❌ Keeper configuration has issues:")
            
            for conflict in validation.get('conflicts', []):
                print(f"   🚫 {conflict}")
            
            for keeper_result in validation['keeper_results']:
                if not keeper_result['valid']:
                    keeper = keeper_result['keeper']
                    print(f"   ❌ {keeper.player_name} (Team {keeper.team_name}):")
                    for issue in keeper_result['issues']:
                        print(f"      • {issue}")
                    if keeper_result['suggestions']:
                        print(f"      Suggestions: {', '.join(keeper_result['suggestions'][:3])}")
        
        self.display_keeper_summary(config, validation)
    
    def show_keepers(self):
        """Display current keeper configuration"""
        config = self.load_keeper_config()
        if not config:
            print("❌ No keeper configuration found. Run '/draft keeper setup' first.")
            return
        
        print("🏆 CURRENT KEEPER CONFIGURATION")
        print("=" * 50)
        
        validation = self.validator.validate_keeper_configuration(config)
        self.display_keeper_summary(config, validation)
    
    def display_keeper_summary(self, config: KeeperConfiguration, validation: dict):
        """Display a formatted keeper summary"""
        print(f"\n📋 Configuration Summary:")
        print(f"   Keeper Rules: {config.keeper_rules}")
        print(f"   Total Keepers: {len(config.keepers)}")
        print(f"   Valid Keepers: {validation['valid_keepers']}/{validation['total_keepers']}")
        
        if validation['summary']['teams_with_keepers']:
            print(f"   Teams with Keepers: {validation['summary']['teams_with_keepers']}")
            print(f"   Position Breakdown: {validation['summary']['positions']}")
        
        print(f"\n🏆 Keeper Details:")
        
        # Group keepers by round
        keepers_by_round = {}
        for keeper in config.keepers:
            round_num = keeper.keeper_round
            if round_num not in keepers_by_round:
                keepers_by_round[round_num] = []
            keepers_by_round[round_num].append(keeper)
        
        for round_num in sorted(keepers_by_round.keys()):
            print(f"\n   Round {round_num}:")
            for keeper in keepers_by_round[round_num]:
                # Get player info
                player_info = None
                for result in validation['keeper_results']:
                    if result['keeper'].player_name == keeper.player_name:
                        player_info = result.get('player_info')
                        break
                
                if player_info:
                    print(f"     {keeper.team_name:12s} (Pick #{keeper.draft_position:2d}): "
                          f"{player_info['name']:20s} ({player_info['position']}) - "
                          f"{player_info['projection']:6.1f} pts")
                else:
                    print(f"     {keeper.team_name:12s} (Pick #{keeper.draft_position:2d}): "
                          f"{keeper.player_name:20s} (❌ Invalid)")
    
    def test_keeper_simulation(self):
        """Run a test simulation with current keeper configuration"""
        config = self.load_keeper_config()
        if not config:
            print("❌ No keeper configuration found. Run '/draft keeper setup' first.")
            return
        
        print("🎯 TESTING KEEPER-AWARE SIMULATION")
        print("=" * 50)
        
        try:
            # Create league configuration (you might want to make this configurable)
            league_config = LeagueConfig(
                num_teams=12,
                draft_position=6,  # This should be configurable
                roster_size=16
            )
            
            # Create keeper-aware simulator
            simulator = KeeperAwareDraftSimulator(league_config, config)
            
            # Analyze user's keeper advantage
            user_advantage = simulator.get_user_keeper_advantage()
            
            print(f"\n🏆 Your Keeper Analysis:")
            if user_advantage['has_keepers']:
                print(f"   You have {len(user_advantage['keepers'])} keeper(s):")
                for keeper in user_advantage['keepers']:
                    consistency = keeper['consistency_tier']
                    consistency_emoji = "🟢" if consistency == "High" else "🟡" if consistency == "Medium" else "🔴"
                    print(f"     {consistency_emoji} {keeper['player_name']:20s} ({keeper['position']}) - "
                          f"Round {keeper['keeper_round']} - {keeper['projection']:.1f} pts")
                    print(f"        Consistency: {consistency} ({keeper['coefficient_of_variation']:.1%} variance)")
            else:
                print("   You have no keepers configured.")
            
            # Run small test simulation
            print(f"\n🎲 Running test simulation (3 rounds, 2 simulations)...")
            strategy = OutcomeAwareStrategy(risk_tolerance=0.5)
            
            results = simulator.simulate_keeper_aware_draft(
                strategy=strategy,
                num_simulations=2,
                rounds_to_simulate=3
            )
            
            scores = results['roster_scores']
            print(f"\n📊 Test Results:")
            print(f"   Mean Score: {scores['mean']:.1f} ± {scores['std']:.1f}")
            print(f"   Score Range: {scores['min']:.1f} - {scores['max']:.1f}")
            
            # Show keeper impact
            if 'keeper_impact' in results:
                impact = results['keeper_impact']
                print(f"\n🏆 Keeper Impact Analysis:")
                print(f"   Total kept projection: {impact['total_kept_projection']:.1f}")
                print(f"   Players removed from pool: {impact['players_removed_from_pool']}")
                print(f"   Draft picks used by keepers: {impact['draft_picks_used']}")
        
        except Exception as e:
            print(f"❌ Error running test simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_keepers(self):
        """Clear all keeper configurations"""
        if Path(self.config_file).exists():
            response = input("🗑️  Are you sure you want to clear all keeper configurations? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                Path(self.config_file).unlink()
                print("✅ Keeper configuration cleared.")
            else:
                print("Clear cancelled.")
        else:
            print("No keeper configuration to clear.")


def handle_keeper_commands(args: list) -> None:
    """Main entry point for keeper CLI commands"""
    cli = KeeperCLI()
    cli.handle_keeper_command(args)


if __name__ == "__main__":
    # Test the CLI
    print("🏆 Testing Keeper CLI")
    print("=" * 30)
    
    cli = KeeperCLI()
    cli.show_keeper_help()
