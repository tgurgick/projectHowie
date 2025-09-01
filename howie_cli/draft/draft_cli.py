"""
CLI integration for draft simulation
Provides commands to run draft analysis from ProjectHowie CLI
"""

from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .models import LeagueConfig, KeeperPlayer
from .analysis_generator import DraftAnalysisGenerator
from .database import DraftDatabaseConnector

console = Console()


class DraftCLI:
    """CLI interface for draft simulation"""
    
    def __init__(self):
        self.analysis_generator = DraftAnalysisGenerator()
        self.db = DraftDatabaseConnector()
    
    def handle_draft_command(self, command_str: str) -> str:
        """Handle draft-related commands using slash format"""
        
        if not command_str.strip():
            return self._show_draft_help()
        
        # Parse slash-separated command
        parts = [part for part in command_str.strip().split('/') if part]
        
        if not parts:
            return self._show_draft_help()
        
        subcommand = parts[0].lower()
        
        if subcommand == "help":
            return self._show_draft_help()
        elif subcommand == "test":
            return self._test_connection()
        elif subcommand == "config":
            return self._handle_config(parts[1:])
        elif subcommand == "analyze":
            return self._run_analysis(parts[1:])
        elif subcommand == "quick":
            return self._quick_analysis()
        elif subcommand == "monte" or subcommand == "montecarlo":
            return self._run_monte_carlo(parts[1:])
        elif subcommand == "simulate":
            return self._run_simulation(parts[1:])
        else:
            return f"Unknown draft command: {subcommand}. Use '/draft/help' for available commands."
    
    def _show_draft_help(self) -> str:
        """Show available draft commands"""
        help_panel = Panel(
            """[bold]Draft Simulation Commands:[/bold]

[green]/draft/test[/green]     - Test database connection
[green]/draft/quick[/green]    - Quick analysis with default settings  
[green]/draft/config[/green]   - Interactive league configuration
[green]/draft/analyze[/green]  - Full draft analysis
[green]/draft/monte[/green]    - Monte Carlo simulation
[green]/draft/simulate[/green] - Advanced simulation with AI opponents
[green]/draft/help[/green]     - Show this help

[bold]Monte Carlo Examples:[/bold]
[dim]/draft/monte/25/8[/dim]                  - 25 simulations, 8 rounds
[dim]/draft/monte/100/15[/dim]                - 100 simulations, 15 rounds
[dim]/draft/monte/25/8/realistic[/dim]        - Use realistic opponents (default)
[dim]/draft/monte/25/8/personalities[/dim]    - Use AI personalities
[dim]/draft/monte/50/12/enhanced[/dim]        - Use enhanced distributions

[bold]Configuration Examples:[/bold]
[dim]/draft/config/position/10[/dim]          - Set draft position to 10
[dim]/draft/config/teams/12[/dim]             - Set league to 12 teams
[dim]/draft/config/scoring/ppr[/dim]          - Set PPR scoring

[bold]Features:[/bold]
• Round-by-round pick recommendations
• Monte Carlo simulation with realistic opponents (ADP+noise)
• Enhanced evaluation (SoS, starter status, injury risk)
• VORP and scarcity analysis with player distributions
• Natural draft variance and realistic outcomes
""",
            title="🏈 Draft Simulation System",
            border_style="green"
        )
        
        console.print(help_panel)
        return ""
    
    def _test_connection(self) -> str:
        """Test database connection and show basic info"""
        try:
            info = self.db.get_database_info()
            
            # Create info table
            table = Table(title="📊 Database Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Database Path", info["db_path"])
            table.add_row("Player Count", str(info["player_count"]))
            table.add_row("Tables Found", str(len(info["tables"])))
            
            console.print(table)
            
            if info["sample_players"]:
                console.print("\n[bold]Top 5 Players:[/bold]")
                for i, player in enumerate(info["sample_players"], 1):
                    name, pos, team, pts = player
                    console.print(f"  {i}. {name} ({pos}, {team}) - {pts:.1f} pts")
            
            return "✅ Database connection successful!"
            
        except Exception as e:
            return f"❌ Database connection failed: {str(e)}"
    
    def _interactive_config(self) -> str:
        """Interactive league configuration wizard"""
        console.print("\n[bold green]🏈 League Configuration Wizard[/bold green]")
        
        try:
            # Basic settings
            console.print("\n[bold]Basic Settings:[/bold]")
            num_teams = self._prompt_int("Number of teams", 12, 8, 16)
            draft_position = self._prompt_int("Your draft position", 6, 1, num_teams)
            scoring_type = self._prompt_choice("Scoring type", ["ppr", "half_ppr", "standard"], "ppr")
            
            # Roster settings  
            console.print("\n[bold]Roster Configuration:[/bold]")
            qb_slots = self._prompt_int("QB slots", 1, 1, 2)
            rb_slots = self._prompt_int("RB slots", 2, 1, 3)
            wr_slots = self._prompt_int("WR slots", 2, 1, 4)
            te_slots = self._prompt_int("TE slots", 1, 1, 2)
            flex_slots = self._prompt_int("FLEX slots", 1, 0, 3)
            
            # Calculate bench slots
            starting_slots = qb_slots + rb_slots + wr_slots + te_slots + flex_slots + 2  # +2 for K/DEF
            roster_size = self._prompt_int("Total roster size", 16, starting_slots + 3, 20)
            bench_slots = roster_size - starting_slots
            
            # Create config
            config = LeagueConfig(
                num_teams=num_teams,
                roster_size=roster_size,
                scoring_type=scoring_type,
                draft_position=draft_position,
                qb_slots=qb_slots,
                rb_slots=rb_slots,
                wr_slots=wr_slots,
                te_slots=te_slots,
                flex_slots=flex_slots,
                bench_slots=bench_slots
            )
            
            # Show summary
            self._show_config_summary(config)
            
            # Ask if they want to run analysis
            if self._prompt_yes_no("Run draft analysis now?", True):
                return self._run_full_analysis(config)
            else:
                return "Configuration saved. Use '/draft analyze' to run analysis."
                
        except KeyboardInterrupt:
            return "Configuration cancelled."
        except Exception as e:
            return f"Configuration error: {str(e)}"
    
    def _quick_analysis(self) -> str:
        """Quick analysis with default settings"""
        console.print("[bold]Running quick draft analysis...[/bold]")
        
        # Default 12-team league, 6th pick
        config = LeagueConfig(
            num_teams=12,
            draft_position=6,
            scoring_type="ppr"
        )
        
        return self._run_full_analysis(config, rounds=6)
    
    def _run_analysis(self, args: List[str]) -> str:
        """Run analysis using slash format"""
        
        # Parse slash-separated arguments: /draft/analyze/position/teams/scoring
        config = LeagueConfig()  # Default config
        
        if len(args) >= 1:
            try:
                config.draft_position = int(args[0])
            except ValueError:
                pass
        
        if len(args) >= 2:
            try:
                config.num_teams = int(args[1])
            except ValueError:
                pass
        
        if len(args) >= 3:
            scoring = args[2].lower()
            if scoring in ["ppr", "half", "standard"]:
                config.scoring_type = scoring
        
        return self._run_full_analysis(config)
    
    def _run_full_analysis(self, config: LeagueConfig, rounds: int = 8) -> str:
        """Run the full draft analysis"""
        
        console.print(f"[bold]Generating draft analysis for {rounds} rounds...[/bold]")
        
        try:
            analysis = self.analysis_generator.generate_pre_draft_analysis(
                config, rounds_to_analyze=rounds
            )
            
            # Display the analysis
            console.print(analysis)
            
            return "\n✅ Draft analysis complete!"
            
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def _prompt_int(self, prompt: str, default: int, min_val: int, max_val: int) -> int:
        """Prompt for integer input with validation"""
        while True:
            try:
                response = input(f"{prompt} [{default}]: ").strip()
                if not response:
                    return default
                
                value = int(response)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    def _prompt_choice(self, prompt: str, choices: List[str], default: str) -> str:
        """Prompt for choice from list"""
        while True:
            choice_str = "/".join(choices)
            response = input(f"{prompt} ({choice_str}) [{default}]: ").strip().lower()
            
            if not response:
                return default
            
            if response in choices:
                return response
            
            print(f"Please choose from: {', '.join(choices)}")
    
    def _prompt_yes_no(self, prompt: str, default: bool) -> bool:
        """Prompt for yes/no response"""
        default_str = "Y/n" if default else "y/N"
        
        while True:
            response = input(f"{prompt} ({default_str}): ").strip().lower()
            
            if not response:
                return default
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            
            print("Please enter 'y' or 'n'")
    
    def _show_config_summary(self, config: LeagueConfig):
        """Show configuration summary"""
        
        table = Table(title="📋 League Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Teams", str(config.num_teams))
        table.add_row("Draft Position", str(config.draft_position))
        table.add_row("Scoring", config.scoring_type.upper())
        table.add_row("Roster Size", str(config.roster_size))
        table.add_row("Starting Lineup", f"QB:{config.qb_slots} RB:{config.rb_slots} WR:{config.wr_slots} TE:{config.te_slots} FLEX:{config.flex_slots}")
        table.add_row("Bench Slots", str(config.bench_slots))
        
        console.print(table)
    
    def _handle_config(self, args: List[str]) -> str:
        """Handle configuration commands using slash format"""
        
        if not args:
            return self._interactive_config()
        
        config_type = args[0].lower()
        
        if config_type == "position" and len(args) >= 2:
            try:
                position = int(args[1])
                if 1 <= position <= 12:
                    console.print(f"[green]✅ Draft position set to {position}[/green]")
                    return f"Draft position configured: #{position}"
                else:
                    return "[red]❌ Draft position must be between 1 and 12[/red]"
            except ValueError:
                return "[red]❌ Invalid position number[/red]"
        
        elif config_type == "teams" and len(args) >= 2:
            try:
                teams = int(args[1])
                if 8 <= teams <= 16:
                    console.print(f"[green]✅ League size set to {teams} teams[/green]")
                    return f"League size configured: {teams} teams"
                else:
                    return "[red]❌ League size must be between 8 and 16 teams[/red]"
            except ValueError:
                return "[red]❌ Invalid team count[/red]"
        
        elif config_type == "scoring" and len(args) >= 2:
            scoring = args[1].lower()
            if scoring in ["ppr", "half", "standard"]:
                console.print(f"[green]✅ Scoring set to {scoring.upper()}[/green]")
                return f"Scoring configured: {scoring.upper()}"
            else:
                return "[red]❌ Scoring must be one of: ppr, half, standard[/red]"
        
        else:
            return "[yellow]Usage: /draft/config/position/6 or /draft/config/teams/12 or /draft/config/scoring/ppr[/yellow]"
    
    def _run_monte_carlo(self, args: List[str]) -> str:
        """Run Monte Carlo simulation using slash format"""
        
        try:
            # Default values
            num_sims = 25
            rounds = 8
            position = 6
            teams = 12
            use_realistic = True
            use_enhanced = False
            
            # Parse slash-separated arguments: /draft/monte/sims/rounds/mode
            if len(args) >= 1:
                try:
                    num_sims = int(args[0])
                except ValueError:
                    pass
            
            if len(args) >= 2:
                try:
                    rounds = int(args[1])
                except ValueError:
                    pass
            
            if len(args) >= 3:
                mode = args[2].lower()
                if mode == "personalities":
                    use_realistic = False
                elif mode == "enhanced":
                    use_enhanced = True
                elif mode == "realistic":
                    use_realistic = True
            
            # Load data
            players = self.db.load_player_universe()
            if not players:
                return "❌ No player data found. Please check database connection."
            
            # Configure league
            config = LeagueConfig(
                draft_position=position,
                num_teams=teams,
                scoring_type="ppr"
            )
            
            # Import and run Monte Carlo simulation
            if use_enhanced:
                # Use enhanced Monte Carlo with distributions
                from .enhanced_monte_carlo import EnhancedMonteCarloSimulator
                
                simulator = EnhancedMonteCarloSimulator(config, players)
                
                console.print(f"🎲 Starting Enhanced Monte Carlo simulation...")
                console.print(f"   Simulations: {num_sims:,}")
                console.print(f"   Rounds: {rounds}")
                console.print(f"   Your Position: #{position} of {teams}")
                console.print(f"   Mode: Enhanced with Player Distributions")
                
                # Run enhanced simulation
                results = simulator.run_enhanced_simulation(
                    num_simulations=num_sims,
                    rounds=rounds,
                    use_distributions=True,
                    num_outcome_samples=min(10000, num_sims * 20)  # Scale samples with sims
                )
                
                # Generate enhanced report
                report = simulator.generate_enhanced_availability_report(results)
                console.print("")
                console.print(report)
                
            else:
                # Use standard Monte Carlo simulation
                from .monte_carlo_simulator import MonteCarloSimulator
                
                simulator = MonteCarloSimulator(config, players, use_realistic_opponents=use_realistic)
                
                opponent_type = "Realistic (ADP+noise)" if use_realistic else "AI Personalities"
                console.print(f"🎲 Starting Monte Carlo simulation...")
                console.print(f"   Simulations: {num_sims:,}")
                console.print(f"   Rounds: {rounds}")
                console.print(f"   Your Position: #{position} of {teams}")
                console.print(f"   Opponent Model: {opponent_type}")
                
                # Run standard simulation
                results = simulator.run_simulation(
                    num_simulations=num_sims,
                    rounds_to_simulate=rounds
                )
                
                # Generate standard report
                report = simulator.generate_availability_report(results)
                console.print("")
                console.print(report)
            
            return ""
            
        except Exception as e:
            return f"❌ Error running Monte Carlo simulation: {str(e)}"
    
    def _run_simulation(self, args: List[str]) -> str:
        """Run advanced simulation with detailed analysis"""
        
        try:
            # Similar to monte carlo but with more detailed output
            return self._run_monte_carlo(args)
            
        except Exception as e:
            return f"❌ Error running simulation: {str(e)}"
