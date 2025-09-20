"""
CLI integration for draft simulation
Provides commands to run draft analysis from ProjectHowie CLI
"""

from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .models import LeagueConfig, KeeperPlayer
from .analysis_generator import DraftAnalysisGenerator
from .database import DraftDatabaseConnector
from .keeper_system import KeeperManager, KeeperValidator, Keeper, KeeperConfiguration
from .player_comparison import compare_players_command

console = Console()


class ConfigState:
    """Tracks configuration wizard state for navigation"""
    
    def __init__(self):
        self.current_step = 0
        self.config_data = {}
        self.steps = [
            ("num_teams", "Number of teams"),
            ("draft_position", "Draft position"),
            ("scoring_type", "Scoring type"),
            ("qb_slots", "QB slots"),
            ("rb_slots", "RB slots"),
            ("wr_slots", "WR slots"),
            ("te_slots", "TE slots"),
            ("flex_slots", "FLEX slots"),
            ("roster_size", "Total roster size"),
            ("keepers_enabled", "Enable keepers"),
            ("keeper_config", "Keeper configuration")
        ]
    
    def set_value(self, key: str, value: Any):
        """Set a configuration value"""
        self.config_data[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config_data.get(key, default)
    
    def get_current_step_name(self) -> str:
        """Get the name of the current step"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step][1]
        return "Unknown"
    
    def can_go_back(self) -> bool:
        """Check if we can go back to previous step"""
        return self.current_step > 0
    
    def go_back(self):
        """Go to previous step"""
        if self.can_go_back():
            self.current_step -= 1
    
    def go_forward(self):
        """Go to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
    
    def show_current_config(self):
        """Display current configuration state"""
        if not self.config_data:
            console.print("\n[dim]No configuration set yet[/dim]")
            return
        
        table = Table(title="🔧 Current Configuration", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Status", style="yellow", width=10)
        
        for i, (key, name) in enumerate(self.steps):
            if key in self.config_data:
                value = self.config_data[key]
                if key == "keepers_enabled":
                    value = "Yes" if value else "No"
                elif key == "keeper_config":
                    if value:
                        value = f"{len(value.keepers)} keepers" if hasattr(value, 'keepers') else "Configured"
                    else:
                        value = "None"
                
                status = "✓ Done" if i < self.current_step else "→ Current" if i == self.current_step else "Pending"
                table.add_row(name, str(value), status)
            else:
                status = "→ Current" if i == self.current_step else "Pending"
                table.add_row(name, "[dim]Not set[/dim]", status)
        
        console.print(table)


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
        elif subcommand == "strategy":
            return self._handle_strategy(parts[1:])
        elif subcommand == "view":
            return self._handle_view_results(parts[1:])
        elif subcommand == "compare":
            return self._handle_player_compare(parts[1:])
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
[green]/draft/strategy[/green] - Strategy management and quick access
[green]/draft/view[/green]     - View Monte Carlo simulation results
[green]/draft/compare[/green]  - Compare two players side-by-side
[green]/draft/strategy/recommendations[/green] - 5-7 options for each round (all 16 rounds)
[green]/draft/strategy/positional[/green]       - Primary & backup plans by position
[green]/draft/help[/green]     - Show this help

[bold]Monte Carlo Examples:[/bold]
[dim]/draft/monte/25/16[/dim]                 - 25 simulations, full 16 rounds
[dim]/draft/monte/config[/dim]                - Choose configuration first
[dim]/draft/monte/100/15[/dim]                - 100 simulations, 15 rounds
[dim]/draft/monte/25/8/realistic[/dim]        - Use realistic opponents (default)
[dim]/draft/monte/25/8/personalities[/dim]    - Use AI personalities
[dim]/draft/monte/50/12/enhanced[/dim]        - Use enhanced distributions

[bold]Configuration Examples:[/bold]
[dim]/draft/config/position/10[/dim]          - Set draft position to 10
[dim]/draft/config/teams/12[/dim]             - Set league to 12 teams
[dim]/draft/config/scoring/ppr[/dim]          - Set PPR scoring

[bold]Strategy Examples:[/bold]
[dim]/draft/strategy[/dim]                    - Show strategy menu
[dim]/draft/strategy/current[/dim]            - Show current strategy details
[dim]/draft/strategy/generate[/dim]           - Generate new optimal strategy
[dim]/draft/strategy/refresh[/dim]            - Force refresh with latest config
[dim]/draft/strategy/migrate[/dim]            - Migrate to unified config system
[dim]/draft/strategy/round/3[/dim]            - Show Round 3 strategy details

[bold]Player Comparison Examples:[/bold]
[dim]/draft/compare/Saquon Barkley/Josh Jacobs[/dim] - Compare two RBs
[dim]/draft/compare/Jefferson/Chase[/dim]             - Compare two WRs (partial names)
[dim]/draft/compare/Lamar/Mahomes[/dim]               - Compare different positions

[bold]View Examples:[/bold]
[dim]/draft/view[/dim]                        - Show results menu
[dim]/draft/view/current[/dim]                - Show last simulation details
[dim]/draft/view/availability[/dim]           - Show player availability analysis

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
        """Interactive league configuration wizard with navigation support"""
        console.print("\n[bold green]🏈 League Configuration Wizard[/bold green]")
        console.print("[dim]Use 'back' to go to previous step • Press Enter for default values[/dim]")
        
        try:
            config_state = ConfigState()
            
            while config_state.current_step < len(config_state.steps):
                step_key, step_name = config_state.steps[config_state.current_step]
                go_back = False
                
                if step_key == "num_teams":
                    value, go_back = self._prompt_int("Number of teams", 
                                                    config_state.get_value("num_teams", 12), 8, 16, config_state)
                    if not go_back:
                        config_state.set_value("num_teams", value)
                
                elif step_key == "draft_position":
                    num_teams = config_state.get_value("num_teams", 12)
                    value, go_back = self._prompt_int("Your draft position", 
                                                    config_state.get_value("draft_position", 6), 1, num_teams, config_state)
                    if not go_back:
                        config_state.set_value("draft_position", value)
                
                elif step_key == "scoring_type":
                    value, go_back = self._prompt_choice("Scoring type", ["ppr", "half_ppr", "standard"], 
                                                       config_state.get_value("scoring_type", "ppr"), config_state)
                    if not go_back:
                        config_state.set_value("scoring_type", value)
                
                elif step_key == "qb_slots":
                    value, go_back = self._prompt_int("QB slots", 
                                                    config_state.get_value("qb_slots", 1), 1, 2, config_state)
                    if not go_back:
                        config_state.set_value("qb_slots", value)
                
                elif step_key == "rb_slots":
                    value, go_back = self._prompt_int("RB slots", 
                                                    config_state.get_value("rb_slots", 2), 1, 3, config_state)
                    if not go_back:
                        config_state.set_value("rb_slots", value)
                
                elif step_key == "wr_slots":
                    value, go_back = self._prompt_int("WR slots", 
                                                    config_state.get_value("wr_slots", 2), 1, 4, config_state)
                    if not go_back:
                        config_state.set_value("wr_slots", value)
                
                elif step_key == "te_slots":
                    value, go_back = self._prompt_int("TE slots", 
                                                    config_state.get_value("te_slots", 1), 1, 2, config_state)
                    if not go_back:
                        config_state.set_value("te_slots", value)
                
                elif step_key == "flex_slots":
                    value, go_back = self._prompt_int("FLEX slots", 
                                                    config_state.get_value("flex_slots", 1), 0, 3, config_state)
                    if not go_back:
                        config_state.set_value("flex_slots", value)
                
                elif step_key == "roster_size":
                    # Calculate minimum roster size
                    qb = config_state.get_value("qb_slots", 1)
                    rb = config_state.get_value("rb_slots", 2) 
                    wr = config_state.get_value("wr_slots", 2)
                    te = config_state.get_value("te_slots", 1)
                    flex = config_state.get_value("flex_slots", 1)
                    starting_slots = qb + rb + wr + te + flex + 2  # +2 for K/DEF
                    
                    value, go_back = self._prompt_int("Total roster size", 
                                                    config_state.get_value("roster_size", 16), 
                                                    starting_slots + 3, 20, config_state)
                    if not go_back:
                        config_state.set_value("roster_size", value)
                        # Calculate bench slots
                        bench_slots = value - starting_slots
                        config_state.set_value("bench_slots", bench_slots)
                
                elif step_key == "keepers_enabled":
                    value, go_back = self._prompt_yes_no("Does your league use keepers?", 
                                                       config_state.get_value("keepers_enabled", False), config_state)
                    if not go_back:
                        config_state.set_value("keepers_enabled", value)
                
                elif step_key == "keeper_config":
                    if config_state.get_value("keepers_enabled", False):
                        console.print("\n[bold]🏆 Keeper Configuration:[/bold]")
                        keeper_config, go_back = self._setup_keepers_interactive_with_nav(
                            config_state.get_value("num_teams"), 
                            config_state.get_value("draft_position"),
                            config_state
                        )
                        if not go_back:
                            config_state.set_value("keeper_config", keeper_config)
                    else:
                        config_state.set_value("keeper_config", None)
                        # Allow going back even when keepers are disabled
                
                # Handle navigation
                if go_back:
                    config_state.go_back()
                else:
                    config_state.go_forward()
            
            # Create final config
            config = LeagueConfig(
                num_teams=config_state.get_value("num_teams"),
                roster_size=config_state.get_value("roster_size"),
                scoring_type=config_state.get_value("scoring_type"),
                draft_position=config_state.get_value("draft_position"),
                qb_slots=config_state.get_value("qb_slots"),
                rb_slots=config_state.get_value("rb_slots"),
                wr_slots=config_state.get_value("wr_slots"),
                te_slots=config_state.get_value("te_slots"),
                flex_slots=config_state.get_value("flex_slots"),
                bench_slots=config_state.get_value("bench_slots"),
                keepers_enabled=config_state.get_value("keeper_config") is not None
            )
            
            # Save the configuration
            try:
                config.save_to_file()
                console.print("[green]💾 League configuration saved![/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not save league config: {e}[/yellow]")
            
            # Show final summary
            console.print("\n[bold green]✅ Configuration Complete![/bold green]")
            self._show_config_summary(config)
            
            # Ask if they want to run analysis
            run_analysis, _ = self._prompt_yes_no("Run draft analysis now?", True)
            if run_analysis:
                return self._run_full_analysis(config, keeper_config=config_state.get_value("keeper_config"))
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
    
    def _run_full_analysis(self, config: LeagueConfig, rounds: int = 16, keeper_config: Optional[KeeperConfiguration] = None) -> str:
        """Run the full draft analysis"""
        
        console.print(f"[bold]Generating draft analysis for {rounds} rounds...[/bold]")
        
        try:
            # Convert keeper configuration to KeeperPlayer format if provided
            keepers = None
            if keeper_config:
                keepers = []
                for keeper in keeper_config.keepers:
                    keeper_player = KeeperPlayer(
                        player_name=keeper.player_name,
                        team_name=keeper.team_name,
                        keeper_round=keeper.keeper_round,
                        original_round=keeper.original_round
                    )
                    keepers.append(keeper_player)
                
                console.print(f"[dim]🏆 Including {len(keepers)} keepers in analysis...[/dim]")
            
            analysis = self.analysis_generator.generate_pre_draft_analysis(
                config, keepers=keepers, rounds_to_analyze=rounds
            )
            
            # Display the analysis
            console.print(analysis)
            
            return "\n✅ Draft analysis complete!"
            
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def _prompt_int(self, prompt: str, default: int, min_val: int, max_val: int, 
                   config_state: Optional[ConfigState] = None) -> tuple[int, bool]:
        """Prompt for integer input with validation and navigation
        
        Returns:
            tuple: (value, go_back_requested)
        """
        while True:
            # Show current config if provided
            if config_state:
                config_state.show_current_config()
                console.print(f"\n[bold cyan]Step {config_state.current_step + 1}/{len(config_state.steps)}: {config_state.get_current_step_name()}[/bold cyan]")
            
            # Build prompt with navigation options
            nav_help = ""
            if config_state and config_state.can_go_back():
                nav_help = " ([yellow]'back'[/yellow] to go back)"
            
            try:
                response = console.input(f"[green]{prompt}[/green] [{default}]{nav_help}: ").strip().lower()
                
                # Handle back navigation
                if response == "back" and config_state and config_state.can_go_back():
                    return default, True
                
                if not response:
                    return default, False
                
                value = int(response)
                if min_val <= value <= max_val:
                    return value, False
                else:
                    console.print(f"[red]Please enter a value between {min_val} and {max_val}[/red]")
            except ValueError:
                if response != "back":  # Don't show error for 'back' command
                    console.print("[red]Please enter a valid number[/red]")
    
    def _prompt_choice(self, prompt: str, choices: List[str], default: str,
                      config_state: Optional[ConfigState] = None) -> tuple[str, bool]:
        """Prompt for choice from list with navigation
        
        Returns:
            tuple: (value, go_back_requested)
        """
        while True:
            # Show current config if provided
            if config_state:
                config_state.show_current_config()
                console.print(f"\n[bold cyan]Step {config_state.current_step + 1}/{len(config_state.steps)}: {config_state.get_current_step_name()}[/bold cyan]")
            
            choice_str = "/".join(choices)
            nav_help = ""
            if config_state and config_state.can_go_back():
                nav_help = " ([yellow]'back'[/yellow] to go back)"
            
            response = console.input(f"[green]{prompt}[/green] ({choice_str}) [{default}]{nav_help}: ").strip().lower()
            
            # Handle back navigation
            if response == "back" and config_state and config_state.can_go_back():
                return default, True
            
            if not response:
                return default, False
            
            if response in choices:
                return response, False
            
            console.print(f"[red]Please choose from: {', '.join(choices)}[/red]")
    
    def _prompt_yes_no(self, prompt: str, default: bool, 
                      config_state: Optional[ConfigState] = None) -> tuple[bool, bool]:
        """Prompt for yes/no response with navigation
        
        Returns:
            tuple: (value, go_back_requested)
        """
        # Show current config if provided
        if config_state:
            config_state.show_current_config()
            console.print(f"\n[bold cyan]Step {config_state.current_step + 1}/{len(config_state.steps)}: {config_state.get_current_step_name()}[/bold cyan]")
        
        default_str = "Y/n" if default else "y/N"
        nav_help = ""
        if config_state and config_state.can_go_back():
            nav_help = " ([yellow]'back'[/yellow] to go back)"
        
        while True:
            response = console.input(f"[green]{prompt}[/green] ({default_str}){nav_help}: ").strip().lower()
            
            # Handle back navigation
            if response == "back" and config_state and config_state.can_go_back():
                return default, True
            
            if not response:
                return default, False
            
            if response in ['y', 'yes']:
                return True, False
            elif response in ['n', 'no']:
                return False, False
            
            console.print("[red]Please enter 'y' or 'n'[/red]")
    
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
            # Check if user wants to configure before running
            config = None
            config_source = "saved"
            
            # Check for "config" argument
            if args and args[0].lower() == "config":
                console.print("[bold]🔧 Configuration Menu[/bold]")
                console.print("1. [yellow]Use saved configuration[/yellow]")
                console.print("2. [yellow]Create new configuration[/yellow]")
                console.print("3. [yellow]Use defaults[/yellow]")
                
                choice = console.input("\n[green]Choose configuration[/green] (1/2/3) [1]: ").strip()
                
                if choice == "2":
                    # Run interactive config
                    console.print("\n[bold]Creating new configuration...[/bold]")
                    config_result = self._interactive_config()
                    if config_result.startswith("❌") or config_result.startswith("Configuration cancelled"):
                        return config_result
                    # Load the newly created config
                    config = LeagueConfig.load_from_file()
                    config_source = "new"
                elif choice == "3":
                    config = LeagueConfig()
                    config_source = "defaults"
                    console.print("[yellow]Using default configuration[/yellow]")
                else:  # choice == "1" or empty
                    config = LeagueConfig.load_from_file()
                    config_source = "saved"
                
                # Remove "config" from args for further processing
                args = args[1:]
            
            # Load configuration if not set yet
            if config is None:
                config = LeagueConfig.load_from_file()
                if config is None:
                    console.print("[yellow]⚠️  No saved configuration found, using defaults[/yellow]")
                    config = LeagueConfig()  # Use defaults
                    config_source = "defaults"
                else:
                    console.print("[green]✅ Using saved league configuration[/green]")
            
            # Default values (can be overridden by args)
            num_sims = 25
            rounds = 16
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
            
            # Load keeper configuration if available
            keeper_config = None
            if config.keepers_enabled:
                try:
                    from .keeper_system import KeeperManager
                    keeper_manager = KeeperManager()
                    keeper_config = keeper_manager.load_keeper_configuration("data/keeper_config.json")
                    console.print(f"[green]🏆 Loaded keeper configuration with {len(keeper_config.keepers)} keepers[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not load keeper config: {e}[/yellow]")
            
            # Import and run Monte Carlo simulation
            if use_enhanced:
                # Use enhanced Monte Carlo with distributions
                if keeper_config:
                    from .keeper_integration import KeeperAwareDraftSimulator
                    simulator = KeeperAwareDraftSimulator(config, keeper_config)
                else:
                    from .enhanced_monte_carlo import EnhancedMonteCarloSimulator
                    simulator = EnhancedMonteCarloSimulator(config)
                
                console.print(f"🎲 Starting Enhanced Monte Carlo simulation...")
                console.print(f"   Simulations: {num_sims:,}")
                console.print(f"   Rounds: {rounds}")
                console.print(f"   Your Position: #{config.draft_position} of {config.num_teams}")
                console.print(f"   Scoring: {config.scoring_type.upper()}")
                keeper_info = f" with {len(keeper_config.keepers)} keepers" if keeper_config else ""
                console.print(f"   Mode: Enhanced with Player Distributions{keeper_info}")
                
                # Run enhanced simulation
                if keeper_config:
                    # Use keeper-aware simulation method
                    from .enhanced_monte_carlo import OutcomeAwareStrategy
                    strategy = OutcomeAwareStrategy(risk_tolerance=0.5)
                    results = simulator.simulate_keeper_aware_draft(
                        strategy=strategy,
                        num_simulations=num_sims,
                        rounds_to_simulate=rounds
                    )
                else:
                    # Use regular enhanced simulation method
                    from .enhanced_monte_carlo import OutcomeAwareStrategy
                    strategy = OutcomeAwareStrategy(risk_tolerance=0.5)
                    results = simulator.simulate_draft_with_outcomes(
                        strategy=strategy,
                        num_simulations=num_sims,
                        rounds_to_simulate=rounds
                    )
                
                # Save results for viewing
                from .monte_carlo_viewer import MonteCarloResultsViewer
                viewer = MonteCarloResultsViewer()
                sim_type = "Keeper-Aware" if keeper_config else "Enhanced"
                session_name = f"{sim_type} MC {num_sims}x{rounds} ({config.num_teams}T {config.scoring_type.upper()} #{config.draft_position})"
                viewer.save_results(results, session_name)
                
                # Generate report
                if keeper_config:
                    # Use keeper-aware report generator
                    report = simulator.generate_keeper_aware_availability_report(results)
                    console.print("")
                    console.print(report)
                else:
                    # For regular enhanced simulation, create a basic summary
                    console.print("\n[bold green]📊 Enhanced Simulation Results[/bold green]")
                    console.print("Simulation completed with player distribution analysis")
                    
                    if 'user_roster' in results:
                        console.print(f"\n[bold]Your Projected Draft Results:[/bold]")
                        user_roster = results['user_roster']
                        for i, player in enumerate(user_roster):
                            round_num = i + 1
                            console.print(f"  Round {round_num}: {player.get('name', 'Unknown')} ({player.get('position', 'N/A')})")
                
            else:
                # Use standard Monte Carlo simulation
                from .monte_carlo_simulator import MonteCarloSimulator
                
                simulator = MonteCarloSimulator(config, players, use_realistic_opponents=use_realistic)
                
                opponent_type = "Realistic (ADP+noise)" if use_realistic else "AI Personalities"
                console.print(f"🎲 Starting Monte Carlo simulation...")
                console.print(f"   Simulations: {num_sims:,}")
                console.print(f"   Rounds: {rounds}")
                console.print(f"   Your Position: #{config.draft_position} of {config.num_teams}")
                console.print(f"   Scoring: {config.scoring_type.upper()}")
                keeper_info = f" (with {len(keeper_config.keepers)} keepers)" if keeper_config else ""
                console.print(f"   Opponent Model: {opponent_type}{keeper_info}")
                
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
    
    def _setup_keepers_interactive(self, num_teams: int, user_draft_position: int) -> Optional[KeeperConfiguration]:
        """Interactive keeper configuration for each team"""
        
        # Initialize keeper system
        validator = KeeperValidator()
        
        # Get keeper rules
        console.print("\n[dim]Keeper Rules:[/dim]")
        console.print("1. [yellow]First Round[/yellow] - All keepers count as 1st round picks")
        console.print("2. [yellow]Round Based[/yellow] - Keepers count as the round they were drafted last year")
        console.print("[dim]Type 'skip' to skip keeper configuration[/dim]")
        
        while True:
            choice_input = console.input("[green]Keeper rules[/green] (1/2) [2]: ").strip().lower()
            
            if choice_input == "skip":
                return None
            
            if not choice_input:
                choice_input = "2"
            
            if choice_input == "1":
                keeper_rules = "first_round"
                break
            elif choice_input == "2":
                keeper_rules = "round_based"
                break
            else:
                console.print("[red]Please choose 1 or 2, or type 'skip'[/red]")
        
        keepers = []
        
        # Get number of teams with keepers
        console.print("[dim]Type 'skip' to skip keeper configuration[/dim]")
        while True:
            response = console.input("[green]How many teams have keepers?[/green] [0]: ").strip().lower()
            
            if response == "skip":
                return None
            
            if not response:
                num_teams_with_keepers = 0
                break
            
            try:
                num_teams_with_keepers = int(response)
                if 0 <= num_teams_with_keepers <= num_teams:
                    break
                else:
                    console.print(f"[red]Please enter a number between 0 and {num_teams}[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number or 'skip'[/red]")
        
        if num_teams_with_keepers == 0:
            return None
        
        # Configure each team's keepers
        for team_num in range(num_teams_with_keepers):
            console.print(f"\n[bold cyan]--- Team {team_num + 1} Keepers ---[/bold cyan]")
            
            # Get team info
            default_team_name = f"Team{team_num + 1}"
            if team_num == 0:  # First team might be the user
                default_team_name = "Your Team"
            
            team_name = console.input(f"[dim]Team {team_num + 1} name[/dim] [yellow]({default_team_name})[/yellow]: ").strip()
            if not team_name:
                team_name = default_team_name
            
            draft_position, _ = self._prompt_int(f"Team {team_name} draft position", user_draft_position if team_num == 0 else 1, 1, num_teams)
            
            # Get keepers for this team
            num_keepers, _ = self._prompt_int(f"How many keepers does {team_name} have?", 0, 0, 3)
            
            for keeper_num in range(num_keepers):
                console.print(f"\n[dim]  Keeper {keeper_num + 1} for {team_name}:[/dim]")
                
                # Get player name with validation
                while True:
                    player_name = console.input("    [yellow]Player name[/yellow]: ").strip()
                    if not player_name:
                        continue
                    
                    # Validate player
                    matches = validator.find_player_matches(player_name)
                    
                    if not matches:
                        console.print(f"    [red]❌ Player '{player_name}' not found[/red]")
                        # Try broader search
                        broad_matches = validator.find_player_matches(player_name, max_suggestions=5)
                        if broad_matches:
                            console.print("    [dim]Suggestions:[/dim]")
                            for i, (suggestion, score) in enumerate(broad_matches[:3]):
                                console.print(f"      {i+1}. [cyan]{suggestion}[/cyan]")
                        continue
                    
                    elif matches[0][1] < 1.0:
                        console.print(f"    [yellow]Did you mean:[/yellow]")
                        for i, (suggestion, score) in enumerate(matches[:3]):
                            console.print(f"      {i+1}. [cyan]{suggestion}[/cyan] ({score:.1%} match)")
                        
                        choice = console.input("    [dim]Choose number or type new name[/dim]: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(matches):
                            player_name = matches[int(choice) - 1][0]
                        else:
                            continue
                    else:
                        player_name = matches[0][0]
                    
                    # Show player info
                    player_info = validator._get_player_info(player_name)
                    if player_info:
                        console.print(f"    [green]✅ {player_info['name']} ({player_info['position']}) - {player_info['team']} - {player_info['projection']:.1f} pts[/green]")
                    break
                
                # Get keeper round
                if keeper_rules == "first_round":
                    keeper_round = 1
                    console.print(f"    [dim]Keeper round: 1 (first round rule)[/dim]")
                else:
                    keeper_round, _ = self._prompt_int("    What round does this keeper cost?", 8, 1, 16)
                
                # Create keeper
                keeper = Keeper(
                    team_name=team_name,
                    draft_position=draft_position,
                    player_name=player_name,
                    keeper_round=keeper_round
                )
                
                keepers.append(keeper)
                
                # Calculate and show pick number
                if keeper_round % 2 == 1:  # Odd rounds
                    pick_in_round = draft_position
                else:  # Even rounds (snake)
                    pick_in_round = num_teams + 1 - draft_position
                overall_pick = (keeper_round - 1) * num_teams + pick_in_round
                
                console.print(f"    [green]✅ Added {player_name} as Round {keeper_round} keeper (Pick #{overall_pick}) for {team_name}[/green]")
        
        # Create and validate configuration
        config = KeeperConfiguration(keepers=keepers, keeper_rules=keeper_rules)
        
        # Show summary
        console.print(f"\n[bold]🏆 Keeper Configuration Summary:[/bold]")
        validation = validator.validate_keeper_configuration(config)
        
        if validation['valid']:
            console.print(f"[green]✅ Configuration valid![/green]")
        else:
            console.print(f"[red]❌ Configuration has issues:[/red]")
            for issue in validation.get('conflicts', []):
                console.print(f"   [red]• {issue}[/red]")
        
        console.print(f"   Total keepers: {len(config.keepers)}")
        console.print(f"   Teams with keepers: {validation['summary']['teams_with_keepers']}")
        
        # Show keepers by round
        keepers_by_round = {}
        for keeper in config.keepers:
            round_num = keeper.keeper_round
            if round_num not in keepers_by_round:
                keepers_by_round[round_num] = []
            keepers_by_round[round_num].append(keeper)
        
        for round_num in sorted(keepers_by_round.keys()):
            console.print(f"\n   [cyan]Round {round_num}:[/cyan]")
            for keeper in keepers_by_round[round_num]:
                # Calculate pick number
                if round_num % 2 == 1:
                    pick_in_round = keeper.draft_position
                else:
                    pick_in_round = num_teams + 1 - keeper.draft_position
                overall_pick = (round_num - 1) * num_teams + pick_in_round
                
                console.print(f"     Pick #{overall_pick:2d}: [yellow]{keeper.player_name:20s}[/yellow] (kept by {keeper.team_name})")
        
        # Save configuration
        try:
            from pathlib import Path
            Path("data").mkdir(exist_ok=True)
            
            manager = KeeperManager()
            manager.save_keeper_configuration(config, "data/keeper_config.json")
            console.print(f"\n[green]💾 Keeper configuration saved to data/keeper_config.json[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not save keeper config: {e}[/yellow]")
        
        return config
    
    def _setup_keepers_interactive_with_nav(self, num_teams: int, user_draft_position: int, config_state: ConfigState) -> tuple[Optional[KeeperConfiguration], bool]:
        """Interactive keeper configuration with navigation support"""
        
        # Show navigation info
        config_state.show_current_config()
        
        console.print("\n[dim]Available options:[/dim]")
        console.print("• [yellow]skip[/yellow] - Skip keeper configuration")
        console.print("• [yellow]back[/yellow] - Go to previous step")
        console.print("• Press Enter to continue with keeper setup")
        
        response = console.input("\n[green]Set up keepers?[/green] (y/n/skip/back) [y]: ").strip().lower()
        
        if response == "back":
            return None, True
        elif response in ["skip", "n", "no"]:
            console.print("[yellow]Skipping keeper configuration[/yellow]")
            return None, False
        elif response in ["", "y", "yes"]:
            # Continue with keeper setup
            try:
                keeper_config = self._setup_keepers_interactive(num_teams, user_draft_position)
                return keeper_config, False
            except KeyboardInterrupt:
                console.print("\n[yellow]Keeper setup cancelled[/yellow]")
                return None, True
        else:
            console.print("[red]Please enter y/n, skip, or back[/red]")
            return self._setup_keepers_interactive_with_nav(num_teams, user_draft_position, config_state)
    
    def _handle_strategy(self, args: List[str]) -> str:
        """Handle strategy commands"""
        from .strategy_manager import StrategyManager
        from .strategy_tree_search import StrategyTreeSearch
        from .strategy_summarizer import StrategyAIAnalyzer, generate_strategy_summary_prompt
        
        manager = StrategyManager()
        
        if not args:
            # Show strategy menu
            return manager.show_strategy_menu()
        
        command = args[0].lower()
        
        if command == "current":
            return manager.show_current_strategy()
        
        elif command == "generate":
            return self._generate_new_strategy(manager)
        
        elif command == "recommendations" or command == "recs":
            return self._generate_round_recommendations()
        
        elif command == "positional" or command == "pos":
            return self._generate_positional_strategy()
        
        elif command == "round":
            if len(args) < 2:
                return "[red]Usage: /draft/strategy/round/3[/red]"
            try:
                round_number = int(args[1])
                return manager.show_round_details(round_number)
            except ValueError:
                return "[red]Invalid round number[/red]"
        
        elif command.isdigit():
            # Load strategy by number from menu
            sessions = manager.list_strategies()
            try:
                strategy_index = int(command) - 1
                if 0 <= strategy_index < len(sessions):
                    session = sessions[strategy_index]
                    loaded_strategy = manager.load_strategy(session.session_id)
                    if loaded_strategy:
                        return manager.show_current_strategy()
                    else:
                        return "[red]Failed to load strategy[/red]"
                else:
                    return f"[red]Strategy number {command} not found. Use /draft/strategy to see available strategies.[/red]"
            except ValueError:
                return "[red]Invalid strategy number[/red]"
        
        elif command == "summary":
            # Generate AI summary
            if not manager.current_strategy:
                return "[yellow]No current strategy loaded. Load a strategy first.[/yellow]"
            
            # Generate AI analysis prompt
            prompt = generate_strategy_summary_prompt(manager.current_strategy)
            
            console.print("[bold]🤖 AI Strategy Analysis Prompt Generated[/bold]")
            console.print("[dim]Copy this prompt to Claude for detailed analysis:[/dim]")
            console.print(Panel(prompt, title="Claude Analysis Prompt", border_style="blue"))
            
            return ""
        
        elif command == "refresh":
            # Force refresh by generating new strategy (clears any cached data)
            return self._generate_new_strategy(manager)
        
        elif command == "migrate":
            return self._migrate_to_unified_config()
        
        else:
            return f"[red]Unknown strategy command: {command}[/red]\nAvailable: current, generate, round/N, summary, or strategy number"
    
    def _generate_new_strategy(self, manager) -> str:
        """Generate a new optimal strategy using tree search"""
        from .strategy_tree_search import StrategyTreeSearch
        
        try:
            # Load configuration
            config = LeagueConfig.load_from_file()
            if config is None:
                console.print("[yellow]⚠️  No saved configuration found. Please run /draft/config first.[/yellow]")
                return "❌ No configuration found. Run /draft/config to set up your league."
            
            # Load players
            players = self.db.load_player_universe()
            if not players:
                return "❌ No player data found. Please check database connection."
            
            console.print("[bold]🌳 Generating optimal draft strategy...[/bold]")
            console.print(f"[dim]League: {config.num_teams} teams, {config.scoring_type.upper()}, position #{config.draft_position}[/dim]")
            console.print("[yellow]⏳ This may take 1-3 minutes (optimized for TUI)...[/yellow]")
            console.print("[dim]💡 Tree search analyzes draft scenarios to find optimal picks[/dim]")
            
            # Run tree search with TUI optimization
            tree_search = StrategyTreeSearch(config, players, fast_mode=True)
            strategy = tree_search.find_optimal_strategy()
            
            # Save strategy
            strategy_name = f"{config.num_teams}T {config.scoring_type.upper()} #{config.draft_position}"
            session_id = manager.save_strategy(strategy, strategy_name)
            
            # Show the generated strategy
            return manager.show_current_strategy()
            
        except Exception as e:
            return f"❌ Error generating strategy: {str(e)}"
    
    def _handle_view_results(self, args: List[str]) -> str:
        """Handle viewing Monte Carlo results"""
        from .monte_carlo_viewer import MonteCarloResultsViewer
        
        viewer = MonteCarloResultsViewer()
        
        if not args:
            # Show results menu
            return viewer.show_results_menu()
        
        command = args[0].lower()
        
        if command == "current":
            return viewer.show_detailed_results()
        
        elif command == "availability":
            return viewer.show_availability_analysis()
        
        elif command.isdigit():
            # Load results by number from menu
            session_ids = viewer.list_saved_results()
            try:
                result_index = int(command) - 1
                if 0 <= result_index < len(session_ids):
                    session_id = session_ids[result_index]
                    loaded_results = viewer.load_results(session_id)
                    if loaded_results:
                        return viewer.show_detailed_results(loaded_results)
                    else:
                        return "[red]Failed to load results[/red]"
                else:
                    return f"[red]Result number {command} not found. Use /draft/view to see available results.[/red]"
            except ValueError:
                return "[red]Invalid result number[/red]"
        
        else:
            return f"[red]Unknown view command: {command}[/red]\nAvailable: current, availability, or result number"
    
    def _migrate_to_unified_config(self) -> str:
        """Migrate from separate configs to unified configuration"""
        from .unified_config import UnifiedConfigManager
        
        try:
            console.print("[bold]🔄 Migrating to Unified Configuration[/bold]")
            console.print("[dim]Combining league_config.json and keeper_config.json...[/dim]")
            
            manager = UnifiedConfigManager()
            success = manager.migrate_and_save()
            
            if success:
                config = manager.get_current_config()
                
                # Show migration results
                console.print("[green]✅ Migration completed successfully![/green]")
                console.print(f"\n[bold]📋 Unified Configuration:[/bold]")
                console.print(f"   League: {config.num_teams}T {config.scoring_type.upper()}")
                console.print(f"   Your Team: {config.your_team_name} (Position #{config.your_draft_position})")
                console.print(f"   Keepers: {len(config.get_all_keepers())} total")
                
                # Show file info
                console.print(f"\n[bold]📁 File Organization:[/bold]")
                console.print(f"   ✅ Created: data/draft_config.json (unified)")
                console.print(f"   📋 Contains: League settings + all team keepers")
                console.print(f"   🔗 No more config mismatches!")
                
                console.print(f"\n[dim]💡 Old files (league_config.json, keeper_config.json) are kept as backup[/dim]")
                
                return "✅ Successfully migrated to unified configuration system"
            else:
                return "❌ Migration failed - check that your config files exist"
                
        except Exception as e:
            return f"❌ Migration error: {str(e)}"
    
    def _generate_round_recommendations(self) -> str:
        """Generate 5-7 recommendations for each of 16 rounds"""
        try:
            # Load configuration
            league_config = LeagueConfig.load_from_file("data/league_config.json")
            if not league_config:
                return "❌ No league configuration found. Please run /draft config first."
            
            # Load players
            db = DraftDatabaseConnector()
            players = db.load_player_universe()
            
            # Create tree search instance
            from .strategy_tree_search import StrategyTreeSearch
            tree_search = StrategyTreeSearch(league_config, players, use_monte_carlo=False)
            
            # Generate round-by-round recommendations
            return tree_search.generate_round_by_round_recommendations()
            
        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"
    
    def _generate_positional_strategy(self) -> str:
        """Generate contingency-based positional strategy guide"""
        try:
            # Load configuration
            league_config = LeagueConfig.load_from_file("data/league_config.json")
            if not league_config:
                return "❌ No league configuration found. Please run /draft config first."
            
            # Load players
            from .database import DraftDatabaseConnector
            db = DraftDatabaseConnector()
            players = db.load_player_universe()
            
            # Create tree search instance
            from .strategy_tree_search import StrategyTreeSearch
            tree_search = StrategyTreeSearch(league_config, players, use_monte_carlo=False)
            
            # Generate positional strategy guide
            return tree_search.generate_positional_strategy_guide()
            
        except Exception as e:
            return f"❌ Error generating positional strategy: {str(e)}"
    
    def _handle_player_compare(self, args: List[str]) -> str:
        """Handle player comparison command"""
        if len(args) < 2:
            return """❌ Player comparison requires two player names.

🔍 Usage Examples:
   /draft/compare/Saquon Barkley/Josh Jacobs
   /draft/compare/Jefferson/Chase  
   /draft/compare/Lamar/Mahomes

💡 You can use partial names - the system will search for matches."""
        
        try:
            # Load configuration and players
            league_config = LeagueConfig.load_from_file("data/league_config.json")
            if not league_config:
                return "❌ No league configuration found. Please run /draft config first."
            
            from .database import DraftDatabaseConnector
            db = DraftDatabaseConnector()
            players = db.load_player_universe()
            
            # Extract player names from args
            player1_name = args[0].replace("_", " ")  # Handle URL encoding
            player2_name = args[1].replace("_", " ")
            
            # Use the comparison function
            return compare_players_command(league_config, players, player1_name, player2_name)
            
        except Exception as e:
            return f"❌ Error comparing players: {str(e)}"
