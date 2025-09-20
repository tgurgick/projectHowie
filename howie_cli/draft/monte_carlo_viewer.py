"""
Monte Carlo Results Viewer
Display and analyze Monte Carlo simulation outcomes
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .models import Player, LeagueConfig
from ..core.paths import get_data_dir


class MonteCarloResultsViewer:
    """View and analyze Monte Carlo simulation results"""
    
    def __init__(self):
        self.console = Console()
        # Use portable data directory for Monte Carlo results
        data_dir = get_data_dir()
        self.results_dir = data_dir / "monte_carlo_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.last_results = None
    
    def save_results(self, results: Dict[str, Any], session_name: str = None) -> str:
        """Save Monte Carlo results for later viewing"""
        import time
        
        session_id = f"mc_{int(time.time())}"
        session_name = session_name or f"Monte Carlo {time.strftime('%m/%d %H:%M')}"
        
        # Prepare results for JSON serialization
        serializable_results = self._make_serializable(results)
        
        result_data = {
            'session_id': session_id,
            'session_name': session_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': serializable_results
        }
        
        # Save to file
        results_file = self.results_dir / f"{session_id}.json"
        with open(results_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        self.last_results = results
        self.console.print(f"[green]💾 Monte Carlo results saved: {session_name} (ID: {session_id})[/green]")
        
        return session_id
    
    def load_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load Monte Carlo results by session ID"""
        results_file = self.results_dir / f"{session_id}.json"
        
        if not results_file.exists():
            self.console.print(f"[red]❌ Results {session_id} not found[/red]")
            return None
        
        try:
            with open(results_file, 'r') as f:
                result_data = json.load(f)
            
            self.last_results = result_data['results']
            self.console.print(f"[green]✅ Loaded results: {result_data['session_name']}[/green]")
            return result_data['results']
            
        except Exception as e:
            self.console.print(f"[red]❌ Error loading results: {e}[/red]")
            return None
    
    def show_results_menu(self) -> str:
        """Display menu of available Monte Carlo results"""
        result_files = list(self.results_dir.glob("mc_*.json"))
        
        if not result_files:
            return "[yellow]No Monte Carlo results found. Run a simulation first.[/yellow]"
        
        # Load and display results
        sessions = []
        for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                sessions.append(data)
            except:
                continue
        
        if not sessions:
            return "[yellow]No valid results found.[/yellow]"
        
        # Create results table
        table = Table(title="🎲 Monte Carlo Results History")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Session", style="bold", width=25)
        table.add_column("Timestamp", style="dim", width=16)
        table.add_column("Simulations", style="green", width=12)
        table.add_column("Type", width=15)
        
        for i, session in enumerate(sessions[:10], 1):  # Show last 10
            results = session['results']
            sim_count = results.get('num_simulations', 'Unknown')
            
            # Determine simulation type
            if 'keeper_impact' in results:
                sim_type = "Keeper-Aware"
            elif 'variance_adjusted_vorp' in str(results):
                sim_type = "Enhanced"
            else:
                sim_type = "Standard"
            
            table.add_row(
                str(i),
                session['session_name'][:24],
                session['timestamp'].split()[1],  # Just time
                str(sim_count),
                sim_type
            )
        
        self.console.print(table)
        
        # Add usage instructions
        instructions = """
[bold]Usage:[/bold]
• [yellow]/draft/monte/view/1[/yellow] - View result #1 details
• [yellow]/draft/monte/view/current[/yellow] - View last simulation
• [yellow]/draft/monte/view/availability[/yellow] - Show player availability
        """
        
        self.console.print(Panel(instructions, title="Commands", border_style="dim"))
        return ""
    
    def show_detailed_results(self, results: Dict[str, Any] = None) -> str:
        """Show detailed Monte Carlo results"""
        if results is None:
            results = self.last_results
        
        if not results:
            return "[yellow]No Monte Carlo results to display. Run a simulation first.[/yellow]"
        
        # Simulation overview
        overview = self._generate_overview_panel(results)
        self.console.print(overview)
        
        # Roster outcomes
        if 'roster_scores' in results:
            scores_panel = self._generate_scores_panel(results['roster_scores'])
            self.console.print(scores_panel)
        
        # Sample rosters
        if 'sample_rosters' in results:
            roster_panel = self._generate_sample_rosters_panel(results['sample_rosters'])
            self.console.print(roster_panel)
        
        # Player frequency
        if 'player_frequency' in results:
            frequency_panel = self._generate_frequency_panel(results['player_frequency'])
            self.console.print(frequency_panel)
        
        # Keeper-specific analysis
        if 'keeper_impact' in results:
            keeper_panel = self._generate_keeper_panel(results['keeper_impact'])
            self.console.print(keeper_panel)
        
        return ""
    
    def show_availability_analysis(self, results: Dict[str, Any] = None) -> str:
        """Show player availability analysis"""
        if results is None:
            results = self.last_results
        
        if not results:
            return "[yellow]No Monte Carlo results to display.[/yellow]"
        
        # Get availability data
        availability_data = None
        if 'player_availability_rates' in results:
            availability_data = results['player_availability_rates']
        elif 'player_frequency' in results and isinstance(results['player_frequency'], dict):
            # Check if this is round-based availability data
            first_player_data = next(iter(results['player_frequency'].values()), {})
            if isinstance(first_player_data, dict) and all(isinstance(k, int) for k in first_player_data.keys()):
                availability_data = results['player_frequency']
        
        if not availability_data:
            return "[yellow]No player availability data found in results.[/yellow]"
        
        # Create availability table for early rounds
        table = Table(title="🎯 Player Availability by Round")
        table.add_column("Player", style="bold", width=20)
        table.add_column("Position", style="cyan", width=8)
        table.add_column("Round 1", style="green", width=8)
        table.add_column("Round 2", style="green", width=8)
        table.add_column("Round 3", style="green", width=8)
        table.add_column("Round 4", style="green", width=8)
        table.add_column("Round 5", style="green", width=8)
        
        # Sort players by average early round availability
        player_scores = []
        for player_name, round_data in availability_data.items():
            if isinstance(round_data, dict):
                early_rounds_avg = sum(round_data.get(r, 0) for r in range(1, 6)) / 5
                player_scores.append((player_name, early_rounds_avg, round_data))
        
        # Sort by highest availability in early rounds
        player_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 15 most available players
        for player_name, _, round_data in player_scores[:15]:
            # Extract position from name if formatted as "Name (POS)"
            if '(' in player_name and ')' in player_name:
                name_part = player_name.split('(')[0].strip()
                pos_part = player_name.split('(')[1].replace(')', '').strip()
            else:
                name_part = player_name
                pos_part = "?"
            
            # Format availability percentages
            round_1 = f"{round_data.get(1, 0):.1%}" if round_data.get(1, 0) > 0 else "-"
            round_2 = f"{round_data.get(2, 0):.1%}" if round_data.get(2, 0) > 0 else "-"
            round_3 = f"{round_data.get(3, 0):.1%}" if round_data.get(3, 0) > 0 else "-"
            round_4 = f"{round_data.get(4, 0):.1%}" if round_data.get(4, 0) > 0 else "-"
            round_5 = f"{round_data.get(5, 0):.1%}" if round_data.get(5, 0) > 0 else "-"
            
            table.add_row(name_part[:19], pos_part, round_1, round_2, round_3, round_4, round_5)
        
        self.console.print(table)
        return ""
    
    def _generate_overview_panel(self, results: Dict[str, Any]) -> Panel:
        """Generate simulation overview panel"""
        num_sims = results.get('num_simulations', 'Unknown')
        
        # Determine simulation type
        sim_type = "Standard Monte Carlo"
        if 'keeper_impact' in results:
            sim_type = "Keeper-Aware Simulation"
            keeper_count = results['keeper_impact'].get('total_keepers', 0) if 'keeper_impact' in results else 0
            sim_type += f" ({keeper_count} keepers)"
        elif 'variance_adjusted_vorp' in str(results):
            sim_type = "Enhanced Monte Carlo"
        
        overview_text = f"""[bold]{sim_type}[/bold]

[dim]Simulations Run:[/dim] {num_sims}
[dim]Rounds Simulated:[/dim] {results.get('rounds_completed', 'Unknown')}
[dim]Analysis:[/dim] Player availability, roster outcomes, frequency analysis"""
        
        if 'keeper_impact' in results:
            keeper_impact = results['keeper_impact']
            overview_text += f"\n[dim]Keepers Removed:[/dim] {keeper_impact.get('total_keepers', 0)} players"
            overview_text += f"\n[dim]Available Pool:[/dim] {keeper_impact.get('available_players', 'Unknown')} players"
        
        return Panel(overview_text, title="🎲 Simulation Overview", border_style="blue")
    
    def _generate_scores_panel(self, scores: Dict[str, float]) -> Panel:
        """Generate roster scores panel"""
        scores_text = f"""[bold]Roster Score Distribution[/bold]

[dim]Mean Score:[/dim] {scores.get('mean', 0):.1f}
[dim]Standard Deviation:[/dim] {scores.get('std', 0):.1f}
[dim]Best Case:[/dim] {scores.get('max', 0):.1f}
[dim]Worst Case:[/dim] {scores.get('min', 0):.1f}
[dim]25th Percentile:[/dim] {scores.get('p25', 0):.1f}
[dim]Median:[/dim] {scores.get('p50', 0):.1f}
[dim]75th Percentile:[/dim] {scores.get('p75', 0):.1f}"""
        
        return Panel(scores_text, title="📊 Score Analysis", border_style="green")
    
    def _generate_sample_rosters_panel(self, sample_rosters: List[Dict]) -> Panel:
        """Generate sample rosters panel"""
        rosters_text = "[bold]Sample Draft Outcomes[/bold]\n\n"
        
        for i, roster_data in enumerate(sample_rosters[:3], 1):
            rosters_text += f"[cyan]Example {i} (Score: {roster_data.get('score', 0):.1f})[/cyan]\n"
            roster_players = roster_data.get('roster', [])
            rosters_text += "  " + " | ".join(roster_players[:8]) + "\n\n"
        
        return Panel(rosters_text, title="🏆 Example Rosters", border_style="yellow")
    
    def _generate_frequency_panel(self, frequency_data: Dict) -> Panel:
        """Generate player frequency panel"""
        # Handle both formats: simple frequency counts or round-based data
        if isinstance(frequency_data, dict):
            first_value = next(iter(frequency_data.values()), None)
            
            if isinstance(first_value, dict):
                # Round-based availability data
                frequency_text = "[bold]Most Available Players (Early Rounds)[/bold]\n\n"
                
                # Calculate average availability across rounds 1-3
                player_avg_availability = {}
                for player, round_data in frequency_data.items():
                    early_rounds = [round_data.get(r, 0) for r in range(1, 4)]
                    avg_availability = sum(early_rounds) / len(early_rounds) if early_rounds else 0
                    player_avg_availability[player] = avg_availability
                
                # Sort by availability
                sorted_players = sorted(player_avg_availability.items(), key=lambda x: x[1], reverse=True)
                
                for player, avg_avail in sorted_players[:8]:
                    frequency_text += f"• {player}: {avg_avail:.1%} available\n"
            else:
                # Simple frequency counts
                frequency_text = "[bold]Most Frequently Drafted Players[/bold]\n\n"
                
                # Sort by frequency
                sorted_frequency = sorted(frequency_data.items(), key=lambda x: x[1], reverse=True)
                
                for player, count in sorted_frequency[:8]:
                    frequency_text += f"• {player}: {count} times\n"
        else:
            frequency_text = "[yellow]No frequency data available[/yellow]"
        
        return Panel(frequency_text, title="📈 Player Frequency", border_style="cyan")
    
    def _generate_keeper_panel(self, keeper_impact: Dict) -> Panel:
        """Generate keeper impact panel"""
        keeper_text = f"""[bold]Keeper Impact Analysis[/bold]

[dim]Total Keepers:[/dim] {keeper_impact.get('total_keepers', 0)}
[dim]Available Players:[/dim] {keeper_impact.get('available_players', 'Unknown')}
[dim]Draft Pool Reduction:[/dim] {keeper_impact.get('pool_reduction_pct', 0):.1%}"""
        
        if 'top_available' in keeper_impact:
            keeper_text += "\n\n[bold]Top Available (Post-Keepers)[/bold]\n"
            for player in keeper_impact['top_available'][:5]:
                keeper_text += f"• {player}\n"
        
        return Panel(keeper_text, title="🔒 Keeper Analysis", border_style="red")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def list_saved_results(self) -> List[str]:
        """List all saved result session IDs"""
        result_files = list(self.results_dir.glob("mc_*.json"))
        session_ids = []
        
        for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
            session_id = result_file.stem
            session_ids.append(session_id)
        
        return session_ids
