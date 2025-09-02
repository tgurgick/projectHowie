"""
Draft Strategy Manager
Stores, retrieves, and manages draft strategies with quick access menu
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .strategy_tree_search import DraftStrategy, PositionTarget
from .models import LeagueConfig


@dataclass
class StrategySession:
    """A stored strategy session with metadata"""
    strategy: DraftStrategy
    session_id: str
    name: str
    created_at: str
    last_accessed: str
    is_favorite: bool = False
    notes: str = ""


class StrategyManager:
    """Manage draft strategies and provide quick access"""
    
    def __init__(self):
        self.console = Console()
        self.strategy_dir = Path("data/strategies")
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        self.current_strategy: Optional[DraftStrategy] = None
    
    def save_strategy(self, strategy: DraftStrategy, name: str = None, notes: str = "") -> str:
        """Save a strategy and return session ID"""
        
        # Generate session ID
        session_id = f"strategy_{int(time.time())}"
        
        # Create strategy session
        session = StrategySession(
            strategy=strategy,
            session_id=session_id,
            name=name or f"Strategy {time.strftime('%m/%d %H:%M')}",
            created_at=strategy.generated_at,
            last_accessed=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes
        )
        
        # Save to file
        session_file = self.strategy_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self._serialize_session(session), f, indent=2)
        
        self.current_strategy = strategy
        
        self.console.print(f"[green]💾 Strategy saved as '{session.name}' (ID: {session_id})[/green]")
        return session_id
    
    def load_strategy(self, session_id: str) -> Optional[DraftStrategy]:
        """Load a strategy by session ID"""
        session_file = self.strategy_dir / f"{session_id}.json"
        
        if not session_file.exists():
            self.console.print(f"[red]❌ Strategy {session_id} not found[/red]")
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            session = self._deserialize_session(session_data)
            
            # Update last accessed
            session.last_accessed = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_session(session)
            
            self.current_strategy = session.strategy
            self.console.print(f"[green]✅ Loaded strategy: {session.name}[/green]")
            return session.strategy
            
        except Exception as e:
            self.console.print(f"[red]❌ Error loading strategy: {e}[/red]")
            return None
    
    def list_strategies(self) -> List[StrategySession]:
        """List all saved strategies"""
        sessions = []
        
        for strategy_file in self.strategy_dir.glob("strategy_*.json"):
            try:
                with open(strategy_file, 'r') as f:
                    session_data = json.load(f)
                sessions.append(self._deserialize_session(session_data))
            except:
                continue
        
        # Sort by last accessed (most recent first)
        sessions.sort(key=lambda s: s.last_accessed, reverse=True)
        return sessions
    
    def show_strategy_menu(self) -> str:
        """Display interactive strategy menu"""
        sessions = self.list_strategies()
        
        if not sessions:
            return "[yellow]No saved strategies found. Run a simulation first to generate strategies.[/yellow]"
        
        # Create strategy menu table
        table = Table(title="🎯 Draft Strategy Quick Menu")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="bold", width=25)
        table.add_column("League", style="dim", width=15)
        table.add_column("Created", style="dim", width=12)
        table.add_column("Summary", width=40)
        
        for i, session in enumerate(sessions[:10], 1):  # Show top 10
            strategy = session.strategy
            league_info = f"{strategy.league_config.num_teams}T, {strategy.league_config.scoring_type.upper()}"
            created = session.created_at.split()[0] if session.created_at else "Unknown"
            
            # Truncate summary
            summary = strategy.strategy_summary.split('\n')[0][:35] + "..." if len(strategy.strategy_summary) > 35 else strategy.strategy_summary
            
            table.add_row(
                str(i),
                session.name,
                league_info,
                created,
                summary
            )
        
        self.console.print(table)
        
        # Add usage instructions
        instructions = """
[bold]Usage:[/bold]
• [yellow]/draft/strategy/1[/yellow] - Load strategy #1
• [yellow]/draft/strategy/current[/yellow] - Show current strategy details
• [yellow]/draft/strategy/generate[/yellow] - Generate new strategy
        """
        
        self.console.print(Panel(instructions, title="Commands", border_style="dim"))
        return ""
    
    def show_current_strategy(self) -> str:
        """Display detailed view of current strategy"""
        if not self.current_strategy:
            return "[yellow]No current strategy loaded. Use '/draft/strategy' to see available strategies.[/yellow]"
        
        strategy = self.current_strategy
        
        # Strategy overview
        overview_panel = Panel(
            f"""[bold]{strategy.strategy_summary}[/bold]

[dim]League:[/dim] {strategy.league_config.num_teams} teams, {strategy.league_config.scoring_type.upper()} scoring
[dim]Draft Position:[/dim] #{strategy.league_config.draft_position}
[dim]Expected Value:[/dim] {strategy.expected_value:.1f}
[dim]Confidence:[/dim] {strategy.confidence_score:.1%}
[dim]Generated:[/dim] {strategy.generated_at}""",
            title="📋 Strategy Overview",
            border_style="green"
        )
        
        self.console.print(overview_panel)
        
        # Round-by-round targets
        targets_table = Table(title="🎯 Round-by-Round Targets")
        targets_table.add_column("Round", style="cyan", width=6)
        targets_table.add_column("Position", style="bold", width=8)
        targets_table.add_column("Target Player", width=20)
        targets_table.add_column("Reasoning", width=35)
        targets_table.add_column("Confidence", width=10)
        
        for target in strategy.position_targets:
            confidence_color = "green" if target.confidence > 0.7 else "yellow" if target.confidence > 0.5 else "red"
            confidence_display = f"[{confidence_color}]{target.confidence:.1%}[/{confidence_color}]"
            
            player_name = target.target_player.name if target.target_player else "TBD"
            
            targets_table.add_row(
                str(target.round_number),
                target.position,
                player_name,
                target.reasoning[:32] + "..." if len(target.reasoning) > 32 else target.reasoning,
                confidence_display
            )
        
        self.console.print(targets_table)
        
        # Key insights
        if strategy.key_insights:
            insights_text = "\n".join([f"• {insight}" for insight in strategy.key_insights])
            insights_panel = Panel(insights_text, title="💡 Key Insights", border_style="blue")
            self.console.print(insights_panel)
        
        return ""
    
    def show_round_details(self, round_number: int) -> str:
        """Show detailed information for a specific round"""
        if not self.current_strategy:
            return "[yellow]No current strategy loaded.[/yellow]"
        
        # Find target for this round
        target = None
        for t in self.current_strategy.position_targets:
            if t.round_number == round_number:
                target = t
                break
        
        if not target:
            return f"[yellow]No strategy data for round {round_number}[/yellow]"
        
        # Round details
        details = f"""[bold]Round {round_number} Strategy[/bold]

[bold]Primary Target:[/bold] {target.position}
[bold]Recommended Player:[/bold] {target.target_player.name if target.target_player else 'To be determined'}
[bold]Confidence:[/bold] {target.confidence:.1%}
[bold]Value Score:[/bold] {target.value_score:.1f}
[bold]Scarcity Urgency:[/bold] {target.scarcity_urgency:.1%}

[bold]Reasoning:[/bold]
{target.reasoning}"""

        if target.alternatives:
            details += f"\n\n[bold]Alternative Positions:[/bold] {', '.join(target.alternatives)}"
        
        # Add contingency plans
        contingencies = self.current_strategy.contingency_plans.get(round_number, [])
        if contingencies:
            details += f"\n\n[bold]Contingency Plans:[/bold]\n"
            details += "\n".join([f"• {plan}" for plan in contingencies])
        
        # Add risk tolerance
        risk_tolerance = self.current_strategy.risk_tolerance_profile.get(round_number, 0.5)
        risk_desc = "Conservative" if risk_tolerance < 0.4 else "Aggressive" if risk_tolerance > 0.6 else "Balanced"
        details += f"\n\n[bold]Risk Profile:[/bold] {risk_desc} ({risk_tolerance:.1%})"
        
        panel = Panel(details, title=f"📍 Round {round_number} Details", border_style="cyan")
        self.console.print(panel)
        
        return ""
    
    def _serialize_session(self, session: StrategySession) -> Dict[str, Any]:
        """Convert session to JSON-serializable format"""
        return {
            'session_id': session.session_id,
            'name': session.name,
            'created_at': session.created_at,
            'last_accessed': session.last_accessed,
            'is_favorite': session.is_favorite,
            'notes': session.notes,
            'strategy': {
                'league_config': asdict(session.strategy.league_config),
                'position_targets': [
                    {
                        'round_number': t.round_number,
                        'position': t.position,
                        'target_player': {
                            'name': t.target_player.name,
                            'position': t.target_player.position,
                            'projection': t.target_player.projection
                        } if t.target_player else None,
                        'confidence': t.confidence,
                        'reasoning': t.reasoning,
                        'alternatives': t.alternatives,
                        'value_score': t.value_score,
                        'scarcity_urgency': t.scarcity_urgency
                    }
                    for t in session.strategy.position_targets
                ],
                'expected_value': session.strategy.expected_value,
                'confidence_score': session.strategy.confidence_score,
                'strategy_summary': session.strategy.strategy_summary,
                'key_insights': session.strategy.key_insights,
                'risk_tolerance_profile': session.strategy.risk_tolerance_profile,
                'contingency_plans': session.strategy.contingency_plans,
                'generated_at': session.strategy.generated_at
            }
        }
    
    def _deserialize_session(self, data: Dict[str, Any]) -> StrategySession:
        """Convert JSON data back to session object"""
        from .models import Player  # Import here to avoid circular imports
        
        strategy_data = data['strategy']
        
        # Reconstruct position targets
        position_targets = []
        for t_data in strategy_data['position_targets']:
            target_player = None
            if t_data['target_player']:
                target_player = Player(
                    name=t_data['target_player']['name'],
                    position=t_data['target_player']['position'],
                    team="",
                    projection=t_data['target_player']['projection'],
                    adp=999,
                    adp_position=99,
                    bye_week=1
                )
            
            target = PositionTarget(
                round_number=t_data['round_number'],
                position=t_data['position'],
                target_player=target_player,
                confidence=t_data['confidence'],
                reasoning=t_data['reasoning'],
                alternatives=t_data['alternatives'],
                value_score=t_data['value_score'],
                scarcity_urgency=t_data['scarcity_urgency']
            )
            position_targets.append(target)
        
        # Reconstruct strategy
        strategy = DraftStrategy(
            league_config=LeagueConfig(**strategy_data['league_config']),
            position_targets=position_targets,
            expected_value=strategy_data['expected_value'],
            confidence_score=strategy_data['confidence_score'],
            strategy_summary=strategy_data['strategy_summary'],
            key_insights=strategy_data['key_insights'],
            risk_tolerance_profile=strategy_data['risk_tolerance_profile'],
            contingency_plans=strategy_data['contingency_plans'],
            generated_at=strategy_data['generated_at']
        )
        
        return StrategySession(
            strategy=strategy,
            session_id=data['session_id'],
            name=data['name'],
            created_at=data['created_at'],
            last_accessed=data['last_accessed'],
            is_favorite=data.get('is_favorite', False),
            notes=data.get('notes', "")
        )
    
    def _save_session(self, session: StrategySession) -> None:
        """Save session to file"""
        session_file = self.strategy_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self._serialize_session(session), f, indent=2)
    
    def delete_strategy(self, session_id: str) -> str:
        """Delete a strategy"""
        session_file = self.strategy_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
            return f"[green]✅ Strategy {session_id} deleted[/green]"
        else:
            return f"[red]❌ Strategy {session_id} not found[/red]"
