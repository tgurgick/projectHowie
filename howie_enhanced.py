#!/usr/bin/env python3
"""
Enhanced Howie CLI with Multi-Model Support
Allows using different AI models for different tasks
"""

import os
import sys
import asyncio
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from howie_cli.core.enhanced_agent import EnhancedHowieAgent
from howie_cli.core.model_manager import ModelManager
from howie_cli.core.context import ConversationContext

# Eagles green color theme
console = Console(style="green")

# Version
__version__ = "2.3.0"


def detect_accidental_rapid_stats(user_input: str) -> bool:
    """
    Detect if user accidentally typed a rapid stats command without the leading slash.
    This prevents expensive API calls for common typos like 'wr/td' instead of '/wr/td'.
    
    Returns True if an accidental pattern is detected and handled.
    """
    if not user_input or user_input.startswith('/'):
        return False
    
    # Define patterns that look like rapid stats commands
    # position/stat[/season] format
    parts = user_input.strip().split('/')
    
    if len(parts) < 2 or len(parts) > 3:
        return False
    
    # Check if first part looks like a position
    valid_positions = ['qb', 'rb', 'wr', 'te', 'k', 'def', 'dst']
    if parts[0].lower() not in valid_positions:
        return False
    
    # Check if second part looks like a stat
    valid_stats = [
        'adp', 'td', 'yards', 'targets', 'rec', 'rush_td', 'rec_td', 'pass_td',
        'fantasy', 'points', 'sacks', 'int', 'projections', 'sos', 'bye',
        'pass_yards', 'rec_yards', 'rush_yards', 'pass_att', 'rush_att',
        'pass_comp', 'receptions', 'fumbles'
    ]
    if parts[1].lower() not in valid_stats:
        return False
    
    # If we have a third part, validate based on the stat type
    if len(parts) == 3:
        if parts[1].lower() == 'bye':
            # For bye weeks, third part should be week number (1-18)
            try:
                week = int(parts[2])
                if week < 1 or week > 18:
                    return False
            except ValueError:
                return False
        else:
            # For other stats, third part should be a year
            try:
                year = int(parts[2])
                if year < 2018 or year > 2030:
                    return False
            except ValueError:
                return False
    
    # Pattern detected! Show helpful message and suggest the correct command
    suggested_command = f"/{user_input}"
    
    console.print(f"[yellow]⚠️  Did you mean [bright_green]{suggested_command}[/bright_green]?[/yellow]")
    console.print(f"[dim]Rapid stats commands need a leading slash to work properly.[/dim]")
    console.print(f"[dim]Type [bright_green]{suggested_command}[/bright_green] or [bright_green]?[/bright_green] for help.[/dim]")
    
    return True





def show_eagles_intro():
    """Display intro screen"""
    console.print("\n" * 2)
    
    # Welcome panel
    welcome_text = (
        "Welcome to HOWIE, your fantasy football assistant!\n\n"
        "Powered by Claude Sonnet 4 • Real-time data • Expert analysis"
    )
    
    console.print(Panel.fit(
        welcome_text,
        title="HOWIE - Fantasy Football AI",
        border_style="bright_green"
    ))
    console.print("\n")


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, version):
    """HOWIE - Fantasy Football AI Assistant
    
    Powered by Claude Sonnet 4 with multi-model support
    
    Use different AI models optimized for different tasks:
    - Claude Sonnet 4 for analysis and strategy
    - Perplexity for real-time research
    - GPT-4o for complex reasoning
    - Fast models for quick lookups
    """
    if version:
        console.print(f"[bright_green]Howie Enhanced CLI v{__version__}[/bright_green]")
        return
    
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@cli.command()
@click.option('--model', help='Default model to use')
@click.option('--config', type=click.Path(), help='Path to model configuration file')
@click.option('--resume', is_flag=True, help='Resume previous session')
@click.option('--no-intro', is_flag=True, help='Skip intro screen')
def chat(model, config, resume, no_intro):
    """Start interactive chat with multi-model support"""
    try:
        # Show intro screen
        if not no_intro:
            show_eagles_intro()
        
        # Initialize enhanced agent
        config_path = Path(config) if config else None
        agent = EnhancedHowieAgent(model=model, model_config_path=config_path)
        
        # Show current model configuration
        console.print(Panel.fit(
            f"[bold bright_green]Howie Enhanced - AI Assistant[/bold bright_green]\n"
            f"Current Model: [bright_green]{agent.model_manager.current_model}[/bright_green]\n"
            f"Type '/' for commands • '?' for help • 'end' to exit",
            border_style="bright_green"
        ))
        
        # Resume session if requested
        if resume:
            try:
                sessions_dir = Path.home() / ".howie" / "sessions"
                if sessions_dir.exists():
                    sessions = list(sessions_dir.glob("*.pkl"))
                    if sessions:
                        latest = max(sessions, key=lambda p: p.stat().st_mtime)
                        session_id = latest.stem
                        agent.context = ConversationContext.load_session(session_id)
                        console.print(f"[bright_green]Resumed session: {session_id}[/bright_green]")
            except Exception as e:
                console.print(f"[yellow]Could not resume session: {e}[/yellow]")
        
        # Start enhanced chat loop
        asyncio.run(enhanced_chat_loop(agent))
        
    except KeyboardInterrupt:
        console.print("\n[bright_green]Goodbye![/bright_green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


async def enhanced_chat_loop(agent: EnhancedHowieAgent):
    """Enhanced chat loop with model commands"""
    
    while True:
        try:
            # Get user input with Claude-like prompt
            user_input = console.input("\n[bold white]>[/bold white] ")
            
            # Early detection for accidental rapid stats patterns (missing leading slash)
            if detect_accidental_rapid_stats(user_input):
                continue
            
            # Check for special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'end', 'e']:
                console.print("[bright_green]Good luck with your fantasy team![/bright_green]")
                break
            
            elif user_input.lower() in ['help', '?']:
                show_enhanced_help()
                console.print("\n[dim]? for help • / for commands • /quit to exit[/dim]")
                continue
            
            # Slash commands
            elif user_input.startswith('/'):
                if user_input == '/':
                    # Show available slash commands
                    console.print("\n[bold bright_green]Available Commands:[/bold bright_green]")
                    console.print("  [bright_green]/model[/bright_green] - Model management (info, switch, config, save)")
                    console.print("  [bright_green]/agent[/bright_green] - Agent management (spawn, list, stop)")
                    console.print("  [bright_green]/cost[/bright_green] - Cost tracking and limits")
                    console.print("  [bright_green]/logs[/bright_green] - Show recent system events")
                    console.print("  [bright_green]/update[/bright_green] - Update ADP data from FantasyPros")
                    console.print("  [bright_green]/help[/bright_green] - Show detailed help")
                    console.print("  [bright_green]/quit[/bright_green] - Exit the application")
                    console.print("\n[bold bright_green]Rapid Stats Commands:[/bold bright_green]")
                    console.print("  [bright_green]/wr/adp[/bright_green] - Top 50 WRs by ADP")
                    console.print("  [bright_green]/qb/td[/bright_green] - Top 50 QBs by total TDs")
                    console.print("  [bright_green]/rb/yards[/bright_green] - Top 50 RBs by total yards")
                    console.print("  [bright_green]/te/rec[/bright_green] - Top 50 TEs by receptions")
                    console.print("  [bright_green]/k/points[/bright_green] - Top 50 Ks by fantasy points")
                    console.print("  [bright_green]/def/sacks[/bright_green] - Top 50 DEFs by sacks")
                    console.print("  [bright_green]/qb/stats[/bright_green] - Show all QB stats available")
                    console.print("  [dim]More: /qb/pass_td, /rb/rush_yards, /wr/targets, /qb/fantasy[/dim]")
                    console.print("  [dim]New: /wr/projections, /qb/sos, /def/sos/playoffs[/dim]")
                    console.print("  [dim]Format: /position/stat/season (e.g., /qb/td/2024)[/dim]")
                    console.print("\n[bright_green]ADP & Draft Tools:[/bright_green]")
                    console.print("  [dim]/adp - Show ADP rankings with 10-team and 12-team round estimates[/dim]")
                    console.print("  [dim]/adp/10 - Show ADP rankings with projected rounds for 10-team league[/dim]")
                    console.print("  [dim]/adp/12 - Show ADP rankings with projected rounds for 12-team league[/dim]")
                    console.print("  [bright_green]/tiers[/bright_green] - Show positional tier analysis and marginal value drops")
                    console.print("  [dim]/tiers/te - Detailed tier breakdown for specific position[/dim]")
                    console.print("  [bright_green]/intel[/bright_green] - Team position intelligence system")
                    console.print("  [dim]/intel/PHI/wr - Get Eagles WR intelligence report[/dim]")
                    console.print("\n[dim]? for help • / for commands • /end to exit[/dim]")
                    continue
                elif user_input.lower().startswith('/model'):
                    handle_model_command(agent, user_input[7:])  # Remove '/model' prefix
                    continue
                elif user_input.lower().startswith('/help'):
                    show_enhanced_help()
                    continue
                elif user_input.lower().startswith('/cost'):
                    handle_cost_command(agent, user_input[6:])  # Remove '/cost' prefix
                    continue
                elif user_input.lower().startswith('/agent'):
                    handle_agent_command(agent, user_input[7:])  # Remove '/agent' prefix
                    continue
                elif user_input.lower().startswith('/logs'):
                    handle_logs_command(agent, user_input[6:])  # Remove '/logs' prefix
                    continue
                elif user_input.lower().startswith('/update'):
                    await handle_update_command(agent, user_input[8:])  # Remove '/update' prefix
                    continue
                elif user_input.lower().startswith('/adp'):
                    handle_adp_command(user_input[1:])  # Remove '/' prefix
                    continue
                elif user_input.lower().startswith('/tiers'):
                    handle_tiers_command(user_input[1:])  # Remove '/' prefix
                    continue
                elif user_input.lower().startswith('/intel'):
                    handle_intel_command(user_input[1:])  # Remove '/' prefix
                    continue
                elif (user_input.lower().startswith('/wr/') or user_input.lower().startswith('/qb/') or 
                      user_input.lower().startswith('/rb/') or user_input.lower().startswith('/te/') or 
                      user_input.lower().startswith('/k/') or user_input.lower().startswith('/def/') or 
                      user_input.lower().startswith('/dst/')):
                    handle_rapid_stats_command(user_input[1:])  # Remove '/' prefix
                    continue
                elif user_input.lower().startswith('/quit') or user_input.lower().startswith('/end') or user_input.lower() == '/e':
                    console.print("[bright_green]Good luck with your fantasy team![/bright_green]")
                    break
                else:
                    console.print("[yellow]Unknown slash command. Type '/' to see available commands.[/yellow]")
                continue
            
            # Model override syntax: @model <query>
            elif user_input.startswith('@'):
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    model_name = parts[0][1:]  # Remove @
                    query = parts[1]
                    
                    if model_name in agent.model_manager.models:
                        console.print(f"[bright_green]Using {model_name} for this query[/bright_green]")
                        response = await agent.process_with_model(query, model_name)
                    else:
                        console.print(f"[red]Unknown model: {model_name}[/red]")
                        continue
                else:
                    console.print("[red]Usage: @model_name your question[/red]")
                    continue
            else:
                # Normal query - use automatic model selection
                console.print("\n[dim]Processing query...[/dim]")
                
                # Show model selection
                recommended_model = agent.model_manager.recommend_model(user_input)
                console.print(f"[dim]Selected model: {recommended_model}[/dim]")
                
                # Process the message
                response = await agent.process_message(user_input)
                
                console.print("[dim]Query processed[/dim]")
            
            # Display response
            console.print("\n[bold bright_green]Howie:[/bold bright_green]")
            from rich.markdown import Markdown
            console.print(Markdown(response))
            
            # Show subtle menu below response
            console.print("\n[dim]? for help • / for commands • /end to exit[/dim]")
            
        except KeyboardInterrupt:
            console.print("[yellow]Use 'quit' to exit properly[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def handle_update_command(agent: EnhancedHowieAgent, command: str):
    """Handle update-related commands"""
    parts = command.split()
    
    if not parts or parts[0] == 'adp':
        # Update ADP data
        console.print("[bright_green]Updating ADP data from FantasyPros...[/bright_green]")
        
        try:
            import sys
            from pathlib import Path
            
            # Add scripts directory to path
            scripts_dir = Path(__file__).parent / "scripts"
            sys.path.insert(0, str(scripts_dir))
            
            from build_fantasypros_adp import build_fantasypros_adp, Args
            
            # Create args object for 2025 season (current)
            args = Args(
                season=2025,
                scoring='ppr',
                test=False,
                db_url="sqlite:///data/fantasy_ppr.db"
            )
            
            # Run the ADP update
            build_fantasypros_adp(args)
            
            console.print("[bright_green]ADP data updated successfully![/bright_green]")
            console.print("[dim]You can now query for current ADP data using commands like 'top WR by ADP'[/dim]")
            
        except ImportError:
            console.print("[red]ADP update script not found. Please check scripts/build_fantasypros_adp.py[/red]")
        except Exception as e:
            console.print(f"[red]Error updating ADP data: {e}[/red]")
    
    elif parts[0] == 'rosters':
        # Update rosters
        console.print("[bright_green]Updating NFL rosters...[/bright_green]")
        
        try:
            import asyncio
            import sys
            from pathlib import Path
            
            # Add scripts directory to path
            scripts_dir = Path(__file__).parent / "scripts"
            sys.path.insert(0, str(scripts_dir))
            
            from update_rosters import main as update_main
            team_changes = asyncio.run(update_main())
            
            console.print("[bright_green]Roster update completed successfully![/bright_green]")
            
            # Display team changes if any
            if team_changes:
                console.print("\n[bold bright_green]Team Changes Detected:[/bold bright_green]")
                for change in team_changes:
                    console.print(f"  • [bold white]{change['player_name']}[/bold white] ([green]{change['position']}[/green]) [red]{change['old_team']}[/red] → [bright_green]{change['new_team']}[/bright_green]")
            else:
                console.print("\n[bright_green]No team changes detected[/bright_green]")
                
        except ImportError:
            console.print("[red]Roster update script not found. Please check scripts/update_rosters.py[/red]")
        except Exception as e:
            console.print(f"[red]Error updating rosters: {e}[/red]")
    
    elif parts[0] == 'intelligence':
        # Update team position intelligence
        console.print("[bright_green]Starting Team Position Intelligence Update...[/bright_green]")
        console.print("[dim]This process will analyze each team's positional groups using Claude + web search + fact-checking[/dim]")
        
        try:
            # Since this function is now async, we can directly await
            await run_team_intelligence_workflow(agent)
            
        except Exception as e:
            console.print(f"[red]Error updating team intelligence: {e}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
    
    else:
        console.print("[yellow]Available update commands:[/yellow]")
        console.print("  [bright_green]/update adp[/bright_green] - Update ADP data from FantasyPros")
        console.print("  [bright_green]/update rosters[/bright_green] - Update NFL roster information")
        console.print("  [bright_green]/update intelligence[/bright_green] - Update team position intelligence (AI-powered analysis)")


def handle_model_command(agent: EnhancedHowieAgent, command: str):
    """Handle model-specific commands"""
    parts = command.split()
    
    if not parts or parts[0] == 'info':
        # Show model information
        info = agent.get_model_info()
        
        # Current model
        console.print(f"\n[bold bright_green]Current Model:[/bold bright_green] {info['current_model']}")
        
        # Available models table
        table = Table(title="Available Models", show_header=True, header_style="bold bright_green")
        table.add_column("Model", style="bright_green", width=20)
        table.add_column("Provider", style="bright_green", width=12)
        table.add_column("Tier", style="bright_green", width=10)
        table.add_column("Cost (I/O per 1K)", style="red", width=15)
        table.add_column("Best For", style="bright_green", width=30)
        
        for name, config in info['available_models'].items():
            table.add_row(
                name,
                config['provider'],
                config['tier'],
                config['cost_per_1k'],
                ", ".join(config['best_for'][:2]) if config['best_for'] else ""
            )
        
        console.print(table)
        
        # Task mappings
        console.print("\n[bold bright_green]Task Mappings:[/bold bright_green]")
        for task, model in info['task_mappings'].items():
            console.print(f"  [bright_green]{task}[/bright_green]: [bright_green]{model}[/bright_green]")
        
        # Usage stats
        if info['usage']['by_model']:
            console.print("\n[bold bright_green]Usage Statistics:[/bold bright_green]")
            console.print(f"Total Cost: [bright_green]${info['usage']['total_cost']:.4f}[/bright_green]")
            for model, stats in info['usage']['by_model'].items():
                console.print(f"  {model}: {stats['calls']} calls, [bright_green]${stats.get('cost', 0):.4f}[/bright_green]")
        
        # Show subtle menu
        console.print("\n[dim]? for help • / for commands • /quit to exit[/dim]")
    
    elif parts[0] == 'switch' and len(parts) > 1:
        # Switch model
        model_name = parts[1]
        try:
            agent.switch_model(model_name)
            console.print(f"[bright_green]Switched to {model_name}[/bright_green]")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif parts[0] == 'config' and len(parts) > 2:
        # Configure task mapping
        task_type = parts[1]
        model_name = parts[2]
        try:
            agent.configure_task_models({task_type: model_name})
            console.print(f"[bright_green]Configured {task_type} to use {model_name}[/bright_green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif parts[0] == 'save':
        # Save configuration
        try:
            agent.model_manager.save_config()
            console.print("[bright_green]Model configuration saved[/bright_green]")
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    else:
        console.print("[yellow]Unknown model command. Try 'model/info' or 'model/switch <name>'[/yellow]")


def handle_cost_command(agent: EnhancedHowieAgent, command: str):
    """Handle cost-related commands"""
    parts = command.split()
    
    if not parts or parts[0] == 'info':
        # Show cost information
        usage = agent.model_manager.get_usage_report()
        
        console.print(f"\n[bold bright_green]Cost Information:[/bold bright_green]")
        console.print(f"Total Cost: [red]${usage['total_cost']:.4f}[/red]")
        
        if usage['by_model']:
            console.print(f"\n[bold bright_green]Cost by Model:[/bold bright_green]")
            for model, stats in usage['by_model'].items():
                cost = stats.get('cost', 0)
                calls = stats.get('calls', 0)
                console.print(f"  [bright_green]{model}[/bright_green]: {calls} calls, [red]${cost:.4f}[/red]")
        
        # Show cost limits if configured
        console.print(f"\n[bold bright_green]Cost Limits:[/bold bright_green]")
        console.print(f"Daily Limit: [red]$10.00[/red] (default)")
        console.print(f"Per Query Limit: [red]$0.50[/red] (default)")
        console.print(f"Warning Threshold: [red]$5.00[/red] (default)")
        
        # Show subtle menu
        console.print("\n[dim]? for help • / for commands • /quit to exit[/dim]")
    
    elif parts[0] == 'reset':
        # Reset cost tracking
        agent.model_manager.total_cost = 0.0
        agent.model_manager.usage_stats = {}
        console.print("[bright_green]Cost tracking reset[/bright_green]")
    
    elif parts[0] == 'estimate' and len(parts) > 2:
        # Estimate cost for a specific model and token count
        try:
            model = parts[1]
            input_tokens = int(parts[2])
            output_tokens = int(parts[3]) if len(parts) > 3 else 1000
            
            cost = agent.model_manager.estimate_cost(model, input_tokens, output_tokens)
            console.print(f"[red]Estimated cost for {model}: ${cost:.4f}[/red]")
        except (ValueError, IndexError):
            console.print("[red]Usage: /cost estimate <model> <input_tokens> [output_tokens][/red]")
    
    else:
        console.print("[yellow]Unknown cost command. Try '/cost info', '/cost reset', or '/cost estimate <model> <tokens>'[/yellow]")


def handle_agent_command(agent: EnhancedHowieAgent, command: str):
    """Handle agent-related commands"""
    parts = command.split()
    
    if not parts or parts[0] == 'info':
        # Show agent information
        console.print(f"\n[bold bright_green]Agent Information:[/bold bright_green]")
        console.print(f"Current Agent: [bright_green]Enhanced Howie Agent[/bright_green]")
        console.print(f"Model Manager: [bright_green]Active[/bright_green]")
        console.print(f"Available Models: [bright_green]{len(agent.model_manager.models)}[/bright_green]")
        console.print(f"Context Memory: [bright_green]Enabled[/bright_green]")
        
        # Show available agent types
        console.print(f"\n[bold bright_green]Available Agent Types:[/bold bright_green]")
        console.print("  [bright_green]research[/bright_green] - Research and data gathering")
        console.print("  [bright_green]analysis[/bright_green] - Deep analysis and insights")
        console.print("  [bright_green]code[/bright_green] - Code generation and scripts")
        console.print("  [bright_green]optimization[/bright_green] - Lineup and strategy optimization")
        
        # Show subtle menu
        console.print("\n[dim]? for help • / for commands • /quit to exit[/dim]")
    
    elif parts[0] == 'spawn' and len(parts) > 1:
        # Spawn a new agent (placeholder for future implementation)
        agent_type = parts[1]
        console.print(f"[bright_green]Spawning {agent_type} agent...[/bright_green]")
        console.print("[yellow]Agent spawning feature coming soon![/yellow]")
    
    elif parts[0] == 'list':
        # List active agents (placeholder)
        console.print(f"\n[bold bright_green]Active Agents:[/bold bright_green]")
        console.print("  [bright_green]Main Agent[/bright_green] - Enhanced Howie Agent (active)")
        console.print("[yellow]Multi-agent support coming soon![/yellow]")
    
    else:
        console.print("[yellow]Unknown agent command. Try '/agent info', '/agent spawn <type>', or '/agent list'[/yellow]")


def handle_logs_command(agent: EnhancedHowieAgent, command: str):
    """Handle logs-related commands with enhanced agent and tool tracing"""
    parts = command.split()
    
    if not parts or parts[0] == 'info':
        # Show recent logs
        logs = agent.get_recent_logs(25)
        
        if not logs:
            console.print("[dim]No recent events logged[/dim]")
            return
        
        console.print(f"\n[bold bright_green]System Event Log (Last 25 Events):[/bold bright_green]")
        console.print("[dim]Format: [TIMESTAMP] EVENT_TYPE: Detailed Description[/dim]\n")
        
        for log in logs:
            timestamp = log['timestamp']
            event_type = log['type'].upper().replace('_', ' ')
            description = log['description']
            
            # Enhanced descriptions with more detail
            if log['type'] == 'user_input':
                detail = f"User query received: {description}"
                color = "bright_green"
            elif log['type'] == 'api_call':
                detail = f"API call made: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_execution':
                detail = f"Tool execution: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_result':
                if "success" in description.lower():
                    detail = f"Tool completed successfully: {description}"
                    color = "bright_green"
                else:
                    detail = f"Tool failed: {description}"
                    color = "red"
            elif log['type'] == 'ai_response':
                detail = f"AI response generated: {description}"
                color = "bright_green"
            elif log['type'] == 'task_classification':
                detail = f"Task classified: {description}"
                color = "bright_green"
            elif log['type'] == 'model_selection':
                detail = f"Model selected: {description}"
                color = "bright_green"
            elif log['type'] == 'search_planning':
                detail = f"Search planning: {description}"
                color = "bright_green"
            elif log['type'] == 'search_execution':
                detail = f"Search execution: {description}"
                color = "bright_green"
            elif log['type'] == 'search_result':
                if "success" in description.lower():
                    detail = f"Search completed: {description}"
                    color = "bright_green"
                else:
                    detail = f"Search failed: {description}"
                    color = "red"
            elif log['type'] == 'search_verification':
                detail = f"Search verification: {description}"
                color = "bright_green"
            elif log['type'] == 'search_reasoning':
                detail = f"Search reasoning: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_planning_prompt':
                detail = f"Planning prompt: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_planning_response':
                detail = f"Planning response: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_planning_parsed':
                detail = f"Planning parsed: {description}"
                color = "bright_green"
            else:
                detail = f"System event: {description}"
                color = "bright_green"
            
            console.print(f"[dim]{timestamp}[/dim] [{color}]{event_type}[/{color}]: {detail}")
        
        # Show subtle menu
        console.print("\n[dim]Use '/logs detailed' for full prompts/reasoning, '/logs tools' for tool trace, '/logs errors' for error analysis[/dim]")
        console.print("[dim]? for help • / for commands • /quit to exit[/dim]")
    
    elif parts[0] == 'detailed':
        # Show detailed logs with full prompts and reasoning
        logs = agent.get_recent_logs(25)
        
        if not logs:
            console.print("[dim]No recent events logged[/dim]")
            return
        
        console.print(f"\n[bold bright_green]Complete Detailed Event Log - All Events with Full Context:[/bold bright_green]\n")
        
        for i, log in enumerate(logs):
            timestamp = log['timestamp']
            event_type = log['type'].upper().replace('_', ' ')
            description = log['description']
            details = log.get('details', {})
            
            # Show EVERY event with detailed context, not just search events
            console.print(f"\n[bold bright_green]Event {i+1}: {event_type}[/bold bright_green]")
            console.print(f"[dim]Time: {timestamp}[/dim]")
            console.print(f"[bright_green]Description: {description}[/bright_green]")
            
            # Always show details section if any details exist
            if details:
                from rich.panel import Panel
                from rich.json import JSON
                
                # Handle ALL event types with comprehensive detail display
                if log['type'] == 'user_input':
                    console.print(f"[dim]User Query: {details.get('user_query', 'N/A')}[/dim]")
                    if 'query_length' in details:
                        console.print(f"[dim]Query length: {details['query_length']} characters[/dim]")
                
                elif log['type'] == 'task_classification':
                    console.print(f"[dim]Task Type: {details.get('task_type', 'N/A')}[/dim]")
                    if 'reasoning' in details:
                        console.print(Panel(details['reasoning'], title="[dim]Classification Reasoning[/dim]", border_style="dim"))
                    if 'confidence' in details:
                        console.print(f"[dim]Classification confidence: {details['confidence']}[/dim]")
                
                elif log['type'] == 'model_selection':
                    console.print(f"[dim]Selected Model: {details.get('model_name', 'N/A')}[/dim]")
                    console.print(f"[dim]Provider: {details.get('provider', 'N/A')}[/dim]")
                    if 'task_type' in details:
                        console.print(f"[dim]Task type: {details['task_type']}[/dim]")
                    if 'reasoning' in details:
                        console.print(Panel(details['reasoning'], title="[dim]Model Selection Reasoning[/dim]", border_style="dim"))
                
                elif log['type'] == 'tool_planning_prompt':
                    console.print(f"[dim]User Input: {details.get('user_input', 'N/A')}[/dim]")
                    if 'full_prompt' in details:
                        console.print(Panel(
                            details['full_prompt'],
                            title="[dim]Complete Planning Prompt[/dim]",
                            border_style="dim"
                        ))
                    if 'system_message' in details:
                        console.print(Panel(
                            details['system_message'],
                            title="[dim]System Message[/dim]",
                            border_style="dim"
                        ))
                    if 'available_tools' in details:
                        console.print(f"[dim]Available tools ({len(details['available_tools'])}): {', '.join(details['available_tools'])}[/dim]")
                    console.print(f"[dim]Prompt length: {details.get('prompt_length', 'N/A')} characters[/dim]")
                
                elif log['type'] == 'tool_planning_response':
                    console.print(f"[dim]User Input: {details.get('user_input', 'N/A')}[/dim]")
                    console.print(f"[dim]Model: {details.get('model_used', 'N/A')}, Temperature: {details.get('temperature', 'N/A')}[/dim]")
                    console.print(f"[dim]Response length: {details.get('response_length', 'N/A')} characters[/dim]")
                    if 'model_response' in details:
                        console.print(Panel(
                            details['model_response'],
                            title="[dim]Complete Model Response[/dim]",
                            border_style="dim"
                        ))
                
                elif log['type'] == 'tool_planning_parsed':
                    console.print(f"[dim]User Input: {details.get('user_input', 'N/A')}[/dim]")
                    if 'parsed_plan' in details:
                        console.print(Panel(
                            str(details['parsed_plan']),
                            title="[dim]Parsed Plan (JSON)[/dim]",
                            border_style="dim"
                        ))
                    if 'plan_analysis' in details:
                        analysis = details['plan_analysis']
                        console.print(f"[dim]Tools planned: {analysis.get('tools_planned', 'N/A')}[/dim]")
                        console.print(f"[dim]Tool names: {', '.join(analysis.get('tool_names', []))}[/dim]")
                        console.print(f"[dim]Has parameters: {analysis.get('has_parameters', 'N/A')}[/dim]")
                        console.print(f"[dim]Plan complexity: {analysis.get('plan_complexity', 'N/A')}[/dim]")
                    if 'validation' in details:
                        val = details['validation']
                        console.print(f"[dim]Valid list: {val.get('is_valid_list', 'N/A')}, All tools valid: {val.get('all_tools_valid', 'N/A')}[/dim]")
                
                elif log['type'] == 'search_planning':
                    console.print(f"[dim]User Query: {details.get('user_query', 'N/A')}[/dim]")
                    if 'analysis' in details:
                        analysis = details['analysis']
                        console.print(f"[dim]Requires current data: {analysis.get('requires_current_data', 'N/A')}[/dim]")
                        console.print(f"[dim]Context type: {analysis.get('context_type', 'N/A')}[/dim]")
                        console.print(f"[dim]Matched indicators: {', '.join(analysis.get('matched_indicators', []))}[/dim]")
                    if 'search_plan' in details:
                        plan = details['search_plan']
                        console.print(f"[dim]Primary searches: {len(plan.get('primary_searches', []))}[/dim]")
                        console.print(f"[dim]Verification searches: {len(plan.get('verification_searches', []))}[/dim]")
                        console.print(f"[dim]Fallback searches: {len(plan.get('fallback_searches', []))}[/dim]")
                        console.print(Panel(
                            str(plan),
                            title="[dim]Complete Search Plan[/dim]",
                            border_style="dim"
                        ))
                    if 'reasoning' in details:
                        console.print(Panel(
                            details['reasoning'],
                            title="[dim]Search Planning Reasoning[/dim]",
                            border_style="dim"
                        ))
                
                elif log['type'] == 'search_execution':
                    console.print(f"[dim]Search {details.get('search_index', 'N/A')}/{details.get('total_searches', 'N/A')}[/dim]")
                    console.print(f"[dim]Tool: {details.get('tool_name', 'N/A')} ({details.get('search_type', 'N/A')})[/dim]")
                    if 'params' in details:
                        console.print(Panel(
                            str(details['params']),
                            title="[dim]Tool Parameters[/dim]",
                            border_style="dim"
                        ))
                
                elif log['type'] == 'search_result':
                    console.print(f"[dim]Search {details.get('search_index', 'N/A')} - Tool: {details.get('tool_name', 'N/A')}[/dim]")
                    console.print(f"[dim]Status: {details.get('status', 'N/A')}, Has data: {details.get('has_data', 'N/A')}[/dim]")
                    if details.get('data_preview'):
                        console.print(Panel(
                            details['data_preview'],
                            title="[dim]Data Preview (first 200 chars)[/dim]",
                            border_style="dim"
                        ))
                    if details.get('error'):
                        console.print(f"[red]Error: {details['error']}[/red]")
                
                elif log['type'] == 'search_verification':
                    if 'data_assessment' in details:
                        assessment = details['data_assessment']
                        console.print(f"[dim]Has current data: {assessment.get('has_current_data', 'N/A')}[/dim]")
                        console.print(f"[dim]Has database data: {assessment.get('has_database_data', 'N/A')}[/dim]")
                        console.print(f"[dim]Data sources: {len(assessment.get('data_sources', []))}[/dim]")
                        if assessment.get('data_sources'):
                            console.print(Panel(
                                str(assessment['data_sources']),
                                title="[dim]Data Sources Detail[/dim]",
                                border_style="dim"
                            ))
                    if 'quality_metrics' in details:
                        metrics = details['quality_metrics']
                        console.print(f"[dim]Total searches: {metrics.get('total_search_results', 'N/A')}[/dim]")
                        console.print(f"[dim]Successful: {metrics.get('successful_results', 'N/A')}, Failed: {metrics.get('failed_results', 'N/A')}[/dim]")
                    if 'final_recommendations' in details:
                        recs = details['final_recommendations']
                        console.print(f"[dim]Data completeness: {recs.get('data_completeness', 'N/A')}[/dim]")
                        console.print(f"[dim]Confidence level: {recs.get('confidence_level', 'N/A')}[/dim]")
                
                elif log['type'] == 'search_reasoning':
                    console.print(f"[dim]User Query: {details.get('user_query', 'N/A')}[/dim]")
                    console.print(f"[dim]Selected tools: {', '.join(details.get('selected_tools', []))}[/dim]")
                    console.print(f"[dim]Data quality score: {details.get('data_quality_score', 'N/A')}[/dim]")
                    if 'reasoning' in details:
                        console.print(Panel(
                            details['reasoning'],
                            title="[dim]Complete Search Selection Reasoning[/dim]",
                            border_style="dim"
                        ))
                
                elif log['type'] == 'tool_execution':
                    console.print(f"[dim]Tool: {details.get('tool_name', 'N/A')}[/dim]")
                    if 'params' in details:
                        console.print(Panel(
                            str(details['params']),
                            title="[dim]Tool Parameters[/dim]",
                            border_style="dim"
                        ))
                    if 'execution_time' in details:
                        console.print(f"[dim]Execution time: {details['execution_time']}s[/dim]")
                
                elif log['type'] == 'tool_result':
                    console.print(f"[dim]Tool: {details.get('tool_name', 'N/A')}[/dim]")
                    console.print(f"[dim]Status: {details.get('status', 'N/A')}[/dim]")
                    if details.get('result_preview'):
                        console.print(Panel(
                            details['result_preview'],
                            title="[dim]Result Preview[/dim]",
                            border_style="dim"
                        ))
                    if details.get('error'):
                        console.print(f"[red]Error: {details['error']}[/red]")
                
                elif log['type'] == 'api_call':
                    console.print(f"[dim]Model: {details.get('model', 'N/A')}[/dim]")
                    console.print(f"[dim]Provider: {details.get('provider', 'N/A')}[/dim]")
                    if 'input_tokens' in details:
                        console.print(f"[dim]Input tokens: {details['input_tokens']}, Output tokens: {details.get('output_tokens', 'N/A')}[/dim]")
                    if 'cost' in details:
                        console.print(f"[dim]Cost: ${details['cost']:.4f}[/dim]")
                    if 'messages' in details:
                        console.print(Panel(
                            str(details['messages'])[:500] + "..." if len(str(details['messages'])) > 500 else str(details['messages']),
                            title="[dim]API Messages[/dim]",
                            border_style="dim"
                        ))
                
                elif log['type'] == 'ai_response':
                    console.print(f"[dim]Model: {details.get('model', 'N/A')}[/dim]")
                    console.print(f"[dim]Response length: {details.get('response_length', 'N/A')} characters[/dim]")
                    if 'response_preview' in details:
                        console.print(Panel(
                            details['response_preview'],
                            title="[dim]Response Preview[/dim]",
                            border_style="dim"
                        ))
                
                else:
                    # For any other event type, show all details as JSON
                    console.print(Panel(
                        JSON.from_data(details),
                        title="[dim]Complete Event Details[/dim]",
                        border_style="dim"
                    ))
            
            else:
                console.print("[dim]No additional details available[/dim]")
            
            console.print("[dim]" + "─" * 80 + "[/dim]")
        
        console.print(f"\n[dim]Showing detailed view of {len(logs)} events[/dim]")
    
    elif parts[0] == 'tools':
        # Show detailed tool trace
        tool_logs = agent.get_tool_trace()
        
        if not tool_logs:
            console.print("[dim]No tool execution logs found[/dim]")
            return
        
        console.print(f"\n[bold bright_green]Tool Execution Detailed Trace (Last 15 Events):[/bold bright_green]")
        console.print("[dim]Format: [TIMESTAMP] EVENT_TYPE: Tool Name - Status/Result Details[/dim]\n")
        
        for log in tool_logs[-15:]:  # Show last 15 tool events
            timestamp = log['timestamp']
            event_type = log['type'].upper().replace('_', ' ')
            description = log['description']
            
            if log['type'] == 'tool_execution':
                detail = f"Tool initiated: {description}"
                color = "bright_green"
            elif log['type'] == 'tool_result':
                if "success" in description.lower():
                    detail = f"Tool completed successfully: {description}"
                    color = "bright_green"
                elif "error" in description.lower() or "fail" in description.lower():
                    detail = f"Tool execution failed: {description}"
                    color = "red"
                else:
                    detail = f"Tool result: {description}"
                    color = "bright_green"
            else:
                detail = f"Tool event: {description}"
                color = "bright_green"
            
            console.print(f"[dim]{timestamp}[/dim] [{color}]{event_type}[/{color}]: {detail}")
        
        console.print("\n[dim]Use '/logs info' for general events, '/logs errors' for error analysis[/dim]")
    
    elif parts[0] == 'agent':
        # Show detailed agent decision trace
        agent_logs = agent.get_agent_trace()
        
        if not agent_logs:
            console.print("[dim]No agent decision logs found[/dim]")
            return
        
        console.print(f"\n[bold bright_green]Agent Decision and Model Selection Trace (Last 15 Events):[/bold bright_green]")
        console.print("[dim]Format: [TIMESTAMP] DECISION_TYPE: Detailed reasoning and model selection logic[/dim]\n")
        
        for log in agent_logs[-15:]:  # Show last 15 agent events
            timestamp = log['timestamp']
            event_type = log['type'].upper().replace('_', ' ')
            description = log['description']
            
            if log['type'] == 'task_classification':
                detail = f"Task classification determined: {description}"
                color = "bright_green"
            elif log['type'] == 'model_selection':
                detail = f"Model selection logic: {description}"
                color = "bright_green"
            elif log['type'] == 'api_call':
                detail = f"API call initiated: {description}"
                color = "bright_green"
            elif log['type'] == 'ai_response':
                detail = f"AI response processing: {description}"
                color = "bright_green"
            else:
                detail = f"Agent decision: {description}"
                color = "bright_green"
            
            console.print(f"[dim]{timestamp}[/dim] [{color}]{event_type}[/{color}]: {detail}")
        
        console.print("\n[dim]Use '/logs tools' for tool execution details, '/logs errors' for error analysis[/dim]")
    
    elif parts[0] == 'session':
        # Show session information
        session_id = getattr(agent, 'session_id', 'unknown')
        session_logs = agent.get_session_logs()
        
        console.print(f"\n[bold bright_green]Session Information:[/bold bright_green]")
        console.print(f"Session ID: [bright_green]{session_id}[/bright_green]")
        console.print(f"Total Events: [bright_green]{len(session_logs)}[/bright_green]")
        
        if session_logs:
            first_event = session_logs[0]['timestamp']
            last_event = session_logs[-1]['timestamp']
            console.print(f"Session Duration: [bright_green]{first_event} to {last_event}[/bright_green]")
    
    elif parts[0] == 'errors':
        # Show detailed error analysis
        logs = agent.get_recent_logs(50)  # Get more logs for error analysis
        error_logs = [log for log in logs if 'error' in log.get('description', '').lower() or 
                     'fail' in log.get('description', '').lower() or 
                     log.get('type') == 'tool_result' and 'error' in log.get('description', '').lower()]
        
        if not error_logs:
            console.print("[bright_green]No errors found in recent logs[/bright_green]")
            return
        
        console.print(f"\n[bold bright_green]Error Analysis (Last {len(error_logs)} Errors):[/bold bright_green]")
        console.print("[dim]Format: [TIMESTAMP] ERROR_TYPE: Detailed error description and context[/dim]\n")
        
        for log in error_logs[-10:]:  # Show last 10 errors
            timestamp = log['timestamp']
            event_type = log['type'].upper().replace('_', ' ')
            description = log['description']
            
            console.print(f"[dim]{timestamp}[/dim] [red]{event_type}[/red]: {description}")
        
        console.print(f"\n[dim]Showing {min(10, len(error_logs))} of {len(error_logs)} total errors found[/dim]")
        console.print("[dim]Use '/logs info' for general events, '/logs tools' for tool execution details[/dim]")
    
    else:
        console.print("[yellow]Unknown logs command. Available options:[/yellow]")
        console.print("[dim]  /logs info    - General system events (default)[/dim]")
        console.print("[dim]  /logs tools   - Detailed tool execution trace[/dim]") 
        console.print("[dim]  /logs agent   - Agent decision and model selection trace[/dim]")
        console.print("[dim]  /logs session - Session information and statistics[/dim]")
        console.print("[dim]  /logs errors  - Error analysis and debugging information[/dim]")


def show_enhanced_help():
    """Show enhanced help with model commands"""
    help_text = """
# Enhanced Commands

## Quick Help
- **?** or **help** - Show this help message

## Slash Commands
- **/** - Show available commands
- **/model** - Model management
  - **/model/info** - Show available models and usage
  - **/model/switch <name>** - Switch to a different model
  - **/model/config <task> <model>** - Configure model for task type
  - **/model/save** - Save model configuration
- **/cost** - Cost tracking and limits
  - **/cost/info** - Show cost information and usage
  - **/cost/reset** - Reset cost tracking
  - **/cost/estimate <model> <tokens>** - Estimate cost for a query
- **/agent** - Agent management
  - **/agent/info** - Show agent information and types
  - **/agent/spawn <type>** - Spawn a new agent (coming soon)
  - **/agent/list** - List active agents (coming soon)
- **/logs** - System event logging and tracing
  - **/logs/info** - Show recent system events
  - **/logs/tools** - Show detailed tool execution trace
  - **/logs/agent** - Show agent decision trace
  - **/logs/session** - Show session information
- **/adp** - ADP rankings and draft tools
  - **/adp** - Show ADP rankings with 10-team and 12-team round estimates
  - **/adp/10** - Show ADP rankings with projected rounds for 10-team league
  - **/adp/12** - Show ADP rankings with projected rounds for 12-team league
- **/tiers** - Positional tier analysis and marginal value
  - **/tiers** - Show marginal value drops between all position tiers
  - **/tiers/te** - Detailed tier breakdown for specific position
- **/intel** - Team position intelligence system
  - **/intel** - Show help and usage examples
  - **/intel/list** - Show available intelligence data
  - **/intel/PHI/wr** - Get detailed Eagles WR intelligence report
- **/help** - Show detailed help
- **/quit**, **/end**, **/e** - Exit the application

## Model Override
- **@model_name <query>** - Use specific model for one query

## Model Selection Examples
- **@perplexity-sonar** Who won the NFL games yesterday?
- **@claude-sonnet-4** Generate a Python script for analysis
- **@gpt-4o-mini** List all QBs on the Cowboys
- **@claude-3-opus** Complex analysis of playoff scenarios

## Automatic Model Selection
The system automatically chooses the best model based on your query:
- Research queries → Perplexity
- Code generation → Claude Sonnet 4
- Complex analysis → GPT-4o or Claude Opus
- Simple queries → GPT-4o-mini or Claude Haiku

## Command Style
Commands use Claude-like syntax with slashes (/) for submenus and ? for help.
Type "/" to see all available commands. Use 'end', 'e', or '/end' to exit.

## Task Types
- **research**: Current events, player news
- **analysis**: Deep player/team analysis
- **code_generation**: Scripts and SQL
- **optimization**: Lineup optimization
- **simple_query**: Quick lookups
- **classification**: Categorization tasks

## Cost Optimization
- Use **@gpt-4o-mini** or **@claude-3-haiku** for simple queries
- Use **@perplexity-sonar** for research instead of GPT-4
- Check costs with **model:info**
"""
    from rich.markdown import Markdown
    console.print(Markdown(help_text))


@cli.command()
@click.argument('task')
@click.option('--model', help='Model to use for this task')
@click.option('--agent', help='Agent type to spawn (research/analysis/code/optimization)')
def spawn(task, model, agent):
    """Spawn an agent with specific model"""
    try:
        enhanced_agent = EnhancedHowieAgent(model=model)
        
        # Build spawn command
        spawn_cmd = f"Spawn a {agent or 'research'} agent for: {task}"
        if model:
            spawn_cmd = f"@{model} {spawn_cmd}"
        
        response = asyncio.run(enhanced_agent.process_message(spawn_cmd))
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def update_rosters():
    """Update database with current NFL roster information"""
    try:
        console.print("[bright_green]Updating NFL rosters...[/bright_green]")
        
        # Import and run the roster updater
        import asyncio
        import sys
        from pathlib import Path
        
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        try:
            from update_rosters import main as update_main
            team_changes = asyncio.run(update_main())
            
            console.print("[bright_green]Roster update completed successfully![/bright_green]")
            
            # Display team changes if any
            if team_changes:
                console.print("\n[bold bright_green]Team Changes Detected:[/bold bright_green]")
                for change in team_changes:
                    console.print(f"  • [bold white]{change['player_name']}[/bold white] ([green]{change['position']}[/green]) [red]{change['old_team']}[/red] → [bright_green]{change['new_team']}[/bright_green]")
            else:
                console.print("\n[bright_green]No team changes detected[/bright_green]")
                
        except ImportError:
            console.print("[red]Roster update script not found. Please check scripts/update_rosters.py[/red]")
        except Exception as e:
            console.print(f"[red]Error updating rosters: {e}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--season', default=2025, help='Season to update ADP data for (default: 2025)')
@click.option('--scoring', default='ppr', type=click.Choice(['ppr', 'half_ppr', 'standard']), help='Scoring format (default: ppr)')
@click.option('--test', is_flag=True, help='Test mode - show what would be updated without making changes')
def update_adp(season, scoring, test):
    """Update ADP data from FantasyPros"""
    try:
        console.print(f"[bright_green]Updating ADP data for {season} season ({scoring} scoring)...[/bright_green]")
        
        # Import and run the ADP updater
        import asyncio
        import sys
        from pathlib import Path
        
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        try:
            from build_fantasypros_adp import build_fantasypros_adp, Args
            
            # Create args object
            args = Args()
            args.season = season
            args.scoring = scoring
            args.test = test
            args.db_url = "sqlite:///data/fantasy_ppr.db"
            
            # Run the ADP update
            build_fantasypros_adp(args)
            
            if test:
                console.print("[yellow]🧪 Test mode completed - no changes made[/yellow]")
            else:
                console.print("[bright_green]ADP data updated successfully![/bright_green]")
                
        except ImportError:
            console.print("[red]ADP update script not found. Please check scripts/build_fantasypros_adp.py[/red]")
        except Exception as e:
            console.print(f"[red]Error updating ADP data: {e}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def models():
    """List all available models with details"""
    try:
        manager = ModelManager()
        
        # Create detailed table
        table = Table(title="Available AI Models", show_header=True, header_style="bold bright_green")
        table.add_column("Model", style="bright_green", width=20)
        table.add_column("Provider", style="bright_green", width=12)
        table.add_column("Tier", style="bright_green", width=10)
        table.add_column("Input $/1K", style="red", width=10)
        table.add_column("Output $/1K", style="red", width=10)
        table.add_column("Tools", style="bright_green", width=8)
        table.add_column("Vision", style="bright_green", width=8)
        table.add_column("Best For", style="bright_green", width=35)
        
        for name, config in sorted(manager.models.items()):
            table.add_row(
                name,
                config.provider.value.upper(),
                config.tier.value,
                f"${config.cost_per_1k_input:.4f}",
                f"${config.cost_per_1k_output:.4f}",
                "✓" if config.supports_tools else "✗",
                "✓" if config.supports_vision else "✗",
                ", ".join(config.best_for) if config.best_for else ""
            )
        
        console.print(table)
        
        # Show task mappings
        console.print("\n[bold]Default Task Mappings:[/bold]")
        for task, model in manager.task_model_mapping.items():
            console.print(f"  {task:20s} → {model}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--save', type=click.Path(), help='Save configuration to file')
def configure(save):
    """Interactive model configuration"""
    try:
        manager = ModelManager()
        
        # Interactive configuration
        console.print("[bold]Configure Model Settings[/bold]\n")
        
        # Configure task mappings
        tasks = ["research", "analysis", "code_generation", "optimization", "simple_query"]
        models = list(manager.models.keys())
        
        for task in tasks:
            current = manager.task_model_mapping.get(task, "default")
            console.print(f"\n[dim]{task}[/dim] (current: {current})")
            console.print("Available models:", ", ".join(models))
            
            choice = console.input("Select model (or Enter to keep current): ").strip()
            if choice and choice in models:
                manager.set_task_model(task, choice)
                console.print(f"[bright_green]Set {task} → {choice}[/bright_green]")
        
        # Set default model
        current_default = manager.current_model
        console.print(f"\n[dim]Default model[/dim] (current: {current_default})")
        choice = console.input("Select default model (or Enter to keep current): ").strip()
        if choice and choice in models:
            manager.set_model(choice)
            console.print(f"[bright_green]Set default → {choice}[/bright_green]")
        
        # Save configuration
        if save:
            save_path = Path(save)
        else:
            save_path = Path.home() / ".howie" / "models.json"
        
        manager.save_config(save_path)
        console.print(f"\n[bright_green]Configuration saved to {save_path}[/bright_green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('input_tokens', type=int)
@click.argument('output_tokens', type=int)
@click.option('--model', help='Model to estimate cost for')
def estimate_cost(input_tokens, output_tokens, model):
    """Estimate cost for token usage"""
    try:
        manager = ModelManager()
        
        if model:
            models_to_check = [model] if model in manager.models else []
            if not models_to_check:
                console.print(f"[red]Unknown model: {model}[/red]")
                return
        else:
            # Check all models
            models_to_check = list(manager.models.keys())
        
        # Create cost comparison table
        table = Table(title=f"Cost Estimate for {input_tokens:,} input + {output_tokens:,} output tokens",
                     show_header=True, header_style="bold bright_green")
        table.add_column("Model", style="bright_green", width=20)
        table.add_column("Provider", style="bright_green", width=12)
        table.add_column("Input Cost", style="red", width=12)
        table.add_column("Output Cost", style="red", width=12)
        table.add_column("Total Cost", style="red", width=12)
        
        costs = []
        for model_name in models_to_check:
            config = manager.models[model_name]
            input_cost = (input_tokens / 1000) * config.cost_per_1k_input
            output_cost = (output_tokens / 1000) * config.cost_per_1k_output
            total_cost = input_cost + output_cost
            
            costs.append((model_name, config.provider.value, input_cost, output_cost, total_cost))
        
        # Sort by total cost
        costs.sort(key=lambda x: x[4])
        
        for model_name, provider, input_cost, output_cost, total_cost in costs:
            table.add_row(
                model_name,
                provider.upper(),
                f"${input_cost:.4f}",
                f"${output_cost:.4f}",
                f"${total_cost:.4f}"
            )
        
        console.print(table)
        
        # Show cheapest and most expensive
        if len(costs) > 1:
            console.print(f"\n[bright_green]Cheapest: {costs[0][0]} ([red]${costs[0][4]:.4f}[/red])[/bright_green]")
            console.print(f"[red]Most expensive: {costs[-1][0]} (${costs[-1][4]:.4f})[/red]")
            console.print(f"[yellow]Difference: ${costs[-1][4] - costs[0][4]:.4f} ({(costs[-1][4] / costs[0][4] - 1) * 100:.1f}% more)[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def show_available_stats(position: str):
    """Show all available stats for a given position"""
    
    # Define available stats by position
    position_stats = {
        'QB': {
            'Combined Stats': ['td', 'yards', 'fantasy'],
            'Passing Stats': ['pass_td', 'pass_yards', 'pass_att', 'pass_comp', 'int'],
            'Rushing Stats': ['rush_td', 'rush_yards', 'rush_att'],
            'Other': ['fumbles']
        },
        'RB': {
            'Combined Stats': ['td', 'yards', 'fantasy'],
            'Rushing Stats': ['rush_td', 'rush_yards', 'rush_att'],
            'Receiving Stats': ['rec', 'rec_td', 'rec_yards', 'targets'],
            'Other': ['fumbles']
        },
        'WR': {
            'Combined Stats': ['td', 'yards', 'fantasy'],
            'Receiving Stats': ['rec', 'rec_td', 'rec_yards', 'targets'],
            'PFF Route Stats': ['routes', 'route_grade', 'route_depth', 'yprr', 'target_per_route', 'reception_per_route'],
            'PFF Efficiency': ['catch_rate', 'drop_rate', 'drops', 'contested_rate', 'contested_rec', 'contested_targets'],
            'PFF YAC Stats': ['yac', 'yac_per_rec', 'yards_per_rec'],
            'PFF Alignment': ['slot_rate', 'wide_rate', 'inline_rate'],
            'PFF QB Rating': ['qb_rating'],
            'PFF Man Coverage': ['man_yprr', 'man_grade', 'man_target_share', 'man_catch_rate', 'man_drop_rate', 'man_contested_rate'],
            'PFF Zone Coverage': ['zone_yprr', 'zone_grade', 'zone_target_share', 'zone_catch_rate', 'zone_drop_rate', 'zone_contested_rate'],
            'PFF Coverage Comparison': ['yprr_diff', 'grade_diff'],
            'Other': ['fumbles']
        },
        'TE': {
            'Combined Stats': ['td', 'yards', 'fantasy'],
            'Receiving Stats': ['rec', 'rec_td', 'rec_yards', 'targets'],
            'PFF Route Stats': ['routes', 'route_grade', 'route_depth', 'yprr', 'target_per_route', 'reception_per_route'],
            'PFF Efficiency': ['catch_rate', 'drop_rate', 'drops', 'contested_rate', 'contested_rec', 'contested_targets'],
            'PFF YAC Stats': ['yac', 'yac_per_rec', 'yards_per_rec'],
            'PFF Alignment': ['slot_rate', 'wide_rate', 'inline_rate'],
            'PFF QB Rating': ['qb_rating'],
            'PFF Man Coverage': ['man_yprr', 'man_grade', 'man_target_share', 'man_catch_rate', 'man_drop_rate', 'man_contested_rate'],
            'PFF Zone Coverage': ['zone_yprr', 'zone_grade', 'zone_target_share', 'zone_catch_rate', 'zone_drop_rate', 'zone_contested_rate'],
            'PFF Coverage Comparison': ['yprr_diff', 'grade_diff'],
            'Other': ['fumbles']
        },
        'K': {
            'Kicking Stats': ['fg_made', 'fg_att', 'fg_pct', 'pat_made', 'pat_att', 'pat_pct', 'points', 'fantasy']
        },
        'DEF': {
            'Defense Stats': ['sacks', 'int', 'fumbles_forced', 'fumbles_rec', 'def_td', 'safeties', 'return_td', 'return_yds', 'points', 'fantasy']
        },
        'DST': {
            'Defense Stats': ['sacks', 'int', 'fumbles_forced', 'fumbles_rec', 'def_td', 'safeties', 'return_td', 'return_yds', 'points', 'fantasy']
        }
    }
    
    # Get stats for the position
    if position not in position_stats:
        console.print(f"[red]Invalid position: {position}. Valid positions: QB, RB, WR, TE, K, DEF, DST[/red]")
        return
    
    stats = position_stats[position]
    
    # Display available stats
    from rich.table import Table
    table = Table(title=f"Available Stats for {position}s", show_header=True, header_style="bold bright_green")
    table.add_column("Category", style="bright_green", width=20)
    table.add_column("Stats", style="bright_green", width=50)
    table.add_column("Example Command", style="bright_green", width=30)
    
    for category, stat_list in stats.items():
        stats_str = ", ".join(stat_list)
        example = f"/{position.lower()}/{stat_list[0]}" if stat_list else "N/A"
        table.add_row(category, stats_str, example)
    
    console.print(table)
    
    # Show ADP info if applicable
    if position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'DST']:
        console.print(f"\n[bold bright_green]ADP Data:[/bold bright_green]")
        console.print(f"  [bright_green]/{position.lower()}/adp[/bright_green] - Current ADP rankings")
        console.print(f"  [bright_green]/{position.lower()}/adp/2024[/bright_green] - Historical ADP")
    
    # Show projections info
    console.print(f"\n[bold bright_green]Projections Data (2025):[/bold bright_green]")
    console.print(f"  [bright_green]/{position.lower()}/projections[/bright_green] - 2025 preseason projections")
    console.print(f"  [bright_green]/{position.lower()}/projections/2025[/bright_green] - 2025 season projections")
    
    # Show strength of schedule info
    if position in ['QB', 'RB', 'WR', 'TE', 'DEF', 'DST']:
        pos_lower = 'dst' if position in ['DEF', 'DST'] else position.lower()
        console.print(f"\n[bold bright_green]Strength of Schedule (SoS):[/bold bright_green]")
        console.print(f"  [bright_green]/{position.lower()}/sos[/bright_green] - Season SoS rankings")
        console.print(f"  [bright_green]/{position.lower()}/sos/playoffs[/bright_green] - Playoff SoS rankings")
        console.print(f"  [bright_green]/{position.lower()}/sos/1-4[/bright_green] - Weeks 1-4 SoS rankings")
    
    # Show bye week info
    console.print(f"\n[bold bright_green]Bye Weeks (2025):[/bold bright_green]")
    console.print(f"  [bright_green]/{position.lower()}/bye[/bright_green] - All {position}s with bye weeks")
    console.print(f"  [bright_green]/{position.lower()}/bye/5[/bright_green] - {position}s with Week 5 bye")
    console.print(f"  [bright_green]/{position.lower()}/bye/14[/bright_green] - {position}s with Week 14 bye")
    
    # Show season info
    console.print(f"\n[bold bright_green]Season Support:[/bold bright_green]")
    console.print(f"  Available seasons: 2018-2025")
    console.print(f"  Format: /{position.lower()}/stat/season (e.g., /{position.lower()}/fantasy/2024)")
    console.print(f"  Default: Current season (2025) if no season specified")


def handle_rapid_stats_command(command: str):
    """Handle rapid stats commands like /wr/adp, /qb/td/2024, etc."""
    try:
        # Parse command: position/stat[/season]
        parts = command.split('/')
        # Remove empty string if command starts with /
        if parts and parts[0] == '':
            parts = parts[1:]
        
        if len(parts) < 2:
            console.print("[red]Invalid format. Use: /position/stat[/season] (e.g., /wr/adp, /qb/td/2024)[/red]")
            return
        
        position = parts[0].upper()
        stat = parts[1].lower()
        
        # Special handling for SoS - third parameter is period, not season
        if stat == 'sos':
            season = 2025  # SoS is always current season
        elif stat == 'bye':
            # Handle bye week queries
            handle_bye_week_command(position, parts)
            return
        else:
            season = parts[2] if len(parts) > 2 else "2025"  # Default to current season
            # Validate season
            try:
                season = int(season)
            except ValueError:
                console.print(f"[red]Invalid season: {season}. Must be a year (e.g., 2024, 2025)[/red]")
                return
        
        # Validate position
        valid_positions = ['WR', 'RB', 'QB', 'TE', 'K', 'DEF', 'DST']
        if position not in valid_positions:
            console.print(f"[red]Invalid position: {position}. Valid positions: {', '.join(valid_positions)}[/red]")
            return
        
        console.print(f"[bright_green]Fetching top 50 {position}s by {stat} for {season}...[/bright_green]")
        
        # Connect to database
        import sqlite3
        from pathlib import Path
        
        db_path = Path(__file__).parent / "data" / "fantasy_ppr.db"
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            return
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if this is a "stats" command to show available stats
        if stat == 'stats':
            show_available_stats(position)
            return
        
        # Build query based on stat type
        if stat == 'adp':
            # Normalize position for ADP query (DEF -> DST)
            adp_position = 'DST' if position in ['DEF', 'DST'] else position
            
            # Query ADP data joined with projections for bye week info
            # Handle case differences between ADP (uppercase) and projections (lowercase) positions
            query = """
                SELECT a.player_name, a.position, a.team, a.adp_overall, a.espn_adp, a.sleeper_adp, a.avg_adp, p.bye_week
                FROM adp_data a
                LEFT JOIN player_projections p ON a.player_name = p.player_name 
                    AND LOWER(a.position) = p.position AND p.season = 2025 AND p.projection_type = 'preseason'
                WHERE a.season = ? AND a.scoring_format = 'ppr' AND a.position = ?
                ORDER BY a.adp_overall ASC
                LIMIT 50
            """
            cursor.execute(query, (season, adp_position))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No ADP data found for {position}s in {season}[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            table = Table(title=f"Top 50 {position}s by ADP ({season})", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Player", style="bright_green", width=22)
            table.add_column("Team", style="bright_green", width=6)
            table.add_column("Bye", style="bright_green", width=5)
            table.add_column("Overall ADP", style="bright_green", width=12)
            table.add_column("ESPN", style="bright_green", width=8)
            table.add_column("Sleeper", style="bright_green", width=8)
            table.add_column("AVG", style="bright_green", width=8)
            
            for i, row in enumerate(results, 1):
                table.add_row(
                    str(i),
                    row[0],  # player_name
                    row[2] or "FA",  # team
                    f"{int(row[7])}" if row[7] else "N/A",  # bye_week
                    f"{row[3]:.1f}" if row[3] else "N/A",  # adp_overall
                    f"{row[4]:.0f}" if row[4] else "N/A",  # espn_adp
                    f"{row[5]:.0f}" if row[5] else "N/A",  # sleeper_adp
                    f"{row[6]:.0f}" if row[6] else "N/A"   # avg_adp
                )
            
            console.print(table)
            
        elif stat == 'projections':
            # Query projections data
            position_db = 'dst' if position in ['DEF', 'DST'] else position.lower()
            query = """
                SELECT player_name, team_name, fantasy_points, games, bye_week, auction_value
                FROM player_projections 
                WHERE season = ? AND position = ? AND projection_type = 'preseason'
                ORDER BY fantasy_points DESC
                LIMIT 50
            """
            cursor.execute(query, (season, position_db))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No projection data found for {position}s in {season}[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            table = Table(title=f"Top 50 {position}s by 2025 Projections", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Player", style="bright_green", width=25)
            table.add_column("Team", style="bright_green", width=6)
            table.add_column("Fantasy Pts", style="bright_green", width=12)
            table.add_column("Games", style="bright_green", width=8)
            table.add_column("Bye", style="bright_green", width=6)
            table.add_column("Auction $", style="bright_green", width=10)
            
            for i, row in enumerate(results, 1):
                table.add_row(
                    str(i),
                    row[0],  # player_name
                    row[1] or "N/A",  # team_name
                    f"{row[2]:.1f}" if row[2] else "N/A",  # fantasy_points
                    str(row[3]) if row[3] else "N/A",  # games
                    str(row[4]) if row[4] else "N/A",  # bye_week
                    f"${row[5]:.0f}" if row[5] else "N/A"   # auction_value
                )
            
            console.print(table)
            
        elif stat == 'sos':
            # Query strength of schedule data
            position_db = 'dst' if position in ['DEF', 'DST'] else position.lower()
            
            # For SoS, the third part is the period, not season (SoS is always 2025)
            period = parts[2] if len(parts) > 2 else 'season'
            
            if period == 'season':
                order_col = 'season_sos'
                select_cols = 'team, season_sos as sos_value, season_games as games'
                desc = 'Season'
            elif period == 'playoffs':
                order_col = 'playoffs_sos'
                select_cols = 'team, playoffs_sos as sos_value, playoffs_games as games'
                desc = 'Playoffs'
            elif '-' in period:
                # Week range like "1-4"
                try:
                    start_week, end_week = map(int, period.split('-'))
                    week_cols = [f'week_{i}' for i in range(start_week, end_week + 1)]
                    avg_expr = f"({' + '.join([f'COALESCE({col}, 0)' for col in week_cols])}) / {len(week_cols)}"
                    select_cols = f'team, {avg_expr} as sos_value, {end_week - start_week + 1} as games'
                    order_col = 'sos_value'
                    desc = f'Weeks {start_week}-{end_week}'
                except:
                    console.print(f"[red]Invalid week range: {period}. Use format like '1-4'[/red]")
                    return
            else:
                console.print(f"[red]Invalid SoS period: {period}. Use 'season', 'playoffs', or week range like '1-4'[/red]")
                return
            
            query = f"""
                SELECT {select_cols}
                FROM strength_of_schedule 
                WHERE season = 2025 AND position = ? AND {order_col} IS NOT NULL
                ORDER BY {order_col} ASC
                LIMIT 32
            """
            cursor.execute(query, (position_db,))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No SoS data found for {position}s[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            table = Table(title=f"{position} Strength of Schedule - {desc} (2025)", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Team", style="bright_green", width=8)
            table.add_column("SoS Rating", style="bright_green", width=12)
            table.add_column("Games", style="bright_green", width=8)
            table.add_column("Difficulty", style="bright_green", width=12)
            
            for i, row in enumerate(results, 1):
                sos_val = row[1]
                if sos_val is not None:
                    # Lower SoS values = HARDER schedule, Higher values = EASIER schedule
                    # Scale: 0.0 (hardest) to 10.0 (easiest)
                    # Create 5 balanced tiers: 0-2, 2-4, 4-6, 6-8, 8-10
                    if sos_val <= 2.0:
                        difficulty = "Hardest"
                        diff_color = "red"
                    elif sos_val <= 4.0:
                        difficulty = "Hard"
                        diff_color = "red" 
                    elif sos_val <= 6.0:
                        difficulty = "Average"
                        diff_color = "yellow"
                    elif sos_val <= 8.0:
                        difficulty = "Easy"
                        diff_color = "bright_green"
                    else:  # 8.0+
                        difficulty = "Easiest"
                        diff_color = "bright_green"
                    
                    table.add_row(
                        str(i),
                        row[0],  # team
                        f"{sos_val:.1f}",  # sos_value
                        str(row[2]) if row[2] else "N/A",  # games
                        difficulty
                    )
            
            console.print(table)
            
        else:
            # Query player stats from player_game_stats table
            # Map stat names to database columns
            stat_mapping = {
                # Combined stats
                'td': 'pass_tds + rush_tds + rec_tds',
                'yards': 'pass_yards + rush_yards + rec_yards',
                'fantasy': 'fantasy_points',
                
                # QB-specific stats
                'pass_td': 'pass_tds',
                'pass_yards': 'pass_yards',
                'pass_att': 'pass_attempts',
                'pass_comp': 'pass_completions',
                'int': 'interceptions',
                'rush_td': 'rush_tds',
                'rush_yards': 'rush_yards',
                'rush_att': 'rush_attempts',
                
                # RB-specific stats
                'rush_td': 'rush_tds',
                'rush_yards': 'rush_yards',
                'rush_att': 'rush_attempts',
                'rec_td': 'rec_tds',
                'rec_yards': 'rec_yards',
                'targets': 'targets',
                
                # WR/TE-specific stats
                'rec': 'receptions',
                'rec_td': 'rec_tds',
                'rec_yards': 'rec_yards',
                'targets': 'targets',
                
                # General stats
                'fumbles': 'fumbles',
                
                # Kicker stats (2025 projections)
                'fg_made': 'fg_made_total',
                'fg_att': 'fg_att_total', 
                'fg_pct': 'fg_percentage',
                'pat_made': 'pat_made',
                'pat_att': 'pat_att',
                'pat_pct': 'pat_percentage',
                'points': 'fantasy_points',
                
                # Defense stats (2025 projections)
                'sacks': 'dst_sacks',
                'int': 'dst_int',
                'fumbles_forced': 'dst_fumbles_forced',
                'fumbles_rec': 'dst_fumbles_recovered',
                'def_td': 'dst_td',
                'safeties': 'dst_safeties',
                'return_td': 'dst_return_td',
                'return_yds': 'dst_return_yds',
                'pts_allowed': 'dst_pts_allowed'
            }
            
            # PFF Advanced Stats mapping (for WR/TE)
            pff_stat_mapping = {
                # Route running stats
                'routes': 'routes_run',
                'route_grade': 'route_grade',
                'route_depth': 'route_depth',
                'yprr': 'yards_per_route_run',
                'target_per_route': 'target_per_route',
                'reception_per_route': 'reception_per_route',
                
                # Efficiency stats
                'catch_rate': 'catch_rate',
                'drop_rate': 'drop_rate',
                'drops': 'drops',
                'contested_rate': 'contested_catch_rate',
                'contested_rec': 'contested_receptions',
                'contested_targets': 'contested_targets',
                
                # YAC stats
                'yac': 'yac',
                'yac_per_rec': 'yac_per_reception',
                'yards_per_rec': 'yards_per_reception',
                
                # Alignment stats
                'slot_rate': 'slot_rate',
                'wide_rate': 'wide_rate',
                'inline_rate': 'inline_rate',
                
                # QB rating when targeted
                'qb_rating': 'targeted_qb_rating'
            }
            
            # PFF Scheme Stats mapping (man vs zone coverage)
            pff_scheme_stat_mapping = {
                # Man coverage stats
                'man_yprr': 'man_yards_per_route_run',
                'man_grade': 'man_route_grade',
                'man_target_share': 'man_target_share',
                'man_catch_rate': 'man_catch_rate',
                'man_drop_rate': 'man_drop_rate',
                'man_contested_rate': 'man_contested_catch_rate',
                
                # Zone coverage stats
                'zone_yprr': 'zone_yards_per_route_run',
                'zone_grade': 'zone_route_grade',
                'zone_target_share': 'zone_target_share',
                'zone_catch_rate': 'zone_catch_rate',
                'zone_drop_rate': 'zone_drop_rate',
                'zone_contested_rate': 'zone_contested_catch_rate',
                
                # Comparison stats
                'yprr_diff': 'yprr_man_vs_zone_diff',
                'grade_diff': 'route_grade_man_vs_zone_diff'
            }
            
            # Check if this is a PFF stat (for WR/TE)
            if stat in pff_stat_mapping and position in ['WR', 'TE']:
                # Use PFF route stats table
                pff_column = pff_stat_mapping[stat]
                query = f"""
                    SELECT player_name, team, {pff_column} as total_{stat}
                    FROM player_route_stats 
                    WHERE season = ? AND position = ? AND {pff_column} IS NOT NULL
                    ORDER BY {pff_column} DESC
                    LIMIT 50
                """
            elif stat in pff_scheme_stat_mapping and position in ['WR', 'TE']:
                # Use PFF scheme stats table
                pff_column = pff_scheme_stat_mapping[stat]
                query = f"""
                    SELECT player_name, team, {pff_column} as total_{stat}
                    FROM player_scheme_stats 
                    WHERE season = ? AND position = ? AND {pff_column} IS NOT NULL
                    ORDER BY {pff_column} DESC
                    LIMIT 50
                """
            elif (position in ['K', 'DEF', 'DST']) and stat in stat_mapping:
                if position == 'K':
                    # Use 2025 projections table for kickers only
                    position = 'k'  # Normalize to lowercase
                    proj_column = stat_mapping[stat]
                    query = f"""
                        SELECT player_name, team_name, {proj_column} as total_{stat}
                        FROM player_projections 
                        WHERE season = ? AND position = ? AND projection_type = 'preseason' AND {proj_column} IS NOT NULL
                        ORDER BY {proj_column} DESC
                        LIMIT 50
                    """
                elif position in ['DEF', 'DST']:
                    # Use historical team defensive stats for 2018-2024, projections for 2025+
                    if season >= 2025:
                        # Use projections for future seasons
                        position = 'dst'  # Normalize to dst for database
                        proj_column = stat_mapping[stat]
                        query = f"""
                            SELECT player_name, team_name, {proj_column} as total_{stat}
                            FROM player_projections 
                            WHERE season = ? AND position = ? AND projection_type = 'preseason' AND {proj_column} IS NOT NULL
                            ORDER BY {proj_column} DESC
                            LIMIT 50
                        """
                    else:
                        # Use historical defensive stats for 2018-2024
                        team_column_mapping = {
                            'sacks': 'total_sacks',
                            'int': 'total_interceptions',
                            'def_td': 'total_tds_allowed',
                            'fumbles_forced': 'total_hurries',  # Using hurries as proxy
                            'fumbles_rec': 'total_qb_hits',     # Using QB hits as proxy
                        }
                        
                        if stat not in team_column_mapping:
                            console.print(f"[red]Stat '{stat}' not available for historical defensive data. Available: {', '.join(team_column_mapping.keys())}[/red]")
                            return
                        
                        hist_column = team_column_mapping[stat]
                        query = f"""
                            SELECT team as player_name, team as team_name, {hist_column} as total_{stat}
                            FROM team_defensive_stats 
                            WHERE season = ? AND {hist_column} IS NOT NULL
                            ORDER BY {hist_column} DESC
                            LIMIT 50
                        """
            elif stat not in stat_mapping:
                console.print(f"[red]Invalid stat: {stat}. Valid stats: {', '.join(stat_mapping.keys())}[/red]")
                return
            else:
                # Check if we should use projections for 2025
                if season >= 2025:
                    # Use 2025 projections for all positions
                    proj_stat_mapping = {
                        # QB stats
                        'pass_td': 'pass_td',
                        'pass_yards': 'pass_yds', 
                        'pass_att': 'pass_att',
                        'pass_comp': 'pass_comp',
                        'int': 'pass_int',
                        
                        # Rushing stats (QB, RB)
                        'rush_td': 'rush_td',
                        'rush_yards': 'rush_yds',
                        'rush_att': 'rush_att',
                        
                        # Receiving stats (WR, TE, RB)
                        'rec_td': 'recv_td',
                        'rec_yards': 'recv_yds',
                        'rec': 'receptions',
                        'targets': 'targets',
                        
                        # Fantasy points
                        'fantasy': 'fantasy_points',
                        'points': 'fantasy_points'
                    }
                    
                    # Add position-dependent combined stats
                    if position in ['WR', 'TE']:
                        proj_stat_mapping['td'] = 'recv_td'
                        proj_stat_mapping['yards'] = 'recv_yds'
                    elif position == 'RB':
                        proj_stat_mapping['td'] = 'rush_td'
                        proj_stat_mapping['yards'] = 'rush_yds'
                    elif position == 'QB':
                        proj_stat_mapping['td'] = 'pass_td'
                        proj_stat_mapping['yards'] = 'pass_yds'
                    
                    if stat not in proj_stat_mapping:
                        console.print(f"[red]Stat '{stat}' not available in 2025 projections. Available: {', '.join(proj_stat_mapping.keys())}[/red]")
                        return
                    
                    proj_column = proj_stat_mapping[stat]
                    position_db = position.lower()
                    
                    query = f"""
                        SELECT player_name, team_name, {proj_column} as total_{stat}
                        FROM player_projections 
                        WHERE season = ? AND position = ? AND projection_type = 'preseason' AND {proj_column} IS NOT NULL
                        ORDER BY {proj_column} DESC
                        LIMIT 50
                    """
                else:
                    # Build the query for historical stats
                    if stat in ['td', 'yards']:
                        # Combined stats across all categories
                        select_clause = f"SUM({stat_mapping[stat]}) as total_{stat}"
                    else:
                        # Single category stats
                        select_clause = f"SUM({stat_mapping[stat]}) as total_{stat}"
                    
                    query = f"""
                        SELECT p.name, p.team, {select_clause}
                        FROM player_game_stats pgs
                        JOIN players p ON pgs.player_id = p.player_id
                        JOIN games g ON pgs.game_id = g.game_id
                        WHERE g.season = ? AND p.position = ?
                        GROUP BY p.player_id, p.name, p.team
                        ORDER BY total_{stat} DESC
                        LIMIT 50
                    """
            
            # Determine query parameters based on the type of query
            if position in ['DEF', 'DST'] and season < 2025:
                # Historical defensive stats only need season parameter
                cursor.execute(query, (season,))
            elif season >= 2025 and position not in ['K', 'DEF', 'DST']:
                # 2025 projections for QB/RB/WR/TE need season and position
                cursor.execute(query, (season, position_db))
            else:
                # Regular historical stats and K/DEF projections need season and position
                cursor.execute(query, (season, position))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No stats found for {position}s in {season}[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            # Determine if we're showing projections or historical data
            if season >= 2025:
                title_suffix = " (2025 Projections)"
                column_header = f"Proj. {stat.upper()}"
            elif position in ['DEF', 'DST'] and season < 2025:
                title_suffix = f" ({season} Actuals)"
                column_header = f"Total {stat.upper()}"
            else:
                title_suffix = f" ({season})"
                column_header = f"Total {stat.upper()}"
            
            table = Table(title=f"Top 50 {position}s by {stat.upper()}{title_suffix}", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Player", style="bright_green", width=25)
            table.add_column("Team", style="bright_green", width=6)
            table.add_column(column_header, style="bright_green", width=15)
            
            for i, row in enumerate(results, 1):
                table.add_row(
                    str(i),
                    row[0],  # name
                    row[1] or "FA",  # team
                    f"{row[2]:,.0f}" if row[2] else "0"  # total stat
                )
            
            console.print(table)
        
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error fetching stats: {e}[/red]")


def handle_adp_command(command: str):
    """Handle ADP commands like /adp, /adp/10, /adp/12, etc."""
    try:
        parts = command.split('/')
        
        if len(parts) == 1:
            # Just /adp - show overall ADP rankings with both 10-team and 12-team columns
            handle_adp_rankings()
        elif len(parts) == 2:
            # /adp/number - show ADP rankings with projected round for specific league size
            try:
                league_size = int(parts[1])
                if league_size < 6 or league_size > 20:
                    console.print(f"[red]Error: League size must be between 6 and 20 teams[/red]")
                    return
                handle_adp_rankings_with_league_size(league_size)
            except ValueError:
                console.print(f"[red]Error: Invalid league size '{parts[1]}'. Please use a number (e.g., /adp/10)[/red]")
        else:
            console.print(f"[red]Error: Invalid ADP command format. Use /adp or /adp/number[/red]")
            
    except Exception as e:
        console.print(f"[red]Error processing ADP command: {str(e)}[/red]")


def handle_adp_rankings():
    """Show overall ADP rankings"""
    try:
        import sqlite3
        from rich.table import Table
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query ADP data with bye weeks
        query = """
        SELECT a.player_name, a.position, a.team, a.adp_overall, a.adp_position, p.bye_week
        FROM adp_data a
        LEFT JOIN player_projections p ON a.player_name = p.player_name 
            AND LOWER(a.position) = p.position AND p.season = 2025 AND p.projection_type = 'preseason'
        WHERE a.season = 2025
        ORDER BY a.adp_overall ASC
        LIMIT 50
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            console.print("[red]No ADP data found for 2025 season[/red]")
            return
        
        # Create table
        table = Table(title="2025 Fantasy Football ADP Rankings (PPR)", title_style="bold bright_green")
        table.add_column("Rank", style="bright_green", width=6)
        table.add_column("Player", style="white", width=20)
        table.add_column("Pos", style="bright_green", width=5)
        table.add_column("Team", style="bright_green", width=6)
        table.add_column("Overall ADP", style="white", width=12)
        table.add_column("Pos ADP", style="white", width=10)
        table.add_column("Bye", style="dim", width=6)
        table.add_column("Round (10tm)", style="yellow", width=12)
        table.add_column("Round (12tm)", style="yellow", width=12)
        
        for i, row in enumerate(results, 1):
            player_name, position, team, adp_overall, adp_position, bye_week = row
            
            # Calculate rounds for different league sizes
            round_10team = calculate_round(adp_overall, 10)
            round_12team = calculate_round(adp_overall, 12)
            
            bye_display = f"{int(bye_week)}" if bye_week else "N/A"
            
            table.add_row(
                str(i),
                player_name,
                position,
                team or "N/A",
                f"{adp_overall:.1f}",
                f"{adp_position:.1f}" if adp_position else "N/A",
                bye_display,
                round_10team,
                round_12team
            )
        
        console.print(table)
        console.print("\n[dim]Use /adp/number to see round estimation for specific ADP (e.g., /adp/24)[/dim]")
        console.print("[dim]Round calculations: 10-team league = 10 picks/round, 12-team = 12 picks/round[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error showing ADP rankings: {str(e)}[/red]")


def handle_adp_rankings_with_league_size(league_size: int):
    """Show ADP rankings with projected round for specific league size"""
    try:
        import sqlite3
        from rich.table import Table
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query ADP data with bye weeks
        query = """
        SELECT a.player_name, a.position, a.team, a.adp_overall, a.adp_position, p.bye_week
        FROM adp_data a
        LEFT JOIN player_projections p ON a.player_name = p.player_name 
            AND LOWER(a.position) = p.position AND p.season = 2025 AND p.projection_type = 'preseason'
        WHERE a.season = 2025
        ORDER BY a.adp_overall ASC
        LIMIT 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            console.print("[red]No ADP data found for 2025 season[/red]")
            return
        
        # Create table
        table = Table(title=f"2025 Fantasy Football ADP Rankings - {league_size} Team League", title_style="bold bright_green")
        table.add_column("Rank", style="bright_green", width=6)
        table.add_column("Player", style="white", width=20)
        table.add_column("Pos", style="bright_green", width=5)
        table.add_column("Team", style="bright_green", width=6)
        table.add_column("Overall ADP", style="white", width=12)
        table.add_column("Pos ADP", style="white", width=10)
        table.add_column("Bye", style="dim", width=6)
        table.add_column(f"Proj. Round", style="yellow", width=12)
        
        for i, row in enumerate(results, 1):
            player_name, position, team, adp_overall, adp_position, bye_week = row
            
            # Calculate projected round for this league size
            projected_round = calculate_round(adp_overall, league_size)
            
            bye_display = f"{int(bye_week)}" if bye_week else "N/A"
            
            table.add_row(
                str(i),
                player_name,
                position,
                team or "N/A",
                f"{adp_overall:.1f}",
                f"{adp_position:.1f}" if adp_position else "N/A",
                bye_display,
                projected_round
            )
        
        console.print(table)
        console.print(f"\n[dim]Projected rounds based on {league_size}-team league ({league_size} picks per round)[/dim]")
        console.print("[dim]Format: Round.Pick (e.g., 2.05 = Round 2, Pick 5)[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error showing ADP rankings: {str(e)}[/red]")


def handle_adp_round_estimation(adp_position: int):
    """Show round estimation for specific ADP position"""
    try:
        from rich.table import Table
        from rich.panel import Panel
        
        # Calculate rounds for common league sizes
        league_sizes = [8, 10, 12, 14, 16]
        
        table = Table(title=f"Round Estimation for ADP #{adp_position}", title_style="bold bright_green")
        table.add_column("League Size", style="bright_green", width=12)
        table.add_column("Round", style="white", width=8)
        table.add_column("Pick in Round", style="white", width=15)
        table.add_column("Description", style="dim", width=30)
        
        for league_size in league_sizes:
            round_num = calculate_round_number(adp_position, league_size)
            pick_in_round = calculate_pick_in_round(adp_position, league_size)
            
            # Add description based on round
            if round_num == 1:
                description = "Elite tier, first round pick"
            elif round_num == 2:
                description = "High-end starter"
            elif round_num <= 4:
                description = "Solid starter"
            elif round_num <= 6:
                description = "Flex/backup option"
            elif round_num <= 10:
                description = "Bench depth"
            else:
                description = "Deep sleeper/waiver wire"
            
            table.add_row(
                f"{league_size} teams",
                f"Round {round_num}",
                f"Pick {pick_in_round}",
                description
            )
        
        console.print(table)
        
        # Show additional context
        panel_text = f"""
**ADP #{adp_position} Analysis:**

• In a **10-team league**: {calculate_round(adp_position, 10)}
• In a **12-team league**: {calculate_round(adp_position, 12)}

**Draft Strategy:**
• This player is typically selected in the {calculate_round_number(adp_position, 12)} round
• Consider targeting them 1-2 picks earlier than ADP if you really want them
• Good value if available 3-5 picks later than ADP
        """
        
        console.print(Panel(panel_text, border_style="dim", title="Draft Context"))
        
    except Exception as e:
        console.print(f"[red]Error calculating round estimation: {str(e)}[/red]")


def calculate_round(adp: float, league_size: int) -> str:
    """Calculate round display string for ADP"""
    round_num = calculate_round_number(adp, league_size)
    pick_in_round = calculate_pick_in_round(adp, league_size)
    return f"{round_num}.{pick_in_round:02d}"


def calculate_round_number(adp: float, league_size: int) -> int:
    """Calculate which round an ADP falls into"""
    return int((adp - 1) // league_size) + 1


def calculate_pick_in_round(adp: float, league_size: int) -> int:
    """Calculate which pick within the round"""
    return int((adp - 1) % league_size) + 1


def handle_tiers_command(command: str):
    """Handle tiers commands like /tiers, /tiers/wr, /tiers/rb, etc."""
    try:
        parts = command.split('/')
        
        if len(parts) == 1:
            # Just /tiers - show all positions
            handle_positional_tiers()
        elif len(parts) == 2:
            # /tiers/position - show specific position breakdown
            position = parts[1].lower()
            valid_positions = ['qb', 'rb', 'wr', 'te', 'k', 'dst']
            if position in valid_positions:
                handle_specific_position_tiers(position)
            else:
                console.print(f"[red]Error: Invalid position '{position}'. Valid positions: {', '.join(valid_positions)}[/red]")
        else:
            console.print(f"[red]Error: Invalid tiers command format. Use /tiers or /tiers/position[/red]")
            
    except Exception as e:
        console.print(f"[red]Error processing tiers command: {str(e)}[/red]")


def handle_positional_tiers():
    """Show marginal value analysis for all positions"""
    try:
        import sqlite3
        import numpy as np
        from rich.table import Table
        from rich.panel import Panel
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        positions = ['qb', 'rb', 'wr', 'te', 'k', 'dst']
        
        # Create summary table
        table = Table(title="2025 Positional Tier Analysis - Marginal Value Drops", title_style="bold bright_green")
        table.add_column("Position", style="bright_green", width=10)
        table.add_column("Players", style="dim", width=8)
        table.add_column("Tier 1→2", style="red", width=10)
        table.add_column("Tier 2→3", style="yellow", width=10) 
        table.add_column("Tier 3→4", style="white", width=10)
        table.add_column("Biggest Drop", style="bright_red", width=12)
        
        for pos in positions:
            cursor.execute('''
                SELECT fantasy_points 
                FROM player_projections 
                WHERE season = 2025 AND projection_type = 'preseason' AND position = ? 
                ORDER BY fantasy_points DESC
            ''', (pos,))
            
            results = cursor.fetchall()
            if not results:
                continue
                
            points = [r[0] for r in results]
            n = len(points)
            
            # Calculate tier breakpoints
            tier1_end = max(1, int(n * 0.10))
            tier2_end = max(tier1_end + 1, int(n * 0.25))
            tier3_end = max(tier2_end + 1, int(n * 0.50))
            tier4_end = max(tier3_end + 1, int(n * 0.75))
            
            tier1_avg = np.mean(points[:tier1_end]) if tier1_end > 0 else 0
            tier2_avg = np.mean(points[tier1_end:tier2_end]) if tier2_end > tier1_end else 0
            tier3_avg = np.mean(points[tier2_end:tier3_end]) if tier3_end > tier2_end else 0
            tier4_avg = np.mean(points[tier3_end:tier4_end]) if tier4_end > tier3_end else 0
            
            # Calculate drops
            drop_1_2 = tier1_avg - tier2_avg if tier1_avg > 0 and tier2_avg > 0 else 0
            drop_2_3 = tier2_avg - tier3_avg if tier2_avg > 0 and tier3_avg > 0 else 0
            drop_3_4 = tier3_avg - tier4_avg if tier3_avg > 0 and tier4_avg > 0 else 0
            
            biggest_drop = max(drop_1_2, drop_2_3, drop_3_4)
            biggest_tier = ""
            if biggest_drop == drop_1_2:
                biggest_tier = "Tier 1→2"
            elif biggest_drop == drop_2_3:
                biggest_tier = "Tier 2→3"  
            else:
                biggest_tier = "Tier 3→4"
            
            table.add_row(
                pos.upper(),
                str(n),
                f"-{drop_1_2:.1f}",
                f"-{drop_2_3:.1f}",
                f"-{drop_3_4:.1f}",
                f"{biggest_tier} ({biggest_drop:.1f})"
            )
        
        console.print(table)
        
        # Add ADP-based draft round analysis
        draft_analysis = get_draft_round_analysis()
        
        # Key insights panel
        insights = f"""
**Key Draft Strategy Insights:**

• **RB has the largest early drop-offs** - prioritize elite RBs early
• **TE shows major scarcity** - Tier 1 TEs have huge value over Tier 2+
• **WR has consistent drop-offs** - multiple tiers of viable options
• **QB can wait** - smaller gaps between tiers, stream-friendly
• **K/DST minimal differences** - wait until very late rounds

**Tier Definitions:**
• Tier 1: Top 10% of position
• Tier 2: 11-25% of position  
• Tier 3: 26-50% of position
• Tier 4: 51-75% of position

**Draft Round Analysis (When Tiers Get Selected):**

{draft_analysis}
        """
        
        console.print(Panel(insights, border_style="dim", title="Draft Strategy"))
        console.print("\n[dim]Use /tiers/position for detailed breakdown (e.g., /tiers/te)[/dim]")
        
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error showing positional tiers: {str(e)}[/red]")


def handle_specific_position_tiers(position: str):
    """Show detailed tier breakdown for specific position"""
    try:
        import sqlite3
        import numpy as np
        from rich.table import Table
        from rich.panel import Panel
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT player_name, fantasy_points 
            FROM player_projections 
            WHERE season = 2025 AND projection_type = 'preseason' AND position = ? 
            ORDER BY fantasy_points DESC
        ''', (position,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            console.print(f"[red]No projection data found for {position.upper()}[/red]")
            return
            
        points = [r[1] for r in results]
        players = [r[0] for r in results]
        n = len(points)
        
        # Calculate tier breakpoints
        tier1_end = max(1, int(n * 0.10))
        tier2_end = max(tier1_end + 1, int(n * 0.25))
        tier3_end = max(tier2_end + 1, int(n * 0.50))
        tier4_end = max(tier3_end + 1, int(n * 0.75))
        
        # Create detailed table
        table = Table(title=f"{position.upper()} Tier Analysis - 2025 Projections", title_style="bold bright_green")
        table.add_column("Tier", style="bright_green", width=8)
        table.add_column("Range", style="dim", width=12)
        table.add_column("Players", style="white", width=8)
        table.add_column("Avg Points", style="white", width=12)
        table.add_column("Top Players", style="bright_green", width=40)
        table.add_column("Marginal Drop", style="red", width=15)
        
        # Tier 1
        tier1_avg = np.mean(points[:tier1_end])
        tier1_players = ", ".join(players[:min(3, tier1_end)])
        table.add_row("Tier 1", "Top 10%", str(tier1_end), f"{tier1_avg:.1f}", tier1_players, "-")
        
        # Tier 2
        if tier2_end > tier1_end:
            tier2_avg = np.mean(points[tier1_end:tier2_end])
            tier2_players = ", ".join(players[tier1_end:min(tier1_end + 3, tier2_end)])
            drop_1_2 = tier1_avg - tier2_avg
            table.add_row("Tier 2", "11-25%", str(tier2_end - tier1_end), f"{tier2_avg:.1f}", tier2_players, f"-{drop_1_2:.1f}")
            
            # Tier 3
            if tier3_end > tier2_end:
                tier3_avg = np.mean(points[tier2_end:tier3_end])
                tier3_players = ", ".join(players[tier2_end:min(tier2_end + 3, tier3_end)])
                drop_2_3 = tier2_avg - tier3_avg
                table.add_row("Tier 3", "26-50%", str(tier3_end - tier2_end), f"{tier3_avg:.1f}", tier3_players, f"-{drop_2_3:.1f}")
                
                # Tier 4
                if tier4_end > tier3_end:
                    tier4_avg = np.mean(points[tier3_end:tier4_end])
                    tier4_players = ", ".join(players[tier3_end:min(tier3_end + 3, tier4_end)])
                    drop_3_4 = tier3_avg - tier4_avg
                    table.add_row("Tier 4", "51-75%", str(tier4_end - tier3_end), f"{tier4_avg:.1f}", tier4_players, f"-{drop_3_4:.1f}")
                    
                    # Tier 5
                    if n > tier4_end:
                        tier5_avg = np.mean(points[tier4_end:])
                        tier5_players = ", ".join(players[tier4_end:min(tier4_end + 3, n)])
                        drop_4_5 = tier4_avg - tier5_avg
                        table.add_row("Tier 5", "76%+", str(n - tier4_end), f"{tier5_avg:.1f}", tier5_players, f"-{drop_4_5:.1f}")
        
        console.print(table)
        
        # Position-specific insights with ADP analysis
        insights = get_position_insights(position, results)
        adp_analysis = get_position_adp_analysis(position, players)
        
        combined_insights = f"{insights}\n\n**Draft Round Analysis:**\n{adp_analysis}" if adp_analysis else insights
        
        if combined_insights:
            console.print(Panel(combined_insights, border_style="dim", title=f"{position.upper()} Strategy"))
        
    except Exception as e:
        console.print(f"[red]Error showing {position} tiers: {str(e)}[/red]")


def get_draft_round_analysis():
    """Analyze when each position tier typically gets drafted based on ADP"""
    try:
        import sqlite3
        import numpy as np
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        analysis_text = ""
        positions = ['qb', 'rb', 'wr', 'te']
        
        for pos in positions:
            # Get projection data to determine tiers
            cursor.execute('''
                SELECT player_name, fantasy_points 
                FROM player_projections 
                WHERE season = 2025 AND projection_type = 'preseason' AND position = ? 
                ORDER BY fantasy_points DESC
            ''', (pos,))
            
            proj_results = cursor.fetchall()
            if not proj_results:
                continue
                
            points = [r[1] for r in proj_results]
            players = [r[0] for r in proj_results]
            n = len(points)
            
            # Calculate tier breakpoints
            tier1_end = max(1, int(n * 0.10))
            tier2_end = max(tier1_end + 1, int(n * 0.25))
            tier3_end = max(tier2_end + 1, int(n * 0.50))
            
            # Get tier player lists
            tier1_players = players[:tier1_end]
            tier2_players = players[tier1_end:tier2_end] if tier2_end > tier1_end else []
            tier3_players = players[tier2_end:tier3_end] if tier3_end > tier2_end else []
            
            # Get ADP data for these players
            tier_adps = {}
            for tier_num, tier_players in enumerate([tier1_players, tier2_players, tier3_players], 1):
                if not tier_players:
                    continue
                    
                placeholders = ','.join(['?' for _ in tier_players])
                cursor.execute(f'''
                    SELECT adp_overall 
                    FROM adp_data 
                    WHERE season = 2025 AND player_name IN ({placeholders})
                    AND adp_overall IS NOT NULL
                ''', tier_players)
                
                adp_results = cursor.fetchall()
                if adp_results:
                    adps = [r[0] for r in adp_results]
                    tier_adps[tier_num] = {
                        'min': min(adps),
                        'max': max(adps),
                        'avg': np.mean(adps)
                    }
            
            # Convert to round ranges
            analysis_text += f"\n**{pos.upper()}:**\n"
            for tier_num, adp_data in tier_adps.items():
                if adp_data:
                    # Calculate rounds for 10 and 12 team leagues
                    min_round_10 = calculate_round_number(adp_data['min'], 10)
                    max_round_10 = calculate_round_number(adp_data['max'], 10)
                    avg_round_10 = calculate_round_number(adp_data['avg'], 10)
                    
                    min_round_12 = calculate_round_number(adp_data['min'], 12)
                    max_round_12 = calculate_round_number(adp_data['max'], 12)
                    avg_round_12 = calculate_round_number(adp_data['avg'], 12)
                    
                    if min_round_10 == max_round_10:
                        round_str_10 = f"Rd {min_round_10}"
                    else:
                        round_str_10 = f"Rds {min_round_10}-{max_round_10}"
                        
                    if min_round_12 == max_round_12:
                        round_str_12 = f"Rd {min_round_12}"
                    else:
                        round_str_12 = f"Rds {min_round_12}-{max_round_12}"
                    
                    analysis_text += f"• Tier {tier_num}: {round_str_10} (10tm), {round_str_12} (12tm)\n"
        
        conn.close()
        return analysis_text.strip()
        
    except Exception as e:
        return f"Error analyzing draft rounds: {str(e)}"


def get_position_adp_analysis(position: str, players: list):
    """Get ADP-based round analysis for specific position"""
    try:
        import sqlite3
        import numpy as np
        
        # Connect to database
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        n = len(players)
        
        # Calculate tier breakpoints
        tier1_end = max(1, int(n * 0.10))
        tier2_end = max(tier1_end + 1, int(n * 0.25))
        tier3_end = max(tier2_end + 1, int(n * 0.50))
        tier4_end = max(tier3_end + 1, int(n * 0.75))
        
        # Get tier player lists
        tier1_players = players[:tier1_end]
        tier2_players = players[tier1_end:tier2_end] if tier2_end > tier1_end else []
        tier3_players = players[tier2_end:tier3_end] if tier3_end > tier2_end else []
        tier4_players = players[tier3_end:tier4_end] if tier4_end > tier3_end else []
        
        analysis_text = ""
        
        # Analyze each tier
        for tier_num, tier_players in enumerate([tier1_players, tier2_players, tier3_players, tier4_players], 1):
            if not tier_players:
                continue
                
            placeholders = ','.join(['?' for _ in tier_players])
            cursor.execute(f'''
                SELECT adp_overall 
                FROM adp_data 
                WHERE season = 2025 AND player_name IN ({placeholders})
                AND adp_overall IS NOT NULL
            ''', tier_players)
            
            adp_results = cursor.fetchall()
            if adp_results:
                adps = [r[0] for r in adp_results]
                min_adp = min(adps)
                max_adp = max(adps)
                avg_adp = np.mean(adps)
                
                # Calculate rounds for 10 and 12 team leagues
                min_round_10 = calculate_round_number(min_adp, 10)
                max_round_10 = calculate_round_number(max_adp, 10)
                avg_round_10 = calculate_round_number(avg_adp, 10)
                
                min_round_12 = calculate_round_number(min_adp, 12)
                max_round_12 = calculate_round_number(max_adp, 12)
                avg_round_12 = calculate_round_number(avg_adp, 12)
                
                # Format round ranges
                if min_round_10 == max_round_10:
                    round_str_10 = f"Rd {min_round_10}"
                else:
                    round_str_10 = f"Rds {min_round_10}-{max_round_10}"
                    
                if min_round_12 == max_round_12:
                    round_str_12 = f"Rd {min_round_12}"
                else:
                    round_str_12 = f"Rds {min_round_12}-{max_round_12}"
                
                analysis_text += f"• **Tier {tier_num}**: {round_str_10} (10tm), {round_str_12} (12tm)\n"
                
                # Add strategic insight based on tier and position
                if tier_num == 1:
                    if position in ['rb', 'te']:
                        analysis_text += f"  → Must target early if you want elite {position.upper()}\n"
                    elif position == 'qb':
                        analysis_text += f"  → Early QB investment, but can wait for value\n"
                elif tier_num == 2:
                    if position == 'qb':
                        analysis_text += f"  → Sweet spot for QB value\n"
                    elif position == 'wr':
                        analysis_text += f"  → Solid WR1/2 options\n"
                elif tier_num == 3:
                    if position in ['rb', 'wr']:
                        analysis_text += f"  → Flex/depth options\n"
                    elif position == 'te':
                        analysis_text += f"  → Streaming territory\n"
                        
        conn.close()
        return analysis_text.strip()
        
    except Exception as e:
        return f"Error analyzing ADP: {str(e)}"


def get_position_insights(position: str, results: list) -> str:
    """Get position-specific draft strategy insights"""
    insights = {
        'qb': """
**QB Strategy:**
• Large gap between elite QBs and middle tier
• Many streamable options in lower tiers  
• Consider waiting unless you get a top-tier QB
• Rushing QBs provide higher floor/ceiling
        """,
        'rb': """
**RB Strategy:**
• Massive drop-offs between tiers - RB scarcity is real
• Elite RBs are extremely valuable due to large gaps
• Tier 1-2 RBs should be prioritized in early rounds
• Handcuffs become important for injury protection
        """,
        'wr': """
**WR Strategy:**
• More consistent value across tiers than RB
• Deep position with multiple viable tiers
• Can find value in middle rounds
• Target volume and target share over big plays
        """,
        'te': """
**TE Strategy:**
• Huge scarcity after elite tier - "Zero TE" strategy viable
• Either draft elite TE early or wait very late
• Middle tiers offer poor value relative to other positions
• Streaming difficult due to inconsistency
        """,
        'k': """
**K Strategy:**
• Minimal differences between tiers
• Wait until final rounds - no need to reach
• Target high-volume offenses and dome teams
• Streaming is highly effective
        """,
        'dst': """
**DST Strategy:**
• Small gaps between tiers - highly streamable
• Matchups matter more than individual unit quality
• Wait until final rounds
• Consider schedule strength for playoffs
        """
    }
    
    return insights.get(position, "")


async def run_team_intelligence_workflow(agent):
    """Run comprehensive team position intelligence gathering workflow"""
    import sqlite3
    import json
    from datetime import datetime
    from rich.progress import Progress, TaskID, track
    from rich.panel import Panel
    
    # NFL teams and positions to analyze
    nfl_teams = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
        'DET', 'GB', 'HOU', 'IND', 'JAC', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
        'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ]
    positions = ['qb', 'rb', 'wr', 'te', 'def']
    
    total_tasks = len(nfl_teams) * len(positions)
    console.print(f"[bright_green]Starting intelligence gathering for {len(nfl_teams)} teams × {len(positions)} positions = {total_tasks} analyses[/bright_green]")
    
    # Database setup
    db_path = "data/fantasy_ppr.db"
    
    # Create the table if it doesn't exist
    try:
        await create_intelligence_table(db_path)
        console.print("[dim]Database table verified/created[/dim]")
    except Exception as e:
        console.print(f"[red]Error setting up database: {e}[/red]")
        return
    
    success_count = 0
    error_count = 0
    
    with Progress() as progress:
        task = progress.add_task("[bright_green]Analyzing teams...", total=total_tasks)
        
        for team in nfl_teams:
            for position in positions:
                try:
                    console.print(f"\n[dim]Analyzing {team} {position.upper()}...[/dim]")
                    
                    # Step 1: Claude Sonnet 4 with web search analysis
                    intelligence_data = await gather_position_intelligence(agent, team, position)
                    
                    if intelligence_data:
                        # Step 2: Perplexity fact-checking
                        fact_check_result = await fact_check_with_perplexity(agent, team, position, intelligence_data)
                        
                        # Step 3: Save to database
                        await save_intelligence_to_db(db_path, team, position, intelligence_data, fact_check_result)
                        
                        success_count += 1
                        console.print(f"[bright_green]✓ {team} {position.upper()} completed[/bright_green]")
                    else:
                        error_count += 1
                        console.print(f"[red]✗ {team} {position.upper()} failed - no data returned[/red]")
                        
                except Exception as e:
                    error_count += 1
                    console.print(f"[red]✗ {team} {position.upper()} failed: {str(e)[:100]}[/red]")
                
                progress.update(task, advance=1)
    
    # Summary
    console.print(f"\n[bold bright_green]Intelligence Update Complete![/bold bright_green]")
    console.print(f"[bright_green]Successful: {success_count}[/bright_green]")
    console.print(f"[red]Errors: {error_count}[/red]")
    console.print(f"[dim]Data stored in: {db_path}[/dim]")


async def create_intelligence_table(db_path: str):
    """Create the team position intelligence table if it doesn't exist"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_position_intelligence (
            team TEXT,
            position TEXT,
            season INTEGER,
            last_updated TEXT,
            intelligence_summary TEXT,
            key_players TEXT,
            usage_notes TEXT,
            coaching_style TEXT,
            injury_updates TEXT,
            recent_changes TEXT,
            fact_check_status TEXT DEFAULT 'pending',
            fact_check_notes TEXT,
            confidence_score REAL,
            PRIMARY KEY (team, position, season)
        )
    ''')
    
    conn.commit()
    conn.close()


async def gather_position_intelligence(agent, team: str, position: str) -> dict:
    """Use Claude Sonnet 4 with web search to gather position intelligence"""
    try:
        # Force Claude Sonnet 4 for this analysis
        original_model = getattr(agent.model_manager, 'current_model', None)
        agent.model_manager.set_model('claude-sonnet-4')
        
        prompt = f"""Analyze the {team} {position.upper()} position group for the 2025 NFL season. Use web search to find the most current information.

ANALYSIS REQUIREMENTS:
1. **Key Players**: Who are the main players in this position group?
2. **Usage Patterns**: Expected snap counts, target share, carries, etc.
3. **Coaching Style**: How does this team utilize this position?
4. **Injury Updates**: Any current injuries or concerns?
5. **Recent Changes**: New signings, trades, cuts, or scheme changes?
6. **Fantasy Impact**: What does this mean for fantasy football?

WEB SEARCH FOCUS:
- 2025 depth charts and roster moves
- Recent beat reporter updates
- Coaching staff quotes about usage
- Preseason performance and usage
- Injury reports and training camp news

Please provide factual, recent information only. Include sources when possible.
"""
        
        # This will use Claude's native web search capability
        response = await agent.model_manager.complete(
            messages=[{"role": "user", "content": prompt}],
            task_type="research",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Restore original model
        if original_model:
            agent.model_manager.set_model(original_model)
        
        # Parse the response into structured data
        intelligence_data = {
            'raw_analysis': response,
            'key_players': extract_key_players(response),
            'usage_notes': extract_usage_notes(response),
            'coaching_style': extract_coaching_style(response),
            'injury_updates': extract_injury_updates(response),
            'recent_changes': extract_recent_changes(response),
            'fantasy_impact': extract_fantasy_impact(response)
        }
        
        return intelligence_data
        
    except Exception as e:
        console.print(f"[red]Error in Claude analysis for {team} {position}: {e}[/red]")
        return None


async def fact_check_with_perplexity(agent, team: str, position: str, intelligence_data: dict) -> dict:
    """Use Perplexity Pro to fact-check the Claude analysis"""
    try:
        # Force Perplexity Pro for fact-checking
        original_model = getattr(agent.model_manager, 'current_model', None)
        agent.model_manager.set_model('perplexity-sonar-pro')
        
        # Create fact-checking prompt
        claims_to_check = f"""
Key Claims from Analysis:
- Key Players: {intelligence_data.get('key_players', 'N/A')}
- Usage Notes: {intelligence_data.get('usage_notes', 'N/A')}
- Injury Updates: {intelligence_data.get('injury_updates', 'N/A')}
- Recent Changes: {intelligence_data.get('recent_changes', 'N/A')}
"""
        
        fact_check_prompt = f"""Fact-check these claims about the {team} {position.upper()} position group for 2025:

{claims_to_check}

Please verify:
1. Are the key players correctly identified?
2. Are injury reports accurate and current?
3. Are recent roster moves/trades correct?
4. Are usage patterns realistic based on team history?

Provide confidence scores (0-100) for each major claim and flag any inaccuracies."""
        
        fact_check_response = await agent.model_manager.complete(
            messages=[{"role": "user", "content": fact_check_prompt}],
            task_type="research", 
            temperature=0.1,
            max_tokens=1000
        )
        
        # Restore original model
        if original_model:
            agent.model_manager.set_model(original_model)
            
        return {
            'fact_check_response': fact_check_response,
            'confidence_score': extract_confidence_score(fact_check_response),
            'status': 'verified' if 'accurate' in fact_check_response.lower() else 'flagged'
        }
        
    except Exception as e:
        console.print(f"[red]Error in Perplexity fact-check for {team} {position}: {e}[/red]")
        return {'status': 'error', 'fact_check_response': f'Error: {e}', 'confidence_score': 0}


async def save_intelligence_to_db(db_path: str, team: str, position: str, intelligence_data: dict, fact_check_result: dict):
    """Save the intelligence data to the database"""
    import sqlite3
    import json
    from datetime import datetime
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prepare data for insertion
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT OR REPLACE INTO team_position_intelligence (
            team, position, season, last_updated, intelligence_summary,
            key_players, usage_notes, coaching_style, injury_updates,
            recent_changes, fact_check_status, fact_check_notes, confidence_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        team, position, 2025, timestamp,
        intelligence_data.get('raw_analysis', ''),
        intelligence_data.get('key_players', ''),
        intelligence_data.get('usage_notes', ''),
        intelligence_data.get('coaching_style', ''),
        intelligence_data.get('injury_updates', ''),
        intelligence_data.get('recent_changes', ''),
        fact_check_result.get('status', 'pending'),
        fact_check_result.get('fact_check_response', ''),
        fact_check_result.get('confidence_score', 0)
    ))
    
    conn.commit()
    conn.close()


def extract_key_players(response: str) -> str:
    """Extract key players from Claude response"""
    # Simple extraction - look for player names (could be enhanced with NER)
    import re
    lines = response.split('\n')
    key_players = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['key player', 'starter', 'depth chart', 'rb1', 'wr1', 'qb1']):
            # Extract potential player names (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', line)
            key_players.extend(names)
    
    return ', '.join(set(key_players[:5]))  # Top 5 unique names


def extract_usage_notes(response: str) -> str:
    """Extract usage/snap count information"""
    lines = response.split('\n')
    usage_info = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['snap', 'target', 'usage', 'carry', 'touch', 'percent']):
            usage_info.append(line.strip())
    
    return '\n'.join(usage_info[:3])  # Top 3 usage notes


def extract_coaching_style(response: str) -> str:
    """Extract coaching philosophy information"""
    lines = response.split('\n')
    coaching_info = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['coach', 'scheme', 'offense', 'defense', 'system', 'philosophy']):
            coaching_info.append(line.strip())
    
    return '\n'.join(coaching_info[:2])  # Top 2 coaching notes


def extract_injury_updates(response: str) -> str:
    """Extract injury information"""
    lines = response.split('\n')
    injury_info = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['injury', 'injured', 'hurt', 'questionable', 'doubtful', 'ir']):
            injury_info.append(line.strip())
    
    return '\n'.join(injury_info[:3])  # Top 3 injury notes


def extract_recent_changes(response: str) -> str:
    """Extract recent roster moves"""
    lines = response.split('\n')
    changes_info = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['trade', 'sign', 'cut', 'release', 'draft', 'acquire', 'new']):
            changes_info.append(line.strip())
    
    return '\n'.join(changes_info[:3])  # Top 3 recent changes


def extract_fantasy_impact(response: str) -> str:
    """Extract fantasy football implications"""
    lines = response.split('\n')
    fantasy_info = []
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['fantasy', 'draft', 'waiver', 'start', 'sit', 'value']):
            fantasy_info.append(line.strip())
    
    return '\n'.join(fantasy_info[:3])  # Top 3 fantasy notes


def extract_confidence_score(response: str) -> float:
    """Extract confidence score from fact-check response"""
    import re
    
    # Look for confidence percentages
    confidence_matches = re.findall(r'(\d+)%|confidence[:\s]*(\d+)', response.lower())
    
    if confidence_matches:
        # Get the highest confidence score mentioned
        scores = []
        for match in confidence_matches:
            for group in match:
                if group:
                    scores.append(int(group))
        
        return max(scores) if scores else 75.0
    
    # Default confidence if none specified
    return 75.0


def handle_intel_command(command: str):
    """Handle intelligence queries like /intel/team/position"""
    try:
        parts = command.split('/')
        
        if len(parts) == 1:
            # Just /intel - show help
            console.print("[bright_green]Team Position Intelligence System[/bright_green]")
            console.print("\n[dim]Usage examples:[/dim]")
            console.print("  [bright_green]/intel/PHI/wr[/bright_green] - Get Eagles WR intelligence")
            console.print("  [bright_green]/intel/SF/rb[/bright_green] - Get 49ers RB intelligence")
            console.print("  [bright_green]/intel/list[/bright_green] - Show available team/position combinations")
            console.print("\n[dim]Available positions: qb, rb, wr, te, def[/dim]")
            
        elif len(parts) == 2 and parts[1].lower() == 'list':
            # Show available intelligence data
            show_available_intelligence()
            
        elif len(parts) == 3:
            # /intel/team/position - show specific intelligence
            team = parts[1].upper()
            position = parts[2].lower()
            show_team_position_intelligence(team, position)
            
        else:
            console.print(f"[red]Error: Invalid intel command format. Use /intel/team/position[/red]")
            
    except Exception as e:
        console.print(f"[red]Error processing intel command: {str(e)}[/red]")


def show_available_intelligence():
    """Show what intelligence data is available"""
    try:
        import sqlite3
        from rich.table import Table
        
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_position_intelligence'")
        if not cursor.fetchone():
            console.print("[red]No intelligence data found. Run '/update intelligence' first.[/red]")
            conn.close()
            return
        
        # Get summary of available data
        cursor.execute('''
            SELECT team, position, last_updated, fact_check_status, confidence_score
            FROM team_position_intelligence 
            WHERE season = 2025
            ORDER BY team, position
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            console.print("[red]No intelligence data found for 2025 season. Run '/update intelligence' first.[/red]")
            return
        
        # Create summary table
        table = Table(title="Available Team Position Intelligence", title_style="bold bright_green")
        table.add_column("Team", style="bright_green", width=6)
        table.add_column("Position", style="bright_green", width=8)
        table.add_column("Last Updated", style="dim", width=12)
        table.add_column("Status", style="white", width=10)
        table.add_column("Confidence", style="yellow", width=12)
        
        for row in results:
            team, position, last_updated, status, confidence = row
            
            # Format date
            try:
                from datetime import datetime
                date_obj = datetime.fromisoformat(last_updated)
                formatted_date = date_obj.strftime("%m/%d %H:%M")
            except:
                formatted_date = last_updated[:10]
            
            # Color code status
            if status == 'verified':
                status_display = f"[bright_green]{status}[/bright_green]"
            elif status == 'flagged':
                status_display = f"[red]{status}[/red]"
            else:
                status_display = f"[yellow]{status}[/yellow]"
            
            confidence_display = f"{confidence:.0f}%" if confidence else "N/A"
            
            table.add_row(
                team,
                position.upper(),
                formatted_date,
                status_display,
                confidence_display
            )
        
        console.print(table)
        console.print(f"\n[dim]Use /intel/team/position to view detailed analysis (e.g., /intel/PHI/wr)[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error showing intelligence list: {str(e)}[/red]")


def show_team_position_intelligence(team: str, position: str):
    """Show detailed intelligence for specific team/position"""
    try:
        import sqlite3
        from rich.panel import Panel
        from rich.columns import Columns
        
        db_path = "data/fantasy_ppr.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get intelligence data
        cursor.execute('''
            SELECT intelligence_summary, key_players, usage_notes, coaching_style,
                   injury_updates, recent_changes, fact_check_status, fact_check_notes,
                   confidence_score, last_updated
            FROM team_position_intelligence 
            WHERE team = ? AND position = ? AND season = 2025
        ''', (team, position))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            console.print(f"[red]No intelligence data found for {team} {position.upper()}. Try '/intel/list' to see available data.[/red]")
            return
        
        (intelligence_summary, key_players, usage_notes, coaching_style,
         injury_updates, recent_changes, fact_check_status, fact_check_notes,
         confidence_score, last_updated) = result
        
        # Display intelligence in organized panels
        console.print(f"\n[bold bright_green]{team} {position.upper()} Intelligence Report[/bold bright_green]")
        
        # Status info
        status_color = "bright_green" if fact_check_status == 'verified' else "yellow" if fact_check_status == 'pending' else "red"
        console.print(f"[dim]Status: [{status_color}]{fact_check_status}[/{status_color}] | Confidence: {confidence_score:.0f}% | Updated: {last_updated[:10]}[/dim]")
        
        # Main analysis
        if intelligence_summary:
            console.print(Panel(intelligence_summary, title="📊 Analysis Summary", border_style="bright_green"))
        
        # Key details in columns
        panels = []
        
        if key_players:
            panels.append(Panel(key_players, title="👥 Key Players", border_style="blue"))
        
        if usage_notes:
            panels.append(Panel(usage_notes, title="📈 Usage Patterns", border_style="yellow"))
        
        if coaching_style:
            panels.append(Panel(coaching_style, title="🎯 Coaching Style", border_style="green"))
        
        if injury_updates:
            panels.append(Panel(injury_updates, title="🏥 Injury Updates", border_style="red"))
        
        if recent_changes:
            panels.append(Panel(recent_changes, title="🔄 Recent Changes", border_style="cyan"))
        
        # Display panels in columns (2 per row)
        if panels:
            for i in range(0, len(panels), 2):
                if i + 1 < len(panels):
                    console.print(Columns([panels[i], panels[i + 1]]))
                else:
                    console.print(panels[i])
        
        # Fact-check notes if available
        if fact_check_notes and fact_check_status != 'pending':
            console.print(Panel(fact_check_notes, title="✅ Fact-Check Results", border_style="dim"))
        
        console.print(f"\n[dim]Last updated: {last_updated}[/dim]")
        console.print(f"[dim]Use '/update intelligence' to refresh all team data[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error showing intelligence for {team} {position}: {str(e)}[/red]")


def handle_bye_week_command(position: str, parts: list):
    """Handle bye week queries like /wr/bye, /qb/bye/5, etc."""
    try:
        # Parse optional bye week filter: /position/bye[/week]
        week_filter = None
        if len(parts) > 2:
            try:
                week_filter = int(parts[2])
                if week_filter < 1 or week_filter > 18:
                    console.print(f"[red]Invalid week: {week_filter}. Must be between 1-18[/red]")
                    return
            except ValueError:
                console.print(f"[red]Invalid week: {parts[2]}. Must be a number (1-18)[/red]")
                return
        
        # Connect to database
        import sqlite3
        from pathlib import Path
        
        db_path = Path(__file__).parent / "data" / "fantasy_ppr.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        position_db = position.lower() if position not in ['DEF', 'DST'] else 'dst'
        
        if week_filter:
            # Show players with specific bye week
            query = """
                SELECT player_name, team_name, bye_week
                FROM player_projections 
                WHERE season = 2025 AND position = ? AND bye_week = ?
                ORDER BY player_name
            """
            cursor.execute(query, (position_db, week_filter))
            title = f"{position}s with Week {week_filter} Bye"
        else:
            # Show all players with their bye weeks
            query = """
                SELECT player_name, team_name, bye_week
                FROM player_projections 
                WHERE season = 2025 AND position = ? AND bye_week IS NOT NULL
                ORDER BY bye_week, player_name
            """
            cursor.execute(query, (position_db,))
            title = f"All {position}s - 2025 Bye Weeks"
        
        results = cursor.fetchall()
        
        if not results:
            if week_filter:
                console.print(f"[yellow]No {position}s found with Week {week_filter} bye[/yellow]")
            else:
                console.print(f"[yellow]No bye week data found for {position}s[/yellow]")
            return
        
        # Display results
        from rich.table import Table
        table = Table(title=title, show_header=True, header_style="bold bright_green")
        table.add_column("Rank", style="bright_green", width=6)
        table.add_column("Player", style="bright_green", width=25)
        table.add_column("Team", style="bright_green", width=6)
        table.add_column("Bye Week", style="bright_green", width=10)
        
        for i, row in enumerate(results, 1):
            table.add_row(
                str(i),
                row[0],  # player_name
                row[1] or "FA",  # team_name
                f"Week {int(row[2])}" if row[2] else "N/A"  # bye_week
            )
        
        console.print(table)
        
        # Show bye week summary if showing all players
        if not week_filter and results:
            console.print(f"\n[bold bright_green]Bye Week Summary:[/bold bright_green]")
            
            # Count players by bye week
            bye_summary = {}
            for row in results:
                week = int(row[2]) if row[2] else 0
                bye_summary[week] = bye_summary.get(week, 0) + 1
            
            for week in sorted(bye_summary.keys()):
                if week > 0:
                    console.print(f"  [bright_green]Week {week}:[/bright_green] {bye_summary[week]} players")
        
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error fetching bye weeks: {e}[/red]")


if __name__ == '__main__':
    cli()