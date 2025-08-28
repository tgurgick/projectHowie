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
__version__ = "2.2.0"





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
                    console.print("  [bright_green]/qb/stats[/bright_green] - Show all QB stats available")
                    console.print("  [dim]More: /qb/pass_td, /rb/rush_yards, /wr/targets, /qb/fantasy[/dim]")
                    console.print("  [dim]Format: /position/stat/season (e.g., /qb/td/2024)[/dim]")
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
                    handle_update_command(agent, user_input[8:])  # Remove '/update' prefix
                    continue
                elif user_input.lower().startswith('/wr/') or user_input.lower().startswith('/qb/') or user_input.lower().startswith('/rb/') or user_input.lower().startswith('/te/'):
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


def handle_update_command(agent: EnhancedHowieAgent, command: str):
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
            args = Args()
            args.season = 2025
            args.scoring = 'ppr'
            args.test = False
            args.db_url = "sqlite:///data/fantasy_ppr.db"
            
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
    
    else:
        console.print("[yellow]Available update commands:[/yellow]")
        console.print("  [bright_green]/update adp[/bright_green] - Update ADP data from FantasyPros")
        console.print("  [bright_green]/update rosters[/bright_green] - Update NFL roster information")


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
            'Kicking Stats': ['fantasy']  # Kickers only have fantasy points
        }
    }
    
    # Get stats for the position
    if position not in position_stats:
        console.print(f"[red]Invalid position: {position}. Valid positions: QB, RB, WR, TE, K[/red]")
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
    if position in ['QB', 'RB', 'WR', 'TE']:
        console.print(f"\n[bold bright_green]ADP Data:[/bold bright_green]")
        console.print(f"  [bright_green]/{position.lower()}/adp[/bright_green] - Current ADP rankings")
        console.print(f"  [bright_green]/{position.lower()}/adp/2024[/bright_green] - Historical ADP")
    
    # Show season info
    console.print(f"\n[bold bright_green]Season Support:[/bold bright_green]")
    console.print(f"  Available seasons: 2018-2024")
    console.print(f"  Format: /{position.lower()}/stat/season (e.g., /{position.lower()}/fantasy/2024)")
    console.print(f"  Default: Current season (2025) if no season specified")


def handle_rapid_stats_command(command: str):
    """Handle rapid stats commands like /wr/adp, /qb/td/2024, etc."""
    try:
        # Parse command: position/stat[/season]
        parts = command.split('/')
        if len(parts) < 2:
            console.print("[red]Invalid format. Use: /position/stat[/season] (e.g., /wr/adp, /qb/td/2024)[/red]")
            return
        
        position = parts[0].upper()
        stat = parts[1].lower()
        season = parts[2] if len(parts) > 2 else "2025"  # Default to current season
        
        # Validate position
        valid_positions = ['WR', 'RB', 'QB', 'TE', 'K']
        if position not in valid_positions:
            console.print(f"[red]Invalid position: {position}. Valid positions: {', '.join(valid_positions)}[/red]")
            return
        
        # Validate season
        try:
            season = int(season)
        except ValueError:
            console.print(f"[red]Invalid season: {season}. Must be a year (e.g., 2024, 2025)[/red]")
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
            # Query ADP data from the new adp_data table
            query = """
                SELECT player_name, position, team, adp_overall, espn_adp, sleeper_adp, avg_adp
                FROM adp_data 
                WHERE season = ? AND scoring_format = 'ppr' AND position = ?
                ORDER BY adp_overall ASC
                LIMIT 50
            """
            cursor.execute(query, (season, position))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No ADP data found for {position}s in {season}[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            table = Table(title=f"Top 50 {position}s by ADP ({season})", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Player", style="bright_green", width=25)
            table.add_column("Team", style="bright_green", width=6)
            table.add_column("Overall ADP", style="bright_green", width=12)
            table.add_column("ESPN", style="bright_green", width=8)
            table.add_column("Sleeper", style="bright_green", width=8)
            table.add_column("AVG", style="bright_green", width=8)
            
            for i, row in enumerate(results, 1):
                table.add_row(
                    str(i),
                    row[0],  # player_name
                    row[2] or "FA",  # team
                    f"{row[3]:.1f}" if row[3] else "N/A",  # adp_overall
                    f"{row[4]:.0f}" if row[4] else "N/A",  # espn_adp
                    f"{row[5]:.0f}" if row[5] else "N/A",  # sleeper_adp
                    f"{row[6]:.0f}" if row[6] else "N/A"   # avg_adp
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
                'fumbles': 'fumbles'
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
            elif stat not in stat_mapping:
                console.print(f"[red]Invalid stat: {stat}. Valid stats: {', '.join(stat_mapping.keys())}[/red]")
                return
            else:
                # Build the query for regular stats
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
            
            cursor.execute(query, (season, position))
            results = cursor.fetchall()
            
            if not results:
                console.print(f"[yellow]No stats found for {position}s in {season}[/yellow]")
                return
            
            # Display results
            from rich.table import Table
            table = Table(title=f"Top 50 {position}s by {stat.upper()} ({season})", show_header=True, header_style="bold bright_green")
            table.add_column("Rank", style="bright_green", width=6)
            table.add_column("Player", style="bright_green", width=25)
            table.add_column("Team", style="bright_green", width=6)
            table.add_column(f"Total {stat.upper()}", style="bright_green", width=15)
            
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


if __name__ == '__main__':
    cli()