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
__version__ = "2.1.0"


def show_eagles_intro():
    """Display intro screen"""
    console.print("\n" * 2)
    
    # Title with subtle Eagles branding
    title = Text("HOWIE", style="bold bright_green")
    subtitle = Text("Fantasy Football AI Assistant", style="green")
    tagline = Text("🦅 Powered by Claude Sonnet 4 🦅", style="bright_green")
    
    # Center the title
    console.print(Align.center(title))
    console.print(Align.center(subtitle))
    console.print(Align.center(tagline))
    console.print("\n")
    
    # Welcome panel
    welcome_text = (
        "Welcome to HOWIE, your fantasy football assistant!\n\n"
        "Powered by Claude Sonnet 4 • Real-time data • Expert analysis"
    )
    
    console.print(Panel.fit(
        welcome_text,
        title="🏈 WELCOME 🏈",
        border_style="bright_green"
    ))
    console.print("\n")


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, version):
    """🦅 HOWIE - Fantasy Football AI Assistant
    
    Powered by Claude Sonnet 4 with multi-model support 🏈
    
    Use different AI models optimized for different tasks:
    - Claude Sonnet 4 for analysis and strategy
    - Perplexity for real-time research
    - GPT-4o for complex reasoning
    - Fast models for quick lookups
    """
    if version:
        console.print(f"[bright_green]🦅 Howie Enhanced CLI v{__version__}[/bright_green]")
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
            f"Type 'model:info' to see all models, 'model:switch <name>' to change\n"
            f"Type 'help' for commands • Type 'quit' to exit",
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
                console.print(f"[dim yellow]Could not resume session: {e}[/dim yellow]")
        
        # Start enhanced chat loop
        asyncio.run(enhanced_chat_loop(agent))
        
    except KeyboardInterrupt:
        console.print("\n[bright_green]🦅 Goodbye![/bright_green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


async def enhanced_chat_loop(agent: EnhancedHowieAgent):
    """Enhanced chat loop with model commands"""
    
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold bright_green]🏈 You:[/bold bright_green] ")
            
            # Check for special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("[bright_green]🦅 Good luck with your fantasy team![/bright_green]")
                break
            
            elif user_input.lower() == 'help':
                show_enhanced_help()
                continue
            
            # Model-specific commands
            elif user_input.lower().startswith('model:'):
                handle_model_command(agent, user_input[6:])
                continue
            
            # Model override syntax: @model <query>
            elif user_input.startswith('@'):
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    model_name = parts[0][1:]  # Remove @
                    query = parts[1]
                    
                    if model_name in agent.model_manager.models:
                        console.print(f"[green]Using {model_name} for this query[/green]")
                        response = await agent.process_with_model(query, model_name)
                    else:
                        console.print(f"[red]Unknown model: {model_name}[/red]")
                        continue
                else:
                    console.print("[red]Usage: @model_name your question[/red]")
                    continue
            else:
                # Normal query - use automatic model selection
                response = await agent.process_message(user_input)
            
            # Display response
            console.print("\n[bold bright_green]🦅 Howie:[/bold bright_green]")
            from rich.markdown import Markdown
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            console.print("\n[dim yellow]Use 'quit' to exit properly[/dim yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


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
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Provider", style="green", width=12)
        table.add_column("Tier", style="yellow", width=10)
        table.add_column("Cost (I/O per 1K)", style="red", width=15)
        table.add_column("Best For", style="white", width=30)
        
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
            console.print(f"  [green]{task}[/green]: {model}")
        
        # Usage stats
        if info['usage']['by_model']:
            console.print("\n[bold bright_green]Usage Statistics:[/bold bright_green]")
            console.print(f"Total Cost: [green]${info['usage']['total_cost']:.4f}[/green]")
            for model, stats in info['usage']['by_model'].items():
                console.print(f"  {model}: {stats['calls']} calls, [green]${stats.get('cost', 0):.4f}[/green]")
    
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
        console.print("[yellow]Unknown model command. Try 'model:info' or 'model:switch <name>'[/yellow]")


def show_enhanced_help():
    """Show enhanced help with model commands"""
    help_text = """
# Enhanced Commands

## Model Management
- **model:info** - Show available models and usage
- **model:switch <name>** - Switch to a different model
- **model:config <task> <model>** - Configure model for task type
- **model:save** - Save model configuration
- **@model_name <query>** - Use specific model for one query

## Model Selection Examples
- **@perplexity-sonar** Who won the NFL games yesterday?
- **@claude-3-5-sonnet** Generate a Python script for analysis
- **@gpt-4o-mini** List all QBs on the Cowboys
- **@claude-3-opus** Complex analysis of playoff scenarios

## Automatic Model Selection
The system automatically chooses the best model based on your query:
- Research queries → Perplexity
- Code generation → Claude Sonnet
- Complex analysis → GPT-4o or Claude Opus
- Simple queries → GPT-4o-mini or Claude Haiku

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
        console.print("[bright_green]🦅 Updating NFL rosters...[/bright_green]")
        
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
                console.print("\n[bold bright_green]🔄 Team Changes Detected:[/bold bright_green]")
                for change in team_changes:
                    console.print(f"  • [bold white]{change['player_name']}[/bold white] ([green]{change['position']}[/green]) [red]{change['old_team']}[/red] → [bright_green]{change['new_team']}[/bright_green]")
            else:
                console.print("\n[bright_green]✅ No team changes detected[/bright_green]")
                
        except ImportError:
            console.print("[red]Roster update script not found. Please check scripts/update_rosters.py[/red]")
        except Exception as e:
            console.print(f"[red]Error updating rosters: {e}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def models():
    """List all available models with details"""
    try:
        manager = ModelManager()
        
        # Create detailed table
        table = Table(title="Available AI Models", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Provider", style="green", width=12)
        table.add_column("Tier", style="yellow", width=10)
        table.add_column("Input $/1K", style="red", width=10)
        table.add_column("Output $/1K", style="red", width=10)
        table.add_column("Tools", style="blue", width=8)
        table.add_column("Vision", style="blue", width=8)
        table.add_column("Best For", style="white", width=35)
        
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
            console.print(f"\n[cyan]{task}[/cyan] (current: {current})")
            console.print("Available models:", ", ".join(models))
            
            choice = console.input("Select model (or Enter to keep current): ").strip()
            if choice and choice in models:
                manager.set_task_model(task, choice)
                console.print(f"[green]Set {task} → {choice}[/green]")
        
        # Set default model
        current_default = manager.current_model
        console.print(f"\n[cyan]Default model[/cyan] (current: {current_default})")
        choice = console.input("Select default model (or Enter to keep current): ").strip()
        if choice and choice in models:
            manager.set_model(choice)
            console.print(f"[green]Set default → {choice}[/green]")
        
        # Save configuration
        if save:
            save_path = Path(save)
        else:
            save_path = Path.home() / ".howie" / "models.json"
        
        manager.save_config(save_path)
        console.print(f"\n[green]Configuration saved to {save_path}[/green]")
        
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
                     show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Provider", style="green", width=12)
        table.add_column("Input Cost", style="yellow", width=12)
        table.add_column("Output Cost", style="yellow", width=12)
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
            console.print(f"\n[green]Cheapest: {costs[0][0]} (${costs[0][4]:.4f})[/green]")
            console.print(f"[red]Most expensive: {costs[-1][0]} (${costs[-1][4]:.4f})[/red]")
            console.print(f"[yellow]Difference: ${costs[-1][4] - costs[0][4]:.4f} ({(costs[-1][4] / costs[0][4] - 1) * 100:.1f}% more)[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    cli()