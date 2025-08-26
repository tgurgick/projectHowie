#!/usr/bin/env python3
"""
Howie CLI - Claude-like Fantasy Football AI Assistant
Main entry point for the enhanced CLI tool
"""

import os
import sys
import asyncio
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from howie_cli.core.agent import HowieAgent
from howie_cli.core.context import ConversationContext
from howie_cli.tools.registry import global_registry

console = Console()

# Version
__version__ = "2.0.0"


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, version):
    """Howie - Claude-like Fantasy Football AI Assistant
    
    An intelligent CLI tool that provides comprehensive fantasy football
    analysis with file operations, visualizations, code generation,
    real-time data, and ML predictions.
    """
    if version:
        console.print(f"Howie CLI v{__version__}")
        return
    
    if ctx.invoked_subcommand is None:
        # Default to chat mode
        ctx.invoke(chat)


@cli.command()
@click.option('--model', default='gpt-4o', help='OpenAI model to use')
@click.option('--resume', is_flag=True, help='Resume previous session')
@click.option('--session-id', help='Specific session ID to resume')
def chat(model, resume, session_id):
    """Start interactive chat mode (default)"""
    try:
        # Initialize agent
        agent = HowieAgent(model=model)
        
        # Resume session if requested
        if resume or session_id:
            try:
                if session_id:
                    agent.context = ConversationContext.load_session(session_id)
                    console.print(f"[green]Resumed session: {session_id}[/green]")
                else:
                    # Load most recent session
                    sessions_dir = Path.home() / ".howie" / "sessions"
                    if sessions_dir.exists():
                        sessions = list(sessions_dir.glob("*.pkl"))
                        if sessions:
                            latest = max(sessions, key=lambda p: p.stat().st_mtime)
                            session_id = latest.stem
                            agent.context = ConversationContext.load_session(session_id)
                            console.print(f"[green]Resumed latest session: {session_id}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not resume session: {e}[/yellow]")
                console.print("[yellow]Starting new session...[/yellow]")
        
        # Start chat loop
        asyncio.run(agent.chat_loop())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--model', default='gpt-4o', help='OpenAI model to use')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', type=click.Choice(['text', 'json', 'markdown']), default='text')
def ask(query, model, output, format):
    """Ask a single question without entering chat mode"""
    try:
        agent = HowieAgent(model=model)
        
        # Process query
        response = asyncio.run(agent.process_message(query))
        
        # Format output
        if format == 'json':
            output_data = {
                "query": query,
                "response": response,
                "session": agent.get_session_summary()
            }
            formatted = json.dumps(output_data, indent=2)
        elif format == 'markdown':
            formatted = f"# Query\n{query}\n\n# Response\n{response}"
        else:
            formatted = response
        
        # Output results
        if output:
            with open(output, 'w') as f:
                f.write(formatted)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            console.print(formatted)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('file_path')
@click.option('--platform', type=click.Choice(['espn', 'yahoo', 'sleeper', 'generic']), 
              default='generic', help='Platform format')
def import_roster(file_path, platform):
    """Import a fantasy roster from CSV/Excel file"""
    try:
        agent = HowieAgent()
        
        # Import roster
        query = f"Import roster from {file_path} (platform: {platform})"
        response = asyncio.run(agent.process_message(query))
        
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error importing roster: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('players', nargs=-1, required=True)
@click.option('--metrics', '-m', multiple=True, help='Metrics to compare')
@click.option('--visual', is_flag=True, help='Generate visual comparison')
def compare(players, metrics, visual):
    """Compare multiple players"""
    try:
        agent = HowieAgent()
        
        # Build query
        player_list = ", ".join(players)
        query = f"Compare players: {player_list}"
        
        if metrics:
            query += f" using metrics: {', '.join(metrics)}"
        
        if visual:
            query += " and create a comparison chart"
        
        # Process
        response = asyncio.run(agent.process_message(query))
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('player')
@click.option('--weeks', default=1, help='Weeks to project ahead')
@click.option('--method', type=click.Choice(['linear', 'ml', 'ensemble']), 
              default='ensemble', help='Projection method')
def project(player, weeks, method):
    """Generate player projections"""
    try:
        agent = HowieAgent()
        
        query = f"Generate {weeks} week projection for {player} using {method} method"
        response = asyncio.run(agent.process_message(query))
        
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--roster-file', help='File containing your roster')
@click.option('--method', type=click.Choice(['points', 'safety', 'balanced']), 
              default='balanced', help='Optimization method')
@click.option('--week', type=int, help='Week to optimize for')
def optimize(roster_file, method, week):
    """Optimize your fantasy lineup"""
    try:
        agent = HowieAgent()
        
        query = f"Optimize lineup using {method} method"
        
        if roster_file:
            query = f"Import roster from {roster_file} and " + query
        
        if week:
            query += f" for week {week}"
        
        response = asyncio.run(agent.process_message(query))
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--week', type=int, help='NFL week number')
@click.option('--team', help='Specific team to track')
def live(week, team):
    """Get live scores and updates"""
    try:
        agent = HowieAgent()
        
        query = "Get live scores"
        if week:
            query += f" for week {week}"
        if team:
            query += f" for {team}"
        
        response = asyncio.run(agent.process_message(query))
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('requirements')
@click.option('--output', '-o', help='Output file name')
@click.option('--type', 'code_type', 
              type=click.Choice(['script', 'sql', 'analysis']), 
              default='script', help='Type of code to generate')
def generate(requirements, output, code_type):
    """Generate analysis code or queries"""
    try:
        agent = HowieAgent()
        
        if code_type == 'sql':
            query = f"Generate SQL query for: {requirements}"
        elif code_type == 'analysis':
            query = f"Generate analysis script for: {requirements}"
        else:
            query = f"Generate Python script for: {requirements}"
        
        if output:
            query += f" and save to {output}"
        
        response = asyncio.run(agent.process_message(query))
        console.print(response)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def tools():
    """List all available tools"""
    global_registry.display_tools()


@cli.command()
def sessions():
    """List saved sessions"""
    sessions_dir = Path.home() / ".howie" / "sessions"
    
    if not sessions_dir.exists():
        console.print("[yellow]No saved sessions found[/yellow]")
        return
    
    sessions = list(sessions_dir.glob("*.pkl"))
    
    if not sessions:
        console.print("[yellow]No saved sessions found[/yellow]")
        return
    
    table = Table(title="Saved Sessions", show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Size", style="yellow")
    
    for session_file in sorted(sessions, key=lambda p: p.stat().st_mtime, reverse=True):
        session_id = session_file.stem
        created = Path(session_file).stat().st_mtime
        size = session_file.stat().st_size / 1024  # KB
        
        from datetime import datetime
        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(session_id, created_str, f"{size:.1f} KB")
    
    console.print(table)


@cli.command()
def workspace():
    """Show workspace information"""
    try:
        from howie_cli.core.workspace import WorkspaceManager
        ws = WorkspaceManager()
        info = ws.get_workspace_info()
        
        table = Table(title="Workspace Information", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in info.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.confirmation_option(prompt='Clear all workspace files?')
def clear():
    """Clear workspace and temporary files"""
    try:
        from howie_cli.core.workspace import WorkspaceManager
        ws = WorkspaceManager()
        ws.cleanup(keep_reports=False)
        console.print("[green]Workspace cleared![/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    cli()