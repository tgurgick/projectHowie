#!/usr/bin/env python3
"""
Demonstration of Howie's agent capabilities
Similar to Claude's autonomous agent system
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Set up environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-here")

from howie_cli.core.agent import HowieAgent
from howie_cli.core.subagent import AgentManager, ResearchAgent, AnalysisAgent, WorkflowAgent
from howie_cli.tools.registry import global_registry

console = Console()


async def demo_single_agent():
    """Demonstrate a single autonomous agent"""
    console.print(Panel.fit(
        "[bold cyan]Demo 1: Single Autonomous Agent[/bold cyan]\n"
        "Spawning a research agent to investigate a topic independently",
        border_style="cyan"
    ))
    
    # Initialize Howie
    agent = HowieAgent()
    
    # Spawn a research agent
    console.print("\n[yellow]Spawning research agent...[/yellow]")
    response = await agent.process_message(
        "Spawn an agent to research: What factors most impact WR performance in cold weather games?"
    )
    
    console.print(f"\n[green]Agent Response:[/green]\n{response}")


async def demo_parallel_agents():
    """Demonstrate multiple agents working in parallel"""
    console.print(Panel.fit(
        "[bold cyan]Demo 2: Parallel Agents[/bold cyan]\n"
        "Multiple agents working simultaneously on different tasks",
        border_style="cyan"
    ))
    
    agent = HowieAgent()
    
    # Define parallel tasks
    tasks = [
        "Research Josh Allen's performance trends",
        "Analyze the Buffalo Bills offensive scheme",
        "Generate code to predict QB performance",
        "Optimize a lineup featuring Bills players"
    ]
    
    console.print("\n[yellow]Spawning 4 agents to work in parallel...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_id = progress.add_task("Agents working...", total=len(tasks))
        
        # This would use the parallel agents tool
        response = await agent.process_message(
            f"Run these tasks in parallel: {tasks}"
        )
        
        progress.update(task_id, completed=len(tasks))
    
    console.print(f"\n[green]Parallel Results:[/green]\n{response}")


async def demo_workflow_agent():
    """Demonstrate a workflow agent coordinating multiple sub-agents"""
    console.print(Panel.fit(
        "[bold cyan]Demo 3: Workflow Agent[/bold cyan]\n"
        "A master agent coordinating multiple specialized agents",
        border_style="cyan"
    ))
    
    agent = HowieAgent()
    
    workflow_task = """
    Complete a comprehensive analysis for my fantasy playoff decisions:
    1. Research the top playoff defenses and their impact on offensive players
    2. Analyze my current roster's playoff schedule
    3. Generate trade recommendations based on playoff matchups
    4. Create an optimal playoff lineup strategy
    """
    
    console.print("\n[yellow]Initiating complex workflow...[/yellow]")
    
    response = await agent.process_message(
        f"Execute this workflow: {workflow_task}"
    )
    
    console.print(f"\n[green]Workflow Complete:[/green]\n{response}")


async def demo_agent_capabilities():
    """Show different agent types and their capabilities"""
    console.print(Panel.fit(
        "[bold cyan]Agent Capabilities Overview[/bold cyan]",
        border_style="cyan"
    ))
    
    # Create table of agent types
    table = Table(title="Available Agent Types", show_header=True, header_style="bold magenta")
    table.add_column("Agent Type", style="cyan", width=15)
    table.add_column("Purpose", style="green", width=40)
    table.add_column("Example Task", style="yellow", width=50)
    
    agents_info = [
        ("Research", "Gather and compile information", "Research factors affecting RB performance in dome stadiums"),
        ("Analysis", "Deep analysis of players/teams", "Analyze and compare top 5 QBs for playoffs"),
        ("Code", "Generate analysis scripts", "Create a Python script to analyze red zone efficiency"),
        ("Optimization", "Optimize lineups and strategy", "Find the optimal DFS lineup under $50,000 salary"),
        ("Workflow", "Coordinate multiple agents", "Complete full season review with recommendations")
    ]
    
    for agent_type, purpose, example in agents_info:
        table.add_row(agent_type, purpose, example)
    
    console.print(table)


async def interactive_agent_demo():
    """Interactive demonstration where user can spawn agents"""
    console.print(Panel.fit(
        "[bold cyan]Interactive Agent Demo[/bold cyan]\n"
        "You can now spawn agents to handle complex tasks",
        border_style="cyan"
    ))
    
    agent = HowieAgent()
    
    while True:
        console.print("\n[bold]Agent Options:[/bold]")
        console.print("1. Spawn a research agent")
        console.print("2. Spawn an analysis agent")
        console.print("3. Run parallel agents")
        console.print("4. Execute a workflow")
        console.print("5. Check agent status")
        console.print("6. Exit demo")
        
        choice = console.input("\n[cyan]Choose an option (1-6):[/cyan] ")
        
        if choice == "1":
            task = console.input("[yellow]Research task:[/yellow] ")
            response = await agent.process_message(f"Spawn a research agent for: {task}")
            console.print(f"\n{response}")
            
        elif choice == "2":
            task = console.input("[yellow]Analysis task:[/yellow] ")
            response = await agent.process_message(f"Spawn an analysis agent for: {task}")
            console.print(f"\n{response}")
            
        elif choice == "3":
            console.print("[yellow]Enter tasks (comma-separated):[/yellow]")
            tasks = console.input().split(",")
            response = await agent.process_message(
                f"Run these tasks in parallel: {tasks}"
            )
            console.print(f"\n{response}")
            
        elif choice == "4":
            workflow = console.input("[yellow]Workflow objective:[/yellow] ")
            response = await agent.process_message(f"Execute workflow: {workflow}")
            console.print(f"\n{response}")
            
        elif choice == "5":
            agent_id = console.input("[yellow]Agent ID to check:[/yellow] ")
            response = await agent.process_message(f"Check agent status: {agent_id}")
            console.print(f"\n{response}")
            
        elif choice == "6":
            console.print("[green]Exiting demo...[/green]")
            break
        
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")


async def main():
    """Main demo function"""
    console.print(Panel.fit(
        "[bold green]Howie Agent System Demonstration[/bold green]\n"
        "Showcasing Claude-like autonomous agent capabilities",
        border_style="green"
    ))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-key-here":
        console.print("\n[red]Warning: OPENAI_API_KEY not set![/red]")
        console.print("Please set your OpenAI API key to run the demos:")
        console.print("export OPENAI_API_KEY='sk-...'")
        return
    
    # Show capabilities
    await demo_agent_capabilities()
    
    console.print("\n[bold]Choose a demo:[/bold]")
    console.print("1. Single Agent Demo")
    console.print("2. Parallel Agents Demo")
    console.print("3. Workflow Agent Demo")
    console.print("4. Interactive Demo")
    console.print("5. Run All Demos")
    
    choice = console.input("\n[cyan]Select demo (1-5):[/cyan] ")
    
    if choice == "1":
        await demo_single_agent()
    elif choice == "2":
        await demo_parallel_agents()
    elif choice == "3":
        await demo_workflow_agent()
    elif choice == "4":
        await interactive_agent_demo()
    elif choice == "5":
        await demo_single_agent()
        console.input("\n[dim]Press Enter to continue...[/dim]")
        await demo_parallel_agents()
        console.input("\n[dim]Press Enter to continue...[/dim]")
        await demo_workflow_agent()
    else:
        console.print("[red]Invalid choice[/red]")


def main_sync():
    """Synchronous wrapper for the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")

if __name__ == "__main__":
    main_sync()