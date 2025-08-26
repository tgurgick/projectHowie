"""
Main Howie Agent - Claude-like Fantasy Football AI Assistant
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from .context import ConversationContext
from .workspace import WorkspaceManager
from .base_tool import ToolResult, ToolStatus
from ..tools.registry import ToolRegistry, global_registry

# Import and register all tools
from ..tools.file_tools import (
    ReadFileTool, WriteFileTool, ImportRosterTool, 
    CreateReportTool, ListFilesTool
)
from ..tools.visualization_tools import (
    CreateChartTool, PlayerComparisonChartTool,
    SeasonTrendChartTool, ASCIIChartTool
)
from ..tools.code_generation_tools import (
    GenerateAnalysisScriptTool, GenerateSQLQueryTool
)
# from ..tools.realtime_tools import (
#     LiveScoresTool, PlayerNewsUpdatesTool,
#     WeatherUpdatesTool, LiveFantasyTrackerTool
# )
from ..tools.ml_projection_tools import (
    PlayerProjectionTool, LineupOptimizerTool
)
from ..tools.database_tools import (
    DatabaseQueryTool, PlayerStatsTool, TeamAnalysisTool,
    HistoricalTrendsTool, DatabaseInfoTool
)
from ..tools.agent_tools import (
    SpawnAgentTool, ParallelAgentsTool, CheckAgentTool, WorkflowTool
)

console = Console()


class HowieAgent:
    """Main AI agent for fantasy football assistance"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        
        # Initialize components
        self.context = ConversationContext()
        self.workspace = WorkspaceManager()
        self.tool_registry = global_registry
        
        # Register all tools
        self._register_tools()
        
        # System prompt
        self.system_prompt = self._build_system_prompt()
    
    def _register_tools(self):
        """Register all available tools"""
        # File operations
        self.tool_registry.register(ReadFileTool(), aliases=["read", "load"])
        self.tool_registry.register(WriteFileTool(), aliases=["write", "save"])
        self.tool_registry.register(ImportRosterTool(), aliases=["import"])
        self.tool_registry.register(CreateReportTool(), aliases=["report"])
        self.tool_registry.register(ListFilesTool(), aliases=["ls", "list"])
        
        # Visualization
        self.tool_registry.register(CreateChartTool(), aliases=["chart", "plot"])
        self.tool_registry.register(PlayerComparisonChartTool(), aliases=["compare_chart"])
        self.tool_registry.register(SeasonTrendChartTool(), aliases=["trend_chart"])
        self.tool_registry.register(ASCIIChartTool(), aliases=["ascii"])
        
        # Code generation
        self.tool_registry.register(GenerateAnalysisScriptTool(), aliases=["gen_script"])
        self.tool_registry.register(GenerateSQLQueryTool(), aliases=["gen_sql"])
        
        # Real-time data (temporarily disabled - replaced with web search)
        # self.tool_registry.register(LiveScoresTool(), aliases=["scores"])
        # self.tool_registry.register(PlayerNewsUpdatesTool(), aliases=["news"])
        # self.tool_registry.register(WeatherUpdatesTool(), aliases=["weather"])
        # self.tool_registry.register(LiveFantasyTrackerTool(), aliases=["track"])
        
        # ML predictions
        self.tool_registry.register(PlayerProjectionTool(), aliases=["project"])
        self.tool_registry.register(LineupOptimizerTool(), aliases=["optimize"])
        
        # Database access (for existing data)
        self.tool_registry.register(DatabaseQueryTool(), aliases=["query", "sql"])
        self.tool_registry.register(PlayerStatsTool(), aliases=["stats"])
        self.tool_registry.register(TeamAnalysisTool(), aliases=["team"])
        self.tool_registry.register(HistoricalTrendsTool(), aliases=["trends", "history"])
        self.tool_registry.register(DatabaseInfoTool(), aliases=["db_info"])
        
        # Agent spawning (like Claude's Task tool)
        self.tool_registry.register(SpawnAgentTool(), aliases=["agent", "task"])
        self.tool_registry.register(ParallelAgentsTool(), aliases=["parallel"])
        self.tool_registry.register(CheckAgentTool(), aliases=["check"])
        self.tool_registry.register(WorkflowTool(), aliases=["workflow"])
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI"""
        return """You are Howie, an advanced AI assistant specializing in fantasy football analysis.
You have access to a comprehensive set of tools similar to Claude, including:

1. File Operations: Read/write files, import rosters, create reports
2. Visualization: Create charts, comparisons, trends (both image and ASCII)
3. Code Generation: Generate Python scripts and SQL queries
4. Real-time Data: Live scores, player news, weather updates
5. ML Predictions: Player projections, lineup optimization
6. Database Access: Query existing fantasy databases with historical data

You maintain context across conversations and can:
- Analyze player data and statistics
- Generate visualizations and reports
- Write custom analysis scripts
- Track live games and updates
- Optimize lineups using machine learning
- Import and analyze user rosters

Always be helpful, accurate, and provide actionable insights.
When using tools, explain what you're doing and why.
Format responses using markdown for clarity."""
    
    async def process_message(self, user_input: str) -> str:
        """Process a user message and generate response"""
        # Add to context
        self.context.add_message("user", user_input)
        
        # Determine if tools are needed
        tool_calls = await self._plan_tool_usage(user_input)
        
        # Execute tools if needed
        tool_results = []
        if tool_calls:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing tools...", total=len(tool_calls))
                
                for tool_call in tool_calls:
                    result = await self._execute_tool(tool_call)
                    tool_results.append(result)
                    progress.advance(task)
        
        # Generate response
        response = await self._generate_response(user_input, tool_results)
        
        # Add to context
        self.context.add_message("assistant", response)
        
        return response
    
    async def _plan_tool_usage(self, user_input: str) -> List[Dict]:
        """Determine which tools to use based on user input"""
        # Use AI to determine tool usage
        planning_prompt = f"""Based on this user request, determine which tools to use:

User request: {user_input}

Available tool categories:
- file_operations: Read/write files, import rosters
- visualization: Create charts and graphs
- code_generation: Generate Python scripts or SQL queries
- realtime: Get live scores, news, weather
- ml_predictions: Generate projections, optimize lineups

Return a JSON list of tool calls needed, or empty list if no tools needed.
Example: [{{"tool": "read_file", "params": {{"file_path": "roster.csv"}}}}, ...]"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tool planning assistant."},
                    {"role": "user", "content": planning_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract tool calls
            if isinstance(result, dict) and "tool_calls" in result:
                return result["tool_calls"]
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            console.print(f"[yellow]Tool planning failed: {e}[/yellow]")
            return []
    
    async def _execute_tool(self, tool_call: Dict) -> ToolResult:
        """Execute a single tool"""
        tool_name = tool_call.get("tool")
        params = tool_call.get("params", {})
        
        # Execute tool
        result = await self.tool_registry.execute(tool_name, **params)
        
        # Record in context
        self.context.add_tool_execution(tool_name, params, result)
        
        return result
    
    async def _generate_response(self, user_input: str, tool_results: List[ToolResult]) -> str:
        """Generate AI response based on input and tool results"""
        # Build messages for AI
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation context
        for msg in self.context.get_recent_messages(10):
            if msg.role in ["user", "assistant"]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current query
        current_content = user_input
        
        # Add tool results if any
        if tool_results:
            tool_summary = "\n\nTool Results:\n"
            for i, result in enumerate(tool_results, 1):
                if result.status == ToolStatus.SUCCESS:
                    # Summarize data based on type
                    if isinstance(result.data, dict):
                        tool_summary += f"{i}. Success: {json.dumps(result.data, indent=2)[:500]}...\n"
                    elif isinstance(result.data, list):
                        tool_summary += f"{i}. Success: Found {len(result.data)} items\n"
                    else:
                        tool_summary += f"{i}. Success: {str(result.data)[:500]}...\n"
                else:
                    tool_summary += f"{i}. Error: {result.error}\n"
            
            current_content += tool_summary
        
        messages.append({"role": "user", "content": current_content})
        
        # Generate response
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def chat_loop(self):
        """Interactive chat loop"""
        console.print(Panel.fit(
            "[bold cyan]Howie - Fantasy Football AI Assistant[/bold cyan]\n"
            "Type 'help' for commands, 'quit' to exit",
            border_style="cyan"
        ))
        
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold green]You:[/bold green] ")
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print("[yellow]Goodbye! Good luck with your fantasy team![/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'tools':
                    self.tool_registry.display_tools()
                    continue
                
                elif user_input.lower() == 'context':
                    console.print(json.dumps(self.context.get_conversation_summary(), indent=2))
                    continue
                
                elif user_input.lower() == 'workspace':
                    console.print(json.dumps(self.workspace.get_workspace_info(), indent=2))
                    continue
                
                elif user_input.lower().startswith('save'):
                    # Save session
                    self.context.save_session()
                    console.print("[green]Session saved![/green]")
                    continue
                
                # Process message
                response = await self.process_message(user_input)
                
                # Display response
                console.print("\n[bold blue]Howie:[/bold blue]")
                console.print(Markdown(response))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit properly[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
# Available Commands

## Chat Commands
- **help** - Show this help message
- **tools** - List all available tools
- **context** - Show conversation context
- **workspace** - Show workspace information
- **save** - Save current session
- **quit/exit** - Exit the chat

## Tool Categories
- **File Operations** - Read/write files, import rosters
- **Visualization** - Create charts and graphs
- **Code Generation** - Generate analysis scripts
- **Real-time Data** - Live scores and updates
- **ML Predictions** - Projections and optimization

## Example Queries
- "Import my roster from roster.csv"
- "Compare Justin Jefferson vs CeeDee Lamb"
- "Generate a script to analyze RB performance"
- "Show me live scores for week 10"
- "Optimize my lineup for this week"
- "Create a chart showing player trends"
"""
        console.print(Markdown(help_text))
    
    async def execute_command(self, command: str, **kwargs) -> Any:
        """Execute a specific command programmatically"""
        # This allows using the agent programmatically
        return await self.process_message(command)
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            "context": self.context.get_conversation_summary(),
            "workspace": self.workspace.get_workspace_info(),
            "tools_available": len(self.tool_registry.list_tools()),
            "messages_processed": len(self.context.messages)
        }