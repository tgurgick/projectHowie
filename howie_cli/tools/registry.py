"""
Tool registry for managing all available tools
"""

from typing import Dict, List, Optional, Type, Any
from ..core.base_tool import BaseTool, ToolResult, ToolStatus
import asyncio
from rich.console import Console
from rich.table import Table
import importlib
import pkgutil
from pathlib import Path

console = Console()


class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(self, tool: BaseTool, aliases: Optional[List[str]] = None):
        """Register a tool in the registry"""
        tool_name = tool.name.lower()
        self._tools[tool_name] = tool
        
        # Add to category
        category = getattr(tool, 'category', 'general')
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool_name)
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = tool_name
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name or alias"""
        name_lower = name.lower()
        
        # Check direct name
        if name_lower in self._tools:
            return self._tools[name_lower]
        
        # Check aliases
        if name_lower in self._aliases:
            return self._tools[self._aliases[name_lower]]
        
        return None
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Tool '{tool_name}' not found"
            )
        
        return await tool.run(**kwargs)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List all available tools"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def get_categories(self) -> List[str]:
        """Get all tool categories"""
        return list(self._categories.keys())
    
    def display_tools(self):
        """Display all tools in a formatted table"""
        table = Table(title="Available Tools", show_header=True, header_style="bright_green")
        table.add_column("Tool", style="bright_green", no_wrap=True)
        table.add_column("Category", style="dim")
        table.add_column("Description", style="white")
        table.add_column("Aliases", style="yellow")
        
        for name, tool in sorted(self._tools.items()):
            aliases = [alias for alias, target in self._aliases.items() if target == name]
            table.add_row(
                name,
                tool.category,
                tool.description[:50] + "..." if len(tool.description) > 50 else tool.description,
                ", ".join(aliases) if aliases else "-"
            )
        
        console.print(table)
    
    def auto_discover_tools(self, package_path: str = "howie_cli.tools"):
        """Auto-discover and register tools from package"""
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Get package directory
            if hasattr(package, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    if not ispkg and not modname.startswith('_'):
                        # Import the module
                        module_name = f"{package_path}.{modname}"
                        try:
                            module = importlib.import_module(module_name)
                            
                            # Find all tool classes
                            for item_name in dir(module):
                                if not item_name.startswith('_'):
                                    item = getattr(module, item_name)
                                    
                                    # Check if it's a tool class
                                    if (isinstance(item, type) and 
                                        issubclass(item, BaseTool) and 
                                        item is not BaseTool):
                                        
                                        # Instantiate and register
                                        tool_instance = item()
                                        self.register(tool_instance)
                                        
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not load module {module_name}: {e}[/yellow]")
                            
        except Exception as e:
            console.print(f"[red]Error discovering tools: {e}[/red]")
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get schema for a specific tool"""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_schema()
        return None
    
    def get_all_schemas(self) -> Dict[str, Dict]:
        """Get schemas for all tools"""
        return {name: tool.get_schema() for name, tool in self._tools.items()}


# Global registry instance
global_registry = ToolRegistry()