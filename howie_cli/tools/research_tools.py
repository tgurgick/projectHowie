"""
Research tools using Claude's native web search for comprehensive analysis
"""

from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import logging

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.model_manager import ModelManager
import anthropic

logger = logging.getLogger(__name__)


class DeepResearchTool(BaseTool):
    """Tool for comprehensive research using Claude's native web search"""
    
    def __init__(self):
        super().__init__()
        self.name = "deep_research"
        self.description = "Conduct comprehensive research using Claude's web search for detailed analysis"
        self.category = "research"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string", 
                description="Research topic or question for comprehensive analysis",
                required=True
            ),
            ToolParameter(
                name="focus_areas",
                type="string",
                description="Specific areas to focus on (optional)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="max_searches",
                type="integer",
                description="Maximum number of web searches to perform",
                required=False,
                default=5
            )
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute comprehensive research using Claude's web search"""
        try:
            # Ensure .env is loaded
            load_dotenv()
            
            query = kwargs.get('query')
            focus_areas = kwargs.get('focus_areas', '')
            max_searches = kwargs.get('max_searches', 5)
            
            if not query or not isinstance(query, str):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Query parameter is required and must be a string"
                )
            
            # Check if Anthropic API key is available
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
                )
            
            # Perform deep research using Claude's web search
            result = await self._conduct_deep_research(query, focus_areas, max_searches)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                metadata={
                    "search_method": "claude_web_search",
                    "query": query,
                    "max_searches": max_searches
                }
            )
            
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to conduct deep research: {str(e)}"
            )
    
    async def _conduct_deep_research(self, query: str, focus_areas: str, max_searches: int) -> str:
        """Conduct comprehensive research using Claude's native web search"""
        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                raise Exception("Anthropic API key not found")
            
            # Initialize Anthropic client
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            # Construct research prompt
            research_prompt = f"""I need comprehensive research on: {query}

Please conduct a thorough analysis focusing on:
- Current developments and recent news
- Historical context and trends
- Expert opinions and analysis
- Statistical data and metrics
- Fantasy football implications (if applicable)
- Multiple perspectives and sources

{f"Special focus areas: {focus_areas}" if focus_areas else ""}

Provide a detailed, well-structured report with citations and sources."""
            
            # Define web search tool for Claude
            web_search_tool = {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_searches
                # Let Claude search any domain for comprehensive research
            }
            
            # Make API call with web search tool
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Use latest Claude with web search
                max_tokens=4096,
                temperature=0.3,  # Lower temperature for research accuracy
                tools=[web_search_tool],
                messages=[
                    {
                        "role": "user",
                        "content": research_prompt
                    }
                ]
            )
            
            # Extract the response content
            content = response.content[0].text if response.content else "No response generated"
            
            return f"Comprehensive Research Report:\n\n{content}"
            
        except Exception as e:
            logger.error(f"Error in Claude web search: {e}")
            
            # Fallback to regular Claude without web search
            try:
                client = anthropic.AsyncAnthropic(api_key=api_key)
                fallback_prompt = f"""Based on your training data, provide a comprehensive analysis of: {query}

Note: This analysis is based on training data and may not include the most recent developments. For the latest information, please check:
- ESPN.com/nfl
- NFL.com/news
- FantasyPros.com
- Beat reporters on Twitter/X

{f"Focus areas: {focus_areas}" if focus_areas else ""}

Provide as detailed an analysis as possible based on available knowledge."""
                
                response = await client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "user", 
                            "content": fallback_prompt
                        }
                    ]
                )
                
                content = response.content[0].text if response.content else "No response generated"
                
                return f"Research Analysis (Training Data):\n\n{content}\n\nNote: For current information, use the realtime_search tool or check latest sources directly."
                
            except Exception as fallback_error:
                logger.error(f"Fallback research failed: {fallback_error}")
                return f"""Research request: {query}

Deep research temporarily unavailable. For comprehensive information:

1. Use realtime_search tool for current developments
2. Check these sources directly:
   - ESPN.com/nfl/news
   - NFL.com/news
   - FantasyPros.com
   - Pro Football Reference
   - Football Outsiders

3. Follow expert analysts:
   - @AdamSchefter, @RapSheet (breaking news)
   - @FantasyPros (fantasy analysis) 
   - Beat reporters for specific teams

Error: {str(e)}"""
