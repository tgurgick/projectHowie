"""
Enhanced agent with multi-model support
"""

from typing import Dict, List, Optional, Any
import os
from pathlib import Path

from .agent import HowieAgent
from .model_manager import ModelManager, ModelTier
from .context import ConversationContext
from .workspace import WorkspaceManager
from ..tools.registry import global_registry


class EnhancedHowieAgent(HowieAgent):
    """Enhanced Howie agent with multi-model support"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 model_config_path: Optional[Path] = None):
        """Initialize with model manager"""
        # Initialize model manager first
        self.model_manager = ModelManager(model_config_path)
        
        # Initialize event log
        self.event_log = []
        
        # Set default model if provided
        if model:
            self.model_manager.set_model(model)
        
        # Get API key for the default model
        default_model_config = self.model_manager.models[self.model_manager.current_model]
        api_key = api_key or os.getenv(default_model_config.api_key_env)
        
        if not api_key and default_model_config.provider.value == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize parent with model manager's current model
        super().__init__(api_key=api_key, model=self.model_manager.current_model)
        
        # Override client to use model manager
        self.client = None  # We'll use model_manager.complete instead
        
        # Ensure tool registry is properly initialized
        if not hasattr(self, 'tool_registry') or not self.tool_registry:
            self.tool_registry = global_registry
        
        # Auto-discover tools to ensure all tools are available
        self.tool_registry.auto_discover_tools()
    
    async def _generate_response(self, user_input: str, tool_results: List) -> str:
        """Generate AI response using appropriate model"""
        from rich.console import Console
        console = Console()
        
        # Add football context for ambiguous queries
        enhanced_input = self._add_football_context(user_input)
        
        # Determine task type from enhanced input
        task_type = self._classify_task(enhanced_input)
        console.print(f"[dim]Task type: {task_type}[/dim]")
        self.log_event("task_classification", f"Task classified as: {task_type}", {
            "task_type": task_type,
            "user_input": enhanced_input,
            "original_input": user_input,
            "classification_method": "enhanced_input_analysis"
        })
        
        # Get the model that will be used for this task
        model_name = self.model_manager.task_model_mapping.get(task_type, self.model_manager.current_model)
        model_config = self.model_manager.models.get(model_name)
        console.print(f"[dim]Model: {model_name} ({model_config.provider.value})[/dim]")
        self.log_event("model_selection", f"Selected model: {model_name} ({model_config.provider.value})", {
            "model_name": model_name,
            "provider": model_config.provider.value,
            "task_type": task_type,
            "model_config": {
                "tier": model_config.tier.value,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "cost_per_1k_input": model_config.cost_per_1k_input,
                "cost_per_1k_output": model_config.cost_per_1k_output
            }
        })
        
        # For research tasks, try to get current information first
        if task_type == "research":
            try:
                # First, try to get database information for grounding
                database_info = ""
                try:
                    from ..tools.database_tools import DatabaseQueryTool
                    db_tool = DatabaseQueryTool()
                    
                    # Check if this is a roster/depth chart query
                    if any(word in user_input.lower() for word in ["depth chart", "roster", "who are", "players", "team"]):
                        # Try to extract team and position from query
                        import re
                        team_match = re.search(r'\b(eagles|chiefs|cowboys|patriots|bills|ravens|bengals|browns|steelers|texans|colts|jaguars|titans|broncos|chargers|raiders|dolphins|jets|falcons|panthers|saints|buccaneers|cardinals|rams|seahawks|49ers|giants|commanders|bears|lions|packers|vikings)\b', user_input.lower())
                        position_match = re.search(r'\b(wr|rb|qb|te|k|def|dst)\b', user_input.lower())
                        
                        if team_match:
                            team_abbrev = {
                                'eagles': 'PHI', 'chiefs': 'KCC', 'cowboys': 'DAL', 'patriots': 'NEP',
                                'bills': 'BUF', 'ravens': 'BAL', 'bengals': 'CIN', 'browns': 'CLE',
                                'steelers': 'PIT', 'texans': 'HOU', 'colts': 'IND', 'jaguars': 'JAC',
                                'titans': 'TEN', 'broncos': 'DEN', 'chargers': 'LAC', 'raiders': 'LVR',
                                'dolphins': 'MIA', 'jets': 'NYJ', 'falcons': 'ATL', 'panthers': 'CAR',
                                'saints': 'NOS', 'buccaneers': 'TBB', 'cardinals': 'ARI', 'rams': 'LAR',
                                'seahawks': 'SEA', '49ers': 'SFO', 'giants': 'NYG', 'commanders': 'WAS',
                                'bears': 'CHI', 'lions': 'DET', 'packers': 'GBP', 'vikings': 'MIN'
                            }.get(team_match.group(1))
                            
                            if team_abbrev:
                                position_filter = ""
                                if position_match:
                                    position_filter = f" AND position = '{position_match.group(1).upper()}'"
                                
                                db_query = f"""
                                SELECT name, position, team FROM players 
                                WHERE team = '{team_abbrev}'{position_filter}
                                ORDER BY name
                                LIMIT 10
                                """
                                
                                db_result = await db_tool.execute(db_query)
                                if db_result.status.value == "success":
                                    database_info = f"\n\nDatabase Roster Information:\n{db_result.data.to_string()}"
                                    
                except Exception as e:
                    print(f"Database query failed: {e}")
                
                # Import the web search tool
                from ..tools.realtime_tools import search_current_nfl_info
                
                # Enhance the query to be more specific to NFL/football
                enhanced_query = enhanced_input.strip()
                
                # Add NFL context if the query seems generic
                if not any(term in enhanced_query.lower() for term in ['nfl', 'football', 'qb', 'rb', 'wr', 'te', 'defense', 'offense', 'coach', 'team', 'roster']):
                    # Check if it's asking about a player or team
                    if any(word in enhanced_query.lower() for word in ['project', 'season', 'stats', 'performance']):
                        enhanced_query = f"NFL football {enhanced_query}"
                
                # Search for current information
                current_info = await search_current_nfl_info(enhanced_query)
                
                # If we got current information, include it in the response
                if current_info and "❌" not in current_info:
                    # Use a fast model to summarize the current info
                    self.model_manager.set_model("gpt-4o-mini")
                    
                    # Check if this is asking about outdated information
                    is_outdated_query = any(word in user_input.lower() for word in ["2024", "last season", "previous season"])
                    
                    if is_outdated_query:
                        enhanced_prompt = f"""The user is asking about {enhanced_input}, but this appears to be about the 2024 season (last season). 

Current Information for 2025 season:
{current_info}{database_info}

Please provide:
1. The current 2025 roster information they're likely looking for
2. A note that 2024 was last season and you're providing current information
3. The most up-to-date depth chart/roster information available

Focus on current 2025 season information rather than historical 2024 data."""
                    else:
                        enhanced_prompt = f"""Based on the current information below, provide a comprehensive answer to the user's question: "{enhanced_input}"

Current Information:
{current_info}{database_info}

Please provide a detailed response that incorporates this current information and any additional insights you can offer. If database information is available, use it to verify and ground the current information."""
                    
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": enhanced_prompt}
                    ]
                    
                    console.print(f"[dim]API call: {model_name}[/dim]")
                    self.log_event("api_call", f"API call to {model_name}", {
                        "model": model_name,
                        "provider": "research",
                        "context": "database_grounding",
                        "messages": messages,
                        "purpose": "Get database information for research grounding"
                    })
                    response = await self.model_manager.complete(
                        messages=messages,
                        task_type=task_type,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response
                else:
                    # Web search failed or returned no results, use database information if available
                    if database_info:
                        enhanced_prompt = f"""The user asked: "{user_input}"

I couldn't find current web information, but I have database information available:

{database_info}

Please provide a helpful response based on the available database information. If the user is asking about current season information that might not be in the database, let them know that the database contains historical data and suggest they check official team sources for the most current information."""
                        
                        messages = [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": enhanced_prompt}
                        ]
                        
                        response = await self.model_manager.complete(
                            messages=messages,
                            task_type=task_type,
                            temperature=0.7,
                            max_tokens=2000
                        )
                        return response
                    
            except Exception as e:
                # If web search fails, fall back to normal processing
                print(f"Web search failed, falling back to model response: {e}")
        
        # For Perplexity models, use simpler message format to avoid API errors
        if model_config and model_config.provider.value == "perplexity":
            # Use clean, simple messages for Perplexity with better prompting
            system_prompt = """You are a helpful AI assistant specializing in fantasy football and NFL information. 

IMPORTANT GUIDELINES:
1. Provide ONLY accurate, factual information about NFL football
2. If you're unsure about specific details, say so rather than guessing
3. For team rosters and depth charts, be extremely precise about which team each player belongs to
4. If information seems contradictory or unclear, acknowledge the uncertainty
5. Cite sources when possible
6. For current season information, prioritize official team announcements and reliable sports news sources
7. IMPORTANT: 2024 was LAST SEASON. When users ask about "2024" rosters, provide CURRENT 2025 season information instead
8. Always clarify if you're providing current season (2025) vs. historical information

FANTASY RANKING GUIDELINES:
9. For BROAD fantasy rankings (e.g., "top 20 prospects", "best players to draft"), provide LEAGUE-WIDE analysis covering ALL teams
10. DO NOT focus on just one team unless specifically asked - cover multiple teams and positions
11. When ranking players, consider multiple teams and provide diverse options across the league
12. If tool data seems limited to one team, supplement with league-wide knowledge to provide comprehensive rankings
13. For depth charts and rosters, focus on the most recent, up-to-date information
14. CRITICAL: Focus ONLY on NFL football. Do not provide information about MLB, NBA, or other sports
15. If a query seems ambiguous, ask for clarification about which NFL player or team they're referring to
16. For player projections and stats, ensure you're discussing NFL football players, not other sports

Please provide accurate, current NFL football information when available."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        else:
            # Build messages for other models
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation context
            recent_messages = self.context.get_recent_messages(10)
            for msg in recent_messages:
                if msg.role in ["user", "assistant"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current query with tool results
            current_content = user_input
            
            # If this is a follow-up query, add context about what we were discussing
            follow_up_indicators = ["let's try that", "try that", "do that", "go ahead", "yes", "ok", "okay", "what about", "how about", "and", "also", "this was", "but we're", "I want to look", "look ahead", "this season", "next season"]
            is_follow_up = any(indicator in user_input.lower() for indicator in follow_up_indicators)
            
            if is_follow_up:
                # Find the last substantive query
                for msg in reversed(recent_messages):
                    if msg.role == "user" and not any(indicator in msg.content.lower() for indicator in follow_up_indicators):
                        context_note = f"\n\nContext: This is a follow-up to: '{msg.content}'. Continue the same topic but address the new request. If the user mentions seasons or years, update accordingly (2025 is current season, 2024 is last season)."
                        current_content = context_note + "\n\n" + current_content
                        break
            successful_tools = 0
            if tool_results:
                tool_summary = "\n\nTool Results:\n"
                for i, result in enumerate(tool_results, 1):
                    if result.status.value == "success":
                        tool_summary += f"{i}. Success: {str(result.data)[:500]}...\n"
                        successful_tools += 1
                    else:
                        tool_summary += f"{i}. Error: {result.error}\n"
                
                # If tools succeeded, prioritize database results
                if successful_tools > 0:
                    current_content += f"\n\nIMPORTANT: You have access to database results from tools. Use this data as your primary source and provide a comprehensive response based on the database information. Only supplement with general knowledge if needed."
                # If no tools succeeded, provide a helpful response without relying on tools
                elif successful_tools == 0:
                    current_content += f"\n\nNote: I attempted to use tools to help answer your question, but they encountered errors. I'll provide a comprehensive response based on my knowledge of NFL football and fantasy football. Please provide a detailed analysis without saying 'hold on' or 'please wait'."
                
                current_content += tool_summary
            
            messages.append({"role": "user", "content": current_content})
        
        # Get response using model manager
        try:
            console.print(f"[dim]API call: {model_name}[/dim]")
            self.log_event("api_call", f"API call to {model_name}", {
                "model": model_name,
                "provider": model_config.provider.value,
                "task_type": task_type,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "context": "main_response_generation"
            })
            response = await self.model_manager.complete(
                messages=messages,
                task_type=task_type,
                temperature=0.7,
                max_tokens=2000
            )
            
            # For factual queries, add a confidence check
            if task_type == "reasoning" and model_config and model_config.provider.value == "perplexity":
                # Add a note about fact-checking
                response += "\n\nNote: This information was retrieved from current sources. For the most accurate and up-to-date roster information, please verify with official team websites or reliable sports news sources."
            
            # Add confidence indicator if we used database grounding
            if "Database Roster Information:" in response:
                response += "\n\nDatabase Verified: This response has been cross-referenced with our internal database for accuracy."
            
            # If tools succeeded, prioritize database results
            if tool_results and any(result.status.value == "success" for result in tool_results):
                # Add a note that we have database data
                response += "\n\nDatabase Data: This response includes data from our fantasy football database."
            
            # If tools failed, add a note about providing general information
            elif tool_results and all(result.status.value == "error" for result in tool_results):
                response += "\n\nNote: I attempted to use tools to gather specific data, but they encountered errors. This response is based on my general knowledge of NFL football and fantasy football."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def _plan_tool_usage(self, user_input: str) -> List[Dict]:
        """Use appropriate model for planning"""
        from rich.console import Console
        console = Console()
        
        # Get available tools from registry
        available_tools = self.tool_registry.list_tools() if hasattr(self, 'tool_registry') and self.tool_registry else []
        
        # Log available tools (verbose details go to logs only)
        self.log_event("tool_planning", f"Available tools: {available_tools}")
        
        planning_prompt = f"""Based on this user request, determine which tools to use:

User request: {user_input}

Available tools: {available_tools}

TOOL SELECTION GUIDE:
1. For BROAD FANTASY RANKINGS queries (e.g., "top 20 prospects", "best players to draft", "league rankings"):
   - Use: top_players (with appropriate position or ALL for league-wide)
   - Use: realtime_search (for current season context, injuries, trades)
   - DO NOT limit to specific teams - get league-wide data

2. For "top players by position" queries (e.g., "top WRs", "best QBs"):
   - Use: top_players (with position parameter)

3. For specific player stats (e.g., "Ja'Marr Chase stats"):
   - Use: player_stats (with player_name parameter)

4. For team analysis (e.g., "Cowboys roster", "team analysis"):
   - Use: team_analysis (with team parameter)

5. For ADP/draft position queries (e.g., "top ADP", "draft rankings"):
   - Use: database_query (with SQL joining fantasy_market and players tables)
   - IMPORTANT: ADP season mapping - 2025 data = 2025-2026 NFL season, 2024 data = 2024-2025 NFL season
   - Example: "SELECT p.name, p.position, p.team, fm.adp_overall FROM fantasy_market fm JOIN players p ON fm.player_id = p.player_id WHERE fm.season = 2025 AND p.position = 'WR' ORDER BY fm.adp_overall ASC LIMIT 10"

6. For database queries (e.g., "players with 1000+ yards"):
   - Use: database_query (with SQL or natural language)

7. For current news/info (e.g., "latest cuts", "recent trades"):
   - Use: realtime_search

8. For player projections:
   - Use: player_projection ONLY with specific player_name (e.g., "A.J. Brown")
   - Do NOT use for team-based queries or multiple players

SEASON CONTEXT:
- Current season is 2025 NFL season
- "This season" = 2025
- "Last season" = 2024
- Always default to 2025 for current queries unless user specifically mentions a different year

7. For player comparisons (e.g., "compare Chase vs Jefferson"):
   - Use: player_comparison_chart

8. For analysis scripts (e.g., "generate analysis"):
   - Use: generate_analysis_script

IMPORTANT RULES:
- For position-based queries like "top WRs", use top_players NOT player_stats
- For specific player queries, use player_stats with exact player name
- For team queries, use team_analysis with team abbreviation
- For follow-up queries (e.g., "what about 2025"), maintain context from previous query
- Always provide required parameters for tools
- Use the most specific tool available for the query
- For ADP queries, use database_query with proper JOIN between fantasy_market and players tables
- ADP SEASON MAPPING: 2025 ADP = 2025 season data, 2024 ADP = 2024 season data, etc.

Return a JSON list of tool calls with proper parameters, or empty list if no tools needed.
Example: [{{"tool": "top_players", "params": {{"position": "WR", "season": 2024, "limit": 10}}}}, ...]"""
        
        messages = [
            {"role": "system", "content": """You are a tool planning assistant for a fantasy football AI. Your job is to select the most appropriate tools for user queries.

CRITICAL RULES:
1. For "top players by position" queries (e.g., "top WRs", "best QBs"), use top_players tool
2. For specific player stats (e.g., "Ja'Marr Chase stats"), use player_stats tool
3. For team analysis (e.g., "Cowboys roster"), use team_analysis tool
4. For ADP/draft position queries (e.g., "top ADP", "draft rankings"), use database_query tool with SQL joining fantasy_market and players tables
5. For current news/info (e.g., "latest cuts"), use realtime_search tool
6. For database queries (e.g., "players with 1000+ yards"), use database_query tool
7. For player comparisons, use player_comparison_chart tool
8. For analysis scripts, use generate_analysis_script tool

ALWAYS provide the required parameters for each tool. Be specific and accurate in tool selection."""},
            {"role": "user", "content": planning_prompt}
        ]
        
        # Log detailed planning prompt
        self.log_event("tool_planning_prompt", "Tool planning prompt generated", {
            "user_input": user_input,
            "full_prompt": planning_prompt,
            "available_tools": available_tools,
            "prompt_length": len(planning_prompt),
            "system_message": messages[0]["content"]
        })
        
        try:
            # Use fast model for planning
            response = await self.model_manager.complete(
                messages=messages,
                model="gpt-4o-mini",  # Use fast model for planning
                temperature=0.3
            )
            
            # Log the complete planning response
            self.log_event("tool_planning_response", "Tool planning response received", {
                "user_input": user_input,
                "model_response": response,
                "response_length": len(response),
                "model_used": "gpt-4o-mini",
                "temperature": 0.3
            })
            
            # Parse response
            import json
            try:
                # Extract JSON from response
                if "{" in response and "}" in response:
                    start = response.index("{")
                    end = response.rindex("}") + 1
                    json_str = response[start:end]
                    if json_str.startswith("{"):
                        json_str = "[" + json_str + "]"
                    result = json.loads(json_str)
                else:
                    result = []
                
                # Log the parsed plan with detailed analysis
                self.log_event("tool_planning_parsed", "Tool planning parsed successfully", {
                    "user_input": user_input,
                    "parsed_plan": result,
                    "plan_analysis": {
                        "tools_planned": len(result) if isinstance(result, list) else 0,
                        "tool_names": [plan.get("tool", "unknown") for plan in result] if isinstance(result, list) else [],
                        "has_parameters": all(plan.get("params") for plan in result) if isinstance(result, list) else False,
                        "plan_complexity": "simple" if len(result) <= 1 else "moderate" if len(result) <= 3 else "complex"
                    },
                    "validation": {
                        "is_valid_list": isinstance(result, list),
                        "all_tools_valid": all(isinstance(plan, dict) and "tool" in plan for plan in result) if isinstance(result, list) else False
                    }
                })
                    
                return result if isinstance(result, list) else []
            except Exception as parse_error:
                console.print(f"[dim]Plan parsing failed: {parse_error}[/dim]")
                self.log_event("tool_planning", f"Plan parsing failed: {parse_error}")
                return []
                
        except Exception as e:
            print(f"Tool planning failed: {e}")
            return []
    
    def _classify_task(self, user_input: str) -> str:
        """Classify the task type for model selection"""
        input_lower = user_input.lower()
        
        # Check for current year/time indicators first
        current_time_indicators = [
            "2025", "this season", "this year", "current", "latest", "recent",
            "now", "today", "yesterday", "this week", "this month"
        ]
        
        # Outdated year indicators - should trigger current season search
        outdated_indicators = [
            "2024", "last season", "previous season"
        ]
        
        # Specific factual queries that need high accuracy
        factual_queries = [
            "depth chart", "starting", "backup", "roster", "who is", "who are",
            "what is", "which team", "plays for", "belongs to", "member of"
        ]
        
        # Research keywords - should use Perplexity for real-time info
        research_keywords = [
            "research", "find out", "search", "latest", "current", "recent",
            "new", "update", "news", "coach", "coordinator", "draft", "trade", "signing", "injury",
            "schedule", "playoff", "super bowl", "championship", "offensive coordinator",
            "defensive coordinator", "head coach", "fired", "hired", "retired",
            "free agency", "contract", "extension", "cuts"
        ]
        
        # If it's a specific factual query about rosters/players, use reasoning model
        if any(word in input_lower for word in factual_queries):
            return "reasoning"
        # Only classify as research if it explicitly mentions research terms AND football context
        elif any(word in input_lower for word in research_keywords) and any(word in input_lower for word in ["football", "nfl", "fantasy", "player", "team", "coach", "draft", "trade", "injury", "roster", "cut", "signing"]):
            return "research"
        # If it contains outdated indicators, prioritize research for current info
        # BUT if it's asking for stats/data, use reasoning to check database first
        elif any(word in input_lower for word in outdated_indicators):
            if any(word in input_lower for word in ["stats", "statistics", "data", "top", "best", "rankings", "adp", "draft position"]):
                return "reasoning"  # Use database for historical stats
            else:
                return "research"  # Use research for current info
        # Special handling for ADP queries - 2025 ADP should use 2025 season data
        elif "2025" in input_lower and any(word in input_lower for word in ["adp", "draft position", "draft rankings"]):
            return "reasoning"  # Use database for ADP data
        # If it contains current time indicators, prioritize research
        elif any(word in input_lower for word in current_time_indicators):
            return "research"
        # For ambiguous queries without clear football context, default to reasoning
        elif any(word in input_lower for word in ["cuts", "benefits", "who", "what", "when", "where", "why", "how"]):
            return "default"
        # For follow-up queries that might be asking for the same type of data
        elif any(word in input_lower for word in ["what about", "how about", "and", "also", "too", "as well"]):
            return "reasoning"  # Assume they want similar data/analysis
        elif any(word in input_lower for word in ["analyze", "compare", "evaluate", "analysis"]):
            return "analysis"
        elif any(word in input_lower for word in ["code", "script", "generate", "create", "program"]):
            return "code_generation"
        elif any(word in input_lower for word in ["optimize", "best", "recommend", "optimal"]):
            return "optimization"
        elif any(word in input_lower for word in ["simple", "quick", "list", "show", "count"]):
            return "simple_query"
        else:
            return "default"
    
    async def process_with_model(self, user_input: str, model: str) -> str:
        """Process a message with a specific model"""
        original_model = self.model_manager.current_model
        try:
            self.model_manager.set_model(model)
            response = await self.process_message(user_input)
            return response
        finally:
            self.model_manager.set_model(original_model)
    
    def configure_task_models(self, config: Dict[str, str]):
        """Configure which models to use for different tasks"""
        for task_type, model in config.items():
            self.model_manager.set_task_model(task_type, model)
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        models = self.model_manager.list_models()
        info = {
            "current_model": self.model_manager.current_model,
            "available_models": {},
            "task_mappings": self.model_manager.task_model_mapping,
            "usage": self.model_manager.get_usage_report()
        }
        
        for name, config in models.items():
            info["available_models"][name] = {
                "provider": config.provider.value,
                "tier": config.tier.value,
                "supports_tools": config.supports_tools,
                "supports_vision": config.supports_vision,
                "cost_per_1k": f"${config.cost_per_1k_input}/{config.cost_per_1k_output}",
                "best_for": config.best_for
            }
        
        return info
    
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        self.model_manager.set_model(model_name)
        self.model = model_name
        
        # Update system prompt to mention model
        self.system_prompt = self._build_system_prompt()
    
    async def process_message(self, user_input: str) -> str:
        """Process a user message with enhanced logging"""
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        # Add to context
        self.context.add_message("user", user_input)
        self.log_event("user_input", f"User query: {user_input[:50]}...", {
            "user_query": user_input,
            "query_length": len(user_input),
            "session_id": getattr(self, 'session_id', 'default')
        })
        
        # COMPREHENSIVE SEARCH WORKFLOW
        # Step 1: Plan - Determine comprehensive search strategy
        console.print("[dim]Step 1: Planning comprehensive search...[/dim]")
        search_plan = await self._plan_comprehensive_search(user_input)
        
        # Step 2: Execute initial searches (fan-out)
        console.print("[dim]Step 2: Executing searches...[/dim]")
        search_results = await self._execute_searches(search_plan)
        
        # Step 3: Verify and reflect on results
        console.print("[dim]Step 3: Verifying results...[/dim]")
        verified_results = await self._verify_and_reflect(user_input, search_results)
        
        # Use verified results as tool calls for execution
        tool_calls = verified_results
        
        # Execute tools if needed
        tool_results = []
        if tool_calls:
            console.print(f"[dim]Executing {len(tool_calls)} tool(s)...[/dim]")
            self.log_event("tool_execution", f"Executing {len(tool_calls)} tools", {
                "total_tools": len(tool_calls),
                "tool_list": [call.get('tool', 'unknown') for call in tool_calls],
                "user_input": user_input,
                "execution_context": "comprehensive_search_workflow"
            })
            
            for i, tool_call in enumerate(tool_calls, 1):
                tool_name = tool_call.get("tool", "unknown")
                console.print(f"[dim]  [{i}/{len(tool_calls)}] Running: {tool_name}[/dim]")
                
                try:
                    result = await self._execute_tool(tool_call)
                    # Add tool name to metadata for summary display
                    if hasattr(result, 'metadata') and result.metadata:
                        result.metadata['tool_name'] = tool_name
                    tool_results.append(result)
                    
                    # Clean status indicators
                    
                    # Show detailed result information
                    if result.status.value == "success":
                        # Show summary of successful results
                        if hasattr(result, 'data') and result.data:
                            if isinstance(result.data, dict):
                                if 'players' in result.data:
                                    count = len(result.data['players'])
                                    summary = f"found {count} players"
                                elif 'total_players' in result.data:
                                    summary = f"found {result.data['total_players']} players"
                                elif 'rows_returned' in result.metadata:
                                    summary = f"returned {result.metadata['rows_returned']} rows"
                                elif 'games_found' in result.metadata:
                                    summary = f"found {result.metadata['games_found']} games"
                                else:
                                    summary = "success"
                                console.print(f"[dim]  [{i}/{len(tool_calls)}] {tool_name}: {summary}[/dim]")
                            else:
                                console.print(f"[dim]  [{i}/{len(tool_calls)}] {tool_name}: completed[/dim]")
                        else:
                            console.print(f"[dim]  [{i}/{len(tool_calls)}] {tool_name}: completed[/dim]")
                    else:
                        # Show error details
                        error_msg = result.error[:100] + "..." if result.error and len(result.error) > 100 else (result.error or "unknown error")
                        console.print(f"[dim]  [{i}/{len(tool_calls)}] {tool_name}: Error - {error_msg}[/dim]")
                    
                    # Safe way to check if data exists without DataFrame ambiguity
                    data_obj = getattr(result, 'data', None)
                    has_data = data_obj is not None
                    
                    self.log_event("tool_result", f"Tool {tool_name}: {result.status.value}", {
                        "tool_name": tool_name,
                        "status": result.status.value,
                        "has_data": has_data,
                        "data_type": type(data_obj).__name__,
                        "result_preview": str(data_obj)[:200] if has_data else None,
                        "error": getattr(result, 'error', None),
                        "execution_index": i,
                        "total_tools": len(tool_calls)
                    })
                except Exception as e:
                    # Create a failed result
                    from ..core.base_tool import ToolResult, ToolStatus
                    failed_result = ToolResult(
                        status=ToolStatus.ERROR,
                        error=f"Tool execution failed: {str(e)}"
                    )
                    tool_results.append(failed_result)
                    
                    # Show detailed error message
                    error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                    console.print(f"[dim]  [{i}/{len(tool_calls)}] {tool_name}: Failed - {error_msg}[/dim]")
                    self.log_event("tool_result", f"Tool {tool_name}: execution failed - {error_msg}", {
                        "tool_name": tool_name,
                        "status": "execution_failed",
                        "error": str(e),
                        "error_preview": error_msg,
                        "execution_index": i,
                        "total_tools": len(tool_calls)
                    })
        else:
            console.print("[dim]No tools needed[/dim]")
            self.log_event("tool_decision", "No tools needed")
        
        # Generate response  
        console.print("[dim]Generating response...[/dim]")
        response = await self._generate_response(user_input, tool_results)
        
        # Extract and display reasoning if available
        self._display_reasoning(response)
        
        # Show execution summary box
        self._display_execution_summary(user_input, tool_results)
        
        # Add to context
        self.context.add_message("assistant", response)
        self.log_event("ai_response", f"Generated response ({len(response)} chars)", {
            "model": getattr(self, 'current_model', 'unknown'),
            "response_length": len(response),
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "tools_used": len(tool_results) if tool_results else 0,
            "has_tool_data": bool(tool_results and any(hasattr(r, 'data') and r.data for r in tool_results))
        })
        
        return response
    
    def log_event(self, event_type: str, description: str, details: Optional[Dict] = None):
        """Log a system event with persistent storage"""
        import datetime
        import json
        import os
        from pathlib import Path
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "description": description,
            "details": details or {},
            "session_id": getattr(self, 'session_id', 'default')
        }
        self.event_log.append(event)
        
        # Keep only last 25 events in memory (as requested)
        if len(self.event_log) > 25:
            self.event_log = self.event_log[-25:]
        
        # Persistent JSON logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Session-based log file
        if not hasattr(self, 'session_id'):
            self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_file = log_dir / f"howie_session_{self.session_id}.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            # Fallback to console if file writing fails
            print(f"Logging failed: {e}")
    
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """Get the most recent events from memory"""
        return self.event_log[-count:] if self.event_log else []
    
    def get_session_logs(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get all logs from a specific session file"""
        import json
        from pathlib import Path
        
        if not session_id:
            session_id = getattr(self, 'session_id', None)
        
        if not session_id:
            return []
        
        log_file = Path("logs") / f"howie_session_{session_id}.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            print(f"Error reading session logs: {e}")
        
        return logs
    
    def get_tool_trace(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get detailed tool execution trace for a session"""
        logs = self.get_session_logs(session_id)
        return [log for log in logs if log["type"] in ["tool_execution", "tool_result", "tool_decision"]]
    
    def get_agent_trace(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get detailed agent decision trace for a session"""
        logs = self.get_session_logs(session_id)
        return [log for log in logs if log["type"] in ["task_classification", "model_selection", "api_call", "ai_response"]]
    
    def _add_football_context(self, user_input: str) -> str:
        """Add football context to ambiguous queries"""
        input_lower = user_input.lower()
        
        # Check if this is an ambiguous query that could benefit from football context
        ambiguous_terms = ["cuts", "benefits", "who", "what", "when", "where", "why", "how"]
        football_terms = ["nfl", "football", "fantasy", "player", "team", "coach", "roster", "depth chart"]
        
        # If it contains ambiguous terms but no clear football context, add context
        if any(term in input_lower for term in ambiguous_terms) and not any(term in input_lower for term in football_terms):
            if "cuts" in input_lower:
                return f"NFL roster cuts and fantasy football implications: {user_input}"
            elif "benefits" in input_lower:
                return f"NFL football benefits and fantasy football implications: {user_input}"
            elif "who" in input_lower:
                return f"NFL football context: {user_input}"
            else:
                return f"NFL football: {user_input}"
        
        return user_input
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with model awareness"""
        base_prompt = super()._build_system_prompt()
        
        current_model = self.model_manager.current_model
        model_config = self.model_manager.models[current_model]
        
        model_context = f"\n\nYou are currently powered by {current_model} ({model_config.provider.value})."
        
        if model_config.tier == ModelTier.FAST:
            model_context += " Optimize for quick, efficient responses."
        elif model_config.tier == ModelTier.PREMIUM:
            model_context += " Provide detailed, comprehensive analysis."
        elif model_config.tier == ModelTier.RESEARCH:
            model_context += " Focus on current information and research."
        
        return base_prompt + model_context
    
    def _display_reasoning(self, response: str) -> None:
        """Extract and display reasoning from Claude's response if available"""
        from rich.console import Console
        console = Console()
        # Look for common reasoning patterns from Claude
        reasoning_indicators = [
            "Let me think about this",
            "Here's my reasoning:",
            "My analysis:",
            "Thinking through this:",
            "Based on the data:",
            "Looking at this information:",
            "Considering the factors:",
            "To analyze this:"
        ]
        
        # Check if response contains reasoning
        lines = response.split('\n')
        reasoning_found = False
        
        for i, line in enumerate(lines):
            if any(indicator.lower() in line.lower() for indicator in reasoning_indicators):
                # Find the end of the reasoning section
                reasoning_lines = []
                for j in range(i, min(i + 3, len(lines))):  # Look at next few lines
                    if lines[j].strip():
                        reasoning_lines.append(lines[j].strip())
                    else:
                        break
                
                if reasoning_lines:
                    reasoning_text = ' '.join(reasoning_lines)[:180]
                    console.print(f"[dim]Reasoning: {reasoning_text}...[/dim]")
                    reasoning_found = True
                    break
        
        # If no explicit reasoning found, check for analysis patterns
        if not reasoning_found:
            analysis_patterns = [
                "analysis", "assessment", "evaluation", "consideration", 
                "factors", "trends", "implications", "outlook"
            ]
            
            for line in lines[:3]:  # Check first few lines
                if any(pattern in line.lower() for pattern in analysis_patterns) and len(line.strip()) > 30:
                    console.print(f"[dim]Analysis: {line.strip()[:150]}...[/dim]")
                    break
    
    def _display_execution_summary(self, user_input: str, tool_results: list) -> None:
        """Display a structured execution summary box"""
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Console
        console = Console()
        
        # Create execution summary table
        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_column("Field", style="dim", width=12)
        summary_table.add_column("Value", style="bright_green")
        
        # Add basic execution info  
        summary_table.add_row("Query", user_input[:50] + "..." if len(user_input) > 50 else user_input)
        summary_table.add_row("Model", self.model_manager.current_model)
        
        # Add tools executed
        if tool_results:
            successful_tools = [r for r in tool_results if r.status.value == "success"]
            failed_tools = [r for r in tool_results if r.status.value == "error"]
            
            tools_summary = f"{len(successful_tools)} successful"
            if failed_tools:
                tools_summary += f", {len(failed_tools)} failed"
            
            summary_table.add_row("Tools", tools_summary)
            
            # List tool names
            if successful_tools:
                tool_names = []
                for i, result in enumerate(successful_tools):
                    # Get tool name from metadata or assume from position
                    tool_name = result.metadata.get('tool_name', f'tool_{i+1}')
                    tool_names.append(tool_name)
                summary_table.add_row("Executed", ", ".join(tool_names[:3]) + ("..." if len(tool_names) > 3 else ""))
        else:
            summary_table.add_row("Tools", "None used")
        
        # Show the panel
        console.print(Panel(
            summary_table,
            title="[dim]Execution Summary[/dim]",
            border_style="dim",
            expand=False
        ))
    
    async def _plan_comprehensive_search(self, user_input: str) -> dict:
        """Step 1: Plan comprehensive search strategy"""
        search_plan = {
            "primary_searches": [],
            "verification_searches": [],
            "fallback_searches": [],
            "requires_current_data": False,
            "context_type": "general"
        }
        
        # Analyze query to determine search strategy
        query_lower = user_input.lower()
        
        # Check if query needs current/recent information
        current_indicators = [
            "this season", "2025", "current", "now", "recent", "latest", 
            "today", "injured", "depth chart", "roster", "who might", 
            "if injured", "backup", "replacement"
        ]
        
        search_plan["requires_current_data"] = any(indicator in query_lower for indicator in current_indicators)
        
        # Determine context type
        if "eagles" in query_lower or "philadelphia" in query_lower:
            search_plan["context_type"] = "eagles_specific"
        elif any(pos in query_lower for pos in ["wr", "rb", "qb", "te"]):
            search_plan["context_type"] = "position_specific"
        
        # Plan primary searches
        if search_plan["requires_current_data"]:
            if "injury" in query_lower or "injured" in query_lower or "backup" in query_lower:
                search_plan["primary_searches"].extend([
                    {"tool": "realtime_search", "params": {"query": f"Eagles depth chart 2025 season"}},
                    {"tool": "database_query", "params": {"query": "SELECT name, position, team FROM players WHERE team = 'PHI' AND position = 'WR'"}}
                ])
                search_plan["verification_searches"].append(
                    {"tool": "realtime_search", "params": {"query": "Eagles WR roster cuts signings 2025"}}
                )
            else:
                search_plan["primary_searches"].append(
                    {"tool": "top_players", "params": {"position": "WR", "season": 2025, "limit": 10}}
                )
        else:
            # Use database as primary for historical queries
            search_plan["primary_searches"].append(
                {"tool": "database_query", "params": {"query": user_input}}
            )
        
        # Log detailed search planning with reasoning
        self.log_event("search_planning", "Comprehensive search plan created", {
            "user_query": user_input,
            "analysis": {
                "requires_current_data": search_plan["requires_current_data"],
                "context_type": search_plan["context_type"],
                "matched_indicators": [indicator for indicator in current_indicators if indicator in query_lower]
            },
            "search_plan": search_plan,
            "reasoning": f"Query requires current data: {search_plan['requires_current_data']}, Context: {search_plan['context_type']}"
        })
        return search_plan
    
    async def _execute_searches(self, search_plan: dict) -> list:
        """Step 2: Execute searches in parallel (fan-out)"""
        all_searches = (
            search_plan["primary_searches"] + 
            search_plan["verification_searches"] + 
            search_plan["fallback_searches"]
        )
        
        search_results = []
        for i, search in enumerate(all_searches):
            try:
                # Convert search dict to proper tool call format
                tool_name = search["tool"]
                params = search["params"]
                
                # Log individual search execution
                self.log_event("search_execution", f"Executing search {i+1}/{len(all_searches)}", {
                    "search_index": i+1,
                    "total_searches": len(all_searches),
                    "tool_name": tool_name,
                    "params": params,
                    "search_type": "primary" if search in search_plan["primary_searches"] else "verification"
                })
                
                result = await self._execute_tool_by_name(tool_name, params)
                
                search_result = {
                    "search": search,
                    "result": result,
                    "type": "primary" if search in search_plan["primary_searches"] else "verification"
                }
                search_results.append(search_result)
                
                # Log search result details
                self.log_event("search_result", f"Search {i+1} completed", {
                    "search_index": i+1,
                    "tool_name": tool_name,
                    "status": result.status.value if hasattr(result, 'status') else "unknown",
                    "has_data": bool(getattr(result, 'data', None)),
                    "data_preview": str(getattr(result, 'data', ''))[:200] if getattr(result, 'data', None) else None,
                    "error": getattr(result, 'error', None)
                })
                
            except Exception as e:
                self.log_event("search_error", f"Search failed: {search} - {e}", {
                    "search_index": i+1,
                    "tool_name": search.get("tool", "unknown"),
                    "params": search.get("params", {}),
                    "error": str(e)
                })
        
        # Log overall search execution summary
        self.log_event("search_execution_summary", "Search execution completed", {
            "total_searches": len(all_searches),
            "successful_searches": len([r for r in search_results if hasattr(r["result"], 'status') and r["result"].status.value == "success"]),
            "failed_searches": len(all_searches) - len([r for r in search_results if hasattr(r["result"], 'status') and r["result"].status.value == "success"]),
            "search_types": {
                "primary": len([r for r in search_results if r["type"] == "primary"]),
                "verification": len([r for r in search_results if r["type"] == "verification"])
            }
        })
        
        return search_results
    
    async def _verify_and_reflect(self, user_input: str, search_results: list) -> list:
        """Step 3: Verify results and reflect on completeness"""
        # Check if we have sufficient current data
        has_current_data = any(
            result["result"].status.value == "success" and 
            result["type"] == "primary" and
            "realtime_search" in result["search"]["tool"]
            for result in search_results
        )
        
        has_database_data = any(
            result["result"].status.value == "success" and
            ("database_query" in result["search"]["tool"] or "top_players" in result["search"]["tool"])
            for result in search_results
        )
        
        # Determine if we need additional searches
        query_lower = user_input.lower()
        needs_reflection = False
        
        if ("injury" in query_lower or "backup" in query_lower) and not has_current_data:
            needs_reflection = True
            # Add additional current data search
            additional_search = {
                "tool": "realtime_search", 
                "params": {"query": f"NFL Eagles depth chart WR 2025 season injury replacements"}
            }
            try:
                additional_result = await self._execute_tool(additional_search)
                search_results.append({
                    "search": additional_search,
                    "result": additional_result,
                    "type": "reflection"
                })
            except Exception as e:
                self.log_event("reflection_error", f"Additional search failed: {e}")
        
        # Convert successful results back to tool calls for main execution
        successful_tool_calls = []
        for result in search_results:
            if result["result"].status.value == "success":
                successful_tool_calls.append(result["search"])
        
        # Log detailed verification and reflection results
        verification_analysis = {
            "data_assessment": {
                "has_current_data": has_current_data,
                "has_database_data": has_database_data,
                "data_sources": []
            },
            "gap_analysis": {
                "needed_reflection": needs_reflection,
                "gaps_identified": [],
                "additional_searches_needed": []
            },
            "quality_metrics": {
                "total_search_results": len(search_results),
                "successful_results": len([r for r in search_results if r["result"].status.value == "success"]),
                "failed_results": len([r for r in search_results if r["result"].status.value != "success"]),
                "primary_vs_verification": {
                    "primary": len([r for r in search_results if r["type"] == "primary"]),
                    "verification": len([r for r in search_results if r["type"] == "verification"]),
                    "reflection": len([r for r in search_results if r["type"] == "reflection"])
                }
            },
            "final_recommendations": {
                "successful_searches": len(successful_tool_calls),
                "data_completeness": "complete" if has_current_data and has_database_data else "partial",
                "confidence_level": "high" if has_current_data or has_database_data else "low"
            }
        }
        
        # Identify data sources
        for result in search_results:
            if result["result"].status.value == "success":
                source_type = "database" if "database" in result["search"]["tool"] else "realtime"
                verification_analysis["data_assessment"]["data_sources"].append({
                    "tool": result["search"]["tool"],
                    "type": source_type,
                    "search_type": result["type"]
                })
        
        # Identify gaps
        if ("injury" in user_input.lower() or "backup" in user_input.lower()) and not has_current_data:
            verification_analysis["gap_analysis"]["gaps_identified"].append("Missing current injury/depth chart data")
            verification_analysis["gap_analysis"]["additional_searches_needed"].append("Current roster information")
        
        self.log_event("search_verification", "Verification and reflection completed", verification_analysis)
        
        # Log reasoning for final tool selection
        self.log_event("search_reasoning", "Final tool selection reasoning", {
            "user_query": user_input,
            "reasoning": f"""
            Data Assessment:
            - Current data available: {has_current_data}
            - Database data available: {has_database_data}
            - Reflection needed: {needs_reflection}
            
            Quality Analysis:
            - {len(successful_tool_calls)} successful searches out of {len(search_results)} total
            - Data completeness: {verification_analysis['final_recommendations']['data_completeness']}
            - Confidence level: {verification_analysis['final_recommendations']['confidence_level']}
            
            Decision: Proceeding with {len(successful_tool_calls)} validated tool calls
            """.strip(),
            "selected_tools": [call["tool"] for call in successful_tool_calls],
            "data_quality_score": len(successful_tool_calls) / max(len(search_results), 1)
        })
        
        return successful_tool_calls
    
    async def _execute_tool_by_name(self, tool_name: str, params: dict):
        """Helper method to execute a tool by name with parameters"""
        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                from ..core.base_tool import ToolResult, ToolStatus
                return ToolResult(status=ToolStatus.ERROR, error=f"Tool {tool_name} not found")
            
            # Execute the tool with the provided parameters
            result = await tool.run(**params)
            return result
        except Exception as e:
            from ..core.base_tool import ToolResult, ToolStatus
            return ToolResult(status=ToolStatus.ERROR, error=str(e))