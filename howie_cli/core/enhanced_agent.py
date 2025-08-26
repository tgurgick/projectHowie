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
    
    async def _generate_response(self, user_input: str, tool_results: List) -> str:
        """Generate AI response using appropriate model"""
        # Determine task type from user input
        task_type = self._classify_task(user_input)
        
        # Get the model that will be used for this task
        model_name = self.model_manager.task_model_mapping.get(task_type, self.model_manager.current_model)
        model_config = self.model_manager.models.get(model_name)
        
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
                
                # Search for current information
                current_info = await search_current_nfl_info(user_input)
                
                # If we got current information, include it in the response
                if current_info and "❌" not in current_info:
                    # Use a fast model to summarize the current info
                    self.model_manager.set_model("gpt-4o-mini")
                    
                    # Check if this is asking about outdated information
                    is_outdated_query = any(word in user_input.lower() for word in ["2024", "last season", "previous season"])
                    
                    if is_outdated_query:
                        enhanced_prompt = f"""The user is asking about {user_input}, but this appears to be about the 2024 season (last season). 

Current Information for 2025 season:
{current_info}{database_info}

Please provide:
1. The current 2025 roster information they're likely looking for
2. A note that 2024 was last season and you're providing current information
3. The most up-to-date depth chart/roster information available

Focus on current 2025 season information rather than historical 2024 data."""
                    else:
                        enhanced_prompt = f"""Based on the current information below, provide a comprehensive answer to the user's question: "{user_input}"

Current Information:
{current_info}{database_info}

Please provide a detailed response that incorporates this current information and any additional insights you can offer. If database information is available, use it to verify and ground the current information."""
                    
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
1. Provide ONLY accurate, factual information
2. If you're unsure about specific details, say so rather than guessing
3. For team rosters and depth charts, be extremely precise about which team each player belongs to
4. If information seems contradictory or unclear, acknowledge the uncertainty
5. Cite sources when possible
6. For current season information, prioritize official team announcements and reliable sports news sources
7. IMPORTANT: 2024 was LAST SEASON. When users ask about "2024" rosters, provide CURRENT 2025 season information instead
8. Always clarify if you're providing current season (2025) vs. historical information
9. For depth charts and rosters, focus on the most recent, up-to-date information

Please provide accurate, current information when available."""
            
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
            for msg in self.context.get_recent_messages(10):
                if msg.role in ["user", "assistant"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current query with tool results
            current_content = user_input
            if tool_results:
                tool_summary = "\n\nTool Results:\n"
                for i, result in enumerate(tool_results, 1):
                    if result.status.value == "success":
                        tool_summary += f"{i}. Success: {str(result.data)[:500]}...\n"
                    else:
                        tool_summary += f"{i}. Error: {result.error}\n"
                current_content += tool_summary
            
            messages.append({"role": "user", "content": current_content})
        
        # Get response using model manager
        try:
            response = await self.model_manager.complete(
                messages=messages,
                task_type=task_type,
                temperature=0.7,
                max_tokens=2000
            )
            
            # For factual queries, add a confidence check
            if task_type == "reasoning" and model_config and model_config.provider.value == "perplexity":
                # Add a note about fact-checking
                response += "\n\n⚠️ **Note**: This information was retrieved from current sources. For the most accurate and up-to-date roster information, please verify with official team websites or reliable sports news sources."
            
            # Add confidence indicator if we used database grounding
            if "Database Roster Information:" in response:
                response += "\n\n✅ **Database Verified**: This response has been cross-referenced with our internal database for accuracy."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def _plan_tool_usage(self, user_input: str) -> List[Dict]:
        """Use appropriate model for planning"""
        planning_prompt = f"""Based on this user request, determine which tools to use:

User request: {user_input}

Available tool categories:
- file_operations: Read/write files, import rosters
- visualization: Create charts and graphs
- code_generation: Generate Python scripts or SQL queries
- realtime: Get live scores, news, weather
- ml_predictions: Generate projections, optimize lineups
- database: Query existing databases
- agents: Spawn autonomous agents

Return a JSON list of tool calls needed, or empty list if no tools needed.
Example: [{{"tool": "read_file", "params": {{"file_path": "roster.csv"}}}}, ...]"""
        
        messages = [
            {"role": "system", "content": "You are a tool planning assistant."},
            {"role": "user", "content": planning_prompt}
        ]
        
        try:
            # Use fast model for planning
            response = await self.model_manager.complete(
                messages=messages,
                model="gpt-4o-mini",  # Use fast model for planning
                temperature=0.3
            )
            
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
                    
                return result if isinstance(result, list) else []
            except:
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
            "2024", "2025", "this season", "this year", "new", "update", "news",
            "coach", "coordinator", "draft", "trade", "signing", "injury",
            "schedule", "playoff", "super bowl", "championship", "offensive coordinator",
            "defensive coordinator", "head coach", "fired", "hired", "retired",
            "free agency", "contract", "extension"
        ]
        
        # If it contains outdated indicators, prioritize research for current info
        if any(word in input_lower for word in outdated_indicators):
            return "research"
        # If it contains current time indicators, prioritize research
        elif any(word in input_lower for word in current_time_indicators):
            return "research"
        # If it's a specific factual query about rosters/players, use reasoning model
        elif any(word in input_lower for word in factual_queries):
            return "reasoning"
        elif any(word in input_lower for word in research_keywords):
            return "research"
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