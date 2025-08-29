"""
Tool Manager for Hierarchical Tool Selection
Implements workflow-based tool selection to improve accuracy and reduce costs
"""

from typing import Dict, List, Optional, Any
import re
from enum import Enum


class WorkflowType(Enum):
    """Workflow categories for tool selection"""
    ROSTER_MANAGEMENT = "roster_management"
    PLAYER_ANALYSIS = "player_analysis"
    GAME_TIME = "game_time"
    RESEARCH_DEEP_DIVE = "research_deep_dive"
    QUICK_STATS = "quick_stats"


class FantasyToolManager:
    """Manages hierarchical tool selection based on user intent workflows"""
    
    def __init__(self):
        self.workflows = {
            WorkflowType.ROSTER_MANAGEMENT: {
                "description": "Import, analyze, and manage your roster",
                "tools": ["import_roster", "read_file", "create_report", "player_projection", "lineup_optimizer"],
                "keywords": ["roster", "import", "my team", "upload", "lineup", "optimize", "start", "bench"],
                "cost_tier": "analysis",  # Medium cost
                "priority": 1
            },
            WorkflowType.PLAYER_ANALYSIS: {
                "description": "Compare players, analyze trends, get projections",
                "tools": ["smart_player_search", "contextual_search", "top_players", "player_projection", "create_visualization"],
                "keywords": ["compare", "vs", "versus", "analyze", "trends", "better", "rank", "projection", "reliable", "injury", "hurt"],
                "cost_tier": "analysis",
                "priority": 2
            },
            WorkflowType.QUICK_STATS: {
                "description": "Quick stats, rankings, and basic information",
                "tools": ["quick_stats_lookup", "top_players", "team"],
                "keywords": ["stats", "rank", "top", "best", "fantasy points", "season", "who", "how many"],
                "cost_tier": "simple",  # Low cost
                "priority": 3
            },
            WorkflowType.GAME_TIME: {
                "description": "Live scores, weather, fantasy tracking during games",
                "tools": ["get_game_day_data", "live_fantasy_tracker"],  # Consolidated tool
                "keywords": ["live", "scores", "weather", "tracking", "now", "game", "today"],
                "cost_tier": "simple",
                "priority": 4
            },
            WorkflowType.RESEARCH_DEEP_DIVE: {
                "description": "Advanced analysis, custom scripts, SQL queries",
                "tools": ["generate_analysis_script", "generate_sql_query", "create_visualization", "historical_trends"],
                "keywords": ["generate", "script", "sql", "advanced", "custom", "code", "analysis", "deep"],
                "cost_tier": "creative",  # High cost
                "priority": 5
            }
        }
    
    def select_workflow(self, user_query: str) -> WorkflowType:
        """
        Stage 1: Select the most appropriate workflow for the query
        Uses keyword matching and pattern recognition
        """
        query_lower = user_query.lower()
        workflow_scores = {}
        
        for workflow_type, config in self.workflows.items():
            score = 0
            
            # Keyword matching
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1
            
            # Pattern matching
            score += self._pattern_match_score(query_lower, workflow_type)
            
            # Priority boost for common workflows
            score += (6 - config["priority"]) * 0.1
            
            workflow_scores[workflow_type] = score
        
        # Return workflow with highest score
        if workflow_scores:
            best_workflow = max(workflow_scores.items(), key=lambda x: x[1])
            if best_workflow[1] > 0:
                return best_workflow[0]
        
        # Default fallback
        return WorkflowType.PLAYER_ANALYSIS
    
    def _pattern_match_score(self, query: str, workflow_type: WorkflowType) -> float:
        """Additional pattern matching for workflow selection"""
        patterns = {
            WorkflowType.ROSTER_MANAGEMENT: [
                r'should i start',
                r'my (team|roster|lineup)',
                r'import.*roster',
                r'optimize.*lineup'
            ],
            WorkflowType.PLAYER_ANALYSIS: [
                r'(compare|vs\.?|versus)',
                r'(better|worse) than',
                r'who (should|to|is)',
                r'how (reliable|good|consistent)',
                r'(injury|injuries|hurt|health)',
                r'(reliable|consistent|safe)'
            ],
            WorkflowType.QUICK_STATS: [
                r'top \d+',
                r'best \w+',
                r'rank(ing|s)?',
                r'how many'
            ],
            WorkflowType.GAME_TIME: [
                r'live (score|game)',
                r'what.*score',
                r'weather.*game',
                r'track.*points'
            ],
            WorkflowType.RESEARCH_DEEP_DIVE: [
                r'generate.*script',
                r'write.*code',
                r'sql query',
                r'advanced analysis'
            ]
        }
        
        score = 0
        for pattern in patterns.get(workflow_type, []):
            if re.search(pattern, query):
                score += 0.5
        
        return score
    
    def get_workflow_tools(self, workflow: WorkflowType) -> List[str]:
        """
        Stage 2: Return relevant tools for the selected workflow
        Reduces tool count from 20+ to 3-5 tools
        """
        return self.workflows[workflow]["tools"]
    
    def get_cost_tier(self, workflow: WorkflowType) -> str:
        """Get the recommended cost tier (model selection) for workflow"""
        return self.workflows[workflow]["cost_tier"]
    
    def should_use_context_injection(self, workflow: WorkflowType) -> bool:
        """Determine if context injection is beneficial for this workflow"""
        # Player analysis and quick stats benefit most from context injection
        return workflow in [WorkflowType.PLAYER_ANALYSIS, WorkflowType.QUICK_STATS]
    
    def get_workflow_description(self, workflow: WorkflowType) -> str:
        """Get human-readable description of the workflow"""
        return self.workflows[workflow]["description"]
    
    def analyze_query_complexity(self, user_query: str) -> Dict[str, Any]:
        """Analyze query complexity to help with model selection"""
        query_lower = user_query.lower()
        
        complexity_indicators = {
            "simple": ["who", "what", "top", "best", "rank", "stats"],
            "medium": ["compare", "analyze", "should", "better", "trends"],
            "complex": ["generate", "script", "advanced", "custom", "optimize"]
        }
        
        complexity_score = {
            "simple": 0,
            "medium": 0, 
            "complex": 0
        }
        
        for complexity, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    complexity_score[complexity] += 1
        
        # Determine overall complexity
        if complexity_score["complex"] > 0:
            complexity_level = "complex"
        elif complexity_score["medium"] > complexity_score["simple"]:
            complexity_level = "medium"
        else:
            complexity_level = "simple"
        
        return {
            "level": complexity_level,
            "scores": complexity_score,
            "word_count": len(user_query.split()),
            "has_player_names": bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', user_query))
        }


class ModelSelector:
    """Smart model selection based on task requirements and cost optimization"""
    
    def __init__(self):
        self.model_tiers = {
            "simple": {
                "model": "gpt-4o-mini",  # $0.000150/1K input tokens, $0.000600/1K output
                "max_tokens": 1000,
                "temperature": 0.3,
                "description": "Fast, cost-effective for simple queries"
            },
            "analysis": {
                "model": "gpt-4o",  # $0.0025/1K input tokens, $0.010/1K output  
                "max_tokens": 2000,
                "temperature": 0.5,
                "description": "Balanced cost/performance for analysis"
            },
            "creative": {
                "model": "claude-sonnet-4",  # For complex visualizations and scripts
                "max_tokens": 3000,
                "temperature": 0.7,
                "description": "Premium model for complex tasks"
            }
        }
    
    def select_model_for_workflow(self, workflow: WorkflowType, complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal model based on workflow and complexity"""
        # Get base tier from workflow
        workflow_config = FantasyToolManager().workflows[workflow]
        base_tier = workflow_config["cost_tier"]
        
        # Adjust based on complexity
        if complexity["level"] == "complex" and base_tier != "creative":
            tier = "analysis"  # Upgrade for complex queries
        elif complexity["level"] == "simple" and base_tier == "analysis":
            tier = "simple"   # Downgrade for simple queries
        else:
            tier = base_tier
        
        return self.model_tiers[tier]
    
    def estimate_cost(self, model_config: Dict[str, Any], input_tokens: int, output_tokens: int = None) -> float:
        """Estimate cost for a query (placeholder for actual cost calculation)"""
        # This would need actual token counting and pricing
        model = model_config["model"]
        if model == "gpt-4o-mini":
            return (input_tokens * 0.000150 + (output_tokens or 500) * 0.000600) / 1000
        elif model == "gpt-4o":
            return (input_tokens * 0.0025 + (output_tokens or 500) * 0.010) / 1000
        else:  # claude-sonnet-4 (placeholder pricing)
            return (input_tokens * 0.003 + (output_tokens or 500) * 0.015) / 1000


# Performance tracking for optimization
class PerformanceTracker:
    """Track tool selection accuracy, response times, and costs"""
    
    def __init__(self):
        self.metrics = {
            "tool_selections": [],
            "response_times": [],
            "costs": [],
            "success_rates": []
        }
    
    def track_tool_selection(self, query: str, selected_workflow: WorkflowType, 
                           selected_tools: List[str], success: bool, user_feedback: str = None):
        """Track tool selection accuracy"""
        self.metrics["tool_selections"].append({
            "query": query,
            "workflow": selected_workflow.value,
            "tools": selected_tools,
            "success": success,
            "feedback": user_feedback,
            "timestamp": __import__('datetime').datetime.now()
        })
    
    def track_response_time(self, query: str, duration: float, workflow: WorkflowType):
        """Monitor performance improvements"""
        self.metrics["response_times"].append({
            "query": query,
            "duration": duration,
            "workflow": workflow.value,
            "timestamp": __import__('datetime').datetime.now()
        })
    
    def track_cost(self, model: str, estimated_cost: float, workflow: WorkflowType):
        """Monitor cost optimizations"""
        self.metrics["costs"].append({
            "model": model,
            "cost": estimated_cost,
            "workflow": workflow.value,
            "timestamp": __import__('datetime').datetime.now()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not any(self.metrics.values()):
            return {"status": "No data collected yet"}
        
        return {
            "tool_selection_accuracy": self._calculate_accuracy(),
            "avg_response_time": self._calculate_avg_response_time(),
            "avg_cost_per_query": self._calculate_avg_cost(),
            "total_queries": len(self.metrics["tool_selections"])
        }
    
    def _calculate_accuracy(self) -> float:
        """Calculate tool selection accuracy"""
        if not self.metrics["tool_selections"]:
            return 0.0
        
        successful = sum(1 for selection in self.metrics["tool_selections"] if selection["success"])
        return successful / len(self.metrics["tool_selections"])
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.metrics["response_times"]:
            return 0.0
        
        total_time = sum(rt["duration"] for rt in self.metrics["response_times"])
        return total_time / len(self.metrics["response_times"])
    
    def _calculate_avg_cost(self) -> float:
        """Calculate average cost per query"""
        if not self.metrics["costs"]:
            return 0.0
        
        total_cost = sum(cost["cost"] for cost in self.metrics["costs"])
        return total_cost / len(self.metrics["costs"])
