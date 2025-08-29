# ProjectHowie Agent Optimization Guide

## 🎯 Executive Summary

ProjectHowie is a well-architected fantasy football AI assistant with ~20+ tools, but needs optimization for production-scale performance. Current estimated cost: $0.08-0.12 per query. Target: $0.048 per query with 85% tool selection accuracy.

## 🔍 Current Architecture Analysis

### Repository Structure
```
howie_cli/
├── core/
│   ├── agent.py          # REVIEW: Tool selection mechanism
│   ├── context.py        # ✅ Good: Context persistence
│   ├── workspace.py      # ✅ Good: File management
│   └── base_tool.py      # ✅ Good: Extensible framework
├── tools/
│   ├── file_tools.py           # 5 tools
│   ├── visualization_tools.py  # 4 tools (CONSOLIDATE)
│   ├── code_generation_tools.py # 2 tools
│   ├── realtime_tools.py       # 4 tools (BATCH APIs)
│   └── ml_projection_tools.py  # 2+ tools
```

### Current Tools Inventory
- **File Operations:** `read_file`, `write_file`, `import_roster`, `create_report`, `list_files`
- **Visualizations:** `create_chart`, `player_comparison_chart`, `season_trend_chart`, `ascii_chart`
- **Code Generation:** `generate_analysis_script`, `generate_sql_query`
- **Real-time Data:** `live_scores`, `player_news`, `weather_updates`, `live_fantasy_tracker`
- **ML Predictions:** `player_projection`, `lineup_optimizer`

## 🚨 Critical Issues to Fix

### Issue 1: Tool Selection at Scale
**Problem:** 20+ tools overwhelm LLM selection accuracy
**Current Impact:** ~60% tool selection accuracy, high costs
**Solution:** Implement hierarchical tool selection

### Issue 2: Unknown Function Calling Implementation
**Critical:** Verify modern function calling vs legacy text parsing
**Check in agent.py:**
```python
# ❌ LEGACY (update immediately)
if "TOOL:" in llm_response:
    tool_name = extract_tool_name(llm_response)

# ✅ SOTA (verify this exists)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,  # Native function calling
    tool_choice="auto"
)
```

### Issue 3: Redundant API Calls
**Problem:** Separate calls for live_scores + weather + news
**Impact:** 3-4x API cost multiplier
**Solution:** Batch game-day data fetching

### Issue 4: Tool Redundancy
**Problem:** 4 separate chart tools with overlapping functionality
**Impact:** Confuses LLM, increases selection errors
**Solution:** Consolidate into single smart visualization tool

## 🏆 Optimization Solutions

### Solution 1: Hierarchical Tool Selection

Replace technical categories with user-intent workflows:

```python
# File: core/tool_manager.py (NEW FILE)
class FantasyToolManager:
    def __init__(self):
        self.workflows = {
            "roster_management": {
                "description": "Import, analyze, and manage your roster",
                "tools": ["import_roster", "read_file", "create_report", "player_projection"],
                "keywords": ["roster", "import", "my team", "upload"]
            },
            "player_analysis": {
                "description": "Compare players, analyze trends, get projections",
                "tools": ["player_comparison_chart", "season_trend_chart", "player_news", "player_projection"],
                "keywords": ["compare", "vs", "versus", "analyze", "trends"]
            },
            "game_time": {
                "description": "Live scores, weather, fantasy tracking during games",
                "tools": ["live_scores", "weather_updates", "live_fantasy_tracker"],
                "keywords": ["live", "scores", "weather", "tracking", "now"]
            },
            "research_deep_dive": {
                "description": "Advanced analysis, custom scripts, SQL queries",
                "tools": ["generate_analysis_script", "generate_sql_query", "create_chart"],
                "keywords": ["generate", "script", "sql", "advanced", "custom"]
            }
        }
    
    def select_workflow(self, user_query: str) -> str:
        """Stage 1: Cheap GPT-3.5 call to select workflow"""
        # Implementation needed in agent.py
        pass
    
    def get_workflow_tools(self, workflow: str) -> List[dict]:
        """Stage 2: Return 3-5 relevant tools instead of all 20+"""
        # Implementation needed in agent.py
        pass
```

### Solution 2: Tool Consolidation

```python
# File: tools/visualization_tools.py (UPDATE EXISTING)
class UnifiedVisualizationTool(BaseTool):
    """Replaces: create_chart, player_comparison_chart, season_trend_chart, ascii_chart"""
    
    def __init__(self):
        super().__init__()
        self.name = "create_visualization"
        self.description = """Create any type of visualization for fantasy football analysis.
        
        Supports:
        - chart_type: bar, line, scatter, comparison, trend, heatmap
        - format: image, ascii, interactive
        - comparison_players: List of players for comparison charts
        - data_source: player stats, team data, league trends
        """
    
    async def execute(self, 
                     chart_type: str,
                     data_source: str,
                     format: str = "image",
                     comparison_players: List[str] = None,
                     **kwargs) -> ToolResult:
        
        if chart_type == "comparison" and comparison_players:
            return await self._create_comparison_chart(comparison_players, format)
        elif chart_type == "trend":
            return await self._create_trend_chart(data_source, format)
        else:
            return await self._create_standard_chart(chart_type, data_source, format)
```

### Solution 3: Batch API Data Manager

```python
# File: tools/realtime_tools.py (UPDATE EXISTING)
class BatchGameDataTool(BaseTool):
    """Replaces: live_scores, weather_updates, player_news (keep live_fantasy_tracker separate)"""
    
    def __init__(self):
        super().__init__()
        self.name = "get_game_day_data"
        self.cache_ttl = {
            "scores": 30,       # 30 seconds
            "weather": 600,     # 10 minutes
            "news": 300         # 5 minutes
        }
        self.cache = {}
    
    async def execute(self, 
                     data_types: List[str] = ["scores", "weather", "news"],
                     week: int = None,
                     **kwargs) -> ToolResult:
        
        # Check cache first
        cached_data = self._get_cached_data(data_types)
        missing_data = [dt for dt in data_types if dt not in cached_data]
        
        if missing_data:
            # Single API call for all missing data
            fresh_data = await self._batch_fetch(missing_data, week)
            self._update_cache(fresh_data)
            cached_data.update(fresh_data)
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=cached_data
        )
```

### Solution 4: Smart Model Selection

```python
# File: core/agent.py (UPDATE EXISTING)
class HowieAgent:
    def __init__(self):
        self.model_tiers = {
            "simple": {
                "model": "gpt-3.5-turbo",  # $0.002/1K tokens
                "tools": ["read_file", "list_files", "get_game_day_data"],
                "max_tokens": 1000
            },
            "analysis": {
                "model": "gpt-4",  # $0.03/1K tokens
                "tools": ["player_projection", "lineup_optimizer", "generate_sql_query"],
                "max_tokens": 2000
            },
            "creative": {
                "model": "claude-sonnet-4",  # For complex visualizations
                "tools": ["create_visualization", "create_report", "generate_analysis_script"],
                "max_tokens": 3000
            }
        }
    
    def select_model_for_task(self, query: str, selected_tools: List[str]) -> str:
        """Choose cheapest model that can handle the task"""
        for tier_name, tier_config in self.model_tiers.items():
            if any(tool in tier_config["tools"] for tool in selected_tools):
                return tier_config["model"]
        return "gpt-4"  # Default fallback
```

## 📋 Implementation Checklist

### Phase 1: Critical Fixes (Do First)
- [ ] **Verify function calling in agent.py**
  - Check if using `tools=` parameter in OpenAI calls
  - Replace any text parsing with native function calling
- [ ] **Implement hierarchical tool selection**
  - Add `FantasyToolManager` class
  - Update agent.py to use workflow-based selection
  - Test with 2-stage selection (workflow → tools)
- [ ] **Add comprehensive error handling**
  - Tool timeout handling
  - API failure recovery
  - Graceful degradation

### Phase 2: Performance Optimization (Do Second)
- [ ] **Consolidate visualization tools**
  - Replace 4 chart tools with 1 unified tool
  - Update tool descriptions and function signatures
  - Test backward compatibility
- [ ] **Implement batch API data fetching**
  - Combine live_scores + weather_updates + player_news
  - Add intelligent caching layer
  - Reduce API calls by 70%
- [ ] **Smart model selection**
  - Route simple queries to GPT-3.5-turbo
  - Reserve GPT-4/Claude for complex analysis
  - Track cost savings

### Phase 3: Advanced Features (Do Last)
- [ ] **Context-aware tool filtering**
  - Filter tools by season week (early/mid/playoff)
  - Boost relevant tools based on query intent
- [ ] **User behavior learning**
  - Track most-used tool combinations
  - Personalize tool recommendations
- [ ] **Performance monitoring**
  - Tool selection accuracy metrics
  - Response time tracking
  - Cost per query analysis

## 🎯 Expected Performance Improvements

| Metric | Current | Target | Improvement |
|--------|---------|---------|-------------|
| Tool Selection Accuracy | ~60% | 85% | +42% |
| Average Response Time | 8-12s | 4-7s | -50% |
| Cost per Query | $0.08-0.12 | $0.048 | -60% |
| API Calls per Query | 3-5 | 1-2 | -60% |
| Success Rate | ~70% | 90% | +29% |

## 🔧 Files to Modify

### High Priority
1. **core/agent.py** - Tool selection mechanism, model selection
2. **tools/visualization_tools.py** - Consolidate chart tools
3. **tools/realtime_tools.py** - Batch API calls, add caching

### Medium Priority  
4. **core/tool_manager.py** - New file for hierarchical selection
5. **requirements_enhanced.txt** - Add any new dependencies

### Low Priority
6. **howie_enhanced.py** - Update CLI to show performance metrics
7. **tests/** - Add tests for new tool selection logic

## 💡 Quick Wins (Implement Today)

1. **Add tool execution logging** - Track which tools are being selected
2. **Implement basic caching** - Cache player projections for 1 hour
3. **Consolidate chart tools** - Immediate 75% reduction in visualization tool count
4. **Add timeout handling** - Prevent hanging on slow API calls

## 🚀 Success Metrics to Track

```python
# Add to agent.py for monitoring
class PerformanceTracker:
    def track_tool_selection(self, query: str, selected_tool: str, success: bool):
        # Log tool selection accuracy
        pass
    
    def track_response_time(self, query: str, duration: float):
        # Monitor performance improvements
        pass
    
    def track_cost(self, model: str, tokens: int):
        # Monitor cost optimizations
        pass
```

## 🎪 Testing Strategy

1. **Tool Selection Accuracy Test**
   ```python
   test_queries = [
       "Should I start Justin Jefferson or Tyreek Hill?",  # → player_analysis workflow
       "Import my ESPN roster",                            # → roster_management workflow  
       "Show me live scores",                             # → game_time workflow
       "Generate a Python script to analyze RB trends"    # → research_deep_dive workflow
   ]
   ```

2. **Performance Benchmarking**
   - Measure before/after response times
   - Track API call reduction
   - Monitor cost per query

3. **Error Handling Validation**
   - Test with invalid inputs
   - Simulate API failures
   - Verify graceful degradation

---

## 🔗 Implementation Notes

- **Start with Phase 1** - Focus on tool selection accuracy first
- **Measure everything** - Track improvements quantitatively  
- **Test incrementally** - Don't change everything at once
- **Keep user experience** - Maintain all existing functionality

The foundation is excellent. These optimizations will transform ProjectHowie from a prototype into a production-ready, cost-effective fantasy football AI assistant.

**Key insight:** Most gains come from smarter tool selection, not smarter tools. Fix the selection mechanism first, then optimize individual tools.