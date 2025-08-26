# Howie CLI - Claude-like Fantasy Football AI Assistant

## Overview

Howie CLI is an advanced, Claude-inspired AI assistant specifically designed for fantasy football analysis. It combines the power of GPT-4 with a comprehensive tool system to provide file operations, data visualization, code generation, real-time updates, and machine learning predictions.

## 🚀 New Features (v2.0)

### Core Capabilities
- **🗂️ File Operations**: Read/write CSV, Excel, JSON files; import rosters from any platform
- **📊 Visualization**: Generate charts, comparisons, trends (both image and ASCII)
- **💻 Code Generation**: Create Python analysis scripts and SQL queries on demand
- **📡 Real-time Data**: Live scores, player news, weather updates, fantasy tracking
- **🤖 ML Predictions**: Advanced projections, lineup optimization, risk analysis
- **💾 Context Persistence**: Maintains conversation context and session history
- **🏗️ Workspace Management**: Organized file handling and report generation

## Installation

```bash
# Clone the repository
git clone https://github.com/tgurgick/projectHowie.git
cd projectHowie

# Install enhanced version
pip install -r requirements_enhanced.txt

# Or install as package
python setup.py install

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Interactive Chat Mode (Default)
```bash
# Start interactive chat
python howie.py

# Resume previous session
python howie.py chat --resume

# Use specific model
python howie.py chat --model gpt-4o
```

### Single Commands
```bash
# Ask a question
python howie.py ask "Who should I start: Justin Jefferson or Tyreek Hill?"

# Import roster
python howie.py import-roster my_roster.csv --platform espn

# Compare players
python howie.py compare "CeeDee Lamb" "Ja'Marr Chase" --visual

# Generate projections
python howie.py project "Saquon Barkley" --weeks 3 --method ml

# Optimize lineup
python howie.py optimize --roster-file my_team.csv --method balanced

# Get live scores
python howie.py live --week 10

# Generate analysis code
python howie.py generate "analyze RB performance trends" --type script
```

## Tool Categories

### 1. File Operations
- **read_file**: Read CSV, Excel, JSON, text files
- **write_file**: Save data in various formats
- **import_roster**: Import fantasy rosters from any platform
- **create_report**: Generate formatted analysis reports
- **list_files**: Browse workspace files

### 2. Visualization
- **create_chart**: Bar, line, scatter, heatmap charts
- **player_comparison_chart**: Visual player comparisons
- **season_trend_chart**: Trend analysis over time
- **ascii_chart**: Terminal-friendly visualizations

### 3. Code Generation
- **generate_analysis_script**: Create Python analysis scripts
- **generate_sql_query**: Build SQL queries from natural language

### 4. Real-time Data
- **live_scores**: Current NFL game scores
- **player_news**: Latest injury and trade updates
- **weather_updates**: Game-time weather conditions
- **live_fantasy_tracker**: Track your roster's live points

### 5. ML Predictions
- **player_projection**: Advanced ML-based projections
- **lineup_optimizer**: Optimize lineups using multiple strategies

## Example Workflows

### Complete Roster Analysis
```python
# In chat mode
> Import my roster from espn_roster.csv
> Analyze my RB depth
> Project my team's total points for week 10
> Suggest lineup optimizations
> Create a visual report of my team's strengths and weaknesses
```

### Player Research
```python
# In chat mode
> Compare Garrett Wilson vs Chris Olave for the rest of season
> Show me their target share trends
> Generate a chart comparing their performances
> What are the weather conditions for their next games?
> Give me ML projections for both players
```

### Custom Analysis Script
```python
# In chat mode
> Generate a Python script to analyze WR performance against zone coverage
> Modify it to include slot vs outside alignment
> Save the script as wr_zone_analysis.py
> Run the script on 2024 data
```

### Live Game Tracking
```python
# In chat mode
> Show me live scores for all games
> Track my roster's live fantasy points
> Alert me to any injury updates
> How is weather affecting the GB vs CHI game?
```

## Advanced Features

### Context Management
- Sessions are automatically saved
- Resume previous conversations with `--resume`
- View context with `context` command in chat
- Export session history for analysis

### Workspace Organization
```
~/.howie/
├── workspace/         # Working files
│   ├── session_*/     # Per-session files
│   ├── charts/        # Generated visualizations
│   ├── reports/       # Analysis reports
│   └── scripts/       # Generated code
└── sessions/          # Saved conversations
```

### ML Model Types
- **Linear**: Fast, simple projections
- **Random Forest**: Robust predictions with uncertainty
- **Gradient Boosting**: High-accuracy predictions
- **Ensemble**: Combines multiple models for best results

## Architecture

```
howie_cli/
├── core/
│   ├── agent.py           # Main AI agent
│   ├── context.py         # Conversation context
│   ├── workspace.py       # File management
│   └── base_tool.py       # Tool framework
├── tools/
│   ├── file_tools.py      # File operations
│   ├── visualization_tools.py
│   ├── code_generation_tools.py
│   ├── realtime_tools.py
│   └── ml_projection_tools.py
└── __init__.py
```

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."
export HOWIE_MODEL="gpt-4o"  # Default model
export HOWIE_WORKSPACE="~/fantasy_analysis"  # Custom workspace
```

### Settings File
Create `~/.howie/config.json`:
```json
{
  "model": "gpt-4o",
  "scoring_type": "ppr",
  "risk_tolerance": "balanced",
  "favorite_teams": ["SF", "BUF"],
  "analysis_depth": "detailed"
}
```

## Comparison to Original

| Feature | Original | Enhanced (v2.0) |
|---------|----------|-----------------|
| Core Function | Q&A Chatbot | Full AI Assistant |
| File Operations | ❌ | ✅ Read/Write/Import |
| Visualizations | ❌ | ✅ Charts & Graphs |
| Code Generation | ❌ | ✅ Scripts & SQL |
| Real-time Data | ❌ | ✅ Live Updates |
| ML Predictions | Basic | Advanced Multi-model |
| Context Memory | ❌ | ✅ Persistent Sessions |
| Tool System | ❌ | ✅ Extensible Framework |
| Workspace | ❌ | ✅ Organized Files |

## Development

### Adding New Tools
```python
from howie_cli.core.base_tool import BaseTool, ToolResult, ToolStatus

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "my_tool"
        self.category = "custom"
        self.description = "Does something useful"
        
    async def execute(self, **kwargs) -> ToolResult:
        # Tool logic here
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"result": "success"}
        )
```

### Running Tests
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Module Import Errors**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

3. **Database Connection Issues**
   ```bash
   # Rebuild databases
   python scripts/build_fantasy_db.py
   ```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file

## Acknowledgments

- Inspired by Anthropic's Claude
- Built on OpenAI's GPT-4
- Fantasy data from nfl_data_py
- Original projectHowie concept by Trevor Gurgick

## Roadmap

- [ ] Voice interface
- [ ] Web UI companion
- [ ] Mobile app
- [ ] Multi-league portfolio management
- [ ] DFS optimization
- [ ] Betting insights integration
- [ ] Community tool marketplace

---

**Note**: This is a significant enhancement over the original projectHowie, transforming it from a Q&A chatbot into a comprehensive AI assistant with Claude-like capabilities tailored for fantasy football.