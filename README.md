# 🏈 Howie CLI - Fantasy Football AI Command Center

> **Transform your fantasy football strategy with AI-powered analysis, real-time intelligence, and comprehensive draft tools.**

[![Version](https://img.shields.io/badge/version-2.3.0-brightgreen.svg)](https://github.com/tgurgick/projectHowie)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## 🚀 What's New in v2.3.0

### 🎯 **Complete Player Evaluation System**
- **`/player/Name`** - Comprehensive player analysis in vertical format
- **All-in-one reports**: Projections, ADP, tiers, SoS, team intelligence, draft recommendations
- **Mobile-optimized**: Perfect for narrow screens and draft apps

### 📊 **Advanced Draft Tools**  
- **`/adp`** - ADP rankings with multi-league round projections
- **`/adp/10`, `/adp/12`** - League-specific round targeting
- **`/tiers`** - Positional tier analysis with marginal value drops
- **Round-based strategy**: Know exactly when tiers typically get drafted

### 🤖 **AI-Powered Team Intelligence**
- **`/intel/team/position`** - Real-time scouting reports for every team/position
- **Claude + Perplexity workflow**: Web search → analysis → fact-checking
- **`/update intelligence`** - Refresh all 32 teams × 5 positions = 160 analyses
- **95%+ confidence scoring**: Verified by beat reporters and analysts

### ⚡ **Enhanced CLI Experience**
- **Fixed database paths**: Works from any directory
- **Early termination**: Prevents expensive API calls on typos
- **Bye week integration**: Complete schedule awareness
- **Clean color scheme**: Optimized for readability

## 📋 Quick Reference

### **🔍 Player Analysis**
```bash
/player                       # Show help
/player/Christian McCaffrey   # Complete CMC evaluation
/player/A.J. Brown           # Full WR analysis with all data
```

### **🎯 Draft Preparation**
```bash
/adp                         # ADP with round estimates
/adp/12                      # 12-team league targeting
/tiers                       # All position tier analysis
/tiers/rb                    # RB-specific tier breakdown
```

### **🧠 Team Intelligence**
```bash
/intel/list                  # Available intelligence data
/intel/SF/rb                 # 49ers RB situation analysis
/intel/PHI/wr               # Eagles WR depth chart insights
```

### **📊 Rapid Stats (All Positions)**
```bash
# 2025 Projections
/qb/projections  /rb/projections  /wr/projections  /te/projections

# Historical Stats (2018-2024)
/qb/td/2024     /rb/yards/2023   /wr/targets/2022  /te/rec/2021

# Strength of Schedule
/rb/sos/season  /wr/sos/playoffs  /qb/sos/all

# Special Stats
/def/sacks/2024  /k/fg/2023  /bye/12  /bye/7
```

## 🏆 Complete Feature Overview

### **🎯 Draft Command Center**
| Command | Description | Example |
|---------|-------------|---------|
| `/adp` | ADP rankings with round estimates | Shows top 50 with 10/12-team projections |
| `/adp/10` | League-specific rounds | ADP table + projected round for 10-team |
| `/tiers` | Tier value analysis | Marginal points between position tiers |
| `/tiers/rb` | Position-specific tiers | Detailed RB tier breakdown |
| `/player/Name` | Complete evaluation | All data points in mobile format |

### **🧠 Intelligence System**
| Command | Description | Coverage |
|---------|-------------|----------|
| `/intel/list` | Available data | All teams/positions with update times |
| `/intel/team/pos` | Detailed report | Usage, injuries, depth chart, coaching |
| `/update intelligence` | Refresh all data | 32 teams × 5 positions with fact-checking |

### **📊 Comprehensive Stats (All Positions)**
| Position | Commands | Data Range |
|----------|----------|------------|
| **QB** | `/qb/td`, `/qb/yards`, `/qb/projections` | 2018-2025 |
| **RB** | `/rb/td`, `/rb/yards`, `/rb/projections` | 2018-2025 |
| **WR** | `/wr/td`, `/wr/yards`, `/wr/targets` | 2018-2025 |
| **TE** | `/te/td`, `/te/yards`, `/te/targets` | 2018-2025 |
| **DEF** | `/def/sacks`, `/def/int`, `/def/projections` | 2018-2025 |
| **K** | `/k/fg`, `/k/xp`, `/k/projections` | 2025 |

### **📅 Schedule Intelligence**
- **Bye weeks**: `/bye/7`, `/bye/12` - All players on bye
- **Strength of Schedule**: `/pos/sos/season`, `/pos/sos/playoffs`
- **ADP integration**: Bye weeks shown in all draft displays

## 🔧 Installation & Setup

### **📦 Quick Install**
```bash
# Clone repository
git clone https://github.com/tgurgick/projectHowie.git
cd projectHowie

# Install dependencies
pip install -r requirements_enhanced.txt

# Set up API keys (required)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export PERPLEXITY_API_KEY="your-perplexity-key"

# Build databases (one-time setup)
python scripts/build_fantasy_db.py
python scripts/build_pff_projections.py
python scripts/build_defensive_stats.py
```

### **🗄️ Database Setup**
```bash
# Import 2025 projections and SoS
python scripts/build_pff_projections.py

# Import historical defensive stats (2018-2024)
python scripts/build_defensive_stats.py

# Import ADP data
python scripts/build_fantasypros_adp.py

# Verify all databases
python tests/verify_all_databases.py
```

## 🎮 Usage Examples

### **🔍 Complete Player Evaluation**
```bash
> /player/Saquon Barkley

🔍 Comprehensive Analysis: Saquon Barkley
════════════════════════════════════════════════════════════
📋 Saquon Barkley (RB, PHI)

📊 2025 Projections          🎯 Draft Position (ADP)
Fantasy Points: 251.3        Overall ADP: 12.4
Games: 17                    10-team: 2.02
Rush TD: 11.2                12-team: 2.00
Rush Yards: 1,247            14-team: 1.14

🏆 Positional Tier          📅 Strength of Schedule (RB)
Tier: Tier 1 (Elite)       Season: Easy (7.2)
Rank: #6 of 120 RBs         Playoffs: Average (5.8)

🧠 Team Intelligence (PHI RB)
Confidence: 92%
Key Players: Saquon Barkley, Kenneth Gainwell, Will Shipley
Usage: Expects 280+ carries in Sirianni's RB-friendly offense

💡 Draft Recommendation
Action: TARGET EARLY
Confidence: High
Reasoning: Elite tier player, favorable team situation, strong ADP value
```

### **🎯 Draft Strategy Session**
```bash
> /tiers

📊 Positional Tier Analysis - Marginal Value Drops
════════════════════════════════════════════════════════════
Position    Tier 1→2   Tier 2→3   Tier 3→4   Tier 4→5
────────────────────────────────────────────────────────────
QB          -18.3      -12.1      -8.4       -6.2
RB          -31.7      -19.4      -14.8      -11.2
WR          -24.1      -16.3      -12.7      -9.8
TE          -28.9      -15.2      -8.1       -4.3

📋 Draft Strategy Insights:
• RB scarcity most pronounced - secure elite talent early
• TE shows elite-or-bust pattern - target Tier 1 or wait
• QB depth allows streaming - safe to wait until mid-rounds
• WR talent distributed - multiple tiers offer value

Round Analysis (10-team/12-team):
• RB Tier 1: Rounds 1-2, gone by Round 4
• TE Tier 1: Round 3-5, wait for Tier 3 in Round 10+
• QB Tier 1: Round 4-6, Tier 2 still available Round 10+
```

### **🧠 Team Intelligence Deep Dive**
```bash
> /intel/SF/rb

SF RB Intelligence Report
Status: verified | Confidence: 95% | Updated: 2025-08-28
╭─────────────────────── 📊 Analysis Summary ────────────────────────╮
│ The San Francisco 49ers RB room is anchored by Christian McCaffrey │
│ with Jordan Mason as the primary backup. Shanahan's zone scheme     │
│ creates opportunities for any back, but CMC's versatility makes     │
│ him irreplaceable in the offensive system.                         │
╰─────────────────────────────────────────────────────────────────────╯

👥 Key Players: Christian McCaffrey, Jordan Mason, Isaac Guerendo
📈 Usage: CMC 85% snaps, 22+ touches/game in healthy games
🎯 Coaching: Shanahan zone scheme, heavy RB utilization
🏥 Injuries: CMC managing Achilles; expected full workload
```

## 🔧 Advanced Features

### **🤖 Multi-Model AI System**
- **Claude Sonnet 4**: Complex analysis, team intelligence, web search
- **GPT-4o**: General queries, rapid stats, cost-effective operations  
- **Perplexity Pro**: Real-time research, fact-checking, injury updates
- **Automatic routing**: Best model selected for each task type

### **📊 Data Sources**
- **PFF Projections**: 2025 player projections and strength of schedule
- **FantasyPros ADP**: Real-time average draft position data
- **NFL Historical**: Complete stats 2018-2024 via nfl_data_py
- **Real-time Intelligence**: Live injury reports, depth chart changes

### **💾 Database Architecture**
```
data/fantasy_ppr.db
├── player_projections      # 2025 PFF projections
├── adp_data                # FantasyPros ADP 
├── strength_of_schedule    # PFF SoS by position
├── player_defensive_stats  # Historical defense (2018-2024)
├── team_defensive_stats    # Team defense stats
└── team_position_intelligence # AI scouting reports
```

## 🛠️ Configuration

### **⚙️ Environment Setup**
```bash
# Required API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."

# Optional Configuration
export HOWIE_MODEL="claude-sonnet-4"    # Default model
export HOWIE_WORKSPACE="~/fantasy"      # Custom workspace
```

### **🎛️ Model Configuration**
```bash
# In Howie CLI
> /model/info                    # Show available models
> /model/switch gpt-4o          # Switch current model  
> @claude-sonnet-4 <query>      # Use specific model once
> /cost/info                    # Show usage and costs
```

## 📈 Performance & Reliability

### **✅ Quality Assurance**
- **Fact-checking**: Perplexity Pro verifies AI analysis
- **Confidence scoring**: 0-100% reliability metrics
- **Data validation**: Multiple source cross-referencing
- **Error handling**: Graceful fallbacks and clear error messages

### **⚡ Speed Optimizations**
- **Database caching**: Instant rapid stats lookup
- **Parallel tool execution**: Multiple operations simultaneously  
- **Smart model routing**: Cost-effective model selection
- **Path resolution**: Works from any directory

### **💰 Cost Management**
- **Usage tracking**: Real-time cost monitoring per model
- **Smart routing**: Expensive models only when needed
- **Efficient caching**: Reduce redundant API calls
- **Budget alerts**: Cost estimation and warnings

## 🔮 Roadmap & Future Features

### **📅 Coming Soon**
- **Live draft assistant**: Real-time pick recommendations
- **Trade analyzer**: Multi-team trade evaluation
- **Waiver wire AI**: Priority recommendations with reasoning
- **League-specific analysis**: Custom scoring and roster settings
- **Mobile app**: Native iOS/Android companion

### **🎯 Planned Enhancements**
- **Dynasty league support**: Multi-year player valuations
- **Playoff optimizer**: Week 15-17 specific strategies
- **Injury impact analysis**: Replacement value calculations
- **Weather integration**: Game condition impact analysis

## 🤝 Contributing

### **🔧 Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/tgurgick/projectHowie.git
cd projectHowie
pip install -r requirements_enhanced.txt
python -m pytest tests/
```

### **📝 Contributing Guidelines**
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Update documentation** for new commands
5. **Submit** pull request with detailed description

### **🐛 Bug Reports**
- Use GitHub Issues with detailed reproduction steps
- Include Howie version (`/version` command)
- Attach relevant log output (`/logs detailed`)

## 📄 License & Acknowledgments

### **📋 License**
MIT License - see [LICENSE](LICENSE) file for details

### **🙏 Acknowledgments**
- **AI Models**: OpenAI GPT-4o, Anthropic Claude Sonnet 4, Perplexity Pro
- **Data Sources**: nfl_data_py, FantasyPros, Pro Football Focus (PFF)
- **Inspiration**: Anthropic's Claude interface design
- **Creator**: Trevor Gurgick ([@tgurgick](https://github.com/tgurgick))

---

## 🎯 Transform Your Fantasy Strategy

**Howie CLI v2.3.0** isn't just a tool—it's your complete fantasy football command center. From comprehensive player analysis to AI-powered team intelligence, every feature is designed to give you the competitive edge.

**Ready to dominate your league?** 

```bash
git clone https://github.com/tgurgick/projectHowie.git && cd projectHowie
pip install -r requirements_enhanced.txt
python howie_enhanced.py
```

> 🏆 **Join the Howie community** and revolutionize your fantasy football strategy with AI-powered analysis.
