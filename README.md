# 🏈 Howie - Fantasy Football AI Command Center

> **Transform your fantasy football strategy with AI-powered analysis, real-time intelligence, and comprehensive draft tools.**
> 
> **🖥️ TUI Primary Interface** | **💻 CLI Fallback** | **📱 Mobile Optimized**

[![Version](https://img.shields.io/badge/version-2.5.0-brightgreen.svg)](https://github.com/tgurgick/projectHowie)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## 🚀 What's New in v2.5.0

### 🎯 **Three-Layer Draft Analysis System**
- **Pick Recommendations** - Top 10 picks for any round with 6-factor scoring
- **Strategy Tree Search** - Optimal 16-round draft plan with unbiased opponent simulation
- **Monte Carlo Evaluation** - Performance testing against realistic opponent scenarios

### 🧠 **Advanced Draft Intelligence**
- **`/draft analyze`** - Complete pre-draft analysis with round-by-round recommendations
- **`/draft strategy generate`** - Optimal draft strategy using tree search algorithm
- **`/draft monte 25 8`** - Monte Carlo simulation with diverse opponent personalities
- **Unbiased Strategy Creation** - Tree search independent of Monte Carlo variance

### 📊 **Enhanced Player Evaluation**
- **6-Factor Scoring**: VORP, positional scarcity, roster fit, SoS, starter status, injury risk
- **533 Players**: Complete FantasyPros ADP integration (86.7% coverage)
- **Realistic Availability**: Opponent simulation prevents unrealistic draft scenarios
- **Enhanced Intelligence**: AI-gathered scouting data for all teams/positions

### 🎯 **Complete Player Evaluation System**
- **`/player/Name`** - Comprehensive player analysis in vertical format
- **All-in-one reports**: Projections, ADP, tiers, SoS, team intelligence, draft recommendations
- **Mobile-optimized**: Perfect for narrow screens and draft apps

### 🤖 **AI-Powered Team Intelligence**
- **`/intel/team/position`** - Real-time scouting reports for every team/position
- **Claude + Perplexity workflow**: Web search → analysis → fact-checking
- **`/update intelligence`** - Refresh all 32 teams × 5 positions = 160 analyses
- **95%+ confidence scoring**: Verified by beat reporters and analysts

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

### **🎲 Draft Analysis & Strategy**
```bash
/draft analyze                       # Complete pre-draft analysis
/draft strategy generate             # Optimal 16-round strategy
/draft monte 25 8                    # Monte Carlo evaluation (25 sims, 8 rounds)
/draft view                          # View saved strategies & results
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
| `/draft analyze` | Complete pre-draft analysis | Round-by-round recommendations for all 16 rounds |
| `/draft strategy generate` | Optimal draft strategy | Tree search finds best 16-round plan |
| `/draft monte 25 8` | Monte Carlo evaluation | Test strategy against 25 realistic scenarios |
| `/draft view` | View saved results | Access strategies & Monte Carlo history |
| `/adp` | ADP rankings with round estimates | Shows top 50 with 10/12-team projections |
| `/tiers` | Tier value analysis | Marginal points between position tiers |
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

### **📦 Portable Install (Recommended)**
```bash
# Clone repository
git clone https://github.com/tgurgick/projectHowie.git
cd projectHowie

# Create virtual environment
python -m venv howie-env
source howie-env/bin/activate  # On Windows: howie-env\Scripts\activate

# Run portable installer
./install_portable.sh

# Use from anywhere!
cd ~/Desktop
howie                    # 🖥️  Launch TUI (primary interface)
howie-cli chat           # 💻 Launch CLI (fallback)
```

### **🔑 API Keys Setup**
```bash
# Create .env file in your home directory
echo "OPENAI_API_KEY=your-openai-key" >> ~/.env
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> ~/.env  # Optional
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

### **🎯 Draft Strategy Generation**
```bash
> /draft strategy generate

🌳 Starting strategy tree search...
📋 League: 12T HALF_PPR, position #8
🎯 Using unbiased ADP-based opponent simulation for strategy selection
✅ Strategy tree search completed in 12.3 seconds
🎲 Evaluating strategy performance with Monte Carlo...

🎯 OPTIMAL STRATEGY OVERVIEW
══════════════════════════════════════════════════════════
Round  1: WR - Elite receiver value (CeeDee Lamb tier)
Round  2: QB - Secure premium QB position (Josh Allen)
Round  3: RB - Address scarcity concern (Chase Brown)
Round  4: WR - Build receiving depth (Jordan Addison)
Round  5: TE - Premium tight end value (Brock Bowers)
Round  6: RB - Backfield depth (Zack Moss)
Round  7: WR - Target volume receiver (Courtland Sutton)
Round  8: RB - Handcuff/upside play (Blake Corum)

💡 Key Insights:
• 🛡️ Conservative approach - prioritizing safe floors
• 🎯 Position scarcity drives early RB selections  
• 📊 VORP optimization balances value vs need
• 🏈 Realistic opponent simulation prevents overdrafts

Expected Roster Value: 2,847 points | Confidence: 91.6%
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

> 📖 **For technical details**, see [DRAFT_SYSTEM_ARCHITECTURE.md](DRAFT_SYSTEM_ARCHITECTURE.md) for complete documentation of the three-layer analysis system.

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

### **📅 Coming Soon (v2.5.0)**
- **Enhanced Strategy Tree Search**: Multi-outcome rollouts with UCB1 selection
- **Player Outcome Distributions**: Age/tenure variance with injury overlay  
- **Pre-sampled Outcomes Matrix**: 15K+ season simulations for performance
- **Weekly Lineup Optimization**: Season scoring with optimal lineup decisions
- **Strategy Comparison Tool**: A/B testing different draft approaches

### **🎯 Draft Simulation Enhancements**
- **Live draft assistant**: Real-time pick recommendations during drafts
- **Trade analyzer**: Multi-team trade evaluation with VORP analysis
- **Keeper league support**: Draft pick replacement and valuation
- **League-specific analysis**: Custom scoring and roster settings

### **🎮 Advanced Features**
- **Dynasty league support**: Multi-year player valuations
- **Playoff optimizer**: Week 15-17 specific strategies
- **Injury impact analysis**: Replacement value calculations
- **Weather integration**: Game condition impact analysis
- **Mobile app**: Native iOS/Android companion

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

**Howie CLI v2.4.0** isn't just a tool—it's your complete fantasy football command center. From three-layer draft analysis to AI-powered team intelligence, every feature is designed to give you the competitive edge.

**Ready to dominate your league?** 

```bash
git clone https://github.com/tgurgick/projectHowie.git && cd projectHowie
pip install -r requirements_enhanced.txt
python howie_enhanced.py
```

> 🏆 **Join the Howie community** and revolutionize your fantasy football strategy with AI-powered analysis.
