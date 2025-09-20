# 🏈 Howie v2.5.0 Release Notes

## 🎯 Major Updates

### **🖥️ TUI-First Interface**
- **Primary Interface**: `howie` now launches the TUI (Terminal User Interface)
- **CLI Fallback**: `howie-cli` provides command-line interface as backup
- **Auto-scroll**: TUI automatically scrolls to bottom after command execution
- **Session Logging**: All TUI sessions automatically logged to `~/.howie/data/logs/`

### **📦 Portable Installation**
- **Install Once, Use Anywhere**: `./install_portable.sh` creates system-wide installation
- **User Data Directory**: All data stored in `~/.howie/data/` (portable across directories)
- **Database Auto-Migration**: Databases automatically copied to user directory on first run
- **Clean Separation**: User data separate from application code

### **🎯 Draft System Enhancements**
- **Strategy Loading Fixed**: `/draft/strategy/1` now properly displays strategy details in TUI
- **Portable Paths**: All draft data (strategies, Monte Carlo results) use portable user directories
- **Configuration Restored**: Basic league and keeper configurations included as templates
- **16-Round Depth**: Full roster draft strategies and recommendations

### **🔒 Security & Cleanup**
- **No Sensitive Data**: All API keys, .env files, and user configs removed from repository
- **Clean Repository**: Removed all test files, build artifacts, old documentation
- **Production Ready**: Repository safe for public sharing and distribution
- **Essential Files Only**: Streamlined to core application and data files

## 🔧 Technical Improvements

### **Banner & UI**
- **Fixed ASCII Banner**: Proper solid block letters with traditional W shape
- **Consistent Styling**: Kelly green (#4CBB17) coloring maintained
- **Aligned Letters**: All letters properly sized and aligned

### **Path Management**
- **Unified Path System**: All components use `howie_cli.core.paths` for portable paths
- **Database Resolution**: Automatic database discovery and fallback paths
- **Strategy Management**: Strategies saved to `~/.howie/data/strategies/`
- **Monte Carlo Results**: Results saved to `~/.howie/data/monte_carlo_results/`

### **Package Management**
- **Entry Points**: Proper `howie` (TUI) and `howie-cli` (CLI) console scripts
- **MANIFEST.in**: Correct packaging manifest for distribution
- **Requirements**: Clean dependency management

## 📊 Data & Configuration

### **Default Configurations**
- **League Config**: 12-team, Half-PPR, Position #8, 3 WR league
- **Keeper Config**: Brian Thomas Jr. (WR, JAX) kept in 8th round
- **Roster Structure**: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 K, 1 DST, 6 bench

### **Database Files**
- **Multi-Scoring**: PPR, Half-PPR, Standard scoring databases
- **Player Distributions**: 15k outcome matrix for Monte Carlo simulations
- **PFF Data**: 25 projection and historical CSV files

## 🚀 Installation & Usage

### **Quick Start**
```bash
git clone https://github.com/tgurgick/projectHowie.git
cd projectHowie
./install_portable.sh

# Use from anywhere
howie                    # Launch TUI (primary)
howie-cli chat           # Launch CLI (fallback)
```

### **API Keys Setup**
```bash
echo "OPENAI_API_KEY=your-key" >> ~/.env
echo "ANTHROPIC_API_KEY=your-key" >> ~/.env
```

## 🔄 Migration Notes

- **Existing Users**: Run `./install_portable.sh` to migrate to portable installation
- **Data Migration**: Databases automatically copied to `~/.howie/data/` on first run
- **Configuration**: Basic configs included, customize with `/draft/config`
- **Logs**: TUI logs now in `~/.howie/data/logs/` instead of project directory

## 🎯 What's Next

- **Strategy Generation**: Use `/draft/strategy/generate` to create your first strategy
- **Monte Carlo**: Run `/draft/monte 25 8` for draft simulations
- **Player Analysis**: Try `/player/PlayerName` for detailed analysis
- **Draft Help**: Use `/draft/help` to see all available commands

---

**This release transforms Howie into a production-ready, portable fantasy football AI system with a modern TUI interface and clean, secure codebase.**
