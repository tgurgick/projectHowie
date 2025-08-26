# Installation Guide for Howie CLI

## 🚀 Quick Install (Recommended)

### Option 1: Package Installation

```bash
# From the projectHowie directory
./install.sh

# Then you can use:
howie          # Enhanced multi-model CLI
howie-classic  # Original CLI
howie-demo     # Agent demonstration
```

### Option 2: Manual Package Install

```bash
# Install as editable package
pip install -e .

# Now available as commands:
howie --help
howie chat
howie models
```

### Option 3: Shell Aliases

```bash
# Set up aliases in your shell
./setup_aliases.sh

# Restart terminal or run:
source ~/.zshrc  # or ~/.bashrc

# Then use:
howie
howie-classic
howie-demo
```

## 📋 Prerequisites

### Python Requirements
- **Python 3.8+** (3.10+ recommended)
- **pip** package manager

### API Keys (at least one required)
```bash
export OPENAI_API_KEY="sk-..."           # For GPT models
export ANTHROPIC_API_KEY="sk-ant-..."    # For Claude models
export PERPLEXITY_API_KEY="pplx-..."     # For research models
```

## 🔧 Detailed Installation Steps

### Step 1: Clone/Navigate to Project
```bash
cd /Users/trevor.gurgick/projectHowie
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python3 -m venv howie-env
source howie-env/bin/activate  # On Windows: howie-env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_multimodel.txt
```

### Step 4: Install Package
```bash
# Option A: Editable install (for development)
pip install -e .

# Option B: Regular install  
pip install .
```

### Step 5: Set Up API Keys
```bash
# Add to your shell profile (~/.zshrc, ~/.bashrc, etc.)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export PERPLEXITY_API_KEY="your-perplexity-key"

# Or create .env file (not recommended for production)
echo "OPENAI_API_KEY=your-key" > .env
```

### Step 6: Test Installation
```bash
howie --version
howie --help
howie models  # List available models
```

## 🎯 Installation Methods Comparison

| Method | Command | Pros | Cons |
|--------|---------|------|------|
| **Package Install** | `pip install -e .` | Global `howie` command | Requires pip setup |
| **Shell Aliases** | `./setup_aliases.sh` | Simple, flexible | Terminal-specific |
| **Direct Run** | `python howie_enhanced.py` | No setup needed | Must be in directory |

## 🛠️ Post-Installation Setup

### Configure Models
```bash
# Interactive configuration
howie configure

# Or manually create config
cp models_config_example.json ~/.howie/models.json
```

### Test Different Models
```bash
# Test OpenAI
howie ask "Hello" --model gpt-4o-mini

# Test Anthropic (if key configured)
howie ask "Hello" --model claude-3-haiku

# Test Perplexity (if key configured)  
howie ask "What's in the news today?" --model perplexity-sonar
```

### Import Existing Data
```bash
# Your existing databases should work automatically
howie ask "Show me database info"
```

## 🔍 Verification Steps

### 1. Check Installation
```bash
which howie
howie --version
```

### 2. Test Core Functionality
```bash
howie models                    # List models
howie ask "Hello"               # Basic query
howie tools                     # List tools
```

### 3. Test Multi-Model Features
```bash
# Automatic model selection
howie ask "Research current NFL news"    # Should use Perplexity
howie ask "Generate Python code"         # Should use Claude
howie ask "List players"                 # Should use fast model

# Manual model selection
howie ask "@perplexity-sonar What happened today?"
howie ask "@claude-3-5-sonnet Create a function"
```

### 4. Test Agent System
```bash
howie spawn "Research top waiver wire pickups"
howie-demo  # Run agent demonstrations
```

## 🚨 Troubleshooting

### Command Not Found
```bash
# If 'howie' command not found:

# Option 1: Check PATH
echo $PATH
which howie

# Option 2: Reinstall package
pip uninstall howie-cli
pip install -e .

# Option 3: Use aliases
./setup_aliases.sh
source ~/.zshrc

# Option 4: Run directly
python3 howie_enhanced.py
```

### API Key Issues
```bash
# Check if keys are set
env | grep API_KEY

# Test specific provider
howie ask "test" --model gpt-4o-mini        # OpenAI
howie ask "test" --model claude-3-haiku     # Anthropic  
howie ask "test" --model perplexity-sonar   # Perplexity
```

### Import Errors
```bash
# Check dependencies
pip list | grep -E "(openai|anthropic|rich|click)"

# Reinstall dependencies
pip install -r requirements_multimodel.txt

# Check Python path
python3 -c "import howie_cli; print('OK')"
```

### Database Access Issues
```bash
# Check if databases exist
ls -la data/*.db

# Test database access
howie ask "Show database info"
```

## 🔄 Updating

### Update Package
```bash
cd /Users/trevor.gurgick/projectHowie
git pull  # If using git
pip install -e . --upgrade
```

### Update Dependencies
```bash
pip install -r requirements_multimodel.txt --upgrade
```

## 🗑️ Uninstallation

### Remove Package
```bash
pip uninstall howie-cli
```

### Remove Aliases
```bash
# Edit your shell config and remove Howie aliases
nano ~/.zshrc  # or ~/.bashrc
```

### Remove User Data
```bash
rm -rf ~/.howie  # Removes configs, sessions, workspace
```

## 📱 Platform-Specific Notes

### macOS
```bash
# May need to install command line tools
xcode-select --install

# Use homebrew Python if needed
brew install python3
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

### Windows
```powershell
# Use Windows Subsystem for Linux (WSL) or
# Install Python from python.org
# Use pip normally
```

## 🎉 Success Indicators

You'll know the installation worked when:

1. ✅ `howie --help` shows the command help
2. ✅ `howie models` lists available AI models  
3. ✅ `howie ask "Hello"` gets a response
4. ✅ `howie tools` shows all available tools
5. ✅ Your existing database queries work: `howie ask "Show database info"`

## 🆘 Getting Help

If you run into issues:

1. Check the [MULTIMODEL_GUIDE.md](MULTIMODEL_GUIDE.md) for usage help
2. Check the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for database compatibility
3. Run `howie-test` to diagnose issues
4. Check API key configuration
5. Try running `python3 howie_enhanced.py` directly

The installation should be straightforward, and all your existing data will remain accessible!