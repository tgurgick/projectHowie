#!/bin/bash
# Portable Installation Script for Howie CLI + TUI
# This installs Howie so it can be run from anywhere

echo "🏈 Installing Howie CLI + TUI (Portable Version)"
echo "================================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Recommended to use venv."
    echo "   To create one: python -m venv howie-env && source howie-env/bin/activate"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install in development mode (editable install)
echo "📦 Installing Howie in development mode..."
pip install -e .

# Check if installation was successful
if command -v howie &> /dev/null; then
    echo "✅ TUI installed successfully! Try: howie"
else
    echo "❌ TUI installation failed"
    exit 1
fi

if command -v howie-cli &> /dev/null; then
    echo "✅ CLI installed successfully! Try: howie-cli --help"
else
    echo "❌ CLI installation failed"
    exit 1
fi

# Show usage information
echo ""
echo "🎯 Installation Complete!"
echo "========================"
echo ""
echo "🖥️  TUI Usage (PRIMARY - from anywhere):"
echo "   howie                    # Launch TUI (main interface)"
echo ""
echo "💻 CLI Usage (FALLBACK - from anywhere):"
echo "   howie-cli                # Start interactive CLI"
echo "   howie-cli chat           # Start chat mode"
echo "   howie-cli tui            # Launch TUI from CLI"
echo ""
echo "📊 Data Location:"
echo "   ~/.howie/data/           # Database and user files"
echo "   ~/.howie/data/logs/      # Session logs"
echo ""
echo "🔧 Environment Setup:"
echo "   Set API keys in ~/.howie/.env or project .env"
echo "   OPENAI_API_KEY=your_key_here"
echo "   ANTHROPIC_API_KEY=your_key_here"
echo ""
echo "Ready to draft! 🏈"
