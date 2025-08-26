#!/bin/bash
# Installation script for Howie CLI

echo "🏈 Installing Howie CLI - Fantasy Football AI Assistant"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Found Python $python_version"

# Check if we're in a virtual environment (recommended)
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: You're not in a virtual environment"
    echo "   It's recommended to create one:"
    echo "   python3 -m venv howie-env"
    echo "   source howie-env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install package in development mode
echo "📦 Installing Howie CLI package..."
pip install -e .

if [[ $? -eq 0 ]]; then
    echo "✅ Howie CLI installed successfully!"
    echo ""
    echo "Available commands:"
    echo "  howie          - Enhanced multi-model CLI"
    echo "  howie-classic  - Original CLI"  
    echo "  howie-demo     - Agent demonstration"
    echo ""
    echo "🔑 Next steps:"
    echo "1. Set up your API keys:"
    echo "   export OPENAI_API_KEY='sk-...'"
    echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "   export PERPLEXITY_API_KEY='pplx-...'"
    echo ""
    echo "2. Test the installation:"
    echo "   howie --help"
    echo ""
    echo "3. Start using Howie:"
    echo "   howie chat"
    echo ""
    echo "For more info, see README_ENHANCED.md"
else
    echo "❌ Installation failed"
    echo "Try installing dependencies manually:"
    echo "pip install -r requirements_multimodel.txt"
    exit 1
fi