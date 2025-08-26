#!/bin/bash
# Setup shell aliases for Howie CLI

HOWIE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🏈 Setting up Howie CLI aliases"
echo "=============================="

# Detect shell
if [[ -n "$ZSH_VERSION" ]]; then
    SHELL_RC="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [[ -n "$BASH_VERSION" ]]; then
    SHELL_RC="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    echo "⚠️  Could not detect shell type"
    echo "Please add aliases manually to your shell configuration"
    exit 1
fi

echo "Detected $SHELL_NAME shell: $SHELL_RC"

# Backup existing rc file
cp "$SHELL_RC" "$SHELL_RC.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Backed up existing $SHELL_RC"

# Add Howie aliases
echo "" >> "$SHELL_RC"
echo "# Howie CLI aliases" >> "$SHELL_RC"
echo "alias howie='python3 $HOWIE_PATH/howie_enhanced.py'" >> "$SHELL_RC"
echo "alias howie-classic='python3 $HOWIE_PATH/howie.py'" >> "$SHELL_RC"
echo "alias howie-demo='python3 $HOWIE_PATH/demo_agents.py'" >> "$SHELL_RC"
echo "alias howie-test='python3 $HOWIE_PATH/test_howie_cli.py'" >> "$SHELL_RC"

# Add path to Python path (optional)
echo "export PYTHONPATH=\"$HOWIE_PATH:\$PYTHONPATH\"" >> "$SHELL_RC"

echo "✅ Added Howie aliases to $SHELL_RC"
echo ""
echo "Available commands after restart:"
echo "  howie          - Enhanced multi-model CLI"
echo "  howie-classic  - Original CLI"
echo "  howie-demo     - Agent demonstration"
echo "  howie-test     - Test suite"
echo ""
echo "To use immediately, run:"
echo "  source $SHELL_RC"
echo ""
echo "Or restart your terminal"